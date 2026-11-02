"""Offline contracts for the standalone distributed security training flow."""

import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "siliconshield" / "training" / "laya"
sys.path.insert(0, str(TRAINING))
if not os.environ.get("LAYA_ROOT") and (ROOT / "laya").is_dir():
    os.environ["LAYA_ROOT"] = str(ROOT)


def module(name):
    path = TRAINING / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


train = module("security_train")
data = module("build_security_data")
# Package import: security_campaign uses a package-relative import of its
# sibling trainer, so it cannot be loaded as a standalone file module.
from siliconshield.training.laya import security_campaign as campaign


def test_known_metrics_and_alignment():
    logits = [[10, 0, 0], [10, 0, 0], [0, 0, 10], [0, 10, 0]]
    result = train.metrics(logits, [0, 1, 2, 0])
    assert result["accuracy"] == 0.5
    assert result["false_safe_count"] == 1
    assert result["violation_count"] == 1
    assert result["selection_cost"] == 1.5
    assert result["confusion"] == [[1, 1, 0], [1, 0, 0], [0, 0, 1]]
    assert result["degeneracy_count"] == 1
    for z, y in (
        ([], []),
        ([[1, 2, 3]], []),
        ([np.nan, 0, 0], [0]),
        ([1, 2, 3], [3]),
    ):
        with pytest.raises(ValueError):
            train.metrics(z, y)
    assert train.metrics([[1, 0, 0]], [2])["false_safe_rate"] is None


def test_view_diagnostics_respect_uncertainty_budget(monkeypatch):
    calls = []

    def reporting(*args, **kwargs):
        calls.append(kwargs)
        return {"budget": kwargs["bootstrap_samples"]}

    monkeypatch.setattr(train, "evaluation_report", reporting)
    monkeypatch.setattr(train, "hierarchical_weights", lambda rows: [1] * 3)
    rows = [
        {
            "expected": {"security": label},
            "topic": "authored",
            "family_id": "authored",
            "pair_id": "one",
        }
        for label in train.LABELS
    ]
    logits = np.eye(3).tolist()
    disabled = train.diagnostics(rows, logits, bootstrap_samples=0, seed=17)
    enabled = train.diagnostics(rows, logits, bootstrap_samples=7, seed=17)
    assert [call["bootstrap_samples"] for call in calls] == [0, 7]
    assert all(call["seed"] == 17 for call in calls)
    disabled.pop("reporting")
    enabled.pop("reporting")
    assert disabled == enabled


def test_continuation_schedule_uses_absolute_resumed_step():
    factors = [
        train.learning_rate_factor(step, 4, 20, "cosine") for step in range(21)
    ]
    assert factors[:4] == [0.25, 0.5, 0.75, 1.0]
    assert all(a >= b > 0 for a, b in zip(factors[4:], factors[5:]))
    assert factors[-1] == pytest.approx(0.1)
    assert [
        train.learning_rate_factor(step, 4, 20, "cosine")
        for step in range(9, 21)
    ] == factors[9:]
    assert train.learning_rate_factor(50, 4, 0, "constant") == 1.0
    with pytest.raises(ValueError, match="greater than warmup"):
        train.learning_rate_factor(0, 4, 4, "cosine")


def test_hold_decay_schedule_phases():
    warmup, hold, maximum = 10, 20, 100
    at = lambda step: train.learning_rate_factor(  # noqa: E731
        step, warmup, maximum, "hold-decay", hold, 0.1
    )
    assert [at(step) for step in range(10)] == [
        pytest.approx((step + 1) / 10) for step in range(10)
    ]
    assert all(at(step) == 1.0 for step in range(10, 30))
    decay = [at(step) for step in range(30, 101)]
    assert all(a >= b for a, b in zip(decay, decay[1:]))
    assert decay[-1] == pytest.approx(0.1)
    assert at(65) == pytest.approx(0.1 + 0.9 * 0.5, abs=1e-9)
    with pytest.raises(ValueError, match="warmup plus hold"):
        train.learning_rate_factor(0, 10, 30, "hold-decay", 20, 0.1)


def test_hold_decay_zero_hold_matches_cosine():
    for step in range(0, 40):
        assert train.learning_rate_factor(
            step, 4, 40, "hold-decay", 0, 0.1
        ) == train.learning_rate_factor(step, 4, 40, "cosine")


def test_hold_decay_matches_preflight_model():
    # Coupling pin: post-warmup factors must equal the pool/schedule
    # preflight model (bin/preflight_pools.py lr_fraction) exactly,
    # so the 20%-at-first-pass gate describes the executed schedule.
    # Known differences outside the executed range only: inside
    # warmup the trainer ramps (step + 1) / warmup while preflight
    # uses step / warmup, and past the horizon the trainer clamps
    # while preflight keeps oscillating (the trainer never runs
    # past its update budget).
    warmup, hold, maximum, endpoint = 10, 700, 1250, 0.1

    def preflight(update):
        if update < warmup:
            return update / warmup
        decay_start = warmup + hold
        if update < decay_start:
            return 1.0
        span = max(maximum - decay_start, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * (update - decay_start) / span))
        return endpoint + (1 - endpoint) * cosine

    for update in [10, 100, 699, 700, 710, 975, 1249, 1250]:
        assert train.learning_rate_factor(
            update, warmup, maximum, "hold-decay", hold, endpoint
        ) == pytest.approx(preflight(update), abs=1e-12)
    assert train.learning_rate_factor(
        1300, warmup, maximum, "hold-decay", hold, endpoint
    ) == pytest.approx(endpoint)


def test_transfer_acceptance_includes_retention():
    good = {
        "selection_cost": 0,
        "baseline_best_cost": 1,
        "degeneracy_count": 0,
        "worst_family_false_safe_rate": 0,
        "min_class_recall": 1,
        "both_pair_members_correct": 1,
    }
    report = {
        "selection_view": "transfer",
        "development_views": {"transfer": good},
        "retention_check": {"passed": False},
    }
    assert train.acceptance_status(report)["reasons"] == [
        "retention_accuracy_below_budget"
    ]
    report["retention_check"]["passed"] = True
    assert train.acceptance_status(report)["accepted"]


def test_sdk_parity_respects_rounding_and_rejects_wrong_temperature():
    probabilities = np.array([0.072031, 0.870206, 0.057763], dtype=np.float32)
    logits = np.log(probabilities) * 5.0
    serialized = [0.072, 0.8702, 0.0578]
    train.assert_sdk_probabilities(serialized, logits, 5.0)
    with pytest.raises(AssertionError):
        train.assert_sdk_probabilities(serialized, logits, 1.0)
    with pytest.raises(AssertionError):
        train.assert_sdk_probabilities([0.072, 0.8702, 0.0579], logits, 5.0)


@pytest.mark.parametrize("n", [1, 5, 25, 129])
def test_equal_rank_steps_padding_and_partial_group_weighting(n):
    shards = [train.epoch_indices(n, 3, rank, 41, 0) for rank in range(3)]
    masks = [train.epoch_valid_mask(n, 3, rank, 41, 0) for rank in range(3)]
    assert len({len(x) for x in shards}) == len({len(x) for x in masks}) == 1
    combined = sum(shards, [])
    valid = sum(masks, [])
    assert len(combined) == len(valid)
    assert sum(valid) == n
    assert sorted(
        index for index, flag in zip(combined, valid) if flag
    ) == list(range(n))
    seen = set()
    for index, flag in zip(combined, valid):
        if index in seen:
            assert flag == 0
        else:
            assert flag == 1
        seen.add(index)
    for indices in shards:
        groups = list(train.accumulation_groups(indices, 4, 4))
        assert sum(
            len(chunk) for _, chunks in groups for chunk, _, _ in chunks
        ) == len(indices)
        for _, chunks in groups:
            assert sum(weight for _, weight, _ in chunks) == pytest.approx(1)
            assert sum(final for _, _, final in chunks) == 1
            assert chunks[-1][2]


def test_accumulation_matches_complete_group_gradient():
    torch.manual_seed(7)
    left = torch.nn.Linear(3, 1)
    right = torch.nn.Linear(3, 1)
    right.load_state_dict(left.state_dict())
    x, y = torch.randn(13, 3), torch.randn(13, 1)
    for _, chunks in train.accumulation_groups(list(range(13)), 4, 4):
        for ids, weight, _ in chunks:
            (((left(x[ids]) - y[ids]) ** 2).mean() * weight).backward()
    ((right(x) - y) ** 2).mean().backward()
    for a, b in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


def semantic_row(
    split,
    label,
    index,
    source="HardSecBench",
    mechanism="hold",
    design=None,
    variants=1,
):
    design = design or split
    state = (
        "REQUIREMENT\n%s-%d\nRTL\nmodule topmodule; wire value_%d; endmodule"
        % (split, index, index)
    )
    return {
        "id": "%s-%d" % (split, index),
        "family_id": split + "-family",
        "split_group_id": split + "-group",
        "split": split,
        "base_design_id": design,
        "mechanism_id": mechanism,
        "rewrite_group_id": design,
        "pair_id": "%s-%d" % (split, index // 3),
        "variant": index % 3,
        "topic": mechanism,
        "state": state,
        "questions": copy.deepcopy(data.QUESTION),
        "expected": {"security": label},
        "validation": {"accepted": True},
        "source": {"name": source},
    }


def test_split_rejects_lineage_duplicates_and_incomplete_triplets(tmp_path):
    for split in ("train", "dev", "calibration", "test"):
        rows = [
            semantic_row(split, label, index)
            for index, label in enumerate(train.LABELS)
        ]
        (tmp_path / (split + ".jsonl")).write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    assert len(train.load_splits(tmp_path, final=True)) == 4
    path = tmp_path / "dev.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    row["split_group_id"] = "train-group"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0] = row
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="split group crosses"):
        train.load_splits(tmp_path, final=True)


def test_split_rejects_conflicting_duplicate_inputs(tmp_path):
    for split in ("train", "dev", "calibration", "test"):
        rows = [
            semantic_row(split, label, index)
            for index, label in enumerate(train.LABELS)
        ]
        (tmp_path / (split + ".jsonl")).write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    rows = [
        json.loads(line)
        for line in (tmp_path / "dev.jsonl").read_text().splitlines()
    ]
    rows[1]["state"] = rows[0]["state"]
    (tmp_path / "dev.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match="duplicate input"):
        train.load_splits(tmp_path, final=True)


def test_hierarchical_weights_match_declared_mass():
    rows = []
    for source, mechanism, count in (
        ("HardSecBench", "hold", 2),
        ("authored-semantic", "direct", 2),
        ("OpenTitan", "combinational_exact", 1),
    ):
        for design in range(count):
            for variant in range(3 if source == "authored-semantic" else 1):
                for label in train.LABELS:
                    row = semantic_row(
                        "train",
                        label,
                        len(rows),
                        source,
                        mechanism,
                        "%s-%s-%d" % (source, mechanism, design),
                    )
                    row["variant"] = variant
                    row["pair_id"] += "-%d" % variant
                    rows.append(row)
    weights = train.hierarchical_weights(rows)
    report = train.weighting_report(rows, weights)
    assert report["source_mass"] == pytest.approx(train.SOURCE_MASS)
    assert report["label_mass"] == pytest.approx(
        {label: 1 / 3 for label in train.LABELS}
    )
    assert np.all(weights > 0)


def selection_result(
    cost=0.1,
    family_cost=None,
    worst_cost=None,
    false_safe=0.0,
    degeneracy=0,
    pair=1.0,
    triplet=1.0,
    recall=1.0,
    baseline=0.3,
):
    family_cost = cost if family_cost is None else family_cost
    return {
        "selection_cost": cost,
        "family_selection_cost": family_cost,
        "worst_family_selection_cost": cost
        if worst_cost is None
        else worst_cost,
        "worst_family_false_safe_rate": false_safe,
        "degeneracy_count": degeneracy,
        "both_pair_members_correct": pair,
        "complete_triplets_correct": triplet,
        "min_class_recall": recall,
        "family_macro_f1": 0.8,
        "macro_f1": 0.8,
        "nll": 0.2,
        "baseline_best_cost": baseline,
    }


def test_selection_is_baseline_safety_and_family_aware():
    assert train.selection_key(
        selection_result(family_cost=0.3, worst_cost=0.1)
    ) < train.selection_key(
        selection_result(cost=0.05, family_cost=0.5, worst_cost=0.1)
    )
    assert train.selection_key(
        selection_result(degeneracy=1, cost=0.01)
    ) > train.selection_key(selection_result(cost=0.2))
    assert train.selection_key(
        selection_result(cost=0.4)
    ) > train.selection_key(selection_result(cost=0.2))
    assert train.selection_key(
        selection_result(false_safe=0.5)
    ) > train.selection_key(selection_result(false_safe=0.0))
    assert train.selection_key(
        selection_result(pair=0.2)
    ) > train.selection_key(selection_result(pair=0.8))
    assert train.acceptance_status(selection_result())["accepted"]
    assert not train.acceptance_status(selection_result(degeneracy=1))[
        "accepted"
    ]


def test_selection_rejects_false_safe_shortcut_and_retains_balanced_progress():
    biased = selection_result(
        cost=0.292, pair=0.125, recall=0.125, false_safe=0
    )
    useful = selection_result(
        cost=0.347, pair=0.5, recall=0.5, false_safe=2 / 3
    )
    assert train.selection_key(useful) < train.selection_key(biased)
    earlier = selection_result(
        cost=0.417, pair=0.5, recall=0.5, false_safe=2 / 3
    )
    later = selection_result(cost=0.403, pair=0.625, recall=0.792, false_safe=1)
    later["family_macro_f1"] = 0.87
    assert train.selection_key(later) < train.selection_key(earlier)
    assert not train.acceptance_status(later)["accepted"]


def test_grouped_sampler_preserves_global_balance_and_resume_order():
    rows = []
    for source in train.SOURCE_MASS:
        for design in range(5):
            for variant in range(3):
                for label in train.LABELS:
                    rows.append(
                        semantic_row(
                            "train",
                            label,
                            len(rows),
                            source,
                            "m%d" % (design % 2),
                            source + str(design),
                        )
                    )
    shards = [
        train.grouped_epoch_indices(rows, 3, rank, 41, 0, 48)
        for rank in range(3)
    ]
    assert len({len(shard) for shard in shards}) == 1
    order = [index for triplet in zip(*shards) for index in triplet]
    for offset in range(0, len(order), 48):
        batch = [rows[index] for index in order[offset : offset + 48]]
        assert {
            label: sum(row["expected"]["security"] == label for row in batch)
            for label in train.LABELS
        } == {label: 16 for label in train.LABELS}
        # Expected allocation follows the configured recipe (largest remainder), not a fixed recipe.
        slots = 48 // 3
        raw = {
            name: train.SOURCE_MASS[name] * slots for name in train.SOURCE_MASS
        }
        quotas = {name: int(value) for name, value in raw.items()}
        for name in sorted(
            raw, key=lambda n: (raw[n] - quotas[n], n), reverse=True
        )[: slots - sum(quotas.values())]:
            quotas[name] += 1
        assert {
            source: sum(row["source"]["name"] == source for row in batch)
            for source in train.SOURCE_MASS
        } == {name: q * 3 for name, q in quotas.items()}
    assert shards[0] == train.grouped_epoch_indices(rows, 3, 0, 41, 0, 48)
    assert shards[0] != train.grouped_epoch_indices(rows, 3, 0, 41, 1, 48)
    assert (
        len({rows[index]["expected"]["security"] for index in shards[0]}) == 3
    )
    with pytest.raises(ValueError, match="complete triplets"):
        train.grouped_epoch_indices(rows[:-1], 3, 0, 41, 0, 48)


def test_grouped_sampler_mid_epoch_resume_matches_optimizer_continuation(
    tmp_path,
):
    rows = []
    for source in train.SOURCE_MASS:
        for design in range(12):
            for label in train.LABELS:
                rows.append(
                    semantic_row(
                        "train",
                        label,
                        len(rows),
                        source,
                        "hold",
                        source + str(design),
                    )
                )
    indices = train.grouped_epoch_indices(rows, 1, 0, 41, 2, 48)
    torch.manual_seed(9)
    model = torch.nn.Linear(2, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x = torch.randn(len(rows), 2)
    y = torch.tensor(
        [train.LABELS.index(row["expected"]["security"]) for row in rows]
    )

    def update(network, opt, batch):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(
            network(x[batch]), y[batch]
        ).backward()
        opt.step()

    update(model, optimizer, indices[:48])
    path = tmp_path / "resume.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "offset": 48,
        },
        path,
    )
    for offset in range(48, len(indices), 48):
        update(model, optimizer, indices[offset : offset + 48])
    state = torch.load(path, weights_only=False)
    resumed = torch.nn.Linear(2, 3)
    resumed.load_state_dict(state["model"])
    opt = torch.optim.AdamW(resumed.parameters(), lr=0.001)
    opt.load_state_dict(state["optimizer"])
    replay = train.grouped_epoch_indices(rows, 1, 0, 41, 2, 48)
    for offset in range(state["offset"], len(replay), 48):
        update(resumed, opt, replay[offset : offset + 48])
    for left, right in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_policy_reversal_preserves_rtl_and_reverses_known_labels(tmp_path):
    yosys = os.environ.get("LAYA_TEST_YOSYS")
    if not yosys:
        pytest.skip("set LAYA_TEST_YOSYS for solver-backed policy reversals")
    expansion = module("expand_security_data")
    policies = module("augment_security_policies")
    rows, failure = expansion.expand_one(
        expansion.scenario(421, 0), yosys, tmp_path
    )
    assert failure is None
    for variant in range(3):
        original = [row for row in rows if row["variant"] == variant]
        alternate, failure = policies.augment_group(original, yosys, tmp_path)
        assert failure is None
        assert len(alternate) == 3
        for before, after in zip(original, alternate):
            assert (
                before["state"].split("RTL\n", 1)[1]
                == after["state"].split("RTL\n", 1)[1]
            )
            assert before["state"] != after["state"]
            label = before["expected"]["security"]
            expected = dict(
                zip(
                    train.LABELS,
                    [train.LABELS[1], train.LABELS[0], train.LABELS[2]],
                )
            )[label]
            assert after["expected"]["security"] == expected
            assert after["split_group_id"] == before["split_group_id"]
            assert after["base_design_id"] == before["base_design_id"]
            assert after["policy_counterpart_id"] == before["id"]
        assert (
            alternate[2]["validation"]["good_fill_sha256"]
            == original[2]["validation"]["bad_fill_sha256"]
        )


def test_conditional_hold_does_not_require_authorized_writes(tmp_path):
    yosys = os.environ.get("LAYA_TEST_YOSYS")
    if not yosys:
        pytest.skip("set LAYA_TEST_YOSYS for partial contract semantics")
    expansion = module("expand_security_data")
    guards = module("augment_security_guards")
    rows, failure = expansion.expand_one(
        expansion.scenario(421, 0), yosys, tmp_path
    )
    assert failure is None
    result, failure = guards.guard_group(
        [row for row in rows if row["variant"] == 0], yosys, tmp_path
    )
    assert failure is None and len(result) == 3
    code = "module topmodule(input clk, rst_n, wr, input [3:0] wdata, output reg [3:0] value); always @(posedge clk) if (!rst_n) value <= 0; endmodule"
    hold = data.verify(
        code, data.hold_check("value", "!wr"), yosys, tmp_path, "rst_n"
    )
    exact = data.verify(
        code, expansion.exact_check("wr"), yosys, tmp_path, "rst_n"
    )
    assert hold["accepted"] and hold["verdict"] == train.LABELS[0]
    assert exact["accepted"] and exact["verdict"] == train.LABELS[1]


def test_plateau_waits_for_update_budget_and_survives_resume():
    progress = {"step": 352, "stale": 8, "reduced": False}
    assert train.plateau_action(progress, 7, 800, 4) is None
    progress["step"] = 800
    assert train.plateau_action(progress, 7, 800, 4) == "reduce_lr"
    progress.update(reduced=True, stale=0)
    restored = json.loads(json.dumps(progress))
    assert train.plateau_action(restored, 8, 800, 4) is None
    restored["stale"] = 4
    assert train.plateau_action(restored, 12, 800, 4) == "stop"


def test_plateau_acts_on_update_budget_in_early_epochs():
    # An update-based warmup can finish before the epoch-based warmup.
    progress = {"step": 90, "stale": 5, "reduced": False}
    assert train.plateau_action(progress, 1, 40, 3) == "reduce_lr"
    progress.update(reduced=True, stale=0)
    assert train.plateau_action(progress, 1, 40, 3) is None
    # Without an update budget the epoch warmup still applies.
    assert (
        train.plateau_action(
            {"step": 90, "stale": 5, "reduced": False}, 1, 0, 3
        )
        is None
    )
    assert (
        train.plateau_action(
            {"step": 90, "stale": 5, "reduced": False}, 3, 0, 3
        )
        == "reduce_lr"
    )


def test_alpha_renaming_preserves_verilog_literals():
    code = "module topmodule(input [1:0] priv, output wire value); assign value = priv == 2'b11; endmodule"
    rule = "value must be true when priv equals 2'b11."
    rendered, renamed = data.rename_example(code, rule, 7)
    assert "2'b11" in rendered and "2'b11" in renamed
    assert "priv_7" in rendered and "priv_7" in renamed
    assert "b11_7" not in rendered


def test_reference_completion_reconstruction_and_structural_ownership():
    sources = os.environ.get("LAYA_TEST_SOURCES")
    if not sources:
        pytest.skip("set LAYA_TEST_SOURCES to run source-backed contracts")
    cases = list(data.hardsec_cases(Path(sources))) + list(
        data.opentitan_cases(Path(sources))
    )
    for case in cases:
        for variant in data.VARIANTS:
            rendered = data.render_variant(case, variant)
            good = data.complete_rendered(
                rendered["opaque"],
                rendered["relevant_hole"],
                rendered["good_fill"],
                case["hole_width"],
            )
            bad = data.complete_rendered(
                rendered["opaque"],
                rendered["relevant_hole"],
                rendered["bad_fill"],
                case["hole_width"],
            )
            assert good == rendered["good"]
            assert bad == rendered["bad"]
    assert not [
        item
        for item in data.structural_audit(cases)
        if item["left_split"] != item["right_split"]
    ]


def test_antecedent_solver_reads_control_file():
    command, _ = data.solver_commands(
        "always @* if (enable) assert(1'b0);", "rst_n", "antecedent.v"
    )
    assert "read_verilog -formal -sv antecedent.v" in command
    assert "checked.v" not in command


@pytest.mark.parametrize("index", range(9))
def test_semantic_templates_have_verified_counterfactuals_and_irrelevant_holes(
    index, tmp_path
):
    yosys = os.environ.get("LAYA_TEST_YOSYS")
    if not yosys:
        pytest.skip(
            "set LAYA_TEST_YOSYS to run solver-backed dataset contracts"
        )
    expansion = module("expand_security_data")
    rows, failure = expansion.expand_one(
        expansion.scenario(421, index), yosys, tmp_path
    )
    assert failure is None
    assert len(rows) == 9
    assert {
        label: sum(row["expected"]["security"] == label for row in rows)
        for label in train.LABELS
    } == {label: 3 for label in train.LABELS}
    for variant in range(3):
        group = [row for row in rows if row["variant"] == variant]
        assert len({row["pair_id"] for row in group}) == 1
        assert {row["expected"]["security"] for row in group} == set(
            train.LABELS
        )
        unknown = next(
            row
            for row in group
            if row["expected"]["security"] == train.LABELS[2]
        )
        assert unknown["validation"]["good_completion"] == "byte_exact"
        assert unknown["validation"]["bad_completion"] == "byte_exact"
        assert (
            unknown["validation"]["satisfying_completion"]
            != unknown["validation"]["violating_completion"]
        )
    assert any(row["validation"].get("irrelevant_hole") for row in rows)


def test_grouped_calibration_has_no_family_leakage():
    rows, logits, targets = [], [], []
    for family in range(8):
        for label in train.LABELS:
            for _ in range(3):
                rows.append(semantic_row("calibration", label, len(rows)))
                rows[-1]["family_id"] = "family-%d" % family
                targets.append(train.LABELS.index(label))
                logits.append(
                    [4.0 if index == targets[-1] else 0.0 for index in range(3)]
                )
    fitted, report = campaign.grouped_calibration(rows, logits, targets)
    assert fitted["temperature"]
    assert len(report["folds"]) == 2
    for fold in report["folds"]:
        assert not set(fold["fit_families"]) & set(fold["validation_families"])
        assert fold["fit_n"] + fold["validation_n"] == len(rows)


class FakeAgent:
    def __init__(self):
        self.calls = []

    def predict_batch(self, states, questions, batch_size, sort_by_length):
        order = questions["security"].get("option_order", [0, 1, 2])
        self.calls.append((len(states), list(order)))
        answers = []
        for _ in states:
            probabilities = {
                label: value
                for label, value in zip(train.LABELS, [0.7, 0.2, 0.1])
            }
            answers.append(
                {
                    "answers": {"security": {"probabilities": probabilities}},
                    "usage": {"truncated": False, "input_tokens": 10},
                }
            )
        return answers


def test_sdk_option_audit_covers_every_row_and_order(tmp_path):
    rows = [
        semantic_row("test", label, index)
        for index, label in enumerate(train.LABELS)
    ]
    for row in rows:
        row["questions"] = copy.deepcopy(data.QUESTION)
    agent = FakeAgent()
    result = campaign.sdk_split(
        agent, rows, tmp_path, "test", {"run_id": "run"}, 4
    )
    assert result["option_order"]["rows"] == 3
    assert result["option_order"]["orders"] == 6
    assert result["option_order"]["all_rows_all_orders_evaluated"]
    assert result["option_order"]["stable_fraction"] == 1.0
    assert len(list((tmp_path / "finalization").glob("test-*.json"))) == 6
    cached = campaign.sdk_split(
        agent, rows, tmp_path, "test", {"run_id": "run"}, 4
    )
    assert cached["option_order"]["orders"] == 6


def test_resume_restores_checkpoint_eligible_best_and_new_lineage(tmp_path):
    model = torch.nn.Linear(2, 1)
    first = train.save_best(
        model, tmp_path, 0, 0, [0], {"n": 1}, [[1.0, 0.0, 0.0]], [0], 0
    )
    train.save_best(
        model, tmp_path, 1, 10, [-1], {"n": 1}, [[1.0, 0.0, 0.0]], [0], 0
    )
    progress = {
        "best_generation": first["generation"],
        "best_key": [0],
        "best_epoch": 0,
        "step": 0,
    }
    pointer, _, _ = train.restore_best_lineage(tmp_path, progress, 0, 0)
    assert pointer["generation"] == first["generation"]
    replay = train.save_best(
        model, tmp_path, 1, 10, [-2], {"n": 1}, [[1.0, 0.0, 0.0]], [0], 1
    )
    assert replay["generation"] != first["generation"]
    assert replay["generation"].endswith("lineage-000001")


def test_report_after_resume_keeps_initial_development(tmp_path, monkeypatch):
    # Real report producer (finish_stage) with a resumed progress lineage.
    model = torch.nn.Linear(2, 1)
    initial_metrics = {"n": 4, "accuracy": 0.5, "nll": 0.9}
    selected_metrics = {
        "n": 4,
        "accuracy": 0.75,
        "nll": 0.6,
        "selection_cost": 0.3,
        "min_class_recall": 0.5,
        "both_pair_members_correct": 0.5,
        "degeneracy_count": 0,
        "worst_family_false_safe_rate": 0.0,
    }
    train.save_best(
        model,
        tmp_path,
        0,
        0,
        [1],
        initial_metrics,
        [[1.0, 0.0, 0.0]] * 4,
        [0] * 4,
        0,
    )
    train.save_best(
        model,
        tmp_path,
        1,
        10,
        [0],
        selected_metrics,
        [[1.0, 0.0, 0.0]] * 4,
        [0] * 4,
        0,
    )
    monkeypatch.setattr(
        train, "export_selected", lambda *a, **k: {"run_id": "resumed-run"}
    )
    progress = {
        "step": 10,
        "stop_reason": "complete",
        "lineage": 3,
        "initial_generation": train.generation_name(0, 0, 0),
    }
    train.finish_stage(
        None,
        None,
        {},
        tmp_path,
        {"data": {"train": "x", "dev": "y"}},
        progress,
        "resumed-run",
        0,
        1,
        lambda event: None,
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["initial_development"] == initial_metrics
    assert report["development"] == selected_metrics
    from siliconshield.research_loop.episodes.laya_results import adapt_report

    summary, _, _ = adapt_report(report, source_sha256="e" * 64)
    assert summary["initial"] == {"decisions": 4, "agreement": 0.5, "nll": 0.9}
    # Older resumable state without the reference also recovers via the epoch-zero scan.
    del progress["initial_generation"]
    assert train.load_initial_development(tmp_path, progress) == initial_metrics


def test_existing_final_model_requires_complete_calibration_lineage(tmp_path):
    final_model = tmp_path / "final_model"
    final_model.mkdir()
    weights = {"weight": torch.ones(1)}
    save_file(weights, str(final_model / "model.safetensors"))
    cfg = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}
    fitted = {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}
    (final_model / "rl_agent_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="calibration lineage"):
        campaign.prepare_final_model(
            tmp_path,
            cfg,
            fitted,
            {"folds": []},
            train.sha256(final_model / "model.safetensors"),
        )


def test_optimizer_groups_give_adapters_their_own_lr_and_exclude_frozen():
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.residual_adapters = torch.nn.Linear(4, 4)
            self.head = torch.nn.Linear(4, 3)

    model = FakeModel()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    groups = train.optimizer_param_groups(model, 1e-5, 3e-5, 1e-4)
    by_kind = {group["group_name"]: group for group in groups}
    assert set(by_kind) == {"adapter", "head"}
    assert by_kind["adapter"]["lr"] == 3e-5 and by_kind["head"]["lr"] == 1e-4
    assert by_kind["adapter"]["lr"] != by_kind["head"]["lr"]
    seen = {
        name
        for group in groups
        for name in [
            n
            for n, p in model.named_parameters()
            if any(p is q for q in group["params"])
        ]
    }
    assert not any(name.startswith("encoder.") for name in seen)
    assert any("residual_adapters" in name for name in seen)


def test_undefined_baseline_is_not_given_label_metadata():
    good = semantic_row("test", train.LABELS[0], 0)
    good["state"] = "module topmodule; wire condition_expr; endmodule"
    bad = semantic_row("test", train.LABELS[1], 1)
    bad["state"] = (
        "module topmodule; wire condition_expr; assign status = condition_expr; endmodule"
    )
    unknown = semantic_row("test", train.LABELS[2], 2)
    unknown["state"] = (
        "module topmodule; wire condition_expr; assign request = condition_expr; endmodule"
    )
    assert not train.surface_undefined(good)
    assert train.surface_undefined(bad)
    assert train.surface_undefined(unknown)


def test_fitting_subset_keeps_complete_designs_and_train_only_load(tmp_path):
    rows = []
    for source in ("HardSecBench", "OpenTitan"):
        for design in range(4):
            for index, label in enumerate(train.LABELS):
                row = semantic_row("train", label, design * 3 + index)
                row.update(
                    id=source + "-" + str(design) + "-" + str(index),
                    family_id=source + str(design),
                    base_design_id=source + str(design),
                    pair_id=source + str(design),
                    source={"name": source},
                    mechanism="fixture",
                )
                row["state"] = row["state"] + "\n// " + row["id"]
                rows.append(row)
    # Deliberately interleave the verdicts, as shuffled real datasets do.
    rows.sort(key=lambda r: r["expected"]["security"])
    chosen = train.fitting_subset(rows, 8, 7)
    assert len(chosen) >= 8
    assert {r["source"]["name"] for r in chosen} == {
        "HardSecBench",
        "OpenTitan",
    }
    assert train.fitting_subset(rows, 8, 7) == chosen
    assert np.isfinite(train.hierarchical_weights(chosen)).all()
    (tmp_path / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    (tmp_path / "dev.jsonl").write_text("must never be parsed")
    assert set(train.load_splits(tmp_path, train_only=True)) == {"train"}


def test_head_only_and_adapter_only_are_distinct_and_receive_gradients():
    from types import SimpleNamespace

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="modernbert", hidden_size=4
            )
            self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

        def forward(self, x):
            return self.layers[0](x)

    def model():
        value = torch.nn.Module()
        value.encoder, value.head, value.act_head = (
            Encoder(),
            torch.nn.Linear(4, 3),
            torch.nn.Linear(4, 1),
        )
        value.residual_adapters = None
        return value

    head = model()
    assert (
        train.configure_trainable_model(head, {}, True, True, None, False)
        is None
    )
    groups = train.optimizer_param_groups(head, 1e-5, 1e-4, 1e-4)
    assert {g["group_name"] for g in groups} == {"head"}
    adapted = model()
    train.configure_trainable_model(adapted, {}, True, False, 1.0, False)
    groups = train.optimizer_param_groups(adapted, 1e-5, 1e-4, 1e-4)
    assert {g["group_name"] for g in groups} == {"adapter", "head"}
    optimizer = torch.optim.AdamW(groups)
    x = torch.randn(2, 3, 4)
    for step in range(2):
        optimizer.zero_grad()
        adapted.head(adapted.encoder(x)).square().sum().backward()
        branch = adapted.residual_adapters.branches["0"]
        assert branch.up.weight.grad.norm() > 0
        if step:
            assert branch.down.weight.grad.norm() > 0
        assert all(p.grad is None for p in adapted.encoder.parameters())
        optimizer.step()


def test_full_finetune_config_trains_encoder_and_head_only():
    from types import SimpleNamespace

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="modernbert", hidden_size=4
            )
            self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

        def forward(self, x):
            return self.layers[0](x)

    def model():
        value = torch.nn.Module()
        value.encoder, value.head, value.act_head = (
            Encoder(),
            torch.nn.Linear(4, 3),
            torch.nn.Linear(4, 1),
        )
        value.residual_adapters = None
        return value

    full = model()
    assert (
        train.configure_trainable_model(full, {}, False, True, None, False)
        is None
    )
    assert full.residual_adapters is None
    assert all(p.requires_grad for p in full.encoder.parameters())
    assert all(p.requires_grad for p in full.head.parameters())
    assert all(not p.requires_grad for p in full.act_head.parameters())
    groups = train.optimizer_param_groups(full, 1e-5, 2e-6, 2e-6)
    assert {g["group_name"] for g in groups} == {"encoder", "head"}
    optimizer = torch.optim.AdamW(groups)
    optimizer.zero_grad()
    full.head(full.encoder(torch.randn(2, 3, 4))).square().sum().backward()
    assert full.encoder.layers[0].weight.grad.norm() > 0
    assert full.head.weight.grad.norm() > 0
    assert all(p.grad is None for p in full.act_head.parameters())

    adapted = model()
    adapted.residual_adapters = object()
    with pytest.raises(ValueError, match="continue-training"):
        train.configure_trainable_model(adapted, {}, False, True, None, False)
    with pytest.raises(ValueError, match="unadapted base"):
        train.configure_trainable_model(adapted, {}, False, True, None, True)


def test_encoder_trainable_blocks_unfreezes_top_layers_only():
    from types import SimpleNamespace

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="modernbert", hidden_size=4
            )
            self.emb = torch.nn.Embedding(8, 4)
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4) for _ in range(4)]
            )

        def forward(self, x):
            value = self.emb(x)
            for layer in self.layers:
                value = layer(value)
            return value

    def model():
        value = torch.nn.Module()
        value.encoder, value.head, value.act_head = (
            Encoder(),
            torch.nn.Linear(4, 3),
            torch.nn.Linear(4, 1),
        )
        value.residual_adapters = None
        return value

    top2 = model()
    train.configure_trainable_model(top2, {}, False, False, None, False, 2)
    for index, layer in enumerate(top2.encoder.layers):
        assert all(
            p.requires_grad == (index >= 2) for p in layer.parameters()
        ), index
    assert all(not p.requires_grad for p in top2.encoder.emb.parameters())
    assert all(p.requires_grad for p in top2.residual_adapters.parameters())
    assert all(p.requires_grad for p in top2.head.parameters())
    assert all(not p.requires_grad for p in top2.act_head.parameters())
    groups = train.optimizer_param_groups(top2, 1e-7, 1e-6, 1e-6)
    assert {g["group_name"] for g in groups} == {"encoder", "adapter", "head"}
    encoder_params = [
        p for g in groups if g["group_name"] == "encoder" for p in g["params"]
    ]
    assert {id(p) for p in encoder_params} == {
        id(p) for layer in top2.encoder.layers[2:] for p in layer.parameters()
    }
    optimizer = torch.optim.AdamW(groups)
    optimizer.zero_grad()
    x = torch.randint(0, 8, (2, 3))
    top2.head(top2.encoder(x)).square().sum().backward()
    for index, layer in enumerate(top2.encoder.layers):
        for p in layer.parameters():
            assert (p.grad is not None) == (index >= 2), index
    assert all(p.grad is None for p in top2.encoder.emb.parameters())
    assert top2.head.weight.grad.norm() > 0
    assert all(p.grad is None for p in top2.act_head.parameters())

    bad = model()
    with pytest.raises(ValueError, match="contradicts"):
        train.configure_trainable_model(bad, {}, True, False, None, False, 2)
    for invalid in (0, 5, True):
        with pytest.raises(ValueError, match="1\\.\\.4"):
            train.configure_trainable_model(
                model(), {}, False, False, None, False, invalid
            )


def test_gradient_norms_report_actual_post_clip_values():
    model = torch.nn.Module()
    model.head = torch.nn.Linear(1, 1, bias=False)
    model.head.weight.grad = torch.tensor([[6.0]])
    assert train.component_gradient_norms(model)["head"] == 6
    returned = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert returned == 6
    assert train.component_gradient_norms(model)["head"] == pytest.approx(
        1.0, abs=1e-6
    )


def test_terminal_evaluation_updates_resume_and_exports_trained_weights(
    tmp_path, monkeypatch
):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    initial = {"n": 3, "accuracy": 1 / 3, "nll": 1.2}
    selected = {
        **initial,
        "accuracy": 1.0,
        "nll": 0.1,
        "selection_cost": 0.0,
        "min_class_recall": 1.0,
        "both_pair_members_correct": 1.0,
        "degeneracy_count": 0,
        "worst_family_false_safe_rate": 0.0,
    }
    initial_ptr = train.save_best(
        model, tmp_path, 0, 0, [1], initial, [[1, 0, 0]] * 3, [0, 1, 2], 0
    )
    with torch.no_grad():
        model.weight.add_(1)
    progress = {
        "step": 7,
        "last_eval_step": 0,
        "epoch": 0,
        "offset": 3,
        "lineage": 0,
        "stop_reason": "update_budget",
        "initial_generation": initial_ptr["generation"],
    }
    signature = {
        "evaluation_role": "train_fitting",
        "data": {"train": "fixture"},
        "fitting_subset": {"rows": 3, "ids_sha256": "fixture"},
    }
    snapshots = []

    def save_resume(model, optimizer, progress, signature, output, rank, world):
        snapshots.append(copy.deepcopy(progress))

    def observe(epoch, consumed):
        assert consumed == progress["offset"]
        pointer = train.save_best(
            model,
            tmp_path,
            epoch + 1,
            progress["step"],
            [0],
            selected,
            [[1, 0, 0]] * 3,
            [0, 1, 2],
            0,
        )
        progress.update(
            last_eval_step=progress["step"],
            last_metrics=selected,
            best_generation=pointer["generation"],
        )

    expected = model.weight.detach().clone()

    def export(model, *args):
        torch.testing.assert_close(model.weight, expected)
        return {"weights_checked": True}

    monkeypatch.setattr(train, "save_resume", save_resume)
    monkeypatch.setattr(train, "export_selected", export)
    train.finalize_training(
        model,
        optimizer,
        None,
        {},
        tmp_path,
        signature,
        progress,
        "fixture",
        0,
        1,
        lambda event: None,
        observe,
        "cpu",
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["selected_checkpoint"]["step"] == 7
    assert snapshots[-1]["last_eval_step"] == 7
    assert report["initial_train"] == initial and report["train"] == selected
    assert "development" not in report and report["train_accuracy"] == 1.0
    from siliconshield.research_loop.episodes.laya_results import adapt_report

    with pytest.raises((ValueError, KeyError)):
        adapt_report(report)


def test_terminal_evaluation_off_boundary_passes_consumed_offset(
    tmp_path, monkeypatch
):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    selected = {
        "n": 3,
        "accuracy": 1.0,
        "nll": 0.1,
        "selection_cost": 0.0,
        "min_class_recall": 1.0,
        "both_pair_members_correct": 1.0,
        "degeneracy_count": 0,
        "worst_family_false_safe_rate": 0.0,
    }
    initial_ptr = train.save_best(
        model,
        tmp_path,
        0,
        0,
        [1],
        {**selected, "accuracy": 1 / 3},
        [[1, 0, 0]] * 3,
        [0, 1, 2],
        0,
    )
    progress = {
        "step": 7,
        "last_eval_step": 0,
        "epoch": 0,
        "offset": 5,
        "lineage": 0,
        "stop_reason": "operator_pause",
        "initial_generation": initial_ptr["generation"],
    }
    signature = {
        "evaluation_role": "train_fitting",
        "data": {"train": "fixture"},
        "fitting_subset": {"rows": 3, "ids_sha256": "fixture"},
    }
    calls = []

    def observe(epoch, consumed):
        calls.append((epoch, consumed))
        pointer = train.save_best(
            model,
            tmp_path,
            epoch + 1,
            progress["step"],
            [0],
            selected,
            [[1, 0, 0]] * 3,
            [0, 1, 2],
            0,
        )
        progress.update(
            last_eval_step=progress["step"],
            last_metrics=selected,
            best_generation=pointer["generation"],
        )

    def export(_model, *_args):
        return {"weights_checked": True}

    monkeypatch.setattr(train, "save_resume", lambda *args: None)
    monkeypatch.setattr(train, "export_selected", export)
    train.finalize_training(
        model,
        optimizer,
        None,
        {},
        tmp_path,
        signature,
        progress,
        "fixture",
        0,
        1,
        lambda event: None,
        observe,
        "cpu",
    )
    assert calls == [(0, 5)]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["selected_checkpoint"]["step"] == 7


def test_trainer_and_final_report_schemas_are_named_and_distinct():
    from siliconshield.research_loop.episodes import laya_results
    from siliconshield.training.laya import (
        build_security_data,
        build_security_external,
        expand_security_data,
    )

    assert train.REPORT_SCHEMA == laya_results.TRAINER_SCHEMA == 2
    assert data.DATASET_MANIFEST_SCHEMA == 3
    assert campaign.FINAL_REPORT_SCHEMA == 3
    assert campaign.FINAL_REPORT_SCHEMA != train.REPORT_SCHEMA
    assert build_security_data.EVIDENCE_SCHEMA == "security-sat-v3"
    assert expand_security_data.EXPANDED_DATASET_MANIFEST_SCHEMA == 2
    assert build_security_external.EXTERNAL_DATASET_MANIFEST_SCHEMA == 2
    from siliconshield.training.laya import augment_security_policies

    assert augment_security_policies.AUGMENTED_TRAIN_MANIFEST_SCHEMA == 3


def _pool_fixture(tmp_path):
    from siliconshield.training.laya import dataset_versions

    rows = []
    for triplet in range(8):
        for label in train.LABELS:
            rows.append(
                semantic_row(
                    "train",
                    label,
                    len(rows),
                    "HardSecBench",
                    "hold",
                    "design-%d" % triplet,
                )
            )
    ids = tmp_path / "ids.jsonl"
    ids.write_text(
        "".join(json.dumps({"id": row["id"]}) + "\n" for row in rows)
    )
    pools = tmp_path / "pools.json"
    dataset_versions.assign_pools(
        ids,
        {
            "new": {
                "rows": [row["id"] for row in rows[:18]],
                "weight": 3.0,
            },
            "old": {
                "rows": [row["id"] for row in rows[18:]],
                "weight": 1.0,
            },
        },
        pools,
    )
    return rows, pools


def test_pool_assignment_agrees_across_implementations(tmp_path):
    from siliconshield.training.laya import dataset_versions

    rows, pools = _pool_fixture(tmp_path)
    assert train.pool_assignment_for_rows(
        pools, rows
    ) == dataset_versions.pool_assignment(pools, rows)


def test_grouped_sampler_honors_pool_quotas(tmp_path):
    rows, pools = _pool_fixture(tmp_path)
    assignment, mass, modes = train.pool_assignment_for_rows(pools, rows)
    indices = train.grouped_epoch_indices(
        rows, 1, 0, 41, 0, 12, (assignment, mass, modes)
    )
    assert len(indices) == len(rows)
    for offset in range(0, len(indices), 12):
        batch = indices[offset : offset + 12]
        owned = [assignment[index] for index in batch]
        assert owned.count("new") == 9 and owned.count("old") == 3


def test_grouped_sampler_rejects_pool_violations(tmp_path):
    rows, pools = _pool_fixture(tmp_path)
    assignment, mass, modes = train.pool_assignment_for_rows(pools, rows)
    split = dict(assignment)
    split[0] = "old"
    with pytest.raises(ValueError, match="stratum boundary"):
        train.grouped_epoch_indices(rows, 1, 0, 41, 0, 12, (split, mass, modes))
    with pytest.raises(ValueError, match="every configured pool"):
        train.grouped_epoch_indices(
            rows, 1, 0, 41, 0, 12, (assignment, {"new": 1.0})
        )


def test_pool_modes_default_cycle_and_reject_unknown(tmp_path):
    from siliconshield.training.laya import dataset_versions

    rows, pools = _pool_fixture(tmp_path)
    _, _, modes = dataset_versions.pool_assignment(pools, rows)
    assert modes == {"new": "cycle", "old": "cycle"}
    manifest = json.loads(pools.read_text())
    manifest["pools"]["new"]["mode"] = "flat"
    manifest["pools"]["old"]["mode"] = "sideways"
    pools.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cycle or flat"):
        dataset_versions.pool_assignment(pools, rows)
    with pytest.raises(ValueError, match="cycle or flat"):
        train.pool_assignment_for_rows(pools, rows)


def test_grouped_sampler_flat_mode_draws_uniformly(tmp_path):
    rows, pools = _pool_fixture(tmp_path)
    manifest = json.loads(pools.read_text())
    manifest["pools"]["new"]["mode"] = "flat"
    manifest["pools"]["old"]["mode"] = "flat"
    pools.write_text(json.dumps(manifest))
    plan = train.pool_assignment_for_rows(pools, rows)
    assert plan[2] == {"new": "flat", "old": "flat"}
    indices = train.grouped_epoch_indices(rows, 1, 0, 41, 0, 12, plan)
    assert len(indices) == len(rows)
    for offset in range(0, len(indices), 12):
        batch = indices[offset : offset + 12]
        owned = [plan[0][index] for index in batch]
        assert owned.count("new") == 9 and owned.count("old") == 3


def _queue_rows():
    rows = []
    for triplet in range(10):
        for label in train.LABELS:
            rows.append(
                semantic_row(
                    "train",
                    label,
                    len(rows),
                    "HardSecBench",
                    "hold",
                    "design-%d" % triplet,
                )
            )
    return rows


def test_grouped_sampler_persistent_queues_carry_and_replay(tmp_path):
    from siliconshield.training.laya import dataset_versions

    rows = _queue_rows()
    pools = tmp_path / "pools.json"
    ids = tmp_path / "ids.jsonl"
    ids.write_text(
        "".join(json.dumps({"id": row["id"]}) + "\n" for row in rows)
    )
    dataset_versions.assign_pools(
        ids,
        {
            "new": {
                "rows": [row["id"] for row in rows[:15]],
                "weight": 1.0,
            },
            "old": {
                "rows": [row["id"] for row in rows[15:]],
                "weight": 1.0,
            },
        },
        pools,
    )
    plan = train.pool_assignment_for_rows(pools, rows)
    carried: dict = {}
    first = train.grouped_epoch_indices(
        rows, 1, 0, 41, 0, 12, plan, _queue_state=carried
    )
    second = train.grouped_epoch_indices(
        rows, 1, 0, 41, 1, 12, plan, _queue_state=carried
    )
    assert carried, "carried queues must persist state"
    fresh = train.grouped_epoch_indices(rows, 1, 0, 41, 1, 12, plan)
    assert second != fresh
    replay: dict = {}
    train.grouped_epoch_indices(
        rows, 1, 0, 41, 0, 12, plan, _queue_state=replay
    )
    again = train.grouped_epoch_indices(
        rows, 1, 0, 41, 1, 12, plan, _queue_state=replay
    )
    assert again == second
    assert first == train.grouped_epoch_indices(rows, 1, 0, 41, 0, 12, plan)


def test_replica_digest_stable_and_sensitive():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    first = train.replica_digest(model, True)
    assert first == train.replica_digest(model, True)
    assert first == train.assert_replicas_consistent(model, 0, 1, "cpu")
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert train.replica_digest(model, True) != first


def test_verify_frozen_steady_refuses_movement():
    model = torch.nn.Linear(3, 3)
    model.weight.requires_grad_(False)
    expected = train.replica_digest(model, False)
    assert train.verify_frozen_steady(model, expected, "cpu") == expected
    with torch.no_grad():
        model.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="frozen parameters moved"):
        train.verify_frozen_steady(model, expected, "cpu")


def test_save_eval_snapshot_retains_weights(tmp_path):
    model = torch.nn.Linear(3, 3)
    manifest = train.save_eval_snapshot(
        model, tmp_path, 1, 25, [0, 1], {"accuracy": 0.5}, 0
    )
    assert manifest["step"] == 25
    assert (
        tmp_path / "eval_snapshots" / "step-00000025" / "model.safetensors"
    ).exists()
    again = train.save_eval_snapshot(
        model, tmp_path, 1, 25, [0, 1], {"accuracy": 0.5}, 0
    )
    assert again["weights_sha256"] == manifest["weights_sha256"]


def test_trainer_runs_bare_script_without_installed_package(tmp_path):
    env = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    probe = subprocess.run(
        [sys.executable, "-c", "import siliconshield"],
        capture_output=True,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    if probe.returncode == 0:
        pytest.skip("siliconshield is installed in this environment")
    run = subprocess.run(
        [sys.executable, str(TRAINING / "security_train.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert "usage" in run.stdout.lower()


def _reference_view():
    return {
        "accuracy": 0.99,
        "false_safe_count": 1,
        "violation_count": 10,
        "min_class_recall": 0.9,
        "both_pair_members_correct": 0.95,
        "mc_correct": 10,
        "mc_total": 10,
        "per_family": {
            "famA": {"false_safe_count": 1, "violation_count": 6},
        },
    }


def _selection_reference():
    return {
        "schema": "siliconshield-selection-reference-v1",
        "report_digest": "abc",
        "transfer": _reference_view(),
        "retention": _reference_view(),
        "policy": {
            "min_retention_accuracy": 0.95,
            "max_retention_false_safe_increase": 0,
            "require_no_missing_context_regression": True,
            "require_no_pair_regression": True,
            "min_new_coverage_gain_pp": 0.0,
        },
    }


def test_recipe_selection_reference_parses_and_validates():
    assert train.SELECTION_REFERENCE is None
    assert train._selection_reference_from_recipe({}) is None
    reference = _selection_reference()
    assert (
        train._selection_reference_from_recipe(
            {"selection_reference": reference}
        )
        == reference
    )
    with pytest.raises(ValueError, match="schema"):
        train._selection_reference_from_recipe(
            {"selection_reference": {"schema": "other"}}
        )


def test_trainer_recipe_reference_validates_at_import(tmp_path):
    good = tmp_path / "good-recipe.json"
    good.write_text(json.dumps({"selection_reference": _selection_reference()}))
    bad = tmp_path / "bad-recipe.json"
    bad.write_text(json.dumps({"selection_reference": {"schema": "other"}}))
    for path, valid in ((good, True), (bad, False)):
        env = {
            key: value
            for key, value in os.environ.items()
            if key != "PYTHONPATH"
        }
        env["SILICONSHIELD_SECURITY_RECIPE"] = str(path)
        run = subprocess.run(
            [sys.executable, str(TRAINING / "security_train.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=tmp_path,
            env=env,
            check=False,
        )
        assert (run.returncode == 0) == valid, run.stderr[-2000:]
        if not valid:
            assert "Selection reference" in run.stderr


def test_trainer_dry_run_validates_settings_without_data(
    tmp_path, monkeypatch, capsys
):
    argv = [
        "security_train.py",
        "--model",
        str(tmp_path / "missing-model"),
        "--data",
        str(tmp_path / "missing-data"),
        "--output",
        str(tmp_path / "out"),
        "--run-id",
        "dry",
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert train.main() is None
    assert "settings valid" in capsys.readouterr().out


def test_trainer_dry_run_rejects_hold_beyond_budget(tmp_path, monkeypatch):
    argv = [
        "security_train.py",
        "--model",
        "m",
        "--data",
        "d",
        "--output",
        "o",
        "--run-id",
        "r",
        "--dry-run",
        "--lr-schedule",
        "hold-decay",
        "--max-updates",
        "30",
        "--warmup",
        "5",
        "--hold-updates",
        "50",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        train.main()
    assert caught.value.code == 2


def test_trainer_dry_run_rejects_views_in_fitting(tmp_path, monkeypatch):
    views = tmp_path / "views.json"
    views.write_text("{}")
    argv = [
        "security_train.py",
        "--model",
        "m",
        "--data",
        "d",
        "--output",
        "o",
        "--run-id",
        "r",
        "--dry-run",
        "--limit-train-rows",
        "18",
        "--development-views",
        str(views),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        train.main()
    assert caught.value.code == 2
