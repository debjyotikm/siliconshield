"""Checkpoint selection, development views, independence gates, reporting, metrics, and DDP guard tests: synthetic fixtures."""

import json

import numpy as np
import pytest

from siliconshield.schemas import CHECKPOINT_OBJECTIVE_SCHEMA
from siliconshield.training.laya import development_views as views
from siliconshield.training.laya.checkpoint_selection import (
    objective_selection,
    population_indices,
    triplet_counts,
)
from siliconshield.training.laya.security_reporting import (
    LABELS,
)


# ---- Merged from test_checkpoint_selection.py (aliased: block, policy, result) ----
def _selection_block(matrix, pairs):
    """Build a consistent synthetic three-class metric block."""
    n = sum(map(sum, matrix))
    return {
        "confusion": matrix,
        "n": n,
        "accuracy": sum(matrix[i][i] for i in range(3)) / n,
        "false_safe_count": matrix[1][0],
        "complete_pairs": sum(matrix[0]),
        "both_pair_members_correct": pairs / sum(matrix[0]),
        "nll": 0.1,
    }


def _selection_policy():
    return {
        "version": CHECKPOINT_OBJECTIVE_SCHEMA,
        "protected": {
            "old": {"rows": 9, "min_correct": 9, "max_false_safe": 0},
            "new": {"rows": 9, "max_false_safe": 0},
        },
        "view": "new",
        "targets": {
            "min_class_correct": [2, 3, 3],
            "min_pairs_correct": 2,
        },
    }


def _selection_result(new, old=None):
    return {
        "development_views": {
            "old": old
            or _selection_block([[3, 0, 0], [0, 3, 0], [0, 0, 3]], 3),
            "new": new,
        }
    }


def test_new_task_gain_cannot_compensate_for_old_false_safe():
    initial = _selection_result(
        _selection_block([[0, 3, 0], [0, 3, 0], [0, 0, 3]], 0)
    )
    damaged = _selection_result(
        _selection_block([[3, 0, 0], [0, 3, 0], [0, 0, 3]], 3),
        _selection_block([[3, 0, 0], [1, 2, 0], [0, 0, 3]], 2),
    )
    assert objective_selection(
        initial, _selection_policy()
    ) < objective_selection(damaged, _selection_policy())
    assert not damaged["checkpoint_acceptance"]["protected_passed"]


# ---- Merged from test_development_views.py (aliased: block, policy, result) ----
def _views_block(
    accuracy=0.99,
    false_safe=1,
    violations=10,
    recall=0.9,
    pairs=0.95,
    mc=(10, 10),
    families=None,
    nll=0.002,
    n=30,
    complete_pairs=10,
):
    mc_correct, mc_total = mc
    return {
        "accuracy": accuracy,
        "false_safe_count": false_safe,
        "violation_count": violations,
        "min_class_recall": recall,
        "both_pair_members_correct": pairs,
        "complete_pairs": complete_pairs,
        "confusion": [
            [8, 1, 1],
            [1, 8, 1],
            [mc_total - mc_correct, 0, mc_correct],
        ],
        "per_family": families
        if families is not None
        else {"famA": {"false_safe_count": 1, "violation_count": 6}},
        "n": n,
        "nll": nll,
    }


def _views_policy(**overrides):
    base = {
        "min_retention_accuracy": 0.95,
        "max_retention_false_safe_increase": 0,
        "require_no_missing_context_regression": True,
        "require_no_pair_regression": True,
        "min_new_coverage_gain_pp": 0.0,
    }
    base.update(overrides)
    return base


def reference(**overrides):
    transfer = _views_block()
    retention = _views_block()
    return views.build_selection_reference(
        transfer, retention, _views_policy(**overrides), "digest"
    )


def _views_result(transfer, retention):
    return {"development_views": {"transfer": transfer, "retention": retention}}


def config(**overrides):
    base = {
        "selection_view": "transfer",
        "retention_view": "retention",
        "max_retention_accuracy_drop": 0.01,
    }
    base.update(overrides)
    return base


def base_key(view):
    return (view["nll"],)


def test_joint_selection_preserves_pairs_over_lower_nll():
    ref = reference(
        max_new_coverage_false_safe_increase=0,
        family_false_safe_caps={"famA": 1},
    )
    good = _views_result(_views_block(nll=0.002), _views_block(pairs=0.95))
    bad = _views_result(_views_block(nll=0.001), _views_block(pairs=0.90))
    assert views.selection(
        good, config(selection_reference=ref), 0.99, base_key
    ) < views.selection(bad, config(selection_reference=ref), 0.99, base_key)


def adapted_view(false_safe=1, pairs=(9, 10), families="default"):
    pairs_correct, pairs_total = pairs
    return {
        "counts": {
            "n": 30,
            "correct": 28,
            "false_safe": false_safe,
            "violations": 10,
            "mc_correct": 10,
            "mc_total": 10,
            "pairs_correct": pairs_correct,
            "pairs_total": pairs_total,
            "classes": [
                {"correct": 10, "total": 10},
                {"correct": 8, "total": 10},
                {"correct": 10, "total": 10},
            ],
        },
        "families": (
            {"famA": {"false_safe": 1, "violations": 6}}
            if families == "default"
            else families
        ),
    }


def stored_results(**overrides):
    stored = {
        "new_coverage": adapted_view(false_safe=2),
        "retention": adapted_view(false_safe=1),
    }
    stored.update(overrides)
    return stored


# ---- Merged from test_independence_gates.py ----
def _row(row_id, rtl, group, mechanism="locking"):
    return {
        "id": row_id,
        "state": "REQUIREMENT\nr\nASSUMPTIONS\na\nRTL\n%s\n" % rtl,
        "split_group_id": group,
        "dependency_group_id": group,
        "mechanism_id": mechanism,
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _accept_audit(manifest, extra):
    from siliconshield.schemas import DATASET_AUDIT_SCHEMA

    audit = {
        "schema": DATASET_AUDIT_SCHEMA,
        "splits": {
            "train": manifest["splits"]["train"]["sha256"],
        },
        "cross_split_exact_rtl_overlap": 0,
        "retired_rtl_overlap": 0,
        "truncated": 0,
        "lineage_overlap": 0,
        "canonical_overlap": 0,
        "max_group_share": 0.05,
        "max_mechanism_share": 0.15,
        "review_ledger_sha256": "ab" * 32,
        "reviewed_templates": 2,
        "release_class": "narrow-pilot",
    }
    audit.update(extra)
    return audit


# ---- Merged from test_security_reporting.py ----
def fixture():
    rows = []
    for group in range(2):
        for label in range(3):
            rows.append(
                {
                    "id": str(len(rows)),
                    "split_group_id": "g" + str(group),
                    "pair_id": "pair" + str(group),
                    "source": {"name": "source-a"},
                    "mechanism_id": "access",
                    "expected": {"security": LABELS[label]},
                }
            )
    p = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    return rows, np.log(p), np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0])


# ---- Merged from test_inference_metrics.py ----


# ---- Merged from test_ddp_guard.py ----


def test_population_indices_reject_malformed_rows_with_value_error():
    import pytest

    with pytest.raises(ValueError, match="no source name"):
        population_indices([{}], ["s"])
    with pytest.raises(ValueError, match="no source name"):
        population_indices([{"source": "s"}], ["s"])
    with pytest.raises(ValueError, match="not an object"):
        population_indices([["x"]], ["s"])
    rows = [
        {"id": "a", "source": {"name": "s"}, "split_group_id": "g"},
        {"id": "b", "source": {"name": "t"}, "split_group_id": "g"},
    ]
    assert population_indices(rows, ["s"]) == [0]
    assert population_indices(rows, {"split_groups": ["g"]}) == [0, 1]


def test_triplet_counts_reject_malformed_rows_with_value_error():
    import pytest

    labels = ["x", "y", "z"]
    with pytest.raises(ValueError, match="not row 0"):
        triplet_counts([["x"], ["y"], ["z"]], [0, 1, 2], labels)
    with pytest.raises(ValueError, match="no expected label"):
        triplet_counts([{"pair_id": "p"}] * 3, [0, 1, 2], labels)
    with pytest.raises(ValueError, match="no expected label"):
        triplet_counts(
            [{"pair_id": "p", "expected": {"security": {"x": 1}}}] * 3,
            [0, 1, 2],
            labels,
        )
    rows = [{"pair_id": "p", "expected": {"security": name}} for name in labels]
    assert triplet_counts(rows, [0, 1, 2], labels) == {
        "triplets": 1,
        "triplets_correct": 1,
    }


def test_objective_selection_rejects_malformed_results_with_value_error():
    import pytest

    policy = _selection_policy()
    with pytest.raises(ValueError, match="result object"):
        objective_selection([], policy)
    with pytest.raises(ValueError, match="named development views"):
        objective_selection({}, policy)
    with pytest.raises(ValueError, match="named development views"):
        objective_selection({"development_views": []}, policy)
    with pytest.raises(ValueError, match="not an object"):
        objective_selection(
            {"development_views": {"old": [], "new": []}}, policy
        )
    good = _selection_result(
        _selection_block([[3, 0, 0], [0, 3, 0], [0, 0, 3]], 3)
    )
    assert objective_selection(good, policy) is not None


def test_view_counts_reject_bool_accuracy_and_pairless_views():
    import pytest

    from siliconshield.training.laya.checkpoint_selection import view_counts

    base = _selection_block([[2, 0, 0], [0, 2, 0], [0, 0, 2]], 2)
    with pytest.raises(ValueError, match="disagrees with counts"):
        view_counts(dict(base, accuracy=True))
    with pytest.raises(ValueError, match="needs complete pairs"):
        view_counts(dict(base, complete_pairs=0, both_pair_members_correct=0.0))
    assert view_counts(base)["correct"] == 6


def test_views_manifest_rejects_malformed_shapes_with_value_error():
    import pytest

    from siliconshield.training.laya.development_views import (
        attach_views,
        config_from_views_manifest,
    )

    base = {
        "schema": "siliconshield-dev-views-v3",
        "n_rows": 3,
        "views": [
            {
                "review": "agree",
                "record_id": "r",
                "triplets": [
                    {
                        "satisfies": "a",
                        "violates": "b",
                        "insufficient": "c",
                    }
                ],
            }
        ],
    }
    assert config_from_views_manifest(base)["selection_view"] == "fresh_dev"
    bad_templates = [
        dict(base, views={}),
        dict(base, views=["x"]),
        dict(
            base,
            views=[{"review": "agree", "record_id": "r", "triplets": {}}],
        ),
        dict(base, views=[{"review": "agree", "triplets": ["x"]}]),
        dict(
            base,
            views=[
                {
                    "review": "agree",
                    "triplets": [{"satisfies": "a"}],
                }
            ],
        ),
        dict(
            base,
            views=[
                {
                    "review": "agree",
                    "triplets": [
                        {
                            "satisfies": {},
                            "violates": "b",
                            "insufficient": "c",
                        }
                    ],
                }
            ],
        ),
    ]
    for manifest in bad_templates:
        with pytest.raises(ValueError):
            config_from_views_manifest(manifest)
    with pytest.raises(ValueError, match="config object"):
        attach_views({}, [], [], [], lambda r, l: {})
    with pytest.raises(ValueError, match="named populations"):
        attach_views({}, [], [], {"views": []}, lambda r, l: {})
    with pytest.raises(ValueError, match="Missing selection view"):
        attach_views(
            {},
            [{"id": "a", "source": {"name": "s"}}],
            [0],
            {"views": {"v": ["s"]}},
            lambda r, l: {},
        )


def test_confusion_counts_reject_malformed_cells():
    import pytest

    for matrix in (
        [["a", 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, -1]],
        [[True, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, None]],
        [[0, 0, 0], [0, 0, 0]],
    ):
        with pytest.raises(ValueError):
            views.mc_recall_from_confusion(matrix)
    assert views.mc_recall_from_confusion(
        [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    ) == pytest.approx(0.5)
    assert (
        views.mc_recall_from_confusion([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) == 1.0
    )


def test_coverage_helpers_reject_malformed_inputs():
    from siliconshield.training.laya import coverage_audit as audit

    with pytest.raises(ValueError, match="row objects"):
        audit.triplet_integrity(5, {}, set())
    with pytest.raises(ValueError, match="row objects"):
        audit.triplet_integrity([5], {}, set())
    with pytest.raises(ValueError, match="pool for every row"):
        audit.triplet_integrity([{"pair_id": "p"}], {}, set())
    with pytest.raises(ValueError, match="string pair ids"):
        audit.triplet_integrity([{"pair_id": 5}], {0: "a"}, set())
    with pytest.raises(ValueError, match="round index"):
        audit.quotas_for_round({0: {"a": 1}}, "x")
    sampler = lambda *a, **k: []
    with pytest.raises(ValueError, match="callable"):
        audit.simulate_coverage([], ({},), 1, 1, grouped_epoch_indices=5)
    with pytest.raises(ValueError, match="Horizon must be an integer"):
        audit.simulate_coverage(
            [], ({},), 1, "x", grouped_epoch_indices=sampler
        )
    with pytest.raises(ValueError, match="pool plan"):
        audit.simulate_coverage([], 5, 1, 1, grouped_epoch_indices=sampler)
    with pytest.raises(ValueError, match="World size"):
        audit.simulate_coverage(
            [], ({},), 1, 1, world="x", grouped_epoch_indices=sampler
        )
    with pytest.raises(ValueError, match="Epoch must be an integer"):
        audit.reconstruct_seen(
            [], ({},), 1, "x", 0, grouped_epoch_indices=sampler
        )
    with pytest.raises(ValueError, match="exposure object"):
        audit.exposure_gate_passed(5)
    with pytest.raises(ValueError, match="exposure object"):
        audit.exposure_gate_passed({})
    with pytest.raises(ValueError, match="draw counts"):
        audit.delivered_exposure([1], [], {})
    with pytest.raises(ValueError, match="draw counts"):
        audit.delivered_exposure({0: "x"}, [{"pair_id": "p"}], {0: "a"})


def test_ledger_guard_and_inventory_reject_malformed_inputs(tmp_path):
    from siliconshield.training.laya import ddp_guard as guard
    from siliconshield.training.laya import inventory
    from siliconshield.training.laya import review_ledger as ledger

    with pytest.raises(ValueError, match="must be an object"):
        ledger.requirement_of(5)
    with pytest.raises(ValueError, match="must be an object"):
        ledger.assumptions_of(5)
    base = {
        "template_id": "t",
        "requirement_sha256": "a",
        "assertion_sha256": "b",
        "assumptions_sha256": "c",
        "reviewer_type": "human",
        "interpretation": "i",
        "verdict": "agree",
    }
    with pytest.raises(ValueError, match="must be text"):
        ledger.validate_review(dict(base, resolution=5))
    with pytest.raises(ValueError, match="must be a path"):
        ledger.validate_ledger(5)
    with pytest.raises(ValueError, match="review objects"):
        ledger.ledger_digest(5)
    with pytest.raises(ValueError, match="review objects"):
        ledger.template_ids([5])
    with pytest.raises(ValueError, match="row objects"):
        ledger.require_template_coverage(5, [])
    with pytest.raises(ValueError, match="review objects"):
        ledger.require_template_coverage([], [5])
    with pytest.raises(ValueError, match="JSON-serializable"):
        ledger.packet_assertion_sha({1, 2})
    with pytest.raises(ValueError, match="review objects"):
        ledger.require_packet_binding([5], {"t": {}})
    with pytest.raises(ValueError, match="source path"):
        guard.count_backward_calls(5)
    with pytest.raises(ValueError, match="source path"):
        guard.training_loop_uses_no_sync(5)
    bad = tmp_path / "bad.py"
    bad.write_text("def f(:\n")
    with pytest.raises(ValueError, match="does not parse"):
        guard.count_backward_calls(bad)
    with pytest.raises(ValueError, match="row objects"):
        inventory.summarize(5)
    good = [{"origin": "o", "split_group_id": "g"}]
    with pytest.raises(ValueError, match="DEV inventory"):
        inventory.summarize(good, dev_rows=5)
    with pytest.raises(ValueError, match="inventory report"):
        inventory.feasibility(5)
    report = {
        "train_groups": 1,
        "dev_groups": 1,
        "calibration_groups": 1,
        "test_groups": 1,
    }
    with pytest.raises(ValueError, match="map names to integers"):
        inventory.feasibility(report, targets=5)
    with pytest.raises(ValueError, match="map names to integers"):
        inventory.feasibility(report, targets={"train_groups": "x"})


def test_question_contracts_reject_malformed_inputs(tmp_path):
    from siliconshield.training.laya import question_contracts as contracts

    with pytest.raises(ValueError, match="contract validation"):
        contracts.validate_rows(5, "siliconshield-requirement-context-v1")
    with pytest.raises(ValueError, match="unknown question contract"):
        contracts.validate_rows([], ["v1"])
    with pytest.raises(ValueError, match="must be a path"):
        contracts.write_export_record(5, "siliconshield-requirement-context-v1")


def test_campaign_helpers_reject_malformed_inputs():
    pytest.importorskip("torch")
    import os

    if not os.environ.get("LAYA_ROOT"):
        pytest.skip("LAYA runtime is not configured")
    from siliconshield.training.laya import security_campaign as campaign

    with pytest.raises(ValueError, match="argument list"):
        campaign.run_process("echo", "/tmp/x", {}, 5)
    with pytest.raises(ValueError, match="timeout must be positive"):
        campaign.run_process(["echo"], "/tmp/x", {}, "x")
    with pytest.raises(ValueError, match="must be an object"):
        campaign.temperature_for(5)
    assert (
        campaign.temperature_for(
            {
                "temperature_by_options": {"choice:3-5": 1.5},
                "temperature": [1.0],
            }
        )
        == 1.5
    )
    with pytest.raises(ValueError, match="family ids"):
        campaign.grouped_calibration(5, [], [])
    with pytest.raises(ValueError, match="per-row logits"):
        campaign.grouped_calibration([{"family_id": "a"}], [[1.0]], [])
    assert campaign.cache_valid(5, {}) is False
    with pytest.raises(ValueError, match="must be a path"):
        campaign.validate_training_report(5, "r", 1, "d", {})


def test_trainer_helpers_reject_malformed_inputs(tmp_path):
    pytest.importorskip("torch")
    import os

    if not os.environ.get("LAYA_ROOT"):
        pytest.skip("LAYA runtime is not configured")
    from siliconshield.training.laya import security_train as trainer

    with pytest.raises(ValueError, match="state text"):
        trainer.canonical_input(5)
    with pytest.raises(ValueError, match="must be a path"):
        trainer.load_splits(5)
    bad = tmp_path / "data"
    bad.mkdir()
    (bad / "train.jsonl").write_text("[1]\n")
    (bad / "dev.jsonl").write_text("[1]\n")
    with pytest.raises(ValueError, match="non-object row"):
        trainer.load_splits(bad)
    with pytest.raises(ValueError, match="integer world"):
        trainer.grouped_epoch_indices([], "x", 0, 1, 0, 48)
    with pytest.raises(ValueError, match="row objects"):
        trainer.grouped_epoch_indices(5, 1, 0, 1, 0, 3)
    with pytest.raises(ValueError, match="pool plan triple"):
        trainer.grouped_epoch_indices([], 1, 0, 1, 0, 3, pool_plan=5)
    with pytest.raises(ValueError, match="queue state"):
        trainer.grouped_epoch_indices([], 1, 0, 1, 0, 3, _queue_state=5)
