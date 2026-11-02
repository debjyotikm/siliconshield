"""Campaign registry, selection, transactions, and the round runner."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from siliconshield.research_loop.adapters import laya_training
from siliconshield.research_loop.episodes.contracts import digest
from siliconshield.research_loop.execution import laya_campaign
from siliconshield.research_loop.execution.laya_campaign import (
    selection_inputs_from_report,
)
from siliconshield.research_loop.execution.operator import (
    Operator,
    idempotency_key,
)
from siliconshield.training.laya import dataset_versions

POP_SHA = "ab" * 32
INF_DIGEST = "cd" * 32


def write(path, text="fixture"):
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def binding_receipt(path, export, results_dict, experiment_id, round_id):
    """Write a receipt binding an export, results, and run identity."""
    content = {
        "run": {"experiment_id": experiment_id, "round_id": round_id},
        "export_manifest": laya_campaign.export_manifest(Path(export)),
        "results_sha256": digest(results_dict),
    }
    text = json.dumps(content, sort_keys=True)
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def policy():
    return {
        "research": {
            "min_retention_accuracy": 0.9,
            "max_retention_false_safe_increase": 0,
            "require_no_missing_context_regression": True,
            "require_no_pair_regression": True,
            "min_new_coverage_gain_pp": 5.0,
        },
        "deployment": {
            "max_worst_family_false_safe_rate": 0.2,
            "min_class_recall": 0.5,
            "min_pair_accuracy": 0.5,
        },
    }


def view(
    n,
    correct,
    false_safe,
    violations,
    mc,
    pairs,
    classes,
    worst,
    split,
    sha=POP_SHA,
    digest=INF_DIGEST,
    families=None,
    nll=None,
):
    mc_correct, mc_total = mc
    pairs_correct, pairs_total = pairs
    built = {
        "population": {
            "dataset_version": "d1",
            "split": split,
            "sha256": sha,
            "n": n,
        },
        "inference": {"digest": digest},
        "nll": nll,
        "counts": {
            "n": n,
            "correct": correct,
            "false_safe": false_safe,
            "violations": violations,
            "mc_correct": mc_correct,
            "mc_total": mc_total,
            "pairs_correct": pairs_correct,
            "pairs_total": pairs_total,
            "classes": [{"correct": c, "total": t} for c, t in classes],
        },
        "worst_family": {
            "id": worst[0],
            "false_safe": worst[1],
            "violations": worst[2],
        },
    }
    if families is not None:
        built["families"] = families
    return built


def retention(
    correct=60,
    false_safe=1,
    pairs=(19, 20),
    classes=((20, 20), (19, 20), (20, 20)),
    **extra,
):
    return view(
        60,
        correct,
        false_safe,
        20,
        (20, 20),
        pairs,
        classes,
        ("w0", 1, 4),
        "dev-retention",
        **extra,
    )


def coverage(
    correct=24,
    false_safe=3,
    pairs=(8, 10),
    classes=((8, 10), (7, 10), (9, 10)),
    worst=("w1", 2, 6),
    **extra,
):
    return view(
        30,
        correct,
        false_safe,
        10,
        (10, 10),
        pairs,
        classes,
        worst,
        "dev-transfer",
        **extra,
    )


def results(retention_view=None, coverage_view=None):
    return {
        "retention": retention_view or retention(),
        "new_coverage": coverage_view or coverage(),
    }


@pytest.fixture
def campaign(tmp_path):
    export = tmp_path / "parent"
    export.mkdir()
    weights = write(export / "model.safetensors")
    recipe = tmp_path / "recipe.json"
    recipe_sha = write(recipe, json.dumps({"training": {"max_updates": 10}}))
    receipt = tmp_path / "receipt.json"
    receipt_sha = binding_receipt(
        receipt, export, results(), "genesis-eval", None
    )
    laya_campaign.init_campaign(
        tmp_path / "campaign",
        {
            "campaign_id": "demo",
            "parent": {"export": str(export), "weights_sha256": weights},
            "recipe": {"path": str(recipe), "sha256": recipe_sha},
            "selection": policy(),
            "incumbent": {
                "export": str(export),
                "results": results(),
                "receipt": {"path": str(receipt), "sha256": receipt_sha},
            },
        },
    )
    return tmp_path / "campaign"


@pytest.fixture
def versioned(campaign, tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text('{"id": "t1"}\n')
    dev = tmp_path / "dev.jsonl"
    dev.write_text('{"id": "d1"}\n')
    manifest = dataset_versions.write_version(
        tmp_path / "datasets",
        "d1",
        {"train": str(train), "dev": str(dev)},
    )
    from siliconshield.schemas import DATASET_AUDIT_SCHEMA

    dataset_versions.accept_version(
        tmp_path / "datasets",
        "d1",
        {
            "schema": DATASET_AUDIT_SCHEMA,
            "splits": {
                role: entry["sha256"]
                for role, entry in manifest["splits"].items()
            },
            "cross_split_exact_rtl_overlap": 0,
            "retired_rtl_overlap": 0,
            "truncated": 0,
            "lineage_overlap": 0,
            "canonical_overlap": 0,
            "max_group_share": 0.05,
            "max_mechanism_share": 0.15,
            "review_ledger_sha256": "ab" * 32,
            "reviewed_templates": 1,
            "release_class": "narrow-pilot",
        },
    )
    manifest_path = tmp_path / "datasets/versions/d1/manifest.json"
    laya_campaign.register_dataset(
        campaign,
        "d1",
        str(manifest_path),
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    return campaign


def test_init_registers_and_freezes(campaign):
    stored = laya_campaign.load_campaign(campaign)
    assert stored["campaign_id"] == "demo"
    assert laya_campaign.load_incumbent(campaign)["round_id"] is None
    with pytest.raises(ValueError, match="already registered"):
        laya_campaign.init_campaign(campaign, {"campaign_id": "demo"})


def test_init_rejects_identity_mismatch(tmp_path):
    export = tmp_path / "parent"
    export.mkdir()
    (export / "model.safetensors").write_text("weights")
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}")
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    base = {
        "campaign_id": "demo",
        "parent": {"export": str(export), "weights_sha256": "0" * 64},
        "recipe": {
            "path": str(recipe),
            "sha256": hashlib.sha256(b"{}").hexdigest(),
        },
        "selection": policy(),
        "incumbent": {
            "export": str(export),
            "results": results(),
            "receipt": {
                "path": str(receipt),
                "sha256": hashlib.sha256(b"{}").hexdigest(),
            },
        },
    }
    with pytest.raises(ValueError, match="weights mismatch"):
        laya_campaign.init_campaign(tmp_path / "one", base)
    base["parent"]["weights_sha256"] = hashlib.sha256(b"weights").hexdigest()
    base["recipe"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="recipe mismatch"):
        laya_campaign.init_campaign(tmp_path / "two", base)
    base["recipe"]["sha256"] = hashlib.sha256(b"{}").hexdigest()
    base["incumbent"]["receipt"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt bytes"):
        laya_campaign.init_campaign(tmp_path / "three", base)


def test_round_lifecycle_enforces_order_and_idempotency(versioned):
    campaign = versioned
    with pytest.raises(ValueError, match="hypothesis"):
        laya_campaign.prepare_round(campaign, "r1", "d1", "  ")
    with pytest.raises(ValueError, match="Unknown dataset"):
        laya_campaign.prepare_round(campaign, "r1", "v9", "try")
    first = laya_campaign.prepare_round(campaign, "r1", "d1", "try")
    assert laya_campaign.prepare_round(campaign, "r1", "d1", "try") == first
    with pytest.raises(ValueError, match="different spec"):
        laya_campaign.prepare_round(campaign, "r1", "d1", "other")
    with pytest.raises(ValueError, match="Invalid round transition"):
        laya_campaign.advance(campaign, "r1", "submit")
    laya_campaign.advance(campaign, "r1", "validate_data")
    failed = laya_campaign.advance(campaign, "r1", "failed", note="x")
    assert failed["phase"] == "failed"
    with pytest.raises(ValueError, match="already terminal"):
        laya_campaign.advance(campaign, "r1", "done")


def test_round_preparation_revalidates_registered_inputs(versioned, tmp_path):
    campaign = versioned
    (tmp_path / "recipe.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="recipe changed"):
        laya_campaign.prepare_round(campaign, "r1", "d1", "try")


def test_round_identity_binds_policy_and_incumbent(versioned):
    campaign = versioned
    laya_campaign.prepare_round(campaign, "r1", "d1", "try")
    stored = json.loads((campaign / "campaign.json").read_text())
    stored["selection"]["research"]["min_new_coverage_gain_pp"] = 99.0
    (campaign / "campaign.json").write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="different spec"):
        laya_campaign.prepare_round(campaign, "r1", "d1", "try")


def _observed(campaign, round_id="r1"):
    laya_campaign.prepare_round(campaign, round_id, "d1", "try")
    for phase in (
        "validate_data",
        "preflight",
        "acquire",
        "submit",
        "observe",
    ):
        laya_campaign.advance(campaign, round_id, phase)


def _evaluated(campaign, tmp_path, candidate_results, round_id="r1"):
    _observed(campaign, round_id)
    export = tmp_path / ("candidate-" + round_id)
    export.mkdir(exist_ok=True)
    (export / "model.safetensors").write_text("candidate-weights")
    receipt = tmp_path / ("candidate-receipt-" + round_id + ".json")
    receipt_sha = binding_receipt(
        receipt, export, candidate_results, "demo-" + round_id, round_id
    )
    return laya_campaign.record_candidate(
        campaign,
        round_id,
        str(export),
        candidate_results,
        {"path": str(receipt), "sha256": receipt_sha},
    )


def promoted_results():
    return results(
        coverage_view=coverage(correct=27, false_safe=1),
    )


def test_promotion_updates_incumbent_and_keeps_previous(versioned, tmp_path):
    _evaluated(versioned, tmp_path, promoted_results())
    receipt = laya_campaign.decide(versioned, "r1")
    assert receipt["decision"]["promote_research"] is True
    assert receipt["decision"]["deployment_accepted"] is False
    assert "previous_incumbent" in receipt
    assert laya_campaign.load_incumbent(versioned)["round_id"] == "r1"
    closed = laya_campaign.close_round(versioned, "r1", "promoted")
    assert closed["phase"] == "done"


def test_retention_preserves_incumbent_and_records_reasons(versioned, tmp_path):
    before = laya_campaign.load_incumbent(versioned)
    _evaluated(
        versioned,
        tmp_path,
        results(
            retention_view=retention(false_safe=2),
            coverage_view=coverage(correct=25),
        ),
    )
    receipt = laya_campaign.decide(versioned, "r1")
    assert receipt["decision"]["promote_research"] is False
    assert receipt["decision"]["research_reasons"]
    assert receipt["previous_incumbent"] == before
    assert laya_campaign.load_incumbent(versioned) == before


def test_decide_refuses_when_policy_changed(versioned, tmp_path):
    before = laya_campaign.load_incumbent(versioned)
    _evaluated(versioned, tmp_path, results())
    stored = json.loads((versioned / "campaign.json").read_text())
    stored["selection"]["research"]["min_new_coverage_gain_pp"] = 0.0
    (versioned / "campaign.json").write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="re-specify"):
        laya_campaign.decide(versioned, "r1")
    assert laya_campaign.load_incumbent(versioned) == before


def test_decide_refuses_a_damaged_policy_binding(versioned, tmp_path):
    _evaluated(versioned, tmp_path, promoted_results())
    path = versioned / "rounds/r1/round.json"
    receipt = json.loads(path.read_text())
    receipt["bindings"]["policy_content"]["research"][
        "min_new_coverage_gain_pp"
    ] = 0.0
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="damaged"):
        laya_campaign.decide(versioned, "r1")


def test_retried_decision_rechecks_policy(versioned, tmp_path, monkeypatch):
    _evaluated(versioned, tmp_path, promoted_results())
    _crash_after(2, monkeypatch)
    with pytest.raises(RuntimeError, match="injected crash"):
        laya_campaign.decide(versioned, "r1")
    monkeypatch.undo()
    stored = json.loads((versioned / "campaign.json").read_text())
    stored["selection"]["research"]["min_new_coverage_gain_pp"] = 0.0
    (versioned / "campaign.json").write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="re-specify"):
        laya_campaign.decide(versioned, "r1")


def test_decide_refuses_when_incumbent_moved(versioned, tmp_path):
    _evaluated(versioned, tmp_path, promoted_results(), round_id="r1")
    _evaluated(versioned, tmp_path, promoted_results(), round_id="r2")
    receipt = laya_campaign.decide(versioned, "r2")
    assert receipt["decision"]["promote_research"] is True
    with pytest.raises(ValueError, match="moved"):
        laya_campaign.decide(versioned, "r1")


def _candidate_export(tmp_path):
    export = tmp_path / "candidate"
    export.mkdir(exist_ok=True)
    (export / "model.safetensors").write_text("candidate-weights")
    return export


def test_record_candidate_rejects_a_bare_receipt(versioned, tmp_path):
    _observed(versioned)
    export = _candidate_export(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt_sha = write(receipt, "{}")
    with pytest.raises(ValueError, match="bind these results"):
        laya_campaign.record_candidate(
            versioned,
            "r1",
            str(export),
            results(),
            {"path": str(receipt), "sha256": receipt_sha},
        )


def test_record_candidate_rejects_a_receipt_for_other_results(
    versioned, tmp_path
):
    _observed(versioned)
    export = _candidate_export(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt_sha = binding_receipt(
        receipt, export, promoted_results(), "demo-r1", "r1"
    )
    with pytest.raises(ValueError, match="bind these results"):
        laya_campaign.record_candidate(
            versioned,
            "r1",
            str(export),
            results(),
            {"path": str(receipt), "sha256": receipt_sha},
        )


def test_record_candidate_rejects_a_receipt_from_another_round(
    versioned, tmp_path
):
    _observed(versioned)
    export = _candidate_export(tmp_path)
    receipt = tmp_path / "receipt.json"
    candidate_results = results()
    receipt_sha = binding_receipt(
        receipt, export, candidate_results, "demo-r2", "r2"
    )
    with pytest.raises(ValueError, match="different run"):
        laya_campaign.record_candidate(
            versioned,
            "r1",
            str(export),
            candidate_results,
            {"path": str(receipt), "sha256": receipt_sha},
        )


def test_decide_rechecks_candidate_weights(versioned, tmp_path):
    before = laya_campaign.load_incumbent(versioned)
    _evaluated(versioned, tmp_path, promoted_results())
    (tmp_path / "candidate-r1/model.safetensors").write_text("tampered")
    with pytest.raises(ValueError, match="changed after evaluation"):
        laya_campaign.decide(versioned, "r1")
    assert laya_campaign.load_incumbent(versioned) == before


def test_decide_refuses_a_deleted_receipt(versioned, tmp_path):
    _evaluated(versioned, tmp_path, promoted_results())
    (tmp_path / "candidate-receipt-r1.json").unlink()
    with pytest.raises(ValueError, match="changed after evaluation"):
        laya_campaign.decide(versioned, "r1")


def test_retried_decision_rechecks_candidate_weights(
    versioned, tmp_path, monkeypatch
):
    before = laya_campaign.load_incumbent(versioned)
    _evaluated(versioned, tmp_path, promoted_results())
    _crash_after(2, monkeypatch)
    with pytest.raises(RuntimeError, match="injected crash"):
        laya_campaign.decide(versioned, "r1")
    monkeypatch.undo()
    (tmp_path / "candidate-r1/model.safetensors").write_text("tampered")
    with pytest.raises(ValueError, match="changed after evaluation"):
        laya_campaign.decide(versioned, "r1")
    assert laya_campaign.load_incumbent(versioned) == before


def test_gain_gate_uses_shared_trainer_predicate():
    short = results(coverage_view=coverage(correct=25))
    verdict = laya_campaign.apply_selection_policy(policy(), results(), short)
    assert verdict["promote_research"] is False
    assert any("gain" in reason for reason in verdict["research_reasons"])
    assert not laya_training.new_coverage_gain_ok(25 / 30, 24 / 30, 5.0)
    assert laya_training.new_coverage_gain_ok(27 / 30, 24 / 30, 5.0)


def test_verdict_records_v2_ranking():
    ranked = policy()
    ranked["research"]["selection_ranking"] = "v2-retention"
    incumbent = results(
        retention_view=retention(nll=0.30),
        coverage_view=coverage(nll=0.20),
    )
    better = results(
        retention_view=retention(false_safe=0, nll=0.25),
        coverage_view=coverage(correct=27, nll=0.18),
    )
    verdict = laya_campaign.apply_selection_policy(ranked, incumbent, better)
    assert verdict["promote_research"] is True
    assert verdict["ranking"] == {
        "version": "v2-retention",
        "incumbent": [1, -60, -19, 0.30, 0.20],
        "candidate": [0, -60, -19, 0.25, 0.18],
        "candidate_ranks_first": True,
    }
    plain = laya_campaign.apply_selection_policy(policy(), incumbent, better)
    assert "ranking" not in plain


def test_campaign_rank_matches_shared_key():
    adapted = {
        "retention": view(
            30,
            29,
            2,
            10,
            (10, 10),
            (9, 10),
            ((10, 10), (9, 10), (10, 10)),
            ("w0", 1, 4),
            "dev-retention",
            nll=0.05,
        ),
        "new_coverage": view(
            30,
            24,
            3,
            10,
            (10, 10),
            (8, 10),
            ((8, 10), (7, 10), (9, 10)),
            ("w1", 2, 6),
            "dev-transfer",
            nll=0.01,
        ),
    }
    assert laya_campaign.candidate_rank_key(adapted) == list(
        laya_training.retention_rank_key(2, 29, 9, 0.05, 0.01)
    )
    unknown = results()
    assert laya_campaign.candidate_rank_key(unknown)[-2:] == [None, None]


def test_deployment_acceptance_is_reported_separately():
    verdict = laya_campaign.apply_selection_policy(
        policy(), results(), results(coverage_view=coverage())
    )
    assert verdict["promote_research"] is False  # no coverage gain
    accepted = laya_campaign.apply_selection_policy(
        policy(),
        results(),
        results(coverage_view=coverage(worst=("w1", 1, 10))),
    )
    assert accepted["deployment_accepted"] is True


def test_incomparable_populations_are_refused():
    with pytest.raises(ValueError, match="Incomparable"):
        laya_campaign.apply_selection_policy(
            policy(),
            results(),
            results(coverage_view=coverage(sha="00" * 32)),
        )
    with pytest.raises(ValueError, match="Incomparable"):
        laya_campaign.apply_selection_policy(
            policy(),
            results(),
            results(retention_view=retention(digest="00" * 32)),
        )


def test_invalid_counts_are_refused():
    bad = retention()
    bad["counts"]["correct"] = 61
    with pytest.raises(ValueError, match="outside its population"):
        laya_campaign.apply_selection_policy(
            policy(), results(), results(retention_view=bad)
        )
    bad = coverage()
    bad["worst_family"]["violations"] = 0
    with pytest.raises(ValueError, match="counts are invalid"):
        laya_campaign.apply_selection_policy(
            policy(), results(), results(coverage_view=bad)
        )
    bad = coverage()
    bad["population"]["sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="hex digest"):
        laya_campaign.apply_selection_policy(
            policy(), results(), results(coverage_view=bad)
        )


def block(matrix, pairs, families):
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(3))
    supports = [sum(row) for row in matrix]
    recalls = [matrix[i][i] / supports[i] for i in range(3)]
    pairs_correct, pairs_total = pairs
    worst = max(
        (entry["false_safe_count"] / entry["violation_count"], name)
        for name, entry in families.items()
    )
    return {
        "n": total,
        "accuracy": correct / total,
        "confusion": matrix,
        "class_support": supports,
        "false_safe_count": matrix[1][0],
        "violation_count": supports[1],
        "min_class_recall": min(recalls),
        "both_pair_members_correct": pairs_correct / pairs_total,
        "complete_pairs": pairs_total,
        "worst_family_false_safe_rate": worst[0],
        "per_family": families,
        "nll": 0.2,
    }


def populations():
    binding = {
        "dataset_version": "d1",
        "split": "dev",
        "sha256": POP_SHA,
    }
    return {"transfer": dict(binding), "retention": dict(binding)}


def test_adapter_normalizes_trainer_blocks():
    transfer = block(
        [[8, 1, 1], [1, 8, 1], [1, 0, 9]],
        (8, 10),
        {
            "w1": {"false_safe_count": 1, "violation_count": 4},
            "w2": {"false_safe_count": 0, "violation_count": 6},
        },
    )
    retention_block = block(
        [[20, 0, 0], [1, 19, 0], [0, 0, 20]],
        (19, 20),
        {"w0": {"false_safe_count": 1, "violation_count": 20}},
    )
    normalized = selection_inputs_from_report(
        transfer,
        retention_block,
        populations=populations(),
        inference={"digest": INF_DIGEST},
    )
    coverage = normalized["new_coverage"]["counts"]
    assert (coverage["correct"], coverage["n"]) == (25, 30)
    assert normalized["new_coverage"]["worst_family"]["id"] == "w1"
    assert normalized["retention"]["counts"]["mc_correct"] == 20


def test_adapter_carries_family_table():
    transfer = block(
        [[8, 1, 1], [1, 8, 1], [1, 0, 9]],
        (8, 10),
        {
            "w1": {"false_safe_count": 1, "violation_count": 4},
            "w2": {"false_safe_count": 0, "violation_count": 6},
        },
    )
    retention_block = block(
        [[20, 0, 0], [1, 19, 0], [0, 0, 20]],
        (19, 20),
        {"w0": {"false_safe_count": 1, "violation_count": 20}},
    )
    normalized = selection_inputs_from_report(
        transfer,
        retention_block,
        populations=populations(),
        inference={"digest": INF_DIGEST},
    )
    assert normalized["new_coverage"]["families"] == {
        "w1": {"false_safe": 1, "violations": 4},
        "w2": {"false_safe": 0, "violations": 6},
    }


def test_adapter_requires_view_nll():
    transfer = block(
        [[8, 1, 1], [1, 8, 1], [1, 0, 9]],
        (8, 10),
        {"w1": {"false_safe_count": 1, "violation_count": 4}},
    )
    retention_block = block(
        [[20, 0, 0], [1, 19, 0], [0, 0, 20]],
        (19, 20),
        {"w0": {"false_safe_count": 1, "violation_count": 20}},
    )
    normalized = selection_inputs_from_report(
        transfer,
        retention_block,
        populations=populations(),
        inference={"digest": INF_DIGEST},
    )
    assert normalized["new_coverage"]["nll"] == 0.2
    assert normalized["retention"]["nll"] == 0.2
    for bad in (-0.1, float("nan"), "0.2", None):
        broken = dict(transfer, nll=bad)
        with pytest.raises(ValueError, match="NLL"):
            selection_inputs_from_report(
                broken,
                retention_block,
                populations=populations(),
                inference={"digest": INF_DIGEST},
            )
    missing = dict(transfer)
    del missing["nll"]
    with pytest.raises(ValueError, match="NLL"):
        selection_inputs_from_report(
            missing,
            retention_block,
            populations=populations(),
            inference={"digest": INF_DIGEST},
        )


def test_selection_refuses_new_coverage_false_safe_increase():
    gaining = results(
        coverage_view=coverage(correct=26, false_safe=4),
    )
    strict = policy()
    strict["research"]["max_new_coverage_false_safe_increase"] = 0
    verdict = laya_campaign.apply_selection_policy(strict, results(), gaining)
    assert verdict["promote_research"] is False
    assert any(
        "new-coverage" in reason for reason in verdict["research_reasons"]
    )
    strict["research"]["max_new_coverage_false_safe_increase"] = 1
    verdict = laya_campaign.apply_selection_policy(strict, results(), gaining)
    assert verdict["promote_research"] is True
    verdict = laya_campaign.apply_selection_policy(policy(), results(), gaining)
    assert verdict["promote_research"] is True


def test_selection_enforces_family_caps():
    families = {
        "w1": {"false_safe": 2, "violations": 6},
        "w2": {"false_safe": 0, "violations": 4},
    }
    gaining = results(
        coverage_view=coverage(correct=26, families=families),
    )
    capped = policy()
    capped["research"]["family_false_safe_caps"] = {"w1": 2}
    verdict = laya_campaign.apply_selection_policy(capped, results(), gaining)
    assert verdict["promote_research"] is True
    capped["research"]["family_false_safe_caps"] = {"w1": 1}
    verdict = laya_campaign.apply_selection_policy(capped, results(), gaining)
    assert verdict["promote_research"] is False
    assert any("w1" in reason for reason in verdict["research_reasons"])
    capped["research"]["family_false_safe_caps"] = {"w1": 2}
    verdict = laya_campaign.apply_selection_policy(
        capped, results(), results(coverage_view=coverage(correct=26))
    )
    assert verdict["promote_research"] is False
    assert any("lacks" in reason for reason in verdict["research_reasons"])


def test_selection_rejects_invalid_strengthened_policy():
    for budget in (-1, True, 1.5, "0"):
        broken = policy()
        broken["research"]["max_new_coverage_false_safe_increase"] = budget
        with pytest.raises(ValueError, match="coverage budget"):
            laya_campaign.apply_selection_policy(broken, results(), results())
    for gain in (True, "5", float("nan"), float("inf"), None):
        broken = policy()
        broken["research"]["min_new_coverage_gain_pp"] = gain
        with pytest.raises(ValueError, match="gain threshold"):
            laya_campaign.apply_selection_policy(broken, results(), results())
    for budget in (True, 1.5, "0", None):
        broken = policy()
        broken["research"]["max_retention_false_safe_increase"] = budget
        with pytest.raises(ValueError, match="retention budget"):
            laya_campaign.apply_selection_policy(broken, results(), results())
    for ranking in ("v9", 2, None):
        broken = policy()
        broken["research"]["selection_ranking"] = ranking
        with pytest.raises(ValueError, match="unknown ranking"):
            laya_campaign.apply_selection_policy(broken, results(), results())
    for caps in ({"w1": -1}, {"w1": True}, ["w1"], "w1"):
        broken = policy()
        broken["research"]["family_false_safe_caps"] = caps
        with pytest.raises(ValueError, match="family caps"):
            laya_campaign.apply_selection_policy(broken, results(), results())


def test_reference_comparison_is_record_only():
    reference = results(
        coverage_view=coverage(correct=20, false_safe=5),
    )
    gaining = results(
        coverage_view=coverage(correct=27, false_safe=1),
    )
    verdict = laya_campaign.apply_selection_policy(
        policy(), results(), gaining, reference
    )
    assert verdict["promote_research"] is True
    comparison = verdict["reference_comparison"]["new_coverage"]
    assert comparison["accuracy_pp"] == pytest.approx(70 / 3)
    assert comparison["false_safe_delta"] == -4
    assert comparison["mc_recall_delta"] == 0
    assert comparison["pair_accuracy_delta"] == 0
    assert verdict["reference_comparison"]["retention"]["false_safe_delta"] == 0
    stagnant = laya_campaign.apply_selection_policy(
        policy(), results(), results(), reference
    )
    assert stagnant["promote_research"] is False
    assert "reference_comparison" in stagnant
    assert "reference_comparison" not in laya_campaign.apply_selection_policy(
        policy(), results(), gaining
    )
    foreign = results(
        coverage_view=coverage(correct=20, sha="00" * 32),
    )
    with pytest.raises(ValueError, match="Incomparable"):
        laya_campaign.apply_selection_policy(
            policy(), results(), gaining, foreign
        )


def test_decide_records_reference_comparison(versioned, tmp_path):
    _evaluated(versioned, tmp_path, promoted_results())
    receipt = laya_campaign.decide(versioned, "r1", results())
    comparison = receipt["decision"]["reference_comparison"]
    assert comparison["new_coverage"]["false_safe_delta"] == -2


def test_round_accessors_list_and_read(versioned, tmp_path):
    assert laya_campaign.list_rounds(versioned) == []
    laya_campaign.prepare_round(versioned, "r1", "d1", "try")
    assert laya_campaign.list_rounds(versioned) == ["r1"]
    receipt = laya_campaign.read_round(versioned, "r1")
    assert receipt["phase"] == "prepare"
    (Path(versioned) / "rounds" / "r1" / "round.json").write_text("{}")
    with pytest.raises(ValueError, match="Unknown round schema"):
        laya_campaign.read_round(versioned, "r1")


def test_update_selection_registers_between_rounds(versioned):
    updated = policy()
    updated["research"]["min_new_coverage_gain_pp"] = 6.0
    laya_campaign.prepare_round(versioned, "r1", "d1", "try")
    with pytest.raises(ValueError, match="while a round is active"):
        laya_campaign.update_campaign_selection(
            versioned, updated, "strengthen"
        )
    laya_campaign.advance(versioned, "r1", "failed", reason="fixture")
    stored = laya_campaign.update_campaign_selection(
        versioned, updated, "strengthen after r1"
    )
    assert stored["selection"]["research"]["min_new_coverage_gain_pp"] == 6.0
    history = (
        (Path(versioned) / "selection_history.jsonl").read_text().splitlines()
    )
    assert len(history) == 1
    entry = json.loads(history[0])
    assert entry["reason"] == "strengthen after r1"
    assert entry["policy_digest"] == digest(updated)
    laya_campaign.prepare_round(versioned, "r2", "d1", "again")
    bound = laya_campaign.read_round(versioned, "r2")["bindings"]
    assert bound["policy"] == digest(updated)
    assert (
        bound["policy_content"]["research"]["min_new_coverage_gain_pp"] == 6.0
    )
    with pytest.raises(ValueError, match="recorded reason"):
        laya_campaign.update_campaign_selection(versioned, updated, " ")
    broken = policy()
    del broken["research"]["min_new_coverage_gain_pp"]
    with pytest.raises(ValueError, match="missing"):
        laya_campaign.update_campaign_selection(versioned, broken, "bad")


def test_update_selection_refuses_relaxation(versioned):
    laya_campaign.prepare_round(versioned, "r1", "d1", "try")
    laya_campaign.advance(versioned, "r1", "failed", reason="fixture")
    strict = policy()
    strict["research"]["max_new_coverage_false_safe_increase"] = 0
    strict["research"]["family_false_safe_caps"] = {"w1": 2}
    laya_campaign.update_campaign_selection(versioned, strict, "tighten")
    loosened = dict(strict["research"])
    for key, value in (
        ("min_retention_accuracy", 0.5),
        ("max_retention_false_safe_increase", 1),
        ("require_no_missing_context_regression", False),
        ("require_no_pair_regression", False),
        ("min_new_coverage_gain_pp", 1.0),
        ("max_new_coverage_false_safe_increase", 1),
    ):
        relaxed = {
            "research": dict(loosened),
            "deployment": strict["deployment"],
        }
        relaxed["research"][key] = value
        with pytest.raises(ValueError, match="relaxes"):
            laya_campaign.update_campaign_selection(
                versioned, relaxed, "loosen"
            )
    relaxed = {"research": dict(loosened), "deployment": strict["deployment"]}
    relaxed["research"]["family_false_safe_caps"] = {"w1": 3}
    with pytest.raises(ValueError, match="relaxes"):
        laya_campaign.update_campaign_selection(versioned, relaxed, "loosen")
    reranked = {"research": dict(loosened), "deployment": strict["deployment"]}
    reranked["research"]["selection_ranking"] = "v2-retention"
    with pytest.raises(ValueError, match="relaxes"):
        laya_campaign.update_campaign_selection(versioned, reranked, "rerank")
    required = {"research": dict(loosened), "deployment": strict["deployment"]}
    required["research"]["max_retention_false_safe_increase"] = -1
    stored = laya_campaign.update_campaign_selection(
        versioned, required, "require one fewer false-safe"
    )
    assert (
        stored["selection"]["research"]["max_retention_false_safe_increase"]
        == -1
    )
    relaxed["research"]["family_false_safe_caps"] = {}
    with pytest.raises(ValueError, match="relaxes"):
        laya_campaign.update_campaign_selection(versioned, relaxed, "drop")
    relaxed = {
        "research": dict(loosened),
        "deployment": dict(strict["deployment"]),
    }
    relaxed["deployment"]["max_worst_family_false_safe_rate"] = 0.9
    with pytest.raises(ValueError, match="relaxes"):
        laya_campaign.update_campaign_selection(versioned, relaxed, "loosen")
    relaxed["research"]["unknown_future_key"] = True
    with pytest.raises(ValueError, match="relaxes"):
        laya_campaign.update_campaign_selection(versioned, relaxed, "novel")
    laya_campaign.update_campaign_selection(
        versioned, relaxed, "explicit override", allow_relaxation=True
    )


def test_update_recipe_rotates_between_rounds(versioned, tmp_path):
    staged = tmp_path / "recipe2.json"
    sha = write(staged, json.dumps({"training": {"max_updates": 20}}))
    laya_campaign.prepare_round(versioned, "r1", "d1", "try")
    with pytest.raises(ValueError, match="while a round is active"):
        laya_campaign.update_campaign_recipe(versioned, str(staged), sha, "r2")
    laya_campaign.advance(versioned, "r1", "failed", reason="fixture")
    with pytest.raises(ValueError, match="bytes mismatch"):
        laya_campaign.update_campaign_recipe(
            versioned, str(staged), "0" * 64, "r2"
        )
    stored = laya_campaign.update_campaign_recipe(
        versioned, str(staged), sha, "r2 pools"
    )
    assert stored["recipe"]["sha256"] == sha
    laya_campaign.prepare_round(versioned, "r2", "d1", "again")
    bound = laya_campaign.read_round(versioned, "r2")["bindings"]["recipe"]
    assert bound["sha256"] == sha


def test_prepare_freezes_round_recipe_with_reference(versioned):
    receipt = laya_campaign.prepare_round(versioned, "r1", "d1", "try")
    bound = receipt["bindings"]["round_recipe"]
    path = Path(bound["path"])
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == bound["sha256"]
    spec = json.loads(path.read_text())
    assert spec["training"] == {"max_updates": 10}
    reference = spec["selection_reference"]
    laya_training.check_selection_reference(reference)
    assert reference["policy"] == policy()["research"]
    assert reference["report_digest"] == digest(results())
    assert reference["transfer"]["false_safe_count"] == 3
    assert reference["retention"]["false_safe_count"] == 1
    assert reference["retention"]["accuracy"] == pytest.approx(1.0)
    assert laya_campaign.prepare_round(versioned, "r1", "d1", "try") == receipt


def test_verify_round_bindings_matches_runner_check(tmp_path):
    root, _ = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "try")
    bind_profile_recipe(tmp_path, root, "r1")
    profile = json.loads((tmp_path / "profile.json").read_text())
    laya_campaign.verify_round_bindings(root, "r1", profile)
    profile["model"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="bound incumbent"):
        laya_campaign.verify_round_bindings(root, "r1", profile)


def test_leaderboard_records_terminal_rounds(versioned, tmp_path):
    _evaluated(versioned, tmp_path, promoted_results())
    with pytest.raises(ValueError, match="terminal rounds"):
        laya_campaign.record_leaderboard_entry(versioned, "r1")
    laya_campaign.decide(versioned, "r1")
    laya_campaign.close_round(versioned, "r1", "promoted")
    entry = laya_campaign.record_leaderboard_entry(versioned, "r1")
    assert entry["phase"] == "done"
    assert entry["summary"]["new_coverage"]["correct"] == 27
    assert entry["decision"]["promote_research"] is True
    again = laya_campaign.record_leaderboard_entry(versioned, "r1")
    assert again == entry
    assert (
        len((Path(versioned) / "leaderboard.jsonl").read_text().splitlines())
        == 1
    )
    laya_campaign.prepare_round(versioned, "r2", "d1", "again")
    laya_campaign.advance(versioned, "r2", "failed", reason="fixture")
    failed = laya_campaign.record_leaderboard_entry(versioned, "r2")
    assert failed["summary"] == {} and failed["decision"] is None


def test_adapter_refuses_inconsistent_blocks():
    good = block(
        [[8, 1, 1], [1, 8, 1], [1, 0, 9]],
        (8, 10),
        {"w1": {"false_safe_count": 1, "violation_count": 4}},
    )
    tampered = dict(good, accuracy=0.5)
    with pytest.raises(ValueError, match="disagrees"):
        selection_inputs_from_report(
            tampered,
            good,
            populations=populations(),
            inference={"digest": INF_DIGEST},
        )
    tampered = dict(good, per_family={})
    with pytest.raises(ValueError, match="Per-family"):
        selection_inputs_from_report(
            tampered,
            good,
            populations=populations(),
            inference={"digest": INF_DIGEST},
        )
    no_pairs = dict(good, complete_pairs=0)
    with pytest.raises(ValueError, match="Pair total"):
        selection_inputs_from_report(
            no_pairs,
            good,
            populations=populations(),
            inference={"digest": INF_DIGEST},
        )


def test_candidate_export_and_receipt_are_validated(versioned, tmp_path):
    campaign = versioned
    campaign_dir = Path(campaign)
    laya_campaign.prepare_round(campaign, "r1", "d1", "try")
    for phase in (
        "validate_data",
        "preflight",
        "acquire",
        "submit",
        "observe",
    ):
        laya_campaign.advance(campaign, "r1", phase)
    with pytest.raises(ValueError, match="export is missing"):
        laya_campaign.record_candidate(
            campaign,
            "r1",
            str(tmp_path / "absent"),
            results(),
            {"path": str(tmp_path / "receipt.json"), "sha256": "0" * 64},
        )
    export = tmp_path / "candidate"
    export.mkdir()
    with pytest.raises(ValueError, match="lacks weights"):
        laya_campaign.record_candidate(
            campaign,
            "r1",
            str(export),
            results(),
            {"path": str(tmp_path / "receipt.json"), "sha256": "0" * 64},
        )
    (export / "model.safetensors").write_text("weights")
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="receipt bytes"):
        laya_campaign.record_candidate(
            campaign,
            "r1",
            str(export),
            results(),
            {"path": str(receipt), "sha256": "0" * 64},
        )
    assert campaign_dir.exists()


def _crash_after(calls, monkeypatch):
    real = laya_campaign.atomic_json
    state = {"count": 0}

    def flaky(path, value):
        state["count"] += 1
        if state["count"] == calls:
            raise RuntimeError("injected crash")
        return real(path, value)

    monkeypatch.setattr(laya_campaign, "atomic_json", flaky)


@pytest.mark.parametrize("crash_at", [1, 2, 3, 4])
def test_interrupted_promotion_retries_same_decision(
    versioned, tmp_path, monkeypatch, crash_at
):
    campaign = versioned
    before = laya_campaign.load_incumbent(campaign)
    _evaluated(campaign, tmp_path, promoted_results())
    _crash_after(crash_at, monkeypatch)
    with pytest.raises(RuntimeError, match="injected crash"):
        laya_campaign.decide(campaign, "r1")
    monkeypatch.undo()
    receipt = laya_campaign.decide(campaign, "r1")
    assert receipt["decision"]["promote_research"] is True
    assert receipt["previous_incumbent"] == before
    incumbent = laya_campaign.load_incumbent(campaign)
    assert incumbent["round_id"] == "r1"
    assert "candidate" in incumbent["export"]
    assert not (campaign / "rounds/r1/decision-intent.json").exists()


@pytest.mark.parametrize("crash_at", [2, 3])
def test_interrupted_close_retries_into_done(
    versioned, tmp_path, monkeypatch, crash_at
):
    campaign = versioned
    _evaluated(campaign, tmp_path, promoted_results())
    laya_campaign.decide(campaign, "r1")
    _crash_after(crash_at, monkeypatch)
    with pytest.raises(RuntimeError, match="injected crash"):
        laya_campaign.close_round(campaign, "r1", "promoted")
    monkeypatch.undo()
    closed = laya_campaign.close_round(campaign, "r1", "promoted")
    assert closed["phase"] == "done"
    assert closed["report"]["summary"] == "promoted"


@pytest.mark.parametrize("crash_at", [1, 2, 3])
def test_interrupted_retention_retries_same_decision(
    versioned, tmp_path, monkeypatch, crash_at
):
    campaign = versioned
    before = laya_campaign.load_incumbent(campaign)
    _evaluated(
        campaign,
        tmp_path,
        results(
            retention_view=retention(false_safe=2),
            coverage_view=coverage(correct=25),
        ),
    )
    _crash_after(crash_at, monkeypatch)
    with pytest.raises(RuntimeError, match="injected crash"):
        laya_campaign.decide(campaign, "r1")
    monkeypatch.undo()
    receipt = laya_campaign.decide(campaign, "r1")
    assert receipt["decision"]["promote_research"] is False
    assert laya_campaign.load_incumbent(campaign) == before
    assert not (campaign / "rounds/r1/decision-intent.json").exists()


def test_dataset_reregistration_with_new_hash_is_refused(versioned, tmp_path):
    manifest = tmp_path / "datasets/versions/d1/manifest.json"
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert (
        laya_campaign.register_dataset(versioned, "d1", str(manifest), sha)[
            "sha256"
        ]
        == sha
    )
    with pytest.raises(ValueError, match="new hash"):
        laya_campaign.register_dataset(versioned, "d1", str(manifest), "0" * 64)


class RunnerJobs:
    """Fake job/systemd backend behind the real Operator.

    The Operator runs unmodified; only process launch, the systemd
    runner, and GPU probing are simulated. Job completion artifacts are
    written by the test, exactly as a finished owned job would leave
    them, so observation and finalization exercise the real shapes.
    """

    def __init__(self, profile):
        self.profile = profile
        self.units = {}
        self.launches = []
        self.crash_before_launch = False
        self.crash_after_launch = False
        self.gpu_free = True
        self.gpu_script = []

    def gpu_available(self):
        if self.gpu_script:
            return self.gpu_script.pop(0)
        return self.gpu_free

    def status(self, experiment_id):
        output = Path(self.profile["runs"]) / experiment_id
        return {
            name: json.loads((output / name).read_text())
            for name in ("job.json", "execution.json", "report.json")
            if (output / name).exists()
        }

    def launch_owned(self, manifest, path, resources, action):
        if self.crash_before_launch:
            self.crash_before_launch = False
            raise RuntimeError("submitter crash")
        self.launches.append((list(manifest["command"]), action))
        output = Path(manifest["output"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "job.json").write_text(json.dumps({"phase": "submitted"}))
        self.units[manifest["unit"]] = {
            "LoadState": "loaded",
            "ActiveState": "active",
            "Result": "success",
            "Description": "siliconshield:" + manifest["idempotency_key"],
        }
        if self.crash_after_launch:
            self.crash_after_launch = False
            raise RuntimeError("submitter crash")
        return {"phase": "submitted"}

    def runner(self, argv):
        return self.units.get(argv[3], {"LoadState": "not-found"})

    def complete_unit(self):
        unit = next(iter(self.units))
        show = dict(self.units[unit])
        show["ActiveState"] = "inactive"
        self.units[unit] = show


def runner_infra(tmp_path):
    """Build the source/model/data tree a real preflight requires."""
    infra = tmp_path / "infra"
    folders = (
        "source/siliconshield/training/laya",
        "source/siliconshield/research_loop/episodes",
        "source/siliconshield/research_loop/execution",
        "model",
        "data",
        "runs",
        "resources",
    )
    for folder in folders:
        (infra / folder).mkdir(parents=True, exist_ok=True)
    trainer = infra / "source/siliconshield/training/laya/security_train.py"
    trainer.write_text("")
    base = infra / "source/siliconshield/research_loop"
    for package in (
        infra / "source/siliconshield/__init__.py",
        base / "__init__.py",
        base / "episodes/__init__.py",
        base / "execution/__init__.py",
    ):
        package.write_text("")
    (base / "episodes/contracts.py").write_text(
        "def atomic_json(*args):\n    return None\n\n\n"
        "def read_json(*args):\n    return {}\n"
    )
    (base / "execution/owned_job.py").write_text(
        "from siliconshield.research_loop.episodes.contracts import (\n"
        "    atomic_json,\n"
        "    read_json,\n"
        ")\n"
    )
    for split in ("train", "dev"):
        (infra / "data" / (split + ".jsonl")).write_text("{}\n")
    (infra / "model" / "config.json").write_text("{}")
    return infra


def runner_profile(infra):
    return {
        "python": sys.executable,
        "source": str(infra / "source"),
        "model": str(infra / "model"),
        "data": str(infra / "data"),
        "runs": str(infra / "runs"),
        "resource_root": str(infra / "resources"),
        "defaults": {"max_updates": 10},
    }


def operator_factory_for(jobs):
    """Build real Operators over one shared fake backend."""

    def factory(profile_path):
        jobs.profile = json.loads(Path(profile_path).read_text())
        return Operator(None, jobs=jobs, runner=jobs.runner)

    return factory


def trainer_report(experiment_id):
    block = {"n": 30, "accuracy": 0.9, "nll": 0.2}
    return {
        "schema": 2,
        "run_id": experiment_id,
        "signature": {"data": {"train": "aa", "dev": "bb"}},
        "development": dict(block),
        "initial_development": dict(block),
        "progress": {"step": 3},
    }


def complete_job(profile, experiment_id, identity=None):
    """Leave finished-job artifacts, as the owned worker would."""
    output = Path(profile["runs"]) / experiment_id
    output.mkdir(parents=True, exist_ok=True)
    key = identity
    if key is None:
        key = idempotency_key(profile, experiment_id, {})
    (output / "execution.json").write_text(
        json.dumps({"identity": key, "phase": "completed", "started": 1})
    )
    (output / "report.json").write_text(
        json.dumps(trainer_report(experiment_id))
    )


def candidate_evaluator(tmp_path):
    def evaluator(root, round_id, job):
        assert job["terminal"] == "complete"
        transfer, retention_block = candidate_blocks()
        export = tmp_path / "candidate"
        export.mkdir(exist_ok=True)
        (export / "model.safetensors").write_text("candidate-weights")
        normalized = selection_inputs_from_report(
            transfer,
            retention_block,
            populations=populations_for_campaign(),
            inference={"digest": INF_DIGEST},
        )
        receipt = tmp_path / ("receipt-" + round_id + ".json")
        receipt_sha = binding_receipt(
            receipt, export, normalized, "demo-" + round_id, round_id
        )
        return {
            "export": str(export),
            "results": normalized,
            "receipt": {"path": str(receipt), "sha256": receipt_sha},
        }

    return evaluator


def runner_campaign(tmp_path, pools=None):
    export = tmp_path / "parent"
    export.mkdir()
    weights = write(export / "model.safetensors")
    recipe = tmp_path / "recipe.json"
    recipe_sha = write(recipe, json.dumps({"training": {"max_updates": 10}}))
    receipt = tmp_path / "receipt.json"
    infra = runner_infra(tmp_path)
    receipt_sha = binding_receipt(
        receipt, export, results(), "genesis-eval", None
    )
    (infra / "data/train.jsonl").write_text('{"id": "t1"}\n{"id": "t2"}\n')
    (infra / "data/dev.jsonl").write_text('{"id": "d1"}\n')
    profile = runner_profile(infra)
    profile["model"] = str(export)
    profile["recipe"] = str(recipe)
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    root = tmp_path / "campaign"
    laya_campaign.init_campaign(
        root,
        {
            "campaign_id": "demo",
            "parent": {"export": str(export), "weights_sha256": weights},
            "recipe": {"path": str(recipe), "sha256": recipe_sha},
            "selection": policy(),
            "operator_profile": str(tmp_path / "profile.json"),
            "incumbent": {
                "export": str(export),
                "results": results(),
                "receipt": {"path": str(receipt), "sha256": receipt_sha},
            },
        },
    )
    datasets_root = tmp_path / "datasets"
    manifest = dataset_versions.write_version(
        datasets_root,
        "d1",
        {
            "train": str(infra / "data/train.jsonl"),
            "dev": str(infra / "data/dev.jsonl"),
        },
    )
    if pools is not None:
        pools_path = tmp_path / "pools.json"
        dataset_versions.assign_pools(
            infra / "data/train.jsonl", pools, pools_path
        )
        manifest = dataset_versions.bind_pools(datasets_root, "d1", pools_path)
        profile["defaults"]["pools"] = str(pools_path)
        (tmp_path / "profile.json").write_text(json.dumps(profile))
    from siliconshield.schemas import DATASET_AUDIT_SCHEMA

    dataset_versions.accept_version(
        datasets_root,
        "d1",
        {
            "schema": DATASET_AUDIT_SCHEMA,
            "splits": {
                role: entry["sha256"]
                for role, entry in manifest["splits"].items()
            },
            "cross_split_exact_rtl_overlap": 0,
            "retired_rtl_overlap": 0,
            "truncated": 0,
            "lineage_overlap": 0,
            "canonical_overlap": 0,
            "max_group_share": 0.05,
            "max_mechanism_share": 0.15,
            "review_ledger_sha256": "ab" * 32,
            "reviewed_templates": 1,
            "release_class": "narrow-pilot",
        },
    )
    manifest_path = tmp_path / "datasets/versions/d1/manifest.json"
    laya_campaign.register_dataset(
        root,
        "d1",
        str(manifest_path),
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    return root, profile


def candidate_blocks():
    transfer = block(
        [[9, 1, 0], [0, 9, 1], [0, 0, 10]],
        (9, 10),
        {
            "w1": {"false_safe_count": 0, "violation_count": 6},
            "w2": {"false_safe_count": 0, "violation_count": 4},
        },
    )
    retention_block = block(
        [[20, 0, 0], [1, 19, 0], [0, 0, 20]],
        (19, 20),
        {"w0": {"false_safe_count": 1, "violation_count": 20}},
    )
    return transfer, retention_block


def populations_for_campaign():
    # Same registered populations the incumbent was evaluated on.
    transfer = {
        "dataset_version": "d1",
        "split": "dev-transfer",
        "sha256": POP_SHA,
    }
    retention = {
        "dataset_version": "d1",
        "split": "dev-retention",
        "sha256": POP_SHA,
    }
    return {"transfer": transfer, "retention": retention}


def fake_laya_import(monkeypatch):
    monkeypatch.setattr(
        "siliconshield.research_loop.execution.operator.verify_laya",
        lambda *args: True,
    )


def test_runner_completes_a_round_end_to_end(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    jobs = RunnerJobs(profile)
    factory = operator_factory_for(jobs)
    evaluator = candidate_evaluator(tmp_path)

    waiting = laya_campaign.run_round(
        root, "r1", operator_factory=factory, evaluator=evaluator
    )
    assert waiting["phase"] == "observe"
    assert len(jobs.launches) == 1

    complete_job(profile, "demo-r1")
    jobs.complete_unit()
    final = laya_campaign.run_round(
        root, "r1", operator_factory=factory, evaluator=evaluator
    )
    assert final["phase"] == "done"
    assert final["decision"]["promote_research"] is True
    assert laya_campaign.load_incumbent(root)["round_id"] == "r1"
    assert len(jobs.launches) == 1


def test_runner_resumes_after_submission_crash(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    jobs = RunnerJobs(profile)
    jobs.crash_after_launch = True
    factory = operator_factory_for(jobs)
    evaluator = candidate_evaluator(tmp_path)

    with pytest.raises(RuntimeError, match="submitter crash"):
        laya_campaign.run_round(
            root, "r1", operator_factory=factory, evaluator=evaluator
        )
    assert len(jobs.launches) == 1
    complete_job(profile, "demo-r1")
    jobs.complete_unit()
    final = laya_campaign.run_round(
        root, "r1", operator_factory=factory, evaluator=evaluator
    )
    assert final["phase"] == "done"
    submits = [
        entry for entry in final["history"] if entry["phase"] == "submit"
    ]
    assert len(submits) == 1
    assert len(jobs.launches) == 1


def test_runner_resumes_after_evaluator_crash(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    jobs = RunnerJobs(profile)
    factory = operator_factory_for(jobs)
    waiting = laya_campaign.run_round(
        root,
        "r1",
        operator_factory=factory,
        evaluator=candidate_evaluator(tmp_path),
    )
    assert waiting["phase"] == "observe"
    complete_job(profile, "demo-r1")
    jobs.complete_unit()
    calls = {"count": 0}
    inner = candidate_evaluator(tmp_path)

    def flaky(root_arg, round_id, job):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("evaluator crash")
        return inner(root_arg, round_id, job)

    with pytest.raises(RuntimeError, match="evaluator crash"):
        laya_campaign.run_round(
            root, "r1", operator_factory=factory, evaluator=flaky
        )
    assert len(jobs.launches) == 1
    operator_manifest = json.loads(
        (tmp_path / "infra/runs/demo-r1/run-manifest.json").read_text()
    )
    assert operator_manifest["phase"] == "report"
    final = laya_campaign.run_round(
        root, "r1", operator_factory=factory, evaluator=flaky
    )
    assert final["phase"] == "done"
    assert calls["count"] == 2
    assert len(jobs.launches) == 1


def test_runner_fails_cleanly_when_submit_crashed_before_launch(
    tmp_path, monkeypatch
):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    jobs = RunnerJobs(profile)
    jobs.crash_before_launch = True
    factory = operator_factory_for(jobs)
    evaluator = candidate_evaluator(tmp_path)

    with pytest.raises(RuntimeError, match="submitter crash"):
        laya_campaign.run_round(
            root, "r1", operator_factory=factory, evaluator=evaluator
        )
    assert jobs.launches == []
    final = laya_campaign.run_round(
        root, "r1", operator_factory=factory, evaluator=evaluator
    )
    assert final["phase"] == "failed"
    assert jobs.launches == []


def test_runner_waits_for_a_running_job(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    jobs = RunnerJobs(profile)
    calls = []

    def evaluator(r, rid, job):
        calls.append(rid)
        raise AssertionError("must not evaluate a running job")

    factory = operator_factory_for(jobs)
    waiting = laya_campaign.run_round(
        root, "r1", operator_factory=factory, evaluator=evaluator
    )
    assert waiting["phase"] == "observe"
    assert calls == []
    assert len(jobs.launches) == 1


def test_runner_rejects_a_report_from_the_wrong_run(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    jobs = RunnerJobs(profile)
    factory = operator_factory_for(jobs)
    evaluator = candidate_evaluator(tmp_path)

    complete_job(profile, "demo-r1", identity="0" * 64)
    with pytest.raises(ValueError, match="identity"):
        laya_campaign.run_round(
            root, "r1", operator_factory=factory, evaluator=evaluator
        )
    assert jobs.launches == []


def test_runner_resumes_from_report_phase(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    jobs = RunnerJobs(profile)
    factory = operator_factory_for(jobs)
    evaluator = candidate_evaluator(tmp_path)

    waiting = laya_campaign.run_round(
        root, "r1", operator_factory=factory, evaluator=evaluator
    )
    assert waiting["phase"] == "observe"
    complete_job(profile, "demo-r1")
    jobs.complete_unit()

    real_advance = laya_campaign.advance

    def crash_on_done(root_arg, round_arg, phase, **kwargs):
        if phase == "done":
            raise RuntimeError("crash between report and done")
        return real_advance(root_arg, round_arg, phase, **kwargs)

    monkeypatch.setattr(laya_campaign, "advance", crash_on_done)
    with pytest.raises(RuntimeError, match="between report and done"):
        laya_campaign.run_round(
            root, "r1", operator_factory=factory, evaluator=evaluator
        )
    monkeypatch.setattr(laya_campaign, "advance", real_advance)
    interrupted = json.loads(
        (root / "rounds" / "r1" / "round.json").read_text()
    )
    assert interrupted["phase"] == "report"
    final = laya_campaign.run_round(
        root, "r1", operator_factory=factory, evaluator=evaluator
    )
    assert final["phase"] == "done"
    assert final["report"]["summary"] == "promoted"


def rewrite_profile(tmp_path, mutate):
    path = tmp_path / "profile.json"
    profile = json.loads(path.read_text())
    mutate(profile)
    path.write_text(json.dumps(profile))


def bind_profile_recipe(tmp_path, root, round_id):
    """Repoint the job profile at the round's frozen recipe.

    Mirrors the supervisor commit step after preparing a round:
    the profile's recipe must name the round recipe carrying the
    frozen selection reference before the runner launches.
    Returns the bound path so callers can sync in-memory copies
    that feed the job identity.
    """
    bound = laya_campaign.read_round(root, round_id)["bindings"]["round_recipe"]
    rewrite_profile(tmp_path, lambda p: p.update(recipe=bound["path"]))
    return bound["path"]


def test_runner_refuses_a_changed_policy(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    stored = json.loads((root / "campaign.json").read_text())
    stored["selection"]["research"]["min_new_coverage_gain_pp"] = 99.0
    (root / "campaign.json").write_text(json.dumps(stored))
    jobs = RunnerJobs(profile)
    with pytest.raises(ValueError, match="re-specify"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=operator_factory_for(jobs),
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


def drift_policy(root):
    stored = json.loads((root / "campaign.json").read_text())
    stored["selection"]["research"]["min_new_coverage_gain_pp"] = 99.0
    (root / "campaign.json").write_text(json.dumps(stored))


def test_runner_rechecks_policy_when_leaving_preflight(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    jobs = RunnerJobs(profile)
    # Free for operator preflight, busy at the campaign GPU gate.
    jobs.gpu_script = [True, False]
    factory = operator_factory_for(jobs)
    waiting = laya_campaign.run_round(
        root,
        "r1",
        operator_factory=factory,
        evaluator=candidate_evaluator(tmp_path),
    )
    assert jobs.gpu_script == []
    assert waiting["phase"] == "preflight"
    assert jobs.launches == []
    drift_policy(root)
    with pytest.raises(ValueError, match="re-specify"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=factory,
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


def test_runner_rechecks_policy_when_resuming_acquire(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    jobs = RunnerJobs(profile)
    jobs.crash_before_launch = True
    factory = operator_factory_for(jobs)
    with pytest.raises(RuntimeError, match="submitter crash"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=factory,
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []
    interrupted = json.loads((root / "rounds/r1/round.json").read_text())
    assert interrupted["phase"] == "acquire"
    drift_policy(root)
    with pytest.raises(ValueError, match="re-specify"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=factory,
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


def test_runner_refuses_a_stale_model(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    rewrite_profile(
        tmp_path, lambda p: p.update(model=str(tmp_path / "infra/model"))
    )
    jobs = RunnerJobs(profile)
    with pytest.raises(ValueError, match="bound incumbent"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=operator_factory_for(jobs),
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


def test_runner_refuses_swapped_training_data(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    decoy = tmp_path / "decoy-data"
    decoy.mkdir()
    (decoy / "train.jsonl").write_text('{"id": "evil"}\n')
    (decoy / "dev.jsonl").write_text('{"id": "evil-dev"}\n')
    rewrite_profile(tmp_path, lambda p: p.update(data=str(decoy)))
    jobs = RunnerJobs(profile)
    with pytest.raises(ValueError, match="bound train split"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=operator_factory_for(jobs),
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


def test_runner_refuses_a_missing_recipe(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    rewrite_profile(tmp_path, lambda p: p.pop("recipe"))
    jobs = RunnerJobs(profile)
    with pytest.raises(ValueError, match="bound recipe"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=operator_factory_for(jobs),
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


def test_runner_refuses_a_retuned_setting(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    rewrite_profile(tmp_path, lambda p: p["defaults"].update(max_updates=99))
    jobs = RunnerJobs(profile)
    with pytest.raises(ValueError, match="bound recipe"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=operator_factory_for(jobs),
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


def test_runner_refuses_an_unbound_setting(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    rewrite_profile(tmp_path, lambda p: p["defaults"].update(seed=7))
    jobs = RunnerJobs(profile)
    with pytest.raises(ValueError, match="missing from the bound recipe"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=operator_factory_for(jobs),
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


def test_runner_refuses_training_overrides(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(
        root, "r1", "d1", "synthetic try", overrides={"max_updates": 5}
    )
    jobs = RunnerJobs(profile)
    with pytest.raises(ValueError, match="bound recipe"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=operator_factory_for(jobs),
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


def test_runner_refuses_a_decoy_recipe(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    decoy = tmp_path / "decoy-recipe.json"
    decoy.write_text('{"other": true}')
    rewrite_profile(tmp_path, lambda p: p.update(recipe=str(decoy)))
    jobs = RunnerJobs(profile)
    with pytest.raises(ValueError, match="bound recipe"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=operator_factory_for(jobs),
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


def test_runner_refuses_registered_recipe_without_repoint(
    tmp_path, monkeypatch
):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    jobs = RunnerJobs(profile)
    with pytest.raises(ValueError, match="bound recipe"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=operator_factory_for(jobs),
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


def pools_spec():
    return {
        "a": {"rows": ["t1"], "weight": 1.0},
        "b": {"rows": ["t2"], "weight": 3.0},
    }


def test_runner_completes_a_round_with_bound_pools(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path, pools=pools_spec())
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    jobs = RunnerJobs(profile)
    factory = operator_factory_for(jobs)
    evaluator = candidate_evaluator(tmp_path)
    waiting = laya_campaign.run_round(
        root, "r1", operator_factory=factory, evaluator=evaluator
    )
    assert waiting["phase"] == "observe"
    command, _ = jobs.launches[0]
    assert command[command.index("--pools") + 1] == str(tmp_path / "pools.json")
    complete_job(profile, "demo-r1")
    jobs.complete_unit()
    final = laya_campaign.run_round(
        root, "r1", operator_factory=factory, evaluator=evaluator
    )
    assert final["phase"] == "done"


def test_runner_refuses_missing_pools(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path, pools=pools_spec())
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    rewrite_profile(tmp_path, lambda p: p["defaults"].pop("pools"))
    jobs = RunnerJobs(profile)
    with pytest.raises(ValueError, match="bound sampling pools"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=operator_factory_for(jobs),
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


def test_runner_refuses_wrong_pools(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path, pools=pools_spec())
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    decoy = tmp_path / "decoy-pools.json"
    decoy.write_text("{}")
    rewrite_profile(tmp_path, lambda p: p["defaults"].update(pools=str(decoy)))
    jobs = RunnerJobs(profile)
    with pytest.raises(ValueError, match="bound pools"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=operator_factory_for(jobs),
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


def test_runner_refuses_unbound_pools(tmp_path, monkeypatch):
    fake_laya_import(monkeypatch)
    root, profile = runner_campaign(tmp_path)
    laya_campaign.prepare_round(root, "r1", "d1", "synthetic try")
    profile["recipe"] = bind_profile_recipe(tmp_path, root, "r1")
    extra = tmp_path / "extra-pools.json"
    extra.write_text("{}")
    rewrite_profile(tmp_path, lambda p: p["defaults"].update(pools=str(extra)))
    jobs = RunnerJobs(profile)
    with pytest.raises(ValueError, match="did not bind"):
        laya_campaign.run_round(
            root,
            "r1",
            operator_factory=operator_factory_for(jobs),
            evaluator=candidate_evaluator(tmp_path),
        )
    assert jobs.launches == []


OT_SHA = "ee" * 32


def expansion_policy():
    found = policy()
    found["research"]["expansion_views"] = {
        "ot": {
            "population_sha256": OT_SHA,
            "min_correct": 11,
            "max_false_safe": 1,
            "max_group_false_safe": {"g1": 0, "g2": 1},
        }
    }
    return found


def ot_view(correct=11, false_safe=1, groups=None, sha=OT_SHA, **extra):
    return view(
        12,
        correct,
        false_safe,
        4,
        (4, 4),
        (3, 4),
        ((4, 4), (3, 4), (4, 4)),
        ("g0", 1, 2),
        "ot-dev",
        sha=sha,
        families=(
            groups
            if groups is not None
            else {
                "g1": {"false_safe": 0, "violations": 2},
                "g2": {"false_safe": 1, "violations": 2},
            }
        ),
        **extra,
    )


def passing_candidate(extra_view=None):
    built = results(
        coverage_view=coverage(correct=26, worst=("w1", 1, 6)),
    )
    if extra_view is not None:
        built["ot"] = extra_view
    return built


def test_expansion_view_gate_passes():
    verdict = laya_campaign.apply_selection_policy(
        expansion_policy(), results(), passing_candidate(ot_view())
    )
    assert verdict["promote_research"] is True
    assert all(
        "expansion view" not in reason for reason in verdict["research_reasons"]
    )


def test_expansion_view_correct_below():
    verdict = laya_campaign.apply_selection_policy(
        expansion_policy(), results(), passing_candidate(ot_view(correct=10))
    )
    assert verdict["promote_research"] is False
    assert any(
        "correct 10 below 11" in reason
        for reason in verdict["research_reasons"]
    )


def test_expansion_view_false_safe_above():
    verdict = laya_campaign.apply_selection_policy(
        expansion_policy(),
        results(),
        passing_candidate(ot_view(false_safe=2)),
    )
    assert verdict["promote_research"] is False
    assert any(
        "false-safe errors 2 above cap 1" in reason
        for reason in verdict["research_reasons"]
    )


def test_expansion_view_group_above():
    groups = {
        "g1": {"false_safe": 1, "violations": 2},
        "g2": {"false_safe": 0, "violations": 2},
    }
    verdict = laya_campaign.apply_selection_policy(
        expansion_policy(),
        results(),
        passing_candidate(ot_view(groups=groups)),
    )
    assert verdict["promote_research"] is False
    assert any(
        "group g1 false-safe errors 1 above cap 0" in reason
        for reason in verdict["research_reasons"]
    )


def test_expansion_view_missing_group():
    verdict = laya_campaign.apply_selection_policy(
        expansion_policy(),
        results(),
        passing_candidate(
            ot_view(groups={"g1": {"false_safe": 0, "violations": 2}})
        ),
    )
    assert verdict["promote_research"] is False
    assert any(
        "lacks gated group: g2" in reason
        for reason in verdict["research_reasons"]
    )


def test_expansion_view_missing_refuses():
    verdict = laya_campaign.apply_selection_policy(
        expansion_policy(), results(), passing_candidate()
    )
    assert verdict["promote_research"] is False
    assert any(
        "lacks required expansion view" in reason
        for reason in verdict["research_reasons"]
    )


def test_expansion_view_wrong_population_refuses():
    with pytest.raises(ValueError, match="another population"):
        laya_campaign.apply_selection_policy(
            expansion_policy(),
            results(),
            passing_candidate(ot_view(sha=POP_SHA)),
        )


def test_expansion_view_policy_invalid():
    bad = expansion_policy()
    del bad["research"]["expansion_views"]["ot"]["population_sha256"]
    with pytest.raises(ValueError, match="population sha256"):
        laya_campaign.apply_selection_policy(
            bad, results(), passing_candidate(ot_view())
        )
    bad = expansion_policy()
    bad["research"]["expansion_views"]["ot"]["min_correct"] = True
    with pytest.raises(ValueError, match="min_correct"):
        laya_campaign.apply_selection_policy(
            bad, results(), passing_candidate(ot_view())
        )
    bad = expansion_policy()
    bad["research"]["expansion_views"]["ot"]["max_group_false_safe"] = {
        "g1": -1
    }
    with pytest.raises(ValueError, match="group cap"):
        laya_campaign.apply_selection_policy(
            bad, results(), passing_candidate(ot_view())
        )


def test_expansion_view_ignored_without_policy():
    verdict = laya_campaign.apply_selection_policy(
        policy(), results(), passing_candidate(ot_view(correct=0))
    )
    assert verdict["promote_research"] is True


def test_expansion_view_update_comparison():
    plain = policy()
    gated = expansion_policy()
    assert laya_campaign.selection_relaxes(plain, gated) is False
    assert laya_campaign.selection_relaxes(gated, plain) is True
    tighter = expansion_policy()
    tighter["research"]["expansion_views"]["ot"]["min_correct"] = 12
    assert laya_campaign.selection_relaxes(gated, tighter) is False
    weaker = expansion_policy()
    weaker["research"]["expansion_views"]["ot"]["min_correct"] = 10
    assert laya_campaign.selection_relaxes(gated, weaker) is True
    weaker = expansion_policy()
    weaker["research"]["expansion_views"]["ot"]["max_false_safe"] = 2
    assert laya_campaign.selection_relaxes(gated, weaker) is True
    weaker = expansion_policy()
    weaker["research"]["expansion_views"]["ot"]["max_group_false_safe"] = {
        "g1": 1,
        "g2": 1,
    }
    assert laya_campaign.selection_relaxes(gated, weaker) is True
    weaker = expansion_policy()
    del weaker["research"]["expansion_views"]["ot"]["max_group_false_safe"][
        "g2"
    ]
    assert laya_campaign.selection_relaxes(gated, weaker) is True
    weaker = expansion_policy()
    weaker["research"]["expansion_views"]["ot"]["population_sha256"] = POP_SHA
    assert laya_campaign.selection_relaxes(gated, weaker) is True
