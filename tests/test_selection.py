"""Checkpoint selection, development views, independence gates, reporting, metrics, and DDP guard tests: synthetic fixtures."""

import copy
import fcntl
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from siliconshield.research_loop.controller.client import run_agent
from siliconshield.research_loop.controller.inference_metrics import (
    GenerationStreamer,
    TokenTiming,
)
from siliconshield.research_loop.controller.serve_fast import ServerConfig
from siliconshield.research_loop.execution.security_jobs import command_for
from siliconshield.schemas import CHECKPOINT_OBJECTIVE_SCHEMA
from siliconshield.training.laya import development_views as views
from siliconshield.training.laya.checkpoint_selection import (
    objective_selection,
    population_indices,
    view_counts,
)
from siliconshield.training.laya.dataset_versions import (
    accept_version,
    audit_canonical_overlap,
    audit_concentration,
    audit_lineage_overlap,
    audit_split_groups,
    canonical_rtl_key,
    write_version,
)
from siliconshield.training.laya.ddp_guard import (
    require_coherent_training,
    require_single_backward,
)
from siliconshield.training.laya.development_views import (
    attach_views,
    selection,
)
from siliconshield.training.laya.security_reporting import (
    LABELS,
    evaluation_report,
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


def test_new_task_gain_is_visible_to_real_selection():
    initial = _selection_result(
        _selection_block([[0, 3, 0], [0, 3, 0], [0, 0, 3]], 0)
    )
    improved = _selection_result(
        _selection_block([[2, 1, 0], [0, 3, 0], [0, 0, 3]], 2)
    )
    config = {"checkpoint_objective": _selection_policy()}
    assert selection(improved, config, 1.0, None) < selection(
        initial, config, 1.0, None
    )
    assert improved["checkpoint_acceptance"]["quality_targets_passed"]
    assert not initial["checkpoint_acceptance"]["quality_targets_passed"]


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


def test_missing_context_gain_does_not_satisfy_readiness():
    candidate = _selection_result(
        _selection_block([[0, 3, 0], [0, 3, 0], [0, 0, 3]], 0)
    )
    objective_selection(candidate, _selection_policy())
    assert candidate["checkpoint_acceptance"]["protected_passed"]
    assert candidate["checkpoint_acceptance"]["quality_target_failures"] == [
        "class_0",
        "min_pairs_correct",
    ]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
def test_nonfinite_or_invalid_nll_refuses(bad):
    metrics = _selection_block([[3, 0, 0], [0, 3, 0], [0, 0, 3]], 3)
    metrics["nll"] = bad
    with pytest.raises(ValueError, match="NLL"):
        view_counts(metrics)


def test_corrupt_counts_and_unprotected_views_refuse():
    metrics = _selection_block([[3, 0, 0], [0, 3, 0], [0, 0, 3]], 3)
    altered = copy.deepcopy(metrics)
    altered["false_safe_count"] = 1
    with pytest.raises(ValueError, match="confusion"):
        view_counts(altered)
    candidate = _selection_result(metrics)
    candidate["development_views"]["another"] = metrics
    with pytest.raises(ValueError, match="Every development view"):
        objective_selection(candidate, _selection_policy())
    changed_policy = _selection_policy()
    changed_policy["protected"]["old"]["rows"] = 12
    with pytest.raises(ValueError, match="population size"):
        objective_selection(_selection_result(metrics), changed_policy)


def test_source_and_group_selectors_partition_without_relabeling():
    rows = [
        {"source": {"name": "vendor"}, "split_group_id": "old"},
        {"source": {"name": "vendor"}, "split_group_id": "new"},
    ]
    original = copy.deepcopy(rows)
    config = {
        "views": {
            "old": {
                "source_names": ["vendor"],
                "exclude_split_groups": ["new"],
            },
            "new": {"split_groups": ["new"]},
        },
        "selection_view": "new",
        "retention_view": "old",
    }
    values = {}
    attach_views(values, rows, [[1], [2]], config, lambda r, p: p)
    assert values["development_views"] == {"old": [[1]], "new": [[2]]}
    assert rows == original
    assert population_indices(rows, ["vendor"]) == [0, 1]
    config["views"]["old"] = ["vendor"]
    with pytest.raises(ValueError, match="partition"):
        attach_views({}, rows, [[1], [2]], config, lambda r, p: p)


def test_unknown_selector_or_constraint_refuses():
    with pytest.raises(ValueError, match="Unknown development population"):
        population_indices([], {"group": ["typo"]})
    invalid = _selection_policy()
    invalid["protected"]["old"]["max_false_sfae"] = 0
    candidate = _selection_result(
        _selection_block([[3, 0, 0], [0, 3, 0], [0, 0, 3]], 3)
    )
    with pytest.raises(ValueError, match="Unknown development constraint"):
        objective_selection(candidate, invalid)


def test_evaluation_batch_is_forwarded_without_changing_optimizer_batch(
    tmp_path,
):
    profile = {
        "source": str(tmp_path),
        "runs": str(tmp_path / "runs"),
        "python": "python",
        "model": str(tmp_path / "model"),
        "data": str(tmp_path / "data"),
        "defaults": {
            "max_updates": 8,
            "micro_batch": 2,
            "accumulation": 4,
            "eval_batch_size": 8,
        },
    }
    command, _ = command_for(profile, "example", {})
    assert command[command.index("--eval-batch-size") + 1] == "8"
    assert command[command.index("--micro-batch") + 1] == "2"
    assert command[command.index("--accumulation") + 1] == "4"
    with pytest.raises(ValueError, match="positive integer"):
        command_for(profile, "example", {"eval_batch_size": -1})


def test_plateau_exposure_gate_is_forwarded(tmp_path):
    profile = {
        "source": str(tmp_path),
        "runs": str(tmp_path / "runs"),
        "python": "python",
        "model": str(tmp_path / "model"),
        "data": str(tmp_path / "data"),
        "defaults": {
            "max_updates": 8,
            "plateau_requires_full_exposure": True,
        },
    }
    command, _ = command_for(profile, "example", {})
    assert "--plateau-requires-full-exposure" in command
    command, _ = command_for(
        profile, "example", {"plateau_requires_full_exposure": False}
    )
    assert "--plateau-requires-full-exposure" not in command
    with pytest.raises(ValueError, match="Flag setting must be boolean"):
        command_for(profile, "example", {"plateau_requires_full_exposure": 1})


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


def test_legacy_selection_ranks_transfer_nll_only():
    good = _views_result(_views_block(nll=0.002), _views_block(pairs=0.95))
    bad = _views_result(_views_block(nll=0.001), _views_block(pairs=0.90))
    assert views.selection(good, config(), 0.99, base_key) > (
        views.selection(bad, config(), 0.99, base_key)
    )


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


def test_joint_selection_enforces_coverage_gain():
    incumbent = _views_block(accuracy=84 / 90)
    ref = views.build_selection_reference(
        incumbent,
        _views_block(),
        _views_policy(min_new_coverage_gain_pp=4.0),
        "digest",
    )
    cfg = config(selection_reference=ref)
    feasible = _views_result(
        _views_block(accuracy=88 / 90, nll=0.002), _views_block()
    )
    short = _views_result(
        _views_block(accuracy=87 / 90, nll=0.001), _views_block()
    )
    assert views.joint_flags(feasible, cfg)[-1] == 0
    assert views.joint_flags(short, cfg)[-1] == 1
    assert views.selection(feasible, cfg, 0.99, base_key) < views.selection(
        short, cfg, 0.99, base_key
    )
    assert views.selection(
        feasible, config(), 0.99, base_key
    ) > views.selection(short, config(), 0.99, base_key)


def test_v2_ranking_prefers_retention_quality_over_transfer_nll():
    v2 = reference(
        selection_ranking="v2-retention",
        max_retention_false_safe_increase=1,
    )
    v1 = reference(max_retention_false_safe_increase=1)
    safer = _views_result(_views_block(nll=0.002), _views_block(false_safe=1))
    cheaper = _views_result(_views_block(nll=0.001), _views_block(false_safe=2))
    v2_config = config(selection_reference=v2)
    assert sum(views.joint_flags(safer, v2_config)) == 0
    assert sum(views.joint_flags(cheaper, v2_config)) == 0
    safer_key = views.selection(safer, v2_config, 0.99, base_key)
    cheaper_key = views.selection(cheaper, v2_config, 0.99, base_key)
    assert safer_key < cheaper_key
    assert tuple(safer_key[-5:]) == views.retention_rank_key(
        1, 30, 10, 0.002, 0.002
    )
    assert tuple(cheaper_key[-5:]) == views.retention_rank_key(
        2, 30, 10, 0.002, 0.001
    )
    v1_config = config(selection_reference=v1)
    assert views.selection(safer, v1_config, 0.99, base_key) > views.selection(
        cheaper, v1_config, 0.99, base_key
    )


def test_v2_ranking_orders_correct_pairs_then_nll():
    ref = views.build_selection_reference(
        _views_block(),
        _views_block(pairs=0.85),
        _views_policy(selection_ranking="v2-retention"),
        "digest",
    )
    cfg = config(selection_reference=ref)

    def rank(transfer, retention):
        return views.selection(
            _views_result(transfer, retention), cfg, 0.99, base_key
        )

    more_correct = rank(_views_block(), _views_block(accuracy=0.99))
    less_correct = rank(_views_block(), _views_block(accuracy=29 / 30))
    assert more_correct < less_correct
    more_pairs = rank(_views_block(), _views_block(pairs=1.0))
    fewer_pairs = rank(_views_block(), _views_block(pairs=0.9))
    assert more_pairs < fewer_pairs
    lower_retention_nll = rank(_views_block(), _views_block(nll=0.001))
    higher_retention_nll = rank(_views_block(), _views_block(nll=0.002))
    assert lower_retention_nll < higher_retention_nll
    lower_transfer_nll = rank(_views_block(nll=0.001), _views_block())
    higher_transfer_nll = rank(_views_block(nll=0.002), _views_block())
    assert lower_transfer_nll < higher_transfer_nll


def test_ranking_defaults_to_v1_transfer():
    assert views.ranking_version(_views_policy()) == "v1-transfer"
    assert views.ranking_version(
        _views_policy(selection_ranking="v2-retention")
    ) == ("v2-retention")
    with pytest.raises(ValueError, match="unknown ranking"):
        views.ranking_version(_views_policy(selection_ranking="v9"))
    with pytest.raises(ValueError, match="unknown ranking"):
        views.build_selection_reference(
            _views_block(),
            _views_block(),
            _views_policy(selection_ranking="v9"),
            "d",
        )


def test_negative_budget_requires_a_decrease():
    ref = reference(max_retention_false_safe_increase=-1)
    cfg = config(selection_reference=ref)
    held = _views_result(_views_block(), _views_block(false_safe=1))
    improved = _views_result(_views_block(), _views_block(false_safe=0))
    assert views.joint_flags(held, cfg)[1] == 1
    assert views.joint_flags(improved, cfg)[1] == 0


def test_shared_gain_predicate_matches_campaign_math():
    assert views.new_coverage_gain_ok(88 / 90, 84 / 90, 4.0)
    assert not views.new_coverage_gain_ok(87 / 90, 84 / 90, 4.0)
    assert views.new_coverage_gain_ok(0.9, 0.9, 0.0)
    assert not views.new_coverage_gain_ok(0.9, 0.9, 0.01)


def test_joint_flags_cover_budgets_and_caps():
    ref = reference(
        max_new_coverage_false_safe_increase=0,
        family_false_safe_caps={
            "new_coverage": {"famA": 1},
            "retention": {"famR": 0},
        },
    )
    held = _views_result(
        _views_block(
            families={"famA": {"false_safe_count": 0, "violation_count": 6}}
        ),
        _views_block(
            false_safe=1,
            families={"famR": {"false_safe_count": 0, "violation_count": 4}},
        ),
    )
    assert views.joint_flags(held, config(selection_reference=ref)) == (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    over_budget = _views_result(
        _views_block(false_safe=3),
        _views_block(false_safe=1),
    )
    assert (
        views.joint_flags(over_budget, config(selection_reference=ref))[6] == 1
    )
    capped = _views_result(
        _views_block(
            families={"famA": {"false_safe_count": 2, "violation_count": 6}}
        ),
        _views_block(false_safe=1),
    )
    assert views.joint_flags(capped, config(selection_reference=ref))[7] == 1
    missing = _views_result(
        _views_block(
            families={"other": {"false_safe_count": 0, "violation_count": 6}}
        ),
        _views_block(false_safe=1),
    )
    assert views.joint_flags(missing, config(selection_reference=ref))[7] == 1
    retention_capped = _views_result(
        _views_block(),
        _views_block(
            false_safe=1,
            families={"famR": {"false_safe_count": 1, "violation_count": 4}},
        ),
    )
    assert (
        views.joint_flags(retention_capped, config(selection_reference=ref))[7]
        == 1
    )


def test_joint_flags_detect_mc_regression():
    ref = reference()
    regressed = _views_result(_views_block(), _views_block(mc=(9, 10)))
    flags = views.joint_flags(regressed, config(selection_reference=ref))
    assert flags[2] == 1


def test_normalize_family_caps_shapes():
    assert views.normalize_family_caps({"a": 1}) == {
        "new_coverage": {"a": 1},
        "retention": {},
    }
    nested = {"new_coverage": {"a": 1}, "retention": {"b": 0}}
    assert views.normalize_family_caps(nested) == nested
    assert views.normalize_family_caps({}) == {
        "new_coverage": {},
        "retention": {},
    }
    with pytest.raises(ValueError, match="family caps"):
        views.normalize_family_caps({"a": -1})
    with pytest.raises(ValueError, match="family caps"):
        views.normalize_family_caps({"a": 1, "retention": {}})
    with pytest.raises(ValueError, match="family caps"):
        views.normalize_family_caps({"unknown": {}})


def test_reference_builder_validates_blocks():
    with pytest.raises(ValueError, match="confusion"):
        views.build_selection_reference(
            {}, _views_block(), _views_policy(), "d"
        )
    with pytest.raises(ValueError, match="retention floor"):
        views.build_selection_reference(
            _views_block(), _views_block(), {"min_retention_accuracy": 2}, "d"
        )
    incomplete = _views_policy()
    del incomplete["min_new_coverage_gain_pp"]
    with pytest.raises(ValueError, match="gain threshold"):
        views.build_selection_reference(
            _views_block(), _views_block(), incomplete, "d"
        )
    with pytest.raises(ValueError, match="schema"):
        views.check_selection_reference({"schema": "other"})


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


def test_reference_builder_freezes_adapted_results():
    ref = views.build_selection_reference_from_results(
        stored_results(),
        _views_policy(max_new_coverage_false_safe_increase=0),
        "report-digest",
    )
    assert ref["schema"] == views.REFERENCE_SCHEMA
    assert ref["report_digest"] == "report-digest"
    assert ref["transfer"]["false_safe_count"] == 2
    assert ref["retention"]["false_safe_count"] == 1
    assert ref["transfer"]["accuracy"] == pytest.approx(28 / 30)
    assert ref["retention"]["min_class_recall"] == pytest.approx(0.8)
    assert ref["retention"]["both_pair_members_correct"] == pytest.approx(0.9)
    assert ref["retention"]["per_family"] == {
        "famA": {"false_safe_count": 1, "violation_count": 6}
    }


def test_reference_builder_from_results_allows_unknown_families():
    stored = stored_results(
        retention=adapted_view(families=None),
        new_coverage=adapted_view(families={}),
    )
    ref = views.build_selection_reference_from_results(
        stored, _views_policy(), "report-digest"
    )
    assert ref["transfer"]["per_family"] == {}
    assert ref["retention"]["per_family"] == {}


def test_reference_builder_from_results_validates():
    with pytest.raises(ValueError, match="new_coverage"):
        views.build_selection_reference_from_results(
            {"retention": adapted_view()}, _views_policy(), "d"
        )
    bad_counts = stored_results()
    bad_counts["retention"]["counts"]["false_safe"] = 99
    with pytest.raises(ValueError, match="false-safe"):
        views.build_selection_reference_from_results(
            bad_counts, _views_policy(), "d"
        )
    bad_family = stored_results(
        retention=adapted_view(
            families={"famA": {"false_safe": 9, "violations": 6}}
        ),
    )
    with pytest.raises(ValueError, match="famA"):
        views.build_selection_reference_from_results(
            bad_family, _views_policy(), "d"
        )
    with pytest.raises(ValueError, match="retention floor"):
        views.build_selection_reference_from_results(
            stored_results(), {"min_retention_accuracy": 2}, "d"
        )


def test_mc_recall_uses_insufficient_class():
    assert views.mc_recall_from_confusion(
        [[1, 0, 0], [0, 1, 0], [1, 0, 9]]
    ) == (pytest.approx(0.9))
    assert views.mc_recall_from_confusion(
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    ) == (1.0)
    with pytest.raises(ValueError, match="3x3"):
        views.mc_recall_from_confusion([[1]])


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


def test_lineage_and_canonical_overlap_refuse(tmp_path):
    train = _write(
        tmp_path / "train.jsonl",
        [_row("t1", "assign q = a;", "g1")],
    )
    dev = _write(
        tmp_path / "dev.jsonl",
        [_row("d1", "assign q = a;", "g1")],
    )
    splits = {"train": train, "dev": dev}
    assert audit_lineage_overlap(splits)["lineage_overlap"] == 1
    assert audit_canonical_overlap(splits)["canonical_overlap"] == 1
    assert audit_split_groups(splits) == {"train": 1, "dev": 1}


def test_concentration_caps_enforced(tmp_path):
    rows = [_row("t%d" % i, "assign q%d = a;" % i, "big") for i in range(10)]
    train = _write(tmp_path / "train.jsonl", rows)
    report = audit_concentration(train)
    assert report["max_group_share"] == 1.0
    manifest = write_version(tmp_path / "v", "v1", {"train": str(train)})
    audit = _accept_audit(manifest, {"max_group_share": 1.0})
    try:
        accept_version(tmp_path / "v", "v1", audit)
    except ValueError as exc:
        assert "max_group_share" in str(exc)
    else:
        raise AssertionError("concentration cap not enforced")


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


def test_acceptance_records_release_class(tmp_path):
    rows = [
        _row("t%d" % i, "assign q%d = a;" % i, "g%d" % i) for i in range(10)
    ]
    train = _write(tmp_path / "train.jsonl", rows)
    manifest = write_version(tmp_path / "v", "v1", {"train": str(train)})
    audit = _accept_audit(manifest, {"train_groups": 10})
    recorded = accept_version(tmp_path / "v", "v1", audit)
    assert recorded["audit"]["release_class"] == "narrow-pilot"
    # Re-accepting with the original audit is idempotent.
    again = accept_version(tmp_path / "v", "v1", dict(audit))
    assert again["audit_sha256"] == recorded["audit_sha256"]


def test_acceptance_refuses_false_broad_claim(tmp_path):
    rows = [
        _row("t%d" % i, "assign q%d = a;" % i, "g%d" % i) for i in range(10)
    ]
    train = _write(tmp_path / "train.jsonl", rows)
    manifest = write_version(tmp_path / "v", "v2", {"train": str(train)})
    audit = _accept_audit(
        manifest, {"train_groups": 10, "release_class": "broad"}
    )
    try:
        accept_version(tmp_path / "v", "v2", audit)
    except ValueError as exc:
        assert "broad release" in str(exc)
    else:
        raise AssertionError("false broad claim not refused")


def test_canonical_key_ignores_consistent_renames():
    base = _row("t1", "assign q = a && b;", "g1")
    renamed = _row("t2", "assign sig_x = sig_y && sig_z;", "g2")
    assert canonical_rtl_key(base) == canonical_rtl_key(renamed)


def test_canonical_key_keeps_operators_widths_literals():
    base = _row("t1", "assign q = a && b;", "g1")
    swapped = _row("t2", "assign q = a || b;", "g2")
    assert canonical_rtl_key(base) != canonical_rtl_key(swapped)
    narrow = _row("t3", "assign q = 4'd0;", "g3")
    wide = _row("t4", "assign q = 8'd0;", "g4")
    assert canonical_rtl_key(narrow) != canonical_rtl_key(wide)
    commented = _row("t5", "// note\nassign q=a&&b;", "g5")
    assert canonical_rtl_key(base) == canonical_rtl_key(commented)


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


def test_weighted_and_unweighted_metrics_have_known_answers():
    rows, logits, weights = fixture()
    report = evaluation_report(
        rows, logits, weights, {"source-a": 0.8, "source-b": 0.2}, 128, 7
    )
    raw, weighted = (
        report["overall"][name]["metrics"]
        for name in ("unweighted", "weighted")
    )
    assert raw["accuracy"] == pytest.approx(5 / 6)
    assert weighted["accuracy"] == pytest.approx(11 / 12)
    assert weighted["macro_f1"] == pytest.approx((8 / 9 + 6 / 7 + 1) / 3)
    assert raw["false_safe_rate"] == 0.5 and weighted["false_safe_rate"] == 0.25
    assert raw["pair_accuracy"] == 0.5 and weighted["pair_accuracy"] == 0.75
    assert (
        raw["missing_context_recall"] == weighted["missing_context_recall"] == 1
    )
    assert raw["nll"] == pytest.approx(-(5 * np.log(0.8) + np.log(0.1)) / 6)
    assert weighted["nll"] == pytest.approx(
        -(11 * np.log(0.8) + np.log(0.1)) / 12
    )
    assert report["missing_sources"] == ["source-b"]
    assert not report["full_configured_mixture_estimable"]
    assert report["overall"]["weighting"]["observed_source_mass"] == {
        "source-a": 1.0
    }
    interval = report["overall"]["unweighted"]["intervals"]["intervals"][
        "accuracy"
    ]
    assert interval["lower"] == pytest.approx(2 / 3)
    assert interval["upper"] == 1


def test_related_rewrites_do_not_create_more_bootstrap_clusters():
    rows, logits, weights = fixture()
    first = evaluation_report(rows, logits, weights, {"source-a": 1}, 128, 3)
    copies = []
    for rewrite in range(4):
        for row in rows:
            copy_row = copy.deepcopy(row)
            copy_row["id"] += "-rewrite-" + str(rewrite)
            copy_row["pair_id"] += "-rewrite-" + str(rewrite)
            copies.append(copy_row)
    second = evaluation_report(
        copies,
        np.tile(logits, (4, 1)),
        np.tile(weights, 4),
        {"source-a": 1},
        128,
        3,
    )
    assert second["overall"]["support"]["split_groups"] == 2
    for view in ("weighted", "unweighted"):
        a, b = (
            report["overall"][view]["intervals"] for report in (first, second)
        )
        assert a["source_cluster_counts"] == b["source_cluster_counts"]
        for metric in a["intervals"]:
            assert a["intervals"][metric] == pytest.approx(
                b["intervals"][metric]
            )


def test_singleton_source_does_not_get_a_false_precision_interval():
    rows, logits, weights = fixture()
    for row in rows[3:]:
        row["source"]["name"] = "source-b"
        row["mechanism_id"] = "another-access"
    report = evaluation_report(
        rows, logits, weights, {"source-a": 0.5, "source-b": 0.5}, 64
    )
    assert report["full_configured_mixture_estimable"]
    assert (
        report["overall"]["unweighted"]["intervals"]["status"] == "unavailable"
    )
    assert (
        report["per_source"]["source-b"]["weighted"]["intervals"]["reason"]
        == "fewer_than_two_clusters_in_source"
    )
    assert (
        report["per_source_mechanism"]["source-b"]["another-access"][
            "unweighted"
        ]["metrics"]["accuracy"]
        == 1
    )
    assert (
        report["per_source"]["source-a"]["unweighted"]["metrics"][
            "false_safe_rate"
        ]
        == 1
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_weights_rejected(bad):
    rows, logits, weights = fixture()
    weights[0] = bad
    with pytest.raises(ValueError):
        evaluation_report(rows, logits, weights, {"source-a": 1}, 0)


def test_misaligned_arrays_and_mixed_ownership_rejected():
    rows, logits, weights = fixture()
    with pytest.raises(ValueError, match="misaligned"):
        evaluation_report(rows, logits[:-1], weights, {"source-a": 1}, 0)
    rows[1]["source"]["name"] = "source-b"
    with pytest.raises(ValueError, match="spans sources"):
        evaluation_report(
            rows, logits, weights, {"source-a": 0.5, "source-b": 0.5}, 0
        )
    rows, logits, weights = fixture()
    rows[3]["pair_id"] = rows[0]["pair_id"]
    with pytest.raises(ValueError, match="pair crosses"):
        evaluation_report(rows, logits, weights, {"source-a": 1}, 0)


def test_probability_and_logit_paths_agree_and_missing_class_is_explicit():
    rows, logits, weights = fixture()
    left = evaluation_report(rows, logits, weights, {"source-a": 1}, 0)
    right = evaluation_report(
        rows,
        np.exp(logits),
        weights,
        {"source-a": 1},
        0,
        from_probabilities=True,
    )
    for mode in ("weighted", "unweighted"):
        assert left["overall"][mode]["metrics"] == pytest.approx(
            right["overall"][mode]["metrics"]
        )
    chosen = [0, 2, 3, 5]
    partial = evaluation_report(
        [rows[i] for i in chosen],
        logits[chosen],
        weights[chosen],
        {"source-a": 1},
        0,
    )
    assert (
        partial["overall"]["unweighted"]["metrics"]["false_safe_rate"] is None
    )
    assert partial["overall"]["unweighted"]["metrics"]["pair_accuracy"] is None


def test_trainer_attaches_reporting_without_changing_selection(monkeypatch):
    # Importing the trainer requires the complete optional training runtime.
    # The remaining reporting tests need only NumPy.
    for dependency in ("torch", "transformers", "safetensors.torch", "laya"):
        pytest.importorskip(dependency)
    from argparse import Namespace

    from siliconshield.training.laya import security_train as train

    rows, logits, _ = fixture()
    for row in rows:
        row["base_design_id"] = row["split_group_id"]
    monkeypatch.setattr(train, "SOURCE_MASS", {"source-a": 1.0})
    result = train.metrics(logits, [0, 1, 2, 0, 1, 2])
    before = train.selection_key(result)
    train.attach_reporting(
        result, rows, logits, Namespace(report_bootstrap=0, seed=9), rank=0
    )
    assert result["reporting"]["overall"]["support"]["decision_rows"] == 6
    assert train.selection_key(result) == before


# ---- Merged from test_inference_metrics.py ----
def test_prefill_is_separate_and_prompt_tokens_are_not_generated() -> None:
    times = iter([12.0, 12.5, 13.0])
    timer = TokenTiming(10.0)
    streamer = GenerationStreamer(timer, clock=lambda: next(times))
    streamer.put(SimpleNamespace(numel=lambda: 4096))
    for _ in range(3):
        streamer.put(SimpleNamespace(numel=lambda: 1))
    streamer.end()
    result = timer.summary(13.0)
    assert result["generated_tokens"] == 3
    assert result["first_token_seconds"] == 2.0
    assert result["generation_tokens_per_second"] == 1.0
    assert result["decode_tokens_per_second"] == 2.0


def test_speculative_first_block_is_excluded_from_decode_rate() -> None:
    timer = TokenTiming(0.0)
    timer.record(4, 2.0)
    timer.record(3, 2.5)
    assert timer.summary(2.5)["decode_tokens_per_second"] == 6.0
    assert timer.summary(2.5)["generated_tokens"] == 7


def test_no_decode_interval_is_not_reported_as_zero_speed() -> None:
    timer = TokenTiming(0.0)
    assert timer.summary(1.0)["decode_tokens_per_second"] is None
    timer.record(1, 2.0)
    assert timer.summary(2.0)["decode_tokens_per_second"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_timestamps_are_rejected(value: float) -> None:
    timer = TokenTiming(0.0)
    with pytest.raises(ValueError):
        timer.record(1, value)
    with pytest.raises(ValueError):
        timer.summary(value)


def test_usage_is_saved_without_becoming_model_visible(tmp_path) -> None:
    messages = [{"role": "user", "content": "Authored fixture"}]

    class Client:
        model = "fixture"
        last_completion_metrics = {
            "usage": {"prompt_tokens": 20, "completion_tokens": 4}
        }

        def complete(self, history, tools):
            assert history == messages
            return {"role": "assistant", "content": "Done"}

    path = tmp_path / "trajectory.jsonl"
    result = run_agent(Client(), messages, [], lambda *a: None, path)
    event = json.loads(path.read_text().splitlines()[-1])
    assert event["completion_metrics"]["usage"]["completion_tokens"] == 4
    assert event["completion_seconds"] >= 0
    assert result[-1] == {"role": "assistant", "content": "Done"}


def test_native_inference_keeps_bf16_and_explicit_adapter_names() -> None:
    config = ServerConfig(
        "/fixture/model", adapters=(("before", "/fixture/adapter with spaces"),)
    )
    command = config.command()
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "before=/fixture/adapter with spaces" in command
    assert "--speculative-config" not in command
    assert command[command.index("--safetensors-load-strategy") + 1] == "lazy"


def test_speculation_requires_explicit_draft_configuration() -> None:
    command = ServerConfig("/fixture/model", draft="/fixture/draft").command()
    value = json.loads(command[command.index("--speculative-config") + 1])
    assert value == {
        "model": "/fixture/draft",
        "method": "dflash",
        "num_speculative_tokens": 15,
    }


def test_draft_capacity_covers_all_sequences() -> None:
    config = ServerConfig(
        "fixture", draft="draft", max_sequences=8, max_batched_tokens=127
    )
    with pytest.raises(ValueError, match="every draft position"):
        config.command()
    command = ServerConfig(
        "fixture", draft="draft", max_sequences=8, max_batched_tokens=128
    ).command()
    assert command[command.index("--max-num-batched-tokens") + 1] == "128"


@pytest.mark.parametrize("fraction", [float("nan"), 0, 1, 2])
def test_native_inference_rejects_invalid_memory_fraction(fraction) -> None:
    with pytest.raises(ValueError):
        ServerConfig("fixture", memory_fraction=fraction).command()


def test_native_inference_rejects_ambiguous_adapter_identity() -> None:
    with pytest.raises(ValueError, match="unique"):
        ServerConfig(
            "fixture", adapters=(("same", "a"), ("same", "b"))
        ).command()


def test_native_server_inherits_the_exclusive_gpu_lease(
    tmp_path, monkeypatch
) -> None:
    from siliconshield.research_loop.controller import serve_fast

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("SILICONSHIELD_GPU_LOCK", str(lock))
    monkeypatch.setattr(
        serve_fast,
        "verify_snapshot",
        lambda *a, **k: {"fingerprint": "fixture"},
    )
    monkeypatch.setattr(sys, "argv", ["serve_fast", "--model", str(model)])

    def exec_probe(executable, command):
        assert executable == "vllm"
        with lock.open("a") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inherited = [
            int(p.name)
            for p in Path("/proc/self/fd").iterdir()
            if p.exists() and p.resolve() == lock
        ]
        assert len(inherited) == 1
        assert os.get_inheritable(inherited[0])
        raise FileNotFoundError("Authored missing-runtime fixture")

    monkeypatch.setattr(os, "execvp", exec_probe)
    with pytest.raises(FileNotFoundError):
        serve_fast.main()
    with lock.open("a") as released:
        fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)


# ---- Merged from test_ddp_guard.py ----
def test_trainer_uses_single_backward():
    source = (
        Path(__file__).resolve().parents[1]
        / "siliconshield/training/laya/security_train.py"
    )
    assert require_single_backward(source) == 1
    report = require_coherent_training(source)
    assert report["backward_calls"] == 1
    assert report["no_sync"] is True


def test_count_ignores_comments_and_strings(tmp_path):
    from siliconshield.training.laya.ddp_guard import count_backward_calls

    source = tmp_path / "fake_trainer.py"
    source.write_text(
        '"""Docstring mentioning loss.backward() twice: x.backward()."""\n'
        "# comment: y.backward() is not a call\n"
        "z = 'w.backward()'\n"
        "(loss * scale).backward()\n"
    )
    assert count_backward_calls(source) == 1
