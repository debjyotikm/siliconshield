"""Checkpoint selection, development views, independence gates, reporting, metrics, and DDP guard tests: synthetic fixtures."""

import json

import numpy as np

from siliconshield.schemas import CHECKPOINT_OBJECTIVE_SCHEMA
from siliconshield.training.laya import development_views as views
from siliconshield.training.laya.checkpoint_selection import (
    objective_selection,
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
