"""Protect previous tasks while measuring a separate development objective."""

import copy

import pytest

from siliconshield.research_loop.execution.security_jobs import command_for
from siliconshield.schemas import CHECKPOINT_OBJECTIVE_SCHEMA
from siliconshield.training.laya.checkpoint_selection import (
    objective_selection,
    population_indices,
    view_counts,
)
from siliconshield.training.laya.development_views import (
    attach_views,
    selection,
)


def block(matrix, pairs):
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


def policy():
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


def result(new, old=None):
    return {
        "development_views": {
            "old": old or block([[3, 0, 0], [0, 3, 0], [0, 0, 3]], 3),
            "new": new,
        }
    }


def test_new_task_gain_is_visible_to_real_selection():
    initial = result(block([[0, 3, 0], [0, 3, 0], [0, 0, 3]], 0))
    improved = result(block([[2, 1, 0], [0, 3, 0], [0, 0, 3]], 2))
    config = {"checkpoint_objective": policy()}
    assert selection(improved, config, 1.0, None) < selection(
        initial, config, 1.0, None
    )
    assert improved["checkpoint_acceptance"]["quality_targets_passed"]
    assert not initial["checkpoint_acceptance"]["quality_targets_passed"]


def test_new_task_gain_cannot_compensate_for_old_false_safe():
    initial = result(block([[0, 3, 0], [0, 3, 0], [0, 0, 3]], 0))
    damaged = result(
        block([[3, 0, 0], [0, 3, 0], [0, 0, 3]], 3),
        block([[3, 0, 0], [1, 2, 0], [0, 0, 3]], 2),
    )
    assert objective_selection(initial, policy()) < objective_selection(
        damaged, policy()
    )
    assert not damaged["checkpoint_acceptance"]["protected_passed"]


def test_missing_context_gain_does_not_satisfy_readiness():
    candidate = result(block([[0, 3, 0], [0, 3, 0], [0, 0, 3]], 0))
    objective_selection(candidate, policy())
    assert candidate["checkpoint_acceptance"]["protected_passed"]
    assert candidate["checkpoint_acceptance"]["quality_target_failures"] == [
        "class_0",
        "min_pairs_correct",
    ]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
def test_nonfinite_or_invalid_nll_refuses(bad):
    metrics = block([[3, 0, 0], [0, 3, 0], [0, 0, 3]], 3)
    metrics["nll"] = bad
    with pytest.raises(ValueError, match="NLL"):
        view_counts(metrics)


def test_corrupt_counts_and_unprotected_views_refuse():
    metrics = block([[3, 0, 0], [0, 3, 0], [0, 0, 3]], 3)
    altered = copy.deepcopy(metrics)
    altered["false_safe_count"] = 1
    with pytest.raises(ValueError, match="confusion"):
        view_counts(altered)
    candidate = result(metrics)
    candidate["development_views"]["another"] = metrics
    with pytest.raises(ValueError, match="Every development view"):
        objective_selection(candidate, policy())
    changed_policy = policy()
    changed_policy["protected"]["old"]["rows"] = 12
    with pytest.raises(ValueError, match="population size"):
        objective_selection(result(metrics), changed_policy)


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
    invalid = policy()
    invalid["protected"]["old"]["max_false_sfae"] = 0
    candidate = result(block([[3, 0, 0], [0, 3, 0], [0, 0, 3]], 3))
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
