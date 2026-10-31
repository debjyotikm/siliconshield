"""Prevent aggregate gains from concealing operator action regressions."""

import copy

import pytest

from siliconshield.research_loop.episodes.contracts import OPERATOR_ACTIONS
from siliconshield.training.glimmer.selection import (
    METRICS,
    compare_replay,
    select_replay_candidate,
)


def report() -> dict:
    """Create a small, complete authored operator population."""
    rows = [
        {
            "episode_id": action + str(i),
            "group_id": action,
            "expected_action": action,
            "checkpoint_review": action == "leave_unchanged",
            "execution_complete": True,
            "execution_error": None,
            **dict.fromkeys(METRICS, i == 0),
        }
        for action in OPERATOR_ACTIONS
        for i in range(2)
    ]
    result = {
        "n": len(rows),
        "results": rows,
        "valid_execution": True,
        "test_evaluated": False,
        "source_identity": "source",
        "evaluator_sha256": "evaluator",
        "dataset_manifest_sha256": "dataset",
        "decoding": {"temperature": 0},
    }
    recount(result)
    return result


def recount(value: dict) -> None:
    """Recompute fixture totals after deliberate outcome changes."""
    value["totals"] = {
        key: sum(r[key] for r in value["results"]) for key in METRICS
    }


def outcome(value: dict, identity: str, passed: bool) -> None:
    """Change one fixture episode, preserving accurate summary counts."""
    next(r for r in value["results"] if r["episode_id"] == identity).update(
        dict.fromkeys(METRICS, passed)
    )
    recount(value)


def test_aggregate_gain_does_not_hide_stop_regression() -> None:
    parent = report()
    child = copy.deepcopy(parent)
    outcome(child, "stop0", False)
    outcome(child, "finalize_retry1", True)
    outcome(child, "resume1", True)
    result = compare_replay(parent, child)
    assert result["verified_gain"] == 1
    assert not result["passed"]
    assert "action:stop:action_correct" in result["issues"]


def test_preserve_both_reference_strengths_and_require_improvement() -> None:
    serving = report()
    trained = copy.deepcopy(serving)
    outcome(trained, "stop0", False)
    outcome(trained, "finalize_retry1", True)
    repaired = copy.deepcopy(trained)
    outcome(repaired, "stop0", True)
    result = select_replay_candidate(
        {"serving": serving, "trained": trained},
        {"unchanged": trained, "repaired": repaired},
    )
    assert result["selected"] == "repaired"
    assert result["deployed"] is False
    assert (
        select_replay_candidate({"base": serving}, {"same": serving})[
            "selected"
        ]
        is None
    )


@pytest.mark.parametrize("problem", ["duplicate", "missing", "total", "bool"])
def test_malformed_reports_refuse(problem: str) -> None:
    parent, child = report(), report()
    if problem == "duplicate":
        child["results"][0]["episode_id"] = child["results"][1]["episode_id"]
    elif problem == "missing":
        child["results"] = child["results"][:-2]
        child["n"] -= 2
        recount(child)
    elif problem == "total":
        child["totals"]["verified"] += 1
    else:
        child["results"][0]["verified"] = "true"
    with pytest.raises(ValueError):
        compare_replay(parent, child)


@pytest.mark.parametrize(
    "key", ["source_identity", "decoding", "test_evaluated"]
)
def test_identity_and_test_boundaries_refuse(key: str) -> None:
    parent, child = report(), report()
    child[key] = True if key == "test_evaluated" else "changed"
    with pytest.raises(ValueError):
        compare_replay(parent, child)


def test_execution_errors_refuse_even_with_passing_totals() -> None:
    parent, child = report(), report()
    child["results"][0]["execution_error"] = "transport failed"
    assert not compare_replay(parent, child)["passed"]
