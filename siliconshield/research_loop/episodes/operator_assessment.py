"""Compute decision-point facts independently of an operator's recommendation."""

import math
from typing import Any


def checkpoint_review(summary: dict[str, Any]) -> dict[str, Any]:
    """Check every supplied candidate against population-specific count limits.

    Constraints are supplied with the observation, rather than inferred from
    a checkpoint name or a low loss. Missing observations remain unknown.
    """
    candidates = summary["checkpoint_candidates"]
    constraints = summary["checkpoint_constraints"]
    if not candidates or not constraints:
        raise ValueError("Checkpoint review needs candidates and constraints")
    failures: dict[str, list[str]] = {}
    unknown: dict[str, list[str]] = {}
    for candidate in candidates:
        name = candidate["id"]
        if not isinstance(name, str) or name in failures:
            raise ValueError("Checkpoint IDs must be unique strings")
        failures[name], unknown[name] = [], []
        for constraint in constraints:
            view, metric = constraint["view"], constraint["metric"]
            relation, limit = constraint["relation"], constraint["limit"]
            if relation not in ("min", "max") or not _finite(limit):
                raise ValueError("Invalid checkpoint constraint")
            key = view + "." + metric
            value = candidate.get("views", {}).get(view, {}).get(metric)
            if value is None:
                unknown[name].append(key)
            elif not _finite(value):
                raise ValueError("Non-finite checkpoint evidence")
            elif (relation == "min" and value < limit) or (
                relation == "max" and value > limit
            ):
                failures[name].append(key)
    feasible = sorted(
        name for name in failures if not failures[name] and not unknown[name]
    )
    selected = summary.get("selected_checkpoint_id")
    return {
        "feasible_ids": feasible,
        "selected_feasible": selected in feasible
        if selected in failures
        else None,
        "failed_constraints": failures,
        "unknown_constraints": unknown,
    }


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def assessment(summary):
    """Return observed status and comparable development deltas, without future data."""
    failure = summary.get("execution_failed")
    if type(failure) is not bool:
        failure = True if summary.get("systemd") == "failed" else None
    acceptance = summary.get("acceptance", {})
    accepted = acceptance.get("accepted")
    acceptance_state = (
        "unknown"
        if type(accepted) is not bool
        else ("passed" if accepted else "failed")
    )
    selected = summary.get(
        "selected_step", summary.get("selection", {}).get("selected_step")
    )
    selected = selected if type(selected) is int else None
    reductions = summary.get("schedule", {}).get("plateau_lr_reductions")
    comparisons = {}
    entries = summary.get("development_history", {}).get("entries", [])
    if len(entries) >= 2 and summary.get("metrics_comparable", True):
        before, after = entries[-2:]
        if before.get("population_id") is not None and before.get(
            "population_id"
        ) == after.get("population_id"):
            views = {
                "development": (
                    before.get("metrics", {}),
                    after.get("metrics", {}),
                )
            }
            for name in sorted(
                set(before.get("views", {})) & set(after.get("views", {}))
            ):
                views[name] = (before["views"][name], after["views"][name])
            for name, (left, right) in views.items():
                if not left or not right:
                    continue
                if left.get("n") != right.get("n"):
                    continue
                deltas = {}
                for source, target, scale in (
                    ("accuracy", "accuracy_delta_pp", 100),
                    ("nll", "nll_delta", 1),
                    ("false_safe_rate", "false_safe_delta_pp", 100),
                ):
                    a, b = left.get(source), right.get(source)
                    deltas[target] = (
                        round((b - a) * scale, 8)
                        if _finite(a) and _finite(b)
                        else None
                    )
                comparisons[name] = deltas
    result = {
        "execution_failed": failure,
        "acceptance": acceptance_state,
        "selected_step": selected,
        "plateau_reductions": len(reductions)
        if isinstance(reductions, list)
        else None,
        "comparisons": comparisons,
    }
    # The key is always present (null without candidates) so the
    # Muse CLI output schema and the field-equality check agree.
    if "checkpoint_candidates" in summary:
        result["checkpoint_review"] = checkpoint_review(summary)
    else:
        result["checkpoint_review"] = None
    return result


def assessment_issues(summary, reported):
    """Reject omitted, invented, or non-finite facts and incorrectly bound deltas."""
    issues = []

    def compare(expected, actual, path):
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(expected) != set(actual):
                issues.append(path + "_fields_differ")
                return
            for key, value in expected.items():
                compare(value, actual[key], path + "." + key)
        elif isinstance(expected, float):
            if not _finite(actual) or abs(expected - actual) > 1e-5:
                issues.append(path + "_value_differs")
        elif type(actual) is not type(expected) or actual != expected:
            issues.append(path + "_value_differs")

    compare(assessment(summary), reported, "assessment")
    return issues


def parameter_issues(row, decision):
    """Check proposed intervention parameters against the observed run and budget."""
    state, resources = row["summary"], row.get("resources", {})
    action = decision.get("action")
    issues = []
    if action == "adjust_lr":
        bounds = state.get(
            "lr_bounds", resources.get("budget", {}).get("lr_bounds")
        )
        value = decision.get("lr")
        expected = row.get("target", {}).get("lr")
        if not _finite(value):
            issues.append("lr_missing_or_nonfinite")
        elif bounds and not bounds[0] <= value <= bounds[1]:
            issues.append("lr_outside_budget")
        elif _finite(expected) and not math.isclose(
            value, expected, rel_tol=1e-6, abs_tol=1e-12
        ):
            issues.append("lr_differs_from_registered_intervention")
    if action == "resume" or (
        action == "finalize_retry" and "checkpoint_dir" in decision
    ):
        expected = state.get(
            "checkpoint_dir", row.get("target", {}).get("checkpoint_dir")
        )
        if expected is not None and decision.get("checkpoint_dir") != expected:
            issues.append("checkpoint_not_owned_by_run")
    return issues
