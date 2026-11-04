"""Compute decision-point facts independently of an operator's recommendation."""

import math
from typing import Any


def checkpoint_review(summary: dict[str, Any]) -> dict[str, Any]:
    """Check every supplied candidate against population-specific count limits.

    Constraints are supplied with the observation, rather than inferred from
    a checkpoint name or a low loss. Missing observations remain unknown.
    """
    if not isinstance(summary, dict):
        raise ValueError("Checkpoint review needs a summary object")
    candidates = summary.get("checkpoint_candidates")
    constraints = summary.get("checkpoint_constraints")
    if (
        not isinstance(candidates, list)
        or not candidates
        or not isinstance(constraints, list)
        or not constraints
    ):
        raise ValueError("Checkpoint review needs candidates and constraints")
    failures: dict[str, list[str]] = {}
    unknown: dict[str, list[str]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Checkpoint candidates must be objects")
        name = candidate.get("id")
        if not isinstance(name, str) or name in failures:
            raise ValueError("Checkpoint IDs must be unique strings")
        failures[name], unknown[name] = [], []
        views = candidate.get("views", {})
        if not isinstance(views, dict):
            raise ValueError("Checkpoint evidence must be an object")
        for constraint in constraints:
            if not isinstance(constraint, dict):
                raise ValueError("Invalid checkpoint constraint")
            view, metric = constraint.get("view"), constraint.get("metric")
            relation, limit = (
                constraint.get("relation"),
                constraint.get("limit"),
            )
            if (
                not isinstance(view, str)
                or not isinstance(metric, str)
                or relation not in ("min", "max")
                or not _finite(limit)
            ):
                raise ValueError("Invalid checkpoint constraint")
            key = view + "." + metric
            view_block = views.get(view, {})
            if not isinstance(view_block, dict):
                raise ValueError("Checkpoint evidence must be an object")
            value = view_block.get(metric)
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
    if not isinstance(summary, dict):
        raise ValueError("Operator assessment needs a summary object")
    failure = summary.get("execution_failed")
    if type(failure) is not bool:
        failure = True if summary.get("systemd") == "failed" else None
    acceptance = summary.get("acceptance", {})
    if not isinstance(acceptance, dict):
        raise ValueError("Operator acceptance must be an object")
    accepted = acceptance.get("accepted")
    acceptance_state = (
        "unknown"
        if type(accepted) is not bool
        else ("passed" if accepted else "failed")
    )
    selection = summary.get("selection", {})
    if not isinstance(selection, dict):
        raise ValueError("Operator selection must be an object")
    selected = summary.get("selected_step", selection.get("selected_step"))
    selected = selected if type(selected) is int else None
    schedule = summary.get("schedule", {})
    if not isinstance(schedule, dict):
        raise ValueError("Operator schedule must be an object")
    reductions = schedule.get("plateau_lr_reductions")
    comparisons = {}
    history = summary.get("development_history", {})
    if not isinstance(history, dict):
        raise ValueError("Operator development history must be an object")
    entries = history.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Operator history entries must be a list")
    if len(entries) >= 2 and summary.get("metrics_comparable", True):
        before, after = entries[-2:]
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError("Operator history entries must be objects")
        if before.get("population_id") is not None and before.get(
            "population_id"
        ) == after.get("population_id"):
            before_metrics = before.get("metrics", {})
            after_metrics = after.get("metrics", {})
            before_views = before.get("views", {})
            after_views = after.get("views", {})
            if not (
                isinstance(before_metrics, dict)
                and isinstance(after_metrics, dict)
                and isinstance(before_views, dict)
                and isinstance(after_views, dict)
            ):
                raise ValueError("Operator history metrics must be objects")
            views = {"development": (before_metrics, after_metrics)}
            for name in sorted(set(before_views) & set(after_views)):
                views[name] = (before_views[name], after_views[name])
            for name, (left, right) in views.items():
                if not isinstance(left, dict) or not isinstance(right, dict):
                    raise ValueError("Operator history metrics must be objects")
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
    if not isinstance(row, dict) or not isinstance(decision, dict):
        raise ValueError("Intervention check needs row and decision objects")
    state = row.get("summary")
    if not isinstance(state, dict):
        raise ValueError("Intervention check needs a summary object")
    resources = row.get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError("Intervention resources must be an object")
    target = row.get("target", {})
    if not isinstance(target, dict):
        raise ValueError("Intervention target must be an object")
    action = decision.get("action")
    issues = []
    if action == "adjust_lr":
        budget = resources.get("budget", {})
        if not isinstance(budget, dict):
            raise ValueError("Intervention budget must be an object")
        bounds = state.get("lr_bounds", budget.get("lr_bounds"))
        value = decision.get("lr")
        expected = target.get("lr")
        if bounds is not None and (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or not all(_finite(end) for end in bounds)
        ):
            raise ValueError("Intervention LR bounds must be two numbers")
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
        expected = state.get("checkpoint_dir", target.get("checkpoint_dir"))
        if expected is not None and decision.get("checkpoint_dir") != expected:
            issues.append("checkpoint_not_owned_by_run")
    return issues
