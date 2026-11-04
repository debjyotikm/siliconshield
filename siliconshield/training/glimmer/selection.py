"""Select replay candidates without hiding regressions in individual actions."""

from collections import defaultdict
from typing import Any

from siliconshield.research_loop.episodes.contracts import OPERATOR_ACTIONS

METRICS = (
    "action_correct",
    "evidence_correct",
    "explanation_correct",
    "final_consistent",
    "verified",
)
IDENTITIES = (
    "source_identity",
    "evaluator_sha256",
    "dataset_manifest_sha256",
    "decoding",
)


def _rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate recorded outcomes before deriving selection statistics."""
    if not isinstance(report, dict):
        raise ValueError("Replay selection requires a report object")
    values = report.get("results")
    if not isinstance(values, list) or not values:
        raise ValueError("Replay selection requires nonempty results")
    if report.get("n") != len(values):
        raise ValueError("Replay population count differs")
    rows = {}
    for row in values:
        if not isinstance(row, dict):
            raise ValueError("Replay rows must be objects")
        identity = row.get("episode_id")
        if not isinstance(identity, str) or not identity or identity in rows:
            raise ValueError("Replay IDs must be unique nonempty strings")
        for key in ("group_id", "expected_action"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError("Replay row needs " + key)
        if type(row.get("checkpoint_review")) is not bool:
            raise ValueError("Replay row needs checkpoint_review")
        for key in METRICS:
            if type(row.get(key)) is not bool:
                raise ValueError("Replay metric must be boolean: " + key)
        rows[identity] = row
    totals = report.get("totals", {})
    if not isinstance(totals, dict):
        raise ValueError("Replay totals must be an object")
    for key in METRICS:
        if totals.get(key) != sum(row[key] for row in values):
            raise ValueError("Replay totals differ: " + key)
    for key in IDENTITIES:
        if not report.get(key):
            raise ValueError("Replay identity missing: " + key)
    if report.get("test_evaluated") is not False:
        raise ValueError("Selection requires development-only evaluation")
    missing = set(OPERATOR_ACTIONS) - {row["expected_action"] for row in values}
    if missing:
        raise ValueError(
            "Replay omits operator actions: " + str(sorted(missing))
        )
    return rows


def _groups(rows: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Include every action and both checkpoint-review cohorts."""
    groups: dict[str, list[str]] = defaultdict(list)
    for identity, row in rows.items():
        groups["all"].append(identity)
        groups["action:" + row["expected_action"]].append(identity)
        cohort = "checkpoint_review" if row["checkpoint_review"] else "other"
        groups[cohort].append(identity)
    return dict(groups)


def compare_replay(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Require non-regression per action on identical development episodes.

    Raises:
        ValueError: A report is malformed, incomparable, or touches final test.
    """
    old, new = _rows(reference), _rows(candidate)
    if old.keys() != new.keys():
        raise ValueError("Replay episode populations differ")
    for key in IDENTITIES:
        if reference[key] != candidate[key]:
            raise ValueError("Replay identity differs: " + key)
    for identity in old:
        for key in ("group_id", "expected_action", "checkpoint_review"):
            if old[identity][key] != new[identity][key]:
                raise ValueError("Replay membership differs: " + key)
    issues = []
    for name, report in (("reference", reference), ("candidate", candidate)):
        if report.get("valid_execution") is not True or any(
            row.get("execution_error")
            or row.get("execution_complete") is not True
            for row in report["results"]
        ):
            issues.append(name + ":execution_incomplete")
    comparisons = {}
    for group, identities in _groups(old).items():
        metrics = {}
        for key in METRICS:
            before = sum(old[i][key] for i in identities)
            after = sum(new[i][key] for i in identities)
            metrics[key] = {"reference": before, "candidate": after}
            if after < before:
                issues.append(group + ":" + key)
        comparisons[group] = {"n": len(identities), "metrics": metrics}
    return {
        "passed": not issues,
        "issues": issues,
        "groups": comparisons,
        "verified_gain": candidate["totals"]["verified"]
        - reference["totals"]["verified"],
    }


def select_replay_candidate(
    references: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Preserve each reference's action strengths and require a net gain.

    A candidate must pass against every reference and exceed the best reference
    verified total. This permits using the serving adapter and a newer research
    adapter together without trading one adapter's strengths for the other's.
    """
    if not isinstance(references, dict) or not references:
        raise ValueError("At least one reference is required")
    if not isinstance(candidates, dict):
        raise ValueError("Replay selection needs candidate reports")
    checks = {}
    eligible = []
    for name, report in candidates.items():
        comparisons = {
            baseline: compare_replay(reference, report)
            for baseline, reference in references.items()
        }
        improved = report["totals"]["verified"] > max(
            ref["totals"]["verified"] for ref in references.values()
        )
        checks[name] = {"comparisons": comparisons, "improved": improved}
        if improved and all(c["passed"] for c in comparisons.values()):
            eligible.append(name)
    selected = max(
        eligible,
        key=lambda name: (
            candidates[name]["totals"]["verified"],
            candidates[name]["totals"]["evidence_correct"],
            candidates[name]["totals"]["action_correct"],
        ),
        default=None,
    )
    return {
        "schema": "siliconshield-action-selection-v1",
        "selected": selected,
        "checks": checks,
        "test_evaluated": False,
        "deployed": False,
    }
