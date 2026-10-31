"""Source inventory and feasibility checks for fresh data.

The dataset plan sets inventory targets (for example 80-120 TRAIN
groups with 20/40/20 OpenTitan/HardSecBench/authored origins) that
the current exclusions may not support. This helper counts
eligible origins, groups, and mechanisms from a proposed catalog
so a release is explicitly classified as a narrower pilot when
the inventory cannot meet the broad target. No datasets live in
Git; this operates on caller-supplied rows and local manifests.
"""

from collections import Counter
from typing import Any

BROAD_TARGETS = {
    "train_groups": 80,
    "train_groups_goal": 120,
    "dev_groups": 24,
    "calibration_groups": 12,
    "test_groups": 24,
}


def _origin(row: dict[str, Any]) -> str:
    """Return the source-origin label for one row."""
    for key in ("origin", "source_name", "independent_origin"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict) and value.get("name"):
            return str(value["name"])
    source = row.get("source")
    if isinstance(source, dict) and source.get("name"):
        return str(source["name"])
    if isinstance(source, str) and source:
        return source
    raise ValueError("Row lacks origin: " + repr(row.get("id")))


def _group(row: dict[str, Any]) -> str:
    """Return the split-group identity for one row."""
    for key in ("split_group_id", "dependency_group_id", "family_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("Row lacks group: " + repr(row.get("id")))


def summarize(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Count origins, groups, and mechanisms for a proposed corpus."""
    if not train_rows:
        raise ValueError("TRAIN inventory is empty")
    origins = Counter(_origin(row) for row in train_rows)
    groups = {_group(row) for row in train_rows}
    mechanisms = Counter(
        str(row.get("mechanism_id", "unknown")) for row in train_rows
    )
    report: dict[str, Any] = {
        "train_rows": len(train_rows),
        "train_origins": len(origins),
        "train_groups": len(groups),
        "train_mechanisms": len(mechanisms),
        "origin_counts": dict(origins),
        "max_group_share": max(
            Counter(_group(row) for row in train_rows).values()
        )
        / len(train_rows),
    }
    if dev_rows is not None:
        report["dev_groups"] = len({_group(row) for row in dev_rows})
    return report


def feasibility(
    report: dict[str, Any],
    targets: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compare an inventory summary against broad-release targets.

    Returns per-target pass flags and an overall ``broad_release``
    verdict. A failing inventory must be released as a narrower
    pilot with its missing coverage disclosed, not by inflating
    group counts with renames or wrappers.
    """
    goals = dict(BROAD_TARGETS)
    if targets:
        goals.update(targets)
    checks = {}
    for key in (
        "train_groups",
        "dev_groups",
        "calibration_groups",
        "test_groups",
    ):
        value = report.get(key)
        checks[key] = (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= goals[key]
        )
    broad = all(checks.values())
    return {
        "targets": goals,
        "checks": checks,
        "broad_release": broad,
        "release_class": "broad" if broad else "narrow-pilot",
    }
