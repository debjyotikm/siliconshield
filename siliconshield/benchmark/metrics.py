"""Pure benchmark metrics: overall, per-area, per-mechanism (Qt-free).

Scoring takes validated rows plus one label and probability mapping
per row id. Triplets group by pair_id; a triplet counts only when
all three members are correct. No thresholds or gates live here:
callers compare reports and set their own bars, so future models
are judged by the same arithmetic as today's.
"""

import math
from typing import Any, TypedDict

from siliconshield.benchmark.dataset import LABELS


class ClassScore(TypedDict):
    """Correct/total for one label."""

    label: str
    correct: int
    total: int


class SplitScore(TypedDict):
    """Complete score for one population slice."""

    rows: int
    correct: int
    classes: list[ClassScore]
    false_safe: int
    false_violation: int
    unnecessary_abstention: int
    triplets_correct: int
    triplets_total: int
    triplets_failed: int
    triplets_incomplete: int
    nll: float | None
    errors: int


class ModelReport(TypedDict):
    """Whole-benchmark report for one model."""

    model_id: str
    checkpoint_sha256: str
    benchmark_source: str
    benchmark_sha256: str
    overall: SplitScore
    by_area: dict[str, SplitScore]
    by_mechanism: dict[str, SplitScore]
    predictions: dict[str, str]
    row_errors: list[dict[str, str]]


def _empty() -> SplitScore:
    return {
        "rows": 0,
        "correct": 0,
        "classes": [
            {"label": name, "correct": 0, "total": 0} for name in LABELS
        ],
        "false_safe": 0,
        "false_violation": 0,
        "unnecessary_abstention": 0,
        "triplets_correct": 0,
        "triplets_total": 0,
        "triplets_failed": 0,
        "triplets_incomplete": 0,
        "nll": 0.0,
        "errors": 0,
    }


def _finish(split: SplitScore) -> SplitScore:
    scored = split["rows"] - split["errors"]
    total = split["nll"]
    # _empty seeds a float running total and _finish runs once per
    # split, so anything else here means double finalization.
    if not isinstance(total, float):
        raise ValueError("Split score was already finalized.")
    # No scored rows means no measured NLL: null, not Infinity,
    # which strict JSON parsers reject.
    split["nll"] = total / scored if scored else None
    return split


def score_predictions(
    rows: list[dict[str, Any]],
    labels: dict[str, str],
    probabilities: dict[str, dict[str, float]],
    row_errors: dict[str, str] | None = None,
) -> tuple[SplitScore, dict[str, SplitScore], dict[str, SplitScore]]:
    """Score predictions into overall, per-area, and per-mechanism.

    rows with recorded errors stay in class support (counted wrong)
    and fail their triplets; nothing error-related ever counts
    correct. Raises ValueError for unknown labels or invalid target
    scores on rows without recorded errors.
    """
    row_errors = row_errors or {}
    overall = _empty()
    areas: dict[str, SplitScore] = {}
    mechs: dict[str, SplitScore] = {}
    hits: dict[str, bool] = {}
    for row in rows:
        row_id = row["id"]
        expected = row["expected"]["security"]
        area = row["benchmark_area"]
        mech = row.get("mechanism_id", "unknown")
        splits = (
            overall,
            areas.setdefault(area, _empty()),
            mechs.setdefault(mech, _empty()),
        )
        for split in splits:
            split["rows"] += 1
            for entry in split["classes"]:
                if entry["label"] == expected:
                    entry["total"] += 1
        if row_id in row_errors:
            for split in splits:
                split["errors"] += 1
            hits[row_id] = False
            continue
        predicted = labels.get(row_id)
        if predicted not in LABELS:
            raise ValueError(
                "Row %s has unknown label %r." % (row_id, predicted)
            )
        probs = probabilities.get(row_id, {})
        target = probs.get(expected, 0.0)
        if (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not 0.0 <= target <= 1.0
        ):
            raise ValueError("Row %s has an invalid target score." % row_id)
        # Floor matches the campaign scorer: worker probabilities round
        # to 4 decimals, so confident misses legitimately score 0.0.
        target = max(target, 1e-12)
        hit = predicted == expected
        hits[row_id] = hit
        for split in splits:
            split["correct"] += 1 if hit else 0
            for entry in split["classes"]:
                if entry["label"] == expected:
                    entry["correct"] += 1 if hit else 0
            running = split["nll"] or 0.0
            split["nll"] = running - math.log(target)
        if (
            expected == "violates_requirement"
            and predicted == "satisfies_requirement"
        ):
            for split in splits:
                split["false_safe"] += 1
        if (
            expected == "satisfies_requirement"
            and predicted == "violates_requirement"
        ):
            for split in splits:
                split["false_violation"] += 1
        if (
            expected != "insufficient_context"
            and predicted == "insufficient_context"
        ):
            for split in splits:
                split["unnecessary_abstention"] += 1
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["pair_id"], []).append(row)
    for members in groups.values():
        complete = (
            len(members) == 3
            and len({m["expected"]["security"] for m in members}) == 3
        )
        failed = any(m["id"] in row_errors for m in members)
        correct = (
            complete and not failed and all(hits[m["id"]] for m in members)
        )
        homes = {m["benchmark_area"] for m in members}
        targets = [overall]
        if len(homes) == 1:
            targets.append(areas[next(iter(homes))])
        mechnames = {m.get("mechanism_id", "unknown") for m in members}
        if len(mechnames) == 1:
            targets.append(mechs[next(iter(mechnames))])
        for split in targets:
            split["triplets_total"] += 1
            if not complete:
                split["triplets_incomplete"] += 1
            elif failed:
                split["triplets_failed"] += 1
            elif correct:
                split["triplets_correct"] += 1
    return (
        _finish(overall),
        {k: _finish(v) for k, v in areas.items()},
        {k: _finish(v) for k, v in mechs.items()},
    )


def compare(reports: list[ModelReport]) -> dict[str, Any]:
    """Build a head-to-head comparison across model reports.

    Every report must cover the same benchmark hash; areas union so
    a model missing an area shows null there instead of silently
    dropping the comparison.
    """
    if not isinstance(reports, list) or not reports:
        raise ValueError("Comparison needs at least one report.")
    for report in reports:
        if not isinstance(report, dict):
            raise ValueError("Comparison needs report objects.")
        for key in ("benchmark_sha256", "model_id", "by_area", "overall"):
            if key not in report:
                raise ValueError("Comparison report is missing %r." % key)
    digests = {report["benchmark_sha256"] for report in reports}
    if len(digests) != 1:
        raise ValueError(
            "Reports cover different benchmarks: %s" % sorted(digests)
        )
    names = [report["model_id"] for report in reports]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate model ids in comparison.")
    areas: set[str] = set()
    for report in reports:
        areas.update(report["by_area"])
    table: dict[str, dict[str, Any]] = {}
    for area in sorted(areas):
        row: dict[str, Any] = {}
        for report in reports:
            split = report["by_area"].get(area)
            row[report["model_id"]] = (
                {
                    "correct": split["correct"],
                    "rows": split["rows"],
                    "false_safe": split["false_safe"],
                }
                if split is not None
                else None
            )
        table[area] = row
    return {
        "benchmark_sha256": reports[0]["benchmark_sha256"],
        "models": [
            {
                "model_id": report["model_id"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "correct": report["overall"]["correct"],
                "rows": report["overall"]["rows"],
                "false_safe": report["overall"]["false_safe"],
            }
            for report in reports
        ],
        "by_area": table,
    }
