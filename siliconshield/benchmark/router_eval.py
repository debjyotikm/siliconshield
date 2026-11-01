"""Evaluate routing over sampled benchmark examples (Qt-free).

Samples whole triplets per area, routes each row's scope through the
shared service, consults the routed experts, and checks the outcome
against the known label: did a reliable expert get consulted, what
did it say, and did the experts agree? Optional Muse arbitration
spot-checks a smaller subsample; arbitration needs the slow Muse
CLI, so it defaults off and never gates the routing verdict.
"""

import random
from collections import Counter
from typing import Any, TypedDict

from siliconshield.benchmark.dataset import BenchmarkData
from siliconshield.experts.identity import state_sections
from siliconshield.experts.routing_data import area_for_tag
from siliconshield.experts.service import InferenceService


class AreaEval(TypedDict):
    """Routing outcome stats for one area sample."""

    area: str
    rows: int
    thin_rows: int
    expert_correct: dict[str, dict[str, int]]
    errors: int
    agree_rows: int
    routed_tiers: dict[str, int]


class RouterEvalReport(TypedDict):
    """Whole router-eval report across sampled areas."""

    benchmark_source: str
    benchmark_sha256: str
    seed: int
    per_area_triplets: int
    by_area: dict[str, AreaEval]
    arbitration: list[dict[str, Any]]


def sample_triplets(
    rows: list[dict[str, Any]], per_area: int, seed: int
) -> list[dict[str, Any]]:
    """Sample whole triplets per area with a fixed seed.

    Non-positive per_area takes every triplet in every area.
    Raises ValueError when an area holds fewer triplets than asked:
    silent undersampling would skew the area comparison.
    """
    by_area: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        trips = by_area.setdefault(row["benchmark_area"], {})
        trips.setdefault(row["pair_id"], []).append(row)
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for area in sorted(by_area):
        keys = sorted(by_area[area])
        if per_area <= 0:
            chosen = keys
        else:
            if len(keys) < per_area:
                raise ValueError(
                    "Area %s holds %d triplets, fewer than %d."
                    % (area, len(keys), per_area)
                )
            chosen = rng.sample(keys, per_area)
        for key in chosen:
            sampled.extend(by_area[area][key])
    return sampled


def _requirement_of(state: str) -> str:
    """Extract the REQUIREMENT section head for arbitration context."""
    return state_sections(state)[0]


def evaluate_routing(
    data: BenchmarkData,
    service: InferenceService,
    transport: Any,
    per_area: int,
    seed: int,
    arbitrate_per_area: int = 0,
    arbitrate: Any | None = None,
    progress: Any | None = None,
) -> RouterEvalReport:
    """Route and consult sampled rows; score against known labels."""
    sampled = sample_triplets(data["rows"], per_area, seed)
    total = len(sampled)
    by_area: dict[str, AreaEval] = {}
    arbitration: list[dict[str, Any]] = []
    arbitrated: Counter = Counter()
    for done, row in enumerate(sampled, start=1):
        if progress is not None:
            progress(done, total)
        area = row["benchmark_area"]
        scope = [area]
        mechanism = row.get("mechanism_id")
        if mechanism and area_for_tag(mechanism) is not None:
            scope.append(mechanism)
        assessment = service.assess(scope, row["state"], transport)
        entry = by_area.setdefault(
            area,
            {
                "area": area,
                "rows": 0,
                "thin_rows": 0,
                "expert_correct": {},
                "agree_rows": 0,
                "errors": 0,
                "routed_tiers": {},
            },
        )
        entry["rows"] += 1
        if assessment["routing"]["thin_evidence"]:
            entry["thin_rows"] += 1
        expected = row["expected"]["security"]
        votes = []
        for response in assessment["responses"]:
            expert = response["expert_id"]
            tier = assessment["routing"]["tiers"].get(expert, "unrated")
            entry["routed_tiers"][tier] = entry["routed_tiers"].get(tier, 0) + 1
            tally = entry["expert_correct"].setdefault(
                expert, {"correct": 0, "total": 0}
            )
            tally["total"] += 1
            tally["correct"] += 1 if response["label"] == expected else 0
            votes.append(response["label"])
        entry["errors"] += len(assessment["errors"])
        if len(votes) >= 2 and len(set(votes)) == 1:
            entry["agree_rows"] += 1
        if (
            arbitrate is not None
            and arbitrated[area] < arbitrate_per_area * 3
            and len(assessment["responses"]) > 0
        ):
            arbitrated[area] += 1
            try:
                verdict = arbitrate(
                    dict(assessment),
                    row["state"],
                    _requirement_of(row["state"]),
                    service.profiles_for(
                        [item["expert_id"] for item in assessment["responses"]]
                    ),
                )
            except (ValueError, RuntimeError, TimeoutError) as exc:
                verdict = {"conclusion": "arbitration_failed: %s" % exc}
            arbitration.append(
                {
                    "row_id": row["id"],
                    "area": area,
                    "expected": expected,
                    "verdict": verdict,
                }
            )
    return {
        "benchmark_source": data["source"],
        "benchmark_sha256": data["rows_sha256"],
        "seed": seed,
        "per_area_triplets": per_area,
        "by_area": by_area,
        "arbitration": arbitration,
    }
