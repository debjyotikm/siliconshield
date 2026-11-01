"""Route assessment scopes to experts by demonstrated expertise.

Routing matches scope mechanism tags against capability profiles,
then ranks qualified experts by measured area reliability. A scope
consults every expert whose profile covers all of its tags;
overlapping scopes consult several at once. Experts covering only
some tags are reported as partial coverage so callers can split
the scope -- they are never silently promoted to full consults.
Tiers aggregate worst-area-first: one weak area makes the whole
consult thin. Scopes no profile covers are reported unsupported
explicitly. Pure functions, Qt-free.
"""

from collections.abc import Mapping
from typing import Any, TypedDict

from siliconshield.experts.capabilities import CapabilityProfile
from siliconshield.experts.routing_data import area_for_tag

TIER_RANK = {"strong": 0, "partial": 1, "weak": 2, "none": 3, "unrated": 4}
"""Lower ranks first; unrated sorts last so evidence always wins."""


class PartialCover(TypedDict):
    """One expert covering some but not all scope tags."""

    expert_id: str
    covered: list[str]
    missing: list[str]


class RoutingDecision(TypedDict):
    """One routing outcome for an assessment scope."""

    scope: list[str]
    experts: list[str]
    unsupported: list[str]
    tiers: dict[str, str]
    area_tiers: dict[str, dict[str, str]]
    coverage: dict[str, list[str]]
    partial: list[PartialCover]
    rationale: list[str]
    thin_evidence: bool


def _area_tier(
    profile: CapabilityProfile,
    area: str,
    execution: Mapping[str, Any] | None = None,
) -> str:
    """Use measured counts only for the matching artifact and protocol."""
    evidence = profile.get("routing_evidence")
    if not evidence:
        return "unrated"
    identity = evidence["execution_identity"]
    if any(
        identity[key] != dict(profile)[key]
        for key in ("checkpoint_sha256", "question_contract")
    ):
        return "unrated"
    if execution is not None and identity != execution:
        return "unrated"
    counts = evidence["areas"].get(area)
    if counts is None or counts["errors"]:
        return "unrated"
    correct, total = counts["correct"], counts["total"]
    rank = (
        0
        if correct * 5 >= total * 4
        else 1
        if correct * 5 >= total * 3
        else 2
        if correct * 5 >= total * 2
        else 3
    )
    if counts["false_safe"] * 100 > counts["violations"] * 15:
        rank = min(rank + 1, 3)
    return ("strong", "partial", "weak", "none")[rank]


def _worst(tiers: list[str]) -> str:
    """Worst tier with unrated ranking below every measurement."""
    return max(tiers, key=lambda tier: TIER_RANK[tier])


def route_scope(
    scope: list[str],
    profiles: list[CapabilityProfile],
    execution_identities: Mapping[str, Mapping[str, Any]] | None = None,
) -> RoutingDecision:
    """Route one scope to qualified experts, ranked by evidence.

    An expert is consulted only when its supported mechanisms cover
    every scope tag. Consulted experts sort by worst area tier,
    profile order breaking ties. Partial-cover experts and per-tag
    coverage are reported for scope splitting, never consulted.
    thin_evidence is true when a tag is unsupported or unratable,
    no expert covers the scope, or the best consulted worst-tier is
    weak or worse. Rationale lines quote per-area tiers so callers
    and skills cite evidence instead of paraphrasing it.
    """
    if not isinstance(scope, list) or any(
        not isinstance(tag, str) or not tag.strip() for tag in scope
    ):
        raise ValueError("Scope must be a list of nonempty mechanism tags")
    tags = list(dict.fromkeys(scope))
    covered_tags: dict[str, list[str]] = {tag: [] for tag in tags}
    full: list[CapabilityProfile] = []
    partial: list[PartialCover] = []
    for profile in profiles:
        supported = profile["supported_mechanisms"]
        claimed = [tag for tag in tags if tag in supported]
        for tag in claimed:
            covered_tags[tag].append(profile["id"])
        if tags and len(claimed) == len(tags):
            full.append(profile)
        elif claimed:
            partial.append(
                {
                    "expert_id": profile["id"],
                    "covered": claimed,
                    "missing": [tag for tag in tags if tag not in supported],
                }
            )
    unsupported = [tag for tag in tags if not covered_tags[tag]]
    unratable = sorted({tag for tag in tags if area_for_tag(tag) is None})
    area_tiers: dict[str, dict[str, str]] = {}
    for profile in full:
        per_area = {}
        for tag in tags:
            area = area_for_tag(tag)
            if area is not None:
                per_area[area] = _area_tier(
                    profile,
                    area,
                    execution_identities.get(profile["id"], {})
                    if execution_identities is not None
                    else None,
                )
        area_tiers[profile["id"]] = per_area
    tiers = {
        expert: _worst(list(per_area.values())) if per_area else "unrated"
        for expert, per_area in area_tiers.items()
    }
    order = [item["id"] for item in profiles]
    ranked = sorted(
        full,
        key=lambda item: (
            TIER_RANK[tiers[item["id"]]],
            order.index(item["id"]),
        ),
    )
    experts = [item["id"] for item in ranked]
    rationale: list[str] = []
    for tag in tags:
        area = area_for_tag(tag)
        if area is None:
            rationale.append("%s: no benchmark area; evidence unrated." % tag)
            continue
        claims = ", ".join(
            "%s %s" % (expert, area_tiers[expert].get(area, "unrated"))
            for expert in experts
        )
        if claims:
            rationale.append("%s -> %s: %s." % (tag, area, claims))
        else:
            union = ", ".join(covered_tags[tag]) or "nobody"
            rationale.append(
                "%s -> %s: no expert covers the scope (tag claimed by %s)."
                % (tag, area, union)
            )
    for tag in unsupported:
        rationale.append("%s: unsupported by every profile." % tag)
    for item in partial:
        rationale.append(
            "%s covers %s but not %s; split the scope to consult it."
            % (
                item["expert_id"],
                ",".join(item["covered"]),
                ",".join(item["missing"]),
            )
        )
    if experts:
        rationale.append(
            "scope tier: %s (worst area per expert; best consulted %s)."
            % (
                _worst([tiers[expert] for expert in experts]),
                experts[0],
            )
        )
    best = min(
        (TIER_RANK[tiers[expert]] for expert in experts),
        default=TIER_RANK["unrated"],
    )
    thin = (
        bool(unsupported or unratable)
        or not experts
        or best >= TIER_RANK["weak"]
    )
    void = not tags
    return {
        "scope": list(tags),
        "experts": [] if void else experts,
        "unsupported": unsupported,
        "tiers": {} if void else tiers,
        "area_tiers": {} if void else area_tiers,
        "coverage": covered_tags,
        "partial": [] if void else partial,
        "rationale": ["empty scope routes nowhere."] if void else rationale,
        "thin_evidence": True if void else thin,
    }
