"""Machine-readable expert capability profiles (Qt-free, no Torch).

A profile binds measured claims about one decision model to its
exact checkpoint: strengths, weaknesses, supported design patterns,
known failure cases, and the evaluations behind them. The router
matches assessment scopes against supported mechanisms; Muse
arbitration reads strengths and known failures. Profiles live next
to the registry (outside Git); validation is CPU-only and never
touches the GPU.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, NotRequired, TypedDict, Union

from siliconshield.experts.identity import (
    is_digest,
    validate_execution_identity,
)

DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CapabilityMeasurement(TypedDict):
    """One measured evaluation behind a capability claim."""

    eval: str
    correct: int
    total: int


class CapabilityProfile(TypedDict):
    """One validated capability profile."""

    id: str
    release: str
    checkpoint_sha256: str
    question_contract: str
    strengths: list[str]
    weaknesses: list[str]
    supported_mechanisms: list[str]
    known_failures: list[str]
    qualification_ref: str
    measurements: list[CapabilityMeasurement]
    routing_evidence: NotRequired[dict[str, Any]]


def _fail(reason: str) -> ValueError:
    return ValueError("Invalid capability profile: " + reason)


def _need_str(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _fail("missing non-empty %r" % key)
    return value


def _need_str_list(raw: dict[str, object], key: str) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise _fail("%r must be a list of non-empty strings" % key)
    return list(value)


def validate_capability_profile(data: object) -> CapabilityProfile:
    """Validate one decoded profile mapping.

    Raises ValueError for schema mismatches, malformed digests,
    invalid measurement counts, or empty claim lists. Strengths,
    supported mechanisms, and measurements must be non-empty: a
    profile with nothing demonstrated qualifies for nothing.
    """
    from siliconshield.schemas import EXPERT_CAPABILITY_SCHEMA

    if not isinstance(data, dict):
        raise _fail("top-level object must be a mapping")
    if data.get("schema") != EXPERT_CAPABILITY_SCHEMA:
        raise _fail("unsupported schema %r" % (data.get("schema"),))
    profile_id = _need_str(data, "id")
    digest = _need_str(data, "checkpoint_sha256")
    if not DIGEST_PATTERN.match(digest):
        raise _fail("malformed checkpoint_sha256")
    raw_measurements = data.get("measurements")
    if not isinstance(raw_measurements, list) or not raw_measurements:
        raise _fail("measurements must be a non-empty list")
    measurements: list[CapabilityMeasurement] = []
    for index, raw in enumerate(raw_measurements):
        if not isinstance(raw, dict):
            raise _fail("measurement %d must be an object" % index)
        correct = raw.get("correct")
        total = raw.get("total")
        if (
            not isinstance(correct, int)
            or isinstance(correct, bool)
            or not isinstance(total, int)
            or isinstance(total, bool)
            or total <= 0
            or not 0 <= correct <= total
        ):
            raise _fail("measurement %d has invalid counts" % index)
        measurements.append(
            {
                "eval": _need_str(raw, "eval"),
                "correct": correct,
                "total": total,
            }
        )
    strengths = _need_str_list(data, "strengths")
    supported = _need_str_list(data, "supported_mechanisms")
    if not strengths or not supported:
        raise _fail("strengths and supported_mechanisms must be non-empty")
    profile: CapabilityProfile = {
        "id": profile_id,
        "release": _need_str(data, "release"),
        "checkpoint_sha256": digest,
        "question_contract": _need_str(data, "question_contract"),
        "strengths": strengths,
        "weaknesses": _need_str_list(data, "weaknesses"),
        "supported_mechanisms": supported,
        "known_failures": _need_str_list(data, "known_failures"),
        "qualification_ref": _need_str(data, "qualification_ref"),
        "measurements": measurements,
    }
    evidence = data.get("routing_evidence")
    if evidence is not None:
        profile["routing_evidence"] = validate_routing_evidence(
            evidence, profile
        )
    return profile


def validate_routing_evidence(
    value: Any, profile: CapabilityProfile
) -> dict[str, Any]:
    """Validate measured counts bound to one complete inference identity."""
    if not isinstance(value, dict):
        raise _fail("routing_evidence must be an object")
    execution = validate_execution_identity(value.get("execution_identity"))
    for key in ("checkpoint_sha256", "question_contract"):
        if execution[key] != profile[key]:
            raise _fail("routing evidence differs from profile " + key)
    if not is_digest(value.get("benchmark_sha256")):
        raise _fail("routing evidence needs benchmark bytes")
    if (
        not isinstance(value.get("benchmark_source"), str)
        or not value["benchmark_source"]
    ):
        raise _fail("routing evidence needs benchmark source")
    areas = value.get("areas")
    if not isinstance(areas, dict) or not areas:
        raise _fail("routing evidence needs area counts")
    for area, counts in areas.items():
        if (
            not isinstance(area, str)
            or not area
            or not isinstance(counts, dict)
        ):
            raise _fail("invalid routing evidence area")
        for key in ("correct", "total", "violations", "false_safe", "errors"):
            if type(counts.get(key)) is not int or counts[key] < 0:
                raise _fail("invalid routing evidence count " + key)
        if not (
            counts["total"] > 0
            and counts["correct"] + counts["errors"] <= counts["total"]
            and counts["false_safe"] <= counts["violations"] <= counts["total"]
        ):
            raise _fail("inconsistent routing evidence counts")
    return dict(value)


def load_capability_profile(
    path: Union[str, os.PathLike[str]],
) -> CapabilityProfile:
    """Load and validate one expert capability profile file.

    Raises ValueError with an actionable message for unreadable
    files or invalid profile content.
    """
    root = Path(path)
    try:
        data = json.loads(root.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read capability profile: %s" % exc)
    return validate_capability_profile(data)


def covers(profile: CapabilityProfile, mechanism: str) -> bool:
    """Whether the profile claims the mechanism as supported."""
    return mechanism in profile["supported_mechanisms"]
