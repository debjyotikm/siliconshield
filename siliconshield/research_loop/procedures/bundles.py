"""Publish and consume qualified prompt guidance without importing Harness2."""

from pathlib import Path
from typing import Any

from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    digest,
    read_json,
)
from siliconshield.schemas import PROCEDURE_BUNDLE_SCHEMA

GRADE_FIELDS = (
    "execution_complete",
    "action_correct",
    "evidence_correct",
    "explanation_correct",
    "final_consistent",
    "verified",
)


def qualifies(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Require successful checks and no per-task/component regression."""
    if (
        not isinstance(baseline, dict)
        or not isinstance(candidate, dict)
        or not baseline
        or baseline.keys() != candidate.keys()
        or any(not isinstance(v, dict) for v in baseline.values())
        or any(not isinstance(v, dict) for v in candidate.values())
    ):
        return False
    return all(
        candidate[key].get("verified") is True
        and all(candidate[key].get(field) is True for field in GRADE_FIELDS)
        and all(
            not value or candidate[key].get(field) is True
            for field, value in baseline[key].items()
            if field in GRADE_FIELDS or field == "parameters_correct"
        )
        for key in baseline
    )


def publish_bundle(
    path: Path,
    guidance: str,
    identity: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Atomically replace guidance only after replay qualification succeeds."""
    if not guidance.strip() or not qualifies(baseline, candidate):
        raise ValueError("Procedure did not pass replay qualification")
    body: dict[str, Any] = {
        "schema": PROCEDURE_BUNDLE_SCHEMA,
        "scope": "development_procedure_replay",
        "guidance": guidance,
        "identity": identity,
        "baseline": baseline,
        "candidate": candidate,
        "not_evidence_of_training_improvement": True,
    }
    bundle = {**body, "digest": digest(body)}
    atomic_json(path, bundle)
    return bundle


def read_bundle(path: Path, model: str) -> dict[str, Any]:
    """Validate a locally trusted bundle before using it in a session.

    The digest detects damage; it is not a signature against a malicious writer.
    The bundle directory must have the same owner as the campaign configuration.
    """
    bundle = read_json(path)
    if not isinstance(bundle, dict):
        raise ValueError("Procedure bundle must be an object")
    body = {key: value for key, value in bundle.items() if key != "digest"}
    identity = body.get("identity", {})
    if (
        body.get("schema") != PROCEDURE_BUNDLE_SCHEMA
        or body.get("scope") != "development_procedure_replay"
        or bundle.get("digest") != digest(body)
        or not isinstance(identity, dict)
        or identity.get("model") != model
        or not isinstance(body.get("guidance"), str)
        or not body["guidance"].strip()
        or not qualifies(body.get("baseline", {}), body.get("candidate", {}))
    ):
        raise ValueError(
            "Procedure bundle is damaged, unqualified, or for another model"
        )
    return dict(bundle)
