"""Muse arbitration over expert evidence (Qt-free, no Torch).

Uses the repository's tool-free ``MuseClient`` (local `muse` CLI, login
via the existing Muse setup, no credentials copied) with model
``muse-spark-1.3-contributor`` to arbitrate one assessment from expert
responses. The verdict is strictly validated; any Muse failure raises
ValueError so callers fall back explicitly, never silently.
"""

import json
import os
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, Callable

from siliconshield.experts.capabilities import validate_capability_profile
from siliconshield.experts.identity import digest, state_sections, text_digest
from siliconshield.experts.inspection import inspect_structure
from siliconshield.experts.router import route_scope
from siliconshield.schemas import EXPERT_CAPABILITY_SCHEMA


def default_muse_audit_dir() -> Path:
    """Return the user-level Muse audit location (outside Git)."""
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "siliconshield-experts" / "muse-audit"


ARBITRATE_TOOL = {
    "type": "function",
    "function": {
        "name": "experts.arbitrate",
        "description": "Explain a conclusion from expert evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "conclusion": {
                    "type": "string",
                    "enum": [
                        "satisfies_requirement",
                        "violates_requirement",
                        "insufficient_context",
                        "unresolved",
                    ],
                },
                "explanation": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "expert_id": {"type": "string"},
                            "relevance": {"type": "string"},
                        },
                        "required": ["expert_id", "relevance"],
                        "additionalProperties": False,
                    },
                },
                "open_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "conclusion",
                "explanation",
                "evidence",
                "open_questions",
            ],
            "additionalProperties": False,
        },
    },
}

_ARBITRATION_CONCLUSIONS = frozenset(
    {
        "satisfies_requirement",
        "violates_requirement",
        "insufficient_context",
        "unresolved",
    }
)


def validate_arbitration(
    arguments: Any, consulted: Collection[str]
) -> dict[str, Any]:
    """Strictly validate one experts.arbitrate tool call payload.

    Raises ValueError for any shape or content violation: unknown
    top-level keys, unknown conclusion, empty explanation, missing
    or duplicate evidence, evidence entries that do not name a
    consulted expert and its relevance, or open questions that are
    not non-empty strings. Only an unresolved conclusion may carry
    empty evidence, since it makes no claim.
    """
    if not isinstance(arguments, dict):
        raise ValueError("Arbitration payload must be an object.")
    unknown = [
        key
        for key in arguments
        if key
        not in ("conclusion", "explanation", "evidence", "open_questions")
    ]
    if unknown:
        raise ValueError(
            "Arbitration payload has unknown keys: %s" % sorted(unknown)
        )
    conclusion = arguments.get("conclusion")
    if conclusion not in _ARBITRATION_CONCLUSIONS:
        raise ValueError(
            "Arbitration conclusion %r is invalid." % (conclusion,)
        )
    explanation = arguments.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Arbitration needs a non-empty explanation.")
    evidence = arguments.get("evidence")
    if not isinstance(evidence, list) or (
        not evidence and conclusion != "unresolved"
    ):
        raise ValueError("Arbitration needs non-empty evidence.")
    allowed = set(consulted)
    checked_evidence: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in evidence:
        if not isinstance(entry, dict):
            raise ValueError("Arbitration evidence must be objects.")
        extra = [key for key in entry if key not in ("expert_id", "relevance")]
        if extra:
            raise ValueError(
                "Arbitration evidence has unknown keys: %s" % sorted(extra)
            )
        expert_id = entry.get("expert_id")
        relevance = entry.get("relevance")
        if not isinstance(expert_id, str) or not expert_id.strip():
            raise ValueError("Arbitration evidence needs an expert id.")
        if expert_id in seen:
            raise ValueError(
                "Arbitration cites %r more than once." % (expert_id,)
            )
        seen.add(expert_id)
        if expert_id not in allowed:
            raise ValueError(
                "Arbitration cites %r, which was not consulted." % (expert_id,)
            )
        if not isinstance(relevance, str) or not relevance.strip():
            raise ValueError("Arbitration evidence needs relevance text.")
        checked_evidence.append(
            {"expert_id": expert_id, "relevance": relevance}
        )
    questions = arguments.get("open_questions")
    if not isinstance(questions, list) or any(
        not isinstance(item, str) or not item.strip() for item in questions
    ):
        raise ValueError("Arbitration open_questions must be strings.")
    return {
        "conclusion": conclusion,
        "explanation": explanation,
        "evidence": checked_evidence,
        "open_questions": list(questions),
    }


def _normalize(text: str) -> str:
    """Collapse whitespace for requirement-in-state matching."""
    return " ".join(text.split())


def _check_effective_routing(
    routing: Any, effective: Mapping[str, Any], planned: Mapping[str, Any]
) -> None:
    """Check effective routing against recomputed and planned parts.

    Experts, tiers, and thin flags must match the recomputation from
    usable responses; the scope picture (unsupported, partial,
    coverage, planned rationale lines) must match the plan, since
    failures narrow who answered, never what was covered.
    """
    if not isinstance(routing, dict):
        raise ValueError("Assessment routing must be an object")
    for key in ("scope", "experts", "tiers", "area_tiers", "thin_evidence"):
        if routing.get(key) != effective.get(key):
            raise ValueError(
                "Assessment routing is stale or differs from usable evidence"
            )
    for key in ("unsupported", "partial", "coverage"):
        if routing.get(key) != planned.get(key):
            raise ValueError("Assessment routing lost its planned coverage")
    planned_lines = planned.get("rationale", [])
    if routing.get("rationale", [])[: len(planned_lines)] != planned_lines:
        raise ValueError("Assessment routing lost its planned rationale")


def build_arbitration_input(
    assessment: Mapping[str, Any],
    state: str,
    requirement: str,
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
    structure: Mapping[str, Any] | None = None,
    assumptions: str | None = None,
) -> dict[str, Any]:
    """Bind one assessment into the exact object Muse arbitrates.

    The requirement must appear in the judged state (whitespace
    ignored): arbitrating a different requirement than the experts
    judged is refused. Profiles are trimmed to identity, release,
    contract, tiers evidence, and claims; anything else stays out.
    Raises ValueError for empty state/requirement or a mismatch.
    """
    if not isinstance(state, str) or not isinstance(requirement, str):
        raise ValueError("Arbitration needs state and requirement text")
    judged_requirement, judged_assumptions, judged_rtl = state_sections(state)
    if _normalize(requirement) != _normalize(judged_requirement):
        raise ValueError("Requirement differs from the full judged requirement")
    if assumptions is not None and _normalize(assumptions) != _normalize(
        judged_assumptions
    ):
        raise ValueError("Assumptions differ from the judged assumptions")
    assumptions = judged_assumptions
    state_hash = text_digest(state)
    scope = assessment.get("scope")
    if not isinstance(scope, list):
        raise ValueError("Assessment scope must be a list")
    request_id = assessment.get("request_id")
    if (
        not isinstance(request_id, str)
        or not request_id
        or assessment.get("state_sha256") != state_hash
        or assessment.get("scope_sha256") != digest(scope)
    ):
        raise ValueError("Assessment is not bound to this input and scope")
    responses = assessment.get("responses", [])
    if not isinstance(responses, list):
        raise ValueError("Assessment responses must be a list")
    if profiles is not None and not isinstance(profiles, Mapping):
        raise ValueError("Arbitration profiles must be an object")
    checked_profiles = []
    identities = {}
    contracts = set()
    for response in responses:
        if not isinstance(response, dict):
            raise ValueError("Assessment response must be an object")
        expert_id = response.get("expert_id")
        if not isinstance(expert_id, str) or not expert_id:
            raise ValueError("Assessment response names no expert")
        label = response.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("Assessment response has no label")
        if expert_id in identities:
            raise ValueError("Duplicate expert response")
        if (
            response.get("state_sha256") != state_hash
            or response.get("assessment_id") != request_id
            or response.get("scope_sha256") != digest(scope)
        ):
            raise ValueError("Response does not belong to this assessment")
        raw = (profiles or {}).get(expert_id)
        if raw is None:
            raise ValueError("Missing profile for successful expert")
        if not isinstance(raw, Mapping):
            raise ValueError("Arbitration profile must be an object")
        profile = validate_capability_profile(
            {**raw, "schema": EXPERT_CAPABILITY_SCHEMA}
        )
        for key in ("release", "checkpoint_sha256", "question_contract"):
            if response.get(key) != dict(profile)[key]:
                raise ValueError("Response/profile identity mismatch")
        from siliconshield.experts.identity import validate_execution_identity

        identity = validate_execution_identity(
            response.get("execution_identity")
        )
        if any(
            identity[key] != dict(profile)[key]
            for key in ("checkpoint_sha256", "question_contract")
        ):
            raise ValueError("Response execution identity differs from profile")
        identities[expert_id] = identity
        contracts.add(
            (identity["question_contract"], identity["question_sha256"])
        )
        checked_profiles.append(profile)
    if len(contracts) > 1:
        raise ValueError("Experts used incompatible question contracts")
    effective = route_scope(scope, checked_profiles, identities)
    planned = assessment.get("planned_routing")
    current = assessment.get("routing")
    if isinstance(planned, dict):
        if not isinstance(current, dict):
            raise ValueError("Assessment has planned routing but no routing")
        _check_effective_routing(current, effective, planned)
    elif current != effective:
        raise ValueError(
            "Assessment routing is stale or differs from usable evidence"
        )
    responder_ids = {profile["id"] for profile in checked_profiles}
    trimmed = {}
    for expert_id, profile_claim in (profiles or {}).items():
        if expert_id not in responder_ids:
            continue
        if not isinstance(profile_claim, Mapping):
            raise ValueError("Arbitration profile must be an object")
        trimmed[expert_id] = {
            key: profile_claim.get(key)
            for key in (
                "id",
                "release",
                "checkpoint_sha256",
                "question_contract",
                "strengths",
                "weaknesses",
                "supported_mechanisms",
                "known_failures",
            )
        }
        measured = profile_claim.get("routing_evidence")
        if measured and measured["execution_identity"] == identities.get(
            expert_id
        ):
            trimmed[expert_id]["routing_evidence"] = measured
            trimmed[expert_id]["measurement_status"] = (
                "matched to this execution"
            )
        else:
            trimmed[expert_id]["measurement_status"] = (
                "unrated for this execution"
            )
    raw_errors = assessment.get("errors", [])
    if not isinstance(raw_errors, list):
        raise ValueError("Assessment errors must be a list")
    bound_errors = []
    for item in raw_errors:
        if not isinstance(item, dict):
            raise ValueError("Assessment error must be an object")
        failed_id = item.get("expert_id")
        stage = item.get("stage")
        if not isinstance(failed_id, str) or not failed_id:
            raise ValueError("Assessment error names no expert")
        if not isinstance(stage, str) or not stage:
            raise ValueError("Assessment error has no stage")
        bound_errors.append(
            {
                "expert_id": failed_id,
                "stage": stage,
                "detail": item.get("detail", ""),
            }
        )
    if structure is None:
        # No caller supplies its own inspection yet; ground the
        # arbitration in lexical hints rather than an empty map.
        # An explicit mapping (even empty) still wins.
        structure = inspect_structure(judged_rtl, [])
    return {
        "request_id": request_id,
        "state_sha256": state_hash,
        "scope_sha256": digest(scope),
        "requirement": requirement,
        "assumptions": assumptions,
        "state": state,
        "structure": dict(structure),
        "scope": list(assessment.get("scope", [])),
        "routing": assessment.get("routing", {}),
        "profiles": trimmed,
        "responses": list(assessment.get("responses", [])),
        "errors": bound_errors,
    }


def arbitrate_assessment(
    assessment: Mapping[str, Any],
    state: str,
    requirement: str,
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
    structure: Mapping[str, Any] | None = None,
    assumptions: str | None = None,
    audit_dir: Path | None = None,
    model: str = "muse-spark-1.3-contributor",
    timeout: int = 300,
    client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Ask Muse to arbitrate bound assessment evidence; return verdict.

    Muse sees the judged state, requirement, routing tiers, thin
    flags, profiles, and every response and error, and weighs expert
    suitability, RTL evidence, and disagreements; confidence or
    majority voting alone must not decide. With no successful
    responses there is nothing to arbitrate, so unresolved returns
    directly without calling Muse. Usable responses are arbitrated
    even when other experts failed; the failures stay disclosed in
    the bound errors. Cited experts must come from the successful
    set. Raises ValueError when inputs mismatch, the CLI is
    missing, the reply has no tool call, or validation fails.
    """
    bound = build_arbitration_input(
        assessment, state, requirement, profiles, structure, assumptions
    )
    responses = bound["responses"]
    binding = {
        key: bound[key]
        for key in ("request_id", "state_sha256", "scope_sha256")
    }
    binding["requirement_sha256"] = text_digest(_normalize(requirement))
    if not responses:
        return {
            **binding,
            "conclusion": "unresolved",
            "explanation": (
                "No expert produced a usable response for this assessment."
            ),
            "evidence": [
                {
                    "expert_id": item["expert_id"],
                    "relevance": "Consultation failed at stage %s."
                    % item["stage"],
                }
                for item in bound["errors"]
            ],
            "open_questions": [
                "Consult a qualified expert or fix the execution errors, "
                "then arbitrate again."
            ],
            "agreement": False,
        }
    if client_factory is None:
        import shutil

        executable = shutil.which("muse")
        if executable is None:
            raise ValueError("`muse` CLI not found on PATH.")
        from siliconshield.research_loop.procedures.muse_client import (
            MuseClient,
        )

        target = audit_dir or default_muse_audit_dir()

        def client_factory(executable_path: str, audit: Path) -> Any:
            return MuseClient(
                executable=Path(executable_path),
                audit_dir=audit,
                model=model,
                effort="low",
                timeout=timeout,
            )

        client = client_factory(executable, target)
    else:
        client = client_factory()
    messages = [
        {
            "role": "system",
            "content": (
                "You arbitrate RTL security assessments from the bound "
                "evidence object: judged state, requirement, routing "
                "tiers and thin flags, expert profiles, responses, and "
                "errors. Use exactly one call to experts.arbitrate, "
                "citing only experts with successful responses in this "
                "assessment. Explain the conclusion from expert "
                "suitability, RTL evidence, and disagreements; "
                "confidence or majority voting alone must not decide. "
                "Quote numerical capability claims only from matched "
                "routing_evidence. Historical profile claims are not current "
                "measurements. Use unresolved when evidence conflicts "
                "or coverage fails, and list what would settle it."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                bound,
                sort_keys=True,
                allow_nan=False,
            ),
        },
    ]
    consulted = [item["expert_id"] for item in responses]
    reply = client.complete(messages, tools=[ARBITRATE_TOOL])
    # A malformed model reply must fail as a validation error, not
    # leak TypeError/AttributeError to callers that only expect
    # ValueError for bad arbitration output.
    content = reply.get("content") if isinstance(reply, dict) else reply
    calls = reply.get("tool_calls", []) if isinstance(reply, dict) else []
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError(
            "Arbitration client must return exactly one arbitration "
            "call; reply content: %r" % (content,)
        )
    first = calls[0]
    function = first.get("function", {}) if isinstance(first, dict) else {}
    if function.get("name") != "experts.arbitrate":
        raise ValueError(
            "Arbitration reply contained no experts.arbitrate call; "
            "reply content: %r" % (reply.get("content"),)
        )
    payload = function.get("arguments", {})
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError as exc:
            raise ValueError(
                "Arbitration payload must be an object: %s" % exc
            ) from exc
    verdict = {
        **validate_arbitration(payload, consulted),
        **binding,
    }
    labels = [item["label"] for item in responses]
    if (
        labels
        and all(label == "insufficient_context" for label in labels)
        and verdict["conclusion"] not in ("insufficient_context", "unresolved")
    ):
        raise ValueError(
            "Arbitration overrides unanimous insufficient_context with %r."
            % (verdict["conclusion"],)
        )
    verdict["agreement"] = len(set(labels)) <= 1
    return verdict
