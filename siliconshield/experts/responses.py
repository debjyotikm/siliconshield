"""Standardized multi-expert responses (Qt-free, no Torch).

Every consulted expert returns one envelope preserving its three-way
label, all scores, checkpoint identity, the exact input judged, and
coverage status, validated through the existing result protocol.
Execution errors stay separate from model verdicts: a failed consult
yields an ExpertError, never an insufficient_context label. The
transport is injected so the CLI, Muse, and the browser demo share
this envelope over different worker back ends.
"""

from collections.abc import Callable, Mapping
from typing import Any, NotRequired, TypedDict

from siliconshield.experts import contracts
from siliconshield.experts.capabilities import CapabilityProfile
from siliconshield.experts.contracts import JsonValue
from siliconshield.experts.identity import (
    text_digest,
    validate_execution_identity,
)
from siliconshield.experts.worker import WorkerLoadError


class ExpertResponse(TypedDict):
    """One validated expert assessment with provenance."""

    expert_id: str
    release: str
    checkpoint_sha256: str
    label: str
    scores: dict[str, float]
    state_sha256: str
    truncated: bool
    execution_identity: dict[str, Any]
    question_contract: str
    assessment_id: NotRequired[str]
    scope_sha256: NotRequired[str]


class ExpertError(TypedDict):
    """One failed consult, kept separate from model verdicts."""

    expert_id: str
    stage: str
    detail: NotRequired[str]


Transport = Callable[[str, str], Mapping[str, Any]]
"""Send state text to one expert; return its raw result payload."""


def build_expert_response(
    profile: CapabilityProfile, state: str, payload: Mapping[str, JsonValue]
) -> ExpertResponse:
    """Validate one raw worker payload into an expert envelope.

    Raises ValueError when the payload fails the result protocol
    (missing labels, non-unit scores, verdict/score mismatch, or
    truncated input) or names a different model than the profile:
    callers must surface the failure, never judge from it.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("Worker result must be an object")
    worker_id = payload.get("model_id")
    if worker_id != profile["id"]:
        raise ValueError(
            "Worker model %r does not match profile %r."
            % (worker_id, profile["id"])
        )
    identity = validate_execution_identity(payload.get("execution_identity"))
    for key in ("checkpoint_sha256", "question_contract"):
        if identity[key] != profile[key]:
            raise ValueError("Worker differs from profile " + key)
    if payload.get("state_sha256") != text_digest(state):
        raise ValueError("Worker result belongs to a different input")
    checked = contracts.validate_result(payload)
    label = checked["label"]
    if not isinstance(label, str):
        raise ValueError("Worker result has no string verdict")
    scores = checked["scores"]
    if not isinstance(scores, dict):
        raise ValueError("Worker result has no score mapping")
    numbers: dict[str, float] = {}
    for name, value in scores.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Worker result has a non-numeric score")
        numbers[name] = float(value)
    return {
        "expert_id": profile["id"],
        "release": profile["release"],
        "checkpoint_sha256": profile["checkpoint_sha256"],
        "label": label,
        "scores": numbers,
        "state_sha256": text_digest(state),
        "truncated": False,
        "execution_identity": identity,
        "question_contract": identity["question_contract"],
    }


def consult_experts(
    expert_ids: list[str],
    profiles: Mapping[str, CapabilityProfile],
    state: str,
    transport: Transport,
) -> tuple[list[ExpertResponse], list[ExpertError]]:
    """Consult routed experts; collect responses and errors separately.

    Unknown expert ids and transport failures yield ExpertError
    entries; validation failures do too, whether raised while
    building the envelope or by the transport itself. Responses keep
    profile order. Raises ValueError for empty state text or a
    malformed expert-id list. Responses need request stamping
    before arbitration: route them through InferenceService.assess,
    which binds each response to its request, state, and scope.
    """
    if not isinstance(state, str) or not state.strip():
        raise ValueError("Cannot consult experts on empty state text")
    if isinstance(expert_ids, str) or not isinstance(expert_ids, list):
        raise ValueError("Expert ids must be a list of names")
    for expert_id in expert_ids:
        if not isinstance(expert_id, str) or not expert_id:
            raise ValueError("Expert ids must be a list of names")
    seen: set[str] = set()
    ordered: list[str] = []
    for expert_id in expert_ids:
        if expert_id not in seen:
            seen.add(expert_id)
            ordered.append(expert_id)
    responses: list[ExpertResponse] = []
    errors: list[ExpertError] = []
    for expert_id in ordered:
        profile = profiles.get(expert_id)
        if profile is None:
            errors.append(
                {
                    "expert_id": expert_id,
                    "stage": "route",
                    "detail": "unknown expert id",
                }
            )
            continue
        try:
            payload = transport(expert_id, state)
        except ValueError as exc:
            errors.append(
                {
                    "expert_id": expert_id,
                    "stage": "validate",
                    "detail": "%s" % exc,
                }
            )
            continue
        except WorkerLoadError as exc:
            # A model that never loaded is not a transport flake;
            # mislabeling it would send GPU debugging astray.
            errors.append(
                {
                    "expert_id": expert_id,
                    "stage": "load",
                    "detail": "%s" % exc,
                }
            )
            continue
        except Exception as exc:  # noqa: BLE001 - envelope, not verdict
            errors.append(
                {
                    "expert_id": expert_id,
                    "stage": "transport",
                    "detail": "%s" % exc,
                }
            )
            continue
        try:
            responses.append(build_expert_response(profile, state, payload))
        except ValueError as exc:
            errors.append(
                {
                    "expert_id": expert_id,
                    "stage": "validate",
                    "detail": "%s" % exc,
                }
            )
    return responses, errors
