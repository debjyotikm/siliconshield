"""GUI/worker protocol: message shapes, validation, and verdict rules.

The transport is newline-delimited JSON on local pipes. Stdout carries
protocol lines only; diagnostics go to stderr. Every worker message carries
the worker's session id so a delayed reply can never overwrite a newer run.
Validation rejects malformed traffic loudly: no verdict is ever produced
from unobserved, truncated, or inconsistent content.
"""

import math
from collections.abc import Mapping
from typing import Union

JsonValue = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JsonValue"],
    dict[str, "JsonValue"],
]
"""JSON-serializable values exchanged with the worker process."""

PROTOCOL_VERSION = 1

LOAD_TIMEOUT_SECONDS = 180
ANALYZE_TIMEOUT_SECONDS = 60

SCORE_SUM_TOLERANCE = 1e-3
"""Allowed absolute deviation of the three scores from a unit total."""


def make_load_request(
    request_id: str, model_id: str | None = None
) -> dict[str, JsonValue]:
    """Build the single load command sent after worker startup."""
    request: dict[str, JsonValue] = {
        "type": "load",
        "request_id": request_id,
        "protocol": PROTOCOL_VERSION,
    }
    if model_id is not None:
        request["model_id"] = model_id
    return request


def make_analyze_request(request_id: str, state: str) -> dict[str, JsonValue]:
    """Build one analysis request for an immutable source snapshot."""
    return {
        "type": "analyze",
        "request_id": request_id,
        "protocol": PROTOCOL_VERSION,
        "state": state,
    }


def make_shutdown_request() -> dict[str, JsonValue]:
    """Build the graceful shutdown command for the owned worker."""
    return {"type": "shutdown", "protocol": PROTOCOL_VERSION}


def message_kind(message: Mapping[str, JsonValue]) -> str:
    """Return the validated message type, or raise ValueError."""
    kind = message.get("type")
    if not isinstance(kind, str):
        raise ValueError("Worker message type must be a string")
    if kind not in ("progress", "ready", "result", "error"):
        raise ValueError("Unknown worker message type: %r" % (kind,))
    return kind


def _require_finite_number(value: JsonValue, name: str) -> float:
    """Return a finite number or raise ValueError naming the field."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Worker field is not a number: %s" % name)
    if not math.isfinite(value):
        raise ValueError("Worker field is not finite: %s" % name)
    return float(value)


def validate_result(
    payload: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Validate a worker result and return the checked assessment fields.

    Requires the three fixed labels, finite scores in [0, 1] summing to a
    unit total, an explicit verdict consistent with the top score, and
    coverage fields proving the model saw the complete input. Raises
    ValueError with an actionable message otherwise; callers must surface
    the failure instead of producing a verdict.
    """
    label = payload.get("label")
    probabilities = payload.get("probabilities")
    usage = payload.get("usage")
    if not isinstance(probabilities, dict):
        raise ValueError("Worker result has no probability mapping")
    expected = (
        "satisfies_requirement",
        "violates_requirement",
        "insufficient_context",
    )
    missing = [name for name in expected if name not in probabilities]
    if missing:
        raise ValueError(
            "Worker result is missing labels: %s" % sorted(missing)
        )
    scores: dict[str, float] = {}
    for name in expected:
        number = _require_finite_number(
            probabilities[name], "probabilities[%s]" % name
        )
        if not 0.0 <= number <= 1.0:
            raise ValueError("Worker score out of [0, 1]: %s" % name)
        scores[name] = number
    total = sum(scores[name] for name in expected)
    if abs(total - 1.0) > SCORE_SUM_TOLERANCE:
        raise ValueError(
            "Worker scores do not sum to one within tolerance: %r" % (total,)
        )
    best = max(expected, key=lambda name: scores[name])
    if label != best:
        raise ValueError(
            "Worker verdict %r disagrees with top score %r" % (label, best)
        )
    coverage = payload.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Worker result is missing coverage evidence")
    dropped = coverage.get("state_tokens_dropped")
    extra_truncated = coverage.get("truncated_questions")
    if coverage.get("truncated") or (
        isinstance(dropped, (int, float))
        and not isinstance(dropped, bool)
        and dropped > 0
    ):
        raise ValueError(
            "Model input was truncated; no verdict may be produced"
        )
    if isinstance(extra_truncated, list) and extra_truncated:
        raise ValueError(
            "Model questions were truncated; no verdict may be produced"
        )
    if usage is not None and not isinstance(usage, dict):
        raise ValueError("Worker usage must be a mapping when present")
    scores_json: dict[str, JsonValue] = dict(scores)
    checked: dict[str, JsonValue] = {
        "label": label if isinstance(label, str) else None,
        "scores": scores_json,
        "usage": usage if isinstance(usage, dict) else None,
        "truncated": False,
    }
    return checked
