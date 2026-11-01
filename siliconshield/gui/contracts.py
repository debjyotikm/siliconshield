"""Demo shim: the worker protocol lives in experts.contracts."""

from siliconshield.experts.contracts import (
    ANALYZE_TIMEOUT_SECONDS,
    LOAD_TIMEOUT_SECONDS,
    PROTOCOL_VERSION,
    SCORE_SUM_TOLERANCE,
    JsonValue,
    make_analyze_request,
    make_load_request,
    make_shutdown_request,
    message_kind,
    validate_result,
)

__all__ = [
    "ANALYZE_TIMEOUT_SECONDS",
    "LOAD_TIMEOUT_SECONDS",
    "PROTOCOL_VERSION",
    "SCORE_SUM_TOLERANCE",
    "JsonValue",
    "make_analyze_request",
    "make_load_request",
    "make_shutdown_request",
    "message_kind",
    "validate_result",
]
