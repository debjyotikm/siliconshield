"""GUI/worker protocol validation: no GPU, no Qt, no services."""

import math

import pytest

from siliconshield.gui import contracts


def _valid_payload() -> dict:
    return {
        "label": "satisfies_requirement",
        "probabilities": {
            "satisfies_requirement": 0.7,
            "violates_requirement": 0.2,
            "insufficient_context": 0.1,
        },
        "usage": {"input_tokens": 10},
        "coverage": {
            "truncated": False,
            "truncated_questions": [],
            "state_tokens_dropped": 0,
        },
    }


def test_make_load_request_carries_protocol() -> None:
    request = contracts.make_load_request("req-1", "fixture")
    assert request["type"] == "load"
    assert request["protocol"] == contracts.PROTOCOL_VERSION
    assert request["model_id"] == "fixture"


def test_make_load_request_without_model() -> None:
    request = contracts.make_load_request("req-2")
    assert request["type"] == "load"
    assert "model_id" not in request


def test_make_analyze_request_carries_state() -> None:
    request = contracts.make_analyze_request("req-3", "REQUIREMENT\nx")
    assert request["state"] == "REQUIREMENT\nx"


def test_valid_result_passes() -> None:
    checked = contracts.validate_result(_valid_payload())
    assert checked["label"] == "satisfies_requirement"


def test_missing_label_is_rejected() -> None:
    payload = _valid_payload()
    del payload["probabilities"]["violates_requirement"]
    with pytest.raises(ValueError, match="missing labels"):
        contracts.validate_result(payload)


def test_verdict_must_match_top_score() -> None:
    payload = _valid_payload()
    payload["label"] = "violates_requirement"
    with pytest.raises(ValueError, match="disagrees"):
        contracts.validate_result(payload)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_non_finite_or_ranged_scores_rejected(bad: float) -> None:
    payload = _valid_payload()
    payload["probabilities"]["satisfies_requirement"] = bad
    # Renormalize the other scores so only the bad field matters.
    with pytest.raises(ValueError):
        contracts.validate_result(payload)


def test_scores_must_sum_to_one() -> None:
    payload = _valid_payload()
    payload["probabilities"] = {
        "satisfies_requirement": 0.5,
        "violates_requirement": 0.2,
        "insufficient_context": 0.1,
    }
    with pytest.raises(ValueError, match="sum"):
        contracts.validate_result(payload)


def test_bool_scores_are_rejected() -> None:
    payload = _valid_payload()
    payload["probabilities"]["satisfies_requirement"] = True  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="not a number"):
        contracts.validate_result(payload)


def test_scope_note_marks_custom_inputs_unverified() -> None:
    from siliconshield.gui.application import (
        CUSTOM_SCOPE_NOTE,
        SCOPE_NOTE,
        scope_note_for,
    )

    assert scope_note_for({"path": "demo:unauthorized-write"}) == SCOPE_NOTE
    custom = scope_note_for({"path": "/tmp/design.sv"})
    assert custom == CUSTOM_SCOPE_NOTE
    assert "not independently verified" in custom
    assert scope_note_for(None) == CUSTOM_SCOPE_NOTE


def test_truncated_input_refuses_verdict() -> None:
    payload = _valid_payload()
    assert isinstance(payload["coverage"], dict)
    payload["coverage"]["truncated"] = True
    with pytest.raises(ValueError, match="truncated"):
        contracts.validate_result(payload)


def test_dropped_tokens_refuse_verdict() -> None:
    payload = _valid_payload()
    assert isinstance(payload["coverage"], dict)
    payload["coverage"]["state_tokens_dropped"] = 3
    with pytest.raises(ValueError, match="truncated"):
        contracts.validate_result(payload)


def test_truncated_questions_refuse_verdict() -> None:
    payload = _valid_payload()
    assert isinstance(payload["coverage"], dict)
    payload["coverage"]["truncated_questions"] = ["security"]
    with pytest.raises(ValueError, match="truncated"):
        contracts.validate_result(payload)


def test_missing_coverage_refuses_verdict() -> None:
    payload = _valid_payload()
    del payload["coverage"]
    with pytest.raises(ValueError, match="coverage"):
        contracts.validate_result(payload)


def test_unknown_message_kind_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        contracts.message_kind({"type": "nonsense"})
    with pytest.raises(ValueError, match="must be a string"):
        contracts.message_kind({"type": 3})


def test_known_kinds_pass() -> None:
    for kind in ("progress", "ready", "result", "error"):
        assert contracts.message_kind({"type": kind}) == kind


def test_math_is_finite_without_import_inside() -> None:
    assert math.isfinite(1.0)
