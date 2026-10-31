"""Controller state machine: Qt-free, fixture transports only."""

from typing import Any

import pytest

from siliconshield.gui.controller import (
    ANALYZING,
    ERROR,
    LOADING,
    READY,
    UNLOADED,
    Controller,
)


class FakeTransport:
    """Scripted worker transport without processes or GPUs."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any], str]] = []
        self.terminated = False
        self.on_message = lambda _m: None
        self.on_finished = lambda _c, _e: None
        self.on_error = lambda _d: None

    def send_load(self, model: dict[str, Any], request_id: str) -> None:
        self.sent.append(("load", dict(model), request_id))

    def send_analyze(self, inputs: dict[str, Any], request_id: str) -> None:
        self.sent.append(("analyze", dict(inputs), request_id))

    def terminate(self) -> None:
        self.terminated = True


def _controller() -> tuple[Controller, dict[str, Any], FakeTransport]:
    seen: dict[str, Any] = {
        "states": [],
        "progress": [],
        "results": [],
        "errors": [],
    }
    control = Controller(
        on_state=lambda name: seen["states"].append(name),
        on_progress=lambda stage, detail: seen["progress"].append(
            (stage, detail)
        ),
        on_result=lambda result: seen["results"].append(result),
        on_error=lambda message, stage: seen["errors"].append((message, stage)),
    )
    transport = FakeTransport()
    control.attach(transport)
    return control, seen, transport


def _model(model_id: str = "fixture") -> dict[str, Any]:
    return {"id": model_id, "display_name": "Fixture"}


def test_load_success_enables_run() -> None:
    control, seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    assert control.state == LOADING
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "progress",
            "request_id": request_id,
            "session_id": None,
            "stage": "Loading model",
            "detail": "weights",
        }
    )
    assert seen["progress"]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.state == READY
    assert control.can_analyze(True)
    assert not control.can_analyze(False)


def test_load_failure_never_produces_ready() -> None:
    control, seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "error",
            "request_id": request_id,
            "session_id": "sess-1",
            "stage": "load",
            "detail": "CUDA is unavailable",
        }
    )
    assert control.state == ERROR
    assert seen["errors"]
    assert not control.can_analyze(True)


def test_cpu_fallback_during_load_refuses_ready() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 1,
        }
    )
    assert control.state == ERROR


def test_cpu_ready_is_accepted() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cpu",
            "cpu_fallback_count": 0,
        }
    )
    assert control.state == READY


def test_unknown_device_ready_is_rejected() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "tpu",
            "cpu_fallback_count": 0,
        }
    )
    assert control.state == ERROR


def test_model_mismatch_disables_run() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.state == READY
    control.select_model("other")
    assert not control.can_analyze(True)


def test_one_worker_survives_two_analyses() -> None:
    control, seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-9",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    for key in ("first", "second"):
        assert control.analyze(
            {
                "requirement": "req",
                "rtl": "module x; endmodule",
                "state": "STATE %s" % key,
            },
            key,
        )
        assert control.state == ANALYZING
        pending = transport.sent[-1][2]
        control.on_transport_message(
            {
                "type": "result",
                "request_id": pending,
                "session_id": "sess-9",
                "label": "satisfies_requirement",
                "probabilities": {
                    "satisfies_requirement": 0.6,
                    "violates_requirement": 0.3,
                    "insufficient_context": 0.1,
                },
                "usage": {},
                "coverage": {
                    "truncated": False,
                    "truncated_questions": [],
                    "state_tokens_dropped": 0,
                },
            }
        )
        assert control.state == READY
    assert len(seen["results"]) == 2
    assert not transport.terminated


def test_stale_replies_are_ignored() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.analyze(
        {"requirement": "req", "rtl": "rtl", "state": "STATE"},
        "key-1",
    )
    control.on_transport_message(
        {
            "type": "result",
            "request_id": "wrong",
            "session_id": "sess-1",
            "label": "satisfies_requirement",
            "probabilities": {
                "satisfies_requirement": 0.6,
                "violates_requirement": 0.3,
                "insufficient_context": 0.1,
            },
            "usage": {},
            "coverage": {
                "truncated": False,
                "truncated_questions": [],
                "state_tokens_dropped": 0,
            },
        }
    )
    assert control.state == ANALYZING
    assert control.stale_replies == 1


def test_truncated_result_refuses_verdict() -> None:
    control, seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.analyze(
        {"requirement": "req", "rtl": "rtl", "state": "STATE"},
        "key-1",
    )
    pending = transport.sent[-1][2]
    control.on_transport_message(
        {
            "type": "result",
            "request_id": pending,
            "session_id": "sess-1",
            "label": "satisfies_requirement",
            "probabilities": {
                "satisfies_requirement": 0.6,
                "violates_requirement": 0.3,
                "insufficient_context": 0.1,
            },
            "usage": {},
            "coverage": {
                "truncated": True,
                "truncated_questions": [],
                "state_tokens_dropped": 0,
            },
        }
    )
    assert control.state == ERROR
    assert seen["errors"]
    assert not seen["results"]


def test_unknown_worker_message_fails_closed() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message({"type": "mystery", "request_id": request_id})
    assert control.state == ERROR


def test_cancel_terminates_owned_worker() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.analyze(
        {"requirement": "req", "rtl": "rtl", "state": "STATE"},
        "key-1",
    )
    assert control.state == ANALYZING
    control.cancel()
    assert transport.terminated
    assert control.state == UNLOADED


def test_load_timeout_terminates_worker() -> None:
    control, seen, _transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    control.on_load_timeout()
    assert control.state == UNLOADED
    assert seen["errors"]


def test_unload_releases_worker() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    control.unload()
    assert transport.terminated
    control.on_transport_finished(False, "exit code 0")
    assert control.state == UNLOADED


def test_unexpected_exit_clears_readiness() -> None:
    control, seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.state == READY
    assert transport.sent
    control.on_transport_finished(True, "exit code 1")
    assert control.state == ERROR
    assert seen["errors"]


def test_input_edits_invalidate_results() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.analyze(
        {"requirement": "req", "rtl": "rtl", "state": "STATE"},
        "key-1",
    )
    pending = transport.sent[-1][2]
    control.on_transport_message(
        {
            "type": "result",
            "request_id": pending,
            "session_id": "sess-1",
            "label": "satisfies_requirement",
            "probabilities": {
                "satisfies_requirement": 0.6,
                "violates_requirement": 0.3,
                "insufficient_context": 0.1,
            },
            "usage": {},
            "coverage": {
                "truncated": False,
                "truncated_questions": [],
                "state_tokens_dropped": 0,
            },
        }
    )
    assert control.has_stale_result("key-2")
    assert not control.has_stale_result("key-1")
    control.clear_result()
    assert not control.has_stale_result("key-2")


@pytest.mark.parametrize("state", [LOADING, ANALYZING])
def test_selection_locked_during_busy(state: str) -> None:
    control, _seen, _transport = _controller()
    control.select_model("fixture")
    control._state = state  # Fixture drives the busy state directly.
    control.select_model("other")
    assert control._selected_id == "fixture"
