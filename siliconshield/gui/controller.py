"""GUI application state machine: no Qt dependency, fully unit-testable.

The controller owns selection, loading, analysis and teardown state. It
never touches widgets, the network, or the GPU directly: the application
wires Qt signals and timers to the ``on_*`` entry points, and a transport
object owns the worker process. Stale worker replies (wrong request or
session id) are counted and ignored so they can never overwrite a newer
run.
"""

from typing import Any, Callable, Optional, Union

from siliconshield.gui import contracts

UNLOADED = "unloaded"
LOADING = "loading"
READY = "ready"
ANALYZING = "analyzing"
UNLOADING = "unloading"
ERROR = "error"


class Controller:
    """One-model-at-a-time GUI session over an owned worker process."""

    def __init__(
        self,
        on_state: Callable[[str], None],
        on_progress: Callable[[str, str], None],
        on_result: Callable[[dict], None],
        on_error: Callable[[str, str], None],
        load_timeout_seconds: int = 180,
        analyze_timeout_seconds: int = 60,
    ) -> None:
        self._on_state = on_state
        self._on_progress = on_progress
        self._on_valid_result = on_result
        self._on_error = on_error
        self.load_timeout_seconds = load_timeout_seconds
        self.analyze_timeout_seconds = analyze_timeout_seconds
        self._transport: Any = None
        self._state = UNLOADED
        self._selected_id: Union[str, None] = None
        self._loaded_id: Union[str, None] = None
        self._session_id: Union[str, None] = None
        self._pending_request: Union[str, None] = None
        self._pending_kind: Union[str, None] = None
        self._pending_inputs_key: Union[str, None] = None
        self._stale_replies = 0
        self._last_result_key: Union[str, None] = None

    @property
    def state(self) -> str:
        """Current UI state name."""
        return self._state

    @property
    def stale_replies(self) -> int:
        """Count of ignored out-of-date worker replies."""
        return self._stale_replies

    def has_loaded_model(self) -> bool:
        """Whether a verified model is resident and ready for analysis."""
        return self._state == READY and self._loaded_id is not None

    def can_analyze(self, inputs_valid: bool) -> bool:
        """Run is enabled only when ready, matched, and inputs are valid."""
        return (
            self._state == READY
            and inputs_valid
            and self._selection_matches_loaded()
        )

    def set_callbacks(
        self,
        on_state: Callable[[str], None],
        on_progress: Callable[[str, str], None],
        on_result: Callable[[dict], None],
        on_error: Callable[[str, str], None],
    ) -> None:
        """Replace UI callbacks after construction (window wiring)."""
        self._on_state = on_state
        self._on_progress = on_progress
        self._on_valid_result = on_result
        self._on_error = on_error

    def attach(self, transport: Any) -> None:
        """Bind the worker transport; the transport calls back on events."""
        self._transport = transport
        transport.on_message = self.on_transport_message
        transport.on_finished = self.on_transport_finished
        transport.on_error = self.on_transport_error

    def select_model(self, model_id: Union[str, None]) -> None:
        """Record a dropdown selection; Run stays disabled until loaded."""
        if self._state in (LOADING, ANALYZING, UNLOADING):
            return
        self._selected_id = model_id or None
        if self._state == ERROR:
            self._set_state(UNLOADED)

    def load(self, model: dict) -> bool:
        """Start loading the selected model; returns False when refused."""
        if self._state in (LOADING, ANALYZING, UNLOADING):
            return False
        if self._selected_id is None or model.get("id") != self._selected_id:
            self._fail("Select a model before loading.", "select")
            return False
        if self._transport is None:
            self._fail("No worker transport is attached.", "load")
            return False
        self._pending_request = _new_request_id()
        self._pending_kind = "load"
        self._set_state(LOADING)
        self._transport.send_load(model, self._pending_request)
        return True

    def analyze(self, inputs: dict, input_key: str) -> bool:
        """Run one analysis of a frozen input snapshot; returns False if refused."""
        if not self.can_analyze(
            bool(inputs.get("requirement")) and bool(inputs.get("rtl"))
        ):
            return False
        if self._transport is None:
            return False
        self._pending_request = _new_request_id()
        self._pending_kind = "analyze"
        self._pending_inputs_key = input_key
        self._set_state(ANALYZING)
        self._transport.send_analyze(inputs, self._pending_request)
        return True

    def cancel(self) -> None:
        """Terminate the owned worker after user cancellation."""
        if self._state == ANALYZING:
            self._terminate_owned(
                "Analysis cancelled; the worker was stopped.", report=True
            )
        elif self._state == LOADING:
            self._terminate_owned(
                "Loading cancelled; the worker was stopped.", report=True
            )

    def unload(self) -> None:
        """Unload the model, releasing the owned worker and GPU context."""
        if self._state in (UNLOADED, UNLOADING):
            return
        self._set_state(UNLOADING)
        self._terminate_owned("Model unloaded.", report=False)

    def shutdown(self) -> None:
        """Release the worker unconditionally during application exit."""
        self._loaded_id = None
        self._session_id = None
        self._pending_request = None
        self._pending_kind = None
        self._pending_inputs_key = None
        if self._transport is not None:
            self._transport.terminate()
        if self._state != UNLOADED:
            self._set_state(UNLOADED)

    def on_load_timeout(self) -> None:
        """Report a load timeout and stop the owned worker."""
        if self._state != LOADING:
            return
        self._terminate_owned(
            "Loading timed out after %d seconds; the worker was stopped."
            % self.load_timeout_seconds,
            report=True,
        )

    def on_analyze_timeout(self) -> None:
        """Report an analysis timeout and stop the owned worker."""
        if self._state != ANALYZING:
            return
        self._terminate_owned(
            "Analysis timed out after %d seconds; the worker was stopped."
            % self.analyze_timeout_seconds,
            report=True,
        )

    def on_transport_message(self, message: dict) -> None:
        """Route one validated worker message to state transitions."""
        kind = message.get("type")
        if kind == "progress":
            if self._matches_pending(message):
                self._on_progress(
                    str(message.get("stage", "")),
                    str(message.get("detail", "")),
                )
            else:
                self._note_stale()
            return
        if kind == "ready":
            self._handle_ready(message)
            return
        if kind == "result":
            self._handle_result(message)
            return
        if kind == "error":
            detail = message.get("detail")
            stage = message.get("stage")
            if not isinstance(detail, str) or not detail:
                detail = "Worker reported a failure."
            if not isinstance(stage, str) or not stage:
                stage = "worker"
            if self._state in (LOADING, ANALYZING):
                if self._matches_pending(message):
                    self._fail(detail, stage)
                else:
                    self._note_stale()
            elif self._state not in (UNLOADED, UNLOADING):
                self._fail(detail, stage)
            return
        self._fail("Worker sent an unrecognized message.", "protocol")

    def on_transport_finished(self, _expected: bool, detail: str) -> None:
        """Handle worker exit; any exit outside unload ends readiness.

        The expected flag is informational only: an exit during
        loading, analysis, or ready always clears readiness, since a
        resident model requires a live worker.
        """
        self._pending_request = None
        self._pending_kind = None
        if self._state == UNLOADING or self._state == UNLOADED:
            self._loaded_id = None
            self._session_id = None
            self._set_state(UNLOADED)
        elif self._state == ERROR:
            self._loaded_id = None
            self._session_id = None
        else:
            was = self._state
            self._loaded_id = None
            self._session_id = None
            self._set_state(ERROR)
            self._on_error(
                "Worker exited during %s: %s" % (was, detail), "worker"
            )

    def on_transport_error(self, detail: str) -> None:
        """Handle a transport failure; readiness is always cleared."""
        self._pending_request = None
        self._pending_kind = None
        if self._state not in (UNLOADED, UNLOADING):
            self._loaded_id = None
            self._session_id = None
            self._set_state(ERROR)
            self._on_error(detail, "transport")

    def has_stale_result(self, current_key: Optional[str]) -> bool:
        """Whether the shown result belongs to older inputs."""
        return (
            self._last_result_key is not None
            and current_key != self._last_result_key
        )

    def clear_result(self) -> None:
        """Drop the stored result key after the inputs change."""
        self._last_result_key = None

    def _set_state(self, state: str) -> None:
        self._state = state
        self._on_state(state)

    def _fail(self, message: str, stage: str) -> None:
        self._loaded_id = None
        self._session_id = None
        self._pending_request = None
        self._pending_kind = None
        self._set_state(ERROR)
        self._on_error(message, stage)

    def _selection_matches_loaded(self) -> bool:
        return (
            self._selected_id is not None
            and self._selected_id == self._loaded_id
            and self._session_id is not None
        )

    def _matches_pending(self, message: dict) -> bool:
        request_id = message.get("request_id")
        session_id = message.get("session_id")
        if self._pending_request is None or request_id != self._pending_request:
            return False
        if (
            self._session_id is not None
            and session_id is not None
            and session_id != self._session_id
        ):
            return False
        return True

    def _note_stale(self) -> None:
        self._stale_replies += 1

    def _terminate_owned(self, detail: str, report: bool) -> None:
        self._pending_request = None
        self._pending_kind = None
        self._pending_inputs_key = None
        if self._transport is not None:
            try:
                self._transport.terminate()
            except RuntimeError as exc:
                detail = "%s (%s)" % (detail, exc)
        self._loaded_id = None
        self._session_id = None
        if self._state != UNLOADED:
            self._set_state(UNLOADED)
        if report:
            self._on_error(detail, "cancelled")

    def _handle_ready(self, message: dict) -> None:
        if self._state != LOADING or not self._matches_pending(message):
            self._note_stale()
            return
        session_id = message.get("session_id")
        model_id = message.get("model_id")
        if not isinstance(session_id, str) or not session_id:
            self._fail("Worker ready message is missing its session.", "load")
            return
        if model_id != self._selected_id:
            self._fail(
                "Worker reported a different model than selected.", "load"
            )
            return
        device = message.get("device")
        if not isinstance(device, str) or not (
            device == "cpu" or device.startswith("cuda:")
        ):
            self._fail(
                "Worker reported an unknown device: %r." % (device,),
                "load",
            )
            return
        fallback = message.get("cpu_fallback_count")
        if (
            isinstance(device, str)
            and device.startswith("cuda")
            and isinstance(fallback, int)
            and fallback > 0
        ):
            self._fail(
                "Worker fell back to CPU; refusing to report GPU readiness.",
                "load",
            )
            return
        self._loaded_id = self._selected_id
        self._session_id = session_id
        self._pending_request = None
        self._pending_kind = None
        self._set_state(READY)

    def _handle_result(self, message: dict) -> None:
        if self._state != ANALYZING or not self._matches_pending(message):
            self._note_stale()
            return
        try:
            checked = contracts.validate_result(message)
        except ValueError as exc:
            self._fail("Invalid worker result: %s." % exc, "analyze")
            return
        self._last_result_key = self._pending_inputs_key
        self._pending_request = None
        self._pending_kind = None
        self._pending_inputs_key = None
        self._set_state(READY)
        self._on_valid_result(checked)


_request_counter = [0]


def _new_request_id() -> str:
    _request_counter[0] += 1
    return "req-%d" % _request_counter[0]
