"""QProcess wrapper for the owned LAYA inference worker.

The GUI thread never blocks on model I/O: stdout is consumed through the
``readyReadStandardOutput`` signal and parsed incrementally. Termination is
asynchronous by default (close stdin, terminate, scheduled kill fallback);
the bounded blocking ``wait_for_exit`` exists for tests and application
shutdown only. Protocol violations fail closed: malformed output produces
an error and schedules termination, never a fabricated result.
"""

import json
from typing import Any, Callable, Optional, Union

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer

OnMessage = Callable[[dict], None]
OnFinished = Callable[[int, bool], None]
OnError = Callable[[str], None]


class WorkerClient(QObject):
    """Owns one worker subprocess and turns its stdout into messages."""

    def __init__(
        self,
        on_message: OnMessage,
        on_finished: OnFinished,
        on_error: OnError,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._on_message = on_message
        self._on_finished = on_finished
        self._on_error = on_error
        self._process: Union[QProcess, None] = None
        self._buffer = b""
        self._terminating = False
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill)

    @property
    def running(self) -> bool:
        """Whether a worker process is currently owned."""
        return (
            self._process is not None
            and self._process.state() != QProcess.NotRunning
        )

    @property
    def pid(self) -> Union[int, None]:
        """PID of the owned worker, or None when no worker runs."""
        if self._process is None:
            return None
        pid = self._process.processId()
        return int(pid) if pid else None

    def start(
        self,
        program: str,
        args: list[str],
        env: Union[dict[str, str], None] = None,
        cwd: Union[str, None] = None,
    ) -> None:
        """Spawn the worker; failures arrive through the callbacks.

        Raises RuntimeError when a worker is already owned; callers
        must terminate the previous worker first to avoid GPU leaks.
        """
        if self.running:
            raise RuntimeError("Cannot start: worker is already running")
        process = QProcess(self)
        process.readyReadStandardOutput.connect(self._on_ready_read)
        process.finished.connect(self._handle_finished)
        process.errorOccurred.connect(self._handle_process_error)
        if env is not None:
            process.setProcessEnvironment(_process_environment(env))
        if cwd is not None:
            process.setWorkingDirectory(cwd)
        self._process = process
        self._buffer = b""
        process.start(program, args)

    @property
    def on_message(self) -> OnMessage:
        """Current message callback (settable for controller wiring)."""
        return self._on_message

    @on_message.setter
    def on_message(self, callback: OnMessage) -> None:
        self._on_message = callback

    @property
    def on_finished(self) -> OnFinished:
        """Current finished callback (settable for controller wiring)."""
        return self._on_finished

    @on_finished.setter
    def on_finished(self, callback: OnFinished) -> None:
        self._on_finished = callback

    @property
    def on_error(self) -> OnError:
        """Current error callback (settable for controller wiring)."""
        return self._on_error

    @on_error.setter
    def on_error(self, callback: OnError) -> None:
        self._on_error = callback

    def send(self, message: dict[str, Any]) -> None:
        """Write one JSON line to the worker's stdin."""
        if (
            self._process is None
            or self._process.state() == QProcess.NotRunning
        ):
            raise RuntimeError("Cannot send: worker is not running")
        line = (json.dumps(message, allow_nan=False) + "\n").encode("utf-8")
        self._process.write(line)

    def send_load(self, model: dict[str, Any], request_id: str) -> None:
        """Send a load command for one validated registry entry."""
        from siliconshield.gui import contracts

        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Model entry is missing its id")
        request = contracts.make_load_request(request_id, model_id)
        request["model"] = dict(model)
        self.send(request)

    def send_analyze(self, inputs: dict[str, Any], request_id: str) -> None:
        """Send one analysis request for a frozen input snapshot."""
        from siliconshield.gui import contracts

        state = inputs.get("state")
        if not isinstance(state, str) or not state:
            raise ValueError("Analysis inputs are missing state text")
        self.send(contracts.make_analyze_request(request_id, state))

    def terminate(self, kill_after_ms: int = 5000) -> None:
        """Stop the owned worker without blocking the caller.

        Closes stdin so a well-behaved worker exits, asks Qt to
        terminate it, and schedules SIGKILL as a bounded fallback.
        Completion always arrives through the finished callback.
        """
        process = self._process
        if process is None or process.state() == QProcess.NotRunning:
            return
        self._terminating = True
        try:
            process.closeWriteChannel()
        except RuntimeError:
            pass
        process.terminate()
        self._kill_timer.start(kill_after_ms)

    def wait_for_exit(self, timeout_ms: int = 10000) -> bool:
        """Block (bounded) until the owned worker exits; for tests and shutdown."""
        process = self._process
        if process is None or process.state() == QProcess.NotRunning:
            return True
        finished = process.waitForFinished(timeout_ms)
        if not finished:
            process.kill()
            finished = process.waitForFinished(timeout_ms)
        return bool(finished)

    def _on_ready_read(self) -> None:
        process = self._process
        if process is None:
            return
        self._buffer += bytes(process.readAllStandardOutput().data())
        *complete, self._buffer = self._buffer.split(b"\n")
        for raw in complete:
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except ValueError as exc:
                self._protocol_failed("worker wrote non-JSON output: %s" % exc)
                return
            if not isinstance(message, dict) or not isinstance(
                message.get("type"), str
            ):
                self._protocol_failed("worker message has no type")
                return
            self._on_message(message)

    def _protocol_failed(self, detail: str) -> None:
        self._on_error(detail)
        self.terminate()

    def _handle_finished(self, exit_code: int, _status: object) -> None:
        self._kill_timer.stop()
        self._buffer = b""
        self._process = None
        self._terminating = False
        self._on_finished(exit_code, exit_code != 0)

    def _handle_process_error(self, _error: object) -> None:
        if self._terminating:
            # Our own terminate() already explains the shutdown through the
            # finished callback; a follow-up "crashed" report would overwrite
            # the real error (e.g. a hash mismatch) with a scary message.
            return
        process = self._process
        detail = "worker process failed to start or crashed"
        if process is not None:
            detail = "worker process error: %s" % process.errorString()
        self._on_error(detail)

    def _force_kill(self) -> None:
        process = self._process
        if process is not None and process.state() != QProcess.NotRunning:
            process.kill()


def _process_environment(env: dict[str, str]) -> QProcessEnvironment:
    merged = QProcessEnvironment.systemEnvironment()
    for key, value in env.items():
        merged.insert(key, value)
    return merged
