"""Owned, identity-checked subprocesses with bounded request lifetimes."""

import copy
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from _thread import RLock
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from siliconshield.experts import contracts
from siliconshield.experts.identity import (
    digest,
    runtime_digest,
    text_digest,
    validate_execution_identity,
)
from siliconshield.experts.registry import ModelEntry
from siliconshield.training.laya.question_contracts import (
    question_for,
    read_export_record,
)


class WorkerLoadError(RuntimeError):
    """Model loading failed; the worker was discarded, never half-ready."""


def _spawn(
    entry: ModelEntry, worker_module: str, extra_env: dict[str, str] | None
) -> subprocess.Popen[str]:
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    laya_root = entry.get("laya_root", "")
    if laya_root:
        env["LAYA_ROOT"] = laya_root
        env["PYTHONPATH"] = laya_root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [entry["python"], "-m", worker_module],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        env=env,
    )


def _pump(stream: Any, box: queue.Queue[str | None]) -> None:
    try:
        for line in stream:
            box.put(line)
    finally:
        stream.close()
        box.put(None)


class WorkerPool:
    """One persistent process per expert, serialized within each expert."""

    def __init__(
        self,
        worker_module: str = "siliconshield.app.inference_worker",
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self._workers: dict[str, subprocess.Popen[str]] = {}
        self._queues: dict[str, queue.Queue[str | None]] = {}
        self._ready: dict[str, dict[str, Any]] = {}
        self._locks: dict[str, RLock] = {}
        self._pool_lock = threading.RLock()
        self._closed = False
        self._module, self._extra_env = worker_module, extra_env

    def __contains__(self, expert_id: object) -> bool:
        """Whether a validated worker is currently ready for analysis."""
        return expert_id in self._ready

    def ready(self, expert_id: str) -> dict[str, Any]:
        """Return an immutable copy of the validated ready receipt."""
        return copy.deepcopy(self._ready[expert_id])

    def load(self, entry: ModelEntry) -> None:
        """Load and validate identity before permitting any analysis.

        Reloading an expert that fails re-validation discards the
        previous generation too: a worker that just failed an
        identity check must not keep serving. Availability yields
        to identity safety here; callers retry the load.
        """
        with self._pool_lock:
            if self._closed:
                raise RuntimeError("Worker pool is closed")
        if not isinstance(entry, Mapping):
            raise WorkerLoadError("Cannot load worker: entry is not an object")
        expert_id = entry.get("id")
        if not isinstance(expert_id, str) or not expert_id:
            raise WorkerLoadError("Cannot load worker: entry names no expert")
        for key in (
            "checkpoint",
            "device",
            "expected_files",
            "laya_root",
            "python",
        ):
            if key not in entry:
                raise WorkerLoadError(
                    "Cannot load %s: entry lacks %s" % (expert_id, key)
                )
        with self._pool_lock:
            expert_lock = self._locks.setdefault(expert_id, threading.RLock())
        with expert_lock:
            with self._pool_lock:
                if self._closed:
                    raise RuntimeError("Worker pool is closed")
                self._ready.pop(expert_id, None)
            try:
                worker = self._workers.get(expert_id)
                if worker is None or worker.poll() is not None:
                    self._discard(expert_id)
                    worker = _spawn(entry, self._module, self._extra_env)
                    if worker.stdout is None:
                        raise RuntimeError("Worker stdout was not piped")
                    box: queue.Queue[str | None] = queue.Queue()
                    threading.Thread(
                        target=_pump, args=(worker.stdout, box), daemon=True
                    ).start()
                    with self._pool_lock:
                        if self._closed:
                            self._workers[expert_id] = worker
                            self._queues[expert_id] = box
                            self._discard(expert_id)
                            raise RuntimeError("Worker pool is closed")
                        self._workers[expert_id], self._queues[expert_id] = (
                            worker,
                            box,
                        )
                request_id = uuid.uuid4().hex
                request: dict[str, Any] = dict(
                    contracts.make_load_request(request_id, expert_id)
                )
                request["model"] = dict(entry)
                self._send(expert_id, request)
                reply = self._wait(
                    expert_id,
                    request_id,
                    ("ready",),
                    contracts.LOAD_TIMEOUT_SECONDS,
                    "load",
                )
                self._validate_ready(entry, reply)
                self._ready[expert_id] = reply
            except (OSError, ValueError, RuntimeError) as exc:
                self._discard(expert_id)
                raise WorkerLoadError(
                    "Cannot load %s: %s" % (expert_id, exc)
                ) from exc

    @staticmethod
    def _validate_ready(entry: ModelEntry, reply: dict[str, Any]) -> None:
        staged = Path(entry["checkpoint"])
        record_dir = staged.parent if staged.is_file() else staged
        expected_contract = (
            entry.get("question_contract")
            or (read_export_record(record_dir)["version"])
        )
        for field, expected in (
            ("model_id", entry["id"]),
            ("device", entry["device"]),
            ("question_contract", expected_contract),
        ):
            if reply.get(field) != expected:
                raise ValueError("Ready reply differs from requested " + field)
        if (
            type(reply.get("cpu_fallback_count")) is not int
            or reply["cpu_fallback_count"] != 0
        ):
            raise ValueError("Ready reply has invalid CPU fallback evidence")
        questions = question_for(expected_contract)
        questions["security"]["option_order"] = [0, 1, 2]
        identity = validate_execution_identity(reply.get("execution_identity"))
        expected_files = entry["expected_files"]
        if (
            identity["checkpoint_sha256"]
            != expected_files.get("model.safetensors")
            or identity["export_sha256"] != digest(expected_files)
            or identity["question_contract"] != expected_contract
            or identity["question_sha256"] != digest(questions)
            or identity["runtime_sha256"] != runtime_digest(entry["laya_root"])
        ):
            raise ValueError(
                "Ready artifact/runtime identity differs from registry"
            )
        settings = identity["inference"]
        if (
            settings["batch_size"] != 1
            or settings["sort_by_length"] is not False
            or settings["device_type"] != entry["device"].split(":")[0]
        ):
            raise ValueError(
                "Ready inference protocol differs from worker request"
            )

    def analyze(self, expert_id: str, state: str) -> dict[str, Any]:
        """Return one result only from the loaded generation and exact input.

        Transport and protocol failures discard the worker and raise
        RuntimeError. Input rejections from result validation raise
        ValueError and keep the loaded worker: a truncated input says
        nothing about the worker's health.
        """
        if not isinstance(state, str) or not state.strip():
            raise ValueError("Cannot analyze empty state text")
        with self._locks.setdefault(expert_id, threading.RLock()):
            if expert_id not in self._ready:
                raise RuntimeError("No validated worker ready for " + expert_id)
            try:
                request_id = uuid.uuid4().hex
                self._send(
                    expert_id, contracts.make_analyze_request(request_id, state)
                )
                try:
                    reply = self._wait(
                        expert_id,
                        request_id,
                        ("result",),
                        contracts.ANALYZE_TIMEOUT_SECONDS,
                        "analyze",
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        "Worker for %s broke protocol: %s" % (expert_id, exc)
                    ) from exc
                ready = self._ready[expert_id]
                for field in ("model_id", "device", "execution_identity"):
                    if reply.get(field) != ready[field]:
                        raise RuntimeError("Result changed loaded " + field)
                if reply.get("state_sha256") != text_digest(state):
                    raise RuntimeError("Result belongs to a different state")
            except RuntimeError as exc:
                self._discard(expert_id)
                raise RuntimeError(
                    "Expert %s analysis failed: %s" % (expert_id, exc)
                ) from exc
            contracts.validate_result(reply)
            return reply

    def transport(self) -> Any:
        """Return the checked analysis callable."""
        return self.analyze

    def _discard(self, expert_id: str) -> None:
        """Invalidate and reap only this pool's owned child."""
        with self._pool_lock:
            self._ready.pop(expert_id, None)
            self._queues.pop(expert_id, None)
            worker = self._workers.pop(expert_id, None)
        if worker is None:
            return
        try:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    try:
                        worker.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            if worker.stdin is not None:
                try:
                    worker.stdin.close()
                except OSError:
                    pass

    def close(self) -> None:
        """Reap all owned children, including failed or timed-out generations.

        One stuck child never blocks the rest: every owned
        generation is discarded, and the first failure surfaces
        only after all siblings were attempted. Close is terminal:
        later loads are refused, and sweeps repeat until no owned
        child remains, so a concurrent spawn cannot leak.
        """
        with self._pool_lock:
            self._closed = True
        first: BaseException | None = None
        while True:
            with self._pool_lock:
                pending = list(self._workers)
            if not pending:
                break
            for expert_id in pending:
                try:
                    with self._locks.setdefault(expert_id, threading.RLock()):
                        self._discard(expert_id)
                except BaseException as exc:  # noqa: BLE001 - reap all first
                    if first is None:
                        first = exc
        if first is not None:
            raise first

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _send(self, expert_id: str, request: dict[str, Any]) -> None:
        worker = self._workers[expert_id]
        if worker.poll() is not None or worker.stdin is None:
            raise RuntimeError("Worker for %s exited early" % expert_id)
        try:
            worker.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            worker.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(
                "Cannot reach worker for %s: %s" % (expert_id, exc)
            ) from exc

    def _wait(
        self,
        expert_id: str,
        request_id: str,
        wanted: tuple[str, ...],
        timeout: float,
        stage: str,
    ) -> dict[str, Any]:
        box = self._queues[expert_id]
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "Worker for %s timed out during %s" % (expert_id, stage)
                )
            try:
                line = box.get(timeout=remaining)
            except queue.Empty as exc:
                raise RuntimeError(
                    "Worker for %s timed out during %s" % (expert_id, stage)
                ) from exc
            if line is None:
                raise RuntimeError(
                    "Worker for %s exited during %s" % (expert_id, stage)
                )
            try:
                message = json.loads(line)
            except ValueError as exc:
                raise RuntimeError(
                    "Worker for %s sent invalid JSON during %s: %s"
                    % (expert_id, stage, exc)
                ) from exc
            if not isinstance(message, dict):
                raise RuntimeError(
                    "Worker for %s sent a non-object message during %s"
                    % (expert_id, stage)
                )
            if message.get("request_id") != request_id:
                continue
            session = message.get("session_id")
            if (
                message.get("protocol") != contracts.PROTOCOL_VERSION
                or not isinstance(session, str)
                or not session
            ):
                raise RuntimeError("Invalid worker protocol/session")
            if (
                expert_id in self._ready
                and session != self._ready[expert_id]["session_id"]
            ):
                raise RuntimeError("Worker session changed during request")
            kind = contracts.message_kind(message)
            if kind == "progress":
                continue
            if kind == "error":
                raise RuntimeError(
                    "Worker for %s failed during %s: %s"
                    % (expert_id, stage, message.get("detail", "unknown"))
                )
            if kind in wanted:
                return dict(message)
            raise RuntimeError(
                "Worker for %s sent unexpected %s during %s"
                % (expert_id, kind, stage)
            )
