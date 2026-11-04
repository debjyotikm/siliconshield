"""Browser-facing orchestration over Glimmer specialists and their skills."""

import copy
import hashlib
import json
import threading
import time
import traceback
import urllib.request
import uuid
from pathlib import Path
from typing import Any, cast

from siliconshield.experts.arbitration import arbitrate_assessment
from siliconshield.experts.identity import assemble_state
from siliconshield.experts.service import InferenceService
from siliconshield.research_loop.controller.client import ChatClient
from siliconshield.research_loop.episodes.contracts import atomic_json
from siliconshield.research_loop.procedures.adapter_specialists import (
    AdapterTransport,
)
from siliconshield.research_loop.procedures.glimmer_specialists import (
    GlimmerTransport,
)
from siliconshield.web_demo.glimmer_client import LocalGlimmerClient

LABELS = [
    "satisfies_requirement",
    "violates_requirement",
    "insufficient_context",
]

ROUTING_SCOPES = ("fifo", "counter", "arbitration", "ot-access-control")


def read_json(path: Path, fallback: Any = None) -> Any:
    """Read an optional JSON document which may not exist yet."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return fallback


def state_text(body: dict[str, Any]) -> str:
    """Preserve supplied RTL and refuse oversized inputs before inference.

    The requirement and assumptions must not smuggle an exact
    section-header line: the judged sections are split on the
    first RTL header, so a smuggled header would truncate the
    judged requirement and pollute the judged RTL.
    """
    if not isinstance(body, dict):
        raise ValueError("Demo request must be an object.")
    fields = {}
    for field, limit in [
        ("rtl", 50000),
        ("requirement", 3000),
        ("assumptions", 3000),
    ]:
        value = body.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(field + " is required.")
        if len(value) > limit:
            raise ValueError(
                field
                + " exceeds the demo input limit. Submit a self-contained region."
            )
        fields[field] = value.strip()
    return assemble_state(
        fields["requirement"], fields["assumptions"], fields["rtl"]
    )


class _RecordingChatClient:
    """Record live Glimmer calls for demo evidence.

    Wraps one chat client and keeps every completion request and
    response: endpoint, model, per-message sizes, the skill digest
    actually sent, timing, and the exact response text. Failed
    calls are recorded with their error instead of vanishing. An
    optional progress callback streams call lifecycle lines while
    each call runs, so observers see activity during long calls.
    """

    def __init__(
        self,
        client: Any,
        endpoint: str,
        model: str,
        progress: Any = None,
        heartbeat_seconds: float = 10.0,
    ) -> None:
        self._client = client
        self._endpoint = endpoint
        self._model = model
        self._progress = progress
        self._heartbeat_seconds = heartbeat_seconds
        self.calls: list[dict[str, Any]] = []

    def _say(self, text: str) -> None:
        if self._progress is not None:
            self._progress(text)

    def complete(self, messages: Any, tools: Any = ()) -> Any:
        started = time.monotonic()
        number = len(self.calls) + 1
        skill = ""
        for message in messages:
            if message.get("role") == "system":
                skill = message.get("content", "")
        digest = hashlib.sha256(skill.encode("utf-8")).hexdigest()[:12]
        self._say(
            "Call %d: POST %s model %s skill %s. Waiting on Glimmer."
            % (number, self._endpoint, self._model, digest)
        )
        done = threading.Event()

        def heartbeat() -> None:
            while not done.wait(self._heartbeat_seconds):
                self._say(
                    "Call %d: still waiting (%ds elapsed)."
                    % (number, time.monotonic() - started)
                )

        pulse = threading.Thread(target=heartbeat, daemon=True)
        pulse.start()
        try:
            reply = self._client.complete(messages, tools)
        except Exception as exc:
            done.set()
            seconds = time.monotonic() - started
            self.calls.append(
                {
                    "endpoint": self._endpoint,
                    "model": self._model,
                    "seconds": seconds,
                    "error": str(exc),
                }
            )
            self._say("Call %d failed after %.1fs: %s" % (number, seconds, exc))
            raise
        finally:
            done.set()
        seconds = time.monotonic() - started
        content = reply.get("content", "") if isinstance(reply, dict) else ""
        request_messages = [
            {"role": message.get("role"), "chars": len(text)}
            for message in messages
            for text in [message.get("content", "")]
        ]
        self.calls.append(
            {
                "endpoint": self._endpoint,
                "model": self._model,
                "seconds": seconds,
                "kind": "chat",
                "request_messages": request_messages,
                "skill_sha256": hashlib.sha256(
                    skill.encode("utf-8")
                ).hexdigest(),
                "response_chars": len(content),
                "response": content,
            }
        )
        self._say(
            "Call %d finished in %.1fs (%d response chars)."
            % (number, seconds, len(content))
        )
        return reply


def _call_with_heartbeat(
    log: Any, start_text: str, wait_text: str, call: Any
) -> Any:
    """Run one blocking agent call with elapsed-time progress updates."""
    started = time.monotonic()
    log(start_text)
    done = threading.Event()

    def heartbeat() -> None:
        while not done.wait(10.0):
            log(wait_text % (time.monotonic() - started))

    pulse = threading.Thread(target=heartbeat, daemon=True)
    pulse.start()
    try:
        return call()
    finally:
        done.set()


class Demo:
    """Consult Glimmer specialists and retain actual execution receipts."""

    def __init__(
        self, config: dict[str, Any], *, health_poll: bool = True
    ) -> None:
        if not isinstance(config, dict):
            raise ValueError("Demo configuration must be an object.")
        if (
            not isinstance(config.get("state_dir"), str)
            or not config["state_dir"]
        ):
            raise ValueError("Demo configuration needs a state directory.")
        entries = config.get("entries")
        if not isinstance(entries, list) or any(
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), str)
            or not entry["id"]
            for entry in entries
        ):
            raise ValueError("Demo configuration needs entries with ids.")
        self.config = config
        self.state = Path(config["state_dir"])
        self.state.mkdir(parents=True, exist_ok=True, mode=448)
        self.entries = {entry["id"]: entry for entry in entries}
        self.jobs: dict[str, dict[str, Any]] = {}
        self._evicted: list[str] = []
        self._threads: list[threading.Thread] = []
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.health: dict[str, Any] = {
            "glimmer": {"status": "not configured"},
        }
        self.stopping = False
        if health_poll:
            threading.Thread(target=self.health_loop, daemon=True).start()

    def glimmer_health(self) -> dict[str, Any]:
        """Check an explicitly configured local service without generation."""
        cfg = self.config.get("glimmer")
        if not cfg:
            return {"status": "not configured"}
        if cfg["endpoint"].endswith("/score"):
            return self._shim_health(cfg)
        if not isinstance(cfg, dict) or not isinstance(cfg.get("model"), str):
            raise ValueError("Glimmer configuration needs a model id.")
        client = LocalGlimmerClient(
            cfg["endpoint"],
            cfg["model"],
            ssh_host=cfg.get("ssh_host"),
            timeout=8,
        )
        reply = client.models()
        if not isinstance(reply, dict) or not isinstance(
            reply.get("data"), list
        ):
            raise RuntimeError("Glimmer service returned a malformed reply.")
        served = [
            model["id"]
            for model in reply["data"]
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        ]
        if cfg["model"] not in served:
            raise RuntimeError("Configured Glimmer model is not served.")
        return {
            "status": "ready",
            "model": cfg["model"],
            "transport": "SSH" if cfg.get("ssh_host") else "Loopback HTTP",
        }

    def _shim_health(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Check the scoring shim through its model list."""
        if not isinstance(cfg.get("model"), str):
            raise ValueError("Adapter configuration needs a model id.")
        url = cfg["endpoint"][: -len("/score")] + "/v1/models"
        with urllib.request.urlopen(url, timeout=8) as reply:
            data = json.loads(reply.read())
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise RuntimeError("Scoring shim returned a malformed reply.")
        served = [
            model["id"]
            for model in data["data"]
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        ]
        if cfg["model"] not in served:
            raise RuntimeError("Configured adapter model is not served.")
        return {
            "status": "ready",
            "model": cfg["model"],
            "transport": "Loopback scoring",
        }

    def health_loop(self) -> None:
        """Poll the Glimmer service without dispatching inference."""
        while not self.stopping:
            try:
                self.health["glimmer"] = self.glimmer_health()
            except Exception as exc:
                self.health["glimmer"] = {
                    "status": "unavailable",
                    "error": str(exc),
                }
            for _ in range(30):
                if self.stopping:
                    return
                time.sleep(1)

    def was_evicted(self, job_id: str) -> bool:
        """Whether a job id names an expired, evicted receipt."""
        with self.lock:
            return job_id in self._evicted

    def close(self) -> None:
        """Drain owned jobs with a bounded wait, then stop polling."""
        self.stopping = True
        with self.lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout=30)

    def submit_task(self, kind: str, fn: Any) -> dict[str, Any]:
        """Run one bounded background request and retain its real result."""
        job_id = uuid.uuid4().hex
        with self.lock:
            if sum((j["status"] == "running" for j in self.jobs.values())) >= 8:
                raise ValueError(
                    "Too many active requests. Wait for the current work."
                )
            self.jobs[job_id] = {
                "id": job_id,
                "kind": kind,
                "status": "running",
                "started": time.time(),
                "messages": [],
            }

        def log(message: str) -> None:
            with self.lock:
                self.jobs[job_id]["messages"].append(
                    {"time": time.time(), "text": message}
                )

        def worker() -> None:
            try:
                value = fn(log)
                with self.lock:
                    self.jobs[job_id].update(
                        status="complete", result=value, finished=time.time()
                    )
            except Exception as exc:
                traceback.print_exc()
                with self.lock:
                    self.jobs[job_id].update(
                        status="error", error=str(exc), finished=time.time()
                    )
            with self.lock:
                snapshot = dict(self.jobs[job_id])
            atomic_json(self.state / f"request-{job_id}.json", snapshot)
            self._evict_jobs()

        thread = threading.Thread(target=worker, daemon=True)
        with self.lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            self._threads.append(thread)
        thread.start()
        return {"job_id": job_id}

    def _evict_jobs(self, keep: int = 8) -> None:
        """Retain only recent terminal jobs in memory and on disk."""
        with self.lock:
            terminal = sorted(
                (
                    job_id
                    for job_id, job in self.jobs.items()
                    if job.get("status") in ("complete", "error")
                    and job.get("finished") is not None
                ),
                key=lambda job_id: self.jobs[job_id]["finished"],
            )
            evict = terminal[:-keep] if len(terminal) > keep else []
            for job_id in evict:
                del self.jobs[job_id]
            # Tombstones distinguish an expired receipt from a
            # forged id for clients still polling evicted jobs.
            self._evicted.extend(evict)
            del self._evicted[:-32]
        for job_id in evict:
            try:
                (self.state / f"request-{job_id}.json").unlink(missing_ok=True)
            except OSError:
                pass

    def _engine_transport(
        self, service: InferenceService, log: Any = None
    ) -> tuple[Any, Any]:
        """Bind the configured engine to registry specialists.

        Selects the chat or scoring transport from the registry
        kind. Either way the serving fingerprint must match the
        registry and profile binding exactly. Returns the transport
        with its live-call recorder.
        """
        cfg = self.config.get("glimmer")
        if not cfg:
            raise ValueError("Configure the Glimmer endpoint before assessing.")
        entries = {
            model_id: service.model_entry(model_id)
            for model_id in service.expert_ids
        }
        if not entries:
            raise ValueError("No usable specialists in the registry.")
        for model_id, entry in entries.items():
            # Non-meta local decision models are not supported. Use at own risk.
            if entry.get("kind", "local-worker") == "local-worker":
                raise ValueError(
                    "Refusing to route %r: local-worker entries are not "
                    "served by this demo." % (model_id,)
                )
        kinds = {entry.get("kind") for entry in entries.values()}
        if kinds == {"adapter"}:
            transport = AdapterTransport(
                cfg["endpoint"],
                cfg["model"],
                cfg["fingerprint"],
                entries,
                progress=log,
            )
            return transport, transport
        if kinds != {"glimmer"}:
            raise ValueError(
                "One demo serves one engine: register glimmer or adapter "
                "specialists, not both."
            )
        recorder = _RecordingChatClient(
            ChatClient(
                cfg["endpoint"],
                cfg["model"],
                ssh_host=cfg.get("ssh_host"),
            ),
            cfg["endpoint"],
            cfg["model"],
            progress=log,
        )
        ssh_host = cfg.get("ssh_host")
        return (
            GlimmerTransport(
                recorder,
                cfg["model"],
                cfg["endpoint"],
                cfg["fingerprint"],
                entries,
                device_type=(
                    "ssh:%s" % ssh_host.strip()
                    if ssh_host and ssh_host.strip()
                    else "remote"
                ),
            ),
            recorder,
        )

    def assess(self, body: dict[str, Any], log: Any) -> dict[str, Any]:
        """Consult one named Glimmer specialist behind its skill."""
        text = state_text(body)
        model = body.get("model")
        if not isinstance(model, str):
            raise ValueError("A model ID is required.")
        if not self.config.get("registry"):
            raise ValueError("Register specialists before assessing.")
        service = InferenceService(self.config["registry"])
        with self.inference_lock:
            transport, recorder = self._engine_transport(service, log)
            log("Consulting the Contract skill through Muse Glimmer.")
            started = time.monotonic()
            assessment = service.assess_expert(model, text, transport)
            if not assessment["responses"]:
                raise RuntimeError(
                    "Specialist %r returned no verdict: %s"
                    % (model, assessment["errors"])
                )
            response = cast(dict[str, Any], assessment["responses"][0])
            result = {
                "model": response["expert_id"],
                "label": response["label"],
                "scores": response["scores"],
                "truncated": response["truncated"],
                "wall_seconds": time.monotonic() - started,
                "glimmer_model": self.config["glimmer"]["model"],
                "glimmer_calls": recorder.calls,
                "input": {
                    k: body[k] for k in ["rtl", "requirement", "assumptions"]
                },
            }
            log("Specialist verdict received with skill evidence.")
            return result

    def route(self, body: dict[str, Any], log: Any) -> dict[str, Any]:
        """Use the capability router and optionally interpret its evidence."""
        text = state_text(body)
        scope = body.get("scope")
        if not isinstance(scope, str) or scope not in ROUTING_SCOPES:
            raise ValueError("Select a measured routing domain.")
        if not self.config.get("registry"):
            raise ValueError("Register specialists before routing.")
        service = InferenceService(self.config["registry"])
        with self.inference_lock:
            transport, recorder = self._engine_transport(service, log)
            log("Routing by measured domain coverage: " + scope + ".")
            assessment = service.assess([scope], text, transport)
        result: dict[str, Any] = {
            "assessment": assessment,
            "broken_experts": service.broken_experts,
            "unprofiled_experts": service.unprofiled_experts,
            "glimmer_calls": recorder.calls,
            "arbitration_requested": body.get("arbitrate") is True,
            "input": {
                k: body[k] for k in ["rtl", "requirement", "assumptions"]
            },
        }
        if body.get("arbitrate") is True:
            agent = self.config.get("arbiter") or self.config.get("glimmer")
            if agent is None:
                result["arbitration_error"] = (
                    "Configure an arbiter: no generative service is "
                    "available. Specialist evidence remains available."
                )
                log("Arbitration refused: no generative arbiter configured.")
                return result
            if agent.get("engine") == "spark":

                def run_spark() -> Any:
                    return arbitrate_assessment(
                        assessment,
                        text,
                        body["requirement"],
                        profiles=service.profiles_for(
                            assessment["routing"]["experts"]
                        ),
                        assumptions=body["assumptions"],
                        audit_dir=self.state / "arbitration",
                        model=agent["model"],
                    )

                try:
                    result["arbitration"] = _call_with_heartbeat(
                        log,
                        "Calling %s via the local muse CLI "
                        "for final assessment." % agent["model"],
                        "Muse Spark final assessment: "
                        "still waiting (%ds elapsed).",
                        run_spark,
                    )
                    result["arbitration_model"] = agent["model"]
                    log("Muse Spark returned a validated arbitration response.")
                except Exception as exc:
                    result["arbitration_error"] = str(exc)
                    log(
                        "Arbitration failed. Specialist evidence remains "
                        "available, without an agent verdict."
                    )
                return result
            if agent["endpoint"].endswith("/score"):
                result["arbitration_error"] = (
                    "Configure an arbiter: the scoring adapter cannot "
                    "arbitrate. Specialist evidence remains available."
                )
                log("Arbitration refused: no generative arbiter configured.")
                return result

            def run_glimmer() -> Any:
                return arbitrate_assessment(
                    assessment,
                    text,
                    body["requirement"],
                    profiles=service.profiles_for(
                        assessment["routing"]["experts"]
                    ),
                    assumptions=body["assumptions"],
                    client_factory=lambda: LocalGlimmerClient(
                        agent["endpoint"],
                        agent["model"],
                        ssh_host=agent.get("ssh_host"),
                        max_tokens=1536,
                        timeout=180,
                        max_calls=1,
                        audit_dir=self.state / "arbitration",
                        decode_arguments=True,
                    ),
                )

            try:
                result["arbitration"] = _call_with_heartbeat(
                    log,
                    "Local Muse Glimmer is reviewing the identity-bound "
                    "specialist evidence.",
                    "Muse Glimmer arbitration: still waiting (%ds elapsed).",
                    run_glimmer,
                )
                result["arbitration_model"] = agent["model"]
                log("Muse Glimmer returned a validated arbitration response.")
            except Exception as exc:
                result["arbitration_error"] = str(exc)
                log(
                    "Arbitration failed. Specialist evidence remains available, without an agent verdict."
                )
        result["input"] = {
            k: body[k] for k in ["rtl", "requirement", "assumptions"]
        }
        return result

    def status(self) -> dict[str, Any]:
        """Report registered specialists, profiles, and Glimmer health."""
        service = None
        registry_error = None
        if self.config.get("registry"):
            try:
                service = InferenceService(self.config["registry"])
            except (ValueError, OSError) as exc:
                # Health polling must survive a transient registry
                # problem; readiness comes from the registry itself.
                registry_error = "%s: %s" % (
                    type(exc).__name__,
                    exc,
                )
        return {
            "models": [
                {
                    "id": key,
                    "display_name": entry.get("display_name", key),
                    "kind": entry.get("kind", "local-worker"),
                    "skill_sha256": entry.get("expected_files", {}).get(
                        "SKILL.md"
                    ),
                    "profile": service.profiles_for([key]).get(key)
                    if service
                    else None,
                }
                for key, entry in self.entries.items()
            ],
            "registry_error": registry_error,
            "health": copy.deepcopy(self.health),
            "glimmer_configured": bool(self.config.get("glimmer")),
            "arbiter_configured": bool(self.config.get("arbiter")),
        }
