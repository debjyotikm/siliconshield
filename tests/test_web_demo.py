"""Browser transport and Glimmer specialist boundaries on synthetic inputs."""

import copy
import hashlib
import http.client
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from siliconshield.experts.service import InferenceService
from siliconshield.web_demo import backend, config
from siliconshield.web_demo.glimmer_client import LocalGlimmerClient
from siliconshield.web_demo.server import DemoServer

BODY = {
    "model": "contract-1",
    "device": "cpu",
    "rtl": "module t; wire q; endmodule",
    "requirement": "Keep q low.",
    "assumptions": "Two-state logic.",
    "scope": "fifo",
}

FINGERPRINT = "ab" * 32

GLIMMER = {
    "endpoint": "http://127.0.0.1:18400/v1/chat/completions",
    "model": "synthetic-glimmer",
    "fingerprint": FINGERPRINT,
}

VERDICT_TEXT = """Synthetic specialist reasoning.

VERDICT: satisfies_requirement
CONFIDENCE: 0.9
EVIDENCE: synthetic guard present
OPEN QUESTIONS: none
"""


def settings(tmp_path: Path, **values: Any) -> dict[str, Any]:
    path = tmp_path / "web.local.json"
    path.write_text(
        json.dumps({"state_dir": str(tmp_path / "state"), **values})
    )
    return config.load_config(path)


@pytest.fixture
def demo(tmp_path):
    app = backend.Demo(settings(tmp_path), health_poll=False)
    yield app
    app.close()


@pytest.fixture
def http_api(demo):
    server = DemoServer(("127.0.0.1", 0), demo)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, body=None, *, headers=None):
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        base = {
            "Content-Type": "application/json",
            "X-Demo-Token": server.token,
            "Origin": f"http://127.0.0.1:{server.server_port}",
        }
        base.update(headers or {})
        conn.request(
            "GET" if body is None else "POST",
            path,
            body=None if body is None else json.dumps(body),
            headers=base,
        )
        response = conn.getresponse()
        data = response.read()
        result = (
            json.loads(data)
            if response.getheader("Content-Type") == "application/json"
            else data.decode()
        )
        code = response.status
        conn.close()
        return code, result

    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


class StubChat:
    """Canned Glimmer completions; records skill and state per call."""

    text = VERDICT_TEXT

    def __init__(self, endpoint, model, ssh_host=None):
        self.endpoint = endpoint
        self.model = model
        self.ssh_host = ssh_host
        self.calls = []

    def complete(self, messages, tools=()):
        self.calls.append(messages)
        return {"content": StubChat.text}


def _skill_dir(tmp_path, name):
    skill = tmp_path / name
    skill.mkdir(exist_ok=True)
    (skill / "SKILL.md").write_text("Synthetic %s skill." % name)
    return skill


def _glimmer_profile(model_id, mechanisms):
    from siliconshield.schemas import EXPERT_CAPABILITY_SCHEMA

    return {
        "schema": EXPERT_CAPABILITY_SCHEMA,
        "id": model_id,
        "release": "glimmer-specialist-v1",
        "checkpoint_sha256": FINGERPRINT,
        "question_contract": "glimmer-specialist-v1",
        "strengths": ["synthetic strength"],
        "weaknesses": ["synthetic weakness"],
        "supported_mechanisms": mechanisms,
        "known_failures": ["synthetic failure"],
        "qualification_ref": "synthetic probe",
        "measurements": [{"eval": "synthetic", "correct": 2, "total": 3}],
    }


def _glimmer_registry(tmp_path):
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    models = []
    for model_id, mechanisms in (
        ("contract-1", ["ot-access-control"]),
        ("contract-2", ["fifo", "counter", "arbitration"]),
    ):
        skill = _skill_dir(tmp_path, model_id)
        digest = hashlib.sha256((skill / "SKILL.md").read_bytes()).hexdigest()
        profile_path = tmp_path / ("%s.profile.json" % model_id)
        profile_path.write_text(
            json.dumps(_glimmer_profile(model_id, mechanisms))
        )
        models.append(
            {
                "id": model_id,
                "kind": "glimmer",
                "display_name": model_id + " specialist",
                "checkpoint": str(skill),
                "expected_files": {
                    "model.safetensors": FINGERPRINT,
                    "SKILL.md": digest,
                },
                "device": "synthetic-spark",
                "profile": str(profile_path),
                "release": "glimmer-specialist-v1",
                "question_contract": "glimmer-specialist-v1",
                "qualification": "synthetic probe",
            }
        )
    registry_path = tmp_path / "models.json"
    registry_path.write_text(
        json.dumps({"schema": GUI_MODEL_REGISTRY_SCHEMA, "models": models})
    )
    return str(registry_path)


def add_specialists(demo, tmp_path, monkeypatch):
    registry = _glimmer_registry(tmp_path)
    new = settings(tmp_path, registry=registry, glimmer=dict(GLIMMER))
    demo.config = new
    demo.entries = {entry["id"]: entry for entry in new["entries"]}
    StubChat.text = VERDICT_TEXT
    monkeypatch.setattr(backend, "ChatClient", StubChat)


def wait_job(http_api, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        _, job = http_api("/api/jobs/" + job_id)
        if job["status"] != "running":
            return job
        time.sleep(0.01)
    pytest.fail("Fixture request did not terminate")


def test_import_and_empty_configuration_needs_no_specialists(tmp_path):
    cfg = settings(tmp_path)
    assert cfg["entries"] == [] and cfg["glimmer"] is None
    assert "gpu" not in cfg and "training_enabled" not in cfg


@pytest.mark.parametrize(
    "values",
    [
        {"gpu": 0},
        {"gpu": True},
        {"port": 0},
        {"training": {"enabled": False}},
        {"source": "/unexpected"},
        {"glimmer": {}},
        {
            "glimmer": {
                "endpoint": "http://127.0.0.1:1/chat/completions",
                "model": "synthetic",
            }
        },
        {
            "glimmer": {
                "endpoint": "http://127.0.0.1:1/chat/completions",
                "model": "synthetic",
                "fingerprint": "not-hex",
            }
        },
    ],
)
def test_configuration_refuses_malformed_authority(tmp_path, values):
    with pytest.raises(ValueError):
        settings(tmp_path, **values)


def test_relative_paths_are_relative_to_config_not_working_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir("/")
    cfg = settings(tmp_path, state_dir="session")
    assert cfg["state_dir"] == str(tmp_path / "session")


def test_http_bootstrap_and_assets_work_without_private_configuration(
    http_api,
):
    status, data = http_api("/api/bootstrap")
    assert status == 200 and data["title"] == "SiliconShield"
    assert data["glimmer_configured"] is False
    assert len(data["examples"]) == 7
    assert data["token"]
    for name in ("/", "/app.js", "/style.css", "/favicon.svg"):
        code, text = http_api(name)
        assert code == 200 and text
    assert http_api("/api/status")[1]["models"] == []


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "foreign.invalid"},
        {"Origin": "https://foreign.invalid"},
        {"X-Demo-Token": ""},
    ],
)
def test_http_rejects_foreign_mutations_before_dispatch(
    http_api, demo, monkeypatch, headers
):
    monkeypatch.setattr(
        demo, "submit_task", lambda *args: pytest.fail("Forbidden dispatch")
    )
    code, _ = http_api("/api/assess", dict(BODY), headers=headers)
    assert code == 403


def test_removed_worker_and_training_routes_are_unknown(http_api):
    for path, body in [
        ("/api/models/load", {"model": "contract-1"}),
        ("/api/models/unload", {"model": "contract-1"}),
        ("/api/training/start", {"confirm": True}),
        ("/api/training/stop", {}),
    ]:
        code, result = http_api(path, body)
        assert code == 400
        assert result["error"] == "Unknown action."
    assert http_api("/api/training")[0] == 404


def test_assess_consults_glimmer_specialist_and_preserves_input(
    http_api, demo, tmp_path, monkeypatch
):
    add_specialists(demo, tmp_path, monkeypatch)
    seen = []
    real_init = StubChat.__init__

    def capture(self, endpoint, model, ssh_host=None):
        real_init(self, endpoint, model, ssh_host=ssh_host)
        seen.append(self)

    monkeypatch.setattr(StubChat, "__init__", capture)
    code, submitted = http_api("/api/assess", BODY)
    assert code == 202
    job = wait_job(http_api, submitted["job_id"])
    assert job["status"] == "complete"
    result = job["result"]
    assert result["label"] == "satisfies_requirement"
    assert set(result["scores"]) == set(backend.LABELS)
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-6
    assert result["scores"]["satisfies_requirement"] == pytest.approx(0.9)
    assert result["input"]["rtl"] == BODY["rtl"]
    assert result["glimmer_model"] == GLIMMER["model"]
    assert result["truncated"] is False
    calls = result["glimmer_calls"]
    assert len(calls) == 1
    assert calls[0]["kind"] == "chat"
    assert calls[0]["endpoint"] == GLIMMER["endpoint"]
    assert calls[0]["model"] == GLIMMER["model"]
    assert calls[0]["seconds"] >= 0
    assert "VERDICT" in calls[0]["response"]
    assert (
        calls[0]["skill_sha256"]
        == hashlib.sha256(
            (tmp_path / "contract-1" / "SKILL.md").read_bytes()
        ).hexdigest()
    )
    assert [message["role"] for message in calls[0]["request_messages"]] == [
        "system",
        "user",
    ]
    assert len(seen) == 1 and len(seen[0].calls) == 1
    system, user = seen[0].calls[0]
    assert system["role"] == "system"
    assert "Synthetic contract-1 skill." in system["content"]
    assert user["role"] == "user" and BODY["rtl"] in user["content"]


def test_assess_without_registry_is_an_error(demo):
    with pytest.raises(ValueError, match="Register specialists"):
        demo.assess(dict(BODY), lambda text: None)


def test_assess_without_glimmer_is_an_error(demo, tmp_path):
    registry = _glimmer_registry(tmp_path)
    new = settings(tmp_path, registry=registry)
    demo.config = new
    demo.entries = {entry["id"]: entry for entry in new["entries"]}
    with pytest.raises(ValueError, match="Configure the Glimmer endpoint"):
        demo.assess(dict(BODY), lambda text: None)


def test_transport_refuses_local_worker_entries(demo):
    demo.config["glimmer"] = dict(GLIMMER)

    class _Service:
        expert_ids = ["contract-1"]

        def model_entry(self, _model_id):
            return {"id": "contract-1"}

    with pytest.raises(ValueError, match="local-worker entries are not served"):
        demo._engine_transport(_Service())


ADAPTER_FINGERPRINT = "cd" * 32

ADAPTER_GLIMMER = {
    "endpoint": "http://127.0.0.1:18401/score",
    "model": "synthetic-adapter",
    "fingerprint": ADAPTER_FINGERPRINT,
}


def _adapter_profile(model_id):
    from siliconshield.schemas import EXPERT_CAPABILITY_SCHEMA

    return {
        "schema": EXPERT_CAPABILITY_SCHEMA,
        "id": model_id,
        "release": "synthetic-adapter",
        "checkpoint_sha256": ADAPTER_FINGERPRINT,
        "question_contract": "siliconshield-adapter-scoring-v1",
        "strengths": ["synthetic strength"],
        "weaknesses": ["synthetic weakness"],
        "supported_mechanisms": ["fifo"],
        "known_failures": ["synthetic failure"],
        "qualification_ref": "synthetic probe",
        "measurements": [{"eval": "synthetic", "correct": 2, "total": 3}],
    }


def _adapter_registry(tmp_path):
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    models = []
    for model_id in ("contract-1", "contract-2"):
        skill = tmp_path / ("adapter-" + model_id)
        skill.mkdir(exist_ok=True)
        (skill / "SKILL.md").write_text("Synthetic %s adapter doc." % model_id)
        digest = hashlib.sha256((skill / "SKILL.md").read_bytes()).hexdigest()
        profile_path = tmp_path / ("adapter-%s.profile.json" % model_id)
        profile_path.write_text(json.dumps(_adapter_profile(model_id)))
        models.append(
            {
                "id": model_id,
                "kind": "adapter",
                "display_name": model_id + " adapter",
                "checkpoint": str(skill),
                "expected_files": {
                    "adapter_model.safetensors": ADAPTER_FINGERPRINT,
                    "SKILL.md": digest,
                },
                "device": "synthetic gpus",
                "profile": str(profile_path),
            }
        )
    registry_path = tmp_path / "adapters.json"
    registry_path.write_text(
        json.dumps({"schema": GUI_MODEL_REGISTRY_SCHEMA, "models": models})
    )
    return str(registry_path)


def add_adapters(demo, tmp_path):
    registry = _adapter_registry(tmp_path)
    new = settings(tmp_path, registry=registry, glimmer=dict(ADAPTER_GLIMMER))
    demo.config = new
    demo.entries = {entry["id"]: entry for entry in new["entries"]}


def test_engine_transport_selects_the_adapter(demo, tmp_path):
    from siliconshield.research_loop.procedures.adapter_specialists import (
        AdapterTransport,
    )

    add_adapters(demo, tmp_path)
    service = InferenceService(demo.config["registry"])
    transport, recorder = demo._engine_transport(service)
    assert isinstance(transport, AdapterTransport)
    assert recorder.calls == []


def test_glimmer_engine_transport_labels_the_ssh_hop(
    demo, tmp_path, monkeypatch
):
    add_specialists(demo, tmp_path, monkeypatch)
    demo.config["glimmer"] = {**GLIMMER, "ssh_host": "spark"}
    service = InferenceService(demo.config["registry"])
    transport, _recorder = demo._engine_transport(service)
    assert transport._device_type == "ssh:spark"


def test_engine_transport_refuses_mixed_registries(demo, tmp_path):
    add_adapters(demo, tmp_path)

    class _Service:
        expert_ids = ["contract-1", "contract-2"]

        def model_entry(self, model_id):
            return {
                "id": model_id,
                "kind": "adapter" if model_id == "contract-1" else "glimmer",
            }

    with pytest.raises(ValueError, match="one engine"):
        demo._engine_transport(_Service())


def test_fingerprint_mismatch_yields_no_verdict(demo, tmp_path, monkeypatch):
    add_specialists(demo, tmp_path, monkeypatch)
    demo.config["glimmer"] = {**GLIMMER, "fingerprint": "cd" * 32}
    with pytest.raises(RuntimeError, match="returned no verdict"):
        demo.assess(dict(BODY), lambda text: None)


def test_malformed_verdict_yields_no_verdict(demo, tmp_path, monkeypatch):
    add_specialists(demo, tmp_path, monkeypatch)
    StubChat.text = "no verdict block here"
    with pytest.raises(RuntimeError, match="returned no verdict"):
        demo.assess(dict(BODY), lambda text: None)


def test_route_consults_routed_specialists_and_preserves_agent_error(
    demo, tmp_path, monkeypatch
):
    add_specialists(demo, tmp_path, monkeypatch)

    def outage(*args, **kwargs):
        raise RuntimeError("synthetic agent outage")

    monkeypatch.setattr(backend, "arbitrate_assessment", outage)
    result = demo.route({**BODY, "arbitrate": True}, lambda text: None)
    responses = result["assessment"]["responses"]
    assert [item["expert_id"] for item in responses] == ["contract-2"]
    assert not result["assessment"]["errors"]
    assert len(result["glimmer_calls"]) == 1
    assert result["arbitration_requested"] is True
    assert "arbitration" not in result
    assert result["arbitration_error"] == "synthetic agent outage"
    assert result["input"]["rtl"] == BODY["rtl"]


def test_recording_client_keeps_failed_calls():
    class _Outage:
        def complete(self, messages, tools=()):
            raise RuntimeError("synthetic outage")

    lines = []
    recorder = backend._RecordingChatClient(
        _Outage(),
        "http://127.0.0.1:1/v1/chat/completions",
        "synthetic",
        progress=lines.append,
    )
    with pytest.raises(RuntimeError, match="synthetic outage"):
        recorder.complete([{"role": "user", "content": "hi"}])
    assert recorder.calls[0]["error"] == "synthetic outage"
    assert any("failed" in line for line in lines)


def test_recording_client_streams_call_lifecycle():
    class _Slow:
        def complete(self, messages, tools=()):
            time.sleep(0.3)
            return {"content": VERDICT_TEXT}

    lines = []
    recorder = backend._RecordingChatClient(
        _Slow(),
        "http://127.0.0.1:1/v1/chat/completions",
        "synthetic",
        progress=lines.append,
        heartbeat_seconds=0.05,
    )
    recorder.complete(
        [
            {"role": "system", "content": "skill text"},
            {"role": "user", "content": "state text"},
        ]
    )
    assert lines[0].startswith("Call 1: POST http://127.0.0.1:1")
    assert any("still waiting" in line for line in lines)
    assert lines[-1].startswith("Call 1 finished in")


def test_assess_streams_call_progress_to_job_log(demo, tmp_path, monkeypatch):
    add_specialists(demo, tmp_path, monkeypatch)
    lines = []
    demo.assess(dict(BODY), lines.append)
    assert any(line.startswith("Call 1: POST") for line in lines)
    assert any(line.startswith("Call 1 finished in") for line in lines)


def test_route_without_arbitration_skips_the_agent(demo, tmp_path, monkeypatch):
    add_specialists(demo, tmp_path, monkeypatch)
    monkeypatch.setattr(
        backend,
        "arbitrate_assessment",
        lambda *args, **kwargs: pytest.fail("Agent consulted"),
    )
    result = demo.route(dict(BODY), lambda text: None)
    assert result["arbitration_requested"] is False
    assert "arbitration" not in result
    assert "arbitration_error" not in result
    assert result["assessment"]["responses"]


def test_route_arbitration_prefers_the_arbiter_service(
    demo, tmp_path, monkeypatch
):
    add_specialists(demo, tmp_path, monkeypatch)
    demo.config["arbiter"] = {
        "endpoint": "http://127.0.0.1:18400/v1/chat/completions",
        "model": "synthetic-arbiter",
        "ssh_host": "spark",
    }
    captured: dict[str, Any] = {}

    class _Client:
        def __init__(self, endpoint, model, ssh_host=None, **kwargs):
            captured.update(endpoint=endpoint, model=model, ssh_host=ssh_host)

    monkeypatch.setattr(backend, "LocalGlimmerClient", _Client)
    monkeypatch.setattr(
        backend,
        "arbitrate_assessment",
        lambda *args, **kwargs: kwargs["client_factory"]() or {},
    )
    demo.route({**BODY, "arbitrate": True}, lambda text: None)
    assert captured == {
        "endpoint": "http://127.0.0.1:18400/v1/chat/completions",
        "model": "synthetic-arbiter",
        "ssh_host": "spark",
    }


def test_route_spark_arbiter_uses_default_spark_model(
    demo, tmp_path, monkeypatch
):
    add_specialists(demo, tmp_path, monkeypatch)
    demo.config["arbiter"] = {
        "engine": "spark",
        "model": "muse-spark-1.3-contributor",
    }
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        backend,
        "arbitrate_assessment",
        lambda *args, **kwargs: captured.update(kwargs) or {},
    )
    lines: list[str] = []
    result = demo.route({**BODY, "arbitrate": True}, lines.append)
    assert "client_factory" not in captured
    assert captured["model"] == "muse-spark-1.3-contributor"
    assert result["arbitration_model"] == "muse-spark-1.3-contributor"
    assert any("muse-spark-1.3-contributor" in line for line in lines)
    assert any("validated arbitration response" in line for line in lines)


def test_call_with_heartbeat_logs_start_and_returns():
    lines: list[str] = []
    result = backend._call_with_heartbeat(
        lines.append, "start now", "wait %d", lambda: "done"
    )
    assert result == "done"
    assert lines == ["start now"]


def test_call_with_heartbeat_releases_on_error():
    lines: list[str] = []

    def _boom() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        backend._call_with_heartbeat(
            lines.append, "start now", "wait %d", _boom
        )
    assert lines == ["start now"]


def test_route_without_any_arbiter_reports_cleanly(demo, tmp_path, monkeypatch):
    from types import SimpleNamespace

    add_specialists(demo, tmp_path, monkeypatch)
    demo.config["glimmer"] = None
    monkeypatch.setattr(
        demo,
        "_engine_transport",
        lambda service, log: (object(), SimpleNamespace(calls=[])),
    )

    class _Service:
        broken_experts: dict[str, Any] = {}
        unprofiled_experts: list[str] = []

        def assess(
            self, scope: Any, text: str, transport: Any
        ) -> dict[str, Any]:
            return {
                "scope": scope,
                "routing": {"experts": []},
                "responses": [],
                "errors": [],
            }

    monkeypatch.setattr(
        backend, "InferenceService", lambda registry: _Service()
    )
    result = demo.route({**BODY, "arbitrate": True}, lambda text: None)
    assert result["arbitration_requested"] is True
    assert "arbitration" not in result
    assert "arbiter" in result["arbitration_error"]


def test_route_without_arbiter_refuses_scoring_arbitration(demo, tmp_path):
    add_adapters(demo, tmp_path)
    demo.config["glimmer"]["endpoint"] = "http://127.0.0.1:9/score"
    result = demo.route({**BODY, "arbitrate": True}, lambda text: None)
    assert result["arbitration_requested"] is True
    assert "arbitration" not in result
    assert "arbiter" in result["arbitration_error"]


def test_shim_health_checks_the_model_list(demo, tmp_path, monkeypatch):
    add_adapters(demo, tmp_path)

    class _Reply:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self.body

    seen = []

    def fake_open(url, timeout=None):
        seen.append(url)
        return _Reply(b'{"data": [{"id": "synthetic-adapter"}]}')

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    health = demo.glimmer_health()
    assert health["status"] == "ready"
    assert health["transport"] == "Loopback scoring"
    assert seen == ["http://127.0.0.1:18401/v1/models"]

    def missing(url, timeout=None):
        return _Reply(b'{"data": []}')

    monkeypatch.setattr("urllib.request.urlopen", missing)
    with pytest.raises(RuntimeError, match="not served"):
        demo.glimmer_health()


def test_changed_skill_is_refused_before_any_consult(tmp_path):
    _glimmer_registry(tmp_path)
    (tmp_path / "contract-1" / "SKILL.md").write_text("Edited skill text.")
    with pytest.raises(ValueError, match="skill digest"):
        settings(tmp_path, registry=str(tmp_path / "models.json"))


def test_status_reports_specialist_skill_binding(demo, tmp_path, monkeypatch):
    add_specialists(demo, tmp_path, monkeypatch)
    status = demo.status()
    assert status["registry_error"] is None
    assert status["glimmer_configured"] is True
    rows = {row["id"]: row for row in status["models"]}
    assert set(rows) == {"contract-1", "contract-2"}
    for row in rows.values():
        assert row["kind"] == "glimmer"
        assert row["profile"] is not None
    skill = tmp_path / "contract-1" / "SKILL.md"
    expected = hashlib.sha256(skill.read_bytes()).hexdigest()
    assert rows["contract-1"]["skill_sha256"] == expected


def test_status_survives_a_missing_registry(demo, tmp_path):
    demo.config["registry"] = str(tmp_path / "missing.json")
    status = demo.status()
    assert status["registry_error"].startswith(
        ("ValueError", "OSError", "FileNotFoundError")
    )
    assert status["models"] == []


def test_glimmer_native_tool_arguments_and_budgets(monkeypatch):
    observed = []

    def reply(self, url, payload=None):
        observed.append(payload)
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call",
                                "function": {
                                    "name": "experts.arbitrate",
                                    "arguments": '{"conclusion":"unresolved"}',
                                },
                            }
                        ]
                    },
                }
            ]
        }

    monkeypatch.setattr(LocalGlimmerClient, "_request", reply)
    client = LocalGlimmerClient(
        "http://127.0.0.1:18400/v1/chat/completions",
        "synthetic",
        max_calls=1,
        decode_arguments=True,
    )
    result = client.complete([], tools=[{"type": "function"}])
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "conclusion": "unresolved"
    }
    assert observed[0]["chat_template_kwargs"] == {"reasoning_strength": "low"}
    with pytest.raises(RuntimeError, match="budget"):
        client.complete([])


def test_glimmer_incomplete_response_is_never_accepted(monkeypatch):
    monkeypatch.setattr(
        LocalGlimmerClient,
        "_request",
        lambda *args: {
            "choices": [
                {"finish_reason": "length", "message": {"content": "partial"}}
            ]
        },
    )
    client = LocalGlimmerClient(
        "http://127.0.0.1:18400/v1/chat/completions", "synthetic"
    )
    with pytest.raises(RuntimeError, match="budget"):
        client.complete([])


def test_glimmer_malformed_responses_raise_value_errors(monkeypatch):
    bodies = [
        {},
        {"choices": []},
        {"choices": [{"message": "not-a-dict"}]},
        {"choices": [{"message": {"tool_calls": "not-a-list"}}]},
        {"choices": [{"message": {"tool_calls": [{"id": "call-1"}]}}]},
        [1, 2],
    ]
    calls = {"count": 0}

    def reply(self, url, payload=None):
        body = bodies[calls["count"]]
        calls["count"] += 1
        return body

    monkeypatch.setattr(LocalGlimmerClient, "_request", reply)
    client = LocalGlimmerClient(
        "http://127.0.0.1:18400/v1/chat/completions", "synthetic"
    )
    for _ in bodies:
        with pytest.raises(ValueError):
            client.complete([])


def test_final_operator_acknowledgement_is_requested_from_glimmer(monkeypatch):
    client = LocalGlimmerClient(
        "http://127.0.0.1:8000/v1/chat/completions", "fixture"
    )
    captured = []
    reply = {
        "role": "assistant",
        "content": '{"action":"stop","run_id":"demo","executed":false,"scope":"operator_replay"}',
    }

    def request(url, payload):
        captured.append(payload)
        return {"choices": [{"finish_reason": "stop", "message": reply}]}

    monkeypatch.setattr(client, "_request", request)
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "decision",
                    "type": "function",
                    "function": {
                        "name": "operator.decide",
                        "arguments": '{"action":"stop","run_id":"demo"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "decision",
            "content": '{"recorded":true}',
        },
    ]
    original = copy.deepcopy(messages)
    assert client.complete(messages, tools=[{"type": "function"}]) == reply
    assert messages == original
    assert len(captured) == 1
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[0]["tool_choice"] == "none"
    assert captured[0]["messages"][-1]["role"] == "user"


def test_state_text_rejects_smuggled_section_headers():
    from siliconshield.experts.identity import state_sections

    good = backend.state_text(dict(BODY))
    requirement, _, _ = state_sections(good)
    assert requirement == BODY["requirement"]
    for field in ("requirement", "assumptions"):
        for header in ("REQUIREMENT", "ASSUMPTIONS", "RTL"):
            body = dict(BODY)
            body[field] = "first line\n%s\nsecond line" % header
            with pytest.raises(ValueError, match="section header"):
                backend.state_text(body)


def test_glimmer_complete_rejects_malformed_messages(monkeypatch):
    def reply(self, url, payload=None):
        raise AssertionError("no network in this test")

    monkeypatch.setattr(LocalGlimmerClient, "_request", reply)
    client = LocalGlimmerClient(
        endpoint="http://127.0.0.1:1/v1/chat/completions",
        model="synthetic",
        max_calls=4,
        timeout=1,
    )
    with pytest.raises(ValueError, match="list of objects"):
        client.complete([{"role": "system"}, "nope"])
    with pytest.raises(ValueError, match="must be text"):
        client.complete([{"role": "system", "content": ["nope"]}])


def test_non_ascii_token_is_forbidden_not_dropped(http_api):
    code, result = http_api(
        "/api/assess", dict(BODY), headers={"X-Demo-Token": "tökén"}
    )
    assert code == 403
    assert "error" in result
    code, _ = http_api("/api/status", None)
    assert code == 200


def test_empty_transfer_encoding_is_rejected(http_api):
    code, _ = http_api(
        "/api/assess", dict(BODY), headers={"Transfer-Encoding": ""}
    )
    assert code == 400


def test_adapter_endpoints_use_the_scoring_path(tmp_path):
    registry = _adapter_registry(tmp_path)
    cfg = settings(
        tmp_path,
        registry=registry,
        glimmer=dict(ADAPTER_GLIMMER),
    )
    assert cfg["glimmer"]["endpoint"].endswith("/score")
    with pytest.raises(ValueError, match="/score"):
        settings(
            tmp_path,
            registry=registry,
            glimmer={
                **ADAPTER_GLIMMER,
                "endpoint": "http://127.0.0.1:18401/chat/completions",
            },
        )
    with pytest.raises(ValueError, match="ssh_host"):
        settings(
            tmp_path,
            registry=registry,
            glimmer={**ADAPTER_GLIMMER, "ssh_host": "spark"},
        )


def test_mixed_engine_registries_are_refused(tmp_path):
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    _adapter_registry(tmp_path)
    skill = tmp_path / "adapter-contract-1"
    digest = hashlib.sha256((skill / "SKILL.md").read_bytes()).hexdigest()
    path = tmp_path / "mixed.json"
    path.write_text(
        json.dumps(
            {
                "schema": GUI_MODEL_REGISTRY_SCHEMA,
                "models": [
                    {
                        "id": "contract-1",
                        "kind": "adapter",
                        "display_name": "adapter",
                        "checkpoint": str(skill),
                        "expected_files": {
                            "adapter_model.safetensors": ADAPTER_FINGERPRINT,
                            "SKILL.md": digest,
                        },
                        "device": "synthetic gpus",
                    },
                    {
                        "id": "contract-2",
                        "kind": "glimmer",
                        "display_name": "glimmer",
                        "checkpoint": str(skill),
                        "expected_files": {
                            "model.safetensors": "ab" * 32,
                            "SKILL.md": digest,
                        },
                        "device": "synthetic spark",
                    },
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="one engine"):
        settings(tmp_path, registry=str(path))


def test_arbiter_section_is_validated(tmp_path):
    arbiter = {
        "endpoint": "http://127.0.0.1:18400/v1/chat/completions",
        "model": "synthetic-arbiter",
        "ssh_host": "spark",
    }
    assert settings(tmp_path, arbiter=dict(arbiter))["arbiter"] == arbiter
    with pytest.raises(ValueError, match="arbiter needs"):
        settings(tmp_path, arbiter={"model": "synthetic-arbiter"})
    with pytest.raises(ValueError, match="arbiter accepts"):
        settings(tmp_path, arbiter={**arbiter, "fingerprint": "ab" * 32})


def test_spark_arbiter_section_is_validated(tmp_path):
    arbiter = {"engine": "spark", "model": "muse-spark-1.3-contributor"}
    assert settings(tmp_path, arbiter=dict(arbiter))["arbiter"] == arbiter
    with pytest.raises(ValueError, match="spark arbiter needs"):
        settings(tmp_path, arbiter={"engine": "spark"})


def test_shipped_example_config_selects_spark_final_assessment():
    root = Path(__file__).resolve().parent.parent
    cfg = config.load_config(root / "configs" / "web-demo.example.json")
    assert cfg["arbiter"] == {
        "engine": "spark",
        "model": "muse-spark-1.3-contributor",
    }


def test_glimmer_config_needs_endpoint_model_and_fingerprint(tmp_path):
    with pytest.raises(ValueError, match="endpoint and model"):
        settings(tmp_path, glimmer={})
    with pytest.raises(ValueError, match="endpoint and model"):
        settings(
            tmp_path,
            glimmer={"endpoint": "http://127.0.0.1:1/chat/completions"},
        )
    with pytest.raises(ValueError, match="fingerprint"):
        settings(
            tmp_path,
            glimmer={
                "endpoint": "http://127.0.0.1:1/chat/completions",
                "model": "synthetic",
            },
        )


def test_glimmer_budgets_validated_at_construction():
    good = {
        "endpoint": "http://127.0.0.1:1/v1/chat/completions",
        "model": "synthetic",
    }
    with pytest.raises(ValueError, match="max_calls"):
        LocalGlimmerClient(**good, max_calls="6")
    with pytest.raises(ValueError, match="max_tokens"):
        LocalGlimmerClient(**good, max_tokens=0)
    with pytest.raises(ValueError, match="timeout"):
        LocalGlimmerClient(**good, timeout=-1)


def _await_quiet(demo, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with demo.lock:
            running = [
                job
                for job in demo.jobs.values()
                if job.get("status") == "running"
            ]
            threads = [thread for thread in demo._threads if thread.is_alive()]
        if not running and not threads:
            return
        time.sleep(0.05)
    raise AssertionError("jobs did not finish")


def test_close_drains_running_jobs(demo):
    started = threading.Event()

    def slow(log):
        started.set()
        time.sleep(0.5)
        return {"ok": True}

    receipt = demo.submit_task("assess", slow)
    assert started.wait(timeout=10)
    demo.close()
    with demo.lock:
        job = demo.jobs[receipt["job_id"]]
    assert job["status"] == "complete"
    assert (demo.state / ("request-%s.json" % receipt["job_id"])).exists()


def test_close_is_idempotent(demo):
    demo.close()
    demo.close()
    assert demo.stopping is True


def test_terminal_jobs_evicted_from_memory_and_disk(demo):
    for _ in range(2):
        for _ in range(5):
            demo.submit_task("assess", lambda log: {"ok": True})
        _await_quiet(demo)
    with demo.lock:
        assert len(demo.jobs) <= 8
    assert len(list(demo.state.glob("request-*.json"))) <= 8


def test_rejected_glimmer_input_keeps_its_budget(tmp_path):
    client = LocalGlimmerClient(
        "http://127.0.0.1:9/v1/chat/completions",
        "synthetic",
        max_calls=1,
        audit_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="list of objects"):
        client.complete([{"role": "system"}, "nope"])
    assert client.calls == 0


def test_route_scope_is_validated_synchronously(http_api):
    code, result = http_api("/api/route", {**BODY, "scope": "not-a-domain"})
    assert code == 400
    assert "measured routing domain" in result["error"]


def test_evicted_jobs_leave_tombstones(demo):
    first = demo.submit_task("assess", lambda log: {"ok": True})["job_id"]
    for _ in range(2):
        for _ in range(5):
            demo.submit_task("assess", lambda log: {"ok": True})
        _await_quiet(demo)
    assert demo.was_evicted(first) is True
    assert demo.was_evicted("0" * 32) is False


def test_evicted_job_polls_as_expired(demo, http_api):
    first = demo.submit_task("assess", lambda log: {"ok": True})["job_id"]
    for _ in range(2):
        for _ in range(5):
            demo.submit_task("assess", lambda log: {"ok": True})
        _await_quiet(demo)
    code, result = http_api(f"/api/jobs/{first}")
    assert code == 410
    assert "expired" in result["error"]
    code, _ = http_api("/api/jobs/" + "0" * 32)
    assert code == 404


def test_malformed_history_never_breaks_the_recorded_scan(monkeypatch):
    client = LocalGlimmerClient(
        endpoint="http://127.0.0.1:1/chat/completions",
        model="synthetic",
        max_calls=3,
    )
    seen = []

    def fake_request(endpoint, payload):
        seen.append(payload)
        return {"choices": [{"message": {"content": "done"}}]}

    monkeypatch.setattr(client, "_request", fake_request)
    histories = [
        [{"role": "assistant", "tool_calls": "operator.decide"}],
        [{"role": "assistant", "tool_calls": ["x", 1]}],
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "1", "function": "operator.decide"}],
            }
        ],
    ]
    for history in histories:
        message = client.complete(
            history + [{"role": "tool", "tool_call_id": "1", "content": "{}"}]
        )
        assert message["content"] == "done"
    assert len(seen) == 3


def test_backend_rejects_malformed_inputs(tmp_path):
    assert backend.read_json(5, "fallback") == "fallback"
    with pytest.raises(ValueError, match="must be an object"):
        backend.state_text(["x"])
    with pytest.raises(ValueError, match="must be an object"):
        backend.Demo(["x"], health_poll=False)
    with pytest.raises(ValueError, match="state directory"):
        backend.Demo({"entries": []}, health_poll=False)
    with pytest.raises(ValueError, match="entries with ids"):
        backend.Demo(
            {"state_dir": str(tmp_path / "s"), "entries": [{"x": 1}]},
            health_poll=False,
        )
    demo = backend.Demo(
        {"state_dir": str(tmp_path / "s"), "entries": []},
        health_poll=False,
    )
    assert demo.glimmer_health() == {"status": "not configured"}
    assert demo.status()["models"] == []


def test_demo_config_accepts_string_paths(tmp_path):
    path = tmp_path / "demo.json"
    path.write_text("{}")
    assert config.load_config(str(path))["port"] == 8765
    with pytest.raises(ValueError, match="must be a path"):
        config.load_config(5)


def test_glimmer_client_rejects_malformed_arguments():
    from siliconshield.web_demo import glimmer_client

    endpoint = "http://127.0.0.1:1/chat/completions"
    with pytest.raises(ValueError, match="non-empty string"):
        glimmer_client.LocalGlimmerClient(5, "model")
    with pytest.raises(ValueError, match="audit directory"):
        glimmer_client.LocalGlimmerClient(endpoint, "model", audit_dir=5)
    client = glimmer_client.LocalGlimmerClient(endpoint, "model")
    with pytest.raises(ValueError, match="tools must be a list"):
        client.complete([], tools=5)
