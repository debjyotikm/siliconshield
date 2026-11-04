"""Adapter scoring transport and registry kind on synthetic inputs."""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from siliconshield.experts.service import InferenceService
from siliconshield.research_loop.procedures.adapter_specialists import (
    AdapterTransport,
)
from siliconshield.training.laya.question_contracts import (
    ADAPTER_CONTRACT,
    known_contracts,
)

FINGERPRINT = "cd" * 32

GOOD_REPLY = {
    "label": "violates_requirement",
    "scores": {
        "satisfies_requirement": 0.05,
        "violates_requirement": 0.9,
        "insufficient_context": 0.05,
    },
    "input_tokens": 120,
    "label_token_mass": 0.99,
    "precision": "bf16",
    "hardware": ["synthetic-gpu", "synthetic-gpu"],
}


def _skill_dir(tmp_path: Path, name: str) -> Path:
    skill = tmp_path / name
    skill.mkdir(exist_ok=True)
    (skill / "SKILL.md").write_text("Synthetic %s adapter doc." % name)
    return skill


def _profile(model_id: str) -> dict[str, Any]:
    from siliconshield.schemas import EXPERT_CAPABILITY_SCHEMA

    return {
        "schema": EXPERT_CAPABILITY_SCHEMA,
        "id": model_id,
        "release": "synthetic-adapter",
        "checkpoint_sha256": FINGERPRINT,
        "question_contract": ADAPTER_CONTRACT,
        "strengths": ["synthetic strength"],
        "weaknesses": ["synthetic weakness"],
        "supported_mechanisms": ["fifo"],
        "known_failures": ["synthetic failure"],
        "qualification_ref": "synthetic probe",
        "measurements": [{"eval": "synthetic", "correct": 2, "total": 3}],
    }


def _registry(tmp_path: Path) -> str:
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    skill = _skill_dir(tmp_path, "contract-1")
    digest = hashlib.sha256((skill / "SKILL.md").read_bytes()).hexdigest()
    profile_path = tmp_path / "contract-1.profile.json"
    profile_path.write_text(json.dumps(_profile("contract-1")))
    registry_path = tmp_path / "models.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": GUI_MODEL_REGISTRY_SCHEMA,
                "models": [
                    {
                        "id": "contract-1",
                        "kind": "adapter",
                        "display_name": "contract-1 adapter",
                        "checkpoint": str(skill),
                        "expected_files": {
                            "adapter_model.safetensors": FINGERPRINT,
                            "SKILL.md": digest,
                        },
                        "device": "synthetic gpus",
                        "profile": str(profile_path),
                    }
                ],
            }
        )
    )
    return str(registry_path)


@pytest.fixture
def stub_shim():
    """In-process scoring shim with a scripted reply per test."""
    state: dict[str, Any] = {"reply": dict(GOOD_REPLY), "status": 200}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(size)
            body = json.dumps(state["reply"]).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, state
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _transport(server, tmp_path, **kwargs):
    from siliconshield.experts.registry import load_registry

    entries = {
        entry["id"]: entry for entry in load_registry(_registry(tmp_path))
    }
    return AdapterTransport(
        "http://127.0.0.1:%d/score" % server.server_port,
        "synthetic-adapter",
        FINGERPRINT,
        entries,
        **kwargs,
    )


def test_adapter_contract_is_registered():
    assert ADAPTER_CONTRACT in known_contracts()


def test_adapter_transport_scores_through_the_router_flow(stub_shim, tmp_path):
    server, _ = stub_shim
    service = InferenceService(_registry(tmp_path))
    transport = _transport(server, tmp_path)
    assessment = service.assess_expert(
        "contract-1", "canonical state", transport
    )
    assert not assessment["errors"]
    (response,) = assessment["responses"]
    assert response["label"] == "violates_requirement"
    assert response["scores"]["violates_requirement"] == pytest.approx(0.9)
    identity = response["execution_identity"]
    assert identity["checkpoint_sha256"] == FINGERPRINT
    assert identity["question_contract"] == ADAPTER_CONTRACT
    assert identity["inference"]["backend"] == "adapter.score"
    assert identity["inference"]["device_type"] == "2x synthetic-gpu"
    assert len(transport.calls) == 1
    assert transport.calls[0]["kind"] == "score"
    assert (
        transport.calls[0]["skill_sha256"]
        == hashlib.sha256(
            (tmp_path / "contract-1" / "SKILL.md").read_bytes()
        ).hexdigest()
    )


def test_adapter_transport_streams_progress(stub_shim, tmp_path):
    server, _ = stub_shim
    lines: list[str] = []
    transport = _transport(server, tmp_path, progress=lines.append)
    transport("contract-1", "canonical state")
    assert lines[0].startswith("Call 1: POST http://127.0.0.1:")
    assert lines[-1].startswith("Call 1 finished in")


def test_adapter_transport_requires_a_score_endpoint(tmp_path):
    from siliconshield.experts.registry import load_registry

    entries = {
        entry["id"]: entry for entry in load_registry(_registry(tmp_path))
    }
    with pytest.raises(ValueError, match="/score"):
        AdapterTransport(
            "http://127.0.0.1:18401/chat/completions",
            "synthetic-adapter",
            FINGERPRINT,
            entries,
        )


def test_adapter_transport_rechecks_the_skill(stub_shim, tmp_path):
    server, _ = stub_shim
    transport = _transport(server, tmp_path)
    (tmp_path / "contract-1" / "SKILL.md").write_text("Edited doc.")
    with pytest.raises(ValueError, match="skill changed"):
        transport("contract-1", "canonical state")


@pytest.mark.parametrize(
    "reply",
    [
        {"label": "unknown-label", "scores": GOOD_REPLY["scores"]},
        {"label": "violates_requirement", "scores": {"a": 1.0}},
        {
            "label": "violates_requirement",
            "scores": {
                "satisfies_requirement": 0.5,
                "violates_requirement": 0.5,
                "insufficient_context": 0.5,
            },
        },
        {
            "label": "satisfies_requirement",
            "scores": GOOD_REPLY["scores"],
        },
        {k: v for k, v in GOOD_REPLY.items() if k != "precision"},
        {**GOOD_REPLY, "hardware": []},
        {**GOOD_REPLY, "hardware": "not-a-list"},
        "not-a-dict",
    ],
)
def test_adapter_transport_refuses_malformed_replies(
    stub_shim, tmp_path, reply
):
    server, state = stub_shim
    state["reply"] = reply
    transport = _transport(server, tmp_path)
    with pytest.raises(ValueError):
        transport("contract-1", "canonical state")


def test_adapter_transport_surfaces_shim_refusals(stub_shim, tmp_path):
    server, state = stub_shim
    state["reply"] = {"error": "Input exceeds 2048 tokens."}
    state["status"] = 400
    transport = _transport(server, tmp_path)
    with pytest.raises(ValueError, match="Input exceeds 2048 tokens"):
        transport("contract-1", "canonical state")


def test_adapter_transport_reports_unreachable_shims(tmp_path):
    from siliconshield.experts.registry import load_registry

    entries = {
        entry["id"]: entry for entry in load_registry(_registry(tmp_path))
    }
    transport = AdapterTransport(
        "http://127.0.0.1:9/score",
        "synthetic-adapter",
        FINGERPRINT,
        entries,
    )
    with pytest.raises(RuntimeError, match="unreachable"):
        transport("contract-1", "canonical state")
    assert transport.calls[0]["error"]


def test_adapter_registry_kind_is_validated(tmp_path):
    from siliconshield.experts.registry import load_registry
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    skill = _skill_dir(tmp_path, "contract-1")
    digest = hashlib.sha256((skill / "SKILL.md").read_bytes()).hexdigest()

    def write(models):
        path = tmp_path / "models.json"
        path.write_text(
            json.dumps({"schema": GUI_MODEL_REGISTRY_SCHEMA, "models": models})
        )
        return str(path)

    base = {
        "id": "contract-1",
        "kind": "adapter",
        "display_name": "contract-1 adapter",
        "checkpoint": str(skill),
        "expected_files": {
            "adapter_model.safetensors": FINGERPRINT,
            "SKILL.md": digest,
        },
        "device": "synthetic gpus",
    }
    assert load_registry(write([base]))[0]["kind"] == "adapter"
    missing = dict(base)
    missing["expected_files"] = {"SKILL.md": digest}
    with pytest.raises(ValueError, match="adapter_model.safetensors"):
        load_registry(write([missing]))
    worker = dict(base)
    worker["python"] = "/bin/false"
    with pytest.raises(ValueError, match="must not name"):
        load_registry(write([worker]))
