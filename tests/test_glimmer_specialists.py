"""Synthetic tests for Glimmer specialists behind the router flow."""

import hashlib
import json

import pytest
from test_multi_expert import _arbitration_payload, _scripted_client

from siliconshield.research_loop.procedures import glimmer_specialists

FINGERPRINT = "ab" * 32


def _skill_dir(tmp_path, name, text="Synthetic specialist skill."):
    skill = tmp_path / name
    skill.mkdir()
    (skill / "SKILL.md").write_text(text)
    return skill


def _glimmer_profile(model_id, mechanisms, fingerprint=FINGERPRINT):
    from siliconshield.schemas import EXPERT_CAPABILITY_SCHEMA

    return {
        "schema": EXPERT_CAPABILITY_SCHEMA,
        "id": model_id,
        "release": "glimmer-specialist-v1",
        "checkpoint_sha256": fingerprint,
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
        ("glimmer-a", ["ot-access-control"]),
        ("glimmer-b", ["fifo"]),
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


class _StubChat:
    """Canned Glimmer completions for transport tests."""

    def __init__(self, text):
        self.text = text
        self.calls = []

    def complete(self, messages, tools=()):
        self.calls.append(messages)
        return {"content": self.text}


VERDICT_TEXT = """Reasoning about the guard.

VERDICT: violates_requirement
CONFIDENCE: 0.8
EVIDENCE: wr_en writes without a privilege check
OPEN QUESTIONS: none
"""


def test_parse_verdict_accepts_exact_block():
    verdict = glimmer_specialists.parse_verdict(VERDICT_TEXT)
    assert verdict == {"label": "violates_requirement", "confidence": 0.8}
    with pytest.raises(ValueError, match="exactly one VERDICT"):
        glimmer_specialists.parse_verdict("no block here")
    with pytest.raises(ValueError, match="exactly one VERDICT"):
        glimmer_specialists.parse_verdict(
            "VERDICT: violates_requirement\n"
            "VERDICT: satisfies_requirement\n"
            "CONFIDENCE: 0.8\n"
        )
    with pytest.raises(ValueError, match="exactly one CONFIDENCE"):
        glimmer_specialists.parse_verdict("VERDICT: violates_requirement\n")
    with pytest.raises(ValueError, match="out of \\[0, 1\\]"):
        glimmer_specialists.parse_verdict(
            "VERDICT: violates_requirement\nCONFIDENCE: 1.5\n"
        )


def test_scores_for_splits_remainder_and_refuses_low_confidence():
    scores = glimmer_specialists.scores_for("violates_requirement", 0.8)
    assert scores["violates_requirement"] == 0.8
    assert scores["satisfies_requirement"] == pytest.approx(0.1)
    assert scores["insufficient_context"] == pytest.approx(0.1)
    assert abs(sum(scores.values()) - 1.0) < 1e-9
    with pytest.raises(ValueError, match="cannot top"):
        glimmer_specialists.scores_for("violates_requirement", 0.2)
    with pytest.raises(ValueError, match="Unknown verdict label"):
        glimmer_specialists.scores_for("maybe", 0.9)


def test_transport_builds_validated_payload(tmp_path):
    from siliconshield.experts.responses import build_expert_response
    from siliconshield.experts.service import InferenceService

    registry_path = _glimmer_registry(tmp_path)
    service = InferenceService(registry_path)
    chat = _StubChat(VERDICT_TEXT)
    transport = glimmer_specialists.GlimmerTransport(
        chat,
        "synthetic-glimmer",
        "http://127.0.0.1:18400/v1/chat/completions",
        FINGERPRINT,
        {"glimmer-a": service.model_entry("glimmer-a")},
    )
    state = "REQUIREMENT\nr\nASSUMPTIONS\na\nRTL\nassign q = d;\n"
    payload = transport("glimmer-a", state)
    assert payload["label"] == "violates_requirement"
    assert payload["model_id"] == "glimmer-a"
    assert (
        payload["execution_identity"]["question_contract"]
        == "glimmer-specialist-v1"
    )
    assert chat.calls[0][0]["role"] == "system"
    assert "Synthetic specialist skill" in chat.calls[0][0]["content"]
    assert chat.calls[0][1]["content"] == state
    profile = service.profiles_for(["glimmer-a"])["glimmer-a"]
    response = build_expert_response(profile, state, payload)
    assert response["scores"]["violates_requirement"] == 0.8


def test_transport_refuses_tampered_skill_and_tight_budget(tmp_path):
    from siliconshield.experts.service import InferenceService

    registry_path = _glimmer_registry(tmp_path)
    service = InferenceService(registry_path)
    entries = {
        "glimmer-a": service.model_entry("glimmer-a"),
        "glimmer-b": service.model_entry("glimmer-b"),
    }
    chat = _StubChat(VERDICT_TEXT)
    transport = glimmer_specialists.GlimmerTransport(
        chat,
        "synthetic-glimmer",
        "http://127.0.0.1:18400/v1/chat/completions",
        FINGERPRINT,
        entries,
    )
    state = "REQUIREMENT\nr\nASSUMPTIONS\na\nRTL\nassign q = d;\n"
    (tmp_path / "glimmer-a" / "SKILL.md").write_text("tampered skill")
    with pytest.raises(ValueError, match="changed since registry load"):
        transport("glimmer-a", state)
    assert chat.calls == []
    tight = glimmer_specialists.GlimmerTransport(
        chat,
        "synthetic-glimmer",
        "http://127.0.0.1:18400/v1/chat/completions",
        FINGERPRINT,
        entries,
        max_tokens=8190,
    )
    with pytest.raises(ValueError, match="exceeds"):
        tight("glimmer-b", state)


def test_transport_device_type_defaults_to_remote(tmp_path):
    from siliconshield.experts.service import InferenceService

    registry_path = _glimmer_registry(tmp_path)
    service = InferenceService(registry_path)
    entries = {"glimmer-a": service.model_entry("glimmer-a")}
    state = "REQUIREMENT\nr\nASSUMPTIONS\na\nRTL\nassign q = d;\n"
    default = glimmer_specialists.GlimmerTransport(
        _StubChat(VERDICT_TEXT),
        "synthetic-glimmer",
        "http://127.0.0.1:18400/v1/chat/completions",
        FINGERPRINT,
        entries,
    )
    assert (
        default("glimmer-a", state)["execution_identity"]["inference"][
            "device_type"
        ]
        == "remote"
    )
    remote = glimmer_specialists.GlimmerTransport(
        _StubChat(VERDICT_TEXT),
        "synthetic-glimmer",
        "http://127.0.0.1:18400/v1/chat/completions",
        FINGERPRINT,
        entries,
        device_type="ssh:spark",
    )
    assert (
        remote("glimmer-a", state)["execution_identity"]["inference"][
            "device_type"
        ]
        == "ssh:spark"
    )
    with pytest.raises(ValueError, match="device type"):
        glimmer_specialists.GlimmerTransport(
            _StubChat(VERDICT_TEXT),
            "synthetic-glimmer",
            "http://127.0.0.1:18400/v1/chat/completions",
            FINGERPRINT,
            entries,
            device_type="  ",
        )


def test_run_specialists_expert_mode_writes_receipt(tmp_path):
    parser = glimmer_specialists.build_parser()
    registry_path = _glimmer_registry(tmp_path)
    design = tmp_path / "demo.sv"
    design.write_text("module topmodule; endmodule\n")
    output = tmp_path / "receipt"
    args = parser.parse_args(
        [
            "--registry",
            registry_path,
            "--expert",
            "glimmer-a",
            "--design",
            str(design),
            "--requirement",
            "Hold the count.",
            "--assumptions",
            "Two-state RTL.",
            "--endpoint",
            "http://127.0.0.1:18400/v1/chat/completions",
            "--model",
            "synthetic-glimmer",
            "--glimmer-fingerprint",
            FINGERPRINT,
            "--output",
            str(output),
            "--brief",
        ]
    )
    chat = _StubChat(VERDICT_TEXT)
    report = glimmer_specialists.run_specialists(
        args, client_factory=lambda: chat
    )
    assert report["assessment"]["responses"][0]["label"] == (
        "violates_requirement"
    )
    assert report["load_seconds"] == 0.0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["expert"] == "glimmer-a"
    assert receipt["glimmer_fingerprint"] == FINGERPRINT
    assert set(receipt["skill_sha256"]) == {"glimmer-a"}
    assert receipt["arbitration"] is None
    assert receipt["arbitration_model"] is None


def test_run_specialists_routed_mode_and_spark_verdict(tmp_path):
    parser = glimmer_specialists.build_parser()
    registry_path = _glimmer_registry(tmp_path)
    design = tmp_path / "demo.sv"
    design.write_text("module topmodule; endmodule\n")
    output = tmp_path / "receipt"
    args = parser.parse_args(
        [
            "--registry",
            registry_path,
            "--scope",
            "fifo",
            "--design",
            str(design),
            "--requirement",
            "Hold the count.",
            "--assumptions",
            "Two-state RTL.",
            "--endpoint",
            "http://127.0.0.1:18400/v1/chat/completions",
            "--model",
            "synthetic-glimmer",
            "--glimmer-fingerprint",
            FINGERPRINT,
            "--output",
            str(output),
            "--arbitrate",
        ]
    )
    chat = _StubChat(VERDICT_TEXT)
    spark = _scripted_client(
        _arbitration_payload(
            conclusion="violates_requirement",
            evidence=[{"expert_id": "glimmer-b", "relevance": "covers fifo"}],
        )
    )
    report = glimmer_specialists.run_specialists(
        args,
        client_factory=lambda: chat,
        arbitration_client_factory=lambda: spark,
    )
    routing = report["assessment"]["routing"]
    assert routing["experts"] == ["glimmer-b"]
    assert report["arbitration"]["conclusion"] == "violates_requirement"
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["arbitration_model"] == "muse-spark-1.3-contributor"
    assert receipt["arbitration"]["conclusion"] == "violates_requirement"


def test_failed_arbitration_preserves_specialist_receipt(tmp_path):
    parser = glimmer_specialists.build_parser()
    registry_path = _glimmer_registry(tmp_path)
    design = tmp_path / "demo.sv"
    design.write_text("module topmodule; endmodule\n")
    output = tmp_path / "receipt"
    args = parser.parse_args(
        [
            "--registry",
            registry_path,
            "--scope",
            "fifo",
            "--design",
            str(design),
            "--requirement",
            "Hold the count.",
            "--assumptions",
            "Two-state RTL.",
            "--endpoint",
            "http://127.0.0.1:18400/v1/chat/completions",
            "--model",
            "synthetic-glimmer",
            "--glimmer-fingerprint",
            FINGERPRINT,
            "--output",
            str(output),
            "--arbitrate",
        ]
    )
    chat = _StubChat(VERDICT_TEXT)

    class _Outage:
        def complete(self, messages, tools=()):
            raise RuntimeError("simulated Spark outage")

    with pytest.raises(RuntimeError, match="simulated Spark outage"):
        glimmer_specialists.run_specialists(
            args,
            client_factory=lambda: chat,
            arbitration_client_factory=lambda: _Outage(),
        )
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["responses"][0]["label"] == "violates_requirement"
    assert receipt["arbitration"] is None
    assert "simulated Spark outage" in receipt["arbitration_error"]


def test_arbitrate_needs_routed_scope(tmp_path):
    parser = glimmer_specialists.build_parser()
    registry_path = _glimmer_registry(tmp_path)
    design = tmp_path / "demo.sv"
    design.write_text("module topmodule; endmodule\n")
    args = parser.parse_args(
        [
            "--registry",
            registry_path,
            "--expert",
            "glimmer-a",
            "--design",
            str(design),
            "--requirement",
            "Hold the count.",
            "--endpoint",
            "http://127.0.0.1:18400/v1/chat/completions",
            "--model",
            "synthetic-glimmer",
            "--glimmer-fingerprint",
            FINGERPRINT,
            "--output",
            str(tmp_path / "receipt"),
            "--arbitrate",
        ]
    )
    chat = _StubChat(VERDICT_TEXT)
    with pytest.raises(ValueError, match="needs --scope"):
        glimmer_specialists.run_specialists(args, client_factory=lambda: chat)
    assert chat.calls == []


def test_run_specialists_refuses_local_worker_entries(tmp_path):
    from test_multi_expert import _profile as _local_profile
    from test_multi_expert import _write_registry as _write_local

    parser = glimmer_specialists.build_parser()
    registry_path = _write_local(
        tmp_path, _local_profile("expert-a", ["fifo"]), {}
    )
    design = tmp_path / "demo.sv"
    design.write_text("module topmodule; endmodule\n")
    args = parser.parse_args(
        [
            "--registry",
            registry_path,
            "--expert",
            "expert-a",
            "--design",
            str(design),
            "--requirement",
            "Hold the count.",
            "--endpoint",
            "http://127.0.0.1:18400/v1/chat/completions",
            "--model",
            "synthetic-glimmer",
            "--glimmer-fingerprint",
            FINGERPRINT,
            "--output",
            str(tmp_path / "receipt"),
        ]
    )
    chat = _StubChat(VERDICT_TEXT)
    with pytest.raises(ValueError, match="not a glimmer specialist"):
        glimmer_specialists.run_specialists(args, client_factory=lambda: chat)
    assert chat.calls == []
