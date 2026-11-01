"""Synthetic tests for multi-expert routing and envelopes."""

import json
import sys
from pathlib import Path

import pytest

from siliconshield.experts import (
    arbitration,
    capabilities,
    findings,
    inspection,
    responses,
)
from siliconshield.experts.cli import build_parser, run_assess
from siliconshield.experts.identity import digest, text_digest
from siliconshield.experts.router import route_scope
from siliconshield.experts.service import InferenceService
from siliconshield.experts.worker import WorkerPool
from siliconshield.training.laya.question_contracts import question_for


def _profile(model_id, mechanisms, **overrides):
    from siliconshield.schemas import EXPERT_CAPABILITY_SCHEMA

    payload = {
        "schema": EXPERT_CAPABILITY_SCHEMA,
        "id": model_id,
        "release": "synthetic-1",
        "checkpoint_sha256": "ab" * 32,
        "supported_mechanisms": mechanisms,
        "strengths": ["synthetic strength"],
        "weaknesses": ["synthetic weakness"],
        "known_failures": ["synthetic failure"],
        "question_contract": "siliconshield-requirement-context-v1",
        "qualification_ref": "synthetic run",
        "measurements": [{"eval": "synthetic", "correct": 1, "total": 2}],
    }
    payload.update(overrides)
    return payload


def _identity(weights="ab" * 32):
    questions = question_for("siliconshield-requirement-context-v1")
    questions["security"]["option_order"] = [0, 1, 2]
    return {
        "checkpoint_sha256": weights,
        "export_sha256": "cd" * 32,
        "question_sha256": digest(questions),
        "runtime_sha256": "ef" * 32,
        "question_contract": "siliconshield-requirement-context-v1",
        "inference": {
            "backend": "laya.predict_batch",
            "precision": "torch.bfloat16",
            "torch_version": "synthetic",
            "device_type": "cpu",
            "batch_size": 1,
            "max_len": 1536,
            "head_max_len": 256,
            "sort_by_length": False,
            "option_order": [0, 1, 2],
        },
    }


def _rated(model_id, mechanisms, counts):
    profile = _profile(model_id, mechanisms)
    profile["routing_evidence"] = {
        "benchmark_source": "synthetic",
        "benchmark_sha256": "12" * 32,
        "execution_identity": _identity(),
        "areas": {
            area: {
                "correct": correct,
                "total": 100,
                "violations": 30,
                "false_safe": 0,
                "errors": 0,
            }
            for area, correct in counts.items()
        },
    }
    return capabilities.validate_capability_profile(profile)


def _result(label="satisfies_requirement", state="state", model_id="expert-a"):
    scores = {
        "satisfies_requirement": 0.7,
        "violates_requirement": 0.2,
        "insufficient_context": 0.1,
    }
    if label != "satisfies_requirement":
        scores = {name: (0.7 if name == label else 0.15) for name in scores}
    return {
        "label": label,
        "probabilities": scores,
        "coverage": {
            "truncated": False,
            "state_tokens_dropped": 0,
            "truncated_questions": [],
        },
        "model_id": model_id,
        "execution_identity": _identity(),
        "state_sha256": text_digest(state),
    }


def test_capability_profile_round_trip(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_profile("expert-a", ["fifo"])))
    loaded = capabilities.load_capability_profile(str(path))
    assert loaded["id"] == "expert-a"
    assert loaded["supported_mechanisms"] == ["fifo"]


def test_capability_profile_rejects_bad_checksum(tmp_path):
    path = tmp_path / "profile.json"
    payload = _profile("expert-a", ["fifo"], checkpoint_sha256="zz")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        capabilities.load_capability_profile(str(path))


def test_router_prefers_qualified_and_flags_unsupported():
    first = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo", "counter"])
    )
    second = capabilities.validate_capability_profile(
        _profile("expert-b", ["fifo"])
    )
    decision = route_scope(["fifo", "counter"], [first, second])
    assert decision["experts"] == ["expert-a"]
    assert decision["unsupported"] == []
    lonely = route_scope(["handshake"], [first, second])
    assert lonely["experts"] == []
    assert lonely["unsupported"] == ["handshake"]
    assert route_scope([], [first])["experts"] == []


def test_router_ranks_by_tier_and_flags_thin():
    strong = _rated(
        "contract-1", ["local-safety-cwe"], {"local-safety-cwe": 90}
    )
    weak = _rated("contract-2", ["local-safety-cwe"], {"local-safety-cwe": 30})
    decision = route_scope(["local-safety-cwe"], [weak, strong])
    assert decision["experts"] == ["contract-1", "contract-2"]
    assert decision["tiers"] == {
        "contract-2": "none",
        "contract-1": "strong",
    }
    assert decision["thin_evidence"] is False
    assert any("local-safety-cwe" in line for line in decision["rationale"])
    thin = route_scope(["guard-semantics"], [weak])
    assert thin["experts"] == []
    assert thin["thin_evidence"] is True
    weak_only = _rated(
        "contract-2", ["guard-semantics"], {"guard-semantics": 50}
    )
    rated = route_scope(["guard-semantics"], [weak_only])
    assert rated["tiers"] == {"contract-2": "weak"}
    assert rated["thin_evidence"] is True


def test_router_mixed_scope_uses_worst_tier():
    both = _rated(
        "contract-1",
        ["fifo", "local-safety-cwe"],
        {"fifo": 30, "local-safety-cwe": 90},
    )
    decision = route_scope(["fifo", "local-safety-cwe"], [both])
    assert decision["tiers"] == {"contract-1": "none"}
    assert decision["area_tiers"] == {
        "contract-1": {"fifo": "none", "local-safety-cwe": "strong"}
    }
    assert decision["thin_evidence"] is True
    assert any("scope tier: none" in line for line in decision["rationale"])


def test_router_stale_weights_unrate():
    upgraded = capabilities.validate_capability_profile(
        _profile("contract-1", ["local-safety-cwe"])
    )
    decision = route_scope(["local-safety-cwe"], [upgraded])
    assert decision["tiers"] == {"contract-1": "unrated"}
    assert decision["thin_evidence"] is True


def test_response_rejects_foreign_worker_model():
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )
    payload = _result()
    payload["model_id"] = "foreign-worker-id"
    with pytest.raises(ValueError):
        responses.build_expert_response(profile, "state", payload)
    payload["model_id"] = "expert-a"
    assert (
        responses.build_expert_response(profile, "state", payload)["label"]
        == "satisfies_requirement"
    )


def test_router_reports_partial_union_without_consulting():
    fifo_only = capabilities.validate_capability_profile(
        _profile("contract-2", ["fifo"])
    )
    local_only = capabilities.validate_capability_profile(
        _profile("contract-1", ["local-safety-cwe"])
    )
    decision = route_scope(
        ["fifo", "local-safety-cwe"], [fifo_only, local_only]
    )
    assert decision["experts"] == []
    assert decision["unsupported"] == []
    assert decision["coverage"] == {
        "fifo": ["contract-2"],
        "local-safety-cwe": ["contract-1"],
    }
    assert {item["expert_id"] for item in decision["partial"]} == {
        "contract-1",
        "contract-2",
    }
    assert decision["thin_evidence"] is True


def test_expert_response_preserves_identity_and_input():
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )
    state = "synthetic state text"
    response = responses.build_expert_response(
        profile, state, _result(state=state)
    )
    assert response["expert_id"] == "expert-a"
    assert response["label"] == "satisfies_requirement"
    assert response["scores"]["satisfies_requirement"] == pytest.approx(0.7)
    assert len(response["state_sha256"]) == 64
    with pytest.raises(ValueError):
        responses.build_expert_response(profile, state, {"verdict": "nope"})


def test_consult_experts_separates_errors():
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )

    def transport(expert_id, state):
        assert state == "hello"
        if expert_id == "expert-a":
            return _result("violates_requirement", state=state)
        raise RuntimeError("worker down")

    replies, errors = responses.consult_experts(
        ["expert-a", "expert-b"], {"expert-a": profile}, "hello", transport
    )
    assert [item["label"] for item in replies] == ["violates_requirement"]
    assert [item["expert_id"] for item in errors] == ["expert-b"]
    assert errors[0]["stage"] == "route"
    with pytest.raises(ValueError):
        responses.consult_experts(
            ["expert-a"], {"expert-a": profile}, "  ", transport
        )


def test_service_routes_and_consults(tmp_path):
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile("expert-a", ["fifo"])))
    registry_path = tmp_path / "models.json"
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    laya = tmp_path / "laya"
    (laya / "laya").mkdir(parents=True)
    (laya / "laya" / "__init__.py").write_text("")
    registry_path.write_text(
        json.dumps(
            {
                "schema": GUI_MODEL_REGISTRY_SCHEMA,
                "models": [
                    {
                        "id": "expert-a",
                        "display_name": "Expert A",
                        "checkpoint": str(tmp_path),
                        "python": str(python),
                        "laya_root": str(laya),
                        "expected_files": {"model.safetensors": "ab" * 32},
                        "device": "cpu",
                        "profile": "profile.json",
                        "release": "synthetic-1",
                        "question_contract": "siliconshield-requirement-context-v1",
                        "qualification": "synthetic run",
                    }
                ],
            }
        )
    )
    service = InferenceService(str(registry_path))
    assert service.expert_ids == ["expert-a"]
    assessment = service.assess(
        ["fifo"], "synthetic state", lambda _eid, _st: _result(state=_st)
    )
    assert assessment["routing"]["experts"] == ["expert-a"]
    assert len(assessment["responses"]) == 1
    assert assessment["errors"] == []
    assert assessment["request_id"]


def _write_registry(tmp_path, profile_payload, entry_overrides):
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    (tmp_path / "profile.json").write_text(json.dumps(profile_payload))
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    laya = tmp_path / "laya"
    (laya / "laya").mkdir(parents=True, exist_ok=True)
    (laya / "laya" / "__init__.py").write_text("")
    entry = {
        "id": "expert-a",
        "display_name": "Expert A",
        "checkpoint": str(tmp_path),
        "python": str(python),
        "laya_root": str(laya),
        "expected_files": {"model.safetensors": "ab" * 32},
        "device": "cpu",
        "profile": "profile.json",
    }
    entry.update(entry_overrides)
    registry_path = tmp_path / "models.json"
    registry_path.write_text(
        json.dumps({"schema": GUI_MODEL_REGISTRY_SCHEMA, "models": [entry]})
    )
    return str(registry_path)


def test_registry_rejects_identity_mismatch(tmp_path):
    from siliconshield.experts.registry import load_registry

    good = _profile("expert-a", ["fifo"])
    load_registry(_write_registry(tmp_path, good, {}))
    wrong_weights = _profile("expert-a", ["fifo"], checkpoint_sha256="ef" * 32)
    with pytest.raises(ValueError):
        load_registry(_write_registry(tmp_path, wrong_weights, {}))
    with pytest.raises(ValueError):
        load_registry(_write_registry(tmp_path, good, {"release": "other"}))
    with pytest.raises(ValueError):
        load_registry(
            _write_registry(tmp_path, good, {"question_contract": "other"})
        )


def test_empty_coverage_is_rejected():
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )
    payload = _result()
    payload["coverage"] = {}
    with pytest.raises(ValueError):
        responses.build_expert_response(profile, "state", payload)
    payload["coverage"] = {"truncated": False}
    with pytest.raises(ValueError):
        responses.build_expert_response(profile, "state", payload)


def test_inspection_maps_structure():
    rtl = (
        "module m(input logic clk, rst_n, wr_en,\n"
        "  input logic [7:0] wdata,\n"
        "  output logic [7:0] q);\n"
        "  logic [7:0] q;\n"
        "  always_ff @(posedge clk) begin\n"
        "    if (!rst_n) q <= 8'h00;\n"
        "    else if (wr_en) q <= wdata;\n"
        "  end\n"
        "endmodule\n"
    )
    known = ["clk", "rst_n", "wr_en", "wdata", "q", "m"]
    structure = inspection.inspect_structure(rtl, known)
    assert structure["writers"]["q"] == [6, 7]
    assert "!rst_n" in structure["guards"]
    assert "q" in structure["state"]
    assert structure["unknown_symbols"] == []


def test_inspection_flags_unknown_symbols():
    structure = inspection.inspect_structure("assign q = mystery;\n", ["q"])
    assert structure["unknown_symbols"] == ["mystery"]


def test_finding_labels_model_assessment():
    assessment = {
        "responses": [
            {"expert_id": "expert-a", "label": "violates_requirement"}
        ],
        "errors": [{"expert_id": "expert-b", "stage": "transport"}],
    }
    verdict = {
        "conclusion": "violates_requirement",
        "explanation": "synthetic",
        "open_questions": ["confirm reset scope"],
    }
    assessment.update(
        request_id="request", state_sha256="ab" * 32, scope_sha256="cd" * 32
    )
    verdict.update(
        {
            key: assessment[key]
            for key in ("request_id", "state_sha256", "scope_sha256")
        }
    )
    verdict["requirement_sha256"] = text_digest("writes need enable")
    finding = findings.build_finding(
        "possible bypass",
        "security: unauthorized write",
        "writes need enable",
        ["top.sv:6"],
        assessment,
        verdict,
    )
    assert finding["assessment_kind"] == "model_assessment"
    assert finding["expert_labels"] == {"expert-a": "violates_requirement"}
    assert any("simulation" in item for item in finding["open_questions"])
    with pytest.raises(ValueError):
        findings.build_finding(" ", "r", "q", [], assessment, verdict)


def _arbitration_payload(**overrides):
    payload = {
        "conclusion": "violates_requirement",
        "explanation": "synthetic arbitration",
        "evidence": [
            {"expert_id": "expert-a", "relevance": "covers the guard"}
        ],
        "open_questions": [],
    }
    payload.update(overrides)
    return payload


_STATE = "REQUIREMENT\nsynthetic requirement\nRTL\nassign q = d;\n"


def _assessment(items, errors=None, state=_STATE):
    profiles = {
        item["expert_id"]: capabilities.validate_capability_profile(
            _profile(item["expert_id"], ["fifo"])
        )
        for item in items
    }
    replies = []
    for item in items:
        response = responses.build_expert_response(
            profiles[item["expert_id"]],
            state,
            _result(state=state, model_id=item["expert_id"]),
        )
        response.update(
            assessment_id="synthetic-request", scope_sha256=digest(["fifo"])
        )
        replies.append(response)
    return {
        "request_id": "synthetic-request",
        "state_sha256": text_digest(state),
        "scope": ["fifo"],
        "scope_sha256": digest(["fifo"]),
        "routing": route_scope(
            ["fifo"],
            list(profiles.values()),
            {key: _identity() for key in profiles},
        ),
        "responses": replies,
        "errors": errors or [],
    }


def _arb_profiles():
    return {
        "expert-a": capabilities.validate_capability_profile(
            _profile("expert-a", ["fifo"])
        )
    }


def test_validate_arbitration_accepts_and_rejects():
    checked = arbitration.validate_arbitration(
        _arbitration_payload(), ["expert-a"]
    )
    assert checked["conclusion"] == "violates_requirement"
    with pytest.raises(ValueError):
        arbitration.validate_arbitration(
            _arbitration_payload(conclusion="maybe")
        )
    with pytest.raises(ValueError):
        arbitration.validate_arbitration(_arbitration_payload(explanation="  "))
    with pytest.raises(ValueError):
        arbitration.validate_arbitration(_arbitration_payload(evidence=[]))
    with pytest.raises(ValueError):
        arbitration.validate_arbitration(
            _arbitration_payload(open_questions=["ok", 7])
        )
    with pytest.raises(ValueError):
        arbitration.validate_arbitration(_arbitration_payload(), ["expert-b"])


def test_arbitrate_assessment_uses_tool_call():
    state = "REQUIREMENT\nsynthetic requirement\nRTL\nassign q = d;\n"

    class FakeClient:
        def complete(self, messages, tools):
            assert tools[0]["function"]["name"] == "experts.arbitrate"
            bound = json.loads(messages[1]["content"])
            assert bound["state"] == state
            assert bound["request_id"] == "synthetic-request"
            return {
                "tool_calls": [
                    {
                        "function": {
                            "name": "experts.arbitrate",
                            "arguments": _arbitration_payload(),
                        }
                    }
                ]
            }

    verdict = arbitration.arbitrate_assessment(
        _assessment([{"expert_id": "expert-a"}]),
        state,
        "synthetic requirement",
        profiles=_arb_profiles(),
        client_factory=lambda: FakeClient(),
    )
    assert verdict["conclusion"] == "violates_requirement"

    class SilentClient:
        def complete(self, messages, tools):
            return {"tool_calls": []}

    with pytest.raises(ValueError):
        arbitration.arbitrate_assessment(
            _assessment([{"expert_id": "expert-a"}]),
            state,
            "synthetic requirement",
            profiles=_arb_profiles(),
            client_factory=lambda: SilentClient(),
        )


def test_arbitrate_assessment_binds_requirement_and_citations():
    state = "REQUIREMENT\nsynthetic requirement\nRTL\nassign q = d;\n"
    with pytest.raises(ValueError):
        arbitration.build_arbitration_input(
            _assessment([{"expert_id": "expert-a"}]),
            state,
            "a different requirement",
        )
    empty = arbitration.arbitrate_assessment(
        _assessment([], [{"expert_id": "expert-a", "stage": "transport"}]),
        state,
        "synthetic requirement",
        profiles=_arb_profiles(),
        client_factory=lambda: None,
    )
    assert empty["conclusion"] == "unresolved"
    assert empty["evidence"] == []


_FAKE_WORKER = """import json
import sys
from siliconshield.experts.identity import digest, runtime_digest, text_digest
from siliconshield.experts.contracts import PROTOCOL_VERSION
from siliconshield.training.laya.question_contracts import question_for
from test_multi_expert import _identity

from pathlib import Path

from siliconshield.experts.identity import digest, text_digest
from siliconshield.training.laya.question_contracts import question_for

for line in sys.stdin:
    try:
        command = json.loads(line)
    except ValueError:
        continue
    kind = command.get("type")
    if kind == "shutdown":
        break
    if kind == "load":
        entry = command["model"]
        identity = _identity()
        identity["export_sha256"] = digest(entry["expected_files"])
        identity["runtime_sha256"] = runtime_digest(entry["laya_root"])
        reply = {"type": "ready", "request_id": command.get("request_id"),
            "model_id": entry["id"], "device": entry["device"],
            "cpu_fallback_count": 0, "question_contract": entry["question_contract"]}
    elif kind == "analyze":
        reply = {
            "type": "result",
            "request_id": command.get("request_id"),
            "label": "violates_requirement",
            "probabilities": {
                "satisfies_requirement": 0.15,
                "violates_requirement": 0.7,
                "insufficient_context": 0.15,
            },
            "coverage": {"truncated": False, "state_tokens_dropped": 0, "truncated_questions": []},
        }
    else:
        reply = {
            "type": "error",
            "request_id": command.get("request_id"),
            "stage": "fake",
            "detail": "unknown command",
        }
    reply.update(protocol=PROTOCOL_VERSION, session_id="synthetic-worker", execution_identity=identity,
                 model_id=entry["id"], device=entry["device"])
    if kind == "analyze":
        reply["state_sha256"] = text_digest(command["state"])
    sys.stdout.write(json.dumps(reply) + "\\n")
    sys.stdout.flush()
"""


def _write_cli_registry(tmp_path):
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    (tmp_path / "fake_worker.py").write_text(_FAKE_WORKER)
    (tmp_path / "profile.json").write_text(
        json.dumps(_profile("expert-a", ["fifo"]))
    )
    laya = tmp_path / "laya"
    (laya / "laya").mkdir(parents=True)
    (laya / "laya" / "__init__.py").write_text("")
    registry_path = tmp_path / "models.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": GUI_MODEL_REGISTRY_SCHEMA,
                "models": [
                    {
                        "id": "expert-a",
                        "display_name": "Expert A",
                        "checkpoint": str(tmp_path),
                        "python": sys.executable,
                        "laya_root": str(laya),
                        "expected_files": {"model.safetensors": "ab" * 32},
                        "device": "cpu",
                        "profile": "profile.json",
                        "release": "synthetic-1",
                        "question_contract": "siliconshield-requirement-context-v1",
                        "qualification": "synthetic run",
                    }
                ],
            }
        )
    )
    return str(registry_path)


def test_worker_pool_round_trip(tmp_path):
    from siliconshield.experts.registry import load_registry

    registry_path = _write_cli_registry(tmp_path)
    entry = load_registry(registry_path)[0]
    pool = WorkerPool(
        worker_module="fake_worker",
        extra_env={
            "PYTHONPATH": ":".join(
                (
                    str(tmp_path),
                    str(Path(__file__).parent),
                    str(Path(__file__).parents[1]),
                )
            )
        },
    )
    with pool:
        pool.load(entry)
        payload = pool.analyze("expert-a", "synthetic state")
    assert payload["label"] == "violates_requirement"
    assert payload["probabilities"]["violates_requirement"] == 0.7


def test_cli_assess_end_to_end(tmp_path, capsys):
    from siliconshield.experts.cli import main

    registry_path = _write_cli_registry(tmp_path)
    state_path = tmp_path / "state.txt"
    state_path.write_text("synthetic state")

    def factory():
        return WorkerPool(
            worker_module="fake_worker",
            extra_env={
                "PYTHONPATH": ":".join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        str(Path(__file__).parents[1]),
                    )
                )
            },
        )

    args = build_parser().parse_args(
        [
            "assess",
            "--registry",
            registry_path,
            "--scope",
            "fifo",
            "--state-file",
            str(state_path),
        ]
    )
    report = run_assess(args, pool_factory=factory)
    assert report["assessment"]["routing"]["experts"] == ["expert-a"]
    assert report["assessment"]["responses"][0]["label"] == (
        "violates_requirement"
    )
    with pytest.raises(SystemExit) as exited:
        main(["assess", "--help"])
    assert exited.value.code == 0
    capsys.readouterr()
