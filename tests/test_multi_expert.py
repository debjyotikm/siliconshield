"""Synthetic tests for multi-expert routing and envelopes."""

import copy
import json
import os
import queue
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from siliconshield.experts import (
    arbitration,
    capabilities,
    contracts,
    findings,
    inspection,
    registry,
    responses,
    routing_data,
)
from siliconshield.experts.cli import build_parser, run_assess
from siliconshield.experts.identity import (
    digest,
    text_digest,
    validate_execution_identity,
)
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
    assert empty["evidence"] == [
        {
            "expert_id": "expert-a",
            "relevance": "Consultation failed at stage transport.",
        }
    ]
    assert empty["agreement"] is False


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


def test_cli_finding_flags_are_gated_before_consult(tmp_path):
    from siliconshield.experts.cli import build_parser, run_assess

    state_path = tmp_path / "state.txt"
    state_path.write_text("synthetic state")
    base = [
        "assess",
        "--registry",
        str(tmp_path / "missing.json"),
        "--scope",
        "fifo",
        "--state-file",
        str(state_path),
    ]
    args = build_parser().parse_args(base + ["--issue", "x"])
    with pytest.raises(ValueError, match="required together"):
        run_assess(args)
    args = build_parser().parse_args(
        base + ["--issue", "x", "--relevance", "y"]
    )
    with pytest.raises(ValueError, match="need --arbitrate"):
        run_assess(args)
    args = build_parser().parse_args(base + ["--arbitrate"])
    with pytest.raises(ValueError, match="needs --requirement"):
        run_assess(args)


def test_worker_load_rejects_non_object_entries():
    from siliconshield.experts.worker import WorkerLoadError, WorkerPool

    pool = WorkerPool()
    try:
        with pytest.raises(WorkerLoadError, match="not an object"):
            pool.load(["not-an-entry"])  # type: ignore[arg-type]
        with pytest.raises(WorkerLoadError, match="names no expert"):
            pool.load({})
    finally:
        pool.close()


def _two_expert_registry(tmp_path, with_identity=True):
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    laya = tmp_path / "laya"
    (laya / "laya").mkdir(parents=True, exist_ok=True)
    (laya / "laya" / "__init__.py").write_text("")
    models = []
    for expert_id in ("expert-a", "expert-b"):
        entry = {
            "id": expert_id,
            "display_name": expert_id,
            "checkpoint": str(tmp_path),
            "python": str(python),
            "laya_root": str(laya),
            "expected_files": {"model.safetensors": "ab" * 32},
            "device": "cpu",
        }
        if with_identity:
            entry["release"] = "synthetic-1"
            entry["question_contract"] = "siliconshield-requirement-context-v1"
        models.append(entry)
    registry_path = tmp_path / "models.json"
    registry_path.write_text(
        json.dumps({"schema": GUI_MODEL_REGISTRY_SCHEMA, "models": models})
    )
    return str(registry_path)


def test_effective_routing_keeps_planned_coverage_after_failure(tmp_path):
    registry = _two_expert_registry(tmp_path)
    service = InferenceService(
        registry,
        profiles={
            "expert-a": _profile("expert-a", ["fifo", "counter"]),
            "expert-b": _profile("expert-b", ["fifo"]),
        },
    )

    def transport(expert_id, state):
        if expert_id == "expert-a":
            raise RuntimeError("synthetic transport failure")
        return _result(state=state, model_id=expert_id)

    assessment = service.assess(["fifo", "counter"], "state", transport)
    planned = assessment["planned_routing"]
    assert planned["experts"] == ["expert-a"]
    assert [item["expert_id"] for item in planned["partial"]] == ["expert-b"]
    routing = assessment["routing"]
    assert routing["unsupported"] == []
    assert routing["partial"] == planned["partial"]
    assert routing["coverage"] == planned["coverage"]
    assert any("expert-a failed" in line for line in routing["rationale"])


def test_cli_rejects_empty_scope(tmp_path):
    state_path = tmp_path / "state.txt"
    state_path.write_text("synthetic state")
    args = build_parser().parse_args(
        [
            "assess",
            "--registry",
            str(tmp_path / "missing.json"),
            "--scope",
            " , , ",
            "--state-file",
            str(state_path),
        ]
    )
    with pytest.raises(ValueError, match="at least one mechanism tag"):
        run_assess(args)


def test_whitespace_scope_tags_are_normalized():
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )
    decision = route_scope([" fifo "], [profile])
    assert decision["experts"] == ["expert-a"]
    assert decision["scope"] == ["fifo"]


def test_empty_scope_routes_nowhere_with_thin_evidence():
    decision = route_scope([], [])
    assert decision["experts"] == []
    assert decision["thin_evidence"] is True
    assert decision["rationale"] == ["empty scope routes nowhere."]


def test_area_coverage_satisfies_same_area_mechanism_tags():
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )
    decision = route_scope(["fifo", "stored-word-hold:pass0"], [profile])
    assert decision["experts"] == ["expert-a"]
    assert decision["unsupported"] == []


def test_transport_value_error_is_a_validate_stage_error():
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )

    def transport(expert_id, state):
        raise ValueError("synthetic input rejection")

    _, errors = responses.consult_experts(
        ["expert-a"], {"expert-a": profile}, "state", transport
    )
    assert errors[0]["stage"] == "validate"


def test_worker_keeps_ready_receipt_on_validation_failure(monkeypatch):
    pool = WorkerPool()
    pool._ready["expert-a"] = {
        "session_id": "synthetic-session",
        "model_id": "expert-a",
        "device": "cpu",
        "execution_identity": _identity(),
    }
    box: queue.Queue = queue.Queue()
    pool._queues["expert-a"] = box
    pool._workers["expert-a"] = SimpleNamespace(
        poll=lambda: None,
        stdin=SimpleNamespace(write=lambda text: None, flush=lambda: None),
    )
    monkeypatch.setattr(
        "siliconshield.experts.worker.uuid.uuid4",
        lambda: SimpleNamespace(hex="fixed-request"),
    )
    reply = _result(state="state")
    reply.update(
        type="result",
        request_id="fixed-request",
        protocol=contracts.PROTOCOL_VERSION,
        session_id="synthetic-session",
        device="cpu",
    )
    reply["coverage"]["truncated"] = True
    box.put(json.dumps(reply))
    with pytest.raises(ValueError, match="truncated"):
        pool.analyze("expert-a", "state")
    assert "expert-a" in pool._ready


def test_worker_rejects_malformed_stdout_loudly():
    pool = WorkerPool()
    pool._ready["expert-a"] = {"session_id": "synthetic-session"}
    box: queue.Queue = queue.Queue()
    pool._queues["expert-a"] = box
    box.put("this is not json")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        pool._wait("expert-a", "request-1", ("result",), 5.0, "analyze")


def test_worker_rejects_wrong_kind_messages():
    pool = WorkerPool()
    pool._ready["expert-a"] = {"session_id": "synthetic-session"}
    box: queue.Queue = queue.Queue()
    pool._queues["expert-a"] = box
    box.put(
        json.dumps(
            {
                "type": "ready",
                "request_id": "request-1",
                "protocol": contracts.PROTOCOL_VERSION,
                "session_id": "synthetic-session",
            }
        )
    )
    with pytest.raises(RuntimeError, match="unexpected ready"):
        pool._wait("expert-a", "request-1", ("result",), 5.0, "analyze")


def test_injected_profiles_skip_unstated_registry_identity(tmp_path):
    registry = _two_expert_registry(tmp_path, with_identity=False)
    service = InferenceService(
        registry,
        profiles={
            "expert-a": _profile("expert-a", ["fifo"]),
            "expert-b": _profile("expert-b", ["counter"]),
        },
    )
    assert service.expert_ids == ["expert-a", "expert-b"]


def test_result_rejects_unknown_probability_labels():
    payload = _result()
    payload["probabilities"]["abstain"] = 0.0
    with pytest.raises(ValueError, match="unknown labels"):
        contracts.validate_result(payload)


def _scripted_client(payload):
    class _Scripted:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools=()):
            self.calls += 1
            return {
                "tool_calls": [
                    {
                        "function": {
                            "name": "experts.arbitrate",
                            "arguments": payload,
                        }
                    }
                ]
            }

    return _Scripted()


def test_malformed_arbitration_replies_raise_value_error():
    assessment = _assessment([{"expert_id": "expert-a"}])

    def attempt(reply):
        class _Client:
            def complete(self, messages, tools=()):
                return reply

        with pytest.raises(ValueError, match="(?i)arbitration"):
            arbitration.arbitrate_assessment(
                assessment,
                _STATE,
                "synthetic requirement",
                profiles=_arb_profiles(),
                client_factory=_Client,
            )

    attempt({"content": "x", "tool_calls": None})
    attempt({"content": "x", "tool_calls": "experts.arbitrate"})
    attempt({"content": "x", "tool_calls": ["experts.arbitrate"]})
    attempt([])
    attempt("x")
    attempt(None)


def test_usable_responses_arbitrate_despite_other_errors():
    assessment = _assessment(
        [{"expert_id": "expert-a"}],
        [{"expert_id": "expert-b", "stage": "transport"}],
    )
    assert assessment["routing"]["thin_evidence"] is True
    client = _scripted_client(_arbitration_payload())
    verdict = arbitration.arbitrate_assessment(
        assessment,
        _STATE,
        "synthetic requirement",
        profiles=_arb_profiles(),
        client_factory=lambda: client,
    )
    assert client.calls == 1
    assert verdict["conclusion"] == "violates_requirement"
    assert verdict["agreement"] is True


def test_direct_unresolved_verdict_round_trips_the_validator():
    assessment = _assessment(
        [], [{"expert_id": "expert-a", "stage": "transport"}]
    )
    verdict = arbitration.arbitrate_assessment(
        assessment,
        _STATE,
        "synthetic requirement",
        profiles=_arb_profiles(),
        client_factory=lambda: None,
    )
    payload = {
        key: verdict[key]
        for key in ("conclusion", "explanation", "evidence", "open_questions")
    }
    checked = arbitration.validate_arbitration(payload, ["expert-a"])
    assert checked["evidence"][0]["expert_id"] == "expert-a"


def test_arbitration_validator_rejects_weak_and_unknown_content():
    good = _arbitration_payload()
    with pytest.raises(ValueError, match="relevance"):
        arbitration.validate_arbitration(
            _arbitration_payload(
                evidence=[{"expert_id": "expert-a", "relevance": "   "}]
            ),
            ["expert-a"],
        )
    with pytest.raises(ValueError, match="open_questions"):
        arbitration.validate_arbitration(
            _arbitration_payload(open_questions=["", "  "]), ["expert-a"]
        )
    with pytest.raises(ValueError, match="more than once"):
        arbitration.validate_arbitration(
            _arbitration_payload(
                evidence=[
                    {"expert_id": "expert-a", "relevance": "first"},
                    {"expert_id": "expert-a", "relevance": "second"},
                ]
            ),
            ["expert-a"],
        )
    with pytest.raises(ValueError, match="unknown keys"):
        arbitration.validate_arbitration(
            _arbitration_payload(extra_key="dropped before"), ["expert-a"]
        )
    with pytest.raises(ValueError, match="not consulted"):
        arbitration.validate_arbitration(
            _arbitration_payload(
                evidence=[{"expert_id": "ghost", "relevance": "invented"}]
            ),
            ["expert-a"],
        )
    with pytest.raises(TypeError):
        arbitration.validate_arbitration(good)  # consulted is required
    checked = arbitration.validate_arbitration(dict(good), ["expert-a"])
    assert checked["conclusion"] == "violates_requirement"


def test_only_responders_reach_the_bound_model_input():
    assessment = _assessment([{"expert_id": "expert-a"}])
    profiles = dict(_arb_profiles())
    profiles["ghost"] = {"id": "ghost"}
    bound = arbitration.build_arbitration_input(
        assessment, _STATE, "synthetic requirement", profiles
    )
    assert set(bound["profiles"]) == {"expert-a"}


def test_string_tool_arguments_arbitrate_successfully():
    assessment = _assessment([{"expert_id": "expert-a"}])
    client = _scripted_client(json.dumps(_arbitration_payload()))
    verdict = arbitration.arbitrate_assessment(
        assessment,
        _STATE,
        "synthetic requirement",
        profiles=_arb_profiles(),
        client_factory=lambda: client,
    )
    assert verdict["conclusion"] == "violates_requirement"


def test_arbitration_errors_name_the_client_and_keep_content():
    assessment = _assessment([{"expert_id": "expert-a"}])

    class _Silent:
        def complete(self, messages, tools=()):
            return {"content": "synthetic refusal text"}

    with pytest.raises(ValueError, match="synthetic refusal text"):
        arbitration.arbitrate_assessment(
            assessment,
            _STATE,
            "synthetic requirement",
            profiles=_arb_profiles(),
            client_factory=_Silent,
        )


def test_split_experts_report_disagreement():
    assessment = _assessment(
        [{"expert_id": "expert-a"}, {"expert_id": "expert-b"}]
    )
    assessment["responses"][1]["label"] = "violates_requirement"
    profiles = {
        "expert-a": capabilities.validate_capability_profile(
            _profile("expert-a", ["fifo"])
        ),
        "expert-b": capabilities.validate_capability_profile(
            _profile("expert-b", ["fifo"])
        ),
    }
    payload = _arbitration_payload(
        evidence=[
            {"expert_id": "expert-a", "relevance": "first"},
            {"expert_id": "expert-b", "relevance": "second"},
        ]
    )
    verdict = arbitration.arbitrate_assessment(
        assessment,
        _STATE,
        "synthetic requirement",
        profiles=profiles,
        client_factory=lambda: _scripted_client(payload),
    )
    assert verdict["agreement"] is False


def test_unanimous_insufficient_context_cannot_be_overridden():
    assessment = _assessment(
        [{"expert_id": "expert-a"}, {"expert_id": "expert-b"}]
    )
    for response in assessment["responses"]:
        response["label"] = "insufficient_context"
    profiles = {
        "expert-a": capabilities.validate_capability_profile(
            _profile("expert-a", ["fifo"])
        ),
        "expert-b": capabilities.validate_capability_profile(
            _profile("expert-b", ["fifo"])
        ),
    }
    with pytest.raises(ValueError, match="unanimous insufficient_context"):
        arbitration.arbitrate_assessment(
            assessment,
            _STATE,
            "synthetic requirement",
            profiles=profiles,
            client_factory=lambda: _scripted_client(_arbitration_payload()),
        )
    allowed = arbitration.arbitrate_assessment(
        assessment,
        _STATE,
        "synthetic requirement",
        profiles=profiles,
        client_factory=lambda: _scripted_client(
            _arbitration_payload(conclusion="insufficient_context")
        ),
    )
    assert allowed["conclusion"] == "insufficient_context"


def _finding_inputs():
    assessment = _assessment(
        [{"expert_id": "expert-a"}],
        [{"expert_id": "expert-b", "stage": "transport"}],
    )
    verdict = {
        "conclusion": "violates_requirement",
        "explanation": "synthetic explanation",
        "evidence": [{"expert_id": "expert-a", "relevance": "covers it"}],
        "open_questions": ["synthetic question"],
        "agreement": True,
        "request_id": "synthetic-request",
        "state_sha256": text_digest(_STATE),
        "scope_sha256": digest(["fifo"]),
        "requirement_sha256": text_digest("synthetic requirement"),
    }
    return assessment, verdict


def test_trailing_newlines_rejected_in_digest_and_device(tmp_path):
    from siliconshield.schemas import EXPERT_CAPABILITY_SCHEMA

    padded = _profile("expert-a", ["fifo"])
    padded["schema"] = EXPERT_CAPABILITY_SCHEMA
    padded["checkpoint_sha256"] = "ab" * 32 + "\n"
    stored = capabilities.validate_capability_profile(padded)
    assert stored["checkpoint_sha256"] == "ab" * 32
    embedded = _profile("expert-a", ["fifo"])
    embedded["schema"] = EXPERT_CAPABILITY_SCHEMA
    embedded["checkpoint_sha256"] = "ab" * 31 + "\n" + "ab"
    with pytest.raises(ValueError, match="malformed checkpoint"):
        capabilities.validate_capability_profile(embedded)
    registry_path = _write_registry(
        tmp_path, _profile("expert-a", ["fifo"]), {}
    )
    data = json.loads(Path(registry_path).read_text())
    data["models"][0]["device"] = "cpu\n"
    Path(registry_path).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="device"):
        registry.load_registry(registry_path)


def test_padded_mechanisms_normalize_and_match():
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", [" fifo "])
    )
    assert profile["supported_mechanisms"] == ["fifo"]
    rated = _rated("expert-a", ["fifo"], {"fifo": 90})
    decision = route_scope(["fifo"], [rated])
    assert decision["experts"] == ["expert-a"]


def test_padded_registry_id_and_empty_filename_rejected(tmp_path):
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    registry_path = _write_registry(
        tmp_path, _profile("expert-a", ["fifo"]), {}
    )
    data = json.loads(Path(registry_path).read_text())
    data["models"][0]["id"] = "expert-a "
    Path(registry_path).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="must be exact"):
        registry.load_registry(registry_path)
    data["models"][0]["id"] = "expert-a"
    data["models"][0]["expected_files"] = {"": "ab" * 32}
    Path(registry_path).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="empty expected filename"):
        registry.load_registry(registry_path)
    assert GUI_MODEL_REGISTRY_SCHEMA


def test_unknown_routing_area_rejected():
    profile = _rated("expert-a", ["fifo"], {"fifo": 90})
    evidence = dict(profile["routing_evidence"])
    evidence["areas"] = {
        "bogus-area": {
            "correct": 1,
            "total": 1,
            "violations": 1,
            "false_safe": 0,
            "errors": 0,
        }
    }
    with pytest.raises(ValueError, match="unknown routing evidence area"):
        capabilities.validate_routing_evidence(evidence, profile)


def test_qualification_mismatch_rejected(tmp_path):
    registry_path = _write_registry(
        tmp_path, _profile("expert-a", ["fifo"]), {}
    )
    data = json.loads(Path(registry_path).read_text())
    data["models"][0]["qualification"] = "other run"
    Path(registry_path).write_text(json.dumps(data))
    service = InferenceService(registry_path)
    assert service.expert_ids == []
    assert "differs from registry" in service.broken_experts["expert-a"]
    clean = _two_expert_registry(tmp_path)
    stored = json.loads(Path(clean).read_text())
    stored["models"][0]["qualification"] = "registry run"
    Path(clean).write_text(json.dumps(stored))
    mismatched = _profile("expert-a", ["fifo"])
    mismatched["qualification_ref"] = "profile run"
    with pytest.raises(ValueError, match="differs from registry"):
        InferenceService(clean, profiles={"expert-a": mismatched})


def test_profile_mismatch_names_the_field_and_fix(tmp_path):
    clean = _two_expert_registry(tmp_path)
    stored = json.loads(Path(clean).read_text())
    stored["models"][0]["qualification"] = "registry run"
    Path(clean).write_text(json.dumps(stored))
    mismatched = _profile("expert-a", ["fifo"])
    mismatched["qualification_ref"] = "profile run"
    with pytest.raises(ValueError) as failed:
        InferenceService(clean, profiles={"expert-a": mismatched})
    message = str(failed.value)
    assert "qualification" in message
    assert "registry run" in message
    assert "profile run" in message
    assert "backfill" in message


def test_scope_sentence_names_worst_tier_and_best_expert_tier():
    strong = _rated("expert-a", ["fifo"], {"fifo": 90})
    poor = _rated("expert-b", ["fifo"], {"fifo": 30})
    decision = route_scope(["fifo"], [poor, strong])
    assert decision["tiers"] == {"expert-a": "strong", "expert-b": "none"}
    assert decision["experts"] == ["expert-a", "expert-b"]
    assert decision["rationale"][-1] == (
        "scope tier: worst consulted none, best consulted expert-a (strong)."
    )


def test_injected_bogus_schema_rejected(tmp_path):
    clean = _two_expert_registry(tmp_path)
    bad = _profile("expert-a", ["fifo"])
    bad["schema"] = "bogus-schema-v99"
    with pytest.raises(ValueError, match="unsupported schema"):
        InferenceService(clean, profiles={"expert-a": bad})


def test_injected_profile_needs_stated_weights(tmp_path):
    clean = _two_expert_registry(tmp_path)
    stored = json.loads(Path(clean).read_text())
    stored["models"][0]["expected_files"] = {"other.bin": "ab" * 32}
    Path(clean).write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="must state model.safetensors"):
        InferenceService(
            clean, profiles={"expert-a": _profile("expert-a", ["fifo"])}
        )


def test_execution_identity_wraps_non_json_settings():
    identity = dict(_identity())
    identity["extra"] = {1, 2}
    with pytest.raises(ValueError, match="non-JSON settings"):
        validate_execution_identity(identity)
    padded = copy.deepcopy(_identity())
    padded["inference"]["device_type"] = "cpu "
    with pytest.raises(ValueError, match="must be exact"):
        validate_execution_identity(padded)


def test_validated_evidence_detaches_from_caller_input():
    profile = _rated("expert-a", ["fifo"], {"fifo": 90})
    raw = {
        "execution_identity": dict(profile["routing_evidence"]),
        "benchmark_sha256": "ab" * 32,
        "benchmark_source": "synthetic",
        "areas": {
            "fifo": {
                "correct": 1,
                "total": 1,
                "violations": 1,
                "false_safe": 0,
                "errors": 0,
            }
        },
    }
    raw["execution_identity"] = dict(_identity())
    checked = capabilities.validate_routing_evidence(raw, profile)
    raw["areas"]["fifo"]["correct"] = 999
    assert checked["areas"]["fifo"]["correct"] == 1


def test_zero_violation_evidence_stays_unrated():
    from siliconshield.experts.router import _area_tier

    profile = _rated("expert-a", ["fifo"], {"fifo": 90})
    evidence = dict(profile["routing_evidence"])
    areas = {
        "fifo": {
            "correct": 100,
            "total": 100,
            "violations": 0,
            "false_safe": 0,
            "errors": 0,
        }
    }
    evidence["areas"] = areas
    profile["routing_evidence"] = capabilities.validate_routing_evidence(
        evidence, profile
    )
    assert _area_tier(profile, "fifo") == "unrated"


def test_profiles_for_skips_unknown_ids_by_contract(tmp_path):
    service = InferenceService(
        _write_registry(tmp_path, _profile("expert-a", ["fifo"]), {})
    )
    assert service.profiles_for(["nope"]) == {}
    with pytest.raises(KeyError):
        service.model_entry("nope")


def test_benchmark_area_set_is_single_sourced():
    assert routing_data.area_for_tag("fifo") == "fifo"
    assert routing_data.area_for_tag("stored-word-hold:pass0") == "fifo"
    assert routing_data.area_for_tag("privilege") is None
    assert set(routing_data.MECHANISM_TO_AREA.values()) <= set(
        routing_data.BENCHMARK_AREAS
    )


def test_casex_casez_need_parser_review():
    flagged = inspection.inspect_structure("casex (sel) a = 1;", [])
    plain = inspection.inspect_structure("assign a = 1;", [])
    assert len(flagged["limitations"]) == len(plain["limitations"]) + 1


def test_direction_ports_are_not_state():
    rtl = (
        "module m(input logic [7:0] a, input logic b);\n"
        "reg internal;\n"
        "endmodule"
    )
    found = inspection.inspect_structure(rtl, [])
    assert found["state"] == ["internal"]


def test_unterminated_comment_masks_rest_of_file():
    rtl = "module m;\n/* unterminated\nassign q = 1;\nendmodule"
    found = inspection.inspect_structure(rtl, [])
    assert found["writers"] == {}
    assert "unterminated" not in found["unknown_symbols"]


def test_parenthesized_comparison_is_not_a_writer():
    rtl = "module m;\nassign y = (a <= b);\nq <= d;\nendmodule"
    found = inspection.inspect_structure(rtl, [])
    assert "a" not in found["writers"]
    assert set(found["writers"]) == {"y", "q"}


def test_finding_separates_failed_experts_and_validates_text():
    assessment, verdict = _finding_inputs()
    found = findings.build_finding(
        "synthetic issue",
        "synthetic relevance",
        "synthetic requirement",
        ["top.sv:1"],
        assessment,
        verdict,
    )
    assert found["experts_consulted"] == ["expert-a"]
    assert found["failed_experts"] == [
        {"expert_id": "expert-b", "stage": "transport"}
    ]
    assert found["agreement"] is True
    assert found["open_questions"][-1].startswith("Confirm by simulation")
    bad_explanation = dict(verdict, explanation=None)
    with pytest.raises(ValueError, match="explanation"):
        findings.build_finding(
            "synthetic issue",
            "synthetic relevance",
            "synthetic requirement",
            [],
            assessment,
            bad_explanation,
        )
    bad_questions = dict(verdict, open_questions=[None])
    with pytest.raises(ValueError, match="open questions"):
        findings.build_finding(
            "synthetic issue",
            "synthetic relevance",
            "synthetic requirement",
            [],
            assessment,
            bad_questions,
        )


def test_consult_stages_non_object_payload_as_validate():
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )
    for payload in (None, [], "nope", 42):
        responses_, errors = responses.consult_experts(
            ["expert-a"],
            {"expert-a": profile},
            "state",
            lambda expert_id, state: payload,
        )
        assert responses_ == []
        assert [item["stage"] for item in errors] == ["validate"]


def test_consult_dedupes_repeated_expert_ids():
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )
    seen = []

    def transport(expert_id, state):
        seen.append(expert_id)
        return _result(state=state, model_id=expert_id)

    responses_, _ = responses.consult_experts(
        ["expert-a", "expert-a"], {"expert-a": profile}, "state", transport
    )
    assert seen == ["expert-a"]
    assert [item["expert_id"] for item in responses_] == ["expert-a"]


def test_validated_identity_detaches_from_payload():
    payload = _result(state="state")
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )
    built = responses.build_expert_response(profile, "state", payload)
    payload["execution_identity"]["inference"]["batch_size"] = 99
    assert built["execution_identity"]["inference"]["batch_size"] == 1


def test_injected_profile_must_be_an_object(tmp_path):
    registry_path = _two_expert_registry(tmp_path)
    with pytest.raises(ValueError, match="must be an object"):
        InferenceService(registry_path, profiles={"expert-a": None})


def test_injected_profile_colliding_with_registry_file_rejected(tmp_path):
    registry_path = _write_registry(
        tmp_path, _profile("expert-a", ["fifo"]), {}
    )
    with pytest.raises(ValueError, match="collides"):
        InferenceService(
            registry_path,
            profiles={"expert-a": _profile("expert-a", ["fifo"])},
        )


def test_service_backfills_unstated_registry_identity(tmp_path):
    registry_path = _two_expert_registry(tmp_path, with_identity=False)
    service = InferenceService(
        registry_path, profiles={"expert-a": _profile("expert-a", ["fifo"])}
    )
    entry = service.model_entry("expert-a")
    assert entry["question_contract"] == "siliconshield-requirement-context-v1"
    assert entry["release"] == "synthetic-1"
    assert entry["qualification"] == "synthetic run"


def test_finding_rejects_duplicate_expert_responses():
    assessment, verdict = _finding_inputs()
    assessment["responses"].append(dict(assessment["responses"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        findings.build_finding(
            "synthetic issue",
            "synthetic relevance",
            "synthetic requirement",
            [],
            assessment,
            verdict,
        )


def test_finding_rejects_malformed_inputs():
    assessment, verdict = _finding_inputs()
    with pytest.raises(ValueError, match="suspected issue"):
        findings.build_finding(
            None,
            "synthetic relevance",
            "synthetic requirement",
            [],
            assessment,
            verdict,
        )
    with pytest.raises(ValueError, match="requirement judged"):
        findings.build_finding(
            "synthetic issue",
            "synthetic relevance",
            123,
            [],
            assessment,
            verdict,
        )
    with pytest.raises(ValueError, match="objects"):
        findings.build_finding(
            "synthetic issue",
            "synthetic relevance",
            "synthetic requirement",
            [],
            [],
            verdict,
        )
    with pytest.raises(ValueError, match="locations"):
        findings.build_finding(
            "synthetic issue",
            "synthetic relevance",
            "synthetic requirement",
            [123],
            assessment,
            verdict,
        )
    broken = dict(assessment, errors=[{"expert_id": "expert-b"}])
    with pytest.raises(ValueError, match="stage"):
        findings.build_finding(
            "synthetic issue",
            "synthetic relevance",
            "synthetic requirement",
            [],
            broken,
            verdict,
        )


def test_arbitration_input_rejects_malformed_shapes():
    state = _STATE
    good = _assessment([{"expert_id": "expert-a"}])
    with pytest.raises(ValueError, match="must be a list"):
        arbitration.build_arbitration_input(
            dict(good, responses={}),
            state,
            "synthetic requirement",
            _arb_profiles(),
        )
    broken = dict(good, responses=[{"label": "satisfies_requirement"}])
    with pytest.raises(ValueError, match="names no expert"):
        arbitration.build_arbitration_input(
            broken, state, "synthetic requirement", _arb_profiles()
        )
    with pytest.raises(ValueError, match="must be an object"):
        arbitration.build_arbitration_input(
            good, state, "synthetic requirement", {"expert-a": []}
        )
    stageless = dict(good, errors=[{"expert_id": "expert-b"}])
    with pytest.raises(ValueError, match="stage"):
        arbitration.build_arbitration_input(
            stageless, state, "synthetic requirement", _arb_profiles()
        )
    with pytest.raises(ValueError, match="state and requirement"):
        arbitration.build_arbitration_input(
            good, None, "synthetic requirement", _arb_profiles()
        )


def test_arbitration_rejects_extra_evidence_keys():
    payload = _arbitration_payload(
        evidence=[
            {
                "expert_id": "expert-a",
                "relevance": "covers it",
                "smuggled": "nope",
            }
        ]
    )
    with pytest.raises(ValueError, match="unknown keys"):
        arbitration.validate_arbitration(payload, ["expert-a"])


def test_unparenthesized_comparison_writes_nothing():
    rtl = (
        "module m(input a, input b, output out);\n"
        "assign out = a <= b;\n"
        "endmodule\n"
    )
    assert set(inspection.inspect_structure(rtl, []).get("writers", {})) == {
        "out"
    }
    rtl = (
        "module m(input clk, input d, output reg q);\n"
        "always_ff @(posedge clk) q <= d;\n"
        "endmodule\n"
    )
    assert set(inspection.inspect_structure(rtl, []).get("writers", {})) == {
        "q"
    }


def test_pool_load_rejects_malformed_entries():
    pool = WorkerPool()
    for entry in (None, [], {}, {"checkpoint": "x"}):
        with pytest.raises(RuntimeError, match="Cannot load"):
            pool.load(entry)


def test_pool_analyze_rejects_empty_state():
    pool = WorkerPool()
    with pytest.raises(ValueError, match="empty state"):
        pool.analyze("expert-a", "   ")


def test_pool_wait_rejects_non_object_messages():
    pool = WorkerPool()
    pool._ready["expert-a"] = {"session_id": "synthetic-session"}
    box: queue.Queue = queue.Queue()
    pool._queues["expert-a"] = box
    box.put("[]")
    with pytest.raises(RuntimeError, match="non-object"):
        pool._wait("expert-a", "request-1", ("result",), 5.0, "analyze")


def test_pool_close_reaps_every_child():
    pool = WorkerPool()
    reaped = []

    class Stuck:
        stdin = None

        def poll(self):
            return None

        def terminate(self):
            reaped.append("stuck")

        def kill(self):
            reaped.append("killed")

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("cmd", timeout)

    class Fine:
        stdin = None

        def poll(self):
            return 0

    pool._workers["stuck"] = Stuck()
    pool._workers["fine"] = Fine()
    pool.close()
    assert pool._workers == {}
    assert reaped == ["stuck", "killed"]


def _two_file_registry(tmp_path, corrupt_broken):
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    (tmp_path / "good.json").write_text(
        json.dumps(_profile("expert-a", ["fifo"]))
    )
    (tmp_path / "broken.json").write_text(corrupt_broken)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    laya = tmp_path / "laya"
    (laya / "laya").mkdir(parents=True, exist_ok=True)
    (laya / "laya" / "__init__.py").write_text("")
    models = []
    for expert_id, profile in (
        ("expert-a", "good.json"),
        ("expert-b", "broken.json"),
    ):
        models.append(
            {
                "id": expert_id,
                "display_name": expert_id,
                "checkpoint": str(tmp_path),
                "python": str(python),
                "laya_root": str(laya),
                "expected_files": {"model.safetensors": "ab" * 32},
                "device": "cpu",
                "profile": profile,
            }
        )
    registry_path = tmp_path / "models.json"
    registry_path.write_text(
        json.dumps({"schema": GUI_MODEL_REGISTRY_SCHEMA, "models": models})
    )
    return str(registry_path)


def test_broken_stored_profile_does_not_block_healthy_scope(tmp_path):
    registry_path = _two_file_registry(tmp_path, "{not json")
    service = InferenceService(registry_path)
    assert service.expert_ids == ["expert-a"]
    assert list(service.broken_experts) == ["expert-b"]
    assert service.unprofiled_experts == []
    decision = service.route(["fifo"])
    assert decision["experts"] == ["expert-a"]


def test_broken_binding_recorded_with_reason(tmp_path):
    mismatched = _profile("expert-b", ["counter"])
    mismatched["checkpoint_sha256"] = "ff" * 32
    registry_path = _two_file_registry(tmp_path, json.dumps(mismatched))
    service = InferenceService(registry_path)
    assert service.expert_ids == ["expert-a"]
    assert "differs from registry" in service.broken_experts["expert-b"]


def test_injected_defects_still_fail_fast(tmp_path):
    registry_path = _two_expert_registry(tmp_path)
    bad = _profile("expert-a", ["fifo"])
    bad["supported_mechanisms"] = []
    with pytest.raises(ValueError, match="non-empty"):
        InferenceService(registry_path, profiles={"expert-a": bad})


def test_evidence_counts_reject_unknown_keys_and_normalize():
    profile = _profile("expert-a", ["fifo", "fifo"])
    stored = capabilities.validate_capability_profile(profile)
    assert stored["supported_mechanisms"] == ["fifo"]
    rated = _rated("expert-a", ["fifo"], {"fifo": 90})
    evidence = dict(rated["routing_evidence"])
    evidence["benchmark_source"] = "  synthetic  "
    evidence["areas"]["fifo"]["bogus"] = 1
    with pytest.raises(ValueError, match="unknown routing evidence"):
        capabilities.validate_routing_evidence(evidence, rated)
    del evidence["areas"]["fifo"]["bogus"]
    checked = capabilities.validate_routing_evidence(evidence, rated)
    assert checked["benchmark_source"] == "synthetic"
    assert set(checked["areas"]["fifo"]) == {
        "correct",
        "total",
        "violations",
        "false_safe",
        "errors",
    }


def test_pool_close_is_terminal():
    pool = WorkerPool()
    pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match="pool is closed"):
        pool.load(
            {
                "id": "expert-a",
                "checkpoint": "x",
                "device": "cpu",
                "expected_files": {},
                "laya_root": "y",
            }
        )


def test_inspection_always_discloses_macro_gap():
    assert any(
        "macro expansion" in line
        for line in inspection.inspect_structure("module m; endmodule", [])[
            "limitations"
        ]
    )


def test_arbitration_input_requires_labels_and_routing():
    state = _STATE
    good = _assessment([{"expert_id": "expert-a"}])
    labelless = dict(good, responses=[{"expert_id": "expert-a", "scores": {}}])
    with pytest.raises(ValueError, match="has no label"):
        arbitration.build_arbitration_input(
            labelless, state, "synthetic requirement", _arb_profiles()
        )
    unrouted = dict(good, planned_routing={"experts": []})
    del unrouted["routing"]
    with pytest.raises(ValueError, match="but no routing"):
        arbitration.build_arbitration_input(
            unrouted, state, "synthetic requirement", _arb_profiles()
        )


def test_padded_expected_filename_fails_fast(tmp_path):
    profile = _profile("expert-a", ["fifo"])
    path = _write_registry(
        tmp_path,
        profile,
        {"expected_files": {" model.safetensors ": "ab" * 32}},
    )
    with pytest.raises(ValueError, match="padded expected filename"):
        registry.load_registry(path)


def test_padded_registry_identity_is_stripped_for_binding(tmp_path):
    profile = _profile("expert-a", ["fifo"])
    path = _write_registry(
        tmp_path,
        profile,
        {
            "release": "  synthetic-1 ",
            "question_contract": " siliconshield-requirement-context-v1 ",
            "qualification": " synthetic run ",
        },
    )
    service = InferenceService(path)
    assert service.expert_ids == ["expert-a"]
    assert service.broken_experts == {}


def test_unknown_registry_keys_are_rejected(tmp_path):
    profile = _profile("expert-a", ["fifo"])
    path = _write_registry(tmp_path, profile, {"releaze": "typo"})
    with pytest.raises(ValueError, match="unknown keys"):
        registry.load_registry(path)


def test_unknown_question_contract_fails_binding():
    with pytest.raises(ValueError, match="unknown question_contract"):
        capabilities.validate_capability_profile(
            _profile("expert-a", ["fifo"], question_contract="v9-future")
        )


def test_assemble_state_rejects_empty_and_non_string():
    from siliconshield.experts.identity import assemble_state

    for kwargs in (
        {"requirement": "", "assumptions": "a", "rtl": "r"},
        {"requirement": "r", "assumptions": "  ", "rtl": "r"},
        {"requirement": "r", "assumptions": "a", "rtl": ""},
        {"requirement": None, "assumptions": "a", "rtl": "r"},
    ):
        with pytest.raises(ValueError, match="non-empty"):
            assemble_state(**kwargs)


def test_assemble_state_matches_training_bytes():
    from siliconshield.experts.identity import assemble_state

    assert assemble_state(requirement="req", assumptions="asm", rtl="rtl") == (
        "REQUIREMENT\nreq\nASSUMPTIONS\nasm\nRTL\nrtl"
    )
    assert assemble_state(
        requirement=" req ", assumptions="asm", rtl="rtl"
    ) == ("REQUIREMENT\n req \nASSUMPTIONS\nasm\nRTL\nrtl")


def test_validate_result_rescales_near_distributions():
    from siliconshield.experts.contracts import validate_result

    payload = _result()
    payload["probabilities"] = {
        "satisfies_requirement": 0.7005,
        "violates_requirement": 0.2,
        "insufficient_context": 0.1,
    }
    checked = validate_result(payload)
    assert checked["label"] == "satisfies_requirement"
    assert abs(sum(checked["scores"].values()) - 1.0) < 1e-9


def test_area_for_tag_strips_whitespace():
    assert routing_data.area_for_tag(" fifo ") == "fifo"
    assert (
        routing_data.area_for_tag(" arbiter-grant-handshake ") == "arbitration"
    )


def test_mapping_payloads_are_accepted():
    from types import MappingProxyType

    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )
    payload = MappingProxyType(_result(state=_STATE, model_id="expert-a"))
    envelope = responses.build_expert_response(profile, _STATE, payload)
    assert envelope["expert_id"] == "expert-a"


def test_consult_rejects_non_string_state():
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )
    with pytest.raises(ValueError, match="empty state text"):
        responses.consult_experts(
            ["expert-a"], {"expert-a": profile}, None, lambda e, s: {}
        )


def test_for_header_does_not_hide_later_writers():
    found = inspection.inspect_structure(
        "module m; always @(posedge clk) for (i = 0; i < 4; i = i + 1) "
        "q[i] <= d; endmodule",
        [],
    )
    assert "q" in found["writers"]


def test_compound_assignments_are_writers():
    found = inspection.inspect_structure(
        "module m; always @(posedge clk) begin cnt += 1; acc -= x; "
        "i++; endmodule",
        [],
    )
    assert {"cnt", "acc", "i"} <= set(found["writers"])


def test_arbitrate_usage_fails_before_consult(tmp_path):
    state_path = tmp_path / "state.txt"
    state_path.write_text("synthetic state")
    args = build_parser().parse_args(
        [
            "assess",
            "--registry",
            str(tmp_path / "missing.json"),
            "--scope",
            "fifo",
            "--state-file",
            str(state_path),
            "--arbitrate",
        ]
    )
    # The missing registry would fail first if the usage check ran late.
    with pytest.raises(ValueError, match="--arbitrate needs --requirement"):
        run_assess(args)


def test_lone_issue_or_relevance_is_rejected(tmp_path):
    state_path = tmp_path / "state.txt"
    state_path.write_text("synthetic state")
    for extra in (["--issue", "x"], ["--relevance", "y"]):
        args = build_parser().parse_args(
            [
                "assess",
                "--registry",
                str(tmp_path / "missing.json"),
                "--scope",
                "fifo",
                "--state-file",
                str(state_path),
                "--arbitrate",
                "--requirement",
                "req",
            ]
            + extra
        )
        with pytest.raises(ValueError, match="required together"):
            run_assess(args)


def test_non_object_profiles_fail_without_responses():
    assessment = _assessment([], state=_STATE)
    with pytest.raises(ValueError, match="must be an object"):
        arbitration.build_arbitration_input(
            assessment, _STATE, "synthetic requirement", ["not-a-mapping"]
        )


def test_worker_load_without_python_is_a_runtime_error():
    pool = WorkerPool()
    try:
        with pytest.raises(RuntimeError, match="lacks python"):
            pool.load(
                {
                    "id": "expert-a",
                    "checkpoint": "c",
                    "device": "cpu",
                    "expected_files": {},
                    "laya_root": "r",
                }
            )
    finally:
        pool.close()


def test_route_scope_deduplicates_profiles():
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )
    decision = route_scope(["fifo"], [profile, profile])
    assert decision["experts"] == ["expert-a"]


def test_finding_carries_error_detail():
    from siliconshield.experts import findings

    assessment = _assessment(
        [{"expert_id": "expert-a"}],
        errors=[
            {
                "expert_id": "expert-b",
                "stage": "consult",
                "detail": "synthetic consult failure",
            }
        ],
    )
    _, verdict = _finding_inputs()
    finding = findings.build_finding(
        "synthetic issue",
        "synthetic relevance",
        "synthetic requirement",
        ["top.sv:1"],
        assessment,
        verdict,
    )
    assert finding["failed_experts"] == [
        {
            "expert_id": "expert-b",
            "stage": "consult",
            "detail": "synthetic consult failure",
        }
    ]


def test_assemble_matches_training_and_backend_bytes():
    from siliconshield.app.laya_assessor import state_text
    from siliconshield.experts.identity import (
        assemble_state,
        state_sections,
    )

    fields = ("synthetic requirement", "synthetic assumptions", "rtl")
    assert assemble_state(*fields) == state_text(*fields)
    assert state_sections(assemble_state(*fields)) == fields


def test_unknown_supported_mechanisms_are_rejected():
    with pytest.raises(ValueError, match="unknown tags"):
        capabilities.validate_capability_profile(
            _profile("expert-a", ["fifo", "not-a-mechanism"])
        )


def test_unknown_profile_keys_are_rejected():
    profile = _profile("expert-a", ["fifo"])
    profile["strength"] = "typo for strengths"
    with pytest.raises(ValueError, match="unknown keys"):
        capabilities.validate_capability_profile(profile)
    profile = _profile("expert-a", ["fifo"])
    profile["measurements"] = [
        {"eval": "synthetic", "correct": 1, "total": 1, "extra": 1}
    ]
    with pytest.raises(ValueError, match="unknown keys"):
        capabilities.validate_capability_profile(profile)


def test_registry_paths_are_absolute_without_dereference(tmp_path):
    link = tmp_path / "pylink"
    link.symlink_to(sys.executable)
    profile = _profile("expert-a", ["fifo"])
    path = _write_registry(tmp_path, profile, {})
    raw = json.loads(Path(path).read_text())
    raw["models"][0]["python"] = str(link)
    raw["models"][0]["checkpoint"] = "relative-dir"
    (tmp_path / "relative-dir").mkdir()
    Path(path).write_text(json.dumps(raw))
    (entry,) = registry.load_registry(path)
    assert os.path.isabs(entry["checkpoint"])
    assert entry["checkpoint"] == os.path.abspath(tmp_path / "relative-dir")
    assert entry["python"] == str(link)


def test_arbitration_inspects_structure_when_absent():
    requirement = "synthetic requirement"
    state = (
        "REQUIREMENT\n"
        + requirement
        + "\nASSUMPTIONS\nsynthetic assumptions\nRTL\n"
        + "module m(input clk, input wr, output reg q); "
        "always @(posedge clk) if (wr) q <= 1'b1; endmodule"
    )
    assessment = _assessment([{"expert_id": "expert-a"}], state=state)
    bound = arbitration.build_arbitration_input(
        assessment, state, requirement, _arb_profiles()
    )
    assert bound["structure"]["writers"] == {"q": [1]}
    explicit = arbitration.build_arbitration_input(
        assessment, state, requirement, _arb_profiles(), structure={}
    )
    assert explicit["structure"] == {}


def test_finding_flags_need_arbitrate(tmp_path):
    state_path = tmp_path / "state.txt"
    state_path.write_text("synthetic state")
    args = build_parser().parse_args(
        [
            "assess",
            "--registry",
            str(tmp_path / "missing.json"),
            "--scope",
            "fifo",
            "--state-file",
            str(state_path),
            "--issue",
            "x",
            "--relevance",
            "y",
        ]
    )
    with pytest.raises(ValueError, match="need --arbitrate"):
        run_assess(args)


def test_checked_scores_sum_to_one():
    payload = _result(state=_STATE, model_id="expert-a")
    payload["probabilities"] = {
        "satisfies_requirement": 0.5004,
        "violates_requirement": 0.4996,
        "insufficient_context": 0.0,
    }
    checked = contracts.validate_result(payload)
    assert abs(sum(checked["scores"].values()) - 1.0) < 1e-12


def test_consult_rejects_malformed_id_lists():
    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )
    for bad in ("expert-a", ["expert-a", None], ["expert-a", ["x"]], ()):
        with pytest.raises(ValueError, match="list of names"):
            responses.consult_experts(
                bad, {"expert-a": profile}, _STATE, lambda e, s: {}
            )


def test_load_failure_reports_load_stage():
    from siliconshield.experts.worker import WorkerLoadError

    profile = capabilities.validate_capability_profile(
        _profile("expert-a", ["fifo"])
    )

    def fail_load(expert_id, state):
        raise WorkerLoadError("Cannot load expert-a: synthetic")

    _, errors = responses.consult_experts(
        ["expert-a"], {"expert-a": profile}, _STATE, fail_load
    )
    assert errors == [
        {
            "expert_id": "expert-a",
            "stage": "load",
            "detail": "Cannot load expert-a: synthetic",
        }
    ]


def test_typedef_and_struct_members_are_not_state():
    found = inspection.inspect_structure(
        "module m (input clk, input [7:0] d, output logic [7:0] q); "
        "typedef logic [7:0] word_t; word_t mem [0:3]; "
        "always_ff @(posedge clk) q <= d; endmodule",
        [],
    )
    assert found["state"] == ["q"]
    assert "word_t" in found["unknown_symbols"]
    assert "mem" in found["unknown_symbols"]
    assert len(found["limitations"]) == 2
    found = inspection.inspect_structure(
        "module m (input clk, input [7:0] d, output logic [7:0] q); "
        "struct packed { logic [3:0] hi; logic [3:0] lo; } halves; "
        "always_ff @(posedge clk) q <= d; endmodule",
        [],
    )
    assert found["state"] == ["q"]
    assert "halves" in found["unknown_symbols"]
    assert len(found["limitations"]) == 2


def test_params_and_loop_indices_are_not_writers():
    found = inspection.inspect_structure(
        "module m #(parameter W = 8) (input clk, input [W-1:0] d, "
        "output reg [W-1:0] q); localparam Z = 0; integer i; "
        "always @(posedge clk) for (i = 0; i < W; i = i + 1) "
        "q[i] <= d[i]; endmodule",
        [],
    )
    assert "W" not in found["writers"]
    assert "Z" not in found["writers"]
    assert "i" not in found["writers"]
    assert "q" in found["writers"]
    assert "q" in found["state"]
    assert "d" not in found["state"]


def test_system_calls_and_keywords_stay_out_of_unknowns():
    found = inspection.inspect_structure(
        'module m; always @(*) begin $display("%d", 1); '
        "assert (a) else $error; end endmodule",
        [],
    )
    assert "display" not in found["unknown_symbols"]
    assert "assert" not in found["unknown_symbols"]
    assert "error" not in found["unknown_symbols"]
