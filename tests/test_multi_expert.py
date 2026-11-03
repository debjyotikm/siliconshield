"""Synthetic tests for multi-expert routing and envelopes."""

import json
import sys
from pathlib import Path

import pytest

from siliconshield.experts import (
    arbitration,
    capabilities,
    responses,
)
from siliconshield.experts.cli import build_parser, run_assess
from siliconshield.experts.identity import digest, text_digest
from siliconshield.experts.router import route_scope
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
