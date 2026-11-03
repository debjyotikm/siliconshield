"""Synthetic tests for Glimmer-driven router sessions."""

import copy
import json

from siliconshield.experts.identity import digest, text_digest
from siliconshield.research_loop.procedures import glimmer_router
from siliconshield.training.laya.question_contracts import question_for


def _identity():
    questions = question_for("siliconshield-requirement-context-v1")
    questions["security"]["option_order"] = [0, 1, 2]
    return {
        "checkpoint_sha256": "ab" * 32,
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


def _reply(label, state):
    return {
        "label": label,
        "probabilities": {
            "satisfies_requirement": 0.7
            if label == "satisfies_requirement"
            else 0.15,
            "violates_requirement": 0.7
            if label == "violates_requirement"
            else 0.15,
            "insufficient_context": 0.7
            if label == "insufficient_context"
            else 0.15,
        },
        "coverage": {
            "truncated": False,
            "state_tokens_dropped": 0,
            "truncated_questions": [],
        },
        "model_id": "expert-a",
        "execution_identity": _identity(),
        "state_sha256": text_digest(state),
    }


class _Pool:
    def __init__(self, label="violates_requirement"):
        self.label = label
        self.loaded = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def load(self, entry):
        self.loaded.append(entry["id"])

    def transport(self):
        def call(expert_id, state):
            return _reply(self.label, state)

        return call


def _registry(tmp_path):
    from siliconshield.schemas import (
        EXPERT_CAPABILITY_SCHEMA,
        GUI_MODEL_REGISTRY_SCHEMA,
    )

    profile = {
        "schema": EXPERT_CAPABILITY_SCHEMA,
        "id": "expert-a",
        "release": "synthetic-1",
        "checkpoint_sha256": "ab" * 32,
        "supported_mechanisms": ["fifo"],
        "strengths": ["synthetic strength"],
        "weaknesses": ["synthetic weakness"],
        "known_failures": ["synthetic failure"],
        "question_contract": "siliconshield-requirement-context-v1",
        "qualification_ref": "synthetic run",
        "measurements": [{"eval": "synthetic", "correct": 1, "total": 2}],
    }
    (tmp_path / "profile.json").write_text(json.dumps(profile))
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
        "release": "synthetic-1",
        "question_contract": "siliconshield-requirement-context-v1",
    }
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps({"schema": GUI_MODEL_REGISTRY_SCHEMA, "models": [entry]})
    )
    return str(path)


class _Client:
    model = "fake-glimmer"

    def __init__(self, script):
        self.script = [copy.deepcopy(item) for item in script]
        self.seen = []

    def complete(self, messages, tools=()):
        self.seen.append({"messages": copy.deepcopy(messages), "tools": tools})
        return self.script.pop(0)


def _call(name, args, call_id="call-1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "function": {"name": name, "arguments": args},
            }
        ],
    }


def test_session_runs_roster_then_assess(tmp_path):
    registry = _registry(tmp_path)
    out = tmp_path / "session"
    skill = tmp_path / "skill.md"
    skill.write_text("Assess with the router tools.")
    args = {
        "scope": "fifo",
        "requirement": "Hold occupancy.",
        "assumptions": "Two-state.",
        "rtl": "module m; endmodule",
    }
    client = _Client(
        [
            _call("router.roster", {}),
            _call("router.assess", args, call_id="call-2"),
            {"role": "assistant", "content": "Verdict: violates."},
        ]
    )
    history = glimmer_router.run_router_session(
        client,
        registry,
        "Assess the FIFO hold requirement.",
        str(out),
        skill=str(skill),
        max_turns=6,
        pool_factory=_Pool,
    )
    assert history[-1]["content"] == "Verdict: violates."
    tool_results = [item for item in history if item.get("role") == "tool"]
    assert len(tool_results) == 2
    assessment = json.loads(tool_results[1]["content"])
    assert assessment["routing"]["experts"] == ["expert-a"]
    assert assessment["responses"][0]["label"] == "violates_requirement"
    traces = list(out.glob("trace-*.jsonl"))
    assert len(traces) == 1
    assert len(traces[0].read_text().strip().splitlines()) >= 4
