"""Unattended operator loop with a fixture Glimmer client: validation gates execution."""

import copy
import json
import re

import pytest

from siliconshield.research_loop.episodes.operator_episodes import (
    SCENARIOS,
    episode_for,
)
from siliconshield.research_loop.execution.glimmer_operator import (
    OperatorTools,
    run_cycle,
)
from siliconshield.research_loop.execution.operator import Operator
from siliconshield.research_loop.execution.strategy import (
    assess_fitting,
    fitting_plan,
)

ARMS = [
    {
        "name": "headonly",
        "overrides": {"freeze_encoder": True, "disable_adapters": True},
    },
    {
        "name": "adapters",
        "overrides": {"freeze_encoder": True, "adapter_lr": 1e-4},
    },
    {
        "name": "full",
        "overrides": {"freeze_encoder": False, "adapter_lr": 1e-4},
    },
]


class FakeJobs:
    def __init__(self, root, profile):
        self.profile, self.root = profile, root
        self.submissions = 0

    def status(self, experiment_id):
        return {}

    def submit(self, experiment_id, overrides, launch=False):
        self.submissions += 1
        return {"phase": "submitted"}


def make_operator(tmp_path):
    profile = {
        "python": "/usr/bin/python3",
        "source": "/source",
        "model": "/m",
        "data": "/d",
        "runs": str(tmp_path / "runs"),
        "gpus": [0, 1, 2],
        "defaults": {"max_updates": 10},
    }
    op = Operator.__new__(Operator)
    op.jobs = FakeJobs(tmp_path / "runs", profile)
    op.profile = profile
    op.runner = lambda argv: {"LoadState": "not-found"}
    from siliconshield.research_loop.execution.operator import (
        ResourceReservation,
    )

    op.reservation = ResourceReservation(op.jobs.root)
    return op


class ScriptedGlimmer:
    """Replays one authored operator trace for the requested action."""

    model = "fixture-glimmer"
    endpoint = "http://fixture.invalid"

    def __init__(self, action):
        self.action = action
        self.calls = 0

    def complete(self, messages, tools=()):
        self.calls += 1
        scenario = next(s for s in SCENARIOS if s[3] == self.action)
        row = episode_for(
            scenario[0],
            scenario[1],
            scenario[2],
            scenario[3],
            scenario[4],
            "fx",
            0,
        )
        message = copy.deepcopy(row["messages"][len(messages)])
        run_id = re.search(r"Inspect run (\S+)", messages[1]["content"])[1]
        for call in message.get("tool_calls", []):
            call["function"]["arguments"]["run_id"] = run_id
        if message.get("role") == "assistant" and message.get("content"):
            final = json.loads(message["content"])
            final["run_id"] = run_id
            message["content"] = json.dumps(final)
        return message


def test_valid_decision_executes_through_operator(tmp_path):
    op = make_operator(tmp_path)
    manifest = op.prepare("exp-a", {})
    manifest["phase"] = "submit"  # state-machine position after a real submit
    manifest, executed = run_cycle(
        op, manifest, ScriptedGlimmer("leave_unchanged"), tmp_path
    )
    assert executed["action"] == "leave_unchanged"


def test_decision_and_execution_recorded(tmp_path):
    op = make_operator(tmp_path)
    manifest = op.prepare("exp-a", {})
    manifest["phase"] = "submit"
    manifest, executed = run_cycle(
        op, manifest, ScriptedGlimmer("leave_unchanged"), tmp_path
    )
    assert executed["action"] == "leave_unchanged"
    assert manifest["interventions"][-1]["action"] == "leave_unchanged"


def test_invalid_resume_decision_is_blocked_deterministically(tmp_path):
    op = make_operator(tmp_path)
    manifest = op.prepare("exp-a", {})
    manifest["phase"] = "submit"
    # Scripted resume carries a checkpoint_dir outside the registered output, so
    # Operator.intervene must reject it deterministically instead of executing.
    with pytest.raises((ValueError, KeyError, TypeError)):
        run_cycle(op, manifest, ScriptedGlimmer("resume"), tmp_path)


class RecordingOperator:
    """Records intervene params without executing, to verify field propagation."""

    def __init__(self):
        self.interventions = []

        class _Jobs:
            def status(self, experiment_id):
                return {}

        class _Reservation:
            def owner(self):
                return None

        self.jobs, self.reservation = _Jobs(), _Reservation()

    def observe(self, manifest):
        return manifest

    def diagnose(self, manifest):
        return manifest

    def intervene(self, manifest, action, **params):
        self.interventions.append({"action": action, "params": params})
        manifest.setdefault("interventions", []).append(
            {"action": action, "params": params}
        )
        return manifest


def stub_manifest():
    return {
        "experiment_id": "exp-a",
        "phase": "submit",
        "history": [],
        "interventions": [],
    }


def test_adjust_lr_propagates_lr_to_execution(tmp_path):
    stub = RecordingOperator()
    manifest, executed = run_cycle(
        stub, stub_manifest(), ScriptedGlimmer("adjust_lr"), tmp_path
    )
    assert executed["action"] == "adjust_lr"
    assert stub.interventions[-1] == {
        "action": "adjust_lr",
        "params": {"lr": 2e-5},
    }


def test_resume_propagates_checkpoint_dir_to_execution(tmp_path):
    stub = RecordingOperator()
    manifest, executed = run_cycle(
        stub, stub_manifest(), ScriptedGlimmer("resume"), tmp_path
    )
    assert executed["action"] == "resume"
    assert stub.interventions[-1]["params"]["checkpoint_dir"].startswith(
        "/private/runs/"
    )


def test_episodes_require_lr_and_checkpoint_dir_fields():
    from siliconshield.research_loop.episodes.contracts import validate_episode

    scenario, state, resources, action, reason = next(
        s for s in SCENARIOS if s[3] == "adjust_lr"
    )
    row = episode_for(
        scenario, state, resources, action, reason, "lineage-z", 0
    )
    call = next(
        m["tool_calls"][0]
        for m in row["messages"]
        if m.get("tool_calls")
        and m["tool_calls"][0]["function"]["name"] == "operator.decide"
    )
    del call["function"]["arguments"]["lr"]
    with pytest.raises(ValueError, match="finite lr"):
        validate_episode(row)


def test_tools_cannot_submit_jobs(tmp_path):
    op = make_operator(tmp_path)
    manifest = op.prepare("exp-a", {})
    tools = OperatorTools(op, manifest)
    with pytest.raises(ValueError):
        tools.dispatch("operator.submit", {"run_id": "exp-a"})
    assert op.jobs.submissions == 0


def test_fitting_plan_is_train_only_and_promotion_is_deterministic():
    plan = fitting_plan(
        "fixture",
        {
            "limit_train_rows": 36,
            "max_updates": 6,
            "max_len": 128,
            "head_max_len": 32,
        },
        arms=ARMS,
    )
    assert len(plan) == 3
    assert all(spec["split_access"] == ["train"] for spec in plan)
    assert all(
        spec["experiment_id"].startswith("fixture-fit-") for spec in plan
    )
    results = [
        {
            "arm": spec["arm"],
            "train_accuracy": 0.62,
            "initial_train_accuracy": 0.40,
            "adapter_not_worse_than_disabled": True,
            "max_component_grad_norm_ratio": 3.0,
        }
        for spec in plan
    ]
    assessed = assess_fitting(results)
    assert assessed["promotable"] == [spec["arm"] for spec in plan]
    results[0]["train_accuracy"] = 0.41
    assert assess_fitting(results)["promotable"] != assessed["promotable"]


def test_fitting_plan_arms_produce_valid_launch_commands(tmp_path):
    from siliconshield.research_loop.execution.security_jobs import command_for

    profile = {
        "python": "/usr/bin/python3",
        "source": str(tmp_path / "src"),
        "model": "/m",
        "data": "/d",
        "runs": str(tmp_path / "runs"),
        "gpus": [0, 1, 2],
        "defaults": {"max_updates": 60},
    }
    seen_flags = set()
    for spec in fitting_plan(
        "fixture",
        {
            "limit_train_rows": 36,
            "max_updates": 6,
            "max_len": 128,
            "head_max_len": 32,
        },
        arms=ARMS,
    ):
        command, output = command_for(
            profile, spec["experiment_id"], spec["overrides"]
        )
        assert str(output).endswith(spec["experiment_id"])
        assert "--limit-train-rows" in command
        assert command[command.index("--limit-train-rows") + 1] == "36"
        assert "--max-len" in command and "--head-max-len" in command
        seen_flags.update(spec["overrides"])
        if spec["arm"] == "headonly":
            assert (
                "--freeze-encoder" in command
                and "--disable-adapters" in command
            )
            assert "--adapter-scale" not in command
        if "adapter_lr" in spec["overrides"]:
            value = float(command[command.index("--adapter-lr") + 1])
            assert value == spec["overrides"]["adapter_lr"]
    assert "freeze_encoder" in seen_flags and "disable_adapters" in seen_flags


def test_operator_loop_honors_poll_sleep_and_persists_failures(tmp_path):
    from siliconshield.research_loop.execution import glimmer_operator as loop

    class LoopStub:
        def __init__(self):
            self.unit = "stub-unit"
            self.runner = lambda argv: {
                "LoadState": "loaded",
                "ActiveState": "active",
                "Result": "success",
            }
            self.interventions = []

            class _Jobs:
                def status(self, experiment_id):
                    return {}

            class _Reservation:
                def owner(self):
                    return None

            self.jobs, self.reservation = _Jobs(), _Reservation()

        def prepare(self, exp, ov=None, budget=None):
            return {
                "experiment_id": exp,
                "phase": "prepare",
                "history": [],
                "interventions": [],
                "unit": self.unit,
            }

        def preflight(self, manifest):
            manifest["phase"] = "preflight"
            return manifest

        def submit(self, manifest):
            manifest["phase"] = "submit"
            return manifest

        def observe(self, manifest):
            manifest["phase"] = "observe"
            return manifest

        def diagnose(self, manifest):
            manifest["phase"] = "diagnose"
            manifest["history"].append({"phase": "diagnose"})
            return manifest

        def intervene(self, manifest, action, **params):
            manifest["phase"] = "intervene"
            self.interventions.append((action, params))
            manifest["interventions"].append({"action": action})
            return manifest

        def finalize(self, manifest):
            manifest["phase"] = "finalize"
            return manifest

        def report(self, manifest):
            manifest["phase"] = "report"
            manifest["terminal"] = "complete"
            return manifest

    attempts = {"n": 0}

    class FlakyGlimmer(ScriptedGlimmer):
        def complete(self, messages, tools=()):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ConnectionError("tunnel reset")
            return super().complete(messages, tools)

    stub, sleeps = LoopStub(), []
    config = {
        "profile": "/nonexistent",
        "experiment_id": "exp-a",
        "endpoint": "http://x",
        "model": "m",
        "output": str(tmp_path / "out"),
        "max_cycles": 2,
        "poll_seconds": 7,
    }
    result = loop.run_operator(
        config,
        client=FlakyGlimmer("continue"),
        operator=stub,
        sleep=sleeps.append,
    )
    assert sleeps == [7, 7]
    status = json.loads((tmp_path / "out/operator-status.json").read_text())
    assert status["cycles"][0]["job_running"] is True
    assert "tunnel reset" in status["cycles"][0]["error"]
    assert status["cycles"][1]["executed"]["action"] == "continue"
    assert result["manifest"]["phase"] == "intervene"


def test_wrong_run_id_cannot_be_recorded(tmp_path):
    op = make_operator(tmp_path)
    tools = OperatorTools(op, op.prepare("exp-a"))
    with pytest.raises(ValueError, match="bound experiment"):
        tools.dispatch("operator.inspect_run", {"run_id": "exp-b"})
    assert not tools.inspected


def test_completed_operator_reentry_is_idempotent(tmp_path):
    from siliconshield.research_loop.episodes.contracts import atomic_json
    from siliconshield.research_loop.execution.glimmer_operator import (
        run_operator,
    )

    op = make_operator(tmp_path)
    manifest = op.prepare("exp-a")
    manifest.update(phase="report", terminal="complete")
    atomic_json(op.manifest_path("exp-a"), manifest)
    result = run_operator(
        {
            "profile": "unused",
            "experiment_id": "exp-a",
            "output": str(tmp_path / "out"),
            "model": "fixture",
        },
        client=ScriptedGlimmer("continue"),
        operator=op,
    )
    assert result["manifest"]["terminal"] == "complete"
    assert not op.jobs.submissions


def test_fitting_rejects_nonfinite_and_missing_measurements():
    with pytest.raises(ValueError):
        assess_fitting(
            [
                {
                    "arm": "x",
                    "train_accuracy": float("nan"),
                    "initial_train_accuracy": 0.3,
                }
            ]
        )
    with pytest.raises(KeyError):
        assess_fitting([{"arm": "x", "train_accuracy": 0.6}])


def test_ssh_chat_transport_passes_payload_on_stdin(monkeypatch):
    from types import SimpleNamespace

    from siliconshield.research_loop.controller import client

    seen = []

    def subprocess_run(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": "ok"}}
                    ]
                }
            )
        )

    monkeypatch.setattr(client.subprocess, "run", subprocess_run)
    model = client.ChatClient(
        "http://127.0.0.1:1234/v1/chat/completions",
        "fixture",
        ssh_host="fixture-host",
    )
    assert (
        model.complete([{"role": "user", "content": "test"}])["content"] == "ok"
    )
    argv, kwargs = seen[0]
    assert argv[0] == "ssh" and "fixture-host" in argv
    envelope = json.loads(kwargs["input"])
    assert envelope["payload"]["messages"][0]["content"] == "test"
    assert envelope["headers"]["Content-Type"] == "application/json"
    assert "StrictHostKeyChecking=no" not in argv


def test_operator_evidence_omits_large_prediction_and_provenance_payloads():
    from siliconshield.research_loop.execution.glimmer_operator import (
        model_status,
    )

    summary = model_status(
        {
            "status.json": {
                "event": "initial_train",
                "metrics": {
                    "accuracy": 0.4,
                    "nll": 1.1,
                    "predictions": [0] * 10000,
                    "reporting": {"large": [1] * 10000},
                },
            },
            "report.json": {
                "evaluation_role": "train_fitting",
                "train": {"accuracy": 0.8, "nll": 0.5},
                "signature": {"private_path": "/private/example"},
            },
        }
    )
    assert len(json.dumps(summary)) < 1000
    assert summary["report"]["train"]["accuracy"] == 0.8
    assert "predictions" not in json.dumps(
        summary
    ) and "private_path" not in json.dumps(summary)


def test_campaign_uses_qualified_recipe_and_removes_fitting_flags(tmp_path):
    from siliconshield.research_loop.execution.strategy import run_campaign

    calls = []

    def execute(config):
        calls.append(config)
        folder = tmp_path / config["experiment_id"]
        folder.mkdir()
        if config["overrides"].get("limit_train_rows"):
            report = {
                "evaluation_role": "train_fitting",
                "train_accuracy": 0.8,
                "initial_train_accuracy": 0.3,
                "train": {"nll": 0.4},
                "selected_checkpoint": {"step": 6},
            }
        else:
            report = {
                "development": {"accuracy": 0.7},
                "development_acceptance": {"accepted": False},
            }
        (folder / "report.json").write_text(json.dumps(report))
        return {"manifest": {"terminal": "complete", "output": str(folder)}}

    config = {
        "experiment_prefix": "fixture",
        "output": str(tmp_path / "operator"),
        "profile": "unused",
        "timeout_seconds": 30,
        "operator": {"model": "fixture"},
        "fitting_common": {
            "limit_train_rows": 36,
            "no_development_in_fitting": True,
            "max_updates": 6,
        },
        "arms": ARMS,
        "development_overrides": {"max_updates": 20},
    }
    run_campaign(config, run=execute)
    assert len(calls) == 4
    assert calls[-1]["experiment_id"] == "fixture-development"
    assert "limit_train_rows" not in calls[-1]["overrides"]
    assert "no_development_in_fitting" not in calls[-1]["overrides"]
    assert not calls[-1]["overrides"].get("disable_adapters")
    assert calls[-1]["overrides"]["max_updates"] == 20


def test_stop_requires_recorded_evidence_not_weak_fitting_metrics(tmp_path):
    from siliconshield.research_loop.execution.glimmer_operator import (
        stop_evidence,
    )

    manifest = {"output": str(tmp_path)}
    status = {
        "execution.json": {"phase": "running"},
        "status.json": {"loss": 1.1},
    }
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "event": "train_evaluation",
                "metrics": {"accuracy": 0.3, "nll": 1.1},
            }
        )
    )
    assert not stop_evidence(manifest, status)
    events.write_text(
        "\n".join(
            json.dumps({"event": "development", "metrics": m})
            for m in (
                {"accuracy": 0.8, "nll": 0.3},
                {"accuracy": 0.6, "nll": 0.6},
            )
        )
    )
    assert stop_evidence(manifest, status)
    events.unlink()
    assert stop_evidence(manifest, {"execution.json": {"phase": "failed"}})


def test_inspect_run_exposes_development_history_with_steps_and_timestamps(
    tmp_path,
):
    from siliconshield.research_loop.execution.glimmer_operator import (
        development_history,
    )

    events = tmp_path / "output" / "events.jsonl"
    events.parent.mkdir(parents=True)
    lines = [
        {
            "event": "initial_development",
            "step": 0,
            "time": 1000.0,
            "metrics": {
                "n": 256,
                "accuracy": 0.34,
                "nll": 1.46,
                "false_safe_rate": 0.94,
            },
        },
        {"event": "training", "step": 1, "time": 1001.0, "loss": 1.7},
        {
            "event": "development",
            "step": 60,
            "time": 2000.0,
            "metrics": {
                "n": 256,
                "accuracy": 0.99,
                "nll": 0.02,
                "false_safe_rate": 0.01,
            },
        },
    ]
    events.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    history = development_history({"output": str(tmp_path / "output")})
    assert history["total"] == 2 and history["complete"] is True
    assert [entry["step"] for entry in history["entries"]] == [0, 60]
    assert [entry["time"] for entry in history["entries"]] == [1000.0, 2000.0]
    assert history["entries"][1]["metrics"]["accuracy"] == 0.99
    assert development_history({}) == {
        "entries": [],
        "complete": True,
        "total": 0,
    }


def test_inspect_observation_carries_history_and_contract_version(tmp_path):
    op = make_operator(tmp_path)
    output = tmp_path / "runs" / "exp-a"
    output.mkdir(parents=True, exist_ok=True)
    (output / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "development",
                "step": 60,
                "time": 2000.0,
                "metrics": {"accuracy": 0.5, "nll": 1.0},
            }
        )
        + "\n"
    )
    manifest = op.prepare("exp-a", {})
    manifest["phase"] = "submit"
    manifest["output"] = str(output)
    tools = OperatorTools(op, manifest)
    observed = tools.dispatch("operator.inspect_run", {"run_id": "exp-a"})
    assert observed["contract_version"] == 4
    assert observed["development_history"]["total"] == 1
    assert observed["development_history"]["entries"][0]["step"] == 60
