"""Operator curriculum, contract, episodes, baseline, and dev tests: synthetic fixtures only."""

import copy
import hashlib
import json
import sys

import pytest

from siliconshield.research_loop.episodes.operator_episodes import (
    SCENARIOS,
    episode_for,
)
from siliconshield.research_loop.execution.operator import (
    Operator,
)


# ---- Merged from test_operator_curriculum.py (aliased: build) ----
def spec():
    return {
        "scenario": "illustrative_recovery",
        "family": "authored-family",
        "index": 0,
        "split": "train",
        "action": "resume",
        "reason": "Resume the validated owned checkpoint.",
        "state": {
            "phase": "paused",
            "systemd": "inactive",
            "checkpoint_valid": True,
            "execution_failed": False,
            "selected_step": 10,
            "schedule": {"plateau_lr_reductions": []},
            "development_history": {
                "entries": [
                    {
                        "population_id": "dev",
                        "metrics": {"n": 100, "accuracy": 0.8, "nll": 0.4},
                    },
                    {
                        "population_id": "dev",
                        "metrics": {"n": 100, "accuracy": 0.9, "nll": 0.3},
                    },
                ]
            },
        },
        "resources": {"gpus_free": 3, "owned_by_this_run": False},
    }


# ---- Merged from test_operator_contract.py ----
class FakeJobs:
    def __init__(self, root):
        self.root = root / "runs"
        self.root.mkdir(exist_ok=True)
        self.profile = {
            "python": sys.executable,
            "source": str(root / "source"),
            "model": str(root / "model"),
            "data": str(root / "data"),
            "runs": str(self.root),
            "resource_root": str(root / "resources"),
            "defaults": {"max_updates": 10, "encoder_lr": 1e-5},
        }
        self.units, self.launches = {}, []
        for folder in (
            "source/siliconshield/training/laya",
            "source/siliconshield/research_loop/episodes",
            "source/siliconshield/research_loop/execution",
            "model",
            "data",
        ):
            (root / folder).mkdir(parents=True, exist_ok=True)
        (
            root / "source/siliconshield/training/laya/security_train.py"
        ).write_text("")
        for package in (
            "source/siliconshield/__init__.py",
            "source/siliconshield/research_loop/__init__.py",
            "source/siliconshield/research_loop/episodes/__init__.py",
            "source/siliconshield/research_loop/execution/__init__.py",
        ):
            (root / package).write_text("")
        (
            root / "source/siliconshield/research_loop/episodes/contracts.py"
        ).write_text(
            "def atomic_json(*args):\n    return None\n\n\n"
            "def read_json(*args):\n    return {}\n"
        )
        (
            root / "source/siliconshield/research_loop/execution/owned_job.py"
        ).write_text(
            "from siliconshield.research_loop.episodes.contracts import (\n"
            "    atomic_json,\n"
            "    read_json,\n"
            ")\n"
        )
        for split in ("train", "dev"):
            (root / "data" / (split + ".jsonl")).write_text("{}\n")

    def status(self, experiment_id):
        output = self.root / experiment_id
        return {
            n: json.loads((output / n).read_text())
            for n in ("job.json", "execution.json", "report.json")
            if (output / n).exists()
        }

    def launch_owned(self, manifest, path, resources, action):
        self.launches.append((list(manifest["command"]), action))
        self.units[manifest["unit"]] = {
            "LoadState": "loaded",
            "ActiveState": "active",
            "Result": "success",
            "Description": "siliconshield:" + manifest["idempotency_key"],
        }
        return {"phase": "submitted"}

    def runner(self, argv):
        return self.units.get(argv[3], {"LoadState": "not-found"})


@pytest.fixture
def operator(tmp_path, monkeypatch):
    monkeypatch.delenv("LAYA_ROOT", raising=False)
    monkeypatch.setattr(
        "siliconshield.research_loop.execution.operator.verify_laya",
        lambda *a: True,
    )
    jobs = FakeJobs(tmp_path)
    return Operator(None, jobs=jobs, runner=jobs.runner)


def prepared(op, name="experiment", overrides=None):
    return op.preflight(op.prepare(name, overrides))


def test_preflight_required_and_overrides_preserved(operator):
    manifest = operator.prepare("experiment", {"seed": 99})
    with pytest.raises(ValueError, match="transition"):
        operator.submit(manifest)
    operator.preflight(manifest)
    operator.submit(manifest)
    command, _ = operator.jobs.launches[0]
    assert command[command.index("--seed") + 1] == "99"
    operator.submit(manifest)
    assert len(operator.jobs.launches) == 1


def test_recipe_and_reporting_contract_bind_the_launch(operator, tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text('{"a": 1}')
    contract = tmp_path / "contract.json"
    contract.write_text('{"b": 2}')
    operator.profile["recipe"] = str(recipe)
    operator.profile["reporting_contract"] = str(contract)
    first = operator.prepare("bound")
    assert (
        first["execution_inputs"][str(recipe)]
        == hashlib.sha256(b'{"a": 1}').hexdigest()
    )
    assert str(contract) in first["execution_inputs"]
    recipe.write_text('{"a": 2}')
    with pytest.raises(ValueError, match="changed"):
        operator.preflight(first)


# ---- Merged from test_operator_episodes.py (aliased: build) ----


def _old_style_row(scenario_name):
    """Simulate a saved pre-contract episode: old tools, missing decide fields."""
    scenario, state, resources, action, reason = next(
        s for s in SCENARIOS if s[0] == scenario_name
    )
    row = episode_for(
        scenario, state, resources, action, reason, "old-lineage", 0
    )
    row = copy.deepcopy(row)
    for message in row["messages"]:
        for call in message.get("tool_calls", []):
            if call["function"]["name"] == "operator.decide":
                call["function"]["arguments"].pop("lr", None)
                call["function"]["arguments"].pop("checkpoint_dir", None)
    for tool in row["tools"]:
        if tool["function"]["name"] == "operator.decide":
            properties = tool["function"]["parameters"]["properties"]
            properties.pop("lr", None)
            properties.pop("checkpoint_dir", None)
    row["target"].pop("lr", None)
    row["target"].pop("checkpoint_dir", None)
    return row


def _operator_row(
    summary, action="continue", reason="Continuing within budget."
):
    from siliconshield.research_loop.episodes.contracts import OPERATOR_TOOLS

    return {
        "id": "repro-1",
        "group_id": "repro-g",
        "tools": OPERATOR_TOOLS,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Inspect run repro-1."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "i",
                        "type": "function",
                        "function": {
                            "name": "operator.decide",
                            "arguments": {"run_id": "repro-1"},
                        },
                    }
                ],
            },
        ],
        "summary": summary,
        "target": {"action": action, "reason": reason},
        "verification": {"kind": "authored_fixture"},
    }


def _operator_delivered(reason, action="continue"):
    return {
        "error": None,
        "read_evidence": True,
        "compared_metrics": True,
        "decision": {"run_id": "repro-1", "action": action, "reason": reason},
        "final": '{"action": "%s", "run_id": "repro-1", "executed": false, "scope": "operator_replay"}'
        % action,
    }


def _failed_acceptance_summary():
    return {
        "run_id": "repro-1",
        "systemd": "running",
        "step": 200,
        "execution_failed": False,
        "acceptance": {
            "accepted": False,
            "reasons": ["worst_family_false_safe_above_budget"],
            "thresholds": {"max_worst_family_false_safe_rate": 0.25},
        },
        "development_history": {
            "entries": [
                {"step": 100, "metrics": {"accuracy": 0.9, "nll": 0.3}},
                {"step": 200, "metrics": {"accuracy": 0.94, "nll": 0.36}},
            ],
            "complete": True,
            "total": 2,
        },
        "schedule": {"plateau_lr_reductions": []},
        "selected_step": 200,
    }


# ---- Merged from test_operator_baseline.py ----
def _fixture_client(rows):
    indexed = {row["summary"]["run_id"]: row for row in rows}

    class FixtureClient:
        model = "baseline-fixture"
        endpoint = "http://fixture.invalid"

        def complete(self, messages, tools=()):
            import re

            match = re.search(r"Inspect run (\S+)", messages[1]["content"])
            row = indexed[match.group(1)]
            return json.loads(json.dumps(row["messages"][len(messages)]))

    return FixtureClient()


# ---- Merged from test_operator_dev.py ----
def _norm_state(state):
    import copy

    state = copy.deepcopy(state)
    state.pop("run_id", None)
    return json.dumps(state, sort_keys=True)


def _failed_unit(operator, manifest):
    operator.jobs.units[manifest["unit"]] = {
        "LoadState": "loaded",
        "ActiveState": "failed",
        "Result": "failure",
        "Description": "siliconshield:" + manifest["idempotency_key"],
    }


def _job_profile(tmp_path):
    return {
        "python": sys.executable,
        "source": str(tmp_path),
        "model": str(tmp_path),
        "data": str(tmp_path),
        "runs": str(tmp_path / "runs"),
        "defaults": {"max_updates": 10},
    }


def test_command_helpers_reject_malformed_profiles(tmp_path):
    from siliconshield.research_loop.execution import operator as opmod
    from siliconshield.research_loop.execution import security_jobs as jobs

    profile = _job_profile(tmp_path)
    with pytest.raises(ValueError, match="must be an object"):
        jobs.command_for(["x"], "e1", {})
    with pytest.raises(ValueError, match="stable identifier"):
        jobs.command_for(profile, 5, {})
    with pytest.raises(ValueError, match="overrides must be an object"):
        jobs.command_for(profile, "e1", None)
    with pytest.raises(ValueError, match="GPU profiles"):
        jobs.command_for(dict(profile, gpus=5), "e1", {})
    bad = {k: v for k, v in profile.items() if k != "source"}
    with pytest.raises(ValueError, match="needs source"):
        jobs.command_for(bad, "e1", {})
    prof_path = tmp_path / "prof.json"
    prof_path.write_text("[1, 2]")
    with pytest.raises(ValueError, match="must be an object"):
        jobs.SecurityJobs(str(prof_path))
    prof_path.write_text(json.dumps({"python": "p"}))
    with pytest.raises(ValueError, match="needs runs"):
        jobs.SecurityJobs(str(prof_path))
    with pytest.raises(ValueError, match="stable identifier"):
        jobs.service_command(profile, 5, ["x"])
    with pytest.raises(ValueError, match="must be an object"):
        opmod.execution_inputs(["x"], {})
    with pytest.raises(ValueError, match="overrides must be an object"):
        opmod.execution_inputs(profile, ["x"])
    with pytest.raises(ValueError, match="non-empty path"):
        opmod.execution_inputs(profile, {"pools": 5})
    with pytest.raises(ValueError, match="unreadable"):
        opmod.systemd_state("u", lambda argv: ["x"])


def test_prepare_rejects_malformed_budget_and_manifest(operator):
    with pytest.raises(ValueError, match="must be an object"):
        operator.prepare("budgeted", {}, intervention_budget="x")
    manifest = operator.prepare("corrupt", {})
    path = operator.manifest_path("corrupt")
    path.write_text("[1, 2]")
    with pytest.raises(ValueError, match="unreadable"):
        operator.prepare("corrupt", {})
    assert manifest["experiment_id"] == "corrupt"


def test_submit_rejects_malformed_state(operator, tmp_path):
    manifest = prepared(operator, "hashes")
    manifest["input_hashes"] = ["x"]
    with pytest.raises(ValueError, match="malformed"):
        operator.submit(manifest)
    manifest = prepared(operator, "owner")
    resources = tmp_path / "resources"
    resources.mkdir(exist_ok=True)
    (resources / "owner.json").write_text("[1]")
    with pytest.raises(ValueError, match="unreadable"):
        operator.submit(manifest)
    (resources / "owner.json").write_text("{}")
    with pytest.raises(ValueError, match="needs unit"):
        operator.submit(manifest)
    (resources / "owner.json").unlink()
    manifest = prepared(operator, "attempts")
    manifest["attempts"] = "x"
    with pytest.raises(ValueError, match="needs attempts"):
        operator.submit(manifest, action="resume")


def test_report_resume_and_intervention_guards(operator):
    manifest = operator.prepare("guards", {})
    assert operator._report_valid(manifest, {"execution.json": ["x"]}) is False
    with pytest.raises(ValueError, match="torch"):
        operator._validate_resume(manifest, manifest["output"])
    with pytest.raises(ValueError, match="checkpoint directory"):
        operator._validate_resume(manifest, 5)
    diagnosed = operator._save(dict(manifest), "diagnose")
    diagnosed["intervention_budget"] = "x"
    with pytest.raises(ValueError, match="malformed"):
        operator.intervene(diagnosed, "adjust_lr", lr=1e-5)
    diagnosed = operator._save(dict(manifest), "diagnose")
    diagnosed["interventions"] = ["x"]
    with pytest.raises(ValueError, match="history is malformed"):
        operator.intervene(diagnosed, "leave_unchanged")


def test_coexistence_and_recipe_helpers_reject_malformed_shapes(tmp_path):
    from siliconshield.research_loop.execution import operator as opmod
    from siliconshield.research_loop.execution import security_jobs as jobs

    assert jobs.qualified_coexistence([], ["x"]) is False
    with pytest.raises(ValueError, match="process identifier"):
        jobs.process_identity(5)
    with pytest.raises(ValueError, match="must name a path"):
        jobs.preservation_teacher_files({"recipe": 5})
    with pytest.raises(ValueError, match="Missing training recipe"):
        jobs.preservation_teacher_files({"recipe": str(tmp_path / "no.json")})
    profile = _job_profile(tmp_path)
    with pytest.raises(ValueError, match="name a path"):
        opmod.execution_inputs(dict(profile, recipe=5), {})
    with pytest.raises(ValueError, match="must be a string"):
        opmod.execution_inputs(dict(profile, reporting_contract=5), {})
