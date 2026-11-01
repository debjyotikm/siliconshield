"""Operator curriculum, contract, episodes, baseline, and dev tests: synthetic fixtures only."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from siliconshield.research_loop.controller.replay import _failure_claims, grade
from siliconshield.research_loop.episodes.contracts import (
    final_payload,
    validate_episode,
)
from siliconshield.research_loop.episodes.operator_assessment import (
    assessment,
    assessment_issues,
    parameter_issues,
)
from siliconshield.research_loop.episodes.operator_baseline import (
    build_baseline,
    evaluate_baseline,
    load_baseline,
)
from siliconshield.research_loop.episodes.operator_curriculum import (
    build as _curriculum_build,
)
from siliconshield.research_loop.episodes.operator_curriculum import (
    episode,
    fixture_verify,
)
from siliconshield.research_loop.episodes.operator_episodes import (
    SCENARIOS,
    episode_for,
    migrate_operator_dataset,
    migrate_operator_row,
    validate_operator_dataset,
)
from siliconshield.research_loop.episodes.operator_episodes import (
    build as _episodes_build,
)
from siliconshield.research_loop.execution.operator import (
    Operator,
    ResourceReservation,
    systemd_state,
)
from siliconshield.training.glimmer.train import load_rows


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


@pytest.mark.parametrize(
    "text,asserts_failure",
    [
        ("This is a learning failure, not an execution failure.", False),
        ("There was no execution failure.", False),
        (
            "This execution failure does not establish unseen-family generalization.",
            True,
        ),
        ("The job crashed. No final-test claim is supported.", True),
    ],
)
def test_failure_negation_is_scoped(text, asserts_failure):
    assert bool(_failure_claims(text)) is asserts_failure


def test_assessment_checks_metric_binding_and_unknowns():
    state = spec()["state"]
    facts = assessment(state)
    assert facts["comparisons"]["development"]["accuracy_delta_pp"] == 10.0
    assert facts["comparisons"]["development"]["nll_delta"] == -0.1
    assert facts["acceptance"] == "unknown"
    assert facts["comparisons"]["development"]["false_safe_delta_pp"] is None
    assert assessment_issues(state, facts) == []
    for value in (float("nan"), float("inf"), True, -10.0):
        wrong = copy.deepcopy(facts)
        wrong["comparisons"]["development"]["accuracy_delta_pp"] = value
        assert assessment_issues(state, wrong)
    state["development_history"]["entries"][-1]["population_id"] = "another"
    assert assessment(state)["comparisons"] == {}
    assert assessment_issues(state, facts)
    for entry in state["development_history"]["entries"]:
        entry.pop("population_id")
    assert assessment(state)["comparisons"] == {}


def test_proposed_parameters_must_match_owned_state():
    row = episode(spec())
    assert parameter_issues(
        row, {"action": "resume", "checkpoint_dir": "/wrong"}
    )
    assert not parameter_issues(row, row["target"])
    row["summary"]["lr_bounds"] = [1e-6, 1e-4]
    row["target"]["lr"] = 2e-5
    assert parameter_issues(row, {"action": "adjust_lr", "lr": 9e-5})
    assert not parameter_issues(row, {"action": "adjust_lr", "lr": 2e-5})


def test_fixture_uses_real_dispatch_and_structured_evidence():
    row = episode(spec())
    fixture_verify(row)
    decision = {
        k: v for k, v in row["target"].items() if k != "require_assessment"
    }
    decision["run_id"] = row["summary"]["run_id"]
    delivered = {
        "decision": decision,
        "read_evidence": True,
        "compared_metrics": True,
        "final": json.dumps(final_payload("resume", run_id=decision["run_id"])),
    }
    assert grade(row, delivered)["verified"]
    decision["assessment"]["selected_step"] = 999
    result = grade(row, delivered)
    assert (
        result["action_correct"]
        and not result["evidence_correct"]
        and not result["verified"]
    )


def test_catalog_family_cannot_cross_split(tmp_path, monkeypatch):
    key = tmp_path / "key"
    key.write_bytes(b"fixture-signing-key")
    monkeypatch.setenv("SILICONSHIELD_DATASET_KEY", str(key))
    train, dev = spec(), spec()
    dev.update(index=1, split="dev")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(json.dumps(train) + "\n" + json.dumps(dev) + "\n")
    with pytest.raises(ValueError, match="crosses dataset splits"):
        _curriculum_build(catalog, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


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
        for folder in ("source/siliconshield/training/laya", "model", "data"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        (
            root / "source/siliconshield/training/laya/security_train.py"
        ).write_text("")
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


def test_input_change_rejected(operator):
    manifest = prepared(operator)
    (Path(operator.profile["data"]) / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        operator.submit(manifest)


def test_foreign_unit_rejected(operator):
    manifest = prepared(operator)
    operator.jobs.units[manifest["unit"]] = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "Description": "unrelated",
    }
    with pytest.raises(RuntimeError, match="another owner"):
        operator.submit(manifest)


def test_resource_owner_survives_submit_call(operator):
    first = prepared(operator, "first")
    operator.submit(first)
    second = prepared(operator, "second")
    with pytest.raises(RuntimeError, match="owned"):
        operator.submit(second)
    assert len(operator.jobs.launches) == 1


def test_failed_and_stale_states():
    for details, expected in (
        ({"ActiveState": "failed", "Result": "exit-code"}, "failed"),
        ({"ActiveState": "inactive", "Result": "success"}, "completed"),
        ({"LoadState": "not-found"}, "missing"),
        ({"ActiveState": "inactive"}, "stale"),
    ):
        assert systemd_state("unit", lambda _: details)["state"] == expected


def test_resource_lock_and_owner(tmp_path):
    first, other = ResourceReservation(tmp_path), ResourceReservation(tmp_path)
    assert first.acquire("one") and first.owner() == "one"
    assert not other.acquire("two")
    first.release()
    assert other.acquire("two")
    other.release()


def test_stop_and_learning_rate_reach_trainer_control(operator):
    from siliconshield.training.laya.operator_control import apply_control

    manifest = prepared(operator)
    operator.submit(manifest)
    operator.observe(manifest)
    operator.diagnose(manifest)
    progress = {}
    operator.intervene(manifest, "adjust_lr", lr=5e-6)
    assert (
        apply_control(
            manifest["output"], manifest["experiment_id"], progress, 1e-5
        )
        == "adjust_lr"
    )
    assert progress["lr_scale"] == 0.5
    assert (
        apply_control(
            manifest["output"], manifest["experiment_id"], progress, 1e-5
        )
        is None
    )
    with pytest.raises(ValueError, match="budget exhausted"):
        operator.intervene(manifest, "adjust_lr", lr=5e-6)
    operator.intervene(manifest, "stop")
    assert (
        apply_control(
            manifest["output"], manifest["experiment_id"], progress, 1e-5
        )
        == "stop"
    )
    with pytest.raises(ValueError, match="not allowlisted"):
        operator.intervene(manifest, "edit_data")


def test_reattach_accepts_code_drift_but_rejects_recipe_change(
    operator, monkeypatch
):
    import siliconshield.research_loop.execution.operator as opmod
    from siliconshield.research_loop.episodes.contracts import atomic_json

    manifest = prepared(operator)
    manifest["attempts"] = 1
    manifest["phase"] = "submit"
    atomic_json(operator.manifest_path("experiment"), manifest)
    monkeypatch.setattr(
        opmod, "idempotency_key", lambda profile, exp, ov: "drifted-key"
    )
    # Same recipe: operator-code drift reattaches to the submitted command.
    again = operator.prepare("experiment", {})
    assert again["command"] == manifest["command"]
    assert any("code drift" in h.get("note", "") for h in again["history"])
    # Changed learning rate: the old recipe must not be silently returned.
    with pytest.raises(ValueError, match="requested recipe differs"):
        operator.prepare("experiment", {"adapter_lr": 1e-3})


def test_empty_report_is_not_a_checkpoint(operator):
    pytest.importorskip("torch")
    manifest = prepared(operator)
    output = Path(manifest["output"])
    (output / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="valid LAYA"):
        operator._validate_resume(manifest, output)
    (output / "test_touch.json").write_text("{}")
    with pytest.raises(ValueError, match="Test-touched"):
        operator._validate_resume(manifest, output)


def test_valid_laya_checkpoint_and_incompatible_data(operator):
    import random

    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    manifest = prepared(operator)
    output = Path(manifest["output"])
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    parameter.sum().backward()
    optimizer.step()
    signature = {
        "run_id": manifest["experiment_id"],
        "world": 3,
        "data": {
            s: hashlib.sha256(
                (Path(operator.profile["data"]) / (s + ".jsonl")).read_bytes()
            ).hexdigest()
            for s in ("train", "dev")
        },
    }
    state = {
        "model": {"weight": parameter.detach()},
        "optimizer": optimizer.state_dict(),
        "signature": signature,
        "progress": {"step": 1, "epoch": 0, "offset": 4},
        "rng": [
            {
                "torch": torch.get_rng_state(),
                "cuda": torch.get_rng_state(),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            }
            for _ in range(3)
        ],
    }
    torch.save(state, output / "latest.pt")
    (output / "configuration.json").write_text(
        json.dumps({"signature": signature})
    )
    assert operator._validate_resume(manifest, output)["step"] == 1
    (Path(operator.profile["data"]) / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="data changed"):
        operator._validate_resume(manifest, output)


def test_finalize_does_not_accept_empty_report(operator):
    manifest = prepared(operator)
    operator.submit(manifest)
    operator.observe(manifest)
    (Path(manifest["output"]) / "report.json").write_text("{}")
    result = operator.finalize(manifest)
    assert result["history"][-1]["finalized"] is False
    assert operator.report(manifest)["terminal"] == "incomplete"


def test_owned_process_writes_actual_exit_and_timeout(tmp_path):
    from siliconshield.research_loop.execution.owned_job import run

    for name, script, timeout, expected in (
        ("success", 'print("executed")', 5, 0),
        ("failure", "raise SystemExit(7)", 5, 1),
        ("timeout", "import time; time.sleep(30)", 0.05, 1),
    ):
        output = tmp_path / name
        output.mkdir()
        manifest = {
            "output": str(output),
            "command": [sys.executable, "-c", script],
            "idempotency_key": name,
            "unit": "fixture-" + name,
            "timeout_seconds": timeout,
            "output_quota_bytes": 10000,
        }
        path = output / "manifest.json"
        path.write_text(json.dumps(manifest))
        assert run(path, tmp_path / "resources", "train") == expected
        receipt = json.loads((output / "execution.json").read_text())
        assert receipt["identity"] == name
        if name == "timeout":
            assert receipt["failure"] == "timeout"
        elif name == "failure":
            assert receipt["exit_code"] == 7


def test_glimmer_training_and_serving_cannot_share_gpu(tmp_path, monkeypatch):
    from siliconshield.research_loop.execution.gpu_lease import acquire_gpu

    monkeypatch.setenv("SILICONSHIELD_GPU_LOCK", str(tmp_path / "gpu.lock"))
    with acquire_gpu():
        with pytest.raises(RuntimeError, match="already owned"):
            acquire_gpu()
    with acquire_gpu() as owned:
        assert not owned.closed


def test_reattach_rejects_changed_data_at_same_path(operator):
    manifest = prepared(operator)
    operator.submit(manifest)
    (Path(operator.profile["data"]) / "train.jsonl").write_text(
        "different data"
    )
    with pytest.raises(ValueError, match="recipe differs"):
        operator.prepare("experiment")
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


def test_recipe_content_alone_moves_the_idempotency_key(operator, tmp_path):
    from siliconshield.research_loop.execution.operator import (
        idempotency_key,
    )

    recipe = tmp_path / "recipe.json"
    operator.profile["recipe"] = str(recipe)
    recipe.write_text('{"a": 1}')
    before = idempotency_key(operator.profile, "fixed", {})
    recipe.write_text('{"a": 2}')
    after = idempotency_key(operator.profile, "fixed", {})
    assert before != after


def test_separate_teacher_bytes_bind_identity_and_refuse_drift(
    operator, tmp_path
):
    from siliconshield.research_loop.execution.operator import (
        idempotency_key,
    )

    teacher = tmp_path / "teacher"
    teacher.mkdir()
    weights = teacher / "model.safetensors"
    weights.write_bytes(b"authored teacher weights")
    (teacher / "rl_agent_config.json").write_text("{}")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {"replay_preservation": {"teacher_checkpoint": str(teacher)}}
        )
    )
    operator.profile["recipe"] = str(recipe)
    before = idempotency_key(operator.profile, "teacher-bound", {})
    manifest = operator.prepare("teacher-bound")
    assert str(weights) in manifest["execution_inputs"]
    weights.write_bytes(b"changed teacher weights")
    after = idempotency_key(operator.profile, "teacher-bound", {})
    assert before != after
    with pytest.raises(ValueError, match="changed"):
        operator.preflight(manifest)
    assert not operator.jobs.launches


def test_incomplete_separate_teacher_refuses_preparation(operator, tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "replay_preservation": {
                    "teacher_checkpoint": str(tmp_path / "missing")
                }
            }
        )
    )
    operator.profile["recipe"] = str(recipe)
    with pytest.raises(ValueError, match="Incomplete preservation teacher"):
        operator.prepare("missing-teacher")


def test_checkpoint_and_diagnostic_flags_reach_command():
    from siliconshield.research_loop.execution.security_jobs import (
        command_for,
    )

    profile = {
        "python": sys.executable,
        "source": "src",
        "model": "model",
        "data": "data",
        "runs": "runs",
        "defaults": {
            "max_updates": 10,
            "checkpoint_interval": 300,
            "no_diagnostic_weights": True,
            "triplet_shards": True,
        },
    }
    command, _ = command_for(profile, "exp-1", {})
    assert command[command.index("--checkpoint-interval") + 1] == "300"
    assert "--no-diagnostic-weights" in command
    assert "--triplet-shards" in command
    profile["defaults"]["triplet_shards"] = False
    command, _ = command_for(profile, "exp-1", {})
    assert "--triplet-shards" not in command


# ---- Merged from test_operator_episodes.py (aliased: build) ----
def test_all_scenarios_produce_valid_episodes():
    for scenario, state, resources, action, reason in SCENARIOS:
        row = episode_for(
            scenario, state, resources, action, reason, "lineage-x", 0
        )
        validate_episode(row)
        visible = json.dumps(row["messages"])
        assert row["outcome"]["recorded_only"] and "outcome" not in visible


def test_qualification_dataset_acceptance_and_splits(tmp_path):
    from siliconshield.schemas import OPERATOR_EPISODES_SCHEMA

    manifest = _episodes_build(tmp_path / "data", lineages_per_scenario=1)
    assert manifest["schema"] == OPERATOR_EPISODES_SCHEMA
    assert sum(manifest["rows"].values()) == len(SCENARIOS)
    rows = []
    owners = {}
    for split in ("train", "dev", "test"):
        loaded = load_rows(
            tmp_path / "data" / (split + ".jsonl"), allowed_splits=(split,)
        )
        rows.extend(loaded)
        for row in loaded:
            assert (
                row["group_id"] not in owners
                or owners[row["group_id"]] == split
            )
            owners[row["group_id"]] = split
    report = validate_operator_dataset(rows)
    assert report["accepted"], report["issues"]
    assert set(report["actions"]) >= {
        "continue",
        "leave_unchanged",
        "stop",
        "adjust_lr",
        "resume",
        "finalize_retry",
    }


def test_leakage_scan_catches_future_fields():
    scenario, state, resources, action, reason = SCENARIOS[0]
    row = episode_for(
        scenario, state, resources, action, reason, "lineage-y", 0
    )
    row["messages"][1]["content"] += " future_metric: improved"
    report = validate_operator_dataset([row] * 3)
    assert not report["accepted"]
    assert any("future_" in issue for issue in report["issues"])


def test_duplicate_scenarios_are_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="not independent"):
        _episodes_build(tmp_path, lineages_per_scenario=6)
    args = SCENARIOS[0]
    row = episode_for(*args, "lineage", 0)
    assert not validate_operator_dataset([row, row])["accepted"]


def test_no_scenario_crosses_splits(tmp_path):
    _episodes_build(tmp_path)
    ownership = {}
    for split in ("train", "dev", "test"):
        for row in load_rows(
            tmp_path / (split + ".jsonl"), allowed_splits=(split,)
        ):
            name = row["verification"]["scenario"]
            assert ownership.setdefault(name, split) == split


def test_coverage_distinguishes_copies_from_operator_signal():
    from siliconshield.research_loop.episodes.operator_episodes import (
        operator_coverage,
    )

    train = [episode_for(*SCENARIOS[0], "group-%d" % i, i) for i in range(600)]
    dev = [episode_for(*s, "dev-" + s[0], 0) for s in SCENARIOS]
    report = operator_coverage(train, dev, dev[:1])
    assert report["train"]["operator_rows"] == 600
    assert report["train"]["distinct_operator_situations"] == 1
    assert "resume" in report["train"]["missing_actions"]
    assert "adjust_lr" in report["replay"]["missing_actions"]
    assert not report["coverage_passed"]


def test_operator_training_rejects_qualification_dataset_before_model_load(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import pytest

    from siliconshield.training.glimmer import train

    rows = [episode_for(*s, "group-" + s[0], 0) for s in SCENARIOS]
    monkeypatch.setattr(train, "training_rows", lambda data: (rows, rows))
    args = SimpleNamespace(
        data=tmp_path,
        output=tmp_path,
        dev_examples=16,
        purpose="operator",
        task_eval_every=100,
    )
    with pytest.raises(ValueError, match="Operator training requires"):
        train._train(args, "a" * 64)
    receipt = json.loads((tmp_path / "operator-coverage.json").read_text())
    assert not receipt["coverage_passed"] and receipt["purpose"] == "operator"


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


def test_migration_fills_resume_and_adjust_lr_fields():
    for scenario_name, field in (
        ("valid_resume", "checkpoint_dir"),
        ("lr_instability", "lr"),
    ):
        row = _old_style_row(scenario_name)
        try:
            validate_episode(row)
            raise AssertionError("old row should fail the current contract")
        except ValueError:
            pass
        migrated, outcome = migrate_operator_row(row)
        assert outcome["status"] == "migrated", outcome
        assert field in migrated["target"]
        validate_episode(migrated)
        assert field in migrated["verification"]["migration"]["filled"]


def test_migration_rejects_obsolete_actions_without_remapping():
    row = _old_style_row("valid_resume")
    for message in row["messages"]:
        for call in message.get("tool_calls", []):
            if call["function"]["name"] == "operator.decide":
                call["function"]["arguments"]["action"] = "investigate_data"
    row["target"]["action"] = "investigate_data"
    migrated, outcome = migrate_operator_row(row)
    assert migrated is None and outcome["status"] == "rejected"


def test_migrate_dataset_preserves_sealed_test(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    (source).mkdir()
    rows = [_old_style_row("valid_resume"), _old_style_row("lr_instability")]
    (source / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (source / "dev.jsonl").write_text("")
    test_payload = '{"sealed": true}\n'
    (source / "test.jsonl").write_text(test_payload)
    (source / "manifest.json").write_text(
        json.dumps({"schema": "x", "groups": {"g": "test"}, "files": {}})
    )
    report = migrate_operator_dataset(source, dest)
    assert report["migrated"] == 2 and not report["rejected"]
    assert (dest / "test.jsonl").read_text() == test_payload
    assert report["test_rows_copied"] == 1


def test_fabricated_execution_failure_is_not_verified():
    import tempfile

    from siliconshield.research_loop.controller.replay import execute, grade

    scenario, state, resources, action, reason = next(
        s for s in SCENARIOS if s[0] == "healthy_progressing"
    )
    row = episode_for(scenario, state, resources, action, reason, "lin-fab", 0)

    class TraceClient:
        model = "fixture"

        def complete(self, messages, tools=()):
            return row["messages"][len(messages)]

    with tempfile.TemporaryDirectory() as attempt:
        delivered = execute(row, TraceClient(), attempt)
    assert grade(row, delivered)["verified"] is True
    bad = copy.deepcopy(delivered)
    bad["decision"] = dict(
        bad["decision"],
        reason=(
            "This execution failure is a loading problem. Stop the run; retrying will repeat the failure."
        ),
    )
    result = grade(row, bad)
    assert result["action_correct"] is True
    assert result["explanation_correct"] is False
    assert result["verified"] is False
    assert (
        "explanation_contradicts_execution_evidence_in_reason"
        in result["explanation_issues"]
    )


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


def test_structured_claims_reject_false_acceptance_train_loss_proof_and_phantom_lr():
    from siliconshield.research_loop.controller.replay import grade

    row = _operator_row(_failed_acceptance_summary())
    delivered = _operator_delivered(
        "All acceptance checks passed. Near-zero training loss proves generalization. "
        "The plateau controller reduced the learning rate twice."
    )
    result = grade(row, delivered)
    assert result["action_correct"] is True
    assert result["explanation_correct"] is False
    assert result["verified"] is False
    assert (
        "explanation_claims_acceptance_despite_failed_criteria_in_reason"
        in result["explanation_issues"]
    )
    assert (
        "explanation_train_loss_proves_quality_in_reason"
        in result["explanation_issues"]
    )
    assert (
        "explanation_plateau_reduction_differs_from_record_in_reason"
        in result["explanation_issues"]
    )


def test_supported_acceptance_and_denials_pass_grading():
    from siliconshield.research_loop.controller.replay import grade

    accepted = {
        "run_id": "repro-1",
        "systemd": "running",
        "step": 200,
        "execution_failed": False,
        "acceptance": {"accepted": True, "reasons": []},
        "development_history": {
            "entries": [
                {"step": 200, "metrics": {"accuracy": 0.94, "nll": 0.36}}
            ],
            "complete": True,
            "total": 1,
        },
        "schedule": {"plateau_lr_reductions": []},
    }
    row = _operator_row(accepted)
    delivered = _operator_delivered(
        "Acceptance passed on every criterion; continuing."
    )
    assert grade(row, delivered)["verified"] is True
    # Hedged or negated phrasing is not a claim.
    row2 = _operator_row(_failed_acceptance_summary())
    delivered2 = _operator_delivered(
        "The training loss does not prove anything yet."
    )
    assert grade(row2, delivered2)["explanation_correct"] is True
    delivered3 = _operator_delivered(
        "No plateau reduction occurred; lr_scale is still 1.0."
    )
    assert (
        grade(row3 := _operator_row(_failed_acceptance_summary()), delivered3)[
            "explanation_correct"
        ]
        is True
    )


def test_plateau_denial_requires_proximity_not_sentence_match():
    from siliconshield.research_loop.controller.replay import (
        _claimed_plateau_reductions,
        grade,
    )

    # Leading denial suppresses the claim.
    assert (
        _claimed_plateau_reductions(
            "No plateau reduction occurred; lr_scale is still 1.0."
        )
        == []
    )
    # Trailing denial in another clause must not suppress an earlier real claim.
    trailing = _claimed_plateau_reductions(
        "The plateau controller reduced the learning rate twice; no further action is needed."
    )
    assert trailing and trailing[0][1] == 2
    row = _operator_row(_failed_acceptance_summary())
    result = grade(
        row,
        _operator_delivered(
            "The plateau controller reduced the learning rate twice; no further action is needed."
        ),
    )
    assert (
        "explanation_plateau_reduction_differs_from_record_in_reason"
        in result["explanation_issues"]
    )


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


def test_baseline_covers_all_six_actions_with_held_out_lineages(tmp_path):
    from siliconshield.schemas import OPERATOR_BASELINE_SCHEMA

    manifest = build_baseline(tmp_path / "base")
    assert manifest["schema"] == OPERATOR_BASELINE_SCHEMA
    assert (
        manifest["rows"] == 9
        and manifest["never_train"]
        and manifest["regression_checks"]
    )
    assert not manifest["held_out"]
    assert set(manifest["actions"]) >= {
        "continue",
        "leave_unchanged",
        "stop",
        "adjust_lr",
        "resume",
        "finalize_retry",
    }
    rows = load_baseline(tmp_path / "base/baseline.jsonl")
    assert all(r["group_id"].startswith("baseline-heldout-") for r in rows)
    # Overlap guard: shared history is regression-check only, never training.
    assert manifest["regression_check_subset"] == ["regression_stop_supported"]
    assert manifest["overlap"]["regression_stop_supported"] == [
        "dpo-stop-grounding-v1",
        "dpo-baseline-stop-supported",
    ]


def test_baseline_grading_separates_action_from_explanation(tmp_path):
    build_baseline(tmp_path / "base")
    rows = load_baseline(tmp_path / "base/baseline.jsonl")
    summary = evaluate_baseline(
        _fixture_client(rows),
        tmp_path / "base/baseline.jsonl",
        tmp_path / "eval",
    )
    assert summary["action_correct"] == 9 and summary["evidence_correct"] == 9
    assert summary["verified"] == 6
    by_scenario = {r["scenario"]: r for r in summary["results"]}
    assert by_scenario["regression_stop_supported"]["verified"] is True
    assert (
        by_scenario["healthy_continue_fabricated_failure"]["verified"] is False
    )
    assert (
        "explanation_contradicts_execution_evidence_in_reason"
        in by_scenario["healthy_continue_fabricated_failure"][
            "explanation_issues"
        ]
    )
    assert (
        "explanation_metric_not_observed_in_reason"
        in by_scenario["stop_invented_numbers"]["explanation_issues"]
    )
    assert (
        by_scenario["stop_with_unsupported_failure_excuse"]["verified"] is False
    )


def test_baseline_rejects_non_held_out_rows(tmp_path):
    from siliconshield.research_loop.episodes.operator_episodes import (
        SCENARIOS,
        episode_for,
    )

    scenario = SCENARIOS[0]
    row = episode_for(
        scenario[0],
        scenario[1],
        scenario[2],
        scenario[3],
        scenario[4],
        "train-x",
        0,
    )
    path = tmp_path / "other.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="held-out"):
        load_baseline(path)


def test_shared_history_marked_regression_only():
    from siliconshield.training.glimmer.dpo_pairs import (
        REGRESSION_ONLY_IDS,
        corrected_stop_pair,
        is_regression_only,
        same_action_pair,
        training_pairs,
        validate_pair,
    )

    assert REGRESSION_ONLY_IDS == {
        "dpo-stop-grounding-v1",
        "dpo-baseline-stop-supported",
    }
    overlap = corrected_stop_pair()
    assert validate_pair(overlap) == []
    assert is_regression_only(overlap) is True
    assert (
        overlap["provenance"]["overlap"] == "baseline regression_stop_supported"
    )
    # Same ID via same_action_pair is also regression-only by construction.
    from siliconshield.research_loop.episodes.operator_baseline import (
        BASELINE_SCENARIOS,
    )

    sc = next(
        s for s in BASELINE_SCENARIOS if s[0] == "regression_stop_supported"
    )
    state = dict(sc[1], run_id="run-baseline")
    res = dict(sc[2], run_id="run-baseline")
    dup = same_action_pair(
        "run-baseline",
        state,
        res,
        "stop",
        "Development accuracy fell from 82.41% at step 202 to 80.22% at step 302, while NLL "
        "increased from 0.0422 to 0.0533. Stop under the configured regression rule and preserve "
        "the selected checkpoint. No execution failure was observed.",
        "The execution layer is failing, so stop the run before more damage is done.",
        "dpo-baseline-stop-supported",
        "baseline regression_stop_supported",
    )
    assert validate_pair(dup) == []
    assert is_regression_only(dup) is True
    assert len(training_pairs([overlap, dup])) == 0


# ---- Merged from test_operator_dev.py ----
def _norm_state(state):
    import copy

    state = copy.deepcopy(state)
    state.pop("run_id", None)
    return json.dumps(state, sort_keys=True)


def test_dev_builds_independent_synthetic_rows(tmp_path):
    from siliconshield.research_loop.episodes.operator_dev import (
        build_dev,
        load_dev,
    )
    from siliconshield.schemas import OPERATOR_DEV_SCHEMA

    manifest = build_dev(tmp_path / "dev")
    assert manifest["schema"] == OPERATOR_DEV_SCHEMA
    assert manifest["rows"] == 6 and manifest["independent"]
    assert not manifest["held_out"]
    rows = load_dev(tmp_path / "dev/dev.jsonl")
    assert all(r["group_id"].startswith("dev-independent-") for r in rows)


def test_dev_disjoint_from_baseline_and_training_states():
    from siliconshield.research_loop.episodes.operator_baseline import (
        BASELINE_SCENARIOS,
    )
    from siliconshield.research_loop.episodes.operator_dev import DEV_SCENARIOS

    baseline_states = {_norm_state(sc[1]) for sc in BASELINE_SCENARIOS}
    dev_states = {_norm_state(sc[1]) for sc in DEV_SCENARIOS}
    assert not (baseline_states & dev_states)
    # Training eligible states (synthetic distinct set) must also avoid both.
    # Spot-check the previously overlapping shapes are now distinct:
    # baseline uses step 40/60/200/300, grad 42.0, exit 1; dev uses 57/63/123/211/311.
    dev_steps = set()
    for _, state, _, _, _, _ in DEV_SCENARIOS:
        if isinstance(state.get("step"), int):
            dev_steps.add(state["step"])
        for entry in state.get("development_history", {}).get("entries", []):
            dev_steps.add(entry["step"])
    assert dev_steps.isdisjoint({40, 60, 200, 300})
    assert 57 in dev_steps and 63 in dev_steps
