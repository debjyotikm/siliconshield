"""Bounded execution tests use synthetic jobs, never live systemd or a GPU."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from siliconshield.research_loop.execution.operator import (
    Operator,
    ResourceReservation,
    systemd_state,
)


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
