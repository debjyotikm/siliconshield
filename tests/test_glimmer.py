"""Glimmer selection, serving, campaign, jobs, and operator tests: synthetic fixtures only. Torch- and harness2-gated suites stay in their own files so their module-level skips keep working, and the contracts module stays put because serving and harness2 import its helpers."""

import builtins
import copy
import hashlib
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from siliconshield.research_loop.controller.serve import parse_completion
from siliconshield.research_loop.episodes.contracts import (
    OPERATOR_ACTIONS,
    TOOLS,
    atomic_json,
    digest,
    read_json,
)
from siliconshield.research_loop.episodes.operator_episodes import (
    SCENARIOS,
    episode_for,
)
from siliconshield.research_loop.execution import campaign
from siliconshield.research_loop.execution.glimmer_operator import (
    OperatorTools,
    run_cycle,
)
from siliconshield.research_loop.execution.operator import Operator
from siliconshield.research_loop.execution.security_jobs import (
    command_for,
    laya_root_for,
    service_command,
    verify_laya,
)
from siliconshield.research_loop.execution.strategy import (
    assess_fitting,
    fitting_plan,
)
from siliconshield.training.glimmer.identity import checkpoint_fingerprint
from siliconshield.training.glimmer.selection import (
    METRICS,
    compare_replay,
    select_replay_candidate,
)
from tests.test_glimmer_contracts import ScriptedClient, episode


# ---- Merged from test_glimmer_selection.py ----
def report() -> dict:
    """Create a small, complete authored operator population."""
    rows = [
        {
            "episode_id": action + str(i),
            "group_id": action,
            "expected_action": action,
            "checkpoint_review": action == "leave_unchanged",
            "execution_complete": True,
            "execution_error": None,
            **dict.fromkeys(METRICS, i == 0),
        }
        for action in OPERATOR_ACTIONS
        for i in range(2)
    ]
    result = {
        "n": len(rows),
        "results": rows,
        "valid_execution": True,
        "test_evaluated": False,
        "source_identity": "source",
        "evaluator_sha256": "evaluator",
        "dataset_manifest_sha256": "dataset",
        "decoding": {"temperature": 0},
    }
    recount(result)
    return result


def recount(value: dict) -> None:
    """Recompute fixture totals after deliberate outcome changes."""
    value["totals"] = {
        key: sum(r[key] for r in value["results"]) for key in METRICS
    }


def outcome(value: dict, identity: str, passed: bool) -> None:
    """Change one fixture episode, preserving accurate summary counts."""
    next(r for r in value["results"] if r["episode_id"] == identity).update(
        dict.fromkeys(METRICS, passed)
    )
    recount(value)


def test_aggregate_gain_does_not_hide_stop_regression() -> None:
    parent = report()
    child = copy.deepcopy(parent)
    outcome(child, "stop0", False)
    outcome(child, "finalize_retry1", True)
    outcome(child, "resume1", True)
    result = compare_replay(parent, child)
    assert result["verified_gain"] == 1
    assert not result["passed"]
    assert "action:stop:action_correct" in result["issues"]


def test_preserve_both_reference_strengths_and_require_improvement() -> None:
    serving = report()
    trained = copy.deepcopy(serving)
    outcome(trained, "stop0", False)
    outcome(trained, "finalize_retry1", True)
    repaired = copy.deepcopy(trained)
    outcome(repaired, "stop0", True)
    result = select_replay_candidate(
        {"serving": serving, "trained": trained},
        {"unchanged": trained, "repaired": repaired},
    )
    assert result["selected"] == "repaired"
    assert result["deployed"] is False
    assert (
        select_replay_candidate({"base": serving}, {"same": serving})[
            "selected"
        ]
        is None
    )


@pytest.mark.parametrize("problem", ["duplicate", "missing", "total", "bool"])
def test_malformed_reports_refuse(problem: str) -> None:
    parent, child = report(), report()
    if problem == "duplicate":
        child["results"][0]["episode_id"] = child["results"][1]["episode_id"]
    elif problem == "missing":
        child["results"] = child["results"][:-2]
        child["n"] -= 2
        recount(child)
    elif problem == "total":
        child["totals"]["verified"] += 1
    else:
        child["results"][0]["verified"] = "true"
    with pytest.raises(ValueError):
        compare_replay(parent, child)


@pytest.mark.parametrize(
    "key", ["source_identity", "decoding", "test_evaluated"]
)
def test_identity_and_test_boundaries_refuse(key: str) -> None:
    parent, child = report(), report()
    child[key] = True if key == "test_evaluated" else "changed"
    with pytest.raises(ValueError):
        compare_replay(parent, child)


def test_execution_errors_refuse_even_with_passing_totals() -> None:
    parent, child = report(), report()
    child["results"][0]["execution_error"] = "transport failed"
    assert not compare_replay(parent, child)["passed"]


# ---- Merged from test_glimmer_serving.py ----
def test_native_atem_call_and_reasoning_are_preserved():
    text = (
        "<|start|>assistant to=self<|message|>Read the source first.<|eom|>"
        "<|start|>assistant to=research.read_run<|message|><atem:function_calls>"
        '<atem:invoke name="research.read_run"><atem:parameter name="run_id">example</atem:parameter>'
        "</atem:invoke></atem:function_calls><|eot|>"
    )
    message = parse_completion(text, TOOLS)
    assert message["tool_calls"][0]["function"]["arguments"] == {
        "run_id": "example"
    }
    assert message["reasoning_content"] == "Read the source first."


def test_unknown_function_cannot_be_dispatched():
    with pytest.raises(ValueError, match="unknown tool"):
        parse_completion(
            '<atem:invoke name="shell"><atem:parameter name="cmd">oops</atem:parameter>'
            "</atem:invoke>",
            TOOLS,
        )


def test_incomplete_call_and_plain_prose_are_not_successful_tool_use():
    with pytest.raises(ValueError):
        parse_completion(
            '<atem:invoke name="research.read_run"></atem:invoke>', TOOLS
        )
    with pytest.raises(ValueError):
        parse_completion("I called the tool and it worked.", TOOLS)


def test_final_user_answer():
    message = parse_completion(
        "<|start|>assistant to=user<|message|>The evidence supports further checks.<|eot|>",
        TOOLS,
    )
    assert message["content"] == "The evidence supports further checks."


def test_checkpoint_identity_preserves_existing_digest(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    contents = {
        "adapter_config.json": b'{"r": 16}',
        "adapter_model.safetensors": b"authored-fixture-weights",
    }
    for name, content in contents.items():
        (adapter / name).write_bytes(content)
    files = {
        name: hashlib.sha256(value).hexdigest()
        for name, value in contents.items()
    }
    assert checkpoint_fingerprint(adapter, "fixture-base") == digest(
        {"base_revision": "fixture-base", "adapter_files": files}
    )
    assert checkpoint_fingerprint(None, "fixture-base") == digest(
        {"base_revision": "fixture-base", "adapter_files": {}}
    )
    with pytest.raises(ValueError, match="does not exist"):
        checkpoint_fingerprint(tmp_path / "missing")


@pytest.mark.parametrize("reference_adapter", [False, True])
def test_evaluation_runs_without_harness2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_adapter: bool
) -> None:
    """Exercise both evaluation arms while forbidding optional imports."""
    from siliconshield.research_loop.controller import evaluate
    from siliconshield.research_loop.execution import gpu_lease
    from siliconshield.training.glimmer import identity

    original_import = builtins.__import__

    def without_harness(name, *args, **kwargs):
        if "harness2" in name:
            raise ModuleNotFoundError("Harness2 must remain optional")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_harness)
    model = SimpleNamespace(
        load_adapter=lambda *a, **k: None,
        set_adapter=lambda *a: None,
        eval=lambda: None,
        disable_adapter=nullcontext,
    )
    factory = SimpleNamespace(from_pretrained=lambda *a, **k: model)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(bfloat16="fixture-bf16")
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=factory))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=factory, MuseGlimmerForConditionalGeneration=factory
        ),
    )
    monkeypatch.setattr(
        gpu_lease, "acquire_gpu", lambda: SimpleNamespace(closed=False)
    )
    monkeypatch.setattr(
        identity,
        "verify_snapshot",
        lambda *a, **k: {"fingerprint": "fixture-base"},
    )
    monkeypatch.setattr(evaluate, "load_rows", lambda *a, **k: [episode()])
    monkeypatch.setattr(
        evaluate, "InProcessClient", lambda *a, **k: ScriptedClient()
    )
    execute = evaluate.execute

    def check_progress(row, client, path):
        status = json.loads((path.parents[1] / "status.json").read_text())
        assert status["phase"] == "evaluating"
        assert status["completed"] == 0
        return execute(row, client, path)

    monkeypatch.setattr(evaluate, "execute", check_progress)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}')
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    episodes = tmp_path / "dev.jsonl"
    episodes.write_text(json.dumps(episode()) + "\n")
    output = tmp_path / "result"
    argv = [
        "evaluate",
        "--model",
        "fixture",
        "--adapter",
        str(adapter),
        "--episodes",
        str(episodes),
        "--output",
        str(output),
        "--examples",
        "1",
    ]
    if reference_adapter:
        argv += ["--reference-adapter", str(adapter)]
    monkeypatch.setattr(sys, "argv", argv)
    evaluate.main()
    report = json.loads((output / "report.json").read_text())
    reference = "reference" if reference_adapter else "base"
    assert set(report["arms"]) == {reference, "adapter"}
    assert all(arm["verified"] == 1 for arm in report["arms"].values())
    assert report["final_evaluation"] is False


def test_requested_harness2_fails_before_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from siliconshield.research_loop.controller import evaluate
    from siliconshield.research_loop.execution import gpu_lease

    original_import = builtins.__import__

    def without_harness(name, *args, **kwargs):
        if "harness2" in name:
            raise ModuleNotFoundError("Harness2 unavailable in fixture")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_harness)
    monkeypatch.setattr(
        gpu_lease,
        "acquire_gpu",
        lambda: pytest.fail("GPU must not be acquired"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--model",
            "fixture",
            "--adapter",
            "fixture",
            "--episodes",
            "fixture",
            "--output",
            "fixture",
            "--harness2-output",
            "fixture",
        ],
    )
    with pytest.raises(ModuleNotFoundError, match="Harness2 unavailable"):
        evaluate.main()


# ---- Merged from test_glimmer_campaign.py ----
def arguments(root):
    return SimpleNamespace(
        root=str(root),
        run="fixture",
        training_container="owned-training",
        evaluation_container="owned-evaluation",
        image="fixture-image",
        examples=10,
    )


def test_failed_training_never_starts_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign.subprocess, "check_output", lambda *a, **k: "1\n"
    )
    with pytest.raises(RuntimeError, match="Training container exited"):
        campaign.follow(arguments(tmp_path))
    assert (
        read_json(tmp_path / "campaign-status.json")["phase"]
        == "waiting_for_training"
    )


def test_zero_exit_still_requires_complete_training_receipt(
    tmp_path, monkeypatch
):
    atomic_json(
        tmp_path / "runs/fixture/status.json",
        {"phase": "paused_after_preflight"},
    )
    monkeypatch.setattr(
        campaign.subprocess, "check_output", lambda *a, **k: "0\n"
    )
    with pytest.raises(
        RuntimeError, match="without a complete training result"
    ):
        campaign.follow(arguments(tmp_path))


def test_completed_training_launches_and_collects_replays(
    tmp_path, monkeypatch
):
    key = tmp_path / "fixture.key"
    key.write_text("authored-fixture")
    monkeypatch.setenv("SILICONSHIELD_DATASET_KEY", str(key))
    root = tmp_path / "runs/fixture"
    atomic_json(root / "status.json", {"phase": "training_complete"})
    atomic_json(
        root / "generation-replay/report.json",
        {"arms": {"adapter": {"analysis_correct": 7}}},
    )
    atomic_json(root / "harness2-replay/report.json", {"verified": 1})
    waits, launches = [], []

    def wait(command, **kwargs):
        waits.append(command[-1])
        return "0\n"

    def run(command, **kwargs):
        if command[:2] == ["docker", "inspect"]:
            return SimpleNamespace(returncode=1, stdout="")
        if command[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=0, stdout="sha256:fixture-image")
        launches.append(command)
        return SimpleNamespace(returncode=0, stdout="container-id")

    monkeypatch.setattr(campaign.subprocess, "check_output", wait)
    monkeypatch.setattr(campaign.subprocess, "run", run)
    campaign.follow(arguments(tmp_path))
    assert waits == ["owned-training", "owned-evaluation"]
    assert (
        len(launches) == 1
        and "siliconshield.research_loop.controller.evaluate" in launches[0]
    )
    assert "--harness2-output" in launches[0] and "--network" in launches[0]
    assert (
        "SILICONSHIELD_GPU_LOCK=/workspace/runtime/glimmer-gpu.lock"
        in launches[0]
    )
    assert "SILICONSHIELD_IMAGE_DIGEST=sha256:fixture-image" in launches[0]
    assert f"{key}:/workspace/dataset-key:ro" in launches[0]
    assert "SILICONSHIELD_DATASET_KEY=/workspace/dataset-key" in launches[0]
    assert (
        "fixture-image" not in launches[0]
        and "sha256:fixture-image" in launches[0]
    )
    result = read_json(tmp_path / "campaign-status.json")
    assert (
        result["phase"] == "complete"
        and not result["automatic_laya_training_started"]
    )
    assert result["harness2"]["verified"] == 1


def test_existing_evaluator_must_belong_to_this_run(tmp_path, monkeypatch):
    atomic_json(
        tmp_path / "runs/fixture/status.json", {"phase": "training_complete"}
    )
    monkeypatch.setattr(
        campaign.subprocess, "check_output", lambda *a, **k: "0\n"
    )
    record = {"Config": {"Labels": {"laya.glimmer.run": "different-run"}}}
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=json.dumps([record])
        ),
    )
    with pytest.raises(RuntimeError, match="owned by another run"):
        campaign.follow(arguments(tmp_path))


def test_duplicate_watcher_rejected_before_waiting(tmp_path, monkeypatch):
    import fcntl

    with (tmp_path / ".glimmer-campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(
            campaign.subprocess,
            "check_output",
            lambda *a, **k: pytest.fail("Duplicate watcher reached Docker"),
        )
        with pytest.raises(RuntimeError, match="already owns"):
            campaign.follow(arguments(tmp_path))


# ---- Merged from test_glimmer_jobs.py ----
PROFILE = {
    "python": "/env/bin/python",
    "source": "/source",
    "model": "/model",
    "data": "/data",
    "runs": "/runs",
    "defaults": {"max_updates": 100, "seed": 41},
}


def test_training_bridge_uses_three_gpus_and_development_only():
    command, output = command_for(PROFILE, "experiment-a", {"head_lr": 0.0001})
    assert "--nproc_per_node=3" in command
    assert "--no-final-evaluation" in command
    assert str(output) == "/runs/experiment-a"
    assert command[command.index("--max-updates") + 1] == "100"


def test_single_gpu_profile_launches_one_rank():
    profile = {**PROFILE, "gpus": [2]}
    command, _ = command_for(profile, "experiment-a", {})
    assert "--nproc_per_node=1" in command
    assert "--nproc_per_node=3" not in command


@pytest.mark.parametrize("gpus", [[0, 1], [0, 1, 2, 3], [], [-1], [True]])
def test_other_gpu_counts_are_rejected(gpus):
    with pytest.raises(ValueError):
        command_for({**PROFILE, "gpus": gpus}, "experiment-a", {})


@pytest.mark.parametrize(
    "name,overrides",
    [
        ("../escape", {}),
        ("okay", {"shell": "anything"}),
        ("okay", {"max_updates": -1}),
        ("okay", {"encoder_lr": float("nan")}),
        ("okay", {"seed": True}),
    ],
)
def test_invalid_recipe_cannot_reach_process_launch(name, overrides):
    with pytest.raises(ValueError):
        command_for(PROFILE, name, overrides)


def test_laya_root_is_validated_and_forwarded_to_service(tmp_path, monkeypatch):
    (tmp_path / "laya").mkdir()
    (tmp_path / "laya/__init__.py").write_text("")
    profile = {**PROFILE, "laya_root": str(tmp_path)}
    command, _ = command_for(profile, "experiment-a", {})
    argv = service_command(
        profile, "experiment-a", command, laya_root_for(profile)
    )
    assert "--setenv=LAYA_ROOT=" + str(tmp_path) in argv
    assert "--setenv=CUDA_VISIBLE_DEVICES=0,1,2" in argv
    assert argv[-len(command) :] == command
    monkeypatch.setenv("LAYA_ROOT", "/nonexistent-checkout")
    with pytest.raises(ValueError, match="no laya/ package"):
        laya_root_for(PROFILE)
    assert verify_laya({**PROFILE, "python": "/usr/bin/python3"}, str(tmp_path))


def test_continuation_carries_views_schedule_and_study_manifest(tmp_path):
    views = tmp_path / "views.json"
    views.write_text("{}")
    study = tmp_path / "study.json"
    study.write_text("{}")
    command, _ = command_for(
        PROFILE,
        "experiment-a",
        {
            "development_views": str(views),
            "new_study_manifest": str(study),
            "lr_schedule": "cosine",
            "sampling": "grouped",
            "continue_training": True,
        },
    )
    assert command[command.index("--development-views") + 1] == str(views)
    assert command[command.index("--new-study-manifest") + 1] == str(study)
    assert command[command.index("--lr-schedule") + 1] == "cosine"
    assert command[command.index("--sampling") + 1] == "grouped"
    assert "--continue-training" in command


@pytest.mark.parametrize(
    "overrides",
    [
        {"lr_schedule": "exponential"},
        {"sampling": "random"},
        {"development_views": ""},
        {"development_views": 42},
        {"new_study_manifest": 42},
    ],
)
def test_invalid_continuation_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        command_for(PROFILE, "experiment-a", overrides)


def test_hold_decay_schedule_flags_reach_trainer():
    command, _ = command_for(
        PROFILE,
        "experiment-a",
        {"lr_schedule": "hold-decay", "hold_updates": 700, "lr_endpoint": 0.1},
    )
    assert command[command.index("--lr-schedule") + 1] == "hold-decay"
    assert command[command.index("--hold-updates") + 1] == "700"
    assert command[command.index("--lr-endpoint") + 1] == "0.1"


@pytest.mark.parametrize(
    "overrides",
    [
        {"hold_updates": -1},
        {"hold_updates": 1.5},
        {"lr_endpoint": 0.0},
        {"lr_endpoint": 1.5},
        {"lr_endpoint": "0.1"},
    ],
)
def test_invalid_hold_decay_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        command_for(PROFILE, "experiment-a", overrides)


def test_encoder_trainable_blocks_reaches_trainer():
    command, _ = command_for(
        PROFILE, "experiment-a", {"encoder_trainable_blocks": 8}
    )
    assert command[command.index("--encoder-trainable-blocks") + 1] == "8"


@pytest.mark.parametrize(
    "overrides",
    [
        {"encoder_trainable_blocks": 0},
        {"encoder_trainable_blocks": -2},
        {"encoder_trainable_blocks": 2.5},
        {"encoder_trainable_blocks": True},
    ],
)
def test_invalid_encoder_block_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        command_for(PROFILE, "experiment-a", overrides)


def test_missing_development_views_file_blocks_submit(tmp_path):
    import json

    from siliconshield.research_loop.execution.security_jobs import SecurityJobs

    source = tmp_path / "source/siliconshield/training/laya"
    source.mkdir(parents=True)
    (source / "security_train.py").write_text("")
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text("{}\n")
    (data / "dev.jsonl").write_text("{}\n")
    profile = {
        "python": "/usr/bin/python3",
        "source": str(tmp_path / "source"),
        "model": "/model",
        "data": str(data),
        "runs": str(tmp_path / "runs"),
        "defaults": {"max_updates": 10},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    jobs = SecurityJobs(str(path))
    with pytest.raises(ValueError, match="Missing development_views file"):
        jobs.submit(
            "experiment-a", {"development_views": str(tmp_path / "absent.json")}
        )


# ---- Merged from test_glimmer_operator.py ----
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
