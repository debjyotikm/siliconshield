import pytest

from siliconshield.research_loop.execution.security_jobs import (
    command_for,
    laya_root_for,
    service_command,
    verify_laya,
)

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
