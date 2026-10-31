"""Exercise unattended follow-up using synthetic container and training receipts."""

import json
from types import SimpleNamespace

import pytest

from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    read_json,
)
from siliconshield.research_loop.execution import campaign


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
