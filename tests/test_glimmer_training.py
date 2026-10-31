"""Checkpoint rotation and resume recovery, including the interrupted-save state."""

import json

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file

from siliconshield.training.glimmer.train import (
    checkpoint_valid,
    find_resume_checkpoint,
    rotate_checkpoint,
)

SIGNATURE = {"base_revision": "test-revision", "rank": 16}


def make_checkpoint(path, step):
    """A small but genuinely serialized, recoverable checkpoint.

    Optimizer state comes from a real AdamW after a real step; RNG fields use
    the same producers and types as the trainer.
    """
    import random

    import numpy as np

    (path / "adapter").mkdir(parents=True)
    (path / "adapter/adapter_config.json").write_text(
        json.dumps({"r": 16, "peft_type": "LORA"})
    )
    save_file(
        {"lora_A.weight": torch.zeros(2, 2)},
        str(path / "adapter/adapter_model.safetensors"),
    )
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    parameter.grad = torch.ones(2, 2)
    optimizer.step()
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "epoch": 0,
            "offset": step,
            "step": step,
            "best": 1.5,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        },
        path / "state.pt",
    )
    (path / "receipt.json").write_text(
        json.dumps(
            {"step": step, "development_loss": 1.5, "signature": SIGNATURE}
        )
    )


def make_empty_checkpoint(path):
    path.mkdir(parents=True)
    (path / "adapter").mkdir()
    (path / "state.pt").write_bytes(b"")
    (path / "receipt.json").write_text("{}")


def test_empty_checkpoint_is_not_recoverable(tmp_path):
    make_empty_checkpoint(tmp_path / "latest")
    assert not checkpoint_valid(tmp_path / "latest", signature=SIGNATURE)
    assert find_resume_checkpoint(tmp_path, SIGNATURE) is None


def test_incomplete_optimizer_and_missing_rng_cannot_resume(tmp_path):
    make_checkpoint(tmp_path / "latest.previous", 7)
    make_checkpoint(tmp_path / "latest", 9)
    state = torch.load(
        tmp_path / "latest/state.pt", map_location="cpu", weights_only=False
    )
    del state["cuda_rng"], state["python_rng"], state["numpy_rng"]
    state["optimizer"] = {
        "state": {},
        "param_groups": [{"lr": 1e-4}],
    }  # no 'params'
    torch.save(state, tmp_path / "latest/state.pt")
    assert not checkpoint_valid(tmp_path / "latest", signature=SIGNATURE)
    # The usable previous generation is selected instead of the broken latest.
    assert (
        find_resume_checkpoint(tmp_path, SIGNATURE)
        == tmp_path / "latest.previous"
    )


@pytest.mark.parametrize("bad_rng", [b"raw-bytes", torch.ones(4)])
def test_non_byte_tensor_torch_rng_is_rejected_and_falls_back(
    tmp_path, bad_rng
):
    make_checkpoint(tmp_path / "latest.previous", 7)
    make_checkpoint(tmp_path / "latest", 9)
    state = torch.load(
        tmp_path / "latest/state.pt", map_location="cpu", weights_only=False
    )
    assert (
        state["torch_rng"].dtype == torch.uint8
    )  # genuine producer representation
    state["torch_rng"] = bad_rng
    torch.save(state, tmp_path / "latest/state.pt")
    with pytest.raises(TypeError):
        torch.set_rng_state(state["torch_rng"])
    assert not checkpoint_valid(tmp_path / "latest", signature=SIGNATURE)
    assert (
        find_resume_checkpoint(tmp_path, SIGNATURE)
        == tmp_path / "latest.previous"
    )


def test_genuine_fixture_actually_restores_optimizer_and_rng(tmp_path):
    import random

    import numpy as np

    make_checkpoint(tmp_path / "latest", 4)
    state = torch.load(
        tmp_path / "latest/state.pt", map_location="cpu", weights_only=False
    )
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = torch.optim.AdamW([parameter], lr=9.9)
    optimizer.load_state_dict(
        state["optimizer"]
    )  # previously reproduced KeyError 'params'
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[0]["params"] == [parameter]
    saved_torch = torch.get_rng_state()
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(saved_torch)


def test_corrupt_state_and_wrong_signature_are_rejected(tmp_path):
    make_checkpoint(tmp_path / "latest", 3)
    (tmp_path / "latest/state.pt").write_bytes(b"not a checkpoint")
    assert not checkpoint_valid(tmp_path / "latest", signature=SIGNATURE)
    make_checkpoint(tmp_path / "latest2", 3)
    assert not checkpoint_valid(
        tmp_path / "latest2", signature={"base_revision": "other"}
    )


def test_resume_recovers_previous_when_latest_missing(tmp_path):
    # Filesystem state after interruption between the two renames.
    make_checkpoint(tmp_path / "latest.previous", 7)
    assert (
        find_resume_checkpoint(tmp_path, SIGNATURE)
        == tmp_path / "latest.previous"
    )


def test_resume_prefers_valid_latest(tmp_path):
    make_checkpoint(tmp_path / "latest", 9)
    make_checkpoint(tmp_path / "latest.previous", 7)
    assert find_resume_checkpoint(tmp_path, SIGNATURE) == tmp_path / "latest"


def test_unrecoverable_latest_falls_back_to_previous(tmp_path):
    make_checkpoint(tmp_path / "latest.previous", 7)
    make_empty_checkpoint(tmp_path / "latest")
    assert (
        find_resume_checkpoint(tmp_path, SIGNATURE)
        == tmp_path / "latest.previous"
    )


def test_rotation_keeps_previous_generation(tmp_path):
    make_checkpoint(tmp_path / "latest", 1)
    stage = tmp_path / "latest.tmp"
    make_checkpoint(stage, 2)
    rotate_checkpoint(stage, tmp_path / "latest")
    assert checkpoint_valid(tmp_path / "latest", signature=SIGNATURE)
    assert (
        json.loads((tmp_path / "latest/receipt.json").read_text())["step"] == 2
    )
    assert checkpoint_valid(tmp_path / "latest.previous", signature=SIGNATURE)
    assert (
        json.loads((tmp_path / "latest.previous/receipt.json").read_text())[
            "step"
        ]
        == 1
    )
    # Simulated interruption after this rotation: remove latest, previous still resumes.
    import shutil

    shutil.rmtree(tmp_path / "latest")
    assert (
        find_resume_checkpoint(tmp_path, SIGNATURE)
        == tmp_path / "latest.previous"
    )


def test_model_revision_is_shared_with_identity():
    import siliconshield.training.glimmer.train as train_module
    from siliconshield.training.glimmer import dpo as dpo_module
    from siliconshield.training.glimmer import identity

    assert train_module.MODEL_REVISION is identity.MODEL_REVISION
    assert dpo_module.MODEL_REVISION is identity.MODEL_REVISION
    assert identity.MODEL_REVISION == "cec94fcfd6512f2cfe708138c32533b346ea4f69"
