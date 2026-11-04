"""Atomic checkpoints and exact adapter restore for trusted local runs."""

import json
import os
import random
from pathlib import Path


def write_json(path, value):
    """Publish a JSON record by atomic rename."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def save(model, optimizer, output, state, binding, contract):
    """Save adapter and optimizer together before advertising a checkpoint."""
    import torch

    destination = output / f"checkpoint-{state['step']:06d}"
    temporary = output / f".checkpoint-{state['step']:06d}.tmp"
    temporary.mkdir(exist_ok=False)
    model.save_pretrained(temporary, safe_serialization=True)
    torch.save(
        {
            **state,
            "binding": binding,
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
        },
        temporary / "training-state.pt",
    )
    write_json(temporary / "decision_contract.json", contract)
    os.replace(temporary, destination)
    write_json(
        output / "latest.json",
        {"directory": destination.name, "step": state["step"]},
    )
    return destination


def restore(model, optimizer, path, binding):
    """Restore trusted pickle state only after matching the study binding."""
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    state = torch.load(
        path / "training-state.pt", map_location="cpu", weights_only=False
    )
    if state["binding"] != binding:
        raise ValueError("Resume configuration, runtime or dataset changed")
    saved = load_file(path / "adapter_model.safetensors")
    set_peft_model_state_dict(model, saved)
    actual = get_peft_model_state_dict(model)
    if set(actual) != set(saved) or not all(
        torch.equal(actual[k].detach().cpu(), saved[k]) for k in saved
    ):
        raise RuntimeError("Adapter reload equality failed")
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return {k: state[k] for k in ("step", "epoch", "offset", "seen")}
