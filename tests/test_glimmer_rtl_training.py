"""Synthetic fixtures for the RTL answer-token trainer. No live models."""

import copy
import json
from pathlib import Path

import pytest

from siliconshield.training.glimmer.rtl.contract import encode, validate_tokens
from siliconshield.training.glimmer.rtl.data import (
    epoch_rows,
    load_rows,
    validate_splits,
)
from siliconshield.training.glimmer.rtl.evaluate import summarize
from siliconshield.training.glimmer.rtl.train import prepare, validate_config

ROOT = Path(__file__).resolve().parents[1]


def rows(split="train"):
    return [
        {
            "id": f"{split}-{domain}-{group}-{label}",
            "domain": domain,
            "pair_id": f"{split}-{group}",
            "split_group": f"{split}-{domain}-{group}",
            "state": f"{split} synthetic input {domain} {group} {label}",
            "label": label,
        }
        for domain in ("local_guards", "temporal")
        for group in range(3)
        for label in range(3)
    ]


def write_rows(path, values):
    path.write_text("\n".join(json.dumps(r) for r in values) + "\n")
    return path


def config():
    return json.loads(
        (ROOT / "configs/glimmer-rtl-training.example.json").read_text()
    )


def test_exposure_is_complete_deterministic_and_grouped():
    original = rows()
    quotas = {"local_guards": 1, "temporal": 4}
    ordered = epoch_rows(original, 44, 0, quotas)
    assert ordered == epoch_rows(original, 44, 0, quotas)
    assert sorted(r["id"] for r in ordered) == sorted(r["id"] for r in original)
    assert original == rows()
    for start in range(0, len(ordered), 3):
        assert len({r["pair_id"] for r in ordered[start : start + 3]}) == 1
    with pytest.raises(ValueError):
        epoch_rows(original, 44, 0, {"local_guards": 0, "temporal": 4})


@pytest.mark.parametrize("key", ["id", "state", "split_group", "pair_id"])
def test_overlap_is_rejected(key):
    train, dev = rows(), rows("dev")
    dev[0][key] = train[0][key]
    with pytest.raises(ValueError):
        validate_splits(train, dev)


def test_incomplete_triplets_and_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="triplet"):
        load_rows(write_rows(tmp_path / "bad.jsonl", rows()[:-1]))
    duplicate = rows()
    duplicate[1]["id"] = duplicate[0]["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        load_rows(write_rows(tmp_path / "bad.jsonl", duplicate))


def test_real_preflight_without_torch_or_model(tmp_path):
    cfg = config()
    cfg["train"] = str(write_rows(tmp_path / "train.jsonl", rows()))
    cfg["dev"] = str(write_rows(tmp_path / "dev.jsonl", rows("dev")))
    cfg["contract"] = str(ROOT / cfg["contract"])
    train, dev, _, identities = prepare(cfg)
    assert len(train) == len(dev) == 18
    assert identities["train"] != identities["dev"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_rows", 5),
        ("learning_rate", float("nan")),
        ("max_updates", 0),
        ("stage_transfers", "yes"),
        ("base_revision", "main"),
    ],
)
def test_invalid_config(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_metrics_keep_domains_and_whole_triplets_separate():
    samples = rows()
    for row in samples:
        row["pred"] = row["label"]
        row["probabilities"] = [0.1, 0.1, 0.1]
        row["probabilities"][row["label"]] = 0.8
    samples[1]["pred"] = 0
    result = summarize(samples)
    assert result["local_guards"]["false_safe"] == 1
    assert result["local_guards"]["triplets_correct"] == 2
    assert result["temporal"]["triplets_correct"] == 3
    with pytest.raises(ValueError, match="complete"):
        summarize(samples[:-1])


class Tokenizer:
    def encode(self, text, **_kwargs):
        return {"A": [45], "B": [46], "C": [47]}.get(
            text, list(range(len(text)))
        )

    def apply_chat_template(self, messages, **_kwargs):
        return "".join(m["content"] for m in messages)


def test_contract_rejects_truncation_and_wrong_tokens():
    contract = json.loads(
        (ROOT / "configs/glimmer-rtl-contract.example.json").read_text()
    )
    tokenizer = Tokenizer()
    assert validate_tokens(tokenizer, contract) == [45, 46, 47]
    assert encode(tokenizer, "RTL", contract)
    with pytest.raises(ValueError, match="sentinel"):
        encode(tokenizer, "SENTINEL_ANSWER", contract)
    small = copy.deepcopy(contract)
    small["max_length"] = 1
    with pytest.raises(ValueError, match="budget"):
        encode(tokenizer, "RTL", small)
    contract["label_token_ids"] = [1, 2, 3]
    with pytest.raises(ValueError, match="match"):
        validate_tokens(tokenizer, contract)


def test_batch_positions_and_gradient_clipping():
    torch = pytest.importorskip("torch")
    from siliconshield.training.glimmer.rtl.evaluate import batch_inputs
    from siliconshield.training.glimmer.rtl.runtime import clip_grad_norm

    batch = batch_inputs([{"input_ids": [2, 3]}, {"input_ids": [4]}], "cpu", 0)
    assert batch["input_ids"].tolist() == [[2, 3], [0, 4]]
    assert batch["position_ids"].tolist() == [[0, 1], [0, 0]]
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([3.0, 4.0])
    assert clip_grad_norm([parameter], 1.0) == 5.0
    assert float(parameter.grad.norm()) == pytest.approx(1.0, abs=1e-6)
    parameter.grad[0] = float("nan")
    with pytest.raises(RuntimeError, match="Nonfinite"):
        clip_grad_norm([parameter], 1.0)


def test_checkpoint_roundtrip_and_next_optimizer_update(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from siliconshield.training.glimmer.rtl.checkpoint import restore, save

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.proj(x)

    torch.manual_seed(5)
    first = peft.get_peft_model(
        Tiny(), peft.LoraConfig(r=2, lora_alpha=2, target_modules=["proj"])
    )
    torch.manual_seed(5)
    second = peft.get_peft_model(
        Tiny(), peft.LoraConfig(r=2, lora_alpha=2, target_modules=["proj"])
    )
    first_opt = torch.optim.AdamW(
        [p for p in first.parameters() if p.requires_grad], lr=0.01
    )
    second_opt = torch.optim.AdamW(
        [p for p in second.parameters() if p.requires_grad], lr=0.01
    )
    inputs = torch.tensor([[1.0, -2.0]])

    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()

    update(first, first_opt)
    state = {"step": 1, "epoch": 0, "offset": 3, "seen": {"example": 1}}
    binding = {"synthetic": True}
    checkpoint = save(first, first_opt, tmp_path, state, binding, {})
    assert json.loads((tmp_path / "latest.json").read_text())["step"] == 1
    with pytest.raises(ValueError, match="changed"):
        restore(second, second_opt, checkpoint, {"synthetic": False})
    assert restore(second, second_opt, checkpoint, binding) == state
    assert torch.equal(first(inputs), second(inputs))
    update(first, first_opt)
    update(second, second_opt)
    for key, value in peft.get_peft_model_state_dict(first).items():
        assert torch.equal(value, peft.get_peft_model_state_dict(second)[key])
    with pytest.raises(OSError):
        save(first, first_opt, tmp_path, state, binding, {})


def test_transfer_hooks_restore_after_failure():
    hooks = pytest.importorskip("accelerate.hooks")
    operations = pytest.importorskip("accelerate.utils.operations")
    from siliconshield.training.glimmer.rtl.runtime import staged_transfers

    original, original_hook = operations.send_to_device, hooks.send_to_device
    with pytest.raises(RuntimeError, match="fixture"):
        with staged_transfers(True):
            assert operations.send_to_device is not original
            raise RuntimeError("fixture")
    assert operations.send_to_device is original
    assert hooks.send_to_device is original_hook
