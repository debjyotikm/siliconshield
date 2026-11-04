"""Train a bounded RTL answer-token adapter from an explicit JSON config."""

import argparse
import collections
import fcntl
import hashlib
import importlib.metadata
import json
import math
import random
import signal
from pathlib import Path

from siliconshield.training.glimmer.loading import staged_weights
from siliconshield.training.glimmer.rtl.checkpoint import (
    restore,
    save,
    write_json,
)
from siliconshield.training.glimmer.rtl.contract import encode, validate_tokens
from siliconshield.training.glimmer.rtl.data import (
    epoch_rows,
    load_rows,
    validate_splits,
)
from siliconshield.training.glimmer.rtl.evaluate import batch_inputs, evaluate
from siliconshield.training.glimmer.rtl.runtime import (
    clip_grad_norm,
    staged_transfers,
)

TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def validate_config(config):
    """Fail before model allocation on ambiguous or unbounded configuration."""
    required = {
        "base",
        "base_revision",
        "train",
        "dev",
        "contract",
        "output",
        "device_map",
        "domain_quotas",
        "seed",
        "max_updates",
        "max_epochs",
        "batch_rows",
        "microbatch_rows",
        "eval_every",
        "learning_rate",
        "warmup_updates",
        "lora_rank",
        "lora_alpha",
        "grad_clip",
        "stage_weights",
        "stage_transfers",
        "weight_decay",
    }
    if set(config) != required:
        raise ValueError(
            f"Config fields differ: {sorted(set(config) ^ required)}"
        )
    for key in (
        "max_updates",
        "max_epochs",
        "batch_rows",
        "microbatch_rows",
        "eval_every",
        "warmup_updates",
        "lora_rank",
        "lora_alpha",
    ):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        config["batch_rows"] % 3
        or config["microbatch_rows"] > config["batch_rows"]
    ):
        raise ValueError(
            "Batch must contain whole triplets and bound microbatches"
        )
    for key in ("learning_rate", "grad_clip", "weight_decay"):
        if (
            type(config[key]) not in (float, int)
            or not math.isfinite(config[key])
            or config[key] < 0
        ):
            raise ValueError(f"Invalid {key}")
    if config["learning_rate"] == 0 or config["grad_clip"] == 0:
        raise ValueError("Learning rate and gradient clip must be positive")
    for key in ("stage_weights", "stage_transfers"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("base", "base_revision", "train", "dev", "contract", "output"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"Missing {key}")
    if len(config["base_revision"]) != 40 or any(
        c not in "0123456789abcdef" for c in config["base_revision"]
    ):
        raise ValueError("Pin base_revision to a full Hub commit")
    if not isinstance(config["device_map"], dict) or not config["device_map"]:
        raise ValueError("Explicit device_map required")


def prepare(config):
    """Validate data and bind file bytes without loading a model."""
    validate_config(config)
    train, dev = load_rows(config["train"]), load_rows(config["dev"])
    validate_splits(train, dev)
    epoch_rows(train, config["seed"], 0, config["domain_quotas"])
    contract = json.loads(Path(config["contract"]).read_text())
    if contract.get("base_revision") != config["base_revision"]:
        raise ValueError("Contract and base revisions differ")
    if (
        type(contract.get("max_length")) is not int
        or contract["max_length"] <= 0
    ):
        raise ValueError("Contract needs a positive token budget")
    identities = {
        k: hashlib.sha256(Path(config[k]).read_bytes()).hexdigest()
        for k in ("train", "dev", "contract")
    }
    return train, dev, contract, identities


def file_digest(path):
    """Hash large model files without reading a complete shard into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config, resume=None):
    """Hold an exclusive process lock for the complete launch or resume."""
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(output.name + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(config, resume)


def _run(config, resume=None):
    """Run sequential model-parallel LoRA updates with recoverable boundaries."""
    import torch
    import torch.nn.functional as functional
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration

    train, dev, contract, identities = prepare(config)
    if not torch.cuda.is_available():
        raise RuntimeError("This trainer requires CUDA")
    runtime = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "safetensors",
        )
    }
    source = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in Path(__file__).parent.glob("*.py")
    }
    base_path = Path(config["base"])
    base_files = {}
    if base_path.is_dir():
        candidates = sorted(
            f
            for f in base_path.rglob("*")
            if f.is_file()
            and f.suffix
            in (".json", ".safetensors", ".model", ".txt", ".jinja")
        )
        if not any(f.suffix == ".safetensors" for f in candidates):
            raise ValueError("Local base needs safetensors weights")
        base_files = {
            str(f.relative_to(base_path)): file_digest(f) for f in candidates
        }
    binding = {
        "base_files": base_files,
        "config": config,
        "files": identities,
        "runtime": runtime,
        "source": source,
        "visible_gpus": torch.cuda.device_count(),
    }
    output = Path(config["output"])
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "study.json", binding)
    else:
        if json.loads((output / "study.json").read_text()) != binding:
            raise ValueError("Run binding changed")
        latest = json.loads((output / "latest.json").read_text())
        if resume.resolve() != (output / latest["directory"]).resolve():
            raise ValueError("Resume must use the latest committed checkpoint")
    stop = [False]

    def request_stop(_signum, _frame):
        stop[0] = True

    handlers = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    try:
        kwargs = (
            {"local_files_only": True}
            if Path(config["base"]).is_dir()
            else {"revision": config["base_revision"]}
        )
        tokenizer = AutoTokenizer.from_pretrained(config["base"], **kwargs)
        label_ids = validate_tokens(tokenizer, contract)
        for row in train + dev:
            row["input_ids"] = encode(tokenizer, row["state"], contract)
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if pad is None:
            raise ValueError("Tokenizer has no padding or EOS token")
        with staged_transfers(config["stage_transfers"]):
            with staged_weights(config["stage_weights"]):
                model = MuseGlimmerForConditionalGeneration.from_pretrained(
                    config["base"],
                    dtype=torch.bfloat16,
                    device_map=config["device_map"],
                    attn_implementation="sdpa",
                    **kwargs,
                )
            if any(
                str(v) in ("cpu", "disk")
                for k, v in model.hf_device_map.items()
                if "language_model" in k or k == "lm_head"
            ):
                raise ValueError("Language model must remain on CUDA")
            model.requires_grad_(False)
            model.config.use_cache = False
            model.config.text_config.use_cache = False
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
            model = get_peft_model(
                model,
                LoraConfig(
                    r=config["lora_rank"],
                    lora_alpha=config["lora_alpha"],
                    lora_dropout=0.0,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=TARGETS,
                ),
            )
            if any(
                "lora_" not in name
                for name, param in model.named_parameters()
                if param.requires_grad
            ):
                raise RuntimeError("Unexpected trainable base parameter")
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                params,
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
            state = {"step": 0, "epoch": 0, "offset": 0, "seen": {}}
            if resume is not None:
                state = restore(model, optimizer, resume, binding)
            device = model.get_input_embeddings().weight.device
            last_saved = state["step"] if resume else -1
            seen = collections.Counter(state["seen"])
            while (
                state["epoch"] < config["max_epochs"]
                and state["step"] < config["max_updates"]
                and not stop[0]
            ):
                ordered = epoch_rows(
                    train,
                    config["seed"],
                    state["epoch"],
                    config["domain_quotas"],
                )
                if state["offset"] >= len(ordered):
                    state["epoch"] += 1
                    state["offset"] = 0
                    continue
                chunk = ordered[
                    state["offset"] : state["offset"] + config["batch_rows"]
                ]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                progress = state["step"] / config["max_updates"]
                lr = (
                    config["learning_rate"]
                    * min(1.0, (state["step"] + 1) / config["warmup_updates"])
                    * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * progress)))
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss_sum = 0.0
                for offset in range(0, len(chunk), config["microbatch_rows"]):
                    micro = chunk[offset : offset + config["microbatch_rows"]]
                    logits = (
                        model(**batch_inputs(micro, device, pad))
                        .logits[:, -1]
                        .float()
                    )
                    target = torch.tensor(
                        [label_ids[r["label"]] for r in micro],
                        device=logits.device,
                    )
                    loss = functional.cross_entropy(logits, target)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite training loss")
                    (loss * len(micro) / len(chunk)).backward()
                    loss_sum += float(loss.detach()) * len(micro)
                norm = clip_grad_norm(params, config["grad_clip"])
                optimizer.step()
                state["step"] += 1
                state["offset"] += len(chunk)
                seen.update(r["id"] for r in chunk)
                state["seen"] = dict(seen)
                record = {
                    "step": state["step"],
                    "ce": loss_sum / len(chunk),
                    "lr": lr,
                    "grad_norm": norm,
                    "rows_seen": sum(seen.values()),
                    "unique_rows": len(seen),
                }
                with (output / "events.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                if state["step"] % config["eval_every"] == 0:
                    save(model, optimizer, output, state, binding, contract)
                    last_saved = state["step"]
                    write_json(
                        output / f"eval-{state['step']:06d}.json",
                        evaluate(model, dev, device, pad, label_ids),
                    )
            if state["step"] != last_saved:
                save(model, optimizer, output, state, binding, contract)
            write_json(
                output / f"eval-{state['step']:06d}.json",
                evaluate(model, dev, device, pad, label_ids),
            )
            write_json(
                output / "completion.json",
                {
                    "step": state["step"],
                    "reason": "requested_stop"
                    if stop[0]
                    else "budget_complete",
                    "rows_seen": sum(seen.values()),
                    "unique_rows": len(seen),
                    "promotion": "not_assessed",
                    "evaluation": "development_only",
                },
            )
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    """Validate a study or explicitly launch/resume its local GPU trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume", type=Path, help="Trusted local checkpoint only"
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.validate_only:
        train, dev, _, _ = prepare(config)
        print(
            json.dumps(
                {
                    "train_rows": len(train),
                    "dev_rows": len(dev),
                    "status": "schema_and_split_validated",
                    "tokenization": "not_run",
                }
            )
        )
    else:
        run(config, args.resume)


if __name__ == "__main__":
    main()
