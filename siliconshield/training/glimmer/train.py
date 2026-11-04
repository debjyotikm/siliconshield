"""Train a Muse Glimmer LoRA adapter on one GPU and save resumable checkpoints."""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    read_json,
    validate_episode,
)
from siliconshield.training.glimmer.identity import MODEL_REVISION
from siliconshield.training.glimmer.tokenization import encode_episode

TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def load_rows(path, allowed_splits=("train", "dev"), require_signed=False):
    if not isinstance(path, (str, Path)):
        raise ValueError("Episode file path must be a path")
    rows = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        validate_episode(row)
    from siliconshield.research_loop.episodes.dataset import (
        verify_split_membership,
    )

    verify_split_membership(
        rows, path, allowed_splits=allowed_splits, require_signed=require_signed
    )
    return rows


def _state_readable(path):
    """Training state must deserialize with the fields resume actually restores."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        import torch
    except ImportError:
        return path.read_bytes()[:2] == b"PK"  # torch.save zip container
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        # Broad catch is deliberate: any load failure means "not a
        # recoverable checkpoint", and the caller falls back or errors.
        return False
    if not isinstance(state, dict):
        return False
    # Every field the resume path in train() accesses unconditionally.
    if not {
        "optimizer",
        "epoch",
        "offset",
        "step",
        "best",
        "torch_rng",
        "cuda_rng",
        "python_rng",
        "numpy_rng",
    } <= set(state):
        return False
    optimizer = state.get("optimizer")
    if not isinstance(optimizer, dict) or not isinstance(
        optimizer.get("state"), dict
    ):
        return False
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or not groups:
        return False
    for group in groups:
        if (
            not isinstance(group, dict)
            or not isinstance(group.get("params"), list)
            or not group["params"]
        ):
            return False
        lr = group.get("lr")
        if (
            not isinstance(lr, (int, float))
            or isinstance(lr, bool)
            or not math.isfinite(lr)
        ):
            return False
    if (
        type(state.get("epoch")) is not int
        or type(state.get("step")) is not int
        or type(state.get("offset")) is not int
    ):
        return False

    # torch.set_rng_state requires exactly the ByteTensor from torch.get_rng_state();
    # raw bytes and non-uint8 tensors pass isinstance checks but fail at restore time.
    def byte_tensor(value):
        return isinstance(value, torch.Tensor) and value.dtype == torch.uint8

    if (
        not byte_tensor(state["torch_rng"])
        or not isinstance(state["cuda_rng"], list)
        or not all(byte_tensor(value) for value in state["cuda_rng"])
        or not isinstance(state["python_rng"], tuple)
        or not isinstance(state["numpy_rng"], tuple)
    ):
        return False
    best = state.get("best")
    return (
        isinstance(best, (int, float))
        and not isinstance(best, bool)
        and math.isfinite(best)
    )


def checkpoint_valid(path, full=True, signature=None):
    """Check adapter files, run metadata, and the fields needed to resume.

    The adapter needs a parseable config and non-empty weights. The receipt
    must include step, development loss, and signature. When full is True,
    the training state must also deserialize and pass the resume checks.
    """
    path = Path(path)
    adapter = path / "adapter"
    if not adapter.is_dir() or not (path / "receipt.json").is_file():
        return False
    try:
        config = read_json(adapter / "adapter_config.json")
        weights = [
            p
            for p in adapter.iterdir()
            if p.name.startswith("adapter_model.") and p.stat().st_size > 0
        ]
        receipt = read_json(path / "receipt.json")
    except (ValueError, OSError):
        return False
    if not isinstance(config, dict) or not config or not weights:
        return False
    if type(receipt.get("step")) is not int or receipt["step"] < 0:
        return False
    loss = receipt.get("development_loss")
    if (
        not isinstance(loss, (int, float))
        or isinstance(loss, bool)
        or not math.isfinite(loss)
    ):
        return False
    if (
        not isinstance(receipt.get("signature"), dict)
        or not receipt["signature"]
    ):
        return False
    if signature is not None and receipt["signature"] != signature:
        return False
    if full and not _state_readable(path / "state.pt"):
        return False
    return True


def find_resume_checkpoint(output, signature=None):
    """Newest recoverable full checkpoint: ``latest``, recovering ``latest.previous``.

    An interruption between the two renames in rotate_checkpoint() leaves
    ``latest`` missing; the previous generation is then the resume source.
    Returns None when no complete generation exists.
    """
    output = Path(output)
    for name in ("latest", "latest.previous"):
        if checkpoint_valid(output / name, signature=signature):
            return output / name
    return None


def rotate_checkpoint(stage, destination):
    """Install a staged checkpoint and keep the last generation as .previous.

    If the final rename fails, restore the previous generation to destination.
    """
    stage, destination = Path(stage), Path(destination)
    previous = destination.with_name(destination.name + ".previous")
    if previous.exists():
        shutil.rmtree(previous)
    if destination.exists():
        destination.rename(previous)
    try:
        stage.rename(destination)
    except OSError:
        if not destination.exists() and previous.exists():
            previous.rename(destination)
        raise
    return destination


def dev_subset(rows, count):
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) for row in rows
    ):
        raise ValueError("Dev subset needs a list of episode rows")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("Dev subset count must be a nonnegative integer")
    by_action = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        target = row.get("target")
        if not isinstance(target, dict) or "action" not in target:
            raise ValueError(
                "Row %s has no target action to stratify" % row.get("id")
            )
        by_action.setdefault(target["action"], []).append(row)
    chosen = []
    group_counts = Counter()
    while len(chosen) < count and any(by_action.values()):
        for key in sorted(by_action):
            if by_action[key] and len(chosen) < count:
                index = min(
                    range(len(by_action[key])),
                    key=lambda i: group_counts[by_action[key][i]["group_id"]],
                )
                row = by_action[key].pop(index)
                chosen.append(row)
                group_counts[row["group_id"]] += 1
    return chosen


def train(args):
    from siliconshield.research_loop.execution.gpu_lease import acquire_gpu
    from siliconshield.training.glimmer.identity import verify_code_identity

    verified_code = verify_code_identity()
    with acquire_gpu():
        return _train(args, verified_code)


def training_rows(data):
    """Only authenticated TRAIN/DEV populations may enter adapter training."""
    if not isinstance(data, (str, Path)):
        raise ValueError("Training data directory must be a path")
    rows = load_rows(
        Path(data) / "train.jsonl",
        allowed_splits=("train",),
        require_signed=True,
    )
    development = load_rows(
        Path(data) / "dev.jsonl", allowed_splits=("dev",), require_signed=True
    )
    if {r["group_id"] for r in rows} & {r["group_id"] for r in development}:
        raise ValueError("TRAIN and DEV share an experiment lineage")
    if not rows or not development:
        raise ValueError(
            "Training and development splits must both be nonempty"
        )
    return rows, development


def selection_mode(args):
    """Resolve task selection without treating token loss as external approval."""
    external = getattr(args, "external_task_selection", False)
    if external and args.task_eval_every:
        raise ValueError("Choose external or in-process task selection")
    if external:
        return "external_pending"
    return "in_process" if args.task_eval_every else "token_loss"


def checkpoint_improved(mode, task_key, previous_key, loss, previous_loss):
    """Keep external selection pending, even when training loss improves."""
    if mode == "external_pending":
        return False
    if mode == "in_process":
        return task_key is not None and (
            previous_key is None or task_key > previous_key
        )
    return loss < previous_loss


def _train(args, verified_code):
    train_rows, all_dev_rows = training_rows(args.data)
    dev_rows = dev_subset(all_dev_rows, args.dev_examples)
    from siliconshield.research_loop.episodes.operator_episodes import (
        operator_coverage,
    )

    coverage = operator_coverage(train_rows, all_dev_rows, dev_rows)
    purpose = getattr(args, "purpose", "qualification")
    mode = selection_mode(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "operator-coverage.json", {"purpose": purpose, **coverage}
    )
    if purpose == "operator" and (
        not coverage["coverage_passed"] or mode == "token_loss"
    ):
        raise ValueError(
            "Operator training requires full action coverage and task-based selection; "
            "inspect operator-coverage.json"
        )
    import numpy as np
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration

    status = output / "status.json"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA device supporting BF16 is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            Path(args.model) / "chat_template.jinja"
        ).read_text()
    encoded = [
        encode_episode(tokenizer, row, args.max_length) for row in train_rows
    ]
    development = [
        encode_episode(tokenizer, row, args.max_length) for row in dev_rows
    ]
    signature = {
        "base_revision": MODEL_REVISION,
        "purpose": purpose,
        "rank": args.rank,
        "max_length": args.max_length,
        "lr": args.lr,
        "accumulation": args.accumulation,
        "seed": args.seed,
        "epochs": args.epochs,
        "targets": TARGETS,
        "task_selection_mode": mode,
        "cpu_staged_loading": getattr(args, "cpu_staged_loading", False),
        "dev_ids": [r["id"] for r in dev_rows],
        "files": {
            s: hashlib.sha256(
                (Path(args.data) / (s + ".jsonl")).read_bytes()
            ).hexdigest()
            for s in ("train", "dev")
        },
        "template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode()
        ).hexdigest(),
    }
    from siliconshield.training.glimmer.identity import (
        adapter_manifest,
        verify_snapshot,
    )

    init_adapter = getattr(args, "init_adapter", None)
    if init_adapter:
        adapter_config = read_json(Path(init_adapter) / "adapter_config.json")
        if adapter_config.get("r") != args.rank:
            raise ValueError(
                "Starting adapter rank differs from the registered training rank"
            )
        signature["init_adapter_fingerprint"] = adapter_manifest(init_adapter)[
            "fingerprint"
        ]
    snapshot = verify_snapshot(
        args.model,
        os.environ.get("SILICONSHIELD_MODEL_FINGERPRINT"),
        revision=MODEL_REVISION,
    )
    signature["snapshot_fingerprint"] = snapshot["fingerprint"]
    signature["image_digest"] = os.environ.get("SILICONSHIELD_IMAGE_DIGEST")
    signature.update(
        code_identity=verified_code,
        expected_code_identity=verified_code,
        eval_every=args.eval_every,
        task_eval_every=args.task_eval_every,
        dataset_manifest=hashlib.sha256(
            (Path(args.data) / "manifest.json").read_bytes()
        ).hexdigest(),
        torch=torch.__version__,
    )
    if not signature["image_digest"]:
        raise ValueError(
            "Set SILICONSHIELD_IMAGE_DIGEST to the resolved execution image ID"
        )
    if (Path(args.data) / ".final-evaluation").exists():
        raise ValueError(
            "Dataset has been used for final evaluation; further training is forbidden"
        )
    current = (
        read_json(output / "configuration.json")
        if (output / "configuration.json").exists()
        else None
    )
    if current and current != signature:
        raise ValueError(
            "Resume configuration/data/template differs from the recorded run"
        )
    if args.resume and find_resume_checkpoint(output, signature) is None:
        raise RuntimeError(
            "Resume requested but no complete checkpoint (latest or latest.previous) "
            "with adapter, receipt and state exists under " + str(output)
        )
    atomic_json(output / "configuration.json", signature)
    atomic_json(
        output / "data_receipt.json",
        {
            "train": len(encoded),
            "dev": len(development),
            "purpose": purpose,
            "operator_coverage": coverage,
            "max_train_tokens": max(len(e["input_ids"]) for e in encoded),
            "truncated": 0,
            "assistant_tokens": sum(
                sum(v != -100 for v in e["labels"]) for e in encoded
            ),
            "actions": dict(Counter(r["target"]["action"] for r in train_rows)),
        },
    )
    atomic_json(
        status, {"phase": "loading_model", "base_revision": MODEL_REVISION}
    )
    from siliconshield.training.glimmer.loading import staged_weights

    load_started = time.monotonic()
    with staged_weights(signature["cpu_staged_loading"]):
        base, loading = MuseGlimmerForConditionalGeneration.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            local_files_only=True,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
    torch.cuda.synchronize()
    atomic_json(
        output / "loading.json",
        {
            "seconds": time.monotonic() - load_started,
            "cpu_staged_loading": signature["cpu_staged_loading"],
            "precision": "bfloat16",
        },
    )
    if any(
        loading.get(k)
        for k in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
    ):
        atomic_json(output / "loading_errors.json", loading)
        raise RuntimeError(
            "Published checkpoint did not load exactly; inspect loading_errors.json"
        )
    base.config.use_cache = False
    base.config.text_config.use_cache = False
    latest = output / "latest"
    resume_from = (
        find_resume_checkpoint(output, signature) if args.resume else None
    )
    if args.resume and resume_from is None:
        raise RuntimeError(
            "Resume requested but no complete checkpoint (latest or latest.previous) "
            "with adapter, receipt and state exists under " + str(output)
        )
    if resume_from is not None and resume_from != latest:
        # Interrupted save: promote the validated previous generation back to latest.
        if latest.exists():
            shutil.rmtree(latest)
        resume_from.rename(latest)
        resume_from = latest
    if resume_from is not None:
        model = PeftModel.from_pretrained(
            base, resume_from / "adapter", is_trainable=True
        )
    else:
        if latest.exists() or (output / "latest.previous").exists():
            raise ValueError("An existing training state requires --resume")
        if init_adapter:
            model = PeftModel.from_pretrained(
                base, init_adapter, is_trainable=True
            )
        else:
            model = get_peft_model(
                base,
                LoraConfig(
                    r=args.rank,
                    lora_alpha=args.rank * 2,
                    lora_dropout=0.05,
                    target_modules=TARGETS,
                    task_type="CAUSAL_LM",
                ),
            )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not params or any("lora_" not in n or "vision" in n for n, _ in params):
        raise ValueError("Only language-model LoRA parameters may be trainable")
    optimizer = torch.optim.AdamW(
        [p for _, p in params], lr=args.lr, weight_decay=0.01
    )
    steps_per_epoch = math.ceil(len(encoded) / args.accumulation)
    total_steps = steps_per_epoch * args.epochs
    epoch, offset, step, best = 0, 0, 0, float("inf")
    best_task_key = None
    observed_loss = None
    if resume_from is not None:
        state = torch.load(
            resume_from / "state.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        epoch, offset, step, best = (
            state[k] for k in ("epoch", "offset", "step", "best")
        )
        best_task_key = state.get("best_task_key")
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])

    def tensor(row):
        return {
            k: torch.tensor([v], device="cuda", dtype=torch.long)
            for k, v in row.items()
        }

    def event(value):
        value = {"step": step, "time": time.time(), **value}
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        atomic_json(status, value)
        print(json.dumps(value, allow_nan=False), flush=True)

    def evaluate():
        model.eval()
        total, tokens = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in development:
                n = sum(x != -100 for x in row["labels"][1:])
                loss = model(**tensor(row)).loss
                total += float(loss) * n
                tokens += n
        model.train()
        return total / tokens

    def save(destination, full):
        stage = destination.with_name(destination.name + ".tmp")
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        model.save_pretrained(stage / "adapter")
        tokenizer.save_pretrained(stage / "adapter")
        if full:
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "offset": offset,
                    "step": step,
                    "best": best,
                    "best_task_key": best_task_key,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                },
                stage / "state.pt",
            )
        atomic_json(
            stage / "receipt.json",
            {
                "step": step,
                "development_loss": observed_loss,
                "task_selection_key": best_task_key,
                "task_selection_mode": mode,
                "selected_for_use": False,
                "signature": signature,
            },
        )
        rotate_checkpoint(stage, destination)

    def evaluate_tasks(dev_loss):
        from siliconshield.research_loop.controller.evaluate import (
            InProcessClient,
        )
        from siliconshield.research_loop.controller.replay import execute, grade

        model.eval()
        client = InProcessClient(model, tokenizer, "glimmer-development")
        client.max_tokens = 1024
        scores = []
        attempt = output / "development-replay" / ("step-%06d" % step)
        for row in dev_rows:
            scores.append(
                {
                    "episode_id": row["id"],
                    "group_id": row["group_id"],
                    "scenario": row.get("verification", {}).get("scenario"),
                    **grade(row, execute(row, client, attempt / row["id"])),
                }
            )
            atomic_json(
                status,
                {
                    "phase": "development_tasks",
                    "step": step,
                    "time": time.time(),
                    "completed": len(scores),
                    "n": len(dev_rows),
                },
            )
        verified = sum(bool(v["verified"]) for v in scores)
        action = sum(bool(v["action_correct"]) for v in scores)
        evidence = sum(bool(v["evidence_correct"]) for v in scores)
        atomic_json(
            attempt / "report.json",
            {
                "step": step,
                "n": len(scores),
                "verified": verified,
                "action_correct": action,
                "evidence_correct": evidence,
                "results": scores,
            },
        )
        event(
            {
                "phase": "development_tasks",
                "n": len(scores),
                "verified": verified,
                "action_correct": action,
                "evidence_correct": evidence,
            }
        )
        model.train()
        return [verified, action, evidence, -dev_loss]

    event(
        {
            "phase": "model_loaded",
            "trainable_parameters": sum(p.numel() for _, p in params),
            "parameter_dtypes": sorted({str(p.dtype) for _, p in params}),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
    )
    if step:
        previous = [
            json.loads(line)
            for line in (output / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        reference = next(
            (
                v["loss"]
                for v in reversed(previous)
                if v.get("step") == step and v.get("phase") == "development"
            ),
            None,
        )
        reloaded = evaluate()
        observed_loss = reloaded
        if reference is None or abs(reloaded - reference) > max(
            1e-4, abs(reference) * 1e-3
        ):
            raise RuntimeError(
                f"Checkpoint reload loss mismatch: saved={reference}, reloaded={reloaded}"
            )
        event(
            {
                "phase": "resume_parity",
                "saved_loss": reference,
                "reloaded_loss": reloaded,
                "optimizer_restored": True,
                "rng_restored": True,
            }
        )
    if step == 0:
        best = evaluate()
        observed_loss = best
        if not math.isfinite(best):
            raise RuntimeError("Non-finite initial development loss")
        event(
            {
                "phase": "initial_development",
                "loss": best,
                "examples": len(development),
            }
        )
        if args.task_eval_every:
            best_task_key = evaluate_tasks(best)
        save(output / "best", False)
    model.train()
    start = time.monotonic()
    start_step = step
    trained_tokens = 0
    while epoch < args.epochs:
        order = list(range(len(encoded)))
        random.Random(args.seed + epoch).shuffle(order)
        while offset < len(order):
            indices = order[offset : offset + args.accumulation]
            denominator = sum(
                sum(v != -100 for v in encoded[i]["labels"][1:])
                for i in indices
            )
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for index in indices:
                row = encoded[index]
                n = sum(v != -100 for v in row["labels"][1:])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**tensor(row)).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                (loss * (n / denominator)).backward()
                loss_sum += float(loss.detach()) * n / denominator
                trained_tokens += len(row["input_ids"])
            gradient = torch.nn.utils.clip_grad_norm_(
                [p for _, p in params], 1.0
            )
            if not torch.isfinite(gradient) or float(gradient) == 0:
                raise RuntimeError("Missing or non-finite adapter gradients")
            if step < 2:
                counts = {
                    letter: sum(
                        p.grad is not None and bool(torch.count_nonzero(p.grad))
                        for name, p in params
                        if f"lora_{letter}" in name
                    )
                    for letter in ("A", "B")
                }
                event({"phase": "gradient_check", "nonzero_tensors": counts})
                if not counts["B"] or (step == 1 and not counts["A"]):
                    raise RuntimeError(
                        "Adapter projections did not receive expected gradients"
                    )
            warmup = max(1, min(10, total_steps // 10))
            lr_factor = min(1.0, (step + 1) / warmup) * (
                0.1
                + 0.9
                * 0.5
                * (1 + math.cos(math.pi * step / max(1, total_steps)))
            )
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_factor
            optimizer.step()
            offset += len(indices)
            step += 1
            event(
                {
                    "phase": "training",
                    "epoch": epoch,
                    "offset": offset,
                    "loss": loss_sum,
                    "lr": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(gradient),
                    "input_tokens_per_second": trained_tokens
                    / (time.monotonic() - start),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated()
                    / 2**30,
                }
            )
            stopping = bool(
                args.max_steps and step - start_step >= args.max_steps
            )
            if step % args.eval_every == 0 or offset == len(order) or stopping:
                dev_loss = evaluate()
                observed_loss = dev_loss
                event(
                    {
                        "phase": "development",
                        "loss": dev_loss,
                        "examples": len(development),
                    }
                )
                if not math.isfinite(dev_loss):
                    raise RuntimeError("Non-finite development loss")
                task_due = args.task_eval_every and (
                    step % args.task_eval_every == 0
                    or offset == len(order)
                    or stopping
                )
                task_key = evaluate_tasks(dev_loss) if task_due else None
                improved = checkpoint_improved(
                    mode, task_key, best_task_key, dev_loss, best
                )
                if improved:
                    best = dev_loss
                    best_task_key = task_key
                    save(output / "best", False)
                save(latest, True)
                if mode == "external_pending":
                    # Preserve each observation for the separate serving
                    # evaluator; no loss-ranked candidate is auto-selected.
                    save(output / ("checkpoint-%06d" % step), False)
            if stopping:
                event(
                    {
                        "phase": "paused_after_preflight",
                        "resumable": True,
                        "best_development_loss": best,
                    }
                )
                return
        epoch += 1
        offset = 0
    save(latest, True)
    event(
        {
            "phase": "training_complete",
            "best_development_loss": best,
            "heldout_test_evaluated": False,
            "task_selection_mode": mode,
            "quality_claim": "not_established_until_generation_replay",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Path to the local base-model checkpoint"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Directory containing the prepared dataset splits",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for this training run; reuse it with --resume",
    )
    parser.add_argument(
        "--purpose",
        choices=("qualification", "operator"),
        default="qualification",
        help="Operator training requires adequate operator and development-action coverage",
    )
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--init-adapter",
        help="Continue supervised adaptation from this frozen LoRA checkpoint",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum tokens per complete example",
    )
    parser.add_argument(
        "--accumulation",
        type=int,
        default=4,
        help="Microbatches per optimizer update",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="LoRA learning rate"
    )
    parser.add_argument("--seed", type=int, default=41, help="Random seed")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=25,
        help="Optimizer updates between development evaluations",
    )
    parser.add_argument(
        "--task-eval-every",
        type=int,
        default=0,
        help="Select by verified development replay every N updates; 0 retains legacy loss selection",
    )
    parser.add_argument(
        "--cpu-staged-loading",
        action="store_true",
        help="Clone each memory-mapped CPU weight before copying it to CUDA",
    )
    parser.add_argument(
        "--external-task-selection",
        action="store_true",
        help="Save evaluation checkpoints for a separate task evaluator; "
        "leave selection pending and never select by token loss",
    )
    parser.add_argument(
        "--dev-examples",
        type=int,
        default=16,
        help="Maximum development examples to evaluate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Pause after this many updates in this invocation (0: finish training)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the saved training state",
    )
    args = parser.parse_args()
    try:
        train(args)
    except Exception as exc:
        # Terminal receipt: record any failure before re-raising so an
        # interrupted run is distinguishable from a missing one.
        atomic_json(
            Path(args.output) / "failure.json",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
