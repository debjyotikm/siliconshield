"""DPO pilot for factual grounding of the Glimmer operator controller.

Starts from the trained operator adapter (reference and init alike) and
teaches explanation grounding from reviewed preference pairs: identical
observations and actions, where only one explanation is supported by the
evidence. Tool-result and prompt tokens never enter the response objective;
only assistant spans contribute, and the shared prefix cancels in the
pairwise difference.

Reference log probabilities are precomputed once from the frozen starting
adapter and stored, so training loads a single base model plus one trainable
adapter copy. The reference must include the starting adapter; disabling it
would silently change the reference to the base model.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

from siliconshield.research_loop.episodes.contracts import (
    OPERATOR_TOOLS,
    atomic_json,
)
from siliconshield.training.glimmer.identity import MODEL_REVISION
from siliconshield.training.glimmer.tokenization import encode_episode
from siliconshield.training.glimmer.train import rotate_checkpoint


def load_pairs(path, include_regression_only=False):
    from siliconshield.training.glimmer.dpo_pairs import (
        is_regression_only,
        validate_pair,
    )

    pairs = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not pairs:
        raise ValueError("No preference pairs")
    for pair in pairs:
        issues = validate_pair(pair)
        if issues:
            raise ValueError("Pair %s invalid: %s" % (pair.get("id"), issues))
    if len({p["id"] for p in pairs}) != len(pairs):
        raise ValueError("Duplicate pair IDs")
    if not include_regression_only:
        filtered = [p for p in pairs if not is_regression_only(p)]
        if not filtered:
            raise ValueError(
                "No eligible preference pairs after regression filter"
            )
        return filtered
    return pairs


def tokenizer_fingerprint(tokenizer, max_length):
    """Identity of the encoding inputs: chat template + vocab + length.

    Reference scores are only valid for the exact tokenizer/template that
    produced them; training must reject a reference built with different
    encoding.
    """
    template = getattr(tokenizer, "chat_template", "") or ""
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = int(getattr(tokenizer, "vocab_size", 0))
    return hashlib.sha256(
        json.dumps(
            {
                "chat_template": template,
                "vocab_size": vocab_size,
                "max_length": max_length,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def pair_binding_hashes(pairs):
    from siliconshield.training.glimmer.dpo_pairs import pair_content_hash

    return {p["id"]: pair_content_hash(p) for p in pairs}


def _row_for(pair, side):
    response = pair[side]
    decide = dict(response["decide"])
    messages = list(pair["prompt"]) + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "decide",
                    "type": "function",
                    "function": {
                        "name": "operator.decide",
                        "arguments": decide,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "decide",
            "content": '{"recorded": true, "executed": false}',
        },
        {
            "role": "assistant",
            "content": json.dumps(response["final"], sort_keys=True),
        },
    ]
    return {
        "id": "%s-%s" % (pair["id"], side),
        "group_id": "dpo-%s" % pair["id"],
        "tools": OPERATOR_TOOLS,
        "messages": messages,
        "summary": {"run_id": response["decide"]["run_id"]},
        "verification": {"kind": "authored_fixture"},
    }


def encode_pair(tokenizer, pair, max_length):
    """Encode both responses; labels cover assistant spans only.

    Prompt tokens and tool results are masked (-100) by the episode encoder,
    so they never enter the response objective. The shared inspect/resources
    prefix is identical on both sides and cancels in the DPO difference.
    """
    return {
        side: encode_episode(tokenizer, _row_for(pair, side), max_length)
        for side in ("chosen", "rejected")
    }


def sequence_logprob_sum(model, enc):
    """Sum of log probabilities over unmasked response tokens."""
    import torch

    ids = torch.tensor([enc["input_ids"]], device="cuda", dtype=torch.long)
    mask = torch.tensor(
        [enc["attention_mask"]], device="cuda", dtype=torch.long
    )
    labels = torch.tensor([enc["labels"]], device="cuda", dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=mask).logits
    logp = torch.log_softmax(logits.float(), dim=-1)
    shifted = logp[:, :-1, :]
    targets = labels[:, 1:].unsqueeze(-1).clamp(min=0)
    gathered = shifted.gather(-1, targets).squeeze(-1)
    kept = (labels[:, 1:] != -100).float()
    if not kept.sum():
        raise ValueError("No response tokens to score")
    return (gathered * kept).sum()


def dpo_loss(chosen_lp, rejected_lp, chosen_ref, rejected_ref, beta):
    """Standard DPO objective on summed response log probabilities."""
    import torch

    margin = beta * ((chosen_lp - rejected_lp) - (chosen_ref - rejected_ref))
    return -torch.nn.functional.logsigmoid(margin)


def reference_dpo(args):
    from siliconshield.research_loop.execution.gpu_lease import acquire_gpu
    from siliconshield.training.glimmer.identity import verify_code_identity

    verified_code = verify_code_identity()
    with acquire_gpu():
        return _compute_reference_logprobs(args, verified_code)


def _compute_reference_logprobs(args, verified_code):
    """Score every eligible pair under the frozen starting adapter; store sums.

    Uses the same verified execution wrapper as training: GPU lock, code
    identity, image digest, and pinned revision. Records the exact bytes the
    scores were computed from so training can reject mismatches before
    loading the GPU model.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration

    from siliconshield.training.glimmer.identity import (
        adapter_manifest,
        verify_snapshot,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    image_digest = os.environ.get("SILICONSHIELD_IMAGE_DIGEST")
    if not image_digest:
        raise ValueError(
            "Set SILICONSHIELD_IMAGE_DIGEST to the resolved execution image ID"
        )
    fingerprints = {
        "adapter": adapter_manifest(args.init_adapter)["fingerprint"]
    }
    snapshot = verify_snapshot(
        args.model,
        os.environ.get("SILICONSHIELD_MODEL_FINGERPRINT"),
        revision=MODEL_REVISION,
    )
    fingerprints["snapshot"] = snapshot["fingerprint"]
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA device supporting BF16 is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            Path(args.model) / "chat_template.jinja"
        ).read_text()
    pairs = load_pairs(args.pairs)
    pairs_sha = hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest()
    tok_fp = tokenizer_fingerprint(tokenizer, args.max_length)
    base, loading = MuseGlimmerForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        attn_implementation="sdpa",
        output_loading_info=True,
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
        raise RuntimeError("Published checkpoint did not load exactly")
    model = PeftModel.from_pretrained(base, args.init_adapter)
    model.eval()
    reference = {
        "init_adapter": args.init_adapter,
        "fingerprints": fingerprints,
        "pairs_sha256": pairs_sha,
        "pair_hashes": pair_binding_hashes(pairs),
        "tokenizer_fingerprint": tok_fp,
        "max_length": args.max_length,
        "base_revision": MODEL_REVISION,
        "code_identity": verified_code,
        "image_digest": image_digest,
        "pairs": {},
    }
    with torch.no_grad():
        for pair in pairs:
            encoded = encode_pair(tokenizer, pair, args.max_length)
            reference["pairs"][pair["id"]] = {
                side: float(sequence_logprob_sum(model, encoded[side]))
                for side in ("chosen", "rejected")
            }
    atomic_json(output / "reference_logprobs.json", reference)
    print(
        json.dumps(
            {
                "pairs": len(pairs),
                "init_adapter_fingerprint": fingerprints["adapter"],
            }
        )
    )
    return reference


# Backwards-compatible entry point used by earlier tooling; enforces the
# same wrapper so direct calls cannot bypass GPU ownership or code identity.
def compute_reference_logprobs(args):
    return reference_dpo(args)


def train_dpo(args):
    from siliconshield.research_loop.execution.gpu_lease import acquire_gpu
    from siliconshield.training.glimmer.identity import verify_code_identity

    verified_code = verify_code_identity()
    with acquire_gpu():
        return _train_dpo(args, verified_code)


def check_resume_state(args, latest):
    """Validate --resume against on-disk state before loading any GPU model.

    Explicit --resume with a missing checkpoint must fail, not silently
    start from the initial adapter. A fresh run with leftover state must
    require --resume.
    """
    from pathlib import Path as _Path

    latest = _Path(latest)
    if getattr(args, "resume", False):
        if not (latest / "dpo_state.pt").exists():
            raise ValueError(
                "Explicit --resume requires an existing checkpoint: %s is missing"
                % str(latest / "dpo_state.pt")
            )
    else:
        if latest.exists():
            raise ValueError("An existing DPO state requires --resume")


def _train_dpo(args, verified_code):
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration

    from siliconshield.research_loop.episodes.contracts import read_json
    from siliconshield.training.glimmer.identity import (
        adapter_manifest,
        verify_snapshot,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.json"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA device supporting BF16 is required")
    pairs = load_pairs(args.pairs)
    reference = json.loads(Path(args.reference_logprobs).read_text())
    missing = [
        p["id"] for p in pairs if p["id"] not in reference.get("pairs", {})
    ]
    if missing:
        raise ValueError(
            "Reference log probabilities missing for pairs: %s" % missing[:3]
        )
    # Bind reference to its exact inputs before touching the GPU model.
    # Editing a pair while keeping its ID must not silently change the objective.
    current_pairs_sha = hashlib.sha256(
        Path(args.pairs).read_bytes()
    ).hexdigest()
    if reference.get("pairs_sha256") != current_pairs_sha:
        raise ValueError(
            "Reference was computed from different pair bytes: %s != %s"
            % (reference.get("pairs_sha256"), current_pairs_sha)
        )
    current_hashes = pair_binding_hashes(pairs)
    ref_hashes = reference.get("pair_hashes") or {}
    mismatched = [
        pid for pid, h in current_hashes.items() if ref_hashes.get(pid) != h
    ]
    if mismatched:
        raise ValueError(
            "Reference scores do not match current pair contents: %s"
            % mismatched[:3]
        )
    if not ref_hashes:
        raise ValueError(
            "Reference file lacks per-pair content binding; regenerate it"
        )
    if reference.get("base_revision") != MODEL_REVISION:
        raise ValueError(
            "Reference was computed from a different base revision"
        )
    if reference.get("max_length") != args.max_length:
        raise ValueError(
            "Reference max_length %s differs from training %s"
            % (reference.get("max_length"), args.max_length)
        )
    init_fingerprint = adapter_manifest(args.init_adapter)["fingerprint"]
    if reference.get("fingerprints", {}).get("adapter") != init_fingerprint:
        raise ValueError(
            "Reference file was computed from a different starting adapter"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            Path(args.model) / "chat_template.jinja"
        ).read_text()
    if reference.get("tokenizer_fingerprint") != tokenizer_fingerprint(
        tokenizer, args.max_length
    ):
        raise ValueError(
            "Reference tokenizer/template differs from training tokenizer"
        )
    snapshot = verify_snapshot(
        args.model,
        os.environ.get("SILICONSHIELD_MODEL_FINGERPRINT"),
        revision=MODEL_REVISION,
    )
    if (
        reference.get("fingerprints", {}).get("snapshot")
        != snapshot["fingerprint"]
    ):
        raise ValueError("Reference snapshot differs from training snapshot")
    encoded = [encode_pair(tokenizer, pair, args.max_length) for pair in pairs]
    signature = {
        "purpose": "dpo-pilot",
        "base_revision": MODEL_REVISION,
        "beta": args.beta,
        "lr": args.lr,
        "seed": args.seed,
        "max_length": args.max_length,
        "accumulation": args.accumulation,
        "init_adapter_fingerprint": init_fingerprint,
        "pairs_sha256": current_pairs_sha,
        "reference_sha256": hashlib.sha256(
            Path(args.reference_logprobs).read_bytes()
        ).hexdigest(),
    }
    signature["snapshot_fingerprint"] = snapshot["fingerprint"]
    signature["image_digest"] = os.environ.get("SILICONSHIELD_IMAGE_DIGEST")
    if not signature["image_digest"]:
        raise ValueError(
            "Set SILICONSHIELD_IMAGE_DIGEST to the resolved execution image ID"
        )
    signature.update(
        code_identity=verified_code,
        eval_every=0,
        task_eval_every=0,
        torch=torch.__version__,
    )
    current = (
        read_json(output / "configuration.json")
        if (output / "configuration.json").exists()
        else None
    )
    if current and current != signature:
        raise ValueError("Resume configuration differs from the recorded run")
    atomic_json(output / "configuration.json", signature)
    base, loading = MuseGlimmerForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        attn_implementation="sdpa",
        output_loading_info=True,
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
        raise RuntimeError("Published checkpoint did not load exactly")
    base.config.use_cache = False
    base.config.text_config.use_cache = False
    latest = output / "latest"
    check_resume_state(args, latest)
    if (latest / "dpo_state.pt").exists():
        model = PeftModel.from_pretrained(
            base, latest / "adapter", is_trainable=True
        )
        saved = torch.load(
            latest / "dpo_state.pt", map_location="cpu", weights_only=False
        )
        if saved.get("signature") != signature:
            raise ValueError(
                "Resume configuration differs from the recorded DPO run"
            )
        step, offset = saved["step"], saved["offset"]
        optimizer_state = saved["optimizer"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        if torch.cuda.is_available():
            if "cuda_rng" not in saved:
                raise ValueError(
                    "Checkpoint lacks CUDA RNG state; cannot resume deterministically"
                )
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    else:
        # check_resume_state already rejected leftover state without --resume.
        # Trainable copy of the starting adapter; the source checkpoint on disk is never modified.
        model = PeftModel.from_pretrained(
            base, args.init_adapter, is_trainable=True
        )
        step, offset, optimizer_state = 0, 0, None
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
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    def event(value):
        value = {"step": step, "time": time.time(), **value}
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        atomic_json(status, value)
        print(json.dumps(value, allow_nan=False), flush=True)

    def save():
        stage = latest.with_name(latest.name + ".tmp")
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        model.save_pretrained(stage / "adapter")
        tokenizer.save_pretrained(stage / "adapter")
        checkpoint = {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "offset": offset,
            "signature": signature,
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }
        if torch.cuda.is_available():
            checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, stage / "dpo_state.pt")
        atomic_json(
            stage / "receipt.json", {"step": step, "signature": signature}
        )
        rotate_checkpoint(stage, latest)

    order = list(range(len(encoded)))
    random.Random(args.seed).shuffle(order)
    model.train()
    start = time.monotonic()
    ref = reference["pairs"]
    while offset < len(order):
        indices = order[offset : offset + args.accumulation]
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for index in indices:
            pair = pairs[index]
            enc = encoded[index]
            chosen_lp = sequence_logprob_sum(model, enc["chosen"])
            rejected_lp = sequence_logprob_sum(model, enc["rejected"])
            loss = dpo_loss(
                chosen_lp,
                rejected_lp,
                torch.tensor(ref[pair["id"]]["chosen"], device="cuda"),
                torch.tensor(ref[pair["id"]]["rejected"], device="cuda"),
                args.beta,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite DPO loss")
            (loss / len(indices)).backward()
            loss_sum += float(loss.detach()) / len(indices)
        gradient = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)
        if not torch.isfinite(gradient) or float(gradient) == 0:
            raise RuntimeError("Missing or non-finite adapter gradients")
        optimizer.step()
        offset += len(indices)
        step += 1
        event(
            {
                "phase": "dpo_training",
                "loss": loss_sum,
                "gradient_norm": float(gradient),
                "pairs_per_second": (offset) / (time.monotonic() - start),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
        save()
        if args.max_steps and step >= args.max_steps:
            event({"phase": "dpo_paused", "resumable": True})
            return
    event(
        {
            "phase": "dpo_complete",
            "quality_claim": "not_established_until_generation_replay",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "train"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--init-adapter", help="Starting operator adapter (reference and init)"
    )
    parser.add_argument(
        "--pairs", help="Reviewed preference-pair JSONL (outside Git)"
    )
    parser.add_argument(
        "--reference-logprobs",
        help="Precomputed reference log probabilities (train mode)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "reference" and not args.init_adapter:
        raise SystemExit(
            "--init-adapter is required: the reference must include the starting adapter"
        )
    try:
        if args.mode == "reference":
            reference_dpo(args)
        else:
            if not args.reference_logprobs:
                raise SystemExit("train mode requires --reference-logprobs")
            train_dpo(args)
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
