"""CPU pre-submit validation for training-study data inputs.

Runs the trainer's own dataset gates without a GPU: split loading with
question-contract bytes, pool-manifest assignment, grouped-sampler
simulation over the planned horizon, and reporting-weight coverage.
A recipe supplies declared source mass; without one the trainer
defaults apply.

An optional model export enables a token-length screen: each state's
encoded length is a lower bound on the real model input, so a row
over the limit provably truncates, while rows under it are only
unflagged, not proven safe.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from siliconshield.training.laya import security_train as train
from siliconshield.training.laya.question_contracts import LEGACY_CONTRACT


def sha256(path):
    """Return the hex digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_recipe_mass(recipe):
    """Return declared source mass from a recipe file, or None."""
    if recipe is None:
        return None
    manifest = json.loads(Path(recipe).read_text(encoding="utf-8"))
    mass = manifest.get("source_mass")
    if mass is None:
        return None
    return {str(name): float(value) for name, value in mass.items()}


def simulate_coverage(rows, pool_plan, seed, global_batch, world, *, cap=20):
    """Simulate grouped sampling until every triplet is drawn.

    Mirrors the trainer loop by carrying one queue state across
    epochs. Returns covered epochs (None past the cap), the update
    horizon for full coverage, and per-stratum draw counts.
    """
    triplets = {row["pair_id"] for row in rows}
    seen, draws, queue = set(), {}, {}
    batches_per_epoch = math.ceil(len(rows) / global_batch)
    for epoch in range(cap):
        for rank in range(world):
            indices = train.grouped_epoch_indices(
                rows,
                world,
                rank,
                seed,
                epoch,
                global_batch,
                pool_plan=pool_plan,
                _queue_state=queue,
            )
            for index in indices:
                if pool_plan is not None:
                    stratum = pool_plan[0][index]
                else:
                    stratum = rows[index]["source"]["name"]
                draws[stratum] = draws.get(stratum, 0) + 1
                seen.add(rows[index]["pair_id"])
        if seen >= triplets:
            epochs = epoch + 1
            return {
                "epochs_to_cover": epochs,
                "updates_to_cover": epochs * batches_per_epoch,
                "triplets": len(triplets),
                "draws": draws,
            }
    return {
        "epochs_to_cover": None,
        "updates_to_cover": None,
        "triplets": len(triplets),
        "seen_triplets": len(seen),
        "draws": draws,
    }


def screen_token_lengths(rows, model, max_len):
    """Flag rows whose encoded state alone exceeds the limit.

    Returns ``(max_tokens, over_limit)``. Skipped without a model
    export or an installed tokenizer backend.
    """
    if model is None:
        return {"status": "skipped", "reason": "no model export given"}
    if AutoTokenizer is None:
        return {"status": "skipped", "reason": "no tokenizer backend"}
    tokenizer = AutoTokenizer.from_pretrained(str(Path(model) / "tokenizer"))
    over, longest = [], 0
    for row in rows:
        tokens = len(tokenizer.encode(row["state"]))
        longest = max(longest, tokens)
        if tokens > max_len:
            over.append(row["id"])
    return {
        "status": "checked",
        "max_state_tokens": longest,
        "max_len": max_len,
        "over_limit": over,
    }


def validate(
    data,
    *,
    question_contract=LEGACY_CONTRACT,
    pools=None,
    recipe=None,
    seed=41,
    global_batch=48,
    world=1,
    max_updates=None,
    limit_train_rows=None,
    model=None,
    max_len=1024,
):
    """Validate study inputs with the trainer's own gates.

    Raises ValueError with the trainer's message on any violation.
    Returns a receipt dict binding inputs by hash.
    """
    if global_batch % 3 or global_batch % world:
        raise ValueError(
            "grouped batch must contain whole triplets and equal rank shards"
        )
    mass, saved = load_recipe_mass(recipe), train.SOURCE_MASS
    if mass is not None:
        train.SOURCE_MASS = mass
    try:
        splits = train.load_splits(
            data,
            train_only=limit_train_rows is not None,
            question_contract=question_contract,
        )
        rows = splits["train"]
        if limit_train_rows is not None:
            rows = train.fitting_subset(rows, limit_train_rows, seed)
        pool_plan = None
        if pools is not None:
            pool_plan = train.pool_assignment_for_rows(pools, rows)
        coverage = simulate_coverage(rows, pool_plan, seed, global_batch, world)
        checked = list(rows) + list(splits.get("dev", []))
        train.hierarchical_weights(checked)
        screen = screen_token_lengths(checked, model, max_len)
    finally:
        train.SOURCE_MASS = saved
    if screen.get("over_limit"):
        raise ValueError(
            "rows provably exceed max_len: " + ",".join(screen["over_limit"])
        )
    inputs = {
        str(Path(data) / (split + ".jsonl")): sha256(
            Path(data) / (split + ".jsonl")
        )
        for split in splits
    }
    if pools is not None:
        inputs[str(pools)] = sha256(pools)
    if recipe is not None:
        inputs[str(recipe)] = sha256(recipe)
    horizon = coverage["updates_to_cover"]
    return {
        "result": "pass",
        "inputs": inputs,
        "counts": {split: len(rows) for split, rows in splits.items()},
        "trained_rows": len(rows),
        "coverage": coverage,
        "budget_covers_data": (
            None
            if max_updates is None or horizon is None
            else max_updates >= horizon
        ),
        "token_screen": screen,
    }


def main(argv=None):
    """Validate study inputs from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset directory")
    parser.add_argument(
        "--question-contract", default=LEGACY_CONTRACT, help="Contract ID"
    )
    parser.add_argument("--pools", default=None, help="Pool manifest")
    parser.add_argument("--recipe", default=None, help="Recipe JSON")
    parser.add_argument("--seed", type=int, default=41, help="Random seed")
    parser.add_argument(
        "--global-batch", type=int, default=48, help="Rows per update"
    )
    parser.add_argument("--world", type=int, default=1, help="GPU ranks")
    parser.add_argument(
        "--max-updates", type=int, default=None, help="Planned budget"
    )
    parser.add_argument(
        "--limit-train-rows",
        type=int,
        default=None,
        help="TRAIN-only fitting selection",
    )
    parser.add_argument("--model", default=None, help="Model export")
    parser.add_argument(
        "--max-len", type=int, default=1024, help="Input token limit"
    )
    parser.add_argument(
        "--output-receipt", default=None, help="Receipt output path"
    )
    args = parser.parse_args(argv)
    receipt = validate(
        args.data,
        question_contract=args.question_contract,
        pools=args.pools,
        recipe=args.recipe,
        seed=args.seed,
        global_batch=args.global_batch,
        world=args.world,
        max_updates=args.max_updates,
        limit_train_rows=args.limit_train_rows,
        model=args.model,
        max_len=args.max_len,
    )
    if args.output_receipt:
        Path(args.output_receipt).write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )
    print(
        "pass: %(counts)s coverage=%(coverage)s"
        % {
            "counts": receipt["counts"],
            "coverage": receipt["coverage"]["updates_to_cover"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
