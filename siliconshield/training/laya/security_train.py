"""Train SiliconShield-RTL security decisions (LAYA runtime, BF16, DDP)."""

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import transformers
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel


def _laya_root():
    """Locate the external LAYA checkout providing the ``laya`` package.

    SiliconShield does not vendor LAYA core. Set ``LAYA_ROOT`` to a laya-next
    checkout, or install the ``laya`` package; the SiliconShield repository
    root is also accepted when it contains a ``laya/`` directory.
    """
    candidates = [
        os.environ.get("LAYA_ROOT"),
        str(Path(__file__).resolve().parents[3]),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "laya").is_dir():
            return Path(candidate)
    return None


_LAYA_ROOT = _laya_root()
if _LAYA_ROOT is not None:
    sys.path.insert(0, str(_LAYA_ROOT))
_SCRIPT_ROOT = Path(__file__).resolve().parents[3]
if str(_SCRIPT_ROOT) not in sys.path:
    # Bare-script launch (torchrun on this file) puts only the training
    # directory on sys.path, but sibling modules also import
    # siliconshield.*. Anchor the script's own tree so the script and
    # the package always come from the same checkout.
    sys.path.insert(0, str(_SCRIPT_ROOT))
from laya._residual import ResidualAdapters
from laya.common import (
    build_sequence,
    collate_items,
    encode_text,
    serialize_state,
)
from laya.train import (
    encode_item,
    load_checkpoint,
    make_item,
    target_from_expected,
    to_internal,
)

from siliconshield.training.laya.replay_preservation import (
    preservation_diagnostics,
    preservation_loss,
    teacher_checkpoint,
    validate_preservation,
)

if __package__:
    from siliconshield.training.laya.checkpoint_selection import (
        triplet_counts,
    )
    from siliconshield.training.laya.coverage_audit import (
        delivered_exposure,
        exposure_gate_passed,
        reconstruct_seen,
    )
    from siliconshield.training.laya.ddp_guard import (
        require_coherent_training,
    )
    from siliconshield.training.laya.development_views import (
        attach_views,
        check_selection_reference,
        ranking_version,
    )
    from siliconshield.training.laya.development_views import (
        selection as view_selection,
    )
    from siliconshield.training.laya.new_study import (
        load_readonly_checkpoint,
        verify_new_study,
    )
    from siliconshield.training.laya.question_contracts import (
        LEGACY_CONTRACT,
        NEW_CONTRACT,
        read_export_record,
        write_export_record,
    )
    from siliconshield.training.laya.question_contracts import (  # type: ignore[no-redef]
        validate_rows as validate_question_rows,
    )
    from siliconshield.training.laya.security_reporting import evaluation_report
else:
    # Bare-script fallback when run directly from the training directory
    # (package import above is the supported path). The ignores below cover
    # the intentional dual import: exactly one branch executes.
    from checkpoint_selection import (  # type: ignore[no-redef]
        triplet_counts,
    )
    from coverage_audit import (  # type: ignore[no-redef]
        delivered_exposure,
        exposure_gate_passed,
        reconstruct_seen,
    )
    from ddp_guard import require_coherent_training  # type: ignore[no-redef]
    from development_views import (  # type: ignore[no-redef]
        attach_views,  # type: ignore[no-redef]
        check_selection_reference,
        ranking_version,
    )
    from development_views import (  # type: ignore[no-redef]
        selection as view_selection,
    )
    from new_study import (  # type: ignore[no-redef]
        load_readonly_checkpoint,
        verify_new_study,
    )
    from question_contracts import (  # type: ignore[no-redef]
        LEGACY_CONTRACT,
        NEW_CONTRACT,
        read_export_record,
        write_export_record,
    )
    from question_contracts import (  # type: ignore[no-redef]
        validate_rows as validate_question_rows,
    )
    from security_reporting import evaluation_report  # type: ignore[no-redef]


LABELS = [
    "satisfies_requirement",
    "violates_requirement",
    "insufficient_context",
]
# Version of the trainer development report (report.json) consumed by the
# research loop through adapters (see episodes.laya_results.TRAINER_SCHEMA,
# which must stay equal; enforced by tests/test_security_training.py).
REPORT_SCHEMA = 2
# Example defaults; load task settings from SILICONSHIELD_SECURITY_RECIPE.
# See examples/security_recipe.example.json for the format.
SOURCE_MASS = {
    "HardSecBench": 1 / 3,
    "authored-semantic": 1 / 3,
    "OpenTitan": 1 / 3,
}
ACCEPTANCE = {
    "max_worst_family_false_safe_rate": 0.25,
    "min_class_recall": 0.50,
    "min_pair_accuracy": 0.50,
    "require_baseline_improvement": True,
}


def _selection_reference_from_recipe(recipe):
    """Validate and return a recipe's frozen selection reference."""
    if "selection_reference" not in recipe:
        return None
    reference = recipe["selection_reference"]
    check_selection_reference(reference)
    return reference


def _apply_local_recipe():
    import os

    selected = os.environ.get("SILICONSHIELD_SECURITY_RECIPE")
    if not selected:
        return
    recipe = json.loads(Path(selected).read_text())
    global SOURCE_MASS, ACCEPTANCE, SELECTION_REFERENCE, REPLAY_PRESERVATION
    if "source_mass" in recipe:
        SOURCE_MASS = {
            str(k): float(v) for k, v in recipe["source_mass"].items()
        }
    if "acceptance" in recipe:
        ACCEPTANCE = dict(recipe["acceptance"])
    SELECTION_REFERENCE = _selection_reference_from_recipe(recipe)
    REPLAY_PRESERVATION = validate_preservation(
        recipe.get("replay_preservation")
    )


SELECTION_REFERENCE = None
REPLAY_PRESERVATION = None


_apply_local_recipe()


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def verify_execution_manifest(run_id):
    """Verify the exact launch inputs again inside the training process."""
    path = os.environ.get("SILICONSHIELD_EXECUTION_MANIFEST")
    if not path:
        return (
            None  # Direct CLI invocations retain their recorded run signature.
        )
    manifest = json.loads(Path(path).read_text())
    if manifest["experiment_id"] != run_id or not manifest.get("input_hashes"):
        raise ValueError(
            "Execution manifest does not identify this training run"
        )
    for name, expected in manifest["input_hashes"].items():
        if sha256(name) != expected:
            raise ValueError("Launch input differs inside trainer: " + name)
    return manifest["idempotency_key"]


def checkpoint_manifest(path):
    path = Path(path)
    return {
        str(file.relative_to(path)): sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def code_manifest():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    names = [
        "siliconshield/training/laya/security_train.py",
        "siliconshield/training/laya/security_campaign.py",
        "siliconshield/training/laya/new_study.py",
        "siliconshield/training/laya/development_views.py",
        "siliconshield/training/laya/checkpoint_selection.py",
        "siliconshield/training/laya/replay_preservation.py",
        "tests/test_checkpoint_selection.py",
        "tests/test_replay_preservation.py",
        "tests/test_laya_new_study.py",
        "siliconshield/training/laya/security_reporting.py",
        "tests/test_security_reporting.py",
        "siliconshield/training/laya/security_continuation.py",
        "siliconshield/training/laya/build_security_data.py",
        "siliconshield/training/laya/build_security_external.py",
        "siliconshield/training/laya/expand_security_data.py",
        "siliconshield/training/laya/augment_security_policies.py",
        "siliconshield/training/laya/augment_security_guards.py",
        "tests/test_security_training.py",
    ]
    result = {
        name: sha256(repo / name) for name in names if (repo / name).is_file()
    }
    if _LAYA_ROOT is not None:
        for name in (
            "laya/common.py",
            "laya/train.py",
            "laya/_residual.py",
            "laya/agent.py",
            "laya/calibrate.py",
        ):
            if (_LAYA_ROOT / name).is_file():
                result[name] = sha256(_LAYA_ROOT / name)
    result["runtime"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    result["combined_sha256"] = digest_text(json.dumps(result, sort_keys=True))
    return result


def canonical_input(state):
    if not isinstance(state, str):
        raise ValueError("Canonical input needs state text")
    text = re.sub(r"\b([A-Za-z_]\w*)_[12]\b", r"\1", state)
    return re.sub(r"\s+", "", text)


def optimizer_param_groups(model, encoder_lr, adapter_lr, head_lr):
    """Encoder/adapter/head optimizer groups with separate learning rates.

    Membership is by module name: residual-adapter branches form their own
    group so their learning rate is independent of the decision head.
    Frozen (requires_grad False) parameters are excluded from every group.
    """
    groups = []
    for kind in ("encoder", "adapter", "head"):
        for decay in (True, False):
            parameters = [
                value
                for name, value in model.named_parameters()
                if value.requires_grad
                and (
                    "residual_adapters" in name
                    if kind == "adapter"
                    else name.startswith("encoder.")
                    if kind == "encoder"
                    else "residual_adapters" not in name
                    and not name.startswith("encoder.")
                )
                and (value.ndim >= 2) == decay
            ]
            if parameters:
                learning_rate = {
                    "encoder": encoder_lr,
                    "adapter": adapter_lr,
                    "head": head_lr,
                }[kind]
                groups.append(
                    {
                        "params": parameters,
                        "lr": learning_rate,
                        "base_lr": learning_rate,
                        "weight_decay": 0.01 if decay else 0.0,
                        "group_name": kind,
                    }
                )
    return groups


def fitting_subset(rows, limit, seed):
    """Select complete designs, retaining triplets and policy counterparts.

    The requested size is a lower bound: the final design is never cut in half.
    Round-robin source selection prevents a small prefix from dropping a source.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source"]["name"], row["base_design_id"])].append(row)
    sources = defaultdict(list)
    for key, values in sorted(groups.items()):
        sources[key[0]].append(values)
    rng = random.Random(seed)
    for values in sources.values():
        rng.shuffle(values)
    selected = []
    while any(sources.values()) and len(selected) < limit:
        for source in sorted(sources):
            if sources[source]:
                selected.extend(sources[source].pop())
    return selected


def configure_trainable_model(
    model,
    cfg,
    freeze_encoder,
    disable_adapters,
    adapter_scale,
    continue_training,
    encoder_trainable_blocks=None,
):
    """Configure branches before freezing; a head-only model has no branches.

    ``encoder_trainable_blocks`` unfreezes only the top-K encoder blocks
    (embeddings and other non-block encoder parameters stay frozen);
    it contradicts ``freeze_encoder``.
    """
    existing = getattr(model, "residual_adapters", None)
    if existing is not None and not continue_training:
        raise ValueError("adapted initialization requires --continue-training")
    if disable_adapters:
        if existing is not None:
            # Removing hooked, already-trained branches would change export behavior.
            raise ValueError(
                "disabled-adapter training requires an unadapted base checkpoint"
            )
        adapter_cfg = None
    elif existing is not None:
        adapter_cfg = cfg["residual_adapters"]
        if adapter_scale is not None and adapter_scale != adapter_cfg["scale"]:
            raise ValueError(
                "continuation cannot silently change adapter scale"
            )
    else:
        adapter_cfg = {
            "version": 1,
            "layers": list(range(len(model.encoder.layers))),
            "rank": 256,
            "scale": 8.0 if adapter_scale is None else adapter_scale,
        }
        model.residual_adapters = ResidualAdapters(model.encoder, adapter_cfg)
    model.encoder.requires_grad_(not freeze_encoder)
    if encoder_trainable_blocks is not None:
        if freeze_encoder:
            raise ValueError(
                "encoder block selection contradicts --freeze-encoder"
            )
        layers = model.encoder.layers
        count = len(layers)
        if (
            not isinstance(encoder_trainable_blocks, int)
            or isinstance(encoder_trainable_blocks, bool)
            or not 1 <= encoder_trainable_blocks <= count
        ):
            raise ValueError(
                "trainable encoder blocks must be in 1..%d" % count
            )
        model.encoder.requires_grad_(False)
        for layer in layers[count - encoder_trainable_blocks :]:
            layer.requires_grad_(True)
    if getattr(model, "residual_adapters", None) is not None:
        model.residual_adapters.requires_grad_(True)
    model.act_head.requires_grad_(False)
    return adapter_cfg


def component_name(name):
    return (
        "adapter"
        if "residual_adapters" in name
        else "encoder"
        if name.startswith("encoder.")
        else "head"
    )


def component_gradient_norms(model):
    """Aggregate on the device, with one host synchronization per component."""
    squared = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            squared[component_name(name)].append(
                parameter.grad.detach().float().square().sum()
            )
    return {
        kind: float(torch.stack(squared[kind]).sum().sqrt())
        if squared[kind]
        else 0.0
        for kind in ("encoder", "adapter", "head")
    }


def load_splits(
    directory,
    final=False,
    train_only=False,
    question_contract=None,
):
    """Load and validate dataset splits with lineage and contract gates.

    Enforces split-group ownership, duplicate-input refusal,
    alpha-equivalent screening, triplet completeness, accepted
    validation, and — when ``question_contract`` is given — exact
    question-bytes agreement on every row.
    """
    if not isinstance(directory, (str, Path)):
        raise ValueError("Dataset directory must be a path")
    directory = Path(directory)
    names = (
        ["train", "dev", "calibration", "test"] if final else ["train", "dev"]
    )
    if train_only:
        if final:
            raise ValueError("TRAIN-only mode cannot read final partitions")
        names = ["train"]
    if final and (directory / "external.jsonl").exists():
        names.append("external")
    splits, owners, inputs, canonical, ids = {}, {}, {}, {}, set()
    for split in names:
        path = directory / (split + ".jsonl")
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError("empty split: " + split)
        pairs = defaultdict(set)
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("split %s has a non-object row" % split)
            if not isinstance(row.get("id"), str):
                raise ValueError(
                    "split %s has a row without a string id" % split
                )
            family = row["family_id"]
            group = row.get("split_group_id", family)
            if owners.setdefault(group, split) != split:
                raise ValueError("split group crosses splits: " + group)
            if row.get("split", split) != split:
                raise ValueError("row split label differs from ownership")
            if row["id"] in ids:
                raise ValueError("duplicate record ID: " + row["id"])
            ids.add(row["id"])
            key = json.dumps([row["state"], row["questions"]], sort_keys=True)
            if key in inputs:
                raise ValueError("duplicate input within or across splits")
            inputs[key] = split
            normalized = canonical_input(row["state"])
            if canonical.setdefault(normalized, split) != split:
                raise ValueError("alpha/format-equivalent input crosses splits")
            canonical[normalized] = split
            if row["expected"]["security"] not in LABELS:
                raise ValueError("unknown verdict")
            if not row.get("validation", {}).get("accepted"):
                raise ValueError("unvalidated record: " + row["id"])
            pairs[row["pair_id"]].add(row["expected"]["security"])
        if any(set(labels) != set(LABELS) for labels in pairs.values()):
            raise ValueError("incomplete decision triplet in " + split)
        row_by_id = {row["id"]: row for row in rows}
        for row in rows:
            counterpart = row.get("policy_counterpart_id")
            if counterpart is None:
                continue
            original = row_by_id.get(counterpart)
            if (
                original is None
                or original["base_design_id"] != row["base_design_id"]
            ):
                raise ValueError(
                    "policy counterpart missing or crosses designs"
                )
            if (
                original["state"].split("RTL\n", 1)[1]
                != row["state"].split("RTL\n", 1)[1]
            ):
                raise ValueError("policy counterpart changes RTL")
            expected = dict(zip(LABELS, (LABELS[1], LABELS[0], LABELS[2])))[
                original["expected"]["security"]
            ]
            if row["expected"]["security"] != expected:
                raise ValueError("policy counterpart target is not reversed")
        if question_contract is not None:
            validate_question_rows(rows, question_contract)
        splits[split] = rows
    return splits


def prepare_items(tok, rows, max_len, head_max_len):
    items = []
    for row in rows:
        question = row["questions"]["security"]
        internal = to_internal("security", question)
        if list(internal["crit"]) != LABELS:
            raise ValueError("canonical verdict order changed")
        text = serialize_state(row["state"]).replace(tok.mask_token, " ")
        state_ids = encode_text(tok, text, add_special_tokens=False)[
            "input_ids"
        ]
        ids, _, stats = build_sequence(
            tok,
            None,
            internal,
            max_len,
            head_max_len,
            state_ids=state_ids,
            return_truncation_stats=True,
        )
        if stats["truncated"]:
            raise ValueError("incomplete input %s: %s" % (row["id"], stats))
        target = target_from_expected(internal, row["expected"]["security"])
        item, error = make_item(
            tok, internal, target, state_ids, head_max_len, max_len
        )
        if error:
            raise ValueError("invalid input %s: %s" % (row["id"], error))
        item["length"] = len(ids)
        item["family_id"] = row["family_id"]
        item["pair_id"] = row["pair_id"]
        item["rewrite_group_id"] = row.get("rewrite_group_id", row["pair_id"])
        item["row_id"] = row["id"]
        items.append(item)
    return items


def epoch_indices(n, world, rank, seed, epoch):
    if n < 1:
        raise ValueError("empty training set")
    order = list(range(n))
    random.Random(seed + epoch).shuffle(order)
    total = math.ceil(n / world) * world
    order = (order * math.ceil(total / n))[:total]
    return order[rank:total:world]


def epoch_valid_mask(n, world, rank, seed, epoch):
    total = math.ceil(n / world) * world
    valid = [1] * n + [0] * (total - n)
    return valid[rank:total:world]


def pool_assignment_for_rows(pools_path, rows):
    """Map training-row indices to sampling pools with weights.

    Mirrors ``dataset_versions.pool_assignment`` without importing it:
    this module also runs as a bare file under torch.distributed.run,
    where sibling package imports are unavailable. The two
    implementations must agree; tests/test_security_training.py
    cross-checks them.
    """
    manifest = json.loads(Path(pools_path).read_text())
    pools = manifest["pools"]
    owner = {}
    for name, spec in pools.items():
        for row_id in spec["rows"]:
            if row_id in owner:
                raise ValueError("Pool manifest assigns twice: " + row_id)
            owner[row_id] = name
    assignment = {}
    for index, row in enumerate(rows):
        row_id = row.get("id")
        if row_id not in owner:
            raise ValueError("Row outside every pool: " + repr(row_id))
        assignment[index] = owner[row_id]
    weights = {}
    modes = {}
    for name, spec in pools.items():
        weight = spec["weight"]
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError("Pool weight must be positive: " + name)
        weights[name] = float(weight)
        mode = spec.get("mode", "cycle")
        if mode not in ("cycle", "flat"):
            raise ValueError("Pool mode must be cycle or flat: " + name)
        modes[name] = mode
    return assignment, weights, modes


def grouped_epoch_indices(
    rows,
    world,
    rank,
    seed,
    epoch,
    global_batch,
    pool_plan=None,
    _queue_state=None,
):
    """Stratify complete triplets by pool/source, mechanism and design.

    Uses unit loss weights. An epoch is a fixed number of draws, not a
    pass over every unique row. Global batches are shuffled before
    sharding so a rank cannot become class-specific.

    Without ``pool_plan`` the top stratum is the row's source name
    with SOURCE_MASS quotas. With ``pool_plan`` (an
    ``(assignment, mass, modes)`` triple from a pool manifest) the top
    stratum is the row's pool with pool-weight quotas, and every
    triplet must sit in one pool. Pools change batch composition
    only; reporting weights elsewhere stay source-based, so
    provenance is untouched. ``flat`` pools draw triplets uniformly;
    ``cycle`` pools stratify mechanism then design. Without
    ``_queue_state`` the draw queues reset every epoch (legacy); with
    a caller-owned dict the queues persist across epochs exactly as
    the trainer loop carries them.
    """
    for name, value in (
        ("world", world),
        ("rank", rank),
        ("seed", seed),
        ("epoch", epoch),
        ("global batch", global_batch),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("grouped sampling needs an integer " + name)
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) for row in rows
    ):
        raise ValueError("grouped sampling needs a list of row objects")
    if pool_plan is not None and (
        not isinstance(pool_plan, (tuple, list))
        or len(pool_plan) != 3
        or not isinstance(pool_plan[0], dict)
        or not isinstance(pool_plan[1], dict)
    ):
        raise ValueError("grouped sampling needs a pool plan triple")
    if _queue_state is not None and not isinstance(_queue_state, dict):
        raise ValueError("grouped sampling queue state must be an object")
    if global_batch % 3 or global_batch % world or not 0 <= rank < world:
        raise ValueError(
            "grouped batch must contain whole triplets and equal rank shards"
        )
    mass = SOURCE_MASS if pool_plan is None else pool_plan[1]

    def top(index):
        if pool_plan is not None:
            return pool_plan[0][index]
        return rows[index]["source"]["name"]

    rng = random.Random(seed * 1000003 + epoch)
    triplets = defaultdict(list)
    for index, row in enumerate(rows):
        triplets[row["pair_id"]].append(index)
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for indices in triplets.values():
        labels = [rows[index]["expected"]["security"] for index in indices]
        if len(indices) != 3 or set(labels) != set(LABELS):
            raise ValueError("grouped sampling requires complete triplets")
        first = rows[indices[0]]
        key = (
            top(indices[0]),
            first["mechanism_id"],
            first["base_design_id"],
        )
        if any(
            (
                top(i),
                rows[i]["mechanism_id"],
                rows[i]["base_design_id"],
            )
            != key
            for i in indices
        ):
            raise ValueError(
                "triplet crosses a stratum boundary"
                if pool_plan is not None
                else "triplet crosses a source, mechanism or design"
            )
        grouped[key[0]][key[1]][key[2]].append(indices)
    if set(grouped) != set(mass):
        raise ValueError(
            "grouped sampling requires every configured pool"
            if pool_plan is not None
            else "grouped sampling requires all configured sources"
        )
    names = sorted(grouped)
    slots = global_batch // 3
    total = sum(mass.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("stratum mass must sum to a positive total")
    raw = {name: mass[name] / total * slots for name in names}
    quotas = {name: int(raw[name]) for name in names}
    for name in sorted(
        names, key=lambda name: (raw[name] - quotas[name], name), reverse=True
    )[: slots - sum(quotas.values())]:
        quotas[name] += 1
    if not all(quotas.values()):
        raise ValueError(
            "global batch is too small for every pool"
            if pool_plan is not None
            else "global batch is too small for every source"
        )
    # Cycling shuffled pools gives rare mechanisms and every design regular exposure.
    pools = {} if _queue_state is None else _queue_state
    modes = pool_plan[2] if pool_plan is not None and len(pool_plan) > 2 else {}
    flat = {}
    if pool_plan is not None:
        for name in names:
            if modes.get(name, "cycle") == "flat":
                flat[name] = [
                    indices
                    for mechanism in grouped[name].values()
                    for design in mechanism.values()
                    for indices in design
                ]

    def take(key, values):
        if not pools.get(key):
            pools[key] = list(values)
            rng.shuffle(pools[key])
        return pools[key].pop()

    result = []
    for _ in range(math.ceil(len(rows) / global_batch)):
        batch = []
        for source in names:
            for _ in range(quotas[source]):
                if source in flat:
                    batch.extend(take((source,), flat[source]))
                    continue
                mechanism = take((source,), sorted(grouped[source]))
                design = take(
                    (source, mechanism), sorted(grouped[source][mechanism])
                )
                indices = take(
                    (source, mechanism, design),
                    grouped[source][mechanism][design],
                )
                batch.extend(indices)
        rng.shuffle(batch)
        result.extend(batch[rank::world])
    return result


def accumulation_groups(indices, micro_batch, accumulation):
    for start in range(0, len(indices), micro_batch * accumulation):
        group = indices[start : start + micro_batch * accumulation]
        chunks = [
            group[i : i + micro_batch]
            for i in range(0, len(group), micro_batch)
        ]
        yield (
            start + len(group),
            [
                (chunk, len(chunk) / len(group), j == len(chunks) - 1)
                for j, chunk in enumerate(chunks)
            ],
        )


def hierarchical_weights(rows):
    grouped = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for index, row in enumerate(rows):
        source = row["source"]["name"]
        if source not in SOURCE_MASS:
            raise ValueError("no declared source mass for " + source)
        mechanism = row["mechanism_id"]
        design = row["base_design_id"]
        label = row["expected"]["security"]
        grouped[source][mechanism][design][label].append(index)
    weights = np.zeros(len(rows), dtype=np.float64)
    for source, mechanisms in grouped.items():
        source_share = SOURCE_MASS[source] / len(mechanisms)
        for mechanism, designs in mechanisms.items():
            mechanism_share = source_share / len(designs)
            for design, classes in designs.items():
                if set(classes) != set(LABELS):
                    raise ValueError(
                        "weighting requires complete label groups: " + design
                    )
                class_share = mechanism_share / len(classes)
                for label, indices in classes.items():
                    for index in indices:
                        weights[index] = class_share / len(indices)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("invalid hierarchical weights")
    weights /= weights.mean()
    return weights


def weighting_report(rows, weights):
    total = float(weights.sum())
    sources = defaultdict(float)
    mechanisms = defaultdict(float)
    labels = defaultdict(float)
    for row, weight in zip(rows, weights):
        sources[row["source"]["name"]] += float(weight)
        mechanisms[row["mechanism_id"]] += float(weight)
        labels[row["expected"]["security"]] += float(weight)
    return {
        "source_mass": {k: v / total for k, v in sources.items()},
        "mechanism_mass": {k: v / total for k, v in mechanisms.items()},
        "label_mass": {k: v / total for k, v in labels.items()},
        "kish_row_ess": float(total**2 / np.square(weights).sum()),
        "configured_source_mass": dict(SOURCE_MASS),
    }


def probabilities_array(values, from_probabilities=False):
    array = np.asarray(values, dtype=np.float64)
    if not from_probabilities:
        return array
    if np.any(array < 0) or np.any(array.sum(axis=1, keepdims=True) <= 0):
        raise ValueError("invalid probabilities")
    array = np.clip(array, 0.00005, 1.0)
    array /= array.sum(axis=1, keepdims=True)
    return np.log(array)


def metrics(logits, targets, temperature=1.0, from_probabilities=False):
    z = probabilities_array(logits, from_probabilities)
    y = np.asarray(targets, dtype=np.int64)
    if (
        z.ndim != 2
        or z.shape != (len(y), 3)
        or not len(y)
        or not np.isfinite(z).all()
    ):
        raise ValueError("invalid/empty/misaligned prediction arrays")
    if (
        np.any((y < 0) | (y > 2))
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("invalid targets or temperature")
    z = z / temperature
    z -= z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    probabilities = np.exp(logp)
    prediction = probabilities.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (y, prediction), 1)
    precision = np.diag(matrix) / np.maximum(1, matrix.sum(axis=0))
    recall = np.diag(matrix) / np.maximum(1, matrix.sum(axis=1))
    f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    support = matrix.sum(axis=1)
    false_safe = int(matrix[1, 0])
    errors = int((prediction != y).sum())
    ece = 0.0
    confidence = probabilities.max(axis=1)
    for low in np.arange(0, 1, 0.1):
        mask = (confidence >= low) & (
            confidence < low + 0.1 if low < 0.9 else confidence <= 1
        )
        if mask.any():
            ece += mask.mean() * abs(
                confidence[mask].mean() - (prediction[mask] == y[mask]).mean()
            )
    result = {
        "n": len(y),
        "accuracy": float((prediction == y).mean()),
        "macro_f1": float(f1[support > 0].mean()),
        "class_support": support.tolist(),
        "false_safe_count": false_safe,
        "violation_count": int(matrix[1].sum()),
        "false_safe_rate": float(false_safe / matrix[1].sum())
        if matrix[1].sum()
        else None,
        "false_safe_prediction_denominator": int(matrix[:, 0].sum()),
        "false_safe_prediction_rate": float(false_safe / matrix[:, 0].sum())
        if matrix[:, 0].sum()
        else None,
        "selection_cost": (errors + 4 * false_safe) / len(y),
        "nll": float(-logp[np.arange(len(y)), y].mean()),
        "brier_sum_over_classes": float(
            ((probabilities - np.eye(3)[y]) ** 2).sum(axis=1).mean()
        ),
        "ece_10_bins": float(ece),
        "confusion": matrix.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "min_class_recall": float(recall[support > 0].min()),
        "degeneracy_count": int((recall[support > 0] < 0.05).sum()),
        "predictions": prediction.tolist(),
    }
    if from_probabilities:
        result["probability_floor"] = 0.00005
    return result


def diagnostics(
    rows,
    logits,
    temperature=1.0,
    from_probabilities=False,
    bootstrap_samples=1000,
    seed=0,
):
    """Compute point diagnostics and explicitly budgeted uncertainty estimates."""
    targets = [LABELS.index(row["expected"]["security"]) for row in rows]
    topics, families, pairs, triplets, rewrites = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(dict),
        defaultdict(dict),
        defaultdict(list),
    )
    predictions = metrics(logits, targets, temperature, from_probabilities)[
        "predictions"
    ]
    for index, row in enumerate(rows):
        topics[row["topic"]].append(index)
        families[row["family_id"]].append(index)
        pairs[row["pair_id"]][targets[index]] = (
            predictions[index] == targets[index]
        )
        triplets[row["pair_id"]][targets[index]] = (
            predictions[index] == targets[index]
        )
        rewrite = row.get("rewrite_group_id", row["pair_id"])
        rewrites[(rewrite, targets[index])].append(int(predictions[index]))
    paired = [
        bool(group[0] and group[1])
        for group in pairs.values()
        if 0 in group and 1 in group
    ]
    tripled = [
        all(group.values())
        for group in triplets.values()
        if set(group) == {0, 1, 2}
    ]
    stable = [
        len(set(values)) == 1 for values in rewrites.values() if len(values) > 1
    ]
    rewrite_correct = [
        all(value == label for value in values)
        for (group, label), values in rewrites.items()
        if group and len(values) > 1
    ]
    family_metrics = {
        family: metrics(
            [logits[i] for i in indices],
            [targets[i] for i in indices],
            temperature,
            from_probabilities,
        )
        for family, indices in sorted(families.items())
    }
    false_safe_rates = [
        result["false_safe_rate"]
        for result in family_metrics.values()
        if result["false_safe_rate"] is not None
    ]
    return {
        "families": len(families),
        "per_topic": {
            topic: metrics(
                [logits[i] for i in indices],
                [targets[i] for i in indices],
                temperature,
                from_probabilities,
            )
            for topic, indices in sorted(topics.items())
        },
        "per_family": family_metrics,
        "family_selection_cost": float(
            np.mean([r["selection_cost"] for r in family_metrics.values()])
        ),
        "family_macro_f1": float(
            np.mean([r["macro_f1"] for r in family_metrics.values()])
        ),
        "worst_family_selection_cost": float(
            max(r["selection_cost"] for r in family_metrics.values())
        ),
        "worst_family_false_safe_rate": float(max(false_safe_rates))
        if false_safe_rates
        else None,
        "both_pair_members_correct": float(np.mean(paired)) if paired else None,
        "complete_pairs": len(paired),
        "complete_triplets_correct": float(np.mean(tripled))
        if tripled
        else None,
        "complete_triplets": len(tripled),
        "equivalent_rewrite_stability": float(np.mean(stable))
        if stable
        else None,
        "equivalent_rewrite_accuracy": float(np.mean(rewrite_correct))
        if rewrite_correct
        else None,
        "rewrite_groups": len(stable),
        "reporting": evaluation_report(
            rows,
            logits,
            hierarchical_weights(rows),
            SOURCE_MASS,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            temperature=temperature,
            from_probabilities=from_probabilities,
        ),
    }


def source_breakdown(selected_rows, selected_logits):
    """Per-source counts; triplets need complete in-source groups.

    A triplet split across sources (a data defect, not a model
    result) must not crash scoring or credit either source: only
    groups with three same-source rows and one label each count.
    """
    members: dict[str, list[int]] = {}
    for position, row in enumerate(selected_rows):
        members.setdefault(row["source"]["name"], []).append(position)
    breakdown = {}
    for name in sorted(members):
        indices = members[name]
        sub_rows = [selected_rows[i] for i in indices]
        sub_logits = [selected_logits[i] for i in indices]
        targets = [
            LABELS.index(row["expected"]["security"]) for row in sub_rows
        ]
        subgroup = metrics(sub_logits, targets)
        matrix = subgroup["confusion"]
        class_correct = [matrix[i][i] for i in range(3)]
        support = [sum(row) for row in matrix]
        predictions = subgroup["predictions"]
        groups: dict[str, list[int]] = {}
        for rank, row in enumerate(sub_rows):
            groups.setdefault(row["pair_id"], []).append(rank)
        triplets = triplets_correct = 0
        for ranks in groups.values():
            expected = [sub_rows[i]["expected"]["security"] for i in ranks]
            if len(ranks) != 3 or set(expected) != set(LABELS):
                continue
            triplets += 1
            if all(
                predictions[i] == LABELS.index(expected[k])
                for k, i in enumerate(ranks)
            ):
                triplets_correct += 1
        if all(support):
            recall = min(a / b for a, b in zip(class_correct, support))
        else:
            # A source missing a class entirely reports zero recall;
            # positive floors on the absent class still refuse.
            recall = 0.0
        breakdown[name] = {
            "n": len(sub_rows),
            "correct": sum(class_correct),
            "class_correct": class_correct,
            "class_support": support,
            "false_safe": subgroup["false_safe_count"],
            "triplets": triplets,
            "triplets_correct": triplets_correct,
            "min_recall": recall,
        }
    return breakdown


def summarize_development_view(
    selected_rows,
    selected_logits,
    bootstrap_samples=0,
    seed=0,
    score_triplets=False,
    score_sources=False,
):
    """Summarize one development view exactly as training evaluation does.

    Combines confusion metrics, diagnostics, and the trivial-baseline
    cost. With score_triplets, whole-triplet correctness from original
    pair_id membership joins the summary for triplet-gated selection.
    With score_sources, a per-source breakdown joins it for
    source-guarded selection.
    """
    targets = [
        LABELS.index(row["expected"]["security"]) for row in selected_rows
    ]
    summary = {
        **metrics(selected_logits, targets),
        **diagnostics(
            selected_rows,
            selected_logits,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "baseline_best_cost": trivial_baselines(selected_rows)[
            "best_selection_cost"
        ],
    }
    if score_triplets:
        summary.update(
            triplet_counts(selected_rows, summary["predictions"], list(LABELS))
        )
    if score_sources:
        summary["by_source"] = source_breakdown(selected_rows, selected_logits)
    return summary


def attach_reporting(result, rows, logits, args, rank):
    """Additional diagnostics do not alter the predeclared selection metrics."""
    if rank == 0:
        result["reporting"] = evaluation_report(
            rows,
            logits,
            hierarchical_weights(rows),
            SOURCE_MASS,
            bootstrap_samples=args.report_bootstrap,
            seed=args.seed,
        )
    return result


def selection_key(result):
    baseline = result.get("baseline_best_cost")
    baseline_gate = (
        int(result["selection_cost"] >= baseline) if baseline is not None else 0
    )
    paired = result.get("both_pair_members_correct")
    paired = paired if paired is not None else 0.0
    competence_gate = int(
        result.get("min_class_recall", 0) < ACCEPTANCE["min_class_recall"]
        or paired < ACCEPTANCE["min_pair_accuracy"]
    )
    # False-safe errors already cost five times an ordinary error. Worst-family
    # rates remain a finite penalty: three correlated examples cannot veto every
    # improvement in pair discrimination. Deployment acceptance is separate.
    score = (
        result.get("family_selection_cost", result["selection_cost"])
        + 0.5 * (1.0 - paired)
        + 0.5 * (1.0 - result.get("family_macro_f1", result["macro_f1"]))
        + 0.1 * (result.get("worst_family_false_safe_rate") or 0.0)
    )
    return (
        result.get("degeneracy_count", 0),
        competence_gate,
        baseline_gate,
        score,
        result["nll"],
    )


def retention_check_for(config, result, reference):
    """Retention budget check for legacy joint selection.

    Objective selection (``checkpoint_objective``) protects every view
    explicitly and has no retention view; it records no check.
    """
    if "checkpoint_objective" in config:
        return None
    accuracy = result["development_views"][config["retention_view"]]["accuracy"]
    limit = config["max_retention_accuracy_drop"]
    return {
        "passed": bool(accuracy >= reference - limit),
        "reference_accuracy": reference,
        "accuracy": accuracy,
        "max_accuracy_drop": limit,
    }


def acceptance_status(result):
    retention = result.get("retention_check")
    if result.get("selection_view"):
        result = result["development_views"][result["selection_view"]]
    reasons = []
    if retention and not retention["passed"]:
        reasons.append("retention_accuracy_below_budget")
    if (
        result.get("baseline_best_cost") is not None
        and result["selection_cost"] >= result["baseline_best_cost"]
    ):
        reasons.append("does_not_beat_best_trivial_baseline")
    if result.get("degeneracy_count", 0):
        reasons.append("one_or_more_classes_have_no_recall")
    if (
        result.get("worst_family_false_safe_rate") is None
        or result["worst_family_false_safe_rate"]
        > ACCEPTANCE["max_worst_family_false_safe_rate"]
    ):
        reasons.append("worst_family_false_safe_above_budget")
    if result.get("min_class_recall", 0) < ACCEPTANCE["min_class_recall"]:
        reasons.append("class_recall_below_minimum")
    if (
        result.get("both_pair_members_correct") is None
        or result["both_pair_members_correct"] < ACCEPTANCE["min_pair_accuracy"]
    ):
        reasons.append("pair_accuracy_below_minimum")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "thresholds": dict(ACCEPTANCE),
    }


def plateau_action(
    progress, epoch, min_updates, patience, exposure_complete=None
):
    """Plateau intervention gated on optimizer updates, not epoch count.

    Update-budget campaigns can finish before the epoch warmup ends, so an
    unconditional epoch gate can suppress every plateau intervention.
    min_updates is the update-scale warmup; the epoch gate applies only when
    no update budget is configured (min_updates == 0). When exposure_complete
    is provided (never None), a plateau intervention additionally requires a
    delivered full first pass over the training rows.
    """
    if progress["step"] < min_updates or progress["stale"] < patience:
        return None
    if min_updates == 0 and epoch < 3:
        return None
    if exposure_complete is False:
        return None
    return "stop" if progress["reduced"] else "reduce_lr"


def learning_rate_factor(
    step, warmup, max_updates, schedule, hold_updates=0, endpoint=0.1
):
    factor = min(1.0, (step + 1) / max(1, warmup))
    if schedule == "cosine":
        if max_updates <= warmup:
            raise ValueError(
                "Cosine scheduling requires max_updates greater than warmup"
            )
        phase = max(0.0, (step - warmup) / (max_updates - warmup))
        factor *= 0.1 + 0.9 * (1 + math.cos(math.pi * min(phase, 1.0))) / 2
    elif schedule == "hold-decay":
        if max_updates <= warmup + hold_updates:
            raise ValueError(
                "Hold-decay scheduling requires max_updates greater "
                "than warmup plus hold"
            )
        decay_start = warmup + hold_updates
        phase = max(0.0, (step - decay_start) / (max_updates - decay_start))
        cosine = (1 + math.cos(math.pi * min(phase, 1.0))) / 2
        factor *= endpoint + (1 - endpoint) * cosine
    elif schedule != "constant":
        raise ValueError("Unknown learning-rate schedule")
    return factor


def surface_undefined(row):
    state = row["state"]
    names = (
        "condition_expr",
        "guard_expr",
        "control_expr",
        "access_expr",
        "request_expr",
        "permission_expr",
        "write_value",
        "reset_request",
        "read_policy",
        "access_policy",
        "output_mask",
        "unmapped_value",
        "reinit_request",
        "read_value",
    )
    for name in names:
        pattern = r"\b%s(?:_[12])?\b" % re.escape(name)
        occurrences = len(re.findall(pattern, state))
        declaration = re.search(
            r"\bwire\s+(?:\[[^]]+\]\s*)?%s(?:_[12])?\s*;" % re.escape(name),
            state,
        )
        if occurrences >= 2 if declaration else occurrences >= 1:
            return True
    return False


def trivial_baselines(rows):
    targets = [LABELS.index(row["expected"]["security"]) for row in rows]
    result = {}
    for label, name in enumerate(LABELS):
        logits = np.eye(3)[label] * np.ones((len(rows), 1))
        result["always_" + name] = metrics(logits, targets)
    prediction = np.array([2 if surface_undefined(row) else 1 for row in rows])
    logits = np.eye(3)[prediction]
    result["undefined_else_violates"] = metrics(logits, targets)
    best = min(value["selection_cost"] for value in result.values())
    return {"metrics": result, "best_selection_cost": best}


def batch_for(tok, items, indices, args, epoch=None, permutation=None):
    encoded = []
    for index in indices:
        order = permutation
        if epoch is not None:
            order = list(range(3))
            random.Random(args.seed + epoch * 1000003 + index * 17).shuffle(
                order
            )
        encoded.append(
            encode_item(
                tok, items[index], args.max_len, args.head_max_len, order
            )
        )
    return collate_items([encoded], tok.pad_token_id)


def forward(model, batch, device, bf16=True):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        logits, _ = model(
            *[
                batch[key].to(device)
                for key in (
                    "input_ids",
                    "attention_mask",
                    "marker_pos",
                    "marker_mask",
                    "qtype",
                )
            ]
        )
    return logits.float()


def assert_sdk_probabilities(actual, logits, temperature):
    z = np.asarray(logits, dtype=np.float32) / temperature
    if z.shape != (3,) or not np.isfinite(z).all():
        raise ValueError("invalid SDK parity logits")
    probabilities = np.exp(z - z.max())
    probabilities /= probabilities.sum()
    expected = [round(float(value), 4) for value in probabilities]
    np.testing.assert_array_equal(actual, expected)


def evaluate(
    model,
    tok,
    items,
    args,
    device,
    rank,
    world,
    permutation=None,
    bf16=True,
    baseline_cost=None,
):
    model.eval()
    records = []
    indices = list(range(rank, len(items), world))
    evaluation_batch = getattr(args, "eval_batch_size", 0) or args.micro_batch
    with torch.inference_mode():
        for start in range(0, len(indices), evaluation_batch):
            chunk = indices[start : start + evaluation_batch]
            batch = batch_for(tok, items, chunk, args, permutation=permutation)
            logits = forward(model, batch, device, bf16).cpu().tolist()
            if permutation is not None:
                logits = [
                    [row[permutation.index(k)] for k in range(3)]
                    for row in logits
                ]
            records.extend((i, row) for i, row in zip(chunk, logits))
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, records)
    else:
        gathered[0] = records
    records = sorted(itertools.chain.from_iterable(gathered))
    if [i for i, _ in records] != list(range(len(items))):
        raise RuntimeError("evaluation missing or duplicated rows")
    logits = [row for _, row in records]
    targets = [int(np.argmax(item["target"])) for item in items]
    model.train()
    result = metrics(logits, targets)
    families = sorted({item["family_id"] for item in items})
    family_metrics = []
    for family in families:
        selected = [
            i for i, item in enumerate(items) if item["family_id"] == family
        ]
        family_metrics.append(
            metrics(
                [logits[i] for i in selected], [targets[i] for i in selected]
            )
        )
    result["family_selection_cost"] = float(
        np.mean([r["selection_cost"] for r in family_metrics])
    )
    result["family_macro_f1"] = float(
        np.mean([r["macro_f1"] for r in family_metrics])
    )
    result["worst_family_selection_cost"] = float(
        max(r["selection_cost"] for r in family_metrics)
    )
    rates = [
        r["false_safe_rate"]
        for r in family_metrics
        if r["false_safe_rate"] is not None
    ]
    result["worst_family_false_safe_rate"] = (
        float(max(rates)) if rates else None
    )
    result["family_count"] = len(families)
    predictions = np.argmax(np.asarray(logits), axis=1)
    pairs, triplets = defaultdict(dict), defaultdict(dict)
    for index, item in enumerate(items):
        correct = bool(predictions[index] == targets[index])
        pairs[item["pair_id"]][targets[index]] = correct
        triplets[item["pair_id"]][targets[index]] = correct
    paired = [
        bool(group[0] and group[1])
        for group in pairs.values()
        if 0 in group and 1 in group
    ]
    tripled = [
        all(group.values())
        for group in triplets.values()
        if set(group) == {0, 1, 2}
    ]
    result["both_pair_members_correct"] = (
        float(np.mean(paired)) if paired else None
    )
    result["complete_triplets_correct"] = (
        float(np.mean(tripled)) if tripled else None
    )
    result["baseline_best_cost"] = baseline_cost
    return result, logits, targets


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def save_resume(model, optimizer, progress, signature, output, rank, world):
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng_state())
    else:
        states[0] = rng_state()
    if rank == 0:
        output = Path(output)
        temporary = output / "latest.pt.tmp"
        with temporary.open("wb") as handle:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "progress": progress,
                    "rng": states,
                    "signature": signature,
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        latest = output / "latest.pt"
        if latest.exists():
            previous_temporary = output / "previous.pt.tmp"
            previous_temporary.unlink(missing_ok=True)
            os.link(latest, previous_temporary)
            os.replace(previous_temporary, output / "previous.pt")
        os.replace(temporary, latest)
        fsync_directory(output)
    if world > 1:
        dist.barrier()


def save_weights(model, path):
    path = Path(path)
    temporary = Path(str(path) + ".tmp")
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in model.state_dict().items()
        },
        str(temporary),
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def generation_name(epoch, step, lineage):
    return "epoch-%06d-step-%08d-lineage-%06d" % (epoch, step, lineage)


def replica_digest(model, trainable):
    """SHA-256 over replica parameters (cross-rank comparable)."""
    checksum = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad != trainable:
            continue
        checksum.update(name.encode())
        checksum.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        checksum.update(str(parameter.dtype).encode())
    if not trainable:
        for name, buffer in sorted(model.named_buffers()):
            checksum.update(name.encode())
            checksum.update(
                buffer.detach().cpu().contiguous().numpy().tobytes()
            )
    return checksum.hexdigest()


def assert_replicas_consistent(model, rank, world, where):
    """Fail immediately when DDP replicas diverge."""
    if world < 2:
        return replica_digest(model, True)
    local = replica_digest(model, True)
    gathered: list = [None] * world
    dist.all_gather_object(gathered, local)
    if any(value != gathered[0] for value in gathered):
        raise RuntimeError(
            "replica parameters differ at %s: %s" % (where, gathered)
        )
    return gathered[0]


def verify_frozen_steady(model, expected, where):
    """Frozen parameters must never move after initialization."""
    local = replica_digest(model, False)
    if local != expected:
        raise RuntimeError("frozen parameters moved at %s" % where)
    return local


def save_eval_snapshot(model, output, epoch, step, key, result, lineage):
    """Retain every evaluation checkpoint's tensors (rank 0)."""
    folder = output / "eval_snapshots" / ("step-%08d" % step)
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / "model.safetensors"
    if not weights.exists():
        save_weights(model, weights)
    manifest = {
        "epoch": epoch,
        "step": step,
        "generation": generation_name(epoch, step, lineage),
        "key": list(key),
        "metrics": result,
        "weights_sha256": sha256(weights),
    }
    write_json(folder / "snapshot.json", manifest)
    return manifest


def save_best(
    model, output, epoch, step, key, result, logits, targets, lineage
):
    output = Path(output)
    generation = generation_name(epoch, step, lineage)
    folder = output / "best_generations" / generation
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / "model.safetensors"
    development = folder / "development.json"
    if development.exists():
        existing = json.loads(development.read_text())
        if existing["key"] != list(key) or not weights.exists():
            raise ValueError(
                "incomplete or conflicting best generation: " + generation
            )
        metadata = existing
    else:
        save_weights(model, weights)
        metadata = {
            "epoch": epoch,
            "step": step,
            "key": list(key),
            "metrics": result,
            "logits": logits,
            "targets": targets,
            "weights_sha256": sha256(weights),
        }
    write_json(folder / "development.json", metadata)
    pointer = {
        "generation": generation,
        "weights_sha256": metadata["weights_sha256"],
        "development_sha256": sha256(folder / "development.json"),
        "epoch": epoch,
        "step": step,
        "key": list(key),
    }
    write_json(output / "best.json", pointer)
    return pointer


def load_best(output):
    output = Path(output)
    pointer = json.loads((output / "best.json").read_text())
    folder = output / "best_generations" / pointer["generation"]
    if sha256(folder / "model.safetensors") != pointer["weights_sha256"]:
        raise ValueError("selected weight hash mismatch")
    if sha256(folder / "development.json") != pointer["development_sha256"]:
        raise ValueError("selected development hash mismatch")
    metadata = json.loads((folder / "development.json").read_text())
    if (
        metadata["weights_sha256"] != pointer["weights_sha256"]
        or metadata["key"] != pointer["key"]
    ):
        raise ValueError("selected generation metadata mismatch")
    return pointer, metadata, folder / "model.safetensors"


def restore_best_lineage(output, progress, rank, world):
    output = Path(output)
    generation = progress.get("best_generation")
    if not generation:
        raise ValueError("resume progress has no selected generation")
    folder = output / "best_generations" / generation
    metadata = json.loads((folder / "development.json").read_text())
    if (
        metadata["key"] != progress["best_key"]
        or metadata["epoch"] != progress["best_epoch"]
    ):
        raise ValueError("resume selected-generation lineage mismatch")
    if metadata["step"] > progress["step"]:
        raise ValueError("selected generation is newer than resume progress")
    weights = folder / "model.safetensors"
    pointer = {
        "generation": generation,
        "weights_sha256": sha256(weights),
        "development_sha256": sha256(folder / "development.json"),
        "epoch": metadata["epoch"],
        "step": metadata["step"],
        "key": metadata["key"],
    }
    if metadata["weights_sha256"] != pointer["weights_sha256"]:
        raise ValueError("resume selected weight hash mismatch")
    if rank == 0:
        write_json(output / "best.json", pointer)
    if world > 1:
        dist.barrier()
    return load_best(output)


def export_selected(model, tok, cfg, output, signature, progress, run_id):
    output = Path(output)
    pointer, best, weights_path = load_best(output)
    export = output / "model"
    expected = pointer["weights_sha256"]
    if export.exists():
        if sha256(export / "model.safetensors") != expected:
            raise ValueError(
                "existing export differs from committed best generation"
            )
    else:
        temporary = output / "model.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        save_weights(model, temporary / "model.safetensors")
        model.encoder.config.save_pretrained(temporary / "encoder")
        tok.save_pretrained(temporary / "tokenizer")
        cfg = dict(cfg)
        cfg["training"] = {
            "method": "supervised_security_bf16_ddp",
            "run_id": run_id,
            "selected_epoch": best["epoch"],
            "selected_step": best["step"],
            "completed_step": progress["step"],
            "world_size": signature["world"],
            "action_head_trained": False,
            "data_hashes": signature["data"],
            "code_manifest": signature["code_manifest"],
        }
        write_json(temporary / "rl_agent_config.json", cfg)
        if sha256(temporary / "model.safetensors") != expected:
            raise ValueError("export does not match selected weights")
        write_export_record(
            temporary,
            signature.get("question_contract", LEGACY_CONTRACT),
        )
        os.replace(temporary, export)
        fsync_directory(output)
    expected_contract = signature.get("question_contract", LEGACY_CONTRACT)
    recorded = read_export_record(export)
    if recorded["version"] != expected_contract:
        raise ValueError(
            "export contract %s differs from run contract %s"
            % (recorded["version"], expected_contract)
        )
    receipt = {
        "run_id": run_id,
        "generation": pointer["generation"],
        "weights_sha256": expected,
        "export_manifest": checkpoint_manifest(export),
        "question_contract": recorded["version"],
    }
    write_json(output / "export.json", receipt)
    return receipt


def load_initial_development(output, progress):
    """Initial (epoch-zero) development metrics, including after a resume.

    Resume increments progress['lineage'], so the original generation must be
    found through the reference recorded at first training (or, for older
    resumable state, by scanning epoch-zero generations), never via the
    current lineage.
    """
    progress = progress or {}
    metrics = progress.get("initial_metrics")
    if isinstance(metrics, dict):
        return metrics
    root = Path(output) / "best_generations"
    candidates = []
    if progress.get("initial_generation"):
        candidates.append(
            root / progress["initial_generation"] / "development.json"
        )
    if root.is_dir():
        candidates.extend(
            sorted(
                root.glob(
                    "epoch-000000-step-00000000-lineage-*/development.json"
                )
            )
        )
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())["metrics"]
    return None


def finish_stage(
    model, tok, cfg, output, signature, progress, run_id, rank, world, log
):
    error = [None]
    if rank == 0:
        try:
            pointer, best, weights_path = load_best(output)
            receipt = export_selected(
                model, tok, cfg, output, signature, progress, run_id
            )
            initial = load_initial_development(output, progress)
            report = {
                "schema": REPORT_SCHEMA,
                "run_id": run_id,
                "development": best["metrics"],
                "initial_development": initial,
                "selection_key": best["key"],
                "development_acceptance": acceptance_status(best["metrics"]),
                "progress": progress,
                "stop_reason": progress["stop_reason"],
                "selected_checkpoint": {
                    "generation": pointer["generation"],
                    "epoch": best["epoch"],
                    "step": best["step"],
                    "weights_sha256": pointer["weights_sha256"],
                },
                "export": receipt,
                "final_evaluation": False,
                "test_evaluated": False,
                "calibration_fitted": False,
                "signature": signature,
                "reporting_contract": signature.get("reporting_contract"),
            }
            if signature.get("evaluation_role") == "train_fitting":
                report["evaluation_role"] = "train_fitting"
                report["train"] = report.pop("development")
                report["initial_train"] = report.pop("initial_development")
                report.pop("development_acceptance")
                report["final_train"] = progress.get(
                    "last_metrics", report["train"]
                )
                report["train_accuracy"] = report["train"]["accuracy"]
                report["initial_train_accuracy"] = initial["accuracy"]
                report["fitting_subset"] = signature["fitting_subset"]
            write_json(Path(output) / "report.json", report)
            log(
                {
                    "event": "stage_complete",
                    "run_id": run_id,
                    "progress": progress,
                    "test_evaluated": False,
                }
            )
        except BaseException as exc:
            error[0] = "%s: %s" % (type(exc).__name__, exc)
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def finalize_training(
    model,
    optimizer,
    tok,
    cfg,
    output,
    signature,
    progress,
    run_id,
    rank,
    world,
    log,
    observe,
    device,
):
    """Evaluate terminal weights, persist selection, then export the selected model."""
    if progress.get("last_eval_step") != progress["step"]:
        observe(progress["epoch"], progress["offset"])
    save_resume(model, optimizer, progress, signature, output, rank, world)
    _, _, weights_path = load_best(output)
    model.load_state_dict(
        load_file(str(weights_path), device=str(device)), strict=True
    )
    finish_stage(
        model, tok, cfg, output, signature, progress, run_id, rank, world, log
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", help="Path to the local base-model checkpoint"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Directory containing the prepared dataset splits",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for checkpoints, reports, and logs",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Identifier written to this run's reports",
    )
    parser.add_argument(
        "--max-len", type=int, default=1024, help="Maximum total input tokens"
    )
    parser.add_argument(
        "--head-max-len",
        type=int,
        default=192,
        help="Maximum tokens in the question and option section",
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=4,
        help="Decision sequences per GPU per forward pass",
    )
    parser.add_argument(
        "--accumulation",
        type=int,
        default=4,
        help="Microbatches per optimizer update",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Evaluation sequences per GPU; zero uses the training microbatch",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=6.5,
        help="Wall-clock training budget in hours",
    )
    parser.add_argument(
        "--encoder-lr", type=float, default=1e-5, help="Encoder learning rate"
    )
    parser.add_argument(
        "--adapter-lr",
        type=float,
        default=None,
        help="Residual-adapter learning rate (default: --head-lr)",
    )
    parser.add_argument(
        "--adapter-scale",
        type=float,
        default=None,
        help="Residual branch gain for newly created adapters (default: 8.0)",
    )
    parser.add_argument(
        "--limit-train-rows",
        type=int,
        default=0,
        help="Select at least N TRAIN rows, retaining complete designs (TRAIN-only check)",
    )
    parser.add_argument(
        "--no-development-in-fitting",
        action="store_true",
        help="Skip development evaluation in fitting mode (default: skip dev when limit-train-rows is set)",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the encoder for head/adaptation-only fitting",
    )
    parser.add_argument(
        "--encoder-trainable-blocks",
        type=int,
        default=None,
        help="Unfreeze only the top-K encoder blocks (contradicts --freeze-encoder)",
    )
    parser.add_argument(
        "--disable-adapters",
        action="store_true",
        help="Disable residual adapters (true head-only control)",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-4,
        help="Decision-layer and adapter learning rate",
    )
    parser.add_argument(
        "--warmup", type=int, default=30, help="Warmup optimizer updates"
    )
    parser.add_argument("--seed", type=int, default=41, help="Random seed")
    parser.add_argument(
        "--checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable activation checkpointing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the saved training state",
    )
    parser.add_argument(
        "--finalize-development-only",
        action="store_true",
        help="Resume solely to export the selected development checkpoint; no optimizer updates",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Stop after a short training check (0: disabled)",
    )
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="Allow initialization from an adapted checkpoint",
    )
    parser.add_argument(
        "--min-updates",
        type=int,
        default=0,
        help="Minimum updates before validation-based early stopping",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Development evaluations without improvement before early stopping",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate settings and exit before loading data",
    )
    parser.add_argument(
        "--plateau-requires-full-exposure",
        action="store_true",
        help="Refuse plateau intervention until delivered draws cover "
        "every training row with no undrawn triplet member "
        "(grouped sampling only)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=float,
        default=900,
        help="Seconds between resumable checkpoint saves",
    )
    parser.add_argument(
        "--sampling",
        choices=("weighted", "grouped"),
        default="weighted",
        help="Weighted rows or complete groups stratified by source and design",
    )
    parser.add_argument(
        "--pools",
        help=(
            "Sampling-pool manifest over the training rows; with grouped "
            "sampling, batch quotas follow pool weights instead of source "
            "mass. Reporting stays source-based."
        ),
    )
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=0,
        help="Evaluate every N updates (0: use epoch boundaries)",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=0,
        help="Maximum optimizer updates (0: use the epoch and time limits)",
    )
    parser.add_argument(
        "--report-bootstrap",
        type=int,
        default=1000,
        help="Whole-split-group bootstrap draws for reporting (0 disables intervals)",
    )
    parser.add_argument(
        "--reporting-contract",
        help="Local reporting contract JSON, hashed into the run signature",
    )
    parser.add_argument(
        "--new-study-manifest",
        help="Explicit fresh-study registration for an evaluated ancestor",
    )
    parser.add_argument(
        "--development-views",
        help="Local transfer/retention development selection contract",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine", "hold-decay"),
        default="constant",
    )
    parser.add_argument(
        "--hold-updates",
        type=int,
        default=0,
        help="Peak-LR updates after warmup before decay (hold-decay only)",
    )
    parser.add_argument(
        "--lr-endpoint",
        type=float,
        default=0.1,
        help="Terminal LR fraction of peak (hold-decay only)",
    )
    parser.add_argument(
        "--question-contract",
        choices=(LEGACY_CONTRACT, NEW_CONTRACT),
        default=LEGACY_CONTRACT,
        help="Pinned model-visible question contract for every row",
    )
    parser.add_argument(
        "--review-ledger",
        help="JSONL review ledger; template coverage is enforced when set",
    )
    parser.add_argument(
        "--review-packets",
        help="JSON review-packet mapping; ledger assertion hashes are "
        "recomputed from packet bytes when set (requires --review-ledger)",
    )
    parser.add_argument(
        "--no-diagnostic-weights",
        action="store_true",
        help="Keep diagnostic metrics without saving alternative full model copies",
    )
    parser.add_argument(
        "--final-evaluation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Final evaluation requires --evaluate-only",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Calibrate and evaluate a completed training run",
    )
    args = parser.parse_args()
    require_coherent_training(Path(__file__).resolve())
    if args.evaluate_only:
        from security_campaign import finalize_only

        finalize_only(args)
        return
    if not args.model:
        parser.error("--model is required for training")
    if args.finalize_development_only and not args.resume:
        parser.error("--finalize-development-only requires --resume")
    if args.final_evaluation:
        parser.error(
            "outer evaluation is available only through --evaluate-only"
        )
    if (
        min(
            args.max_len,
            args.head_max_len,
            args.micro_batch,
            args.accumulation,
            args.epochs,
        )
        <= 0
    ):
        parser.error("length, batch and epoch values must be positive")
    if (
        args.warmup < 0
        or args.min_updates < 0
        or args.patience < 1
        or args.hours <= 0
        or args.checkpoint_interval <= 0
    ):
        parser.error("invalid budget or scheduler configuration")
    if args.plateau_requires_full_exposure and (
        args.sampling != "grouped" or args.smoke_steps
    ):
        parser.error(
            "--plateau-requires-full-exposure needs grouped sampling "
            "without smoke steps"
        )
    if (
        args.eval_every_updates < 0
        or args.max_updates < 0
        or args.eval_batch_size < 0
    ):
        parser.error("update intervals must be nonnegative")
    if args.lr_schedule == "cosine" and args.max_updates <= args.warmup:
        parser.error(
            "Cosine scheduling requires max_updates greater than warmup"
        )
    if args.hold_updates < 0 or not 0.0 < args.lr_endpoint <= 1.0:
        parser.error("invalid hold-decay configuration")
    if args.lr_schedule == "hold-decay" and (
        args.max_updates <= args.warmup + args.hold_updates
    ):
        parser.error(
            "Hold-decay scheduling requires max_updates greater "
            "than warmup plus hold"
        )
    if args.limit_train_rows < 0 or (
        args.no_development_in_fitting and not args.limit_train_rows
    ):
        parser.error("TRAIN-only fitting requires a positive row limit")
    if any(
        not math.isfinite(v) or v < 0
        for v in (
            args.encoder_lr,
            args.head_lr,
            args.adapter_lr if args.adapter_lr is not None else args.head_lr,
        )
    ):
        parser.error("learning rates must be finite and nonnegative")
    if args.adapter_scale is not None and (
        not math.isfinite(args.adapter_scale) or args.adapter_scale <= 0
    ):
        parser.error("adapter scale must be finite and positive")
    if args.report_bootstrap < 0:
        parser.error("bootstrap draw count must be nonnegative")
    if args.development_views and args.limit_train_rows:
        parser.error(
            "Development views are not available in TRAIN-only fitting"
        )
    if args.dry_run:
        print("settings valid")
        return
    reporting_contract = None
    if args.reporting_contract:
        contract = json.loads(Path(args.reporting_contract).read_text())
        if (
            contract.get("uncertainty", {}).get("cluster_key")
            != "split_group_id"
        ):
            parser.error(
                "reporting contract requires an unsupported clustering unit"
            )
        reporting_contract = {
            "sha256": sha256(args.reporting_contract),
            "contract": contract,
        }
    rank, world, local = (
        int(os.getenv(key, value))
        for key, value in (
            ("RANK", "0"),
            ("WORLD_SIZE", "1"),
            ("LOCAL_RANK", "0"),
        )
    )
    execution_identity = verify_execution_manifest(args.run_id)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable")
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    existing_report = output / "report.json"
    if existing_report.exists():
        previous = json.loads(existing_report.read_text())
        if previous.get("test_evaluated"):
            raise ValueError(
                "training is forbidden after outer-test evaluation"
            )
        if not args.resume:
            raise ValueError(
                "Completed output exists; use a new run ID or explicit --resume"
            )
    model_path = Path(args.model)
    new_study = None
    if args.new_study_manifest:
        if not args.continue_training:
            raise ValueError(
                "A new study requires explicit --continue-training"
            )
        new_study = verify_new_study(
            args.new_study_manifest, model_path, args.data, output, args.run_id
        )
    if (model_path.parent / "test_touch.json").exists() and new_study is None:
        raise ValueError(
            "training is forbidden from a test-touched checkpoint lineage"
        )
    parent_report = model_path.parent / "report.json"
    if (
        parent_report.exists()
        and json.loads(parent_report.read_text()).get("test_evaluated")
        and new_study is None
    ):
        raise ValueError("training is forbidden after outer-test evaluation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    fitting = bool(args.limit_train_rows)
    rows = load_splits(
        args.data,
        final=False,
        train_only=fitting,
        question_contract=args.question_contract,
    )
    if fitting:
        rows["train"] = fitting_subset(
            rows["train"], args.limit_train_rows, args.seed
        )
        validate_question_rows(rows["train"], args.question_contract)
    if args.review_packets and not args.review_ledger:
        parser.error("--review-packets requires --review-ledger")
    if args.review_ledger:
        try:
            from siliconshield.training.laya.review_ledger import (
                require_packet_binding,
                require_template_coverage,
                validate_ledger,
            )
        except ImportError:  # Bare-script launch from training dir.
            from review_ledger import (  # type: ignore[no-redef]
                require_packet_binding,
                require_template_coverage,
                validate_ledger,
            )
        ledger = validate_ledger(Path(args.review_ledger))
        require_template_coverage(rows["train"], ledger)
        if args.review_packets:
            packets = json.loads(Path(args.review_packets).read_text())
            require_packet_binding(ledger, packets)
    pool_plan = None
    pools_record = None
    if args.pools:
        if args.sampling != "grouped":
            parser.error("--pools requires grouped sampling")
        pool_plan = pool_assignment_for_rows(args.pools, rows["train"])
        pools_record = {
            "sha256": sha256(args.pools),
            "mass": dict(pool_plan[1]),
        }
    evaluation_split = "train" if fitting else "dev"
    model, tok, cfg = load_readonly_checkpoint(args.model, load_checkpoint)
    if cfg.get("outer_test_evaluated") and new_study is None:
        raise ValueError(
            "training is forbidden from an outer-test-evaluated export"
        )
    adapter_cfg = configure_trainable_model(
        model,
        cfg,
        args.freeze_encoder,
        args.disable_adapters,
        args.adapter_scale,
        args.continue_training,
        args.encoder_trainable_blocks,
    )
    cfg = dict(
        cfg,
        residual_adapters=adapter_cfg,
        max_len=args.max_len,
        head_max_len=args.head_max_len,
        option_layout="sequential",
        amp_dtype="bf16",
        temperature=[1.0, 1.0, 1.0],
        temperature_by_options={},
    )
    if new_study:
        cfg.update(study_provenance=new_study, outer_test_evaluated=False)
    development_views = (
        json.loads(Path(args.development_views).read_text())
        if args.development_views
        else None
    )
    # The settings block rejects this combination first. This guards
    # direct main() callers that bypass argument parsing.
    if development_views and fitting:
        raise ValueError(
            "Development views are not available in TRAIN-only fitting"
        )
    selection_ranking = (
        ranking_version(SELECTION_REFERENCE["policy"])
        if SELECTION_REFERENCE is not None
        else None
    )
    checkpoint_objective = (
        development_views.get("checkpoint_objective")
        if development_views is not None
        else None
    )
    if checkpoint_objective is not None:
        selection_ranking = checkpoint_objective.get("version")
    if development_views is not None:
        if SELECTION_REFERENCE is not None:
            development_views = dict(
                development_views, selection_reference=SELECTION_REFERENCE
            )
        if rank == 0:
            print(
                "selection: protected development views (%s)"
                % selection_ranking
                if checkpoint_objective is not None
                else "selection: joint campaign constraints (%s)"
                % selection_ranking
                if SELECTION_REFERENCE is not None
                else "selection: LEGACY transfer ranking "
                "(no selection_reference in recipe)",
                flush=True,
            )
    for key in ("min_confidence", "abstention_thresholds"):
        cfg.pop(key, None)
    items = {
        split: prepare_items(tok, split_rows, args.max_len, args.head_max_len)
        for split, split_rows in rows.items()
    }
    weights = (
        np.ones(len(rows["train"]), dtype=np.float64)
        if args.sampling == "grouped"
        else hierarchical_weights(rows["train"])
    )
    weights /= weights.mean()
    if args.checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.head_checkpointing = args.checkpointing
    model.to(device)
    teacher = None
    teacher_identity = None
    if REPLAY_PRESERVATION is not None:
        if pool_plan is None or not set(REPLAY_PRESERVATION["pools"]) <= set(
            pool_plan[1]
        ):
            raise ValueError("Teacher preservation requires registered pools")
        teacher_path = teacher_checkpoint(REPLAY_PRESERVATION, args.model)
        teacher_identity = checkpoint_manifest(teacher_path)
        teacher, _, _ = load_readonly_checkpoint(teacher_path, load_checkpoint)
        teacher.requires_grad_(False)
        teacher.to(device).eval()
    adapter_lr = args.head_lr if args.adapter_lr is None else args.adapter_lr
    groups = optimizer_param_groups(
        model, args.encoder_lr, adapter_lr, args.head_lr
    )
    optimizer = torch.optim.AdamW(groups)

    # Pre-training snapshot of the components under investigation (adapters +
    # decision head) for actual parameter-change measurement. The encoder is
    # excluded: it is frozen in fitting mode and moves slowly otherwise.
    def _tracked(name):
        return "residual_adapters" in name or (
            not name.startswith("encoder.") and "residual_adapters" not in name
        )

    tracked_snapshot = {
        name: parameter.detach().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and _tracked(name)
    }
    signature = {
        "run_id": args.run_id,
        "execution_identity": execution_identity,
        "data": {
            split: sha256(Path(args.data) / (split + ".jsonl"))
            for split in rows
        },
        "evaluation_role": "train_fitting" if fitting else "development",
        "fitting_subset": {
            "requested": args.limit_train_rows,
            "rows": len(rows["train"]),
            "ids_sha256": digest_text(
                json.dumps([r["id"] for r in rows["train"]])
            ),
        }
        if fitting
        else None,
        "freeze_encoder": args.freeze_encoder,
        "encoder_trainable_blocks": args.encoder_trainable_blocks,
        "disable_adapters": args.disable_adapters,
        "world": world,
        "seed": args.seed,
        "micro_batch": args.micro_batch,
        "eval_batch_size": args.eval_batch_size,
        "accumulation": args.accumulation,
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "adapter_lr": adapter_lr,
        "warmup": args.warmup,
        "checkpointing": args.checkpointing,
        "checkpoint_interval": args.checkpoint_interval,
        "adapter": adapter_cfg,
        "min_updates": args.min_updates,
        "patience": args.patience,
        "continue_training": args.continue_training,
        "smoke_steps": args.smoke_steps,
        "sampling": args.sampling,
        "pools": pools_record,
        "replay_preservation": REPLAY_PRESERVATION,
        "eval_every_updates": args.eval_every_updates,
        "max_updates": args.max_updates,
        "report_bootstrap": args.report_bootstrap,
        "question_contract": args.question_contract,
        "plateau_requires_full_exposure": bool(
            args.plateau_requires_full_exposure
        ),
        "review_ledger": sha256(args.review_ledger)
        if args.review_ledger
        else None,
        "review_packets": sha256(args.review_packets)
        if args.review_packets
        else None,
        "reporting_contract": reporting_contract,
        "selection": (
            selection_ranking
            if checkpoint_objective is not None
            else "joint_campaign_constraints_v1:" + selection_ranking
            if SELECTION_REFERENCE is not None
            else "competence_finite_security_pair_cost_v2"
        ),
        "acceptance": dict(ACCEPTANCE),
        "source_mass": dict(SOURCE_MASS),
        "code_manifest": code_manifest(),
        "initial_checkpoint": checkpoint_manifest(args.model),
        "torch": torch.__version__,
    }
    signature.update(
        new_study=new_study,
        development_views=development_views,
        lr_schedule=args.lr_schedule,
        hold_updates=args.hold_updates,
        lr_endpoint=args.lr_endpoint,
        no_diagnostic_weights=args.no_diagnostic_weights,
    )
    if teacher_identity is not None:
        signature["preservation_teacher"] = teacher_identity
    if existing_report.exists():
        previous = json.loads(existing_report.read_text())
        if previous.get("signature") != signature:
            raise ValueError(
                "completed run signature differs from requested resume"
            )
        if world > 1:
            dist.destroy_process_group()
        return
    progress = {
        "epoch": 0,
        "offset": 0,
        "step": 0,
        "lr_scale": 1.0,
        "best_key": [1e9, 0, 1e9],
        "best_epoch": -1,
        "best_generation": None,
        "stale": 0,
        "reduced": False,
        "training_seconds": 0.0,
        "stop_reason": None,
        "lineage": 0,
        "last_eval_step": 0,
    }
    resume_rng = None
    lineage_path = output / "lineage.json"
    prior_lineage = (
        json.loads(lineage_path.read_text())["lineage"]
        if lineage_path.exists()
        else -1
    )
    if args.resume:
        checkpoint = torch.load(
            output / "latest.pt", map_location="cpu", weights_only=False
        )
        if checkpoint["signature"] != signature:
            raise ValueError("resume signature differs from saved run")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        progress = checkpoint["progress"]
        if (
            progress.get("stop_reason") == "operator_pause"
            and not args.finalize_development_only
        ):
            progress["stop_reason"] = None
        resume_rng = checkpoint["rng"][rank]
        del checkpoint
        progress["lineage"] = max(progress.get("lineage", 0), prior_lineage) + 1
        if not args.smoke_steps:
            restore_best_lineage(output, progress, rank, world)
    else:
        progress["lineage"] = prior_lineage + 1
    if rank == 0:
        write_json(
            lineage_path,
            {"lineage": progress["lineage"], "run_id": args.run_id},
        )
    ddp = (
        DistributedDataParallel(
            model, device_ids=[local], gradient_as_bucket_view=True
        )
        if world > 1
        else model
    )
    progress["frozen_digest"] = replica_digest(ddp, False)
    torch.manual_seed(args.seed + rank)
    if resume_rng is not None:
        restore_rng(resume_rng)
    if rank == 0:
        write_json(
            output / "configuration.json",
            {
                "arguments": vars(args),
                "signature": signature,
                "counts": {
                    split: len(values) for split, values in items.items()
                },
                "families": {
                    split: len({row["family_id"] for row in split_rows})
                    for split, split_rows in rows.items()
                },
                "lengths": {
                    split: {
                        "min": min(item["length"] for item in values),
                        "max": max(item["length"] for item in values),
                    }
                    for split, values in items.items()
                },
                "weighting": weighting_report(rows["train"], weights),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "effective_batch": world * args.micro_batch * args.accumulation,
            },
        )

    def log(event):
        if rank == 0:
            event = dict(event, run_id=args.run_id, time=time.time())
            print(json.dumps(event, allow_nan=False), flush=True)
            with (output / "events.jsonl").open("a") as handle:
                handle.write(json.dumps(event, allow_nan=False) + "\n")
            write_json(output / "status.json", event)

    baselines = trivial_baselines(rows[evaluation_split])
    baseline_cost = baselines["best_selection_cost"]

    def add_views(result, logits):
        if development_views:
            attach_views(
                result,
                rows[evaluation_split],
                logits,
                development_views,
                lambda selected_rows, selected_logits: (
                    summarize_development_view(
                        selected_rows,
                        selected_logits,
                        bootstrap_samples=args.report_bootstrap,
                        seed=args.seed,
                        score_triplets=bool(
                            development_views.get("score_triplets")
                        ),
                        score_sources=bool(
                            development_views.get("score_sources")
                        ),
                    )
                ),
            )

    def choose_key(result):
        if development_views:
            if "checkpoint_objective" in development_views:
                result["retention_check"] = None
                return view_selection(
                    result, development_views, None, selection_key
                )
            reference = progress["retention_reference_accuracy"]
            key = view_selection(
                result,
                development_views,
                reference,
                selection_key,
            )
            result["retention_check"] = retention_check_for(
                development_views, result, reference
            )
            return key
        return selection_key(result)

    if rank == 0:
        write_json(output / (evaluation_split + "_baselines.json"), baselines)
    if not args.resume and not args.smoke_steps:
        initial, initial_logits, initial_y = evaluate(
            model,
            tok,
            items[evaluation_split],
            args,
            device,
            rank,
            world,
            baseline_cost=baseline_cost,
        )
        attach_reporting(
            initial, rows[evaluation_split], initial_logits, args, rank
        )
        add_views(initial, initial_logits)
        if development_views:
            objective = "checkpoint_objective" in development_views
            if not objective:
                progress["retention_reference_accuracy"] = initial[
                    "development_views"
                ][development_views["retention_view"]]["accuracy"]
        key = choose_key(initial)
        progress.update(
            best_key=list(key),
            best_epoch=0,
            best_generation=generation_name(0, 0, progress["lineage"]),
            initial_generation=generation_name(0, 0, progress["lineage"]),
        )
        if rank == 0:
            save_best(
                model,
                output,
                0,
                0,
                key,
                initial,
                initial_logits,
                initial_y,
                progress["lineage"],
            )
        log(
            {
                "event": "initial_train" if fitting else "initial_development",
                "metrics": initial,
            }
        )
        save_resume(model, optimizer, progress, signature, output, rank, world)

    def observe_development(epoch, consumed):
        if world > 1:
            assert_replicas_consistent(
                ddp, rank, world, "eval-step-%d" % progress["step"]
            )
            verify_frozen_steady(
                ddp,
                progress["frozen_digest"],
                "eval-step-%d" % progress["step"],
            )
        result, logits, targets = evaluate(
            model,
            tok,
            items[evaluation_split],
            args,
            device,
            rank,
            world,
            baseline_cost=baseline_cost,
        )
        attach_reporting(result, rows[evaluation_split], logits, args, rank)
        add_views(result, logits)
        key = choose_key(result)
        if rank == 0:
            save_eval_snapshot(
                model,
                output,
                epoch + 1,
                progress["step"],
                key,
                result,
                progress["lineage"],
            )
        if key < tuple(progress["best_key"]):
            progress.update(
                best_key=list(key),
                best_epoch=epoch + 1,
                stale=0,
                best_generation=generation_name(
                    epoch + 1, progress["step"], progress["lineage"]
                ),
            )
            if rank == 0:
                save_best(
                    model,
                    output,
                    epoch + 1,
                    progress["step"],
                    key,
                    result,
                    logits,
                    targets,
                    progress["lineage"],
                )
        else:
            progress["stale"] += 1
        if rank == 0 and not args.no_diagnostic_weights:
            # Diagnostic alternatives never replace the predeclared selection rule.
            for name, alt_key in (
                (
                    "accuracy",
                    [
                        -result["accuracy"],
                        -result["both_pair_members_correct"],
                        result["nll"],
                    ],
                ),
                ("nll", [result["nll"]]),
            ):
                folder = output / "diagnostic_candidates" / name
                pointer = folder / "best.json"
                if (
                    not pointer.exists()
                    or alt_key < json.loads(pointer.read_text())["key"]
                ):
                    save_best(
                        model,
                        folder,
                        epoch + 1,
                        progress["step"],
                        alt_key,
                        result,
                        logits,
                        targets,
                        progress["lineage"],
                    )
        progress["last_eval_step"] = progress["step"]
        progress["last_metrics"] = result
        exposure_complete = None
        if args.plateau_requires_full_exposure:
            seen = reconstruct_seen(
                rows["train"],
                pool_plan,
                args.seed,
                epoch,
                consumed,
                world=world,
                global_batch=(world * args.micro_batch * args.accumulation),
                grouped_epoch_indices=grouped_epoch_indices,
            )
            assignment = (
                pool_plan[0]
                if pool_plan is not None
                else {
                    index: row["source"]["name"]
                    for index, row in enumerate(rows["train"])
                }
            )
            exposure = delivered_exposure(
                seen,
                rows["train"],
                assignment,
                checkpoint_step=progress["step"],
            )
            exposure_complete = exposure_gate_passed(exposure)
            result["delivered_exposure"] = exposure
        action = plateau_action(
            progress,
            epoch,
            args.min_updates,
            args.patience,
            exposure_complete,
        )
        if action == "reduce_lr":
            progress.update(lr_scale=0.5, stale=0, reduced=True)
            log({"event": "learning_rate_reduced", "epoch": epoch + 1})
        elif action == "stop":
            progress["stop_reason"] = "development_plateau"
        log(
            {
                "event": "train_evaluation" if fitting else "development",
                "epoch": epoch + 1,
                "step": progress["step"],
                "metrics": result,
                "progress": progress,
            }
        )
        return action == "stop"

    model.train()
    start_time, previous_seconds = (
        time.monotonic(),
        progress["training_seconds"],
    )
    last_save = time.monotonic()
    stop = progress["stop_reason"] is not None
    stop = stop or args.finalize_development_only
    sampler_state: dict = {}
    if args.sampling == "grouped" and progress["epoch"] > 0:
        # Rebuild carried queue state deterministically: replaying
        # prior epochs reproduces exactly the queues training had.
        for prior in range(progress["epoch"]):
            grouped_epoch_indices(
                rows["train"],
                world,
                rank,
                args.seed,
                prior,
                world * args.micro_batch * args.accumulation,
                pool_plan,
                _queue_state=sampler_state,
            )
    for epoch in range(progress["epoch"], args.epochs):
        if stop:
            break
        indices = epoch_indices(
            len(items["train"]), world, rank, args.seed, epoch
        )
        valid_mask = epoch_valid_mask(
            len(items["train"]), world, rank, args.seed, epoch
        )
        if args.sampling == "grouped":
            indices = grouped_epoch_indices(
                rows["train"],
                world,
                rank,
                args.seed,
                epoch,
                world * args.micro_batch * args.accumulation,
                pool_plan,
                _queue_state=sampler_state,
            )
            valid_mask = [1] * len(indices)
        if args.smoke_steps:
            item_count = len(items["train"])
            longest = sorted(
                range(item_count),
                key=lambda i: items["train"][i]["length"],
                reverse=True,
            )
            total = math.ceil(item_count / world) * world
            longest = (longest * math.ceil(total / item_count))[:total]
            indices = longest[rank:total:world]
            valid_mask = ([1] * item_count + [0] * (total - item_count))[
                rank:total:world
            ]
        begin = progress["offset"] if epoch == progress["epoch"] else 0
        for end, chunks in accumulation_groups(
            indices, args.micro_batch, args.accumulation
        ):
            if end <= begin:
                continue
            if rank == 0:
                try:
                    from operator_control import apply_control

                    control = apply_control(
                        output, args.run_id, progress, args.encoder_lr
                    )
                    control_result = {"action": control, "progress": progress}
                except Exception as exc:
                    # Isolate the optional control hook: a control-plane
                    # failure is recorded, never allowed to kill training.
                    control_result = {"error": str(exc)}
            else:
                control_result = None
            if world > 1:
                control_objects = [control_result]
                dist.broadcast_object_list(control_objects, src=0)
                control_result = control_objects[0]
            if "error" in control_result:
                raise ValueError(control_result["error"])
            progress.update(control_result["progress"])
            if control_result["action"] == "stop":
                stop = True
                progress["stop_reason"] = "operator_pause"
                break
            if rank == 0:
                expired = (
                    previous_seconds + time.monotonic() - start_time
                    >= args.hours * 3600
                )
                stop = (
                    expired
                    or bool(
                        args.smoke_steps
                        and progress["step"] >= args.smoke_steps
                    )
                    or bool(
                        args.max_updates
                        and progress["step"] >= args.max_updates
                    )
                )
            if world > 1:
                decisions = [stop]
                dist.broadcast_object_list(decisions, src=0)
                stop = decisions[0]
            if stop:
                progress["stop_reason"] = (
                    "smoke_steps"
                    if args.smoke_steps
                    else "update_budget"
                    if args.max_updates and progress["step"] >= args.max_updates
                    else "time_budget"
                )
                break
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            warm = learning_rate_factor(
                progress["step"],
                args.warmup,
                args.max_updates,
                args.lr_schedule,
                args.hold_updates,
                args.lr_endpoint,
            )
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * progress["lr_scale"] * warm
            group_valid = []
            cursor = end - sum(len(chunk) for chunk, _, _ in chunks)
            for chunk, _, _ in chunks:
                group_valid.append(valid_mask[cursor : cursor + len(chunk)])
                cursor += len(chunk)
            real_local = sum(sum(values) for values in group_valid)
            real_total = torch.tensor(real_local, device=device)
            if world > 1:
                dist.all_reduce(real_total)
            if real_total.item() <= 0:
                raise RuntimeError("accumulation group contains only padding")
            scale = world / real_total.item()
            loss_sum = torch.zeros((), device=device)
            preservation_sum = torch.zeros((), device=device)
            preservation_raw_sum = torch.zeros((), device=device)
            preservation_pool_rows = torch.zeros((), device=device)
            preservation_match_rows = torch.zeros((), device=device)
            preservation_agree_rows = torch.zeros((), device=device)
            source_stats = torch.zeros(
                (len(SOURCE_MASS), 2), device=device, dtype=torch.float64
            )
            source_names = sorted(SOURCE_MASS)
            for (chunk, _, synchronize), valid_chunk in zip(
                chunks, group_valid
            ):
                batch = batch_for(tok, items["train"], chunk, args, epoch=epoch)
                with (
                    ddp.no_sync()
                    if world > 1 and not synchronize
                    else nullcontext()
                ):
                    logits = forward(ddp, batch, device)
                    target = batch["target"].to(device)
                    logp = torch.log_softmax(
                        logits.masked_fill(
                            ~batch["marker_mask"].to(device), -1e4
                        ),
                        -1,
                    )
                    per_item = -(target * logp).sum(-1)
                    sample_weights = torch.as_tensor(
                        weights[chunk], device=device, dtype=torch.float32
                    )
                    valid = torch.as_tensor(
                        valid_chunk, device=device, dtype=torch.float32
                    )
                    loss = (per_item * sample_weights * valid).sum()
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits = forward(teacher, batch, device)
                        eligible = torch.tensor(
                            [
                                pool_plan[0][i] in REPLAY_PRESERVATION["pools"]
                                for i in chunk
                            ],
                            device=device,
                        )
                        penalty = preservation_loss(
                            logits,
                            teacher_logits,
                            target,
                            eligible,
                            REPLAY_PRESERVATION["temperature"],
                        )
                        penalty_sum = (
                            penalty * sample_weights * valid
                        ).sum() * REPLAY_PRESERVATION["weight"]
                        loss = loss + penalty_sum
                        preservation_sum += penalty_sum.detach()
                        preservation_raw_sum += (
                            (penalty * sample_weights * valid).sum().detach()
                        )
                        diagnostics = preservation_diagnostics(
                            logits, teacher_logits, target, eligible, valid
                        )
                        preservation_pool_rows += diagnostics["pool_rows"]
                        preservation_match_rows += diagnostics[
                            "teacher_match_rows"
                        ]
                        preservation_agree_rows += diagnostics[
                            "student_teacher_agree_rows"
                        ]
                    (loss * scale).backward()
                    loss_sum += loss.detach()
                    with torch.no_grad():
                        for source_index, source_name in enumerate(
                            source_names
                        ):
                            mask = (
                                torch.tensor(
                                    [
                                        rows["train"][i]["source"]["name"]
                                        == source_name
                                        for i in chunk
                                    ],
                                    device=device,
                                )
                                * valid
                            )
                            source_stats[source_index, 0] += (
                                per_item.detach() * mask
                            ).sum()
                            source_stats[source_index, 1] += mask.sum()
            unclipped = component_gradient_norms(model)
            norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                1.0,
            )
            clipped_norms = component_gradient_norms(model)
            finite = torch.tensor(
                int(torch.isfinite(norm) and torch.isfinite(loss_sum)),
                device=device,
            )
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(
                    "nonfinite loss or gradient; update was not applied"
                )
            if (
                args.smoke_steps
                and adapter_cfg is not None
                and progress["step"] in (0, 1)
            ):
                gradients = {
                    "up": sum(
                        float(branch.up.weight.grad.abs().sum())
                        for branch in model.residual_adapters.branches.values()
                    ),
                    "down": sum(
                        float(branch.down.weight.grad.abs().sum())
                        for branch in model.residual_adapters.branches.values()
                    ),
                }
                if gradients["up"] <= 0 or (
                    progress["step"] == 1 and gradients["down"] <= 0
                ):
                    raise RuntimeError("adapter gradient did not flow")
                log(
                    {
                        "event": "adapter_gradients",
                        "step": progress["step"],
                        **gradients,
                    }
                )
            optimizer.step()
            progress.update(
                epoch=epoch,
                offset=end,
                step=progress["step"] + 1,
                training_seconds=previous_seconds
                + time.monotonic()
                - start_time,
            )
            if world > 1 and progress["step"] <= 3:
                assert_replicas_consistent(
                    ddp, rank, world, "update-%d" % progress["step"]
                )
                verify_frozen_steady(
                    ddp,
                    progress["frozen_digest"],
                    "update-%d" % progress["step"],
                )
            if world > 1:
                dist.all_reduce(loss_sum)
                dist.all_reduce(preservation_sum)
                dist.all_reduce(preservation_raw_sum)
                dist.all_reduce(preservation_pool_rows)
                dist.all_reduce(preservation_match_rows)
                dist.all_reduce(preservation_agree_rows)
                dist.all_reduce(source_stats)
            group_norms = dict(clipped_norms)
            group_norms["clip_scale_applied"] = min(
                1.0, 1.0 / (float(norm) + 1e-6)
            )
            # AdamW movement relative to this invocation's initialization. Keep
            # snapshots on-device to avoid copying the head on every update.
            if rank == 0:
                for target in ("adapter", "head"):
                    deltas = [
                        (parameter.detach().float() - tracked_snapshot[name])
                        .square()
                        .sum()
                        for name, parameter in model.named_parameters()
                        if name in tracked_snapshot
                        and component_name(name) == target
                    ]
                    group_norms[target + "_delta_norm"] = (
                        float(torch.stack(deltas).sum().sqrt())
                        if deltas
                        else 0.0
                    )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - step_started
            log(
                {
                    "event": "train",
                    "epoch": epoch + 1,
                    "step": progress["step"],
                    "loss": float(loss_sum) / real_total.item(),
                    "preservation_loss": float(preservation_sum)
                    / real_total.item(),
                    "supervised_ce": float(loss_sum - preservation_sum)
                    / real_total.item(),
                    "preservation_unweighted": float(preservation_raw_sum)
                    / real_total.item(),
                    "preservation_pool_rows": int(preservation_pool_rows),
                    "preservation_teacher_match_rows": int(
                        preservation_match_rows
                    ),
                    "preservation_agree_rows": int(preservation_agree_rows),
                    "step_seconds": elapsed,
                    "component_grad_norms": group_norms,
                    "examples_per_second": real_total.item() / elapsed,
                    "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "gradient_norm_before_clip": float(norm),
                    "gradient_clipped": bool(norm > 1.0),
                    "source_counts": {
                        name: int(source_stats[i, 1])
                        for i, name in enumerate(source_names)
                    },
                    "source_unweighted_nll": {
                        name: float(source_stats[i, 0] / source_stats[i, 1])
                        if source_stats[i, 1]
                        else None
                        for i, name in enumerate(source_names)
                    },
                    "encoder_lr": args.encoder_lr * progress["lr_scale"] * warm,
                    "unclipped_component_norms": unclipped,
                    "clipped_global_norm": math.sqrt(
                        sum(v * v for v in clipped_norms.values())
                    ),
                }
            )
            decisions = [False]
            if rank == 0:
                decisions = [
                    time.monotonic() - last_save >= args.checkpoint_interval
                ]
            if world > 1:
                dist.broadcast_object_list(decisions, src=0)
            if decisions[0]:
                save_resume(
                    model, optimizer, progress, signature, output, rank, world
                )
                last_save = time.monotonic()
            if (
                not args.smoke_steps
                and args.eval_every_updates
                and progress["step"] % args.eval_every_updates == 0
            ):
                stop = observe_development(epoch, end)
                save_resume(
                    model, optimizer, progress, signature, output, rank, world
                )
                last_save = time.monotonic()
                if stop:
                    break
        if args.smoke_steps:
            if stop:
                break
            progress.update(epoch=epoch + 1, offset=0)
            continue
        if stop:
            if progress["offset"] >= len(indices):
                progress.update(epoch=epoch + 1, offset=0)
            break
        if (
            not args.eval_every_updates
            and progress["last_eval_step"] != progress["step"]
        ):
            stop = observe_development(epoch, len(indices))
        if progress["offset"] >= len(indices):
            progress.update(epoch=epoch + 1, offset=0)
        if stop:
            break
    progress["stop_reason"] = progress.get("stop_reason") or "epoch_budget"
    save_resume(model, optimizer, progress, signature, output, rank, world)
    if args.smoke_steps:
        checksum = hashlib.sha256()
        for name, parameter in model.state_dict().items():
            checksum.update(name.encode())
            checksum.update(
                parameter.detach().cpu().contiguous().numpy().tobytes()
            )
        checksum = checksum.hexdigest()
        checks = [None] * world
        if world > 1:
            dist.all_gather_object(checks, checksum)
        else:
            checks[0] = checksum
        if not all(checksum == value for value in checks):
            raise RuntimeError("replica checksums differ")
        log(
            {
                "event": "smoke_complete",
                "step": progress["step"],
                "replicas_equal": True,
                "weights_sha256": checksum,
                "bf16": True,
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
    else:
        # Terminal evaluation: the update budget may exit on a step with no
        # evaluation. Evaluate the final weights before selecting/exporting so
        # the exported checkpoint is the evaluated one, not a stale best.
        if world > 1:
            assert_replicas_consistent(ddp, rank, world, "pre-finalize")
            verify_frozen_steady(ddp, progress["frozen_digest"], "pre-finalize")
        finalize_training(
            model,
            optimizer,
            tok,
            cfg,
            output,
            signature,
            progress,
            args.run_id,
            rank,
            world,
            log,
            observe_development,
            device,
        )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
