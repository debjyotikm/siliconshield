"""Frozen, reproducible decision-model study plans.

A study manifest binds everything a training run depends on: the
starting checkpoint, the accepted dataset version, code and
runtime identity, sampling, learning settings, frozen evaluation
rules, and the exact launch command. Two modes separate
improvement from new-model work:

- ``improve_existing`` continues a released model and protects
  its established capabilities with retention floors.
- ``create_new`` starts from a verified untouched foundation
  checkpoint and qualifies a new model domain.

Nothing here launches training. Freezing records hashes; later
verification re-checks every byte before a run may start.
"""

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from siliconshield.research_loop.procedures.bundles import read_bundle
from siliconshield.schemas import (
    CHECKPOINT_BALANCED_SCHEMA,
    CHECKPOINT_OBJECTIVE_SCHEMA,
    CHECKPOINT_SOURCED_SCHEMA,
    CHECKPOINT_TRIPLETS_SCHEMA,
)
from siliconshield.schemas import (
    STUDY_MANIFEST_V2_SCHEMA as STUDY_MANIFEST_SCHEMA,
)
from siliconshield.training.laya import dataset_versions
from siliconshield.training.laya.checkpoint_selection import (
    constraint_violations,
    population_indices,
    triplet_counts,
    view_counts,
)
from siliconshield.training.laya.coverage_audit import (
    exposure_gate_passed,
)
from siliconshield.training.laya.question_contracts import (
    LABELS,
    LEGACY_CONTRACT,
    NEW_CONTRACT,
    read_export_record,
)
from siliconshield.training.laya.review_ledger import (
    ledger_digest,
    require_template_coverage,
    validate_ledger,
)

STUDY_MODES = ("improve_existing", "create_new")
TRAINABLE_MODES = ("full_encoder", "partial_blocks", "adapters_only")
DIAGNOSES = (
    "execution_defect",
    "poor_labels",
    "insufficient_exposure",
    "failure_to_fit",
    "poor_generalization",
    "retention_tradeoff",
)
OBJECTIVE_SCHEMAS = (
    CHECKPOINT_OBJECTIVE_SCHEMA,
    CHECKPOINT_TRIPLETS_SCHEMA,
    CHECKPOINT_BALANCED_SCHEMA,
    CHECKPOINT_SOURCED_SCHEMA,
)
CONTRACTS = (LEGACY_CONTRACT, NEW_CONTRACT)
RESERVED_SPLITS = ("qualification", "test")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(name + " must be an object")
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a non-empty string")
    return value


def _require_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(name + " must be an integer count")
    if value < 0:
        raise ValueError(name + " must be non-negative")
    return int(value)


def _require_rate(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be positive and finite")
    return float(value)


def _hash_tree(root: Path) -> dict[str, str]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Not a directory: " + str(root))
    files = {
        str(path.relative_to(root)): _sha_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    if not files:
        raise ValueError("Empty tree: " + str(root))
    return files


def _check_tree(root: Path, expected: dict[str, str], name: str) -> None:
    if _hash_tree(root) != expected:
        raise ValueError(name + " bytes changed since freezing")


def freeze_study_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate a study spec and freeze it into a manifest.

    Hashes the parent export, code tree, train/dev splits, pools,
    and contract file now; re-checks dataset acceptance, semantic
    review coverage, and objective shapes. Reserved
    qualification/test splits are recorded by declared hash and
    never opened here, so freezing cannot leak them into
    training. Returns the manifest without writing it.
    """
    spec = _require_mapping(spec, "Study spec")
    mode = spec.get("mode")
    if mode not in STUDY_MODES:
        raise ValueError("Unknown study mode: " + repr(mode))
    study_id = _require_text(spec.get("study_id"), "study_id")

    parent = _freeze_parent(_require_mapping(spec.get("parent"), "parent"))
    if mode == "create_new" and (
        parent["adapters_present"] or not parent.get("foundation")
    ):
        raise ValueError(
            "create_new needs a pinned adapter-free foundation reference"
        )
    dataset = _freeze_dataset(_require_mapping(spec.get("dataset"), "dataset"))
    code = _freeze_code(_require_mapping(spec.get("code"), "code"))
    training = _freeze_training(
        _require_mapping(spec.get("training"), "training")
    )
    validate_training_settings(training, code, dataset, parent)
    evaluation = _freeze_evaluation(
        _require_mapping(spec.get("evaluation"), "evaluation")
    )
    budget = _freeze_budget(_require_mapping(spec.get("budget"), "budget"))
    procedure = _freeze_procedure(spec.get("procedure"))
    command = spec.get("launch_command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) for part in command)
    ):
        raise ValueError("launch_command must be a non-empty string list")

    manifest = {
        "schema": STUDY_MANIFEST_SCHEMA,
        "study_id": study_id,
        "mode": mode,
        "parent": parent,
        "dataset": dataset,
        "code": code,
        "training": training,
        "evaluation": evaluation,
        "budget": budget,
        "procedure": procedure,
        "launch_command": list(command),
    }
    manifest["manifest_sha256"] = _sha_text(
        _canonical(
            {key: manifest[key] for key in manifest if key != "manifest_sha256"}
        )
    )
    return manifest


def _freeze_parent(parent: dict[str, Any]) -> dict[str, Any]:
    root = Path(_require_text(parent.get("path"), "parent.path")).resolve()
    files = _hash_tree(root)
    weights = _require_text(
        parent.get("weights_sha256"), "parent.weights_sha256"
    )
    if files.get("model.safetensors") != weights:
        raise ValueError("Parent weights hash does not match model.safetensors")
    # Read tensor names without materializing any tensor or requiring torch.
    with (root / "model.safetensors").open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Parent is not a safetensors export")
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= 16 * 1024 * 1024:
            raise ValueError("Parent safetensors header is invalid")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError("Parent safetensors header must be an object")
    adapters = any("adapter" in name.lower() for name in header)
    if parent.get("adapters_present") is not adapters:
        raise ValueError(
            "Declared adapter presence disagrees with tensor names"
        )
    contract = read_export_record(root)["version"]
    if parent.get("question_contract") != contract:
        raise ValueError("Parent question contract disagrees with its export")
    frozen = {
        "path": str(root),
        "files": files,
        "weights_sha256": weights,
        "adapters_present": adapters,
        "question_contract": contract,
    }
    foundation = parent.get("foundation")
    if foundation is not None:
        foundation = _require_mapping(foundation, "parent.foundation")
        path = Path(
            _require_text(foundation.get("reference"), "foundation.reference")
        )
        expected = _require_text(foundation.get("sha256"), "foundation.sha256")
        if _sha_file(path) != expected:
            raise ValueError("Foundation reference hash mismatch")
        reference = _require_mapping(
            json.loads(path.read_text()), "Foundation reference"
        )
        for key in ("checkpoint", "revision"):
            _require_text(reference.get(key), "foundation." + key)
        if reference.get("files") != files:
            raise ValueError("Parent is not the pinned foundation export")
        frozen["foundation"] = {
            "reference": str(path.resolve()),
            "sha256": expected,
            "checkpoint": reference["checkpoint"],
            "revision": reference["revision"],
        }
    return frozen


def _freeze_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    root = Path(_require_text(dataset.get("root"), "dataset.root"))
    version = _require_text(dataset.get("version"), "dataset.version")
    manifest = dataset_versions.verify_version(root, version)
    if not dataset_versions.is_accepted(root, version):
        raise ValueError("Dataset version is not accepted: " + version)
    acceptance = json.loads(
        (root / "versions" / version / "accepted.json").read_text()
    )
    splits = _require_mapping(dataset.get("splits"), "dataset.splits")
    if set(splits) != {"train", "dev"} | set(RESERVED_SPLITS):
        raise ValueError("dataset.splits needs train/dev/qualification/test")
    frozen_splits: dict[str, dict[str, Any]] = {}
    for name, raw in splits.items():
        if name not in ("train", "dev"):
            continue
        entry = _require_mapping(raw, "dataset.splits." + name)
        path = Path(_require_text(entry.get("path"), name + ".path"))
        frozen_splits[name] = {
            "path": str(path.resolve()),
            "sha256": _sha_file(path),
        }
        accepted = manifest["splits"].get(name, {})
        if frozen_splits[name]["sha256"] != accepted.get("sha256"):
            raise ValueError("Study split is not the accepted split: " + name)
    # Reserved populations are hash-bound, never read: opening them
    # here would expose sealed rows to the training side.
    for name in RESERVED_SPLITS:
        entry = _require_mapping(splits[name], "dataset.splits." + name)
        if entry.get("status") == "not_materialized":
            if set(entry) != {"status", "reason"}:
                raise ValueError(
                    "Pending evaluation cannot claim dataset hashes"
                )
            frozen_splits[name] = {
                "status": "not_materialized",
                "reason": _require_text(entry["reason"], name + ".reason"),
                "verified": False,
            }
            continue
        digest = _require_text(entry.get("sha256"), name + ".sha256")
        if len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise ValueError(name + ".sha256 must be a hex digest")
        frozen_splits[name] = {"sha256": digest, "verified": False}
    pools_path = Path(_require_text(dataset.get("pools"), "dataset.pools"))
    pools = dataset_versions.verify_pools(
        Path(frozen_splits["train"]["path"]), pools_path
    )
    if manifest.get("pools") and manifest["pools"]["sha256"] != _sha_file(
        pools_path
    ):
        raise ValueError("Study pools differ from the accepted dataset version")
    ledger_path = Path(
        _require_text(dataset.get("review_ledger"), "dataset.review_ledger")
    )
    if _sha_file(ledger_path) != acceptance["audit"].get(
        "review_ledger_sha256"
    ):
        raise ValueError("Study review ledger is not the accepted ledger")
    records = validate_ledger(ledger_path)
    rows = [
        json.loads(line)
        for line in Path(frozen_splits["train"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    require_template_coverage(rows, records)
    return {
        "root": str(root.resolve()),
        "version": version,
        "version_sha256": _sha_text(_canonical(manifest)),
        "acceptance_sha256": _sha_file(
            root / "versions" / version / "accepted.json"
        ),
        "splits": frozen_splits,
        "pools": {
            "path": str(pools_path.resolve()),
            "sha256": _sha_file(pools_path),
            "names": sorted(pools["pools"]),
        },
        "review_ledger": {
            "path": str(ledger_path.resolve()),
            "digest": ledger_digest(records),
            "sha256": _sha_file(ledger_path),
        },
    }


def _freeze_code(code: dict[str, Any]) -> dict[str, Any]:
    root = Path(_require_text(code.get("path"), "code.path")).resolve()
    files = _hash_tree(root)
    runtime = _require_mapping(code.get("runtime"), "code.runtime")
    runtime_path = Path(
        _require_text(runtime.get("path"), "runtime.path")
    ).resolve()
    tokenizer = Path(
        _require_text(code.get("tokenizer_path"), "tokenizer_path")
    ).resolve()
    contract_path = Path(
        _require_text(code.get("question_contract_path"), "code.contract")
    ).resolve()
    contract_id = _require_text(
        code.get("question_contract"), "code.question_contract"
    )
    record = read_export_record(contract_path.parent)
    if (
        contract_path.name != "question_contract.json"
        or record["version"] != contract_id
    ):
        raise ValueError(
            "Question contract path must name a matching export record"
        )
    return {
        "path": str(root),
        "files": files,
        "source_sha256": _sha_text(_canonical(files)),
        "runtime": {
            "name": _require_text(runtime.get("name"), "runtime.name"),
            "revision": _require_text(
                runtime.get("revision"), "runtime.revision"
            ),
            "path": str(runtime_path),
            "files": _hash_tree(runtime_path),
        },
        "tokenizer": {"path": str(tokenizer), "files": _hash_tree(tokenizer)},
        "python": {
            "path": str(
                Path(_require_text(code.get("python"), "code.python")).resolve()
            ),
            "sha256": _sha_file(Path(code["python"])),
        },
        "image_digest": _require_text(
            code.get("image_digest"), "code.image_digest"
        ),
        "question_contract": contract_id,
        "question_contract_path": str(contract_path),
        "question_contract_sha256": _sha_file(contract_path),
    }


def _freeze_training(training: dict[str, Any]) -> dict[str, Any]:
    trainables = _require_mapping(
        training.get("trainables"), "training.trainables"
    )
    train_mode = trainables.get("mode")
    if train_mode not in TRAINABLE_MODES:
        raise ValueError("Unknown trainables mode: " + repr(train_mode))
    frozen_trainables: dict[str, Any] = {"mode": train_mode}
    if train_mode == "partial_blocks":
        blocks = trainables.get("blocks")
        if (
            not isinstance(blocks, list)
            or not blocks
            or any(type(item) is not int or item < 0 for item in blocks)
        ):
            raise ValueError("partial_blocks needs a non-empty block list")
        frozen_trainables["blocks"] = sorted(set(blocks))
    rates = _require_mapping(
        training.get("learning_rates"), "training.learning_rates"
    )
    frozen_rates = {
        name: _require_rate(rates.get(name), "learning_rates." + name)
        for name in ("adapter", "head")
    }
    if train_mode in ("full_encoder", "partial_blocks"):
        frozen_rates["encoder"] = _require_rate(
            rates.get("encoder"), "learning_rates.encoder"
        )
    batch = _require_mapping(training.get("batch"), "training.batch")
    micro = _require_count(batch.get("micro"), "batch.micro")
    accum = _require_count(batch.get("accumulation"), "batch.accumulation")
    devices = _require_count(batch.get("devices"), "batch.devices")
    total = _require_count(batch.get("global"), "batch.global")
    if (
        min(micro, accum, devices, total) < 1
        or micro * accum * devices != total
    ):
        raise ValueError("Batch geometry disagrees with global batch")
    hardware = _require_mapping(training.get("hardware"), "training.hardware")
    settings = _require_mapping(training.get("settings"), "training.settings")
    recipe = _require_mapping(training.get("recipe"), "training.recipe")
    groups = ["adapter", "head"]
    if train_mode != "adapters_only":
        groups.append("encoder")
    return {
        "settings": settings,
        "recipe": recipe,
        "trainable_groups": groups,
        "trainables": frozen_trainables,
        "learning_rates": frozen_rates,
        "schedule": _require_text(training.get("schedule"), "schedule"),
        "seed": _require_count(training.get("seed"), "seed"),
        "batch": {
            "micro": micro,
            "accumulation": accum,
            "devices": devices,
            "global": total,
        },
        "max_updates": _require_count(
            training.get("max_updates"), "max_updates"
        ),
        "eval_every_updates": _require_count(
            training.get("eval_every_updates"), "eval_every_updates"
        ),
        "hardware": {
            "devices": _require_text(
                hardware.get("devices"), "hardware.devices"
            ),
            "precision": _require_text(
                hardware.get("precision"), "hardware.precision"
            ),
        },
        "fitting_probe": _require_text(
            training.get("fitting_probe"), "fitting_probe receipt"
        ),
        "resource_check": _require_text(
            training.get("resource_check"), "resource_check receipt"
        ),
    }


def _freeze_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    views = _require_mapping(evaluation.get("views"), "evaluation.views")
    objective = _require_mapping(
        evaluation.get("selection_objective"), "selection_objective"
    )
    if objective.get("version") not in OBJECTIVE_SCHEMAS:
        raise ValueError("Unknown selection objective version")
    objectives = _require_mapping(
        evaluation.get("objectives"), "evaluation.objectives"
    )
    baseline = _require_mapping(objectives.get("baseline"), "baseline")
    for section in ("targets", "retention"):
        limits = _require_mapping(objectives.get(section), section)
        for view, view_limits in limits.items():
            # Type-checks limit keys against the shared gate vocabulary
            # using the frozen baseline population of that view.
            view_counts = _require_mapping(
                baseline.get(view), "baseline." + view
            )
            constraint_violations(
                view_counts, _require_mapping(view_limits, view)
            )
    expected = {"version", "protected", "view", "targets"}
    if objective["version"] == CHECKPOINT_SOURCED_SCHEMA:
        expected.add("per_source")
    if set(objective) != expected:
        raise ValueError("Selection objective fields are incomplete or unknown")
    protected = _require_mapping(objective["protected"], "objective.protected")
    if not views or set(protected) != set(views) or set(baseline) != set(views):
        raise ValueError(
            "Selection and baseline must cover every development view"
        )
    for name, limits in protected.items():
        constraint_violations(baseline[name], _require_mapping(limits, name))
    selected = objective["view"]
    if selected not in objectives["targets"]:
        raise ValueError("Selection must rank a declared target view")
    if objective["targets"] != objectives["targets"][selected]:
        raise ValueError("Trainer targets differ from the study target gates")
    for name, limits in objectives["retention"].items():
        if any(
            protected[name].get(key) != value for key, value in limits.items()
        ):
            raise ValueError(
                "Trainer protection differs from study retention gates"
            )
    families = objectives.get("families", {})
    if not isinstance(families, dict):
        raise ValueError("objectives.families must be an object")
    for name, table in families.items():
        if name not in views:
            raise ValueError("Family gate names an unknown view")
        for family, limits in _require_mapping(
            table, "families." + name
        ).items():
            _require_text(family, "family name")
            limits = _require_mapping(limits, family)
            # Family support may be smaller than the enclosing view. Check
            # its declared size against actual family rows at assessment.
            if "rows" in limits:
                _require_count(limits["rows"], family + ".rows")
            constraint_violations(
                baseline[name], {k: v for k, v in limits.items() if k != "rows"}
            )
    policy = _require_mapping(
        evaluation.get("selection_policy"), "selection_policy"
    )
    return {
        "views": views,
        "views_sha256": _sha_text(_canonical(views)),
        "selection_objective": objective,
        "selection_policy_sha256": policy_digest(policy),
        "objectives": {
            "baseline": baseline,
            "targets": objectives["targets"],
            "retention": objectives["retention"],
            "families": families,
        },
    }


def _freeze_procedure(procedure: Any) -> dict[str, Any]:
    """Bind optional Harness² guidance to the study.

    Absent procedure learning stays a plain disabled flag. When
    present, the bundle must validate for the named model now
    (digest, qualification, guidance), and the queue directory
    must exist; the supervisor replays this check every run.
    Guidance never changes policy, tools, or authority.
    """
    if procedure is None:
        return {"enabled": False}
    procedure = _require_mapping(procedure, "procedure")
    model = _require_text(procedure.get("model"), "procedure.model")
    bundle_path = _require_text(procedure.get("bundle"), "procedure.bundle")
    bundle = read_bundle(Path(bundle_path), model)
    queue = Path(_require_text(procedure.get("queue"), "procedure.queue"))
    if not queue.is_dir():
        raise ValueError("Procedure queue is not a directory")
    return {
        "enabled": True,
        "model": model,
        "bundle": {
            "path": str(Path(bundle_path).resolve()),
            "digest": bundle["digest"],
        },
        "queue": str(queue.resolve()),
    }


def _freeze_budget(budget: dict[str, Any]) -> dict[str, Any]:
    max_rounds = _require_count(budget.get("max_rounds"), "max_rounds")
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least one")
    return {
        "max_rounds": max_rounds,
        "max_hours": _require_rate(budget.get("max_hours"), "max_hours"),
    }


def verify_study_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Re-check every frozen byte and acceptance behind a manifest.

    Raises on any drift: changed parent, code, splits, pools,
    contract, dataset acceptance loss, or digest mismatch.
    Reserved splits stay hash-bound and are never opened.
    """
    manifest = _require_mapping(manifest, "Study manifest")
    if manifest.get("schema") != STUDY_MANIFEST_SCHEMA:
        raise ValueError("Unknown study manifest schema")
    digest = manifest.get("manifest_sha256")
    unsigned = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    try:
        canonical = _canonical(unsigned)
    except (TypeError, ValueError) as exc:
        raise ValueError("Study manifest is not JSON-canonical") from exc
    if digest != _sha_text(canonical):
        raise ValueError("Study manifest digest mismatch")
    parent = manifest["parent"]
    _check_tree(Path(parent["path"]), parent["files"], "Parent export")
    dataset = manifest["dataset"]
    root = Path(dataset["root"])
    version = dataset_versions.verify_version(root, dataset["version"])
    if _sha_text(_canonical(version)) != dataset["version_sha256"]:
        raise ValueError("Dataset version identity changed")
    accepted_path = root / "versions" / dataset["version"] / "accepted.json"
    if _sha_file(accepted_path) != dataset["acceptance_sha256"]:
        raise ValueError("Dataset acceptance identity changed")
    if not dataset_versions.is_accepted(root, dataset["version"]):
        raise ValueError("Dataset version lost acceptance")
    for name in ("train", "dev"):
        entry = dataset["splits"][name]
        if _sha_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Study split changed: " + name)
    pools = dataset["pools"]
    dataset_versions.verify_pools(
        Path(dataset["splits"]["train"]["path"]), Path(pools["path"])
    )
    if _sha_file(Path(pools["path"])) != pools["sha256"]:
        raise ValueError("Study pools changed")
    records = validate_ledger(Path(dataset["review_ledger"]["path"]))
    if ledger_digest(records) != dataset["review_ledger"]["digest"]:
        raise ValueError("Study review ledger changed")
    code = manifest["code"]
    _check_tree(Path(code["path"]), code["files"], "Study code tree")
    if _sha_file(Path(code["python"]["path"])) != code["python"]["sha256"]:
        raise ValueError("Python launcher changed")
    _check_tree(
        Path(code["runtime"]["path"]), code["runtime"]["files"], "Runtime"
    )
    _check_tree(
        Path(code["tokenizer"]["path"]), code["tokenizer"]["files"], "Tokenizer"
    )
    if (
        _sha_file(Path(code["question_contract_path"]))
        != code["question_contract_sha256"]
    ):
        raise ValueError("Question contract changed since freezing")
    if _freeze_parent(parent) != parent:
        raise ValueError("Parent provenance changed since freezing")
    procedure = manifest.get("procedure", {"enabled": False})
    if procedure.get("enabled"):
        bundle = read_bundle(
            Path(procedure["bundle"]["path"]), procedure["model"]
        )
        if bundle["digest"] != procedure["bundle"]["digest"]:
            raise ValueError("Study procedure bundle changed")
        if not Path(procedure["queue"]).is_dir():
            raise ValueError("Study procedure queue is missing")
    return {"study_id": manifest["study_id"], "verified": True}


def qualify_fitting_probe(report: dict[str, Any]) -> dict[str, Any]:
    """Judge a small fitting probe before larger runs are allowed.

    The report carries per-group parameter movement plus train-fit
    numbers from a short probe run. Pass requires movement in
    every declared trainable group, none in frozen groups, and a
    strict TRAIN-fit improvement. Reports describe fitting only;
    they never qualify development performance.
    """
    report = _require_mapping(report, "Fitting probe report")
    declared = report.get("trainable_groups")
    if not isinstance(declared, list) or not declared:
        raise ValueError("Probe needs non-empty trainable_groups")
    movement = _require_mapping(report.get("movement"), "movement")
    for group in declared:
        delta = movement.get(group)
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(delta)
            or delta <= 0
        ):
            raise ValueError(
                "No probe movement in trainable group: " + repr(group)
            )
    for group, delta in movement.items():
        if group not in declared and delta != 0:
            raise ValueError("Probe moved a frozen group: " + repr(group))
    fit = _require_mapping(report.get("train_fit"), "train_fit")
    before = _require_count(fit.get("correct_before"), "correct_before")
    after = _require_count(fit.get("correct_after"), "correct_after")
    total = _require_count(fit.get("rows"), "train_fit.rows")
    if not 0 <= before < after <= total:
        raise ValueError("Probe did not improve TRAIN fit")
    return {
        "qualified": True,
        "trainable_groups": list(declared),
        "gain": after - before,
    }


def check_resources(
    requirement: dict[str, Any], available: dict[str, Any]
) -> dict[str, Any]:
    """Refuse a recipe whose resource claim exceeds what is offered."""
    requirement = _require_mapping(requirement, "Resource requirement")
    available = _require_mapping(available, "Available resources")
    for key in ("gpus", "memory_gb", "hours"):
        need = requirement.get(key)
        have = available.get(key)
        if not isinstance(need, (int, float)) or not isinstance(
            have, (int, float)
        ):
            raise ValueError("Resource key needs numbers: " + key)
        if (
            isinstance(need, bool)
            or isinstance(have, bool)
            or not math.isfinite(need)
            or not math.isfinite(have)
            or need < 0
            or have < 0
        ):
            raise ValueError(
                "Resource values must be finite and nonnegative: " + key
            )
        if key == "gpus" and (type(need) is not int or type(have) is not int):
            raise ValueError("GPU counts must be integers")
        if need > have:
            raise ValueError(
                "Recipe needs %s %s but only %s is available"
                % (need, key, have)
            )
    return {"fits": True}


def assess_attempt(
    manifest: dict[str, Any],
    train_report: dict[str, Any],
    dev_results: dict[str, Any],
    exposure: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one finished attempt into diagnosable evidence.

    Target and retention verdicts reuse the frozen objective
    limits; exposure reuses the delivered-exposure gate. Label
    audits stay advisory: their absence never blocks a verdict.
    """
    manifest = _require_mapping(manifest, "Study manifest")
    objectives = manifest["evaluation"]["objectives"]
    train_report = _require_mapping(train_report, "Train report")
    dev_results = _require_mapping(dev_results, "Dev results")
    job = _require_mapping(job, "Job record")
    target_failures: dict[str, list[str]] = {}
    for view, limits in objectives["targets"].items():
        counts = _require_mapping(dev_results.get(view), "results." + view)
        failed = constraint_violations(counts, limits)
        if failed:
            target_failures[view] = failed
    retention_failures: dict[str, list[str]] = {}
    for view, limits in objectives["retention"].items():
        counts = _require_mapping(dev_results.get(view), "results." + view)
        failed = constraint_violations(counts, limits)
        if failed:
            retention_failures[view] = failed
    family_failures = {}
    for view, families in objectives.get("families", {}).items():
        source = _require_mapping(dev_results.get(view), "results." + view)
        table = _require_mapping(source.get("families"), "results.families")
        for family, limits in families.items():
            counts = _require_mapping(table.get(family), "family." + family)
            failures = constraint_violations(counts, limits)
            if failures:
                family_failures[view + "/" + family] = failures
    target_failures.update(family_failures)
    fit = _require_mapping(train_report.get("fit"), "train_report.fit")
    correct = _require_count(fit.get("correct"), "train correct")
    total = _require_count(fit.get("rows"), "train rows")
    if not 0 <= correct <= total or total < 1:
        raise ValueError("TRAIN-fit counts are inconsistent")
    return {
        "job_complete": job.get("complete") is True
        and job.get("report_present") is True,
        "sdk_parity_ok": job.get("sdk_parity_ok"),
        "exposure_complete": exposure_gate_passed(exposure),
        "train_fit_ok": fit.get("meets_probe_criterion") is True,
        "train_correct": _require_count(fit.get("correct"), "train correct"),
        "train_rows": _require_count(fit.get("rows"), "train rows"),
        "target_met": not target_failures,
        "target_failures": target_failures,
        "retention_ok": not retention_failures,
        "retention_failures": retention_failures,
        "label_audit_ok": train_report.get("label_audit_ok"),
    }


def diagnose(evidence: dict[str, Any]) -> dict[str, Any]:
    """Classify why an attempt missed its objectives.

    Precedence is fixed: execution defects first, then exposure,
    then fit, then labels versus generalization, with retention
    trade-offs last. Failed execution or incomplete training is
    never reported as a modeling verdict. An attempt that met
    every objective has no failure and is refused.
    """
    evidence = _require_mapping(evidence, "Attempt evidence")
    reasons: list[str] = []
    if not evidence.get("job_complete"):
        reasons.append("job incomplete or report missing")
        category, advice = "execution_defect", "fix execution"
    elif evidence.get("sdk_parity_ok") is not True:
        reasons.append("SDK parity is missing or failed")
        category, advice = "execution_defect", "fix execution"
    elif not evidence.get("exposure_complete"):
        reasons.append("required pools never completed a pass")
        category, advice = "insufficient_exposure", "repeat with full exposure"
    elif not evidence.get("train_fit_ok"):
        reasons.append(
            "TRAIN fit %s/%s below probe criterion"
            % (evidence.get("train_correct"), evidence.get("train_rows"))
        )
        category, advice = "failure_to_fit", "change objective or capacity"
    elif evidence.get("label_audit_ok") is False:
        reasons.append("label audit flags verification failures")
        category, advice = "poor_labels", "repair data and reviews"
    elif evidence.get("target_met") and not evidence.get("retention_ok"):
        failed = sorted(evidence.get("retention_failures", {}))
        reasons.append(
            "target passed while protected views failed: %s" % failed
        )
        category, advice = "retention_tradeoff", "add preservation or replay"
    elif not evidence.get("target_met"):
        failed = sorted(evidence.get("target_failures", {}))
        reasons.append("TRAIN fits but frozen targets failed: %s" % failed)
        category, advice = (
            "poor_generalization",
            "broaden independent training families",
        )
    else:
        raise ValueError("Attempt met its objectives; nothing to diagnose")
    return _verdict(category, reasons, advice)


def _verdict(category: str, reasons: list[str], advice: str) -> dict[str, Any]:
    if category not in DIAGNOSES:
        raise ValueError("Unknown diagnosis")
    return {"category": category, "reasons": reasons, "next": advice}


def policy_digest(policy: dict[str, Any]) -> str:
    """Canonical digest binding a campaign selection policy."""
    return _sha_text(_canonical(_require_mapping(policy, "Policy")))


def report_counts(
    dev_report: dict[str, Any], dev_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Recompute every count from aligned predictions, refusing contradictions."""
    dev_report = _require_mapping(dev_report, "Development report")
    predictions = dev_report.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(dev_rows):
        raise ValueError("Development predictions do not align with rows")
    if any(type(p) is not int or p not in range(3) for p in predictions):
        raise ValueError(
            "Development predictions must be semantic label indices"
        )
    ids = [row["id"] for row in dev_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Development row IDs must be unique")
    if dev_report.get("row_ids") is not None and dev_report["row_ids"] != ids:
        raise ValueError("Development row order differs from frozen population")
    matrix = [[0, 0, 0] for _ in LABELS]
    for row, prediction in zip(dev_rows, predictions):
        matrix[LABELS.index(row["expected"]["security"])][prediction] += 1
    if dev_report.get("confusion") != matrix:
        raise ValueError("Development confusion disagrees with row predictions")
    triplets = triplet_counts(dev_rows, predictions, list(LABELS))
    pairs: dict[str, list[bool]] = {}
    for row, prediction in zip(dev_rows, predictions):
        expected = row["expected"]["security"]
        if expected != "insufficient_context":
            pairs.setdefault(row["pair_id"], []).append(
                prediction == LABELS.index(expected)
            )
    total_pairs = len(pairs)
    pair_correct = sum(all(hits) and len(hits) == 2 for hits in pairs.values())
    reported_pairs = dev_report.get("complete_pairs")
    # Legacy reporters may omit pair scoring. Never omit count verification.
    if reported_pairs is not None:
        if reported_pairs != total_pairs or not math.isclose(
            dev_report.get("both_pair_members_correct", -1),
            pair_correct / total_pairs,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Development pair summary disagrees with predictions"
            )
    view = dict(
        dev_report,
        complete_pairs=total_pairs,
        both_pair_members_correct=pair_correct / total_pairs,
        **triplets,
    )
    result = view_counts(view)
    result["pairs"] = total_pairs
    result["pairs_correct"] = pair_correct
    return result


def read_terminal_exposure(events_path: Path) -> dict[str, Any]:
    """Read the last delivered-exposure record from trainer events.

    Only runs launched with full-exposure plateau tracking record
    it; anything else is refused as unqualified rather than
    guessed, since exposure claims must come from the run.
    """
    last: dict[str, Any] | None = None
    for line in Path(events_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        # The trainer nests exposure under the eval metrics; accept
        # either placement so older flat records still read.
        candidates = [event.get("delivered_exposure")]
        metrics = event.get("metrics")
        if isinstance(metrics, dict):
            candidates.append(metrics.get("delivered_exposure"))
        for candidate in candidates:
            if isinstance(candidate, dict):
                last = candidate
    if last is None:
        raise ValueError(
            "Run recorded no delivered exposure; relaunch with "
            "full-exposure plateau tracking"
        )
    for key in ("unique_rows", "total_rows"):
        _require_count(last.get(key), "exposure." + key)
    for key in ("broken_triplets", "split_triplets"):
        if not isinstance(last.get(key), list):
            raise ValueError("exposure." + key + " must be a list")
    if last["total_rows"] < 1 or last["unique_rows"] > last["total_rows"]:
        raise ValueError("Exposure counts are inconsistent")
    return last


def assess_round(
    manifest: dict[str, Any],
    report: dict[str, Any],
    events_path: Path,
    train_fit: dict[str, Any],
    sdk_parity_ok: bool | None = None,
    *,
    expected_run_id: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Assess bound selected weights across all frozen development views.

    Callers supply the identity from the owned job and its verified export.
    Row alignment is derived from the frozen development file and selectors.
    A fit probe is diagnostic only. It never replaces SDK parity or dev gates.
    """
    verify_study_manifest(manifest)
    if report.get("run_id") != expected_run_id:
        raise ValueError("Trainer report belongs to another run")
    if (
        report.get("selected_checkpoint", {}).get("weights_sha256")
        != checkpoint_sha256
    ):
        raise ValueError("Trainer report belongs to another checkpoint")
    for flag in ("test_evaluated", "calibration_fitted", "final_evaluation"):
        if report.get(flag) is not False:
            raise ValueError(
                "Assessment requires an explicitly development-only report"
            )
    dev_rows = [
        json.loads(line)
        for line in Path(manifest["dataset"]["splits"]["dev"]["path"])
        .read_text()
        .splitlines()
    ]
    development = _require_mapping(report.get("development"), "development")
    # Validate the aggregate first. It binds predictions to the exact row order.
    report_counts(development, dev_rows)
    definitions = manifest["evaluation"]["views"]
    memberships = {
        name: population_indices(dev_rows, definition)
        for name, definition in definitions.items()
    }
    flat = [i for members in memberships.values() for i in members]
    if sorted(flat) != list(range(len(dev_rows))) or any(
        not x for x in memberships.values()
    ):
        raise ValueError(
            "Development views must partition the frozen rows exactly"
        )
    reported_views = development.get("development_views", {})
    results = {}
    for name, indices in memberships.items():
        rows = [dev_rows[i] for i in indices]
        raw = development if len(memberships) == 1 else reported_views.get(name)
        raw = _require_mapping(raw, "development view " + name)
        if raw.get("predictions") != [
            development["predictions"][i] for i in indices
        ]:
            raise ValueError(
                "View predictions disagree with aggregate: " + name
            )
        counts = report_counts(raw, rows)
        family_table = {}
        for family, limits in (
            manifest["evaluation"]["objectives"]
            .get("families", {})
            .get(name, {})
            .items()
        ):
            members = [
                i
                for i, row in enumerate(rows)
                if row.get("mechanism_id") == family
            ]
            if not members:
                raise ValueError("Missing registered family: " + family)
            subset = [rows[i] for i in members]
            predicted = [raw["predictions"][i] for i in members]
            family_table[family] = counts_from_predictions(subset, predicted)
            constraint_violations(family_table[family], limits)
        counts["families"] = family_table
        results[name] = counts
    exposure = read_terminal_exposure(events_path)
    train_rows = sum(
        1 for _ in Path(manifest["dataset"]["splits"]["train"]["path"]).open()
    )
    if exposure["total_rows"] != train_rows:
        raise ValueError("Exposure belongs to another training population")
    job = {
        "complete": report.get("stop_reason")
        in (
            "update_budget",
            "max_updates",
            "epochs",
            "plateau",
            "patience",
            "completed",
            "epoch_budget",
            "development_plateau",
        ),
        "report_present": True,
        "sdk_parity_ok": sdk_parity_ok,
    }
    return assess_attempt(manifest, {"fit": train_fit}, results, exposure, job)


def counts_from_predictions(
    rows: list[dict[str, Any]], predictions: list[int]
) -> dict[str, Any]:
    """Derive population and family gate counts from semantic predictions."""
    if len(rows) != len(predictions) or not rows:
        raise ValueError("Counts need nonempty aligned predictions")
    matrix = [[0, 0, 0] for _ in LABELS]
    for row, predicted in zip(rows, predictions):
        if type(predicted) is not int or predicted not in range(3):
            raise ValueError("Invalid prediction index")
        matrix[LABELS.index(row["expected"]["security"])][predicted] += 1
    support = [sum(row) for row in matrix]
    correct = [matrix[i][i] for i in range(3)]
    triplets = triplet_counts(rows, predictions, list(LABELS))
    pairs: dict[str, list[bool]] = {}
    for row, predicted in zip(rows, predictions):
        if row["expected"]["security"] != "insufficient_context":
            pairs.setdefault(row["pair_id"], []).append(
                predicted == LABELS.index(row["expected"]["security"])
            )
    return {
        "rows": len(rows),
        "correct": sum(correct),
        "class_support": support,
        "class_correct": correct,
        "false_safe": matrix[1][0],
        "false_violation": matrix[0][1],
        "unnecessary_abstention": matrix[0][2] + matrix[1][2],
        "pairs_correct": sum(all(x) and len(x) == 2 for x in pairs.values()),
        **triplets,
    }


def validate_training_settings(
    training: dict[str, Any],
    code: dict[str, Any],
    dataset: dict[str, Any],
    parent: dict[str, Any],
) -> None:
    """Ensure descriptive fields mean the same thing as effective CLI values."""
    settings = training["settings"]
    expected = {
        "adapter_lr": training["learning_rates"]["adapter"],
        "head_lr": training["learning_rates"]["head"],
        "micro_batch": training["batch"]["micro"],
        "accumulation": training["batch"]["accumulation"],
        "seed": training["seed"],
        "lr_schedule": training["schedule"],
        "max_updates": training["max_updates"],
        "eval_every_updates": training["eval_every_updates"],
        "question_contract": code["question_contract"],
        "plateau_requires_full_exposure": True,
        "sampling": "grouped",
    }
    if training["max_updates"] < 1 or training["eval_every_updates"] < 1:
        raise ValueError("Study needs positive update and evaluation budgets")
    mode = training["trainables"]["mode"]
    if mode != "adapters_only":
        expected["encoder_lr"] = training["learning_rates"]["encoder"]
    if bool(settings.get("freeze_encoder")) != (mode == "adapters_only"):
        raise ValueError("Encoder freezing differs from declared trainables")
    if settings.get("disable_adapters"):
        raise ValueError("This study mode requires enabled residual adapters")
    if mode == "partial_blocks":
        config = json.loads(
            (Path(parent["path"]) / "encoder/config.json").read_text()
        )
        layers = _require_count(
            config.get("num_hidden_layers"), "encoder layers"
        )
        blocks = training["trainables"]["blocks"]
        if blocks != list(range(layers - len(blocks), layers)):
            raise ValueError(
                "The trainer supports a contiguous final block suffix only"
            )
        expected["encoder_trainable_blocks"] = len(blocks)
    elif settings.get("encoder_trainable_blocks", 0) != 0:
        raise ValueError("Unexpected partial encoder setting")
    for name, value in expected.items():
        if settings.get(name) != value:
            raise ValueError(
                "Training description disagrees with CLI setting: " + name
            )
    if settings.get("pools") != dataset["pools"]["path"]:
        raise ValueError("Training settings name different sampling pools")
    if settings.get("review_ledger") != dataset["review_ledger"]["path"]:
        raise ValueError("Training settings name a different review ledger")
    if training["recipe"].get("training") != settings:
        raise ValueError("Recipe training settings differ from the study")
