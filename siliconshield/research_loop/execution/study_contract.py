"""Bind owned jobs to immutable decision studies and reserve study budgets."""

import fcntl
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from siliconshield.research_loop.adapters.laya_training import (
    decision_study,
    development_views,
)
from siliconshield.research_loop.episodes.contracts import atomic_json
from siliconshield.research_loop.execution.security_jobs import command_for


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(profile: dict[str, Any]) -> dict[str, Any] | None:
    """Verify the explicitly selected study, preserving legacy job behavior."""
    path = profile.get("decision_study_manifest")
    if path is None:
        if profile.get("decision_study_sha256") is not None:
            raise ValueError("Study digest needs a manifest path")
        return None
    manifest = json.loads(Path(path).read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Study manifest must be an object")
    if manifest.get("manifest_sha256") != profile.get("decision_study_sha256"):
        raise ValueError("Job study identity differs from the registered study")
    decision_study.verify_study_manifest(manifest)
    return manifest


def _same_file(actual: Any, expected: dict[str, Any], name: str) -> None:
    if not isinstance(actual, str) or _sha(Path(actual)) != expected["sha256"]:
        raise ValueError("Job differs from study " + name)


def verify_job(
    profile: dict[str, Any], experiment_id: str, overrides: dict[str, Any]
) -> dict[str, Any] | None:
    """Check effective settings, actual inputs, and argv before any submission."""
    manifest = load(profile)
    if manifest is None:
        return None
    parent = manifest["parent"]
    if Path(profile["model"]).resolve() != Path(parent["path"]):
        raise ValueError("Job parent differs from frozen study")
    code = manifest["code"]
    for setting, expected in (
        ("source", code["path"]),
        ("laya_root", code["runtime"]["path"]),
    ):
        if Path(profile.get(setting, "")).resolve() != Path(expected):
            raise ValueError("Job " + setting + " differs from frozen study")
    if _sha(Path(profile["python"])) != code["python"]["sha256"]:
        raise ValueError("Job Python launcher differs from frozen study")
    if profile.get("image_digest") != code["image_digest"]:
        raise ValueError("Job image identity differs from frozen study")
    for role in ("train", "dev"):
        _same_file(
            str(Path(profile["data"]) / (role + ".jsonl")),
            manifest["dataset"]["splits"][role],
            role,
        )
    values = {**profile.get("defaults", {}), **overrides}
    if values != manifest["training"]["settings"]:
        raise ValueError("Effective training settings differ from frozen study")
    _same_file(values.get("pools"), manifest["dataset"]["pools"], "pools")
    _same_file(
        values.get("review_ledger"),
        {"sha256": manifest["dataset"]["review_ledger"]["sha256"]},
        "review ledger",
    )
    views_path = values.get("development_views")
    if not isinstance(views_path, str):
        raise ValueError("Job lacks frozen development views")
    views = json.loads(Path(views_path).read_text())
    if views.get("views") != manifest["evaluation"]["views"]:
        raise ValueError("Job development populations differ from study")
    if (
        views.get("checkpoint_objective")
        != manifest["evaluation"]["selection_objective"]
    ):
        raise ValueError("Trainer selection objective differs from study")
    if views.get("selection_view") != views["checkpoint_objective"]["view"]:
        raise ValueError(
            "Trainer selection_view is missing or differs from objective"
        )
    if views.get("score_triplets") is not True:
        raise ValueError("Study views must enable whole-triplet scoring")
    rows = [
        json.loads(line)
        for line in (Path(profile["data"]) / "dev.jsonl")
        .read_text()
        .splitlines()
    ]
    try:
        development_views.attach_views(
            {}, rows, [None] * len(rows), views, lambda _rows, _logits: {}
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Incomplete trainer development-view configuration"
        ) from exc
    recipe = json.loads(Path(profile["recipe"]).read_text())
    # A campaign adds a separately validated incumbent selection reference.
    recipe.pop("selection_reference", None)
    if recipe != manifest["training"]["recipe"]:
        raise ValueError("Job recipe differs from the qualified study recipe")
    gpus = profile.get("gpus", [0, 1, 2])
    if len(gpus) != manifest["training"]["batch"]["devices"]:
        raise ValueError("Job GPU count differs from study")
    timeout = profile.get("timeout_seconds")
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or timeout <= 0
        or timeout > manifest["budget"]["max_hours"] * 3600
    ):
        raise ValueError("Job timeout exceeds the study time budget")
    command, output = command_for(profile, experiment_id, overrides)
    replacements = {"{experiment_id}": experiment_id, "{output}": str(output)}
    expected = [
        replacements.get(part, part) for part in manifest["launch_command"]
    ]
    if command != expected:
        raise ValueError("Launch command differs from frozen study")
    return {
        "path": str(Path(profile["decision_study_manifest"]).resolve()),
        "sha256": manifest["manifest_sha256"],
        "study_id": manifest["study_id"],
    }


def verify_receipts(
    manifest: dict[str, Any], probe: dict[str, Any], resources: dict[str, Any]
) -> None:
    """Accept qualification only for the exact parent, data, and recipe."""
    identity = manifest["manifest_sha256"]
    for name, receipt in (("probe", probe), ("resource", resources)):
        if receipt.get("study_sha256") != identity:
            raise ValueError(name + " receipt belongs to another study")
        if receipt.get("parent_sha256") != manifest["parent"]["weights_sha256"]:
            raise ValueError(name + " receipt belongs to another parent")
        if (
            receipt.get("train_sha256")
            != manifest["dataset"]["splits"]["train"]["sha256"]
        ):
            raise ValueError(
                name + " receipt belongs to another training population"
            )
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError(name + " receipt needs measured artifacts")
        for path, digest in artifacts.items():
            if _sha(Path(path)) != digest:
                raise ValueError(name + " receipt artifact changed")
    declared = set(manifest["training"]["trainable_groups"])
    if set(probe.get("trainable_groups", [])) != declared:
        raise ValueError("Probe did not qualify the declared trainable groups")
    if set(probe.get("movement", {})) != {"encoder", "adapter", "head"}:
        raise ValueError("Probe must measure every parameter group")
    decision_study.qualify_fitting_probe(probe)
    decision_study.check_resources(
        resources["requirement"], resources["available"]
    )
    need = resources["requirement"]
    if need["gpus"] != manifest["training"]["batch"]["devices"]:
        raise ValueError("Resource receipt covers another GPU count")
    if need["memory_gb"] <= 0 or need["hours"] <= 0:
        raise ValueError(
            "Resource receipt needs positive measured requirements"
        )
    if need["hours"] > manifest["budget"]["max_hours"]:
        raise ValueError("Resource requirement exceeds the frozen study budget")


def reserve_budget(profile: dict[str, Any], job: dict[str, Any]) -> None:
    """Reserve worst-case job time atomically, including retries and stop grace.

    Reservations are durable and conservative. They are not reclaimed on a
    crash or failed launch. This prevents retries from exceeding a study cap.
    """
    manifest = load(profile)
    if manifest is None:
        return
    base = Path(profile["decision_study_manifest"]).resolve().parent
    identity = manifest["manifest_sha256"]
    key = job["idempotency_key"] + ":" + str(job["attempts"] + 1)
    charge = float(job["timeout_seconds"]) + 40.0
    with (base / "budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = base / "budget.json"
        record = (
            json.loads(path.read_text())
            if path.exists()
            else {"study_sha256": identity, "reservations": {}}
        )
        if record.get("study_sha256") != identity:
            raise ValueError("Study budget belongs to another manifest")
        entries = record["reservations"]
        if key in entries:
            if entries[key]["seconds"] != charge:
                raise ValueError("Study reservation changed on retry")
            return
        rounds = {entry["experiment_id"] for entry in entries.values()}
        if (
            job["experiment_id"] not in rounds
            and len(rounds) >= manifest["budget"]["max_rounds"]
        ):
            raise ValueError("Study round budget exhausted")
        if (
            sum(entry["seconds"] for entry in entries.values()) + charge
            > manifest["budget"]["max_hours"] * 3600
        ):
            raise ValueError("Study time budget exhausted")
        entries[key] = {
            "experiment_id": job["experiment_id"],
            "seconds": charge,
        }
        atomic_json(path, record)
