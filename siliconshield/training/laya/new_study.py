"""Verify an explicit new study initialized from an evaluated, immutable parent."""

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from siliconshield.schemas import NEW_STUDY_SCHEMA


def load_readonly_checkpoint(path, loader):
    """Let SDK compatibility repairs touch copied metadata, never the parent.

    Tensor files are read through symlinks; small configuration/tokenizer files
    are copied because the SDK may rewrite them while loading. The loaded model
    and tokenizer own their in-memory contents when the temporary tree is removed.
    """
    parent = Path(path).resolve()

    def copy_file(source, destination):
        if Path(source).suffix in (".safetensors", ".bin"):
            Path(destination).symlink_to(Path(source).resolve())
            return destination
        return shutil.copy2(source, destination)

    with tempfile.TemporaryDirectory(prefix="laya-load-") as temporary:
        staged = Path(temporary) / "checkpoint"
        shutil.copytree(parent, staged, copy_function=copy_file)
        return loader(str(staged))


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_new_study(path, model, data, output, run_id):
    """Retire prior evaluation populations without erasing the parent's history.

    Split manifests and parent files are content-bound. New test rows cannot be
    recycled from a retired evaluation. Only hashes and IDs are inspected here;
    no test metrics or labels enter model preparation or checkpoint selection.
    """
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != NEW_STUDY_SCHEMA:
        raise ValueError("Unknown new-study schema")
    if (
        manifest.get("run_id") != run_id
        or manifest.get("retire_parent_evaluations") is not True
    ):
        raise ValueError(
            "New study must name this run and retire parent evaluations"
        )
    parent = Path(model).resolve()
    destination = Path(output).resolve()
    if (
        destination == parent
        or parent in destination.parents
        or destination in parent.parents
    ):
        raise ValueError("New study cannot overwrite the parent checkpoint")
    actual = {
        str(p.relative_to(parent)): file_hash(p)
        for p in sorted(parent.rglob("*"))
        if p.is_file()
    }
    if not actual or actual != manifest.get("parent_checkpoint"):
        raise ValueError("New-study parent checkpoint identity mismatch")
    splits = manifest.get("dataset", {})
    if set(splits) != {"train", "dev", "calibration", "test"}:
        raise ValueError("New study requires all four named data partitions")
    for split, expected in splits.items():
        if file_hash(Path(data) / (split + ".jsonl")) != expected:
            raise ValueError("New-study dataset identity mismatch: " + split)
    retired = manifest.get("retired_evaluations", [])
    if not retired:
        raise ValueError("Record the retired evaluation populations")
    retired_ids, retired_states, retired_groups = set(), set(), set()

    def state_key(row):
        return hashlib.sha256(
            re.sub(r"\s+", "", row["state"]).encode()
        ).hexdigest()

    for population in retired:
        source = Path(population["path"])
        if file_hash(source) != population["sha256"]:
            raise ValueError("Retired evaluation identity mismatch")
        for line in source.read_text().splitlines():
            row = json.loads(line)
            retired_ids.add(row["id"])
            retired_states.add(state_key(row))
            if row.get("split_group_id"):
                retired_groups.add(row["split_group_id"])
    groups, ids, states = {}, set(), {}
    for split in splits:
        for line in (Path(data) / (split + ".jsonl")).read_text().splitlines():
            row = json.loads(line)
            key = state_key(row)
            group = row.get("split_group_id")
            if (
                row["id"] in retired_ids
                or key in retired_states
                or group in retired_groups
            ):
                raise ValueError(
                    "Retired evaluation row reused in new study: " + split
                )
            if row["id"] in ids or states.setdefault(key, split) != split:
                raise ValueError("Duplicate new-study input or ID")
            ids.add(row["id"])
            if group and groups.setdefault(group, split) != split:
                raise ValueError("New-study lineage crosses partitions")
    return {
        "schema": manifest["schema"],
        "run_id": run_id,
        "manifest_sha256": file_hash(path),
        "parent_checkpoint": actual,
        "dataset": splits,
        "retired_evaluations": [
            {"sha256": r["sha256"], "role": r["role"]} for r in retired
        ],
        "retire_parent_evaluations": True,
        "scope": "Fresh study; ancestor evaluation results are historical, not fresh validation.",
    }
