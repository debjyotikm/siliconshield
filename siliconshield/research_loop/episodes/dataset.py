"""Build Glimmer tool-use training examples from completed development reports."""

import argparse
import dataclasses
import hashlib
import hmac
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Union, cast

from siliconshield.research_loop.episodes.contracts import (
    DevelopmentMetrics,
    DevelopmentSummary,
    JsonValue,
    atomic_json,
    digest,
    episode_from_summary,
    finite_number,
    read_json,
)
from siliconshield.research_loop.episodes.laya_results import (
    ADAPTER_SCHEMA,
    TRAINER_SCHEMA,
    adapt_report,
)
from siliconshield.schemas import CONTROLLER_EPISODES_SCHEMA


@dataclasses.dataclass
class _EpisodeCollection:
    """Accumulator threaded explicitly through the build stages.

    Keeps collection state out of module scope: every stage receives the
    collection and mutates only it. ``rows`` holds review rows,
    ``skipped`` counts rejections by reason, ``seen`` fingerprints visible
    evidence for deduplication, and ``source_files`` maps row IDs back to
    their local receipts for the manifest.
    """

    rows: list[dict[str, JsonValue]] = dataclasses.field(default_factory=list)
    skipped: Counter[str] = dataclasses.field(default_factory=Counter)
    seen: dict[str, str] = dataclasses.field(default_factory=dict)
    source_files: dict[str, dict[str, str]] = dataclasses.field(
        default_factory=dict
    )


def _checked_metrics(
    fit: Mapping[str, JsonValue], label: str
) -> DevelopmentMetrics:
    """Pick validated metrics out of one development checkpoint.

    Raises ValueError when the checkpoint is missing or lacks finite
    numbers; downstream review revalidates denominators before any
    comparison.
    """
    try:
        decisions = fit["decisions"]
        agreement = fit["agreement"]
        nll = fit["nll"]
    except KeyError as exc:
        raise ValueError("Development metrics lack " + str(exc)) from exc
    if (
        not isinstance(decisions, int)
        or isinstance(decisions, bool)
        or not finite_number(agreement)
        or not finite_number(nll)
    ):
        raise ValueError("Development metrics must be finite numbers")
    return {
        "decisions": decisions,
        "agreement": cast(float, agreement),
        "nll": cast(float, nll),
    }


def _collect_legacy_rows(
    runs: Union[str, os.PathLike[str]], collection: _EpisodeCollection
) -> None:
    """Collect development-triage episodes from legacy result.json receipts."""
    for path in sorted(Path(runs).glob("*/result.json")):
        try:
            raw = path.read_bytes()
            report = json.loads(raw)
            scope = str(report.get("scope", ""))
            if not (
                "open development" in scope
                or "no reserved evaluation/calibration" in scope
            ):
                collection.skipped["not_explicit_open_development"] += 1
                continue
            if report.get("status") != "complete":
                collection.skipped["not_complete"] += 1
                continue
            spec_path = path.parent / "spec.json"
            spec = read_json(spec_path) if spec_path.exists() else {}
            if not isinstance(spec, dict):
                raise ValueError("Development spec is not an object")
            pair = path.parent.name.rsplit("-", 1)[0]
            raw_package = spec.get("package", {})
            package = raw_package if isinstance(raw_package, dict) else {}
            group = str(package.get("sha256") or digest({"pair": pair}))
            # The model sees development metrics; source paths stay in the local manifest.
            summary: DevelopmentSummary = {
                "run_id": "run-" + digest({"source": str(path)})[:16],
                "split_role": "open_development",
                "phase": cast(str, spec.get("phase", "research")),
                "completed_updates": report["completed_updates"],
                "initial": _checked_metrics(
                    report.get("initial_fit"), "initial"
                ),
            }
            if "smoke" in path.parent.name:
                summary["phase"] = "smoke"
            provenance: dict[str, JsonValue] = {
                "kind": "checked_calculation",
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "scope": "authored triage policy applied to measured development receipts",
                "not_verified_research_optimality": True,
            }
            for checkpoint in ("selected", "final"):
                if checkpoint + "_fit" not in report:
                    continue
                reviewed: DevelopmentSummary = {
                    **summary,
                    "run_id": "run-"
                    + digest({"source": str(path), "part": checkpoint})[:16],
                    "selected": _checked_metrics(
                        report.get(checkpoint + "_fit"), checkpoint
                    ),
                }
                fingerprint = digest(
                    {
                        k: v
                        for k, v in reviewed.items()
                        if k not in ("run_id", "reviewed_checkpoint")
                    }
                )
                if fingerprint in collection.seen:
                    collection.skipped["identical_visible_evidence"] += 1
                    continue
                row = episode_from_summary(reviewed, group, provenance)
                collection.rows.append(row)
                row_id = row.get("id")
                if not isinstance(row_id, str):
                    raise ValueError("Built episode is missing its id")
                collection.seen[fingerprint] = row_id
                collection.source_files[row_id] = {
                    "result": str(path),
                    "spec": str(spec_path),
                    "reviewed_checkpoint": checkpoint,
                }
        except (KeyError, ValueError, TypeError, OSError) as exc:
            collection.skipped[type(exc).__name__] += 1


def _collect_laya_rows(
    runs: Union[str, os.PathLike[str]], collection: _EpisodeCollection
) -> None:
    """Collect development-triage episodes from LAYA trainer reports."""
    # LAYA security-trainer results (e.g. job-bridge run dirs); the trainer
    # report version is shared with laya_results.TRAINER_SCHEMA.
    for path in sorted(Path(runs).glob("*/report.json")):
        try:
            raw = path.read_bytes()
            report = json.loads(raw)
            if report.get("schema") != TRAINER_SCHEMA:
                collection.skipped["not_laya_security_report"] += 1
                continue
            summary, provenance, group_seed = adapt_report(
                report, source_sha256=hashlib.sha256(raw).hexdigest()
            )
            group = digest(group_seed)
            fingerprint = digest(
                {k: v for k, v in summary.items() if k != "run_id"}
            )
            if fingerprint in collection.seen:
                collection.skipped["identical_visible_evidence"] += 1
                continue
            row = episode_from_summary(summary, group, provenance)
            collection.rows.append(row)
            row_id = row.get("id")
            if not isinstance(row_id, str):
                raise ValueError("Built episode is missing its id")
            collection.seen[fingerprint] = row_id
            collection.source_files[row_id] = {
                "report": str(path),
                "adapter": ADAPTER_SCHEMA,
            }
        except (KeyError, ValueError, TypeError, OSError) as exc:
            collection.skipped[type(exc).__name__] += 1


def _partition_groups(groups: Sequence[str]) -> dict[str, str]:
    """Map each lineage group to a split, holding out test/dev first.

    Raises ValueError when fewer than three independent groups exist;
    with fewer, no split can isolate a test lineage.
    """
    if len(groups) < 3:
        raise ValueError(
            "At least three independent experiment groups are required"
        )
    n_eval = max(1, len(groups) // 10)
    return {
        g: ("test" if i < n_eval else "dev" if i < 2 * n_eval else "train")
        for i, g in enumerate(groups)
    }


def _write_splits(
    rows: Sequence[Mapping[str, JsonValue]],
    partition: Mapping[str, str],
    output: Union[str, os.PathLike[str]],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Write one JSONL per split; returns row counts and action counts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    actions: dict[str, dict[str, int]] = {}
    for split in ("train", "dev", "test"):
        subset: list[Mapping[str, JsonValue]] = []
        for r in rows:
            group_id = r.get("group_id")
            if not isinstance(group_id, str):
                raise ValueError("Episode is missing its lineage group")
            if group_id not in partition:
                raise ValueError(
                    "Episode lineage group has no split: " + group_id
                )
            if partition[group_id] != split:
                continue
            subset.append(r)
        with (output / (split + ".jsonl")).open("x") as stream:
            for row in subset:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        counts[split] = len(subset)
        action_counts: dict[str, int] = {}
        for row in subset:
            target = row.get("target")
            action = target.get("action") if isinstance(target, dict) else None
            if not isinstance(action, str):
                raise ValueError("Episode is missing its target action")
            action_counts[action] = action_counts.get(action, 0) + 1
        actions[split] = action_counts
    return counts, actions


def build(
    runs: Union[str, os.PathLike[str]], output: Union[str, os.PathLike[str]]
) -> dict[str, JsonValue]:
    """Build Glimmer tool-use training episodes from development receipts.

    Collects legacy and LAYA reports, deduplicates identical visible
    evidence, partitions lineage groups into splits, and writes the splits
    plus their manifest. Raises ValueError when nothing eligible remains.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    collection = _EpisodeCollection()
    _collect_legacy_rows(runs, collection)
    _collect_laya_rows(runs, collection)
    if not collection.rows:
        raise ValueError("No eligible, valid development receipts")
    group_ids: list[str] = []
    for row in collection.rows:
        row_group = row.get("group_id")
        if not isinstance(row_group, str):
            raise ValueError("Episode is missing its lineage group")
        group_ids.append(row_group)
    groups = sorted(
        set(group_ids),
        key=lambda g: digest({"split_seed": 41, "group": g}),
    )
    partition = _partition_groups(groups)
    counts, actions = _write_splits(collection.rows, partition, output)
    manifest: dict[str, JsonValue] = {
        "schema": CONTROLLER_EPISODES_SCHEMA,
        "rows": counts,
        "groups": {g: partition[g] for g in sorted(partition)},
        "lineage_groups": len(groups),
        "split_rule": "package lineage; both arms and seeds together",
        "actions": actions,
        "skipped": dict(collection.skipped),
        "sources": collection.source_files,
        "files": {
            s: hashlib.sha256(
                (output / (s + ".jsonl")).read_bytes()
            ).hexdigest()
            for s in ("train", "dev", "test")
        },
        "limitations": [
            "Authored triage policy, not expert-labeled optimal experiment design.",
            "Historical development distributions are reused.",
            "Does not teach arbitrary code repair or independent scientific discovery.",
        ],
    }
    write_manifest(output, manifest)
    return {k: v for k, v in manifest.items() if k != "sources"}


def write_manifest(
    output: Union[str, os.PathLike[str]], manifest: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """Hash split contents; optionally authenticate with a key kept outside the dataset."""
    output = Path(output)
    manifest["files"] = {
        s: hashlib.sha256((output / (s + ".jsonl")).read_bytes()).hexdigest()
        for s in ("train", "dev", "test")
    }
    key_path = os.environ.get("SILICONSHIELD_DATASET_KEY")
    if key_path:
        manifest["signature"] = hmac.new(
            Path(key_path).read_bytes(),
            json.dumps(manifest, sort_keys=True, allow_nan=False).encode(),
            hashlib.sha256,
        ).hexdigest()
    atomic_json(output / "manifest.json", manifest)
    return manifest


def verify_split_membership(
    rows: Sequence[Mapping[str, JsonValue]],
    episodes_path: Union[str, os.PathLike[str]],
    manifest_path: Union[str, os.PathLike[str], None] = None,
    allowed_splits: tuple[str, ...] = ("train", "dev"),
    require_signed: bool = False,
) -> str:
    """Verify content and lineage, independent of the filename. Test access is explicit."""
    episodes_path = Path(episodes_path)
    manifest_file = (
        Path(manifest_path)
        if manifest_path
        else episodes_path.parent / "manifest.json"
    )
    if not manifest_file.is_file():
        raise ValueError(
            "No dataset manifest beside episodes; split membership cannot be verified"
        )
    manifest = read_json(manifest_file)
    if not isinstance(manifest, dict):
        raise ValueError("Dataset manifest is not an object")
    key_path = os.environ.get("SILICONSHIELD_DATASET_KEY")
    if require_signed and (not manifest.get("signature") or not key_path):
        raise ValueError(
            "Training requires a signed dataset manifest and SILICONSHIELD_DATASET_KEY"
        )
    if manifest.get("signature") and not key_path:
        raise ValueError(
            "Authenticated dataset requires SILICONSHIELD_DATASET_KEY"
        )
    if key_path:
        signature = manifest.pop("signature", "")
        if not isinstance(signature, str):
            raise ValueError("Dataset manifest signature mismatch")
        expected = hmac.new(
            Path(key_path).read_bytes(),
            json.dumps(manifest, sort_keys=True, allow_nan=False).encode(),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Dataset manifest signature mismatch")
    content = hashlib.sha256(episodes_path.read_bytes()).hexdigest()
    raw_files = manifest.get("files", {})
    if not isinstance(raw_files, dict):
        raise ValueError("Dataset manifest lacks split file hashes")
    files: dict[str, JsonValue] = raw_files
    matches = [s for s, value in files.items() if value == content]
    if len(matches) != 1:
        raise ValueError(
            "Episode content does not match a unique manifest split hash"
        )
    split = matches[0]
    if split not in allowed_splits:
        raise ValueError("Split is not permitted in this execution: " + split)
    groups = manifest.get("groups")
    if not isinstance(groups, dict):
        raise ValueError(
            "Dataset manifest lacks group split membership; rebuild the dataset"
        )
    wrong: list[str] = []
    for r in rows:
        group_id = r.get("group_id")
        if not isinstance(group_id, str) or groups.get(group_id) != split:
            wrong.append(
                group_id if isinstance(group_id, str) else repr(group_id)
            )
    wrong = sorted(set(wrong))
    if wrong:
        raise ValueError(
            "Episodes from another split lineage are present in %s: %s"
            % (split, wrong[:3])
        )
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        required=True,
        help="Directory whose subdirectories contain completed development reports",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Fresh directory for episode splits and their manifest",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.runs, args.output), indent=2))


if __name__ == "__main__":
    main()
