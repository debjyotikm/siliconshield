"""Load and validate the RTL benchmark rows (Qt-free, no Torch).

Sources are `hf://<repo>@<revision>:<path>` for pinned Hub downloads
or plain local paths for vendored/offline copies and CPU tests. Every
row must carry an id, state text, questions, an expected three-way
label, an area tag, and a triplet key; validation rejects anything
else before a single GPU second is spent.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, TypedDict

DEFAULT_DATASET = "debjyotikm/siliconshield-rtl-benchmark"
DEFAULT_PATH = "data/benchmark.jsonl"
DEFAULT_REVISION = "ab2b7dbee3158a292edc4f7a0127379aab996636"

LABELS = (
    "satisfies_requirement",
    "violates_requirement",
    "insufficient_context",
)

REQUIRED_KEYS = frozenset(
    {
        "id",
        "state",
        "questions",
        "expected",
        "benchmark_area",
        "benchmark_source",
        "pair_id",
    }
)


class BenchmarkData(TypedDict):
    """Validated benchmark rows with provenance."""

    rows: list[dict[str, Any]]
    source: str
    rows_sha256: str


def _read_local(path: str) -> tuple[list[dict[str, Any]], str]:
    try:
        blob = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError("Cannot read benchmark %s: %s" % (path, exc)) from exc
    try:
        text = blob.decode()
    except UnicodeDecodeError as exc:
        raise ValueError("Benchmark %s is not UTF-8: %s" % (path, exc)) from exc
    try:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except ValueError as exc:
        raise ValueError(
            "Benchmark %s has invalid JSON: %s" % (path, exc)
        ) from exc
    digest = hashlib.sha256(blob).hexdigest()
    return rows, digest


def _read_hf(
    repo: str, revision: str, path: str
) -> tuple[list[dict[str, Any]], str]:
    try:
        from huggingface_hub import (  # pylint: disable=import-outside-toplevel
            hf_hub_download,
        )
    except ImportError as exc:
        raise ValueError(
            "huggingface_hub is required for hf:// sources "
            "(see requirements/benchmark.txt)."
        ) from exc
    try:
        cached = hf_hub_download(
            repo_id=repo, filename=path, revision=revision, repo_type="dataset"
        )
    except Exception as exc:
        raise ValueError(
            "Cannot download %s@%s:%s: %s" % (repo, revision, path, exc)
        ) from exc
    return _read_local(cached)


def validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate benchmark rows; return them unchanged.

    Raises ValueError for missing keys, empty state, unknown labels,
    duplicate ids, triplet keys that do not group exactly three
    rows, or triplets split across benchmark areas: whole-triplet
    scoring requires complete, area-contained triplets.
    """
    if not rows:
        raise ValueError("Benchmark has no rows.")
    seen: set[str] = set()
    groups: dict[str, int] = {}
    pair_areas: dict[str, set[str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Row %d is not an object." % index)
        missing = REQUIRED_KEYS - set(row)
        if missing:
            raise ValueError(
                "Row %d is missing keys: %s" % (index, sorted(missing))
            )
        row_id = row["id"]
        if not row_id or not isinstance(row_id, str):
            raise ValueError("Row %d has no string id." % index)
        if row_id in seen:
            raise ValueError("Duplicate benchmark id: %s" % row_id)
        seen.add(row_id)
        if not isinstance(row["state"], str) or not row["state"].strip():
            raise ValueError("Row %s has empty state text." % row_id)
        expected = row.get("expected")
        label = expected.get("security") if isinstance(expected, dict) else None
        if label not in LABELS:
            raise ValueError("Row %s has unknown label %r." % (row_id, label))
        key = row["pair_id"]
        if not key or not isinstance(key, str):
            raise ValueError("Row %s has no triplet key." % row_id)
        mechanism = row.get("mechanism_id")
        if mechanism is not None and (
            not isinstance(mechanism, str) or not mechanism
        ):
            raise ValueError("Row %s has no string mechanism id." % row_id)
        area = row.get("benchmark_area")
        if not isinstance(area, str) or not area:
            raise ValueError("Row %s has no string benchmark area." % row_id)
        groups[key] = groups.get(key, 0) + 1
        pair_areas.setdefault(key, set()).add(row["benchmark_area"])
    broken = {key: size for key, size in groups.items() if size != 3}
    if broken:
        raise ValueError(
            "Triplet keys must group 3 rows: %s"
            % (dict(list(broken.items())[:5]))
        )
    split = sorted(key for key, areas in pair_areas.items() if len(areas) > 1)
    if split:
        raise ValueError("Triplets span benchmark areas: %s" % split[:5])
    return rows


def load_benchmark(source: str) -> BenchmarkData:
    """Load one benchmark from an hf:// or local source.

    `hf://` without `@revision` uses the pinned default revision;
    without `:path` it uses the default file. Returns rows plus the
    resolved source description and content hash.
    """
    if source.startswith("hf://"):
        rest = source[len("hf://") :]
        if ":" in rest:
            repo_rev, path = rest.rsplit(":", 1)
        else:
            repo_rev, path = rest, DEFAULT_PATH
        if "@" in repo_rev:
            repo, revision = repo_rev.split("@", 1)
        else:
            repo, revision = repo_rev, DEFAULT_REVISION
        if not repo or not revision or not path:
            raise ValueError("Malformed hf:// source: %s" % source)
        rows, digest = _read_hf(repo, revision, path)
        resolved = "hf://%s@%s:%s" % (repo, revision, path)
    else:
        rows, digest = _read_local(source)
        resolved = "local:%s" % source
    return {
        "rows": validate_rows(rows),
        "source": resolved,
        "rows_sha256": digest,
    }


def default_source() -> str:
    """Return the pinned public benchmark source string."""
    return "hf://%s@%s:%s" % (DEFAULT_DATASET, DEFAULT_REVISION, DEFAULT_PATH)
