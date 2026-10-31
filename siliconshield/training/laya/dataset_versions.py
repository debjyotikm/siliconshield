"""Immutable dataset versions with sampling pools kept off provenance.

A version pins every split file by SHA-256 under ``versions/<name>/``;
re-registering a version with different content is refused, and
``verify_version`` re-hashes pinned bytes before any training use.
Byte integrity is not dataset acceptance: ``accept_version`` records a
split/lineage audit separately, and only accepted versions train.
Sampling pools live apart from source provenance and bind into the
version identity, so pool weights cannot drift silently. A new
OpenTitan example stays OpenTitan no matter which pool draws it.
"""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from siliconshield.research_loop.episodes.contracts import atomic_json
from siliconshield.schemas import (
    DATASET_ACCEPTANCE_SCHEMA,
    DATASET_AUDIT_SCHEMA,
    DATASET_VERSION_SCHEMA,
)
from siliconshield.training.laya.inventory import feasibility


def _sha(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _version_file(root: Path, version: str) -> Path:
    return Path(root) / "versions" / version / "manifest.json"


def _accepted_file(root: Path, version: str) -> Path:
    return Path(root) / "versions" / version / "accepted.json"


def write_version(
    root: Path, version: str, splits: dict[str, str]
) -> dict[str, Any]:
    """Pin one immutable dataset version.

    ``splits`` maps role names (``train``, ``dev``, ...) to JSONL
    paths. Requires at least one split, distinct files per role, and
    a positive row count each. Refuses to overwrite. Returns the
    manifest.
    """
    path = _version_file(root, version)
    if path.exists():
        raise ValueError("Dataset version exists: " + version)
    if not splits:
        raise ValueError("Dataset version needs at least one split")
    seen = set()
    manifest: dict[str, Any] = {
        "schema": DATASET_VERSION_SCHEMA,
        "version": version,
        "splits": {},
    }
    for role, rows in splits.items():
        resolved = str(Path(rows).resolve())
        if resolved in seen:
            raise ValueError("Two roles share one split file: " + role)
        seen.add(resolved)
        with Path(rows).open(encoding="utf-8") as stream:
            count = sum(1 for _ in stream)
        if count <= 0:
            raise ValueError("Split has no rows: " + role)
        manifest["splits"][role] = {
            "path": resolved,
            "sha256": _sha(Path(rows)),
            "rows": count,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_version(root: Path, version: str) -> dict[str, Any]:
    """Re-hash pinned bytes; refuse on any drift.

    This is byte integrity only. Training readiness additionally
    requires the acceptance marker from ``accept_version``.
    """
    manifest: dict[str, Any] = json.loads(
        _version_file(root, version).read_text(encoding="utf-8")
    )
    if manifest.get("schema") != DATASET_VERSION_SCHEMA:
        raise ValueError("Unknown dataset-version schema")
    for role, entry in manifest["splits"].items():
        if _sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Dataset drift in %s split" % role)
    pools = manifest.get("pools")
    if pools and _sha(Path(pools["path"])) != pools["sha256"]:
        raise ValueError("Sampling pools changed under the version")
    return manifest


def _audit_sha(audit: dict[str, Any]) -> str:
    """Return the canonical digest of an audit record."""
    return hashlib.sha256(
        json.dumps(audit, sort_keys=True).encode()
    ).hexdigest()


def _hex_digest(value: Any) -> str:
    """Return the value when it is a SHA-256 hex digest, else refuse."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value.lower())
    ):
        raise ValueError("Audit needs a SHA-256 hex digest")
    return value.lower()


def _check_share(value: Any, cap: float, name: str) -> None:
    """Require a finite concentration share within its cap."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        or value > cap + 1e-9
    ):
        raise ValueError("Audit blocks training: " + name)


def _audit_ok(manifest: dict[str, Any], audit: Any) -> None:
    """Require an audit that covers the manifest and clears training.

    Audits carrying the strict schema must declare every
    independence gate and a bound review ledger; legacy audits
    without the schema keep the old optional-extras behavior so
    historical markers remain readable.
    """
    if not isinstance(audit, dict):
        raise ValueError("Audit is missing")
    for role, entry in manifest["splits"].items():
        if audit.get("splits", {}).get(role) != entry["sha256"]:
            raise ValueError("Audit does not cover split: " + role)
    for key in (
        "cross_split_exact_rtl_overlap",
        "retired_rtl_overlap",
        "truncated",
    ):
        if audit.get(key) != 0:
            raise ValueError("Audit blocks training: " + key)
    if audit.get("schema") != DATASET_AUDIT_SCHEMA:
        # Legacy audit: extended independence gates are optional but
        # enforced when declared.
        if "lineage_overlap" in audit and audit["lineage_overlap"] != 0:
            raise ValueError("Audit blocks training: lineage_overlap")
        if "canonical_overlap" in audit and audit["canonical_overlap"] != 0:
            raise ValueError("Audit blocks training: canonical_overlap")
        if "max_group_share" in audit:
            _check_share(audit["max_group_share"], 0.05, "max_group_share")
        if "max_mechanism_share" in audit:
            _check_share(
                audit["max_mechanism_share"], 0.15, "max_mechanism_share"
            )
        return
    if audit.get("lineage_overlap") != 0:
        raise ValueError("Audit blocks training: lineage_overlap")
    if audit.get("canonical_overlap") != 0:
        raise ValueError("Audit blocks training: canonical_overlap")
    _check_share(audit.get("max_group_share"), 0.05, "max_group_share")
    _check_share(audit.get("max_mechanism_share"), 0.15, "max_mechanism_share")
    if audit.get("release_class") not in ("broad", "narrow-pilot"):
        raise ValueError("Audit release_class must be broad or narrow-pilot")
    _hex_digest(audit.get("review_ledger_sha256"))
    reviewed = audit.get("reviewed_templates")
    if (
        not isinstance(reviewed, int)
        or isinstance(reviewed, bool)
        or reviewed < 1
    ):
        raise ValueError("Audit needs a positive reviewed_templates count")


def accept_version(
    root: Path, version: str, audit: dict[str, Any]
) -> dict[str, Any]:
    """Record a split/lineage audit so a version may train.

    New releases must carry the strict audit schema with the split
    hashes, zero-overlap findings, lineage and canonical overlap,
    concentration caps, a bound review-ledger digest, and a release
    class. The record binds the exact version manifest bytes and the
    full audit, and publishes atomically. Acceptance is immutable:
    re-accepting with a different audit is refused. Historical
    markers written under a legacy audit stay readable through
    ``is_accepted`` but cannot back a new acceptance.
    """
    manifest = verify_version(root, version)
    _audit_ok(manifest, audit)
    if audit.get("schema") != DATASET_AUDIT_SCHEMA:
        raise ValueError(
            "New releases require audit schema " + DATASET_AUDIT_SCHEMA
        )
    claimed = audit.get("release_class")
    if claimed not in ("broad", "narrow-pilot"):
        raise ValueError("Audit release_class must be broad or narrow-pilot")
    counts = {
        key: audit[key]
        for key in (
            "train_groups",
            "dev_groups",
            "calibration_groups",
            "test_groups",
        )
        if key in audit
    }
    verdict = feasibility(counts) if counts else None
    if verdict is not None:
        if claimed == "broad" and not verdict["broad_release"]:
            raise ValueError(
                "Audit claims broad release but group counts fail targets"
            )
        if claimed is None:
            audit = {**audit, "release_class": verdict["release_class"]}
    audit_sha = _audit_sha(audit)
    marker = _accepted_file(root, version)
    if marker.exists():
        try:
            recorded: dict[str, Any] = json.loads(
                marker.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            raise ValueError("Acceptance record is unreadable") from exc
        if not isinstance(recorded, dict):
            raise ValueError("Acceptance record is unreadable")
        if recorded.get("audit_sha256") != audit_sha:
            raise ValueError("Version accepted under a new audit")
        return recorded
    recorded = {
        "schema": DATASET_ACCEPTANCE_SCHEMA,
        "version": version,
        "manifest_sha256": _sha(_version_file(root, version)),
        "audit": audit,
        "audit_sha256": audit_sha,
    }
    atomic_json(marker, recorded)
    return recorded


def _acceptance_ok(root: Path, version: str) -> None:
    """Require a valid acceptance record, raising on any defect."""
    record = json.loads(
        _accepted_file(root, version).read_text(encoding="utf-8")
    )
    if not isinstance(record, dict):
        raise ValueError("Acceptance record is unreadable")
    if record.get("schema") != DATASET_ACCEPTANCE_SCHEMA:
        raise ValueError("Acceptance record names another schema")
    if record.get("version") != version:
        raise ValueError("Acceptance record names another version")
    if record.get("manifest_sha256") != _sha(_version_file(root, version)):
        raise ValueError("Acceptance record is detached from its version")
    manifest = verify_version(root, version)
    audit = record.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("Acceptance record carries no audit")
    if _audit_sha(audit) != record.get("audit_sha256"):
        raise ValueError("Acceptance audit changed under its record")
    _audit_ok(manifest, audit)


def is_accepted(root: Path, version: str) -> bool:
    """Report whether a version carries a valid acceptance record.

    A filename alone establishes nothing: the record must parse,
    name this version, bind the current manifest bytes and audit,
    and still satisfy the acceptance criteria against live files.
    Any failure reads as not accepted.
    """
    try:
        _acceptance_ok(root, version)
    except (OSError, ValueError):
        return False
    return True


def _row_ids(rows_path: Path) -> list[str]:
    """Row ids in file order; ids must be unique strings."""
    ids = [
        json.loads(line)["id"]
        for line in Path(rows_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Rows need unique string ids")
    return ids


def _check_partition(
    ids: list[str], pools: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Validate that pools partition row ids exactly once each."""
    seen: set[str] = set()
    checked = {}
    for name, spec in pools.items():
        weight = spec.get("weight")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError("Pool weight must be positive: " + name)
        members = spec.get("rows", [])
        if len(set(members)) != len(members):
            raise ValueError("Pool repeats a row: " + name)
        overlap = seen & set(members)
        if overlap:
            raise ValueError("Row assigned to two pools: " + sorted(overlap)[0])
        checked[name] = {"rows": list(members), "weight": weight}
        seen |= set(members)
    if seen != set(ids):
        raise ValueError("Pools must partition all %d rows exactly" % len(ids))
    return checked


def assign_pools(
    rows_path: Path, pools: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Write one sampling-pool manifest over a rows file.

    The pools must partition the file's row ids exactly: no row is
    drawn from two pools and none is left out. Weights must be
    positive and finite. Returns the manifest.
    """
    rows_path = Path(rows_path)
    manifest = {
        "rows": str(rows_path.resolve()),
        "rows_sha256": _sha(rows_path),
        "pools": _check_partition(_row_ids(rows_path), pools),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_pools(rows_path: Path, pools_path: Path) -> dict[str, Any]:
    """Confirm a pool manifest matches its rows file; read-only."""
    manifest: dict[str, Any] = json.loads(
        Path(pools_path).read_text(encoding="utf-8")
    )
    rows_path = Path(rows_path)
    if _sha(rows_path) != manifest["rows_sha256"]:
        raise ValueError("Rows file changed under its pool manifest")
    manifest["pools"] = _check_partition(_row_ids(rows_path), manifest["pools"])
    return manifest


def bind_pools(root: Path, version: str, pools_path: Path) -> dict[str, Any]:
    """Pin one pool manifest into a dataset version's identity.

    The pools must cover the version's train split exactly. Binding
    is permanent: pool weights change only in a new version, and
    ``verify_version`` re-hashes the bound file on every check.
    """
    manifest = verify_version(root, version)
    if "pools" in manifest:
        raise ValueError("Version already binds pools: " + version)
    train = manifest["splits"].get("train")
    if train is None:
        raise ValueError("Pool binding needs a train split")
    verify_pools(Path(train["path"]), Path(pools_path))
    manifest["pools"] = {
        "path": str(Path(pools_path).resolve()),
        "sha256": _sha(Path(pools_path)),
    }
    path = _version_file(root, version)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def pool_assignment(
    pools_path: Path, rows: list[dict[str, Any]]
) -> tuple[dict[int, str], dict[str, float], dict[str, str]]:
    """Map row indices to pools with weights for batch sampling.

    ``rows`` are the loaded training rows in file order; each needs
    the ``id`` the pool manifest was built over. Returns
    ``(index_to_pool, pool_weights, pool_modes)``. Raises on any
    mismatch so a stale manifest can never silently skew batches.
    """
    manifest = json.loads(Path(pools_path).read_text(encoding="utf-8"))
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
    modes: dict[str, str] = {}
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


def _overlap_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows; refuse malformed lines."""
    rows = []
    for lineno, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError("Row %d is not JSON: %s" % (lineno, path)) from exc
        if not isinstance(row, dict):
            raise ValueError("Row %d is not an object: %s" % (lineno, path))
        rows.append(row)
    return rows


def rtl_overlap_key(row: dict[str, Any]) -> str:
    """Canonical key for exact RTL overlap checks.

    The state carries a rendered ``RTL`` section; whitespace
    differences do not distinguish circuits here. Rows without an
    RTL section refuse instead of auditing as clean.
    """
    state = row.get("state")
    if not isinstance(state, str) or "RTL\n" not in state:
        raise ValueError("Row lacks an RTL section: " + repr(row.get("id")))
    rtl = state.split("RTL\n", 1)[1]
    return hashlib.sha256(re.sub(r"\s+", "", rtl).encode()).hexdigest()


def audit_partition_overlap(
    splits: dict[str, Path],
    retired: list[Path],
    train_role: str = "train",
) -> dict[str, int]:
    """Count exact RTL collisions across splits and retired rows.

    One map covers every split in order; only collisions across
    different splits count, so within-train requirement twins stay
    legal. Every training row is also checked against the retired
    population. Returns the two overlap findings an acceptance
    audit must record as zero.
    """
    groups: dict[str, str] = {}
    cross = 0
    train_keys: list[str] = []
    roles = [train_role] + sorted(r for r in splits if r != train_role)
    for role in roles:
        for row in _overlap_rows(splits[role]):
            key = rtl_overlap_key(row)
            if role == train_role:
                train_keys.append(key)
            if groups.setdefault(key, role) != role:
                cross += 1
    sealed: set[str] = set()
    for path in retired:
        sealed.update(rtl_overlap_key(row) for row in _overlap_rows(path))
    stale = sum(1 for key in train_keys if key in sealed)
    return {
        "cross_split_exact_rtl_overlap": cross,
        "retired_rtl_overlap": stale,
    }


def lineage_key(row: dict[str, Any]) -> str:
    """Return the split-group identity for lineage overlap checks.

    ``split_group_id`` is the conservative connected component over
    source units, helper bodies, generator origins, and policy
    variants. It falls back to ``family_id`` and then ``pair_id``
    for synthetic fixtures that name only those fields.
    """
    for key in ("split_group_id", "family_id", "pair_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("Row lacks lineage identity: " + repr(row.get("id")))


_RTL_KEYWORDS = frozenset(
    """
    module endmodule input output inout wire reg logic always assign
    if else begin end case casex casez default endcase for while
    function endfunction parameter localparam generate endgenerate
    posedge negedge edge specify endspecify initial forever repeat
    integer real time triand trior tri0 tri1 supply0 supply1
    """.split()
)

_TOKEN = re.compile(
    r"\d+'[bBdDhH][0-9a-fA-F_xXzZ?]+"
    r"|[A-Za-z_][A-Za-z0-9_$]*"
    r"|\d[\d_]*"
    r"|\"(?:[^\"\\]|\\.)*\""
    r"|\S"
)


def canonical_rtl_key(row: dict[str, Any]) -> str:
    """Return an alpha-normalized RTL screening key.

    Maps identifiers to positional placeholders (consistent
    renames collide), keeps Verilog keywords, operators,
    punctuation, widths, and literals verbatim. This is a
    near-duplicate screen, not proof of semantic independence;
    near matches still need review.
    """
    state = row.get("state")
    if not isinstance(state, str) or "RTL\n" not in state:
        raise ValueError("Row lacks an RTL section: " + repr(row.get("id")))
    rtl = state.split("RTL\n", 1)[1]
    rtl = re.sub(r"/\*.*?\*/", " ", rtl, flags=re.S)
    rtl = re.sub(r"//.*", " ", rtl)
    mapping: dict[str, str] = {}
    out: list[str] = []
    for match in _TOKEN.finditer(rtl):
        token = match.group(0)
        if token[:1].isalpha() or token[:1] in "_$":
            lowered = token.lower()
            if lowered in _RTL_KEYWORDS or lowered.startswith("$"):
                out.append(lowered)
            else:
                if lowered not in mapping:
                    mapping[lowered] = "V%d" % len(mapping)
                out.append(mapping[lowered])
        elif token[:1].isdigit():
            out.append(token.lower())
        else:
            out.append(token)
    return hashlib.sha256(" ".join(out).encode()).hexdigest()


def audit_lineage_overlap(
    splits: dict[str, Path], train_role: str = "train"
) -> dict[str, int]:
    """Count cross-split rows from shared lineage groups.

    Every descendant of one conservative component must stay in one
    split. Each row whose group was already claimed by another role
    counts once, so the total scales with the contamination size.
    """
    owners: dict[str, str] = {}
    overlap = 0
    roles = [train_role] + sorted(r for r in splits if r != train_role)
    for role in roles:
        for row in _overlap_rows(splits[role]):
            group = lineage_key(row)
            if owners.setdefault(group, role) != role:
                overlap += 1
    return {"lineage_overlap": overlap}


def audit_canonical_overlap(
    splits: dict[str, Path], train_role: str = "train"
) -> dict[str, int]:
    """Count cross-split rows with colliding normalized RTL.

    Complements exact matching: renames and formatting-only
    rewrites still collide here. Within-split twins stay legal.
    Each colliding row counts once toward the total.
    """
    owners: dict[str, str] = {}
    cross = 0
    roles = [train_role] + sorted(r for r in splits if r != train_role)
    for role in roles:
        for row in _overlap_rows(splits[role]):
            key = canonical_rtl_key(row)
            if owners.setdefault(key, role) != role:
                cross += 1
    return {"canonical_overlap": cross}


def dependency_group(row: dict[str, Any]) -> str:
    """Return the concentration group for sampling-cap checks."""
    for key in ("dependency_group_id", "split_group_id", "family_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("Row lacks dependency group: " + repr(row.get("id")))


def audit_concentration(train_path: Path) -> dict[str, Any]:
    """Measure TRAIN concentration and independent-group counts.

    Returns the largest dependency-group share, the largest
    mechanism share, and the distinct group/mechanism counts.
    Fresh releases must keep the former at or below 5% and the
    latter at or below 15% where the inventory permits it.
    """
    rows = _overlap_rows(train_path)
    if not rows:
        raise ValueError("TRAIN split has no rows")
    groups: dict[str, int] = {}
    mechanisms: dict[str, int] = {}
    for row in rows:
        groups[dependency_group(row)] = groups.get(dependency_group(row), 0) + 1
        mechanism = row.get("mechanism_id", "unknown")
        mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
    total = len(rows)
    return {
        "train_rows": total,
        "train_groups": len(groups),
        "train_mechanisms": len(mechanisms),
        "max_group_share": max(groups.values()) / total,
        "max_mechanism_share": max(mechanisms.values()) / total,
    }


def audit_split_groups(splits: dict[str, Path]) -> dict[str, int]:
    """Count distinct lineage groups per split role."""
    counts: dict[str, int] = {}
    for role, path in splits.items():
        counts[role] = len({lineage_key(row) for row in _overlap_rows(path)})
    return counts
