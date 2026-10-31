"""Verify local model-snapshot identity against recorded provenance.

Recording MODEL_REVISION is not verification. This hashes the snapshot's
configuration, tokenizer, and weight-index files (and, when ``full`` is
true, every weight shard) so an evaluation can prove which weights were
loaded. All helpers raise ``ValueError`` with an actionable message when
identity cannot be established; callers must not catch that and continue.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import TypedDict, Union

from siliconshield.research_loop.episodes.contracts import digest

INDEX_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
    "chat_template.jinja",
)

# Pinned base-model revision shared by training and DPO. Kept with the
# snapshot-identity helpers (not the training entry point) so both trainers
# reference one definition; verify_snapshot() enforces it against download
# provenance.
MODEL_REVISION = "cec94fcfd6512f2cfe708138c32533b346ea4f69"


class SnapshotManifest(TypedDict):
    """Content hashes and fingerprint of one model snapshot directory."""

    files: dict[str, str]
    fingerprint: str


class AdapterManifest(TypedDict):
    """Content hashes and fingerprint of one adapter checkpoint directory."""

    files: dict[str, str]
    fingerprint: str


def _sha(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checkpoint_fingerprint(
    adapter_path: str | os.PathLike[str] | None = None,
    base_revision: str = MODEL_REVISION,
) -> str:
    """Hash the base revision and adapter contents without optional tooling.

    Only an omitted adapter path selects the base model. Missing or empty
    checkpoints raise ValueError rather than reuse the base identity.
    The digest format is shared with existing Harness2 receipts.
    """
    files: dict[str, str] = {}
    if adapter_path is not None:
        root = Path(adapter_path)
        if not root.exists():
            raise ValueError("Adapter checkpoint does not exist: " + str(root))
        if root.is_dir():
            config = root / "adapter_config.json"
            weights = [
                p
                for p in root.iterdir()
                if p.name.startswith("adapter_model.") and p.stat().st_size > 0
            ]
            if not config.is_file() or not weights:
                raise ValueError(
                    "Adapter checkpoint is incomplete (needs adapter_config.json "
                    "and non-empty adapter_model weights): " + str(root)
                )
            paths = sorted(p for p in root.rglob("*") if p.is_file())
        else:
            if root.stat().st_size == 0:
                raise ValueError(
                    "Adapter checkpoint file is empty: " + str(root)
                )
            paths = [root]
        for path in paths:
            name = str(path.relative_to(root)) if root.is_dir() else path.name
            files[name] = _sha(path)
    return digest({"base_revision": base_revision, "adapter_files": files})


def snapshot_manifest(
    model_dir: Union[str, "os.PathLike[str]"],
    full: bool = True,
    revision: Union[str, None] = None,
) -> SnapshotManifest:
    """Hash a model snapshot's config, tokenizer, index, and weight shards.

    Returns content hashes plus a fingerprint binding them together.
    Raises ValueError when the directory is missing files, when ``full``
    is false (partial hashing proves nothing), or when ``revision`` download
    provenance disagrees with the recorded commit.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise ValueError("Model snapshot does not exist: " + str(root))
    files: dict[str, str] = {}
    for name in INDEX_FILES:
        if (root / name).is_file():
            files[name] = _sha(root / name)
    shards = sorted(root.glob("*.safetensors"))
    if not shards and not (root / "model.safetensors").is_file():
        raise ValueError(
            "Model snapshot has no safetensors weights: " + str(root)
        )
    if not full:
        raise ValueError("Weight identity requires full shard hashing")
    index = root / "model.safetensors.index.json"
    if index.exists():
        listed = set(json.loads(index.read_text())["weight_map"].values())
        if listed != {p.name for p in shards}:
            raise ValueError("Weight index and snapshot shards differ")
    for shard in shards:
        if shard.stat().st_size == 0:
            raise ValueError("Empty model weight shard")
        files[shard.name] = _sha(shard)
    if revision:
        for name in files:
            metadata = (
                root / ".cache/huggingface/download" / (name + ".metadata")
            )
            if not metadata.is_file():
                raise ValueError("Missing download provenance for " + name)
            recorded = metadata.read_text().splitlines()
            if len(recorded) < 2 or recorded[0] != revision:
                raise ValueError("Snapshot commit mismatch for " + name)
            if name.endswith(".safetensors") and recorded[1] != files[name]:
                raise ValueError(
                    "Downloaded weight digest mismatch for " + name
                )
            if len(recorded[1]) == 40:
                raw = (root / name).read_bytes()
                git_blob = hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest()
                if git_blob != recorded[1]:
                    raise ValueError(
                        "Downloaded configuration/tokenizer digest mismatch for "
                        + name
                    )
    if "config.json" not in files:
        raise ValueError("Model snapshot lacks config.json: " + str(root))
    return {"files": files, "fingerprint": digest(files)}


def verify_snapshot(
    model_dir: Union[str, "os.PathLike[str]"],
    expected_fingerprint: Union[str, None] = None,
    full: bool = True,
    revision: Union[str, None] = None,
) -> SnapshotManifest:
    """Hash a snapshot and reject it unless the fingerprint matches.

    Returns the manifest on success. Raises ValueError on any mismatch;
    a fingerprint argument of None records without checking.
    """
    manifest = snapshot_manifest(model_dir, full=full, revision=revision)
    if expected_fingerprint and manifest["fingerprint"] != expected_fingerprint:
        raise ValueError(
            "Model snapshot fingerprint mismatch: expected %s, found %s"
            % (expected_fingerprint, manifest["fingerprint"])
        )
    return manifest


def code_identity() -> str:
    """Fingerprint the executing ``siliconshield`` package sources."""
    root = Path(__file__).resolve().parents[2]
    return digest(
        {str(p.relative_to(root)): _sha(p) for p in sorted(root.rglob("*.py"))}
    )


def verify_code_identity(expected: Union[str, None] = None) -> str:
    """Compare executing source with the identity captured before launch.

    Returns the verified identity. Raises ValueError when no expected
    identity is configured, when it is malformed, or when it disagrees
    with the executing source.
    """
    expected = expected or os.environ.get(
        "SILICONSHIELD_EXPECTED_CODE_IDENTITY"
    )
    if not expected:
        raise ValueError(
            "Set SILICONSHIELD_EXPECTED_CODE_IDENTITY from the frozen launch source"
        )
    if len(expected) != 64 or any(
        c not in "0123456789abcdef" for c in expected
    ):
        raise ValueError("Expected code identity must be a SHA-256 hex digest")
    actual = code_identity()
    if actual != expected:
        raise ValueError(
            "Launch code identity mismatch: expected %s, executing %s"
            % (expected, actual)
        )
    return actual


def adapter_manifest(
    adapter_dir: Union[str, "os.PathLike[str]"],
) -> AdapterManifest:
    """Hash an adapter checkpoint directory and fingerprint its contents.

    Raises ValueError when the directory holds no files.
    """
    root = Path(adapter_dir)
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(root))] = _sha(path)
    if not files:
        raise ValueError("Adapter checkpoint is empty: " + str(root))
    return {"files": files, "fingerprint": digest(files)}
