"""Download and verify a pinned public decision-model export, outside Git."""

import argparse
import hashlib
import json
import os
import tempfile
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "model-releases.toml"


def file_sha256(path: Path) -> str:
    """Hash a file without loading the weights into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def prepare_model(record: dict[str, Any], target: Path) -> str:
    """Fetch missing payload files and refuse any existing identity mismatch.

    Files publish individually only after checksum verification. A retry reuses
    verified files. The complete export fingerprint is returned only when every
    payload file matches and no extra files are present.
    """
    if not isinstance(record, dict) or not isinstance(
        record.get("files"), dict
    ):
        raise ValueError("Model record must name its pinned files")
    if not isinstance(target, Path):
        raise ValueError("Model destination must be a path")
    for key in ("repo_id", "revision", "export_sha256"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError("Model record is missing " + key)
    expected = record["files"]
    for name, wanted in expected.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Model record names a non-string file")
        if not isinstance(wanted, str) or not wanted:
            raise ValueError("Model record pins no digest for " + name)
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    existing = {
        str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()
    }
    if existing - expected.keys():
        raise ValueError("Destination has files outside the pinned export")
    for name, wanted in expected.items():
        path = target / name
        resolved = path.resolve()
        if resolved == root or root not in resolved.parents:
            raise ValueError("Pinned file escapes its destination: " + name)
        if path.exists():
            if file_sha256(path) != wanted:
                raise ValueError("Existing file checksum mismatch: " + name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/{record['repo_id']}/resolve/"
            f"{record['revision']}/{name}"
        )
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".download-")
        try:
            with os.fdopen(fd, "wb") as stream:
                with urllib.request.urlopen(url, timeout=60) as response:
                    while block := response.read(8 * 1024 * 1024):
                        stream.write(block)
            if file_sha256(Path(temporary)) != wanted:
                raise ValueError("Downloaded file checksum mismatch: " + name)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    actual = {name: file_sha256(target / name) for name in expected}
    fingerprint = hashlib.sha256(
        json.dumps(
            actual, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if fingerprint != record["export_sha256"]:
        raise ValueError("Export fingerprint differs from release manifest")
    return fingerprint


def main() -> None:
    """Prepare a named immutable payload without changing any Hub repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("contract-1", "contract-2", "all"), default="all"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("Model exports must remain outside the source repository")
    manifest = tomllib.loads(MANIFEST.read_text())
    for name, record in manifest["models"].items():
        if args.model not in ("all", name):
            continue
        fingerprint = prepare_model(record, output / name)
        print(
            json.dumps(
                {
                    "model": name,
                    "revision": record["revision"],
                    "export_sha256": fingerprint,
                    "path": str(output / name),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
