"""Local model registry: validated, file-based, single-machine only.

Registries live outside Git (see the ``--models`` option). Paths may be
absolute or relative to the registry file itself. Validation checks shape,
duplicate ids, executable worker interpreters, the LAYA runtime layout,
and optional capability profiles, release versions, question contracts,
and qualification references, but never imports LAYA or touches the
GPU: artifact hashes are verified by the worker process at load time,
on the machine that serves them.
"""

import json
import os
import re
from pathlib import Path
from typing import TypedDict, Union

from siliconshield.experts.capabilities import load_capability_profile

DEVICE_PATTERN = re.compile(r"^(cpu|cuda:\d+)$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ModelEntry(TypedDict, total=False):
    """One validated registry entry with resolved absolute paths."""

    id: str
    display_name: str
    checkpoint: str
    python: str
    laya_root: str
    expected_files: dict[str, str]
    device: str
    profile: str
    release: str
    question_contract: str
    qualification: str


def _is_non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fail(reason: str) -> ValueError:
    return ValueError("Invalid model registry: " + reason)


def load_registry(path: Union[str, os.PathLike[str]]) -> list[ModelEntry]:
    """Load and validate an expert model registry file.

    Raises ValueError with an actionable message for unreadable files,
    schema mismatches, duplicate ids, missing directories, non-executable
    interpreters, malformed digests, or bad device names (cpu or cuda:N).
    Relative paths resolve against the registry file, never the working
    directory.
    """
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    root = Path(path)
    try:
        data = json.loads(root.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read model registry: %s" % exc)
    if not isinstance(data, dict):
        raise _fail("top-level object must be a mapping")
    if data.get("schema") != GUI_MODEL_REGISTRY_SCHEMA:
        raise _fail("unsupported schema %r" % (data.get("schema"),))
    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise _fail("registry needs a non-empty model list")
    base = root.parent
    entries: list[ModelEntry] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_models):
        entries.append(_validate_entry(raw, index, base, seen))
    return entries


def _resolve(base: Path, value: str) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    return str(candidate)


def _validate_entry(
    raw: object, index: int, base: Path, seen: set[str]
) -> ModelEntry:
    if not isinstance(raw, dict):
        raise _fail("model %d must be an object" % index)
    model_id = raw.get("id")
    if not _is_non_empty_str(model_id):
        raise _fail("model %d is missing a non-empty id" % index)
    if not isinstance(model_id, str):
        raise _fail("model %d has a non-string id" % index)
    if model_id in seen:
        raise _fail("duplicate model id %r" % model_id)
    seen.add(model_id)
    display_name = raw.get("display_name")
    if not _is_non_empty_str(display_name):
        raise _fail("model %r is missing a display name" % model_id)
    if not isinstance(display_name, str):
        raise _fail("model %r has a non-string display name" % model_id)
    checkpoint = _checked_dir(
        raw.get("checkpoint"), "checkpoint", base, model_id
    )
    python = raw.get("python")
    if not isinstance(python, str) or not python:
        raise _fail("model %r is missing its worker interpreter" % model_id)
    resolved_python = _resolve(base, python)
    if not os.path.isfile(resolved_python):
        raise _fail("worker interpreter missing: %s" % resolved_python)
    if not os.access(resolved_python, os.X_OK):
        raise _fail(
            "worker interpreter is not executable: %s" % resolved_python
        )
    laya_root = raw.get("laya_root")
    if not isinstance(laya_root, str) or not laya_root:
        raise _fail("model %r is missing its LAYA runtime" % model_id)
    resolved_laya = _resolve(base, laya_root)
    if not (Path(resolved_laya) / "laya" / "__init__.py").is_file():
        raise _fail("LAYA runtime has no laya package: %s" % resolved_laya)
    expected = raw.get("expected_files")
    if not isinstance(expected, dict) or not expected:
        raise _fail("model %r needs expected_files digests" % model_id)
    digests: dict[str, str] = {}
    for name, digest_value in expected.items():
        if not isinstance(name, str) or not isinstance(digest_value, str):
            raise _fail("model %r has malformed expected_files" % model_id)
        if not DIGEST_PATTERN.match(digest_value):
            raise _fail(
                "model %r has a malformed digest for %r" % (model_id, name)
            )
        digests[name] = digest_value
    device = raw.get("device")
    if not isinstance(device, str) or not DEVICE_PATTERN.match(device):
        raise _fail(
            "model %r needs a cpu or cuda:N device, got %r" % (model_id, device)
        )
    entry: ModelEntry = {
        "id": model_id,
        "display_name": display_name,
        "checkpoint": checkpoint,
        "python": resolved_python,
        "laya_root": resolved_laya,
        "expected_files": digests,
        "device": device,
    }
    profile = raw.get("profile")
    if profile is not None:
        if not isinstance(profile, str) or not profile:
            raise _fail("model %r has a non-string profile" % model_id)
        resolved_profile = _resolve(base, profile)
        if not Path(resolved_profile).is_file():
            raise _fail(
                "capability profile is not a file: %s" % resolved_profile
            )
        loaded = load_capability_profile(resolved_profile)
        if loaded["id"] != model_id:
            raise _fail(
                "profile id %r does not match model %r"
                % (loaded["id"], model_id)
            )
        weights = digests.get("model.safetensors")
        if weights is None or loaded["checkpoint_sha256"] != weights:
            raise _fail(
                "profile checkpoint %s does not match weights %s for %r"
                % (
                    loaded["checkpoint_sha256"][:12],
                    str(weights)[:12],
                    model_id,
                )
            )
        stated_release = raw.get("release")
        if stated_release is not None and stated_release != loaded["release"]:
            raise _fail(
                "registry release %r disagrees with profile %r for %r"
                % (stated_release, loaded["release"], model_id)
            )
        stated_contract = raw.get("question_contract")
        if (
            stated_contract is not None
            and stated_contract != loaded["question_contract"]
        ):
            raise _fail(
                "registry question_contract %r disagrees with profile %r for %r"
                % (stated_contract, loaded["question_contract"], model_id)
            )
        entry["profile"] = resolved_profile
        entry["release"] = loaded["release"]
        entry["question_contract"] = loaded["question_contract"]
    release = raw.get("release")
    if release is not None:
        if not _is_non_empty_str(release):
            raise _fail("model %r has an empty release" % model_id)
        entry["release"] = release
    contract = raw.get("question_contract")
    if contract is not None:
        if not _is_non_empty_str(contract):
            raise _fail("model %r has an empty question_contract" % model_id)
        entry["question_contract"] = contract
    qualification = raw.get("qualification")
    if qualification is not None:
        if not _is_non_empty_str(qualification):
            raise _fail("model %r has an empty qualification" % model_id)
        entry["qualification"] = qualification
    return entry


def _checked_dir(value: object, label: str, base: Path, model_id: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail("model %r is missing %s" % (model_id, label))
    resolved = _resolve(base, value)
    if not Path(resolved).is_dir():
        raise _fail("%s is not a directory: %s" % (label, resolved))
    return resolved
