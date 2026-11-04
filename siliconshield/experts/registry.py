"""Local model registry: validated, file-based, single-machine only.

Registries live outside Git (see the ``--models`` option). Paths may be
absolute or relative to the registry file itself. Validation checks shape,
duplicate ids, executable worker interpreters, the LAYA runtime layout,
and optional capability profiles, release versions, question contracts,
and qualification references, but never imports LAYA or touches the
GPU: artifact hashes are verified by the worker process at load time,
on the machine that serves them.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import TypedDict, Union

DEVICE_PATTERN = re.compile(r"(cpu|cuda:\d+)\Z")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class ModelEntry(TypedDict, total=False):
    """One validated registry entry with resolved absolute paths."""

    id: str
    display_name: str
    kind: str
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
    # Entries promise absolute paths: a relative survivor would
    # misresolve after any working-directory change between
    # registry load and worker spawn. Normalize without resolving
    # symlinks: the configured python is routinely a venv link,
    # and dereferencing it would lose the venv's packages.
    return os.path.abspath(candidate)


def _validate_entry(
    raw: object, index: int, base: Path, seen: set[str]
) -> ModelEntry:
    if not isinstance(raw, dict):
        raise _fail("model %d must be an object" % index)
    unknown = set(raw) - set(ModelEntry.__annotations__)
    if unknown:
        raise _fail("model %d has unknown keys: %s" % (index, sorted(unknown)))
    model_id = raw.get("id")
    if not _is_non_empty_str(model_id):
        raise _fail("model %d is missing a non-empty id" % index)
    if not isinstance(model_id, str):
        raise _fail("model %d has a non-string id" % index)
    if model_id != model_id.strip():
        raise _fail("model id %r must be exact" % model_id)
    if model_id in seen:
        raise _fail("duplicate model id %r" % model_id)
    seen.add(model_id)
    display_name = raw.get("display_name")
    if not _is_non_empty_str(display_name):
        raise _fail("model %r is missing a display name" % model_id)
    if not isinstance(display_name, str):
        raise _fail("model %r has a non-string display name" % model_id)
    kind = raw.get("kind", "local-worker")
    if kind not in ("local-worker", "glimmer", "adapter"):
        raise _fail("model %r has unknown kind %r" % (model_id, kind))
    checkpoint = _checked_dir(
        raw.get("checkpoint"), "checkpoint", base, model_id
    )
    resolved_python = ""
    resolved_laya = ""
    if kind in ("glimmer", "adapter"):
        # A remotely served specialist is used behind its skill
        # document: it must not claim a local worker interpreter
        # or runtime.
        if raw.get("python") is not None:
            raise _fail(
                "model %r is a %s entry and must not name "
                "a worker interpreter" % (model_id, kind)
            )
        if raw.get("laya_root") is not None:
            raise _fail(
                "model %r is a %s entry and must not name "
                "a LAYA runtime" % (model_id, kind)
            )
    else:
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
        if not name.strip():
            raise _fail("model %r has an empty expected filename" % model_id)
        if name != name.strip():
            # Bind compares exact file names; padded names would
            # pass here and fail there with a misleading message.
            raise _fail("model %r has a padded expected filename" % model_id)
        if not DIGEST_PATTERN.match(digest_value):
            raise _fail(
                "model %r has a malformed digest for %r" % (model_id, name)
            )
        digests[name] = digest_value
    if kind == "glimmer":
        for name in ("model.safetensors", "SKILL.md"):
            if name not in digests:
                raise _fail(
                    "model %r is a glimmer entry and needs an "
                    "expected digest for %s" % (model_id, name)
                )
    if kind == "adapter":
        for name in ("adapter_model.safetensors", "SKILL.md"):
            if name not in digests:
                raise _fail(
                    "model %r is an adapter entry and needs an "
                    "expected digest for %s" % (model_id, name)
                )
    if kind in ("glimmer", "adapter"):
        skill_path = Path(checkpoint) / "SKILL.md"
        try:
            skill_bytes = skill_path.read_bytes()
        except OSError as exc:
            raise _fail(
                "model %r names an unreadable skill: %s" % (model_id, exc)
            )
        actual = hashlib.sha256(skill_bytes).hexdigest()
        if actual != digests["SKILL.md"]:
            raise _fail(
                "model %r skill digest mismatch: registry pins a "
                "different SKILL.md" % (model_id,)
            )
    device = raw.get("device")
    if not isinstance(device, str) or not device.strip():
        raise _fail("model %r needs a device string" % (model_id,))
    if kind in ("glimmer", "adapter"):
        device = device.strip()
    elif not DEVICE_PATTERN.match(device):
        raise _fail(
            "model %r needs a cpu or cuda:N device, got %r" % (model_id, device)
        )
    entry: ModelEntry = {
        "id": model_id,
        "display_name": display_name,
        "checkpoint": checkpoint,
        "expected_files": digests,
        "device": device,
    }
    if kind in ("glimmer", "adapter"):
        entry["kind"] = kind
    else:
        entry["python"] = resolved_python
        entry["laya_root"] = resolved_laya
    profile = raw.get("profile")
    if profile is not None:
        if not isinstance(profile, str) or not profile:
            raise _fail("model %r has a non-string profile" % model_id)
        resolved_profile = _resolve(base, profile)
        if not Path(resolved_profile).is_file():
            raise _fail(
                "capability profile is not a file: %s" % resolved_profile
            )
        # Content checks stay with the service: one corrupt profile
        # file must not block loading unrelated registry entries.
        entry["profile"] = resolved_profile
    release = raw.get("release")
    if release is not None:
        if not _is_non_empty_str(release):
            raise _fail("model %r has an empty release" % model_id)
        entry["release"] = release.strip()
    contract = raw.get("question_contract")
    if contract is not None:
        if not _is_non_empty_str(contract):
            raise _fail("model %r has an empty question_contract" % model_id)
        entry["question_contract"] = contract.strip()
    qualification = raw.get("qualification")
    if qualification is not None:
        if not _is_non_empty_str(qualification):
            raise _fail("model %r has an empty qualification" % model_id)
        entry["qualification"] = qualification.strip()
    return entry


def _checked_dir(value: object, label: str, base: Path, model_id: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail("model %r is missing %s" % (model_id, label))
    resolved = _resolve(base, value)
    if not Path(resolved).is_dir():
        raise _fail("%s is not a directory: %s" % (label, resolved))
    return resolved
