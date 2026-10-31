"""One-command local registry setup for the desktop GUI.

This module is Qt-free and Torch-free so ``python -m siliconshield.gui
--init`` works in lightweight environments. It never touches the GPU
and never writes inside Git: registries live in the user config dir
or an explicit ``--models`` path.
"""

import json
import os
from pathlib import Path
from typing import Any


def default_registry_path() -> Path:
    """Return the default user-level registry location (outside Git)."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home and config_home.strip():
        base = Path(config_home.strip()).expanduser()
    else:
        base = Path.home() / ".config"
    return base / "siliconshield" / "models.local.json"


def build_entry(
    model_id: str,
    display_name: str,
    checkpoint: str,
    python: str,
    laya_root: str,
    device: str,
) -> dict[str, Any]:
    """Build one validated registry entry, hashing the checkpoint.

    Raises ValueError with an actionable message when paths are
    missing or digests cannot be established. The caller writes the
    surrounding registry document.
    """
    from siliconshield.app.laya_assessor import checkpoint_manifest

    if not model_id.strip():
        raise ValueError("Model id must be non-empty.")
    if not display_name.strip():
        raise ValueError("Display name must be non-empty.")
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise ValueError("Checkpoint does not exist: %s" % checkpoint_path)
    python_path = Path(python).expanduser()
    if not python_path.is_file():
        raise ValueError("Worker interpreter missing: %s" % python_path)
    if not os.access(python_path, os.X_OK):
        raise ValueError(
            "Worker interpreter is not executable: %s" % python_path
        )
    laya_path = Path(laya_root).expanduser()
    if not (laya_path / "laya" / "__init__.py").is_file():
        raise ValueError("LAYA runtime has no laya package: %s" % laya_path)
    if device != "cpu" and (
        not device.startswith("cuda:") or not device.split(":", 1)[1].isdigit()
    ):
        raise ValueError("Device must be cpu or cuda:N, got %r." % (device,))
    manifest = checkpoint_manifest(checkpoint_path)
    files = manifest["files"]
    if not files:
        raise ValueError("Checkpoint holds no files.")
    return {
        "id": model_id,
        "display_name": display_name,
        "checkpoint": str(checkpoint_path),
        "python": str(python_path),
        "laya_root": str(laya_path),
        "expected_files": dict(files),
        "device": device,
    }


def write_registry(path: Path, entries: list[dict[str, Any]]) -> Path:
    """Write entries to path with the registry schema; return the path."""
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    document = {"schema": GUI_MODEL_REGISTRY_SCHEMA, "models": entries}
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Validate what was written before reporting success.
    from siliconshield.gui import registry as registry_module

    registry_module.load_registry(path)
    return path


def init_from_args(args: Any) -> Path:
    """Create a registry from parsed ``--init`` args; return its path."""
    checkpoint = (
        args.checkpoint
        or os.environ.get("SILICONSHIELD_SECURITY_CHECKPOINT")
        or os.environ.get("SILICONSHIELD_LAYA_CHECKPOINT")
    )
    python = (
        args.python
        or os.environ.get("SILICONSHIELD_SECURITY_PYTHON")
        or os.environ.get("SILICONSHIELD_LAYA_PYTHON")
        or os.environ.get("LAYA_PYTHON")
    )
    laya_root = args.laya_root or os.environ.get("LAYA_ROOT")
    missing = []
    if not checkpoint:
        missing.append("--checkpoint or SILICONSHIELD_SECURITY_CHECKPOINT")
    if not python:
        missing.append("--python or SILICONSHIELD_SECURITY_PYTHON")
    if not laya_root:
        missing.append("--laya-root or LAYA_ROOT")
    if missing:
        raise ValueError("Missing required locations: %s." % ", ".join(missing))
    if (
        not isinstance(checkpoint, str)
        or not isinstance(python, str)
        or not isinstance(laya_root, str)
    ):
        raise ValueError("Registry locations must be strings.")
    entry = build_entry(
        model_id=str(args.model_id),
        display_name=str(args.display_name),
        checkpoint=checkpoint,
        python=python,
        laya_root=laya_root,
        device=str(args.device),
    )
    # Prove the GUI/SDK input path (state formatting, contract
    # bytes, token budget) before recording a usable registry.
    from siliconshield.gui.qualification import (
        build_case_states,
        qualify_states,
    )

    qualify_states(build_case_states())
    models_flag = getattr(args, "models", None)
    env_models = os.environ.get("SILICONSHIELD_GUI_MODELS")
    if isinstance(models_flag, str) and models_flag.strip():
        target = Path(models_flag).expanduser()
    elif isinstance(env_models, str) and env_models.strip():
        target = Path(env_models.strip()).expanduser()
    else:
        target = default_registry_path()
    entries: list[dict[str, Any]] = [entry]
    try:
        from siliconshield.gui import registry as registry_module

        if target.is_file():
            existing = [
                dict(item) for item in registry_module.load_registry(target)
            ]
            merged = False
            for index, item in enumerate(existing):
                if item.get("id") == entry["id"]:
                    existing[index] = entry
                    merged = True
            if not merged:
                existing.append(entry)
            entries = existing
    except ValueError:
        entries = [entry]
    return write_registry(target, entries)
