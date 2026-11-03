"""Portable, administrator-owned settings for the optional local demo."""

import json
import os
import sys
from pathlib import Path
from typing import Any

import siliconshield
from siliconshield.experts.registry import load_registry


def _path(value: object, base: Path, field: str) -> str:
    """Resolve an explicitly configured local path."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty path")
    path = Path(value).expanduser()
    return str(
        (base / path).resolve() if not path.is_absolute() else path.resolve()
    )


def load_config(
    path: Path | None = None, *, port: int | None = None
) -> dict[str, Any]:
    """Validate configuration without models, GPU access, or agent requests."""
    raw = json.loads(path.read_text()) if path else {}
    if not isinstance(raw, dict):
        raise ValueError("Demo configuration must be a JSON object")
    unknown = set(raw) - {
        "registry",
        "state_dir",
        "gpu",
        "port",
        "glimmer",
        "training",
    }
    if unknown:
        raise ValueError(
            "Unknown configuration fields: " + ", ".join(sorted(unknown))
        )
    base = path.resolve().parent if path else Path.cwd()
    source = Path(siliconshield.__file__).resolve().parent.parent
    state_home = Path(
        os.environ.get("XDG_STATE_HOME", "~/.local/state")
    ).expanduser()
    state = _path(
        raw.get("state_dir", str(state_home / "siliconshield/web-demo")),
        base,
        "state_dir",
    )
    # Runtime outputs must never become files in the published source tree.
    if Path(state).is_relative_to(source) or source.is_relative_to(Path(state)):
        raise ValueError("state_dir must be separate from the installed source")
    selected_port = raw.get("port", 8765) if port is None else port
    if type(selected_port) is not int or not 1024 <= selected_port <= 65535:
        raise ValueError("port must be an integer from 1024 through 65535")
    gpu = raw.get("gpu")
    if gpu is not None and (type(gpu) is not int or gpu < 0):
        raise ValueError("gpu must be a nonnegative device index or null")
    registry = raw.get("registry")
    entries = []
    if registry is not None:
        registry = _path(registry, base, "registry")
        entries = load_registry(registry)
        if any(
            entry["id"] not in {"contract-1", "contract-2"} for entry in entries
        ):
            raise ValueError(
                "This demo expects registry IDs contract-1 and contract-2"
            )
    glimmer = raw.get("glimmer")
    if glimmer is not None:
        if not isinstance(glimmer, dict) or set(glimmer) - {
            "endpoint",
            "model",
            "ssh_host",
        }:
            raise ValueError(
                "glimmer accepts endpoint, model, and optional ssh_host"
            )
        from siliconshield.web_demo.glimmer_client import LocalGlimmerClient

        LocalGlimmerClient(**glimmer)
    training = raw.get("training", {})
    if not isinstance(training, dict) or set(training) - {
        "enabled",
        "python",
        "runtime",
        "output_dir",
        "resource_root",
    }:
        raise ValueError("Invalid training configuration fields")
    enabled = training.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("training.enabled must be a boolean")
    result: dict[str, Any] = {
        "registry": registry,
        "entries": entries,
        "state_dir": state,
        "source": str(source),
        "python": sys.executable,
        "gpu": gpu,
        "port": selected_port,
        "glimmer": glimmer,
        "training_enabled": enabled,
        "toy_updates": 3,
        "training_root": None,
        "training_python": None,
        "training_runtime": None,
        "resource_root": None,
    }
    for key, output in [
        ("python", "training_python"),
        ("runtime", "training_runtime"),
        ("output_dir", "training_root"),
        ("resource_root", "resource_root"),
    ]:
        if key in training:
            result[output] = _path(training[key], base, "training." + key)
    if enabled:
        if (
            gpu is None
            or not glimmer
            or not any(e["id"] == "contract-1" for e in entries)
        ):
            raise ValueError(
                "Training needs a GPU index, Glimmer, and a Contract-1 registry entry"
            )
        if any(
            result[key] is None
            for key in (
                "training_root",
                "training_python",
                "training_runtime",
                "resource_root",
            )
        ):
            raise ValueError(
                "Training needs python, runtime, output_dir, and resource_root"
            )
        if not sys.platform.startswith("linux"):
            raise ValueError(
                "Toy training requires Linux with systemd user services"
            )
        if not os.access(result["training_python"], os.X_OK):
            raise ValueError(
                "training.python must be an executable interpreter"
            )
        if not (
            Path(result["training_runtime"]) / "laya/__init__.py"
        ).is_file():
            raise ValueError("training.runtime must contain the LAYA package")
    for key in ("state_dir", "training_root", "resource_root"):
        if result[key] is not None:
            target = Path(result[key])
            if target.is_relative_to(source) or source.is_relative_to(target):
                raise ValueError(
                    key + " must be separate from the installed source"
                )
            for entry in entries:
                checkpoint = Path(entry["checkpoint"]).resolve()
                if target.is_relative_to(
                    checkpoint
                ) or checkpoint.is_relative_to(target):
                    raise ValueError(
                        key + " must not overlap a model checkpoint"
                    )
    return result
