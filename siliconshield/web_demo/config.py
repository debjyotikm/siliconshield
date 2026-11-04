"""Portable, administrator-owned settings for the optional local demo."""

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
    if path is not None and not isinstance(path, (str, Path)):
        raise ValueError("Demo configuration path must be a path")
    path = Path(path) if path else None
    raw = json.loads(path.read_text()) if path else {}
    if not isinstance(raw, dict):
        raise ValueError("Demo configuration must be a JSON object")
    unknown = set(raw) - {
        "registry",
        "state_dir",
        "port",
        "glimmer",
        "arbiter",
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
    kinds = {entry.get("kind", "local-worker") for entry in entries}
    if "adapter" in kinds and "glimmer" in kinds:
        raise ValueError(
            "One demo serves one engine: register glimmer or adapter "
            "specialists, not both."
        )
    adapter_mode = "adapter" in kinds
    glimmer = raw.get("glimmer")
    if glimmer is not None:
        if not isinstance(glimmer, dict) or set(glimmer) - {
            "endpoint",
            "model",
            "ssh_host",
            "fingerprint",
        }:
            raise ValueError(
                "glimmer accepts endpoint, model, fingerprint, "
                "and optional ssh_host"
            )
        if not isinstance(glimmer.get("endpoint"), str) or not isinstance(
            glimmer.get("model"), str
        ):
            raise ValueError("glimmer needs endpoint and model strings")
        fingerprint = glimmer.get("fingerprint")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
        ):
            raise ValueError("glimmer needs the 64-hex serving fingerprint")
        from siliconshield.web_demo.glimmer_client import LocalGlimmerClient

        if adapter_mode:
            if glimmer.get("ssh_host") is not None:
                raise ValueError(
                    "Adapter endpoints are loopback-direct; ssh_host "
                    "belongs on the arbiter."
                )
            _check_score_endpoint(glimmer["endpoint"])
        else:
            LocalGlimmerClient(
                glimmer["endpoint"],
                glimmer["model"],
                ssh_host=glimmer.get("ssh_host"),
            )
    arbiter = raw.get("arbiter")
    if arbiter is not None:
        if not isinstance(arbiter, dict) or set(arbiter) - {
            "endpoint",
            "model",
            "ssh_host",
            "engine",
        }:
            raise ValueError(
                "arbiter accepts engine, endpoint, model, and optional ssh_host"
            )
        if arbiter.get("engine") == "spark":
            if not isinstance(arbiter.get("model"), str):
                raise ValueError("spark arbiter needs a model string")
        else:
            if not isinstance(arbiter.get("endpoint"), str) or not isinstance(
                arbiter.get("model"), str
            ):
                raise ValueError("arbiter needs endpoint and model strings")
            from siliconshield.web_demo.glimmer_client import (
                LocalGlimmerClient,
            )

            LocalGlimmerClient(
                arbiter["endpoint"],
                arbiter["model"],
                ssh_host=arbiter.get("ssh_host"),
            )
    result: dict[str, Any] = {
        "registry": registry,
        "entries": entries,
        "state_dir": state,
        "source": str(source),
        "python": sys.executable,
        "port": selected_port,
        "glimmer": glimmer,
        "arbiter": arbiter,
    }
    return result


def _check_score_endpoint(endpoint: str) -> None:
    """Require a loopback scoring URL ending in /score."""
    parsed = urlparse(endpoint)
    if (
        parsed.hostname not in ("127.0.0.1", "localhost", "::1")
        or parsed.scheme != "http"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith("/score")
    ):
        raise ValueError(
            "Adapter endpoint must be loopback HTTP ending in /score"
        )
