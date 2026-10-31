"""Muse excerpt selection for the scan tab (Qt-free, no Torch).

Uses the repository's tool-free ``MuseClient`` (local `muse` CLI, login
via the existing Muse setup, no credentials copied) with model
``muse-spark-1.3-contributor`` to pick relevant excerpts. The model only
selects; it never judges security and its output is strictly validated
(see ``scan.validate_scan_selection``). Any Muse failure raises
ValueError so callers fall back explicitly to deterministic excerpts.
"""

import os
from pathlib import Path
from typing import Any, Callable


def default_muse_audit_dir() -> Path:
    """Return the user-level Muse audit location (outside Git)."""
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "siliconshield-gui" / "muse-audit"


SELECT_TOOL = {
    "type": "function",
    "function": {
        "name": "scan.select_excerpts",
        "description": "Select candidate excerpts relevant to a requirement.",
        "parameters": {
            "type": "object",
            "properties": {
                "excerpts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "reason": {"type": "string"},
                        },
                        "required": ["index", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["excerpts"],
            "additionalProperties": False,
        },
    },
}


def select_excerpts(
    windows: list[dict[str, Any]],
    requirement: str,
    assumptions: str,
    audit_dir: Path | None = None,
    model: str = "muse-spark-1.3-contributor",
    timeout: int = 300,
    client_factory: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Ask Muse to select relevant windows; return validated entries.

    Raises ValueError when the CLI is missing, the reply has no tool
    call, or the selection fails validation. Callers must catch this
    and fall back explicitly (never silently).
    """
    from siliconshield.gui.scan import validate_scan_selection

    if client_factory is None:
        import shutil

        executable = shutil.which("muse")
        if executable is None:
            raise ValueError("`muse` CLI not found on PATH.")
        from siliconshield.research_loop.procedures.muse_client import (
            MuseClient,
        )

        target = audit_dir or default_muse_audit_dir()

        def client_factory(executable_path: str, audit: Path) -> Any:
            return MuseClient(
                executable=Path(executable_path),
                audit_dir=audit,
                model=model,
                effort="low",
                timeout=timeout,
            )

        client = client_factory(executable, target)
    else:
        client = client_factory()
    catalogue = [
        {
            "index": window["index"],
            "start_line": window["start_line"],
            "end_line": window["end_line"],
            "text": window["text"],
        }
        for window in windows
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You select RTL excerpts relevant to a security "
                "requirement. Use exactly one call to "
                "scan.select_excerpts with the excerpts that could affect "
                "the requirement. Never judge security yourself; "
                "selection only."
            ),
        },
        {
            "role": "user",
            "content": __import__("json").dumps(
                {
                    "requirement": requirement,
                    "assumptions": assumptions,
                    "windows": catalogue,
                },
                allow_nan=False,
            ),
        },
    ]
    reply = client.complete(messages, tools=[SELECT_TOOL])
    for call in reply.get("tool_calls", []):
        function = call.get("function", {})
        if function.get("name") == "scan.select_excerpts":
            return validate_scan_selection(
                function.get("arguments", {}).get("excerpts"),
                len(windows),
            )
    raise ValueError("Muse reply contained no scan.select_excerpts call.")
