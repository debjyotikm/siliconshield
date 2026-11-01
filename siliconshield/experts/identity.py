"""Content identities shared by workers, profiles, and arbitration."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def digest(value: Any) -> str:
    """Hash a canonical finite JSON value."""
    blob = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def text_digest(text: str) -> str:
    """Hash exact input bytes, without whitespace normalization."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_digest(value: Any) -> bool:
    """Whether a value is a lowercase SHA-256 digest."""
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def runtime_digest(root: str | Path) -> str:
    """Hash the Python implementation of the configured LAYA runtime."""
    package = Path(root) / "laya"
    if not (package / "__init__.py").is_file():
        raise ValueError("Configured LAYA runtime has no package")
    files = {
        str(path.relative_to(package)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    return digest(files)


def validate_execution_identity(value: Any) -> dict[str, Any]:
    """Require the artifact, question, runtime, and numerical protocol."""
    if not isinstance(value, dict):
        raise ValueError("Worker lacks execution identity")
    for key in (
        "checkpoint_sha256",
        "export_sha256",
        "question_sha256",
        "runtime_sha256",
    ):
        if not is_digest(value.get(key)):
            raise ValueError("Invalid execution identity " + key)
    if (
        not isinstance(value.get("question_contract"), str)
        or not value["question_contract"]
    ):
        raise ValueError("Execution identity needs a question contract")
    settings = value.get("inference")
    if not isinstance(settings, dict):
        raise ValueError("Execution identity needs inference settings")
    for key in ("backend", "precision", "torch_version", "device_type"):
        if not isinstance(settings.get(key), str) or not settings[key]:
            raise ValueError("Inference identity needs " + key)
    for key in ("batch_size", "max_len", "head_max_len"):
        if type(settings.get(key)) is not int or settings[key] <= 0:
            raise ValueError("Inference identity needs a positive " + key)
    if type(settings.get("sort_by_length")) is not bool:
        raise ValueError("Inference identity needs sort_by_length")
    if settings.get("option_order") != [0, 1, 2]:
        raise ValueError("Expert inference requires identity option order")
    # This also rejects NaN in any extra numerical settings.
    digest(value)
    return dict(value)


def state_sections(state: str) -> tuple[str, str, str]:
    """Parse one requirement, optional assumptions, and a nonempty RTL body.

    Headers are exact complete lines. Only the pre-RTL headers are parsed,
    so comments in the RTL cannot replace the judged requirement.
    """
    lines = state.splitlines()
    if not lines or lines[0] != "REQUIREMENT" or "RTL" not in lines:
        raise ValueError(
            "Expected REQUIREMENT, optional ASSUMPTIONS, and RTL sections"
        )
    rtl_at = lines.index("RTL")
    prefix = lines[1:rtl_at]
    if "REQUIREMENT" in prefix or prefix.count("ASSUMPTIONS") > 1:
        raise ValueError("Ambiguous state section headers")
    if "ASSUMPTIONS" in prefix:
        split = prefix.index("ASSUMPTIONS")
        requirement = "\n".join(prefix[:split]).strip()
        assumptions = "\n".join(prefix[split + 1 :]).strip()
    else:
        requirement, assumptions = "\n".join(prefix).strip(), ""
    rtl = "\n".join(lines[rtl_at + 1 :]).strip()
    if not requirement or not rtl:
        raise ValueError("Requirement and RTL must be nonempty")
    return requirement, assumptions, rtl
