"""Content identities shared by workers, profiles, and arbitration."""

import copy
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
    if not isinstance(text, str):
        raise ValueError("State digest needs text")
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
    contract = value.get("question_contract")
    if not isinstance(contract, str) or not contract.strip():
        raise ValueError("Execution identity needs a question contract")
    if contract != contract.strip():
        raise ValueError("Execution identity contract must be exact")
    settings = value.get("inference")
    if not isinstance(settings, dict):
        raise ValueError("Execution identity needs inference settings")
    for key in ("backend", "precision", "torch_version", "device_type"):
        text = settings.get(key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Inference identity needs " + key)
        if text != text.strip():
            raise ValueError("Inference identity %s must be exact" % key)
    for key in ("batch_size", "max_len", "head_max_len"):
        if type(settings.get(key)) is not int or settings[key] <= 0:
            raise ValueError("Inference identity needs a positive " + key)
    if type(settings.get("sort_by_length")) is not bool:
        raise ValueError("Inference identity needs sort_by_length")
    if settings.get("option_order") != [0, 1, 2]:
        raise ValueError("Expert inference requires identity option order")
    # This also rejects NaN in any extra numerical settings.
    try:
        digest(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Execution identity has non-JSON settings: %s" % exc
        ) from exc
    # Deep copy: nested inference settings must not alias the
    # caller's payload after validation.
    return copy.deepcopy(dict(value))


SECTION_HEADERS = frozenset({"REQUIREMENT", "ASSUMPTIONS", "RTL"})


def assemble_state(requirement: str, assumptions: str, rtl: str) -> str:
    """Build one judged state from separate fields.

    Emits the exact training-renderer bytes (single newlines, no
    trailing newline): the router must feed the model the same
    bytes it trained on, or state digests stop matching training
    rows and inference drifts off-distribution.

    Construction is strict (non-empty sections) while parsing
    stays lenient: a state with empty sections still splits, but
    only complete states are built for inference.

    The requirement and assumptions must not smuggle an exact
    section-header line: sections split on the first RTL header,
    so a smuggled header would truncate the judged requirement
    and pollute the judged RTL.
    """
    for name, value in (
        ("requirement", requirement),
        ("assumptions", assumptions),
        ("rtl", rtl),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("assembled state needs non-empty " + name)
    for name, value in (
        ("requirement", requirement),
        ("assumptions", assumptions),
    ):
        smuggled = SECTION_HEADERS.intersection(value.splitlines())
        if smuggled:
            raise ValueError(
                name
                + " must not contain a section header line: "
                + ",".join(sorted(smuggled))
            )
    return "REQUIREMENT\n%s\nASSUMPTIONS\n%s\nRTL\n%s" % (
        requirement,
        assumptions,
        rtl,
    )


def state_sections(state: str) -> tuple[str, str, str]:
    """Parse one requirement, optional assumptions, and a nonempty RTL body.

    Headers are exact complete lines. Only the pre-RTL headers are parsed,
    so comments in the RTL cannot replace the judged requirement.
    """
    if not isinstance(state, str):
        raise ValueError("State sections need text")
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
