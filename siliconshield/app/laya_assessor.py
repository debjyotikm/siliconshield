"""SiliconShield-RTL three-way assessment behind a small interface.

The workflow never imports LAYA core directly. A real assessment runs in
the environment that owns the LAYA runtime + torch (subprocess worker,
`laya_worker.py`); when no assessor is configured the result is an
explicit `unavailable` with no model judgment. Executable checks
determine the workflow conclusion.

Identity distinguishes three things: `model_id` names the product
family (`siliconshield-rtl`), `checkpoint` names the configured
artifact path, and `checkpoint_fingerprint` binds that artifact's full
content hashes. `runtime` names the LAYA runtime that loads it.
"""

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import TypedDict

from siliconshield.experts.identity import SECTION_HEADERS
from siliconshield.training.laya.question_contracts import (
    LABELS as _CONTRACT_LABELS,
)

LABELS = tuple(_CONTRACT_LABELS)

SECURITY_MODEL_ID = "siliconshield-rtl"
BACKEND_RUNTIME = "laya"


class CheckpointManifest(TypedDict):
    """Full checkpoint contents and their shared manifest digest."""

    files: dict[str, str]
    fingerprint: str


def state_text(requirement, assumptions, rtl):
    """The exact input format used by the security-training builders.

    The byte format is frozen: trained models read it, and
    experts.identity.assemble_state emits these same bytes so the
    router and the backend judge identical inputs. The
    requirement and assumptions must not smuggle an exact
    section-header line, which would truncate the judged
    requirement and pollute the judged RTL.
    """
    for name, value in (
        ("requirement", requirement),
        ("assumptions", assumptions),
    ):
        smuggled = SECTION_HEADERS.intersection(str(value).splitlines())
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


def _sha(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    """Hash a checkpoint file or directory; bind hashes with a fingerprint.

    Returns the adapter-manifest shape: {"files": {name: sha256},
    "fingerprint": digest}. The digest matches
    research_loop.episodes.contracts.digest, so manifests compare equal
    across the repository. Raises ValueError when identity cannot be
    established; partial hashing proves nothing, so every file counts.
    """
    root = Path(path)
    files = {}
    if root.is_file():
        files[root.name] = _sha(root)
    elif root.is_dir():
        for item in sorted(root.rglob("*")):
            if item.is_file():
                files[str(item.relative_to(root))] = _sha(item)
    else:
        raise ValueError("Checkpoint does not exist: " + str(root))
    if not files:
        raise ValueError("Checkpoint holds no files: " + str(root))
    fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return {"files": files, "fingerprint": fingerprint}


def _identity_failure(
    result: dict[str, object], error: Exception
) -> dict[str, object]:
    """Refuse a judgment when its checkpoint identity cannot be verified."""
    detail = f"{type(error).__name__}: {error}"
    return {
        **result,
        "checkpoint_fingerprint": None,
        "checkpoint_error": detail,
        "status": "error",
        "label": None,
        "probabilities": None,
        "detail": "Checkpoint identity refused: " + detail,
    }


class UnavailableAssessor:
    name = "unavailable"

    def assess(self, requirement, assumptions, rtl):
        return {
            "assessor": self.name,
            "model_id": SECURITY_MODEL_ID,
            "runtime": BACKEND_RUNTIME,
            "checkpoint": None,
            "checkpoint_fingerprint": None,
            "checkpoint_error": None,
            "status": "unavailable",
            "label": None,
            "probabilities": None,
            "detail": "No security-model assessor configured (set "
            "SILICONSHIELD_SECURITY_PYTHON and "
            "SILICONSHIELD_SECURITY_CHECKPOINT, or legacy "
            "SILICONSHIELD_LAYA_*); requirement not model-assessed.",
        }


class LayaBackend:
    """Run laya_worker.py with a configured Python/checkpoint.

    The public name is `SecurityAssessor`; this class name keeps the
    LAYA-runtime backend explicit. Records the exact command run.
    Each assessment resolves and hashes its checkpoint before spawning
    a worker, then checks the same resolved path after execution. Identity
    is local to that call; an unreadable or changed checkpoint refuses
    assessment.
    """

    def __init__(self, python, checkpoint, laya_root=None, timeout=1200):
        self.python, self.checkpoint = python, Path(checkpoint)
        self.laya_root, self.timeout = laya_root, timeout
        self.name = "laya-external"

    @classmethod
    def from_environment(cls):
        python = os.environ.get(
            "SILICONSHIELD_SECURITY_PYTHON"
        ) or os.environ.get("SILICONSHIELD_LAYA_PYTHON")
        checkpoint = os.environ.get(
            "SILICONSHIELD_SECURITY_CHECKPOINT"
        ) or os.environ.get("SILICONSHIELD_LAYA_CHECKPOINT")
        if not python or not checkpoint:
            return None
        return cls(python, checkpoint, os.environ.get("LAYA_ROOT"))

    def assess(
        self, requirement: str, assumptions: str, rtl: str
    ) -> dict[str, object]:
        """Assess one input, rejecting checkpoint changes during execution.

        A replacement between calls is allowed and gets a new manifest.
        The resolved path is passed to the worker so retargeting the
        configured symlink cannot redirect a running assessment.
        """
        result: dict[str, object] = {
            "assessor": self.name,
            "model_id": SECURITY_MODEL_ID,
            "runtime": BACKEND_RUNTIME,
            "checkpoint": str(self.checkpoint.absolute()),
            "checkpoint_fingerprint": None,
            "checkpoint_error": None,
        }
        try:
            checkpoint = self.checkpoint.resolve()
            result["checkpoint"] = str(checkpoint)
            manifest = checkpoint_manifest(checkpoint)
        except (OSError, ValueError, RuntimeError) as exc:
            return _identity_failure(result, exc)
        result["checkpoint_fingerprint"] = manifest
        worker = Path(__file__).with_name("laya_worker.py")
        try:
            state = state_text(requirement, assumptions, rtl)
        except ValueError as exc:
            # A smuggled section header rejects this assessment; it
            # must not abort the run without a report.
            return {
                **result,
                "status": "error",
                "label": None,
                "probabilities": None,
                "detail": "Invalid security-model input: %s" % exc,
            }
        payload = json.dumps(
            {
                "state": state,
                "checkpoint": str(checkpoint),
            }
        )
        env = dict(os.environ)
        if self.laya_root:
            env["LAYA_ROOT"] = self.laya_root
            env["PYTHONPATH"] = (
                self.laya_root + os.pathsep + env.get("PYTHONPATH", "")
            )
        command = [self.python, str(worker)]
        result["command"] = command
        try:
            run = subprocess.run(
                command,
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                **result,
                "status": "error",
                "label": None,
                "probabilities": None,
                "detail": "%s: %s" % (type(exc).__name__, exc),
            }
        result.update(
            {
                "exit_code": run.returncode,
                "stderr_tail": run.stderr.strip().splitlines()[-3:],
            }
        )
        try:
            if checkpoint_manifest(checkpoint) != manifest:
                raise ValueError(
                    "Checkpoint changed during assessment; "
                    "use a stable checkpoint generation and retry."
                )
        except (OSError, ValueError) as exc:
            return _identity_failure(result, exc)
        if run.returncode != 0:
            return {
                **result,
                "status": "error",
                "label": None,
                "probabilities": None,
                "detail": "Security-model worker failed "
                "(laya_worker.py); see stderr_tail",
            }
        try:
            answer = json.loads(run.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {
                **result,
                "status": "error",
                "label": None,
                "probabilities": None,
                "detail": "Security-model worker returned no JSON result",
            }
        if answer.get("label") not in LABELS:
            return {
                **result,
                "status": "error",
                "label": None,
                "probabilities": None,
                "detail": "Security-model worker returned an unknown label",
            }
        probabilities = answer.get("probabilities")
        if (
            not isinstance(probabilities, dict)
            or set(probabilities) != set(LABELS)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
                for value in probabilities.values()
            )
            # Same unit-sum rule as the router path: a verdict whose
            # scores do not form a distribution is rejected, not shown.
            or abs(sum(probabilities.values()) - 1.0) > 1e-3
            or answer.get("label")
            != max(LABELS, key=lambda name: probabilities[name])
        ):
            return {
                **result,
                "status": "error",
                "label": None,
                "probabilities": None,
                "detail": "Security-model worker returned "
                "invalid probabilities",
            }
        usage = answer.get("usage")
        if usage is None:
            usage = {}
        if not isinstance(usage, dict):
            return {
                **result,
                "status": "error",
                "label": None,
                "probabilities": None,
                "detail": "Security-model worker returned a non-object usage",
            }
        dropped_tokens = usage.get("state_tokens_dropped")
        truncated_questions = usage.get("truncated_questions")
        if truncated_questions is None:
            truncated_questions = []
        if (
            not isinstance(truncated_questions, list)
            or (
                dropped_tokens is not None
                and (
                    isinstance(dropped_tokens, bool)
                    or not isinstance(dropped_tokens, (int, float))
                )
            )
            or any(
                not isinstance(token, (int, float)) or isinstance(token, bool)
                for token in (
                    usage.get("input_tokens"),
                    usage.get("state_tokens"),
                )
                if token is not None
            )
        ):
            return {
                **result,
                "status": "error",
                "label": None,
                "probabilities": None,
                "detail": "Security-model worker returned "
                "an invalid usage report",
            }
        coverage = {
            "input_tokens": usage.get("input_tokens"),
            "state_tokens": usage.get("state_tokens"),
            "state_tokens_dropped": dropped_tokens,
            "truncated": bool(usage.get("truncated")),
            "truncated_questions": truncated_questions,
        }
        truncated = (
            coverage["truncated"]
            or bool(coverage["truncated_questions"])
            or (dropped_tokens or 0) > 0
        )
        if truncated:
            # The model did not see the complete input; the training pipeline
            # rejects such inputs, and so does the application.
            return {
                **result,
                "status": "truncated",
                "label": None,
                "probabilities": probabilities,
                "coverage": coverage,
                "detail": "Security-model input was truncated "
                "(LAYA runtime); assessment rejected as "
                "incomplete. The report preserves the full RTL, "
                "but the model did not receive it.",
            }
        return {
            **result,
            "status": "assessed",
            "label": answer["label"],
            "probabilities": probabilities,
            "usage": usage,
            "coverage": coverage,
        }


# Public neutral name for the security-model backend.
SecurityAssessor = LayaBackend
# Legacy alias kept for existing consumers and tests.
ExternalLayaAssessor = LayaBackend
