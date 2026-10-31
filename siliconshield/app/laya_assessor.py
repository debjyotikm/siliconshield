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
import os
import subprocess
from pathlib import Path
from typing import TypedDict

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
    """The exact input format used by the security-training builders."""
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
        payload = json.dumps(
            {
                "state": state_text(requirement, assumptions, rtl),
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
        usage = answer.get("usage") or {}
        dropped_tokens = usage.get("state_tokens_dropped")
        coverage = {
            "input_tokens": usage.get("input_tokens"),
            "state_tokens": usage.get("state_tokens"),
            "state_tokens_dropped": dropped_tokens,
            "truncated": bool(usage.get("truncated")),
            "truncated_questions": usage.get("truncated_questions") or [],
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
                "probabilities": answer.get("probabilities"),
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
            "probabilities": answer.get("probabilities"),
            "usage": usage,
            "coverage": coverage,
        }


# Public neutral name for the security-model backend.
SecurityAssessor = LayaBackend
# Legacy alias kept for existing consumers and tests.
ExternalLayaAssessor = LayaBackend
