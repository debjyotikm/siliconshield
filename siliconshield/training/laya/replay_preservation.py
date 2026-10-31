"""Preserve a frozen parent's predictions on explicitly named training pools."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from siliconshield.schemas import REPLAY_PRESERVATION_SCHEMA

if TYPE_CHECKING:
    from torch import Tensor


def validate_preservation(config: dict | None) -> dict | None:
    """Validate the optional, signature-bound teacher loss configuration."""
    if config is None:
        return None
    required = {
        "version",
        "weight",
        "temperature",
        "pools",
    }
    if (
        not isinstance(config, dict)
        or not required <= set(config)
        or set(config) - required - {"teacher_checkpoint"}
    ):
        raise ValueError("Replay preservation configuration is incomplete")
    if config["version"] != REPLAY_PRESERVATION_SCHEMA:
        raise ValueError("Unsupported replay preservation version")
    for name in ("weight", "temperature"):
        value = config[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Replay preservation needs positive " + name)
    pools = config["pools"]
    if (
        not isinstance(pools, list)
        or not pools
        or not all(isinstance(name, str) and name for name in pools)
        or len(set(pools)) != len(pools)
    ):
        raise ValueError("Replay preservation needs distinct training pools")
    if "teacher_checkpoint" in config and (
        not isinstance(config["teacher_checkpoint"], str)
        or not config["teacher_checkpoint"].strip()
    ):
        raise ValueError("Preservation teacher checkpoint must name a path")
    return dict(config, pools=list(pools))


def teacher_checkpoint(config: dict, initial_checkpoint: str) -> str:
    """Resolve a teacher independently of the student's initialization.

    Consolidation may start from a newly fitted student while preserving an
    earlier accepted model. Legacy recipes use the student's initial model.
    The caller hashes the resolved checkpoint into the execution identity.
    """
    checked = validate_preservation(config)
    if checked is None:
        raise ValueError("A teacher requires preservation configuration")
    value = checked.get("teacher_checkpoint", initial_checkpoint)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Preservation teacher checkpoint must name a path")
    return value


def preservation_loss(
    student: Tensor,
    teacher: Tensor,
    targets: Tensor,
    eligible: Tensor,
    temperature: float,
) -> Tensor:
    """Return per-row softened KL from a correct, detached frozen teacher.

    Only explicitly eligible training rows with a correct teacher prediction
    contribute. The caller applies the ordinary accumulation and validity
    weights. No development data or teacher-generated labels are introduced.
    """
    # Torch is optional for the CPU-only orchestration installation.
    from torch.nn import functional

    if student.shape != teacher.shape or student.shape != targets.shape:
        raise ValueError("Teacher, student and target shapes must match")
    if student.ndim != 2 or eligible.shape != student.shape[:1]:
        raise ValueError("Teacher loss needs a per-row eligibility mask")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Teacher temperature must be finite and positive")
    fixed = teacher.detach().float()
    mask = eligible.bool() & (fixed.argmax(-1) == targets.argmax(-1))
    divergences = functional.kl_div(
        functional.log_softmax(student.float() / temperature, dim=-1),
        functional.log_softmax(fixed / temperature, dim=-1),
        reduction="none",
        log_target=True,
    ).sum(-1)
    return divergences.clamp_min(0) * temperature**2 * mask


def preservation_diagnostics(
    student: Tensor,
    teacher: Tensor,
    targets: Tensor,
    eligible: Tensor,
    valid: Tensor,
) -> dict:
    """Count rows under the same mask `preservation_loss` applies.

    Pool rows are explicitly eligible, valid rows. Matched rows add the
    frozen-teacher-correct condition. Agreed rows add student-teacher
    argmax agreement on matched rows. Counts only; no RNG, no gradients.
    """
    fixed = teacher.detach().float()
    teacher_class = fixed.argmax(-1)
    pool = eligible.bool() & (valid > 0)
    match = pool & (teacher_class == targets.argmax(-1))
    agree = match & (student.detach().float().argmax(-1) == teacher_class)
    return {
        "pool_rows": int(pool.sum()),
        "teacher_match_rows": int(match.sum()),
        "student_teacher_agree_rows": int(agree.sum()),
    }
