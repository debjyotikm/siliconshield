"""Expose the LAYA trainer path and job-submission interface.

The scripts live under ``siliconshield/training/laya/`` and run by file path.
"""

from pathlib import Path
from typing import Any

from siliconshield.research_loop.execution.security_jobs import (
    SecurityJobs,
    command_for,
)
from siliconshield.training.laya import dataset_versions, development_views

TRAINER = (
    Path(__file__).resolve().parents[2]
    / "training"
    / "laya"
    / "security_train.py"
)


def verify_dataset_version(version_root: Path, version: str) -> dict[str, Any]:
    """Re-hash every pinned split of one dataset version."""
    return dataset_versions.verify_version(version_root, version)


def dataset_version_accepted(version_root: Path, version: str) -> bool:
    """Report whether a version carries a training-acceptance marker."""
    return dataset_versions.is_accepted(version_root, version)


def normalize_family_caps(caps: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Normalize flat or nested family caps to population sections."""
    normalized: dict[str, dict[str, int]] = (
        development_views.normalize_family_caps(caps)
    )
    return normalized


def check_selection_reference(reference: Any) -> None:
    """Validate a frozen trainer selection reference structurally."""
    development_views.check_selection_reference(reference)


def build_selection_reference_from_results(
    results: Any,
    policy: Any,
    report_digest: str,
) -> dict[str, Any]:
    """Freeze adapted incumbent results plus research gates."""
    reference: dict[str, Any] = (
        development_views.build_selection_reference_from_results(
            results, policy, report_digest
        )
    )
    return reference


def new_coverage_gain_ok(
    new_accuracy: float, old_accuracy: float, min_gain_pp: float
) -> bool:
    """Shared trainer/campaign new-coverage improvement gate."""
    ok: bool = development_views.new_coverage_gain_ok(
        new_accuracy, old_accuracy, min_gain_pp
    )
    return ok


RANKING_V1_TRANSFER = development_views.RANKING_V1_TRANSFER
RANKING_V2_RETENTION = development_views.RANKING_V2_RETENTION


def ranking_version(policy: Any) -> str:
    """Resolve the ranking version; absent means v1-transfer."""
    version: str = development_views.ranking_version(policy)
    return version


def retention_rank_key(
    false_safe: int,
    correct: int,
    pairs_correct: int,
    retention_nll: float,
    transfer_nll: float,
) -> tuple[int, int, int, float, float]:
    """Post-feasibility ordering shared by trainer and campaign."""
    key: tuple[int, int, int, float, float] = (
        development_views.retention_rank_key(
            false_safe, correct, pairs_correct, retention_nll, transfer_nll
        )
    )
    return key


__all__ = [
    "RANKING_V1_TRANSFER",
    "RANKING_V2_RETENTION",
    "TRAINER",
    "SecurityJobs",
    "build_selection_reference_from_results",
    "check_selection_reference",
    "command_for",
    "dataset_version_accepted",
    "new_coverage_gain_ok",
    "normalize_family_caps",
    "ranking_version",
    "retention_rank_key",
    "verify_dataset_version",
]
