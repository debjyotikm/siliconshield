"""Convert SiliconShield-RTL development reports into review summaries.

The trainer's schema-2 reports use different metric names from the controller.
This adapter maps those names, preserves security metrics and checkpoint
references, and rejects reports that used test or final-evaluation data.
"""

import hashlib
from collections.abc import Mapping
from typing import Union, cast

from siliconshield.research_loop.episodes.contracts import (
    DevelopmentMetrics,
    DevelopmentSummary,
    JsonValue,
    finite_number,
)

ADAPTER_SCHEMA = "siliconshield-laya-development-result-v1"
TRAINER_SCHEMA = 2

_SECURITY_FIELDS = (
    "accuracy",
    "macro_f1",
    "nll",
    "false_safe_count",
    "false_safe_rate",
    "worst_family_false_safe_rate",
    "min_class_recall",
    "selection_cost",
    "both_pair_members_correct",
    "complete_pairs",
    "degeneracy_count",
    "class_support",
    "confusion",
)


def _fit(metrics: JsonValue, label: str) -> DevelopmentMetrics:
    """Pick the controller-visible metrics out of one LAYA development block.

    Maps the trainer's ``n``/``accuracy``/``nll`` names onto the shared
    ``decisions``/``agreement``/``nll`` contract. Raises ValueError when the
    block is missing or lacks finite values; downstream review revalidates
    denominators before any comparison.
    """
    if not isinstance(metrics, dict):
        raise ValueError(
            "LAYA report is missing " + label + " development metrics"
        )
    try:
        decisions = metrics["n"]
        agreement = metrics["accuracy"]
        nll = metrics["nll"]
    except KeyError as exc:
        raise ValueError("LAYA " + label + " metrics lack " + str(exc)) from exc
    if (
        not isinstance(decisions, int)
        or isinstance(decisions, bool)
        or not finite_number(agreement)
        or not finite_number(nll)
    ):
        raise ValueError("LAYA " + label + " metrics must be finite numbers")
    return {
        "decisions": decisions,
        "agreement": cast(float, agreement),
        "nll": cast(float, nll),
    }


def adapt_report(
    report: Mapping[str, JsonValue], *, source_sha256: Union[str, None] = None
) -> tuple[DevelopmentSummary, dict[str, JsonValue], dict[str, JsonValue]]:
    """Return (summary, provenance, group_seed) for one trainer report."""
    if report.get("schema") != TRAINER_SCHEMA:
        raise ValueError(
            "Unsupported LAYA trainer report schema: "
            + str(report.get("schema"))
        )
    if report.get("test_evaluated") or report.get("final_evaluation"):
        raise ValueError(
            "Report used final/test evaluation; only development evidence is admitted"
        )
    signature = report.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("Trainer signature must cover only train/dev data")
    data = signature.get("data")
    if (
        not isinstance(data, dict)
        or not data
        or not set(data) <= {"train", "dev"}
    ):
        raise ValueError("Trainer signature must cover only train/dev data")
    selected = _fit(report.get("development"), "selected")
    initial = _fit(report.get("initial_development"), "initial")
    if initial["decisions"] != selected["decisions"]:
        raise ValueError("Initial and selected development populations differ")
    progress_raw = report.get("progress")
    progress = progress_raw if isinstance(progress_raw, dict) else {}
    checkpoint = report.get("selected_checkpoint")
    checkpoint_step = (
        checkpoint.get("step") if isinstance(checkpoint, dict) else None
    )
    step = progress.get("step")
    if not step:
        step = checkpoint_step
    if not step:
        step = 0
    if isinstance(step, bool) or not isinstance(step, (int, float, str)):
        raise ValueError("LAYA report progress step must be a number")
    completed = int(step)
    run_token = hashlib.sha256(
        str(report.get("run_id", "")).encode()
    ).hexdigest()[:10]
    development = report.get("development")
    if not isinstance(development, dict):
        raise ValueError("LAYA report is missing selected development metrics")
    summary: DevelopmentSummary = {
        "run_id": "run-" + run_token + "-selected",
        "split_role": "open_development",
        "phase": "smoke" if completed <= 2 else "research",
        "completed_updates": completed,
        "initial": initial,
        "selected": selected,
        "reviewed_checkpoint": "selected",
        "security": {
            k: development[k] for k in _SECURITY_FIELDS if k in development
        },
        "development_acceptance": report.get("development_acceptance"),
        "selected_checkpoint": report.get("selected_checkpoint"),
        "selection_key": report.get("selection_key"),
        "stop_reason": report.get("stop_reason"),
    }
    provenance: dict[str, JsonValue] = {
        "kind": "checked_calculation",
        "source_sha256": source_sha256,
        "scope": "LAYA security-trainer development split (no final/test evaluation)",
        "adapter": ADAPTER_SCHEMA,
        "trainer_schema": TRAINER_SCHEMA,
        "trainer_run_id_sha256": hashlib.sha256(
            str(report.get("run_id", "")).encode()
        ).hexdigest(),
        "not_verified_research_optimality": True,
    }
    group_seed: dict[str, JsonValue] = {
        "adapter": ADAPTER_SCHEMA,
        "data": data,
        "initial_checkpoint": signature.get("initial_checkpoint"),
    }
    return summary, provenance, group_seed
