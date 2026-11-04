"""Independent development situations for Glimmer evaluation.

Synthetic and deliberately distinct from both the regression checks in
``operator_baseline`` and the DPO training pairs. Steps, metrics, and
wording are authored placeholders (not campaign measurements) so this file
is safe for source-only commit. Real campaign data lives outside Git.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    validate_episode,
)
from siliconshield.research_loop.episodes.operator_episodes import episode_for
from siliconshield.schemas import OPERATOR_DEV_SCHEMA

# Synthetic variations exercise the same tool interface with different observations.
DEV_SCENARIOS = [
    (
        "dev_regression_211_311",
        {
            "phase": "training",
            "systemd": "running",
            "step": 311,
            "execution_failed": False,
            "development_history": {
                "entries": [
                    {
                        "step": 211,
                        "time": 1100.0,
                        "event": "development",
                        "metrics": {"accuracy": 0.813, "nll": 0.412},
                    },
                    {
                        "step": 311,
                        "time": 2100.0,
                        "event": "development",
                        "metrics": {"accuracy": 0.799, "nll": 0.521},
                    },
                ],
                "complete": True,
                "total": 2,
            },
        },
        {"gpus_free": 0, "reservation": "self"},
        "stop",
        "Development accuracy fell from 81.3% at step 211 to 79.9% at step 311, "
        "while NLL increased from 0.412 to 0.521. Stop under the configured regression "
        "rule and preserve the selected checkpoint. No execution failure was observed.",
        True,
    ),
    (
        "dev_resume_63",
        {
            "phase": "paused",
            "systemd": "inactive",
            "checkpoint_valid": True,
            "step": 63,
            "execution_failed": False,
        },
        {"gpus_free": 3, "reservation": "none"},
        "resume",
        "A validated checkpoint exists at step 63; resume from the recorded "
        "checkpoint directory without repeating completed work.",
        True,
    ),
    (
        "dev_adjust_44",
        {
            "phase": "training",
            "systemd": "running",
            "execution_failed": False,
            "loss_trend": "diverging",
            "gradient_norm": 44.5,
            "step": 123,
            "lr_adjustments_used": 0,
            "lr_adjustments_allowed": 1,
        },
        {"gpus_free": 0, "reservation": "self"},
        "adjust_lr",
        "Loss is diverging with gradient norm 44.5 and one bounded LR adjustment remains; "
        "use it within pre-registered limits.",
        True,
    ),
    (
        "dev_healthy_57",
        {
            "phase": "training",
            "systemd": "running",
            "step": 57,
            "execution_failed": False,
            "dev_loss_trend": "falling",
        },
        {"gpus_free": 0, "reservation": "self"},
        "leave_unchanged",
        "Run is healthy and progressing; leaving it unchanged avoids an unnecessary intervention.",
        True,
    ),
    (
        "dev_finalize_9",
        {
            "phase": "training_complete",
            "systemd": "completed",
            "execution_failed": False,
            "report_present": False,
            "optimizer_updates_remaining": 9,
        },
        {"gpus_free": 3, "reservation": "none"},
        "finalize_retry",
        "Training completed but finalization is missing; retry finalization without additional "
        "optimizer updates.",
        True,
    ),
    (
        "dev_failed_3",
        {
            "phase": "training",
            "systemd": "failed",
            "exit_code": 3,
            "execution_failed": True,
            "error": "NVLink error in worker",
            "false_safe_rate": 0.31,
        },
        {"gpus_free": 3, "reservation": "none"},
        "stop",
        "The unit failed at execution level with exit code 3; stop and preserve evidence "
        "rather than treating it as a quality result.",
        True,
    ),
]


def build_dev(output):
    if not isinstance(output, (str, Path)):
        raise ValueError("Dev build needs an output path")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario, state, resources, action, reason, _ in DEV_SCENARIOS:
        rows.append(
            episode_for(
                scenario,
                state,
                resources,
                action,
                reason,
                "dev-independent-" + scenario,
                0,
            )
        )
    with (output / "dev.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    manifest = {
        "schema": OPERATOR_DEV_SCHEMA,
        "rows": len(rows),
        "held_out": False,
        "never_train": False,
        "independent": True,
        "actions": dict(Counter(r["target"]["action"] for r in rows)),
        "lineages": sorted({r["group_id"] for r in rows}),
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def load_dev(path):
    if not isinstance(path, (str, Path)):
        raise ValueError("Dev path must be a path")
    rows = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        validate_episode(row)
        if not str(row.get("group_id", "")).startswith("dev-independent-"):
            raise ValueError(
                "Dev rows must use independent lineages: " + str(row.get("id"))
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build", help="Directory to write independent dev rows"
    )
    args = parser.parse_args()
    if args.build:
        print(json.dumps(build_dev(args.build), indent=2))


if __name__ == "__main__":
    main()
