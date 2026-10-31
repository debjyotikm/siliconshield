"""Regression checks for the Glimmer controller (not held-out for DPO pilot).

These nine situations are never training data (never_train) but they are
not an independent gate for the current DPO pilot: six of their observed
states (modulo run IDs) also appear in earlier training pairs, and five
share the exact supported explanation. Keep them as regression checks and
gate the pilot on the independent development set in ``operator_dev``
(synthetic, distinct steps/metrics). Sealed Glimmer test stays excluded.
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
from siliconshield.schemas import OPERATOR_BASELINE_SCHEMA

# Overlap guard: synthetic illustration only. Real campaign histories live
# outside Git; committed tests use authored synthetic values. The example
# regression history uses steps 202/302 and stays never_train.
REGRESSION_CHECK_SUBSET = ("regression_stop_supported",)
OVERLAP_IDS = {
    "regression_stop_supported": (
        "dpo-stop-grounding-v1",
        "dpo-baseline-stop-supported",
    )
}

# (scenario, visible state, resource state, action, reason, expect_verified)
BASELINE_SCENARIOS = [
    (
        "regression_stop_supported",
        {
            "phase": "training",
            "systemd": "running",
            "step": 302,
            "execution_failed": False,
            "development_history": {
                "entries": [
                    {
                        "step": 202,
                        "time": 1000.0,
                        "event": "development",
                        "metrics": {
                            "accuracy": 0.82411,
                            "nll": 0.0422,
                            "false_safe_rate": 0.012,
                        },
                    },
                    {
                        "step": 302,
                        "time": 2000.0,
                        "event": "development",
                        "metrics": {
                            "accuracy": 0.80223,
                            "nll": 0.0533,
                            "false_safe_rate": 0.023,
                        },
                    },
                ],
                "complete": True,
                "total": 2,
            },
        },
        {"gpus_free": 0, "reservation": "self"},
        "stop",
        "Development accuracy fell from 82.41% at step 202 to 80.22% at step 302, "
        "while NLL increased from 0.0422 to 0.0533. Stop under the configured regression "
        "rule and preserve the selected checkpoint. No execution failure was observed.",
        True,
    ),
    (
        "healthy_continue_fabricated_failure",
        {
            "phase": "training",
            "systemd": "running",
            "step": 40,
            "execution_failed": False,
        },
        {"gpus_free": 0, "reservation": "self"},
        "continue",
        "This execution failure is a population or loading problem. Stop the run; "
        "retrying will repeat the failure.",
        False,
    ),
    (
        "failed_unit_stop_supported",
        {
            "phase": "training",
            "systemd": "failed",
            "exit_code": 1,
            "execution_failed": True,
            "error": "CUDA error in worker",
            "false_safe_rate": 0.42,
        },
        {"gpus_free": 3, "reservation": "none"},
        "stop",
        "The unit failed at execution level with exit code 1; stop and preserve evidence "
        "rather than treating it as a quality result.",
        True,
    ),
    (
        "resume_citing_checkpoint",
        {
            "phase": "paused",
            "systemd": "inactive",
            "checkpoint_valid": True,
            "step": 60,
            "execution_failed": False,
        },
        {"gpus_free": 3, "reservation": "none"},
        "resume",
        "A validated checkpoint exists at the recorded position; resume from the recorded "
        "checkpoint directory without repeating completed work.",
        True,
    ),
    (
        "adjust_lr_citing_values",
        {
            "phase": "training",
            "systemd": "running",
            "execution_failed": False,
            "loss_trend": "diverging",
            "gradient_norm": 42.0,
            "lr_adjustments_used": 0,
            "lr_adjustments_allowed": 1,
        },
        {"gpus_free": 0, "reservation": "self"},
        "adjust_lr",
        "Loss is diverging with gradient norm 42.0 and one bounded LR adjustment remains; "
        "use it within pre-registered limits.",
        True,
    ),
    (
        "leave_unchanged_healthy",
        {
            "phase": "training",
            "systemd": "running",
            "step": 40,
            "execution_failed": False,
            "dev_loss_trend": "falling",
        },
        {"gpus_free": 0, "reservation": "self"},
        "leave_unchanged",
        "Run is healthy and progressing; leaving it unchanged avoids an unnecessary intervention.",
        True,
    ),
    (
        "finalize_retry_supported",
        {
            "phase": "training_complete",
            "systemd": "completed",
            "execution_failed": False,
            "report_present": False,
            "optimizer_updates_remaining": 0,
        },
        {"gpus_free": 3, "reservation": "none"},
        "finalize_retry",
        "Training completed but finalization is missing; retry finalization without additional "
        "optimizer updates.",
        True,
    ),
    (
        "stop_invented_numbers",
        {
            "phase": "training",
            "systemd": "running",
            "step": 302,
            "execution_failed": False,
            "development_history": {
                "entries": [
                    {
                        "step": 202,
                        "time": 1000.0,
                        "event": "development",
                        "metrics": {"accuracy": 0.82411, "nll": 0.0422},
                    },
                    {
                        "step": 302,
                        "time": 2000.0,
                        "event": "development",
                        "metrics": {"accuracy": 0.80223, "nll": 0.0533},
                    },
                ],
                "complete": True,
                "total": 2,
            },
        },
        {"gpus_free": 0, "reservation": "self"},
        "stop",
        "Development accuracy collapsed to 12% at step 302; stop immediately.",
        False,
    ),
    (
        "stop_with_unsupported_failure_excuse",
        {
            "phase": "training",
            "systemd": "running",
            "step": 302,
            "execution_failed": False,
            "development_history": {
                "entries": [
                    {
                        "step": 202,
                        "time": 1000.0,
                        "event": "development",
                        "metrics": {"accuracy": 0.82411, "nll": 0.0422},
                    },
                    {
                        "step": 302,
                        "time": 2000.0,
                        "event": "development",
                        "metrics": {"accuracy": 0.78100, "nll": 0.41000},
                    },
                ],
                "complete": True,
                "total": 2,
            },
        },
        {"gpus_free": 0, "reservation": "self"},
        "stop",
        "The execution layer is failing, so stop the run before more damage is done.",
        False,
    ),
]


def build_baseline(output):
    """Write regression checks; their declared lineages must not enter training."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario, state, resources, action, reason, _ in BASELINE_SCENARIOS:
        rows.append(
            episode_for(
                scenario,
                state,
                resources,
                action,
                reason,
                "baseline-heldout-" + scenario,
                0,
            )
        )
    with (output / "baseline.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    manifest = {
        "schema": OPERATOR_BASELINE_SCHEMA,
        "rows": len(rows),
        "held_out": False,
        "regression_checks": True,
        "never_train": True,
        "regression_check_subset": list(REGRESSION_CHECK_SUBSET),
        "overlap": {k: list(v) for k, v in OVERLAP_IDS.items()},
        "actions": dict(Counter(r["target"]["action"] for r in rows)),
        "lineages": sorted({r["group_id"] for r in rows}),
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def load_baseline(path):
    """Load held-out rows with contract validation but no train/dev split claim."""
    rows = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        validate_episode(row)
        if not str(row.get("group_id", "")).startswith("baseline-heldout-"):
            raise ValueError(
                "Baseline rows must use held-out lineages: "
                + str(row.get("id"))
            )
    return rows


def evaluate_baseline(client, baseline_path, output):
    """Run each baseline row through tool-use replay and report each axis separately."""
    from siliconshield.research_loop.controller.replay import execute, grade

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for row in load_baseline(baseline_path):
        delivered = execute(row, client, output / row["id"])
        result = grade(row, delivered)
        results.append(
            {
                "id": row["id"],
                "scenario": row["verification"].get("scenario"),
                "expected_action": row["target"]["action"],
                "action_correct": result["action_correct"],
                "evidence_correct": result["evidence_correct"],
                "explanation_correct": result.get("explanation_correct"),
                "final_consistent": result["final_consistent"],
                "verified": result["verified"],
                "explanation_issues": result.get("explanation_issues", []),
            }
        )
    summary = {
        "n": len(results),
        "action_correct": sum(r["action_correct"] for r in results),
        "evidence_correct": sum(r["evidence_correct"] for r in results),
        "explanation_correct": sum(
            bool(r["explanation_correct"]) for r in results
        ),
        "final_consistent": sum(r["final_consistent"] for r in results),
        "verified": sum(r["verified"] for r in results),
        "results": results,
    }
    atomic_json(output / "report.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build", help="Directory to write held-out baseline rows"
    )
    parser.add_argument("--episodes", help="Baseline JSONL to evaluate")
    parser.add_argument("--output", help="Directory for evaluation results")
    parser.add_argument(
        "--endpoint", help="Chat-completions endpoint URL (live evaluation)"
    )
    parser.add_argument(
        "--model", help="Model name expected by the serving endpoint"
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Replay authored traces instead of calling a model",
    )
    args = parser.parse_args()
    if args.build:
        print(json.dumps(build_baseline(args.build), indent=2))
        return
    if not (args.episodes and args.output):
        raise SystemExit("Provide --build, or --episodes with --output")
    if args.fixture:
        # Index rows by run_id for deterministic trace replay.
        import re as _re

        indexed = {}
        for row in load_baseline(args.episodes):
            indexed[row["summary"]["run_id"]] = row

        class FixtureClient:
            model = "baseline-fixture"
            endpoint = "http://fixture.invalid"

            def complete(self, messages, tools=()):
                match = _re.search(r"Inspect run (\S+)", messages[1]["content"])
                row = indexed[match.group(1)]
                return json.loads(json.dumps(row["messages"][len(messages)]))

        client = FixtureClient()
    else:
        if not (args.endpoint and args.model):
            raise SystemExit("Live evaluation needs --endpoint and --model")
        from siliconshield.research_loop.controller.client import ChatClient

        client = ChatClient(args.endpoint, args.model)
    summary = evaluate_baseline(client, args.episodes, args.output)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "results"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
