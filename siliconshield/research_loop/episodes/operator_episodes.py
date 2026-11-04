"""Authored qualification episodes for the bounded SiliconShield-RTL
operator (Phase 1).

Each episode shows only decision-point information: current run state,
resources, budget, and development evidence available then. Post-decision
outcomes live in row['outcome'], never in model-visible messages. Scenarios
are synthetic and reviewed; they teach bounded operation, not science.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from siliconshield.research_loop.episodes.contracts import (
    OPERATOR_ACTIONS,
    OPERATOR_SYSTEM,
    OPERATOR_TOOLS,
    atomic_json,
    call_message,
    digest,
    final_payload,
    finite_number,
    validate_episode,
)
from siliconshield.research_loop.episodes.dataset import write_manifest
from siliconshield.schemas import OPERATOR_EPISODES_SCHEMA

# (scenario, visible state, resource state, action, reason)
SCENARIOS = [
    (
        "healthy_running",
        {
            "phase": "training",
            "systemd": "running",
            "step": 40,
            "dev_loss_trend": "falling",
        },
        {"gpus_free": 0, "reservation": "self"},
        "leave_unchanged",
        "Run is healthy and progressing; leaving it unchanged avoids an unnecessary intervention.",
    ),
    (
        "healthy_progressing",
        {
            "phase": "training",
            "systemd": "running",
            "step": 80,
            "dev_accuracy_delta_pp": 2.5,
        },
        {"gpus_free": 0, "reservation": "self"},
        "continue",
        "Development evidence is improving under the current configuration; continue within budget.",
    ),
    (
        "execution_failure",
        {
            "phase": "training",
            "systemd": "failed",
            "exit_code": 1,
            "error": "CUDA error in worker",
        },
        {"gpus_free": 3, "reservation": "none"},
        "leave_unchanged",
        "The unit has already failed; preserve evidence without issuing a stop to an inactive process.",
    ),
    (
        "stalled_learning",
        {
            "phase": "training",
            "systemd": "running",
            "step": 120,
            "stale_evaluations": 4,
            "dev_loss_trend": "flat",
        },
        {"gpus_free": 0, "reservation": "self"},
        "leave_unchanged",
        "No measured development progress across repeated evaluations; leave the run unchanged while the "
        "stall is investigated outside the bounded intervention set.",
    ),
    (
        "overfitting",
        {
            "phase": "training",
            "systemd": "running",
            "train_accuracy": 0.97,
            "dev_accuracy": 0.61,
            "dev_accuracy_delta_pp": -3.0,
        },
        {"gpus_free": 0, "reservation": "self"},
        "stop",
        "Training accuracy rises while development accuracy falls; stop the unhealthy run rather than "
        "continue blindly.",
    ),
    (
        "class_collapse",
        {
            "phase": "training",
            "systemd": "running",
            "min_class_recall": 0.0,
            "degeneracy_count": 1,
        },
        {"gpus_free": 0, "reservation": "self"},
        "stop",
        "A class has no recall; stop the degenerate run instead of spending budget on broken labels.",
    ),
    (
        "data_defect",
        {
            "phase": "preflight",
            "systemd": "missing",
            "data_hash_match": False,
            "empty_splits": ["dev"],
        },
        {"gpus_free": 3, "reservation": "none"},
        "leave_unchanged",
        "Input hashes do not match and a split is empty; do not submit until the data defect is resolved.",
    ),
    (
        "invalid_resume",
        {
            "phase": "paused",
            "systemd": "inactive",
            "checkpoint_valid": False,
            "checkpoint_detail": "state.pt unreadable",
        },
        {"gpus_free": 3, "reservation": "none"},
        "leave_unchanged",
        "Resume state is invalid; stay paused instead of resuming or silently restarting training from scratch.",
    ),
    (
        "valid_resume",
        {
            "phase": "paused",
            "systemd": "inactive",
            "checkpoint_valid": True,
            "step": 60,
        },
        {"gpus_free": 3, "reservation": "none"},
        "resume",
        "A validated checkpoint exists at the recorded position; resume without repeating completed work.",
    ),
    (
        "test_touched_resume",
        {
            "phase": "paused",
            "systemd": "inactive",
            "checkpoint_valid": True,
            "test_touched": True,
        },
        {"gpus_free": 3, "reservation": "none"},
        "leave_unchanged",
        "The checkpoint lineage is test-touched; stay paused because it may not resume into further training.",
    ),
    (
        "duplicate_running",
        {"phase": "training", "systemd": "running", "submit_requested": True},
        {"gpus_free": 0, "reservation": "self"},
        "leave_unchanged",
        "The run already owns the GPUs and is running; a second submission would duplicate the job.",
    ),
    (
        "finalize_retry",
        {
            "phase": "training_complete",
            "systemd": "completed",
            "report_present": False,
            "optimizer_updates_remaining": 0,
        },
        {"gpus_free": 3, "reservation": "none"},
        "finalize_retry",
        "Training completed but finalization is missing; retry finalization without additional optimizer updates.",
    ),
    (
        "lr_instability",
        {
            "phase": "training",
            "systemd": "running",
            "loss_trend": "diverging",
            "gradient_norm": 42.0,
            "lr_adjustments_used": 0,
            "lr_adjustments_allowed": 1,
        },
        {"gpus_free": 0, "reservation": "self"},
        "adjust_lr",
        "Loss is diverging with large gradients and one bounded LR adjustment remains; use it within pre-registered limits.",
    ),
    (
        "missing_metrics",
        {"phase": "training", "systemd": "running", "metrics_present": False},
        {"gpus_free": 0, "reservation": "self"},
        "leave_unchanged",
        "No development metrics are available at this point; leave the run unchanged because deciding "
        "quality either way would be unsupported.",
    ),
    (
        "false_safe_risk",
        {
            "phase": "training",
            "systemd": "running",
            "worst_family_false_safe_rate": 0.42,
            "false_safe_budget": 0.25,
        },
        {"gpus_free": 0, "reservation": "self"},
        "stop",
        "Worst-family false-safe rate 0.42 exceeds the 0.25 budget; stop rather than accept unsafe evidence.",
    ),
    (
        "resource_contention",
        {"phase": "prepare", "systemd": "missing", "submit_requested": True},
        {"gpus_free": 0, "reservation": "other_run"},
        "leave_unchanged",
        "Another run owns the three GPUs; do not submit or disturb the owned process.",
    ),
]

FORBIDDEN_VISIBLE = (
    "outcome",
    "expected_action",
    "final_test",
    "post_decision",
    "future_",
)


def operator_coverage(train, development, replay):
    """Measure operator signal separately from checkpoint triage and template copies."""
    for name, rows in (
        ("train", train),
        ("development", development),
        ("replay", replay),
    ):
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) for row in rows
        ):
            raise ValueError(
                "Operator coverage needs a list of episode rows for " + name
            )
    actions = OPERATOR_TOOLS[-1]["function"]["parameters"]["properties"][
        "action"
    ]["enum"]
    report = {}
    for name, rows in (
        ("train", train),
        ("development", development),
        ("replay", replay),
    ):
        operators = [
            r
            for r in rows
            if any(
                t["function"]["name"] == "operator.decide" for t in r["tools"]
            )
        ]
        counts = Counter(r["target"]["action"] for r in operators)
        groups = {
            a: len(
                {r["group_id"] for r in operators if r["target"]["action"] == a}
            )
            for a in actions
        }
        situations = {
            digest(
                {
                    "state": {
                        k: v for k, v in r["summary"].items() if k != "run_id"
                    },
                    "resources": r["resources"],
                }
            )
            for r in operators
        }
        report[name] = {
            "rows": len(rows),
            "operator_rows": len(operators),
            "distinct_operator_situations": len(situations),
            "authored_fixture_rows": sum(
                r.get("verification", {}).get("kind") == "authored_fixture"
                for r in operators
            ),
            "actions": {a: counts[a] for a in actions},
            "action_lineages": groups,
            "missing_actions": [a for a in actions if not counts[a]],
        }
    # Readiness minima for this training plan, not a claim of statistical sufficiency.
    issues = []
    if report["train"]["distinct_operator_situations"] < 500:
        issues.append(
            "Operator training requires at least 500 distinct training situations"
        )
    if report["train"]["operator_rows"] * 2 < report["train"]["rows"]:
        issues.append(
            "Checkpoint triage outweighs the operator training signal"
        )
    for name, minimum in (("train", 5), ("development", 2)):
        for action, count in report[name]["action_lineages"].items():
            if count < minimum:
                issues.append(
                    "%s action %s needs at least %d lineages"
                    % (name, action, minimum)
                )
    if report["replay"]["missing_actions"]:
        issues.append(
            "Selected development replay omits: "
            + ", ".join(report["replay"]["missing_actions"])
        )
    report["issues"] = issues
    report["coverage_passed"] = not issues
    report["scope"] = (
        "Coverage checks only; independent episode review and operational validation remain required."
    )
    return report


ALLOWED_AT_EXECUTION = set(OPERATOR_ACTIONS)


def episode_for(scenario, state, resources, action, reason, lineage, index):
    for name, value in (
        ("scenario", scenario),
        ("action", action),
        ("reason", reason),
        ("lineage", lineage),
    ):
        if not isinstance(value, str):
            raise ValueError("Episode builder needs text " + name)
    if not isinstance(state, dict) or not isinstance(resources, dict):
        raise ValueError("Episode builder needs state and resource objects")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("Episode builder needs an integer index")
    run_id = "run-%s-%02d" % (
        hashlib.sha256(lineage.encode()).hexdigest()[:8],
        index,
    )
    visible_state = {"run_id": run_id, **state}
    decide_args = {"run_id": run_id, "action": action, "reason": reason}
    if action == "adjust_lr":
        visible_state.update(
            current_lr=4e-5, permitted_lr=2e-5, lr_bounds=[1e-6, 4e-5]
        )
        decide_args["lr"] = visible_state["permitted_lr"]
    if action == "resume":
        visible_state["checkpoint_dir"] = "/private/runs/" + run_id
        decide_args["checkpoint_dir"] = visible_state["checkpoint_dir"]
    messages = [
        {"role": "system", "content": OPERATOR_SYSTEM},
        {
            "role": "user",
            "content": "Inspect run %s and record the appropriate bounded operator action."
            % run_id,
        },
        call_message("operator.inspect_run", {"run_id": run_id}, "inspect"),
        {
            "role": "tool",
            "tool_call_id": "inspect",
            "content": json.dumps(visible_state, sort_keys=True),
        },
        call_message(
            "operator.inspect_resources", {"run_id": run_id}, "resources"
        ),
        {
            "role": "tool",
            "tool_call_id": "resources",
            "content": json.dumps(resources, sort_keys=True),
        },
        call_message("operator.decide", decide_args, "decide"),
        {
            "role": "tool",
            "tool_call_id": "decide",
            "content": '{"recorded": true, "executed": false}',
        },
        {
            "role": "assistant",
            "content": json.dumps(
                final_payload(action, run_id=run_id), sort_keys=True
            ),
        },
    ]
    row = {
        "id": digest(
            {"scenario": scenario, "lineage": lineage, "index": index}
        )[:20],
        "group_id": lineage,
        "tools": OPERATOR_TOOLS,
        "messages": messages,
        "summary": visible_state,
        "resources": resources,
        "target": {
            "action": action,
            "reason": reason,
            **{
                k: decide_args[k]
                for k in ("lr", "checkpoint_dir")
                if k in decide_args
            },
        },
        "verification": {"kind": "authored_fixture", "scenario": scenario},
        # Post-decision information: stored with the episode, never model-visible.
        "outcome": {
            "recorded_only": True,
            "executed_by_episode": False,
            "note": "Outcome assessment happens after the decision, outside the model input.",
        },
    }
    validate_episode(row)
    return row


MIGRATION_DEFAULT_LR = 2e-5


def migrate_operator_row(row):
    """Migrate one saved operator episode to the current contract.

    Rows whose action is no longer honored by execution are rejected, not
    remapped: silently changing the action would fabricate supervision.
    Missing decision fields are filled only for authored fixtures, using the
    same synthetic rules as the current builder, and recorded as migration
    provenance. Returns (migrated_row_or_None, note).
    """
    import copy

    if not isinstance(row, dict):
        raise ValueError("Migration needs an episode object")
    messages = row.get("messages", [])
    if not isinstance(messages, list) or any(
        not isinstance(message, dict) for message in messages
    ):
        raise ValueError("Migration needs a list of message objects")
    for message in messages:
        calls_raw = message.get("tool_calls", [])
        if not isinstance(calls_raw, list) or any(
            not isinstance(call, dict) for call in calls_raw
        ):
            raise ValueError("Migration needs a list of tool calls")
    calls = [
        c
        for m in messages
        for c in m.get("tool_calls", [])
        if isinstance(c.get("function"), dict)
        and c["function"].get("name") == "operator.decide"
    ]
    if not calls:
        return None, {"status": "rejected", "note": "not an operator episode"}
    arguments = calls[0]["function"].get("arguments")
    if not isinstance(arguments, dict):
        return None, {
            "status": "rejected",
            "note": "operator decision has no argument object",
        }
    action = arguments.get("action")
    if action not in OPERATOR_ACTIONS:
        return None, {
            "status": "rejected",
            "note": (
                "obsolete action %r is not honored by execution; excluded, not remapped"
                % (action,)
            ),
        }
    verification = row.get("verification", {})
    if not isinstance(verification, dict):
        return None, {
            "status": "rejected",
            "note": "operator episode has no verification object",
        }
    if verification.get("kind") != "authored_fixture":
        return None, {
            "status": "rejected",
            "note": "only authored fixtures carry synthetic migration defaults",
        }
    row = copy.deepcopy(row)
    args = next(
        c
        for m in row["messages"]
        for c in m.get("tool_calls", [])
        if isinstance(c.get("function"), dict)
        and c["function"].get("name") == "operator.decide"
    )["function"]["arguments"]
    if not isinstance(args, dict):
        return None, {
            "status": "rejected",
            "note": "operator decision has no argument object",
        }
    filled = {}
    summary = row.get("summary")
    if action == "resume" and not isinstance(args.get("checkpoint_dir"), str):
        if not isinstance(summary, dict) or not isinstance(
            summary.get("run_id"), str
        ):
            return None, {
                "status": "rejected",
                "note": "resume migration lacks an owned run id",
            }
        args["checkpoint_dir"] = filled["checkpoint_dir"] = (
            "/private/runs/" + summary["run_id"]
        )
    if action == "adjust_lr" and not finite_number(args.get("lr")):
        args["lr"] = filled["lr"] = MIGRATION_DEFAULT_LR
    row["tools"] = OPERATOR_TOOLS
    target = row.get("target", {})
    if not isinstance(target, dict):
        return None, {
            "status": "rejected",
            "note": "operator episode has no target object",
        }
    target.update({k: v for k, v in filled.items()})
    row["target"] = target
    provenance = dict(row.get("verification", {}))
    provenance["migration"] = {
        "filled": filled,
        "contract": "operator-decide-v3",
    }
    row["verification"] = provenance
    try:
        validate_episode(row)
    except ValueError as exc:
        return None, {
            "status": "rejected",
            "note": "migration failed validation: %s" % exc,
        }
    if filled:
        return row, {"status": "migrated", "note": "filled %s" % sorted(filled)}
    return row, {"status": "current", "note": "already current"}


def migrate_operator_dataset(source_dir, dest_dir):
    """Migrate train/dev operator episodes; the sealed test split is copied byte-identical.

    Never rewrite sealed test content: test rows are verified present and
    copied unchanged, or migration refuses to run.
    """
    from siliconshield.research_loop.episodes.dataset import read_json

    source, dest = Path(source_dir), Path(dest_dir)
    manifest = read_json(source / "manifest.json")
    if (
        "test" not in manifest.get("groups", {}).values()
        and not (source / "test.jsonl").exists()
    ):
        raise ValueError(
            "Refusing migration without the sealed test split present"
        )
    dest.mkdir(parents=True, exist_ok=True)
    report = {
        "migrated": 0,
        "current": 0,
        "rejected": [],
        "test_rows_copied": 0,
    }
    for split in ("train", "dev"):
        rows = [
            json.loads(line)
            for line in (source / (split + ".jsonl")).read_text().splitlines()
            if line.strip()
        ]
        with (dest / (split + ".jsonl")).open("x") as stream:
            for row in rows:
                decides = any(
                    c.get("function", {}).get("name") == "operator.decide"
                    for m in row.get("messages", [])
                    for c in m.get("tool_calls", [])
                )
                if not decides:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    continue
                migrated, outcome = migrate_operator_row(row)
                if migrated is None:
                    report["rejected"].append({"id": row.get("id"), **outcome})
                    continue
                stream.write(json.dumps(migrated, allow_nan=False) + "\n")
                report[outcome["status"]] += 1
        manifest["files"][split] = hashlib.sha256(
            (dest / (split + ".jsonl")).read_bytes()
        ).hexdigest()
    test_rows = (source / "test.jsonl").read_bytes()
    (dest / "test.jsonl").write_bytes(test_rows)
    manifest["files"]["test"] = hashlib.sha256(test_rows).hexdigest()
    report["test_rows_copied"] = len(test_rows.splitlines())
    manifest["migration"] = {
        "tool": "migrate_operator_dataset",
        "rejected": report["rejected"],
    }
    write_manifest(dest, manifest)
    atomic_json(dest / "migration-report.json", report)
    return report


def build(output, lineages_per_scenario=1):
    if lineages_per_scenario != 1:
        raise ValueError(
            "Repeated copies are not independent situations; use one episode per scenario"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario, state, resources, action, reason in SCENARIOS:
        if action not in ALLOWED_AT_EXECUTION:
            raise ValueError(
                "Scenario action %r is not honored by the execution allowlist"
                % action
            )
        for index in range(lineages_per_scenario):
            lineage = digest({"scenario": scenario})[:16]
            rows.append(
                episode_for(
                    scenario, state, resources, action, reason, lineage, index
                )
            )
    groups = sorted(
        {r["group_id"] for r in rows},
        key=lambda g: digest({"split_seed": 43, "group": g}),
    )
    n_eval = max(1, len(groups) // 10)
    partition = {
        g: ("test" if i < n_eval else "dev" if i < 2 * n_eval else "train")
        for i, g in enumerate(groups)
    }
    counts, actions = {}, {}
    for split in ("train", "dev", "test"):
        subset = [r for r in rows if partition[r["group_id"]] == split]
        with (output / (split + ".jsonl")).open("x") as stream:
            for row in subset:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        counts[split] = len(subset)
        actions[split] = dict(Counter(r["target"]["action"] for r in subset))
    manifest = {
        "schema": OPERATOR_EPISODES_SCHEMA,
        "rows": counts,
        "groups": {g: partition[g] for g in sorted(partition)},
        "actions": actions,
        "scenarios": [s[0] for s in SCENARIOS],
        "limitations": [
            "Authored qualification scenarios, not reviewed live-run episodes.",
            "Actions follow the authored bounded-operator policy.",
        ],
    }
    write_manifest(output, manifest)
    return manifest


def validate_operator_dataset(rows):
    """Acceptance checks: contract, leakage, split isolation handled by builder, balance."""
    issues = [] if rows else ["No operator episodes"]
    actions = Counter()
    situations = set()
    for row in rows:
        try:
            validate_episode(row)
        except ValueError as exc:
            issues.append("invalid episode %s: %s" % (row.get("id"), exc))
            continue
        state = {k: v for k, v in row["summary"].items() if k != "run_id"}
        situation = digest({"state": state, "resources": row["resources"]})
        if situation in situations:
            issues.append("Repeated underlying situation: " + situation)
        situations.add(situation)
        actions[row["target"]["action"]] += 1
        visible = json.dumps(row["messages"]).lower()
        for token in FORBIDDEN_VISIBLE:
            if token in visible:
                issues.append(
                    "future/private field %r visible in episode %s"
                    % (token, row["id"])
                )
        import tempfile

        from siliconshield.research_loop.controller.replay import execute, grade

        class TraceClient:
            model = "qualification-fixture"

            def complete(self, messages, tools=()):
                return row["messages"][len(messages)]

        with tempfile.TemporaryDirectory() as attempt:
            if not grade(row, execute(row, TraceClient(), attempt))["verified"]:
                issues.append(
                    "Episode fails actual operator replay: " + row["id"]
                )
    noop = actions.get("leave_unchanged", 0)
    if rows and noop / len(rows) < 0.1:
        issues.append(
            "healthy no-op coverage below 10%%: %d/%d" % (noop, len(rows))
        )
    return {
        "rows": len(rows),
        "distinct_situations": len(situations),
        "actions": dict(actions),
        "issues": issues,
        "accepted": not issues,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lineages-per-scenario", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.lineages_per_scenario), indent=2))


if __name__ == "__main__":
    main()
