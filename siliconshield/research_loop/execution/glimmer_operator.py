"""Unattended bounded operator loop: Glimmer proposes, deterministic code disposes.

Glimmer (served on Spark with the operator-v3 adapter) sees only decision-point
state through OPERATOR_TOOLS. Every recorded action is validated
deterministically before anything executes: allowlist, budget, checkpoint
validity, test-touch status, and duplicate prevention are enforced by the
Operator state machine, not trusted to the model. The loop is bounded by
configuration (max cycles, preauthorized experiment, intervention budget);
it never starts a follow-on campaign.
"""

import argparse
import fcntl
import json
import math
import time
import uuid
from pathlib import Path

from siliconshield.research_loop.controller.client import ChatClient, run_agent
from siliconshield.research_loop.episodes.contracts import (
    OPERATOR_SYSTEM,
    OPERATOR_TOOLS,
    atomic_json,
    final_payload,
)
from siliconshield.research_loop.execution.operator import (
    Operator,
    systemd_state,
)

VIEW_METRICS = (
    "n",
    "accuracy",
    "nll",
    "false_safe_rate",
    "worst_family_false_safe_rate",
    "both_pair_members_correct",
    "min_class_recall",
)


def _metric_subset(metrics, fields=VIEW_METRICS):
    if not isinstance(metrics, dict):
        return {}
    return {
        key: metrics[key]
        for key in fields
        if isinstance(metrics.get(key), (int, float))
        and not isinstance(metrics.get(key), bool)
    }


def model_status(status, manifest=None, schedule=None):
    """Expose decision evidence without predictions, bootstrap arrays or source paths.

    Pooled development metrics are reported alongside per-view (transfer /
    retention) metrics, the acceptance verdict with its failed criteria, and
    the best / selected / latest checkpoint steps, so a controller can tell
    a transfer gain with a failed gate apart from an all-green run.
    """
    result = {}
    for name in ("job.json", "execution.json", "status.json"):
        record = status.get(name)
        if isinstance(record, dict):
            result[name] = {
                key: record[key]
                for key in (
                    "phase",
                    "event",
                    "step",
                    "epoch",
                    "exit_code",
                    "failure",
                    "started",
                    "finished",
                    "loss",
                    "encoder_lr",
                    "peak_gib",
                    "gradient_norm_before_clip",
                )
                if key in record
            }
            metrics = record.get("metrics", {})
            result[name]["metrics"] = {
                key: metrics[key]
                for key in (
                    "n",
                    "accuracy",
                    "nll",
                    "false_safe_rate",
                    "min_class_recall",
                    "both_pair_members_correct",
                )
                if key in metrics
            }
    report = status.get("report.json")
    if isinstance(report, dict):
        result["report"] = {
            "evaluation_role": report.get("evaluation_role", "development"),
            "selected_step": report.get("selected_checkpoint", {}).get("step"),
            "test_evaluated": report.get("test_evaluated"),
        }
        for field in (
            "train",
            "initial_train",
            "development",
            "initial_development",
        ):
            metrics = report.get(field)
            if isinstance(metrics, dict):
                result["report"][field] = {
                    key: metrics[key]
                    for key in (
                        "n",
                        "accuracy",
                        "nll",
                        "false_safe_rate",
                        "min_class_recall",
                        "both_pair_members_correct",
                    )
                    if key in metrics
                }
        acceptance = report.get("development_acceptance")
        if isinstance(acceptance, dict):
            result["report"]["acceptance"] = {
                "accepted": bool(acceptance.get("accepted")),
                "reasons": list(acceptance.get("reasons", [])),
            }
        progress = report.get("progress", {})
        if isinstance(progress, dict):
            result["report"]["selection"] = {
                key: progress.get(key)
                for key in (
                    "best_epoch",
                    "best_generation",
                    "last_eval_step",
                    "stale",
                    "reduced",
                    "lr_scale",
                    "stop_reason",
                )
                if key in progress
            }
            result["report"]["selection"]["selected_step"] = (
                report.get("selected_checkpoint") or {}
            ).get("step")
        if manifest is not None:
            views, retention_passed = _latest_view_metrics(manifest)
            if views:
                result["report"]["views"] = views
            if retention_passed is not None:
                result["report"]["retention_passed"] = retention_passed
        if schedule is not None:
            result["schedule"] = schedule
    return result


def _latest_view_metrics(manifest):
    """Per-view transfer/retention metrics from the newest development evaluation.

    Returns (views_dict_or_None, retention_passed_or_None).
    """
    output = (manifest or {}).get("output")
    if not output:
        return None, None
    events = Path(output) / "events.jsonl"
    if not events.exists():
        return None, None
    views, retention_passed = None, None
    for line in events.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("event") not in ("initial_development", "development"):
            continue
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        dev_views = metrics.get("development_views")
        if isinstance(dev_views, dict) and dev_views:
            candidate = {
                name: _metric_subset(view)
                for name, view in dev_views.items()
                if isinstance(view, dict)
            }
            if any(candidate.values()):
                views = candidate
        retention = metrics.get("retention_check")
        if isinstance(retention, dict) and "passed" in retention:
            retention_passed = bool(retention["passed"])
    return views, retention_passed


HISTORY_METRICS = (
    "n",
    "accuracy",
    "nll",
    "false_safe_rate",
    "min_class_recall",
    "both_pair_members_correct",
)
HISTORY_LIMIT = 12


def development_history(manifest):
    """Decision-point evidence: metric names, values, steps, observation timestamps.

    The stop flag alone does not let the controller explain its decision; the
    measurements behind the flag must be visible. Histories longer than the
    limit keep the most recent entries and report truncation explicitly.
    """
    output = manifest.get("output")
    if not output:
        return {"entries": [], "complete": True, "total": 0}
    events = Path(output) / "events.jsonl"
    if not events.exists():
        return {"entries": [], "complete": True, "total": 0}
    entries = []
    for line in events.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("event") not in ("initial_development", "development"):
            continue
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        entry = {
            "step": record.get("step"),
            "time": record.get("time"),
            "event": record.get("event"),
            "epoch": record.get("epoch"),
            "metrics": {
                key: metrics[key] for key in HISTORY_METRICS if key in metrics
            },
        }
        dev_views = metrics.get("development_views")
        if isinstance(dev_views, dict) and dev_views:
            entry["views"] = {
                name: _metric_subset(view)
                for name, view in dev_views.items()
                if isinstance(view, dict)
            }
        retention = metrics.get("retention_check")
        if isinstance(retention, dict) and "passed" in retention:
            entry["retention_check"] = {"passed": bool(retention["passed"])}
        progress = record.get("progress", {})
        if isinstance(progress, dict):
            entry["training"] = {
                key: progress.get(key)
                for key in (
                    "lr_scale",
                    "stale",
                    "reduced",
                    "stop_reason",
                    "best_epoch",
                    "best_generation",
                )
                if key in progress
            }
        entries.append(entry)
    total = len(entries)
    kept = entries[-HISTORY_LIMIT:]
    return {"entries": kept, "complete": total <= HISTORY_LIMIT, "total": total}


def _regressed(before, after):
    """Accuracy down and NLL up on finite, present metrics."""
    try:
        return (
            math.isfinite(after["accuracy"])
            and math.isfinite(before["accuracy"])
            and math.isfinite(after["nll"])
            and math.isfinite(before["nll"])
            and after["accuracy"] < before["accuracy"]
            and after["nll"] > before["nll"]
        )
    except (KeyError, TypeError):
        return False


def schedule_status(manifest, status):
    """Scheduled LR changes, plateau actions, and operator interventions, separately.

    The controller must be able to distinguish a cosine-schedule LR movement
    from a plateau intervention (lr_scale change) and from an operator action.
    """
    manifest = manifest or {}
    status = status or {}
    report = status.get("report.json", {})
    signature = report.get("signature", {}) if isinstance(report, dict) else {}
    schedule = {
        key: signature.get(key)
        for key in (
            "lr_schedule",
            "warmup",
            "max_updates",
            "min_updates",
            "patience",
            "epochs",
        )
        if key in signature
    }
    entries = development_history(manifest)["entries"]
    if entries:
        training = entries[-1].get("training", {})
        schedule["lr_scale"] = training.get("lr_scale")
        schedule["reduced"] = training.get("reduced")
        schedule["stale"] = training.get("stale")
    reductions = []
    output = manifest.get("output")
    if output:
        events = Path(output) / "events.jsonl"
        if events.exists():
            for line in events.read_text().splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("event") == "learning_rate_reduced":
                    reductions.append(
                        {
                            key: record.get(key)
                            for key in ("step", "epoch", "time")
                            if record.get(key) is not None
                        }
                    )
    schedule["plateau_lr_reductions"] = reductions
    schedule["operator_interventions"] = [
        {
            key: item.get(key)
            for key in ("action", "time", "params")
            if key in item
        }
        for item in manifest.get("interventions", [])
        if isinstance(item, dict)
    ]
    schedule["intervention_budget"] = manifest.get("intervention_budget", {})
    return schedule


def stop_evidence(manifest, status):
    """A model-proposed stop needs an observed failure or development regression.

    Regression is checked on the pooled metrics and on each transfer/retention
    view, so a view-specific regression cannot hide behind a pooled average.
    """
    execution, latest = (
        status.get("execution.json", {}),
        status.get("status.json", {}),
    )
    if execution.get("phase") == "failed" or latest.get("error"):
        return True
    for name in ("loss", "gradient_norm_before_clip"):
        value = latest.get(name)
        if isinstance(value, (int, float)) and not math.isfinite(value):
            return True
    entries = development_history(manifest)["entries"]
    if len(entries) < 2:
        return False
    previous, current = entries[-2:]
    series = [(previous.get("metrics", {}), current.get("metrics", {}))]
    for view in ("transfer", "retention"):
        before, after = (
            previous.get("views", {}).get(view, {}),
            current.get("views", {}).get(view, {}),
        )
        if before and after:
            series.append((before, after))
    return any(_regressed(before, after) for before, after in series)


class OperatorTools:
    """Model-facing tools backed by the Operator; decide only records."""

    def __init__(self, operator, manifest):
        self.operator, self.manifest = operator, manifest
        self.recorded = None
        self.inspected = False

    def dispatch(self, name, args):
        run_id = self.manifest["experiment_id"]
        if args.get("run_id") != run_id:
            raise ValueError(
                "run_id must match the bound experiment: " + run_id
            )
        if name == "operator.inspect_run":
            self.inspected = True
            history = self.manifest.get("history", [])
            status = self.operator.jobs.status(run_id)
            live = (
                self.operator._owned_state(self.manifest)["state"]
                if isinstance(self.operator, Operator)
                else "unknown"
            )
            latest = status.get("status.json", {})
            return {
                "run_id": run_id,
                "phase": "training"
                if live == "running"
                else self.manifest.get("phase"),
                "operator_phase": self.manifest.get("phase"),
                "systemd": live,
                "step": latest.get(
                    "step", latest.get("progress", {}).get("step", 0)
                ),
                "execution_failed": status.get("execution.json", {}).get(
                    "phase"
                )
                == "failed",
                "development_history": development_history(self.manifest),
                "stop_evidence_present": stop_evidence(self.manifest, status)
                if self.manifest.get("output")
                else False,
                "evidence": model_status(status, self.manifest),
                "evaluation_role": "train_fitting"
                if self.manifest.get("overrides", {}).get("limit_train_rows")
                else "development",
                "last_phases": [h["phase"] for h in history[-4:]],
                "interventions_used": len(
                    self.manifest.get("interventions", [])
                ),
                "schedule": schedule_status(self.manifest, status),
                "checkpoint_dir": self.manifest.get("output"),
                "contract_version": 4,
                "test_touched": self.manifest.get("test_touched", False),
            }
        if name == "operator.inspect_resources":
            owner = self.operator.reservation.owner()
            if hasattr(self.operator.reservation, "path"):
                owner_path = (
                    self.operator.reservation.path.parent / "owner.json"
                )
                if owner_path.exists():
                    owner = json.loads(owner_path.read_text()).get("identity")
            return {
                "run_id": run_id,
                "gpus": self.manifest.get("gpus", [0, 1, 2]),
                "reservation_owner": owner,
                "owned_by_this_run": bool(
                    owner and owner == self.manifest.get("idempotency_key")
                ),
                "budget": self.manifest.get("intervention_budget", {}),
            }
        if name == "operator.decide":
            if not self.inspected:
                raise ValueError("Inspect the run before deciding")
            if self.recorded is not None:
                raise ValueError(
                    "Only one decision is permitted per observation"
                )
            from siliconshield.research_loop.execution.operator import (
                BOUNDED_INTERVENTIONS,
            )

            if args.get("action") not in BOUNDED_INTERVENTIONS:
                raise ValueError(
                    "Choose an action from the current tool contract"
                )
            self.recorded = dict(args)
            return {"recorded": True, "executed": False}
        raise ValueError("Unknown tool")


def run_cycle(operator, manifest, client, output, max_turns=12):
    """One observe->propose->validate->execute cycle. Returns the executed action or None."""
    manifest = operator.observe(manifest)
    manifest = operator.diagnose(manifest)
    tools = OperatorTools(operator, manifest)
    run_id = manifest["experiment_id"]
    messages = [
        {"role": "system", "content": OPERATOR_SYSTEM},
        {
            "role": "user",
            "content": "Inspect run %s and record the appropriate bounded action."
            % run_id,
        },
    ]
    trace = Path(output) / ("trace-" + uuid.uuid4().hex + ".jsonl")
    history = run_agent(
        client,
        messages,
        OPERATOR_TOOLS,
        tools.dispatch,
        trace,
        max_turns=max_turns,
    )
    decision = tools.recorded
    if decision is None:
        return manifest, None
    action = decision["action"]
    final = json.loads(history[-1].get("content", ""))
    if final != final_payload(action, run_id=run_id):
        raise ValueError(
            "Final response must match the recorded action and run"
        )
    params = {}
    if action == "adjust_lr":
        params["lr"] = decision.get("lr")
    if action == "resume":
        params["checkpoint_dir"] = decision.get("checkpoint_dir")
    # A job may finish while the model is generating; do not apply a stale action.
    if isinstance(operator, Operator) and operator._report_valid(
        manifest, operator.jobs.status(run_id)
    ):
        return manifest, {
            "action": "leave_unchanged",
            "decision": decision,
            "superseded": "job completed during inference",
        }
    if (
        action == "stop"
        and isinstance(operator, Operator)
        and not stop_evidence(manifest, operator.jobs.status(run_id))
    ):
        raise ValueError(
            "Stop rejected: no execution failure or development regression in the recorded evidence"
        )
    manifest = operator.intervene(manifest, action, **params)
    return manifest, {
        "action": action,
        "decision": decision,
        "final": history[-1].get("content", "") if history else "",
    }


def run_operator(config, client=None, operator=None, sleep=None):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".operator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_operator(config, client, operator, sleep)


def _run_operator(config, client=None, operator=None, sleep=None):
    """Bounded unattended loop over one preauthorized experiment."""
    profile = config["profile"]
    operator = operator or Operator(profile)
    client = client or ChatClient(
        config["endpoint"],
        config["model"],
        max_tokens=4096,
        adapter_path=config.get("adapter_path"),
        checkpoint_fingerprint=config.get("checkpoint_fingerprint"),
        ssh_host=config.get("ssh_host"),
    )
    manifest = operator.prepare(
        config["experiment_id"],
        config.get("overrides", {}),
        config.get("intervention_budget"),
    )
    if manifest["phase"] in ("prepare", "preflight"):
        manifest = operator.preflight(manifest)
        manifest = operator.submit(manifest)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    # The CLI never passes sleep, so the default must honor poll_seconds.
    sleep = sleep or time.sleep
    poll = config.get("poll_seconds", 60)
    status_path = output / "operator-status.json"
    cycles = (
        json.loads(status_path.read_text()).get("cycles", [])
        if status_path.exists()
        else []
    )

    def persist(status_manifest, job_running):
        atomic_json(
            output / "operator-status.json",
            {
                "experiment": config["experiment_id"],
                "phase": status_manifest["phase"],
                "job_running": job_running,
                "cycles": cycles,
                "model": config["model"],
                "checkpoint_fingerprint": config.get("checkpoint_fingerprint"),
            },
        )

    for _ in range(config.get("max_cycles", 8)):
        if hasattr(operator, "manifest_path"):
            path = operator.manifest_path(config["experiment_id"])
            if path.exists():
                manifest = json.loads(path.read_text())
        if (
            manifest["phase"] == "report"
            and manifest.get("terminal") == "complete"
        ):
            persist(manifest, False)
            return {"manifest": manifest, "cycles": cycles}
        if manifest["phase"] == "report":
            manifest = operator._save(
                manifest, "observe", note="reattached incomplete report"
            )
        job_running = (
            systemd_state(manifest["unit"], operator.runner)["state"]
            == "running"
        )
        if (
            not job_running
            and hasattr(operator, "_report_valid")
            and operator._report_valid(
                manifest, operator.jobs.status(config["experiment_id"])
            )
        ):
            manifest = operator.observe(manifest)
            manifest = operator.report(operator.finalize(manifest))
            persist(manifest, False)
            return {"manifest": manifest, "cycles": cycles}
        if not job_running and hasattr(operator, "abandon"):
            try:
                manifest = operator.abandon(manifest)
            except ValueError:
                pass
            else:
                persist(manifest, False)
                return {"manifest": manifest, "cycles": cycles}
        try:
            manifest, executed = run_cycle(operator, manifest, client, output)
        except Exception as exc:  # endpoint/tunnel failure: job stays owned by systemd; keep observing
            if hasattr(operator, "manifest_path"):
                manifest = json.loads(
                    operator.manifest_path(config["experiment_id"]).read_text()
                )
            job_running = (
                systemd_state(manifest["unit"], operator.runner)["state"]
                == "running"
            )
            cycles.append(
                {
                    "time": time.time(),
                    "phase": manifest["phase"],
                    "executed": None,
                    "job_running": job_running,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
            )
            persist(manifest, job_running)
            sleep(poll)
            continue
        job_running = (
            systemd_state(manifest["unit"], operator.runner)["state"]
            == "running"
        )
        cycles.append(
            {
                "time": time.time(),
                "phase": manifest["phase"],
                "executed": executed,
                "job_running": job_running,
            }
        )
        persist(manifest, job_running)
        if (
            manifest["phase"] == "report"
            and manifest.get("terminal") == "complete"
        ):
            break
        sleep(poll)
    # Exhausting an observation window does not terminate an owned training job.
    # The next invocation continues from its durable manifest and cycle history.
    return {"manifest": manifest, "cycles": cycles}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Local operator config JSON (not committed)",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    result = run_operator(config)
    print(
        json.dumps(
            {
                "experiment": config["experiment_id"],
                "phase": result["manifest"]["phase"],
                "cycles": len(result["cycles"]),
                "actions": [
                    c["executed"]["action"]
                    for c in result["cycles"]
                    if c["executed"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
