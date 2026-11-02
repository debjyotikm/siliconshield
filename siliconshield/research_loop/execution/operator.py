"""Transactional operator contract for bounded SiliconShield-RTL runs (Phase 0).

One coherent state machine replaces parallel ad-hoc job handling:

    prepare -> preflight -> submit -> observe -> diagnose
        -> intervene/resume -> finalize -> report

Invariants enforced here: idempotent submission (no duplicate jobs),
exclusive three-GPU ownership via a locked reservation, systemd state
reconciliation (running/completed/failed/missing/stale/foreign), bounded
interventions only, resume only from validated non-test-touched checkpoints,
and finalization retry without new optimizer updates.
"""

import fcntl
import hashlib
import json
import math
import time
from pathlib import Path

from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    digest,
    read_json,
)
from siliconshield.research_loop.execution.security_jobs import (
    PATH_FLAGS,
    SecurityJobs,
    command_for,
    laya_root_for,
    preservation_teacher_files,
    verify_laya,
)
from siliconshield.schemas import RUN_MANIFEST_SCHEMA

PHASES = (
    "prepare",
    "preflight",
    "submit",
    "observe",
    "diagnose",
    "intervene",
    "finalize",
    "report",
)
TERMINAL = ("complete", "stopped", "rejected")
BOUNDED_INTERVENTIONS = (
    "continue",
    "leave_unchanged",
    "stop",
    "adjust_lr",
    "resume",
    "finalize_retry",
)


def execution_inputs(profile, overrides):
    """Immutable training inputs, independent of controller implementation."""
    source = Path(profile["source"])
    files = list((source / "siliconshield/training").rglob("*.py"))
    files += list((source / "requirements").glob("*.txt"))
    values = {**profile.get("defaults", {}), **overrides}
    splits = ("train",) if values.get("limit_train_rows") else ("train", "dev")
    files += [Path(profile["data"]) / (s + ".jsonl") for s in splits]
    files += list(Path(profile["model"]).rglob("*.json")) + list(
        Path(profile["model"]).rglob("*.safetensors")
    )
    files += preservation_teacher_files(profile)
    files += [Path(values[key]) for key in PATH_FLAGS if values.get(key)]
    root = laya_root_for(profile)
    if root:
        files += list((Path(root) / "laya").rglob("*.py"))
    for name in ("recipe", "reporting_contract"):
        if profile.get(name):
            files.append(Path(profile[name]))
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(files))
        if p.is_file()
    }


def idempotency_key(profile, experiment_id, overrides):
    command, _ = command_for(profile, experiment_id, overrides or {})
    from siliconshield.training.glimmer.identity import _sha

    source = Path(profile["source"])
    files = list((source / "siliconshield").rglob("*.py")) + list(
        (source / "requirements").glob("*.txt")
    )
    identity = {str(p): _sha(p) for p in sorted(files) if p.is_file()}
    identity.update(execution_inputs(profile, overrides or {}))
    return digest(
        {
            "experiment": experiment_id,
            "command": command,
            "laya_root": laya_root_for(profile),
            "content": identity,
            "budget": profile.get("timeout_seconds", 28800),
            "source": profile.get("source"),
            "data": profile.get("data"),
        }
    )


def systemd_state(unit, runner):
    """Reconcile a unit: running/completed/failed/missing, from live systemd state."""
    show = runner(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "-p",
            "LoadState,ActiveState,SubState,Result,ExecMainStatus,Description,InvocationID",
        ]
    )
    if show is None or show.get("LoadState") == "not-found":
        return {"state": "missing", "detail": show}
    active, result = show.get("ActiveState"), show.get("Result")
    if active in ("active", "activating"):
        return {"state": "running", "detail": show}
    if active == "failed" or result == "failure":
        return {"state": "failed", "detail": show}
    if active == "inactive" and result == "success":
        return {"state": "completed", "detail": show}
    return {"state": "stale", "detail": show}


def parse_show(text):
    return dict(
        line.split("=", 1) for line in text.strip().splitlines() if "=" in line
    )


class ResourceReservation:
    """Exclusive three-GPU ownership: one LAYA job at a time, flock-held."""

    def __init__(self, root):
        self.path = Path(root) / ".gpu-reservation"
        self.handle = None

    def acquire(self, owner):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"owner": owner, "time": time.time()}) + "\n")
        handle.flush()
        self.handle = handle
        return True

    def owner(self):
        try:
            text = self.path.read_text().strip()
            return json.loads(text).get("owner") if text else None
        except (OSError, ValueError):
            return None

    def release(self):
        if self.handle:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class Operator:
    """Bounded operator over SecurityJobs with a durable run manifest."""

    def __init__(self, profile_path, runner=None, jobs=None):
        self.jobs = jobs or SecurityJobs(profile_path)
        self.profile = self.jobs.profile
        self.runner = runner or self._systemd_runner
        self.reservation = ResourceReservation(
            self.profile.get(
                "resource_root", Path.home() / ".local/state/siliconshield/laya"
            )
        )

    @staticmethod
    def _systemd_runner(argv):
        import subprocess

        run = subprocess.run(argv, capture_output=True, text=True)
        if run.returncode != 0 and "not-found" not in run.stdout:
            return None
        return parse_show(run.stdout)

    def manifest_path(self, experiment_id):
        _, output = command_for(self.profile, experiment_id, {})
        return Path(output) / "run-manifest.json"

    def prepare(self, experiment_id, overrides=None, intervention_budget=None):
        from siliconshield.research_loop.execution import study_contract

        overrides = overrides or {}
        study = study_contract.verify_job(
            self.profile, experiment_id, overrides
        )
        key = idempotency_key(self.profile, experiment_id, overrides)
        inputs = execution_inputs(self.profile, overrides)
        command, output = command_for(self.profile, experiment_id, overrides)
        path = self.manifest_path(experiment_id)
        if path.exists():
            existing = read_json(path)
            if existing["idempotency_key"] != key:
                # A submitted run is authoritative: reattach after operator-code
                # upgrades instead of orphaning an owned job. Drift is recorded.
                if existing.get("attempts", 0) > 0 or existing.get(
                    "phase"
                ) not in (
                    "prepare",
                    "preflight",
                ):
                    submitted = existing["command"]
                    current, _ = command_for(
                        self.profile, experiment_id, overrides
                    )
                    code_drift = (
                        submitted == current
                        and existing.get("execution_inputs") == inputs
                    )
                    kinds = "code" if code_drift else "recipe"
                    existing.setdefault("history", []).append(
                        {
                            "phase": existing.get("phase", "prepare"),
                            "time": time.time(),
                            "note": "reattached with %s drift" % kinds,
                            "submitted_key": existing["idempotency_key"],
                            "current_key": key,
                        }
                    )
                    if not code_drift:
                        raise ValueError(
                            "Submitted run exists; requested recipe differs (not operator-code drift). "
                            "Use the submitted recipe or a new experiment ID."
                        )
                    atomic_json(path, existing)
                    return existing
                raise ValueError(
                    "Experiment ID already belongs to a different recipe (idempotency mismatch)"
                )
            return existing
        manifest = {
            "schema": RUN_MANIFEST_SCHEMA,
            "study": study,
            "study_profile": dict(self.profile) if study else None,
            "experiment_id": experiment_id,
            "idempotency_key": key,
            "phase": "prepare",
            "execution_inputs": inputs,
            "history": [{"phase": "prepare", "time": time.time()}],
            "command": command,
            "overrides": overrides,
            "output": str(output),
            "timeout_seconds": self.profile.get("timeout_seconds", 28800),
            "output_quota_bytes": self.profile.get(
                "output_quota_bytes", 128 * 1024 * 1024
            ),
            "max_retries": self.profile.get("max_retries", 2),
            "attempts": 0,
            "gpus": self.profile.get("gpus", [0, 1, 2]),
            "laya_root": laya_root_for(self.profile),
            "unit": "laya-glimmer-job-" + experiment_id,
            # None takes the default budget. An explicit {} disables
            # interventions rather than silently re-enabling them.
            "intervention_budget": (
                {"max_lr_adjustments": 1, "lr_bounds": [1e-6, 1e-3]}
                if intervention_budget is None
                else intervention_budget
            ),
            "interventions": [],
            "test_touched": False,
            "terminal": None,
        }
        atomic_json(path, manifest)
        return manifest

    def _save(self, manifest, phase, **details):
        manifest["phase"] = phase
        manifest["history"].append(
            {"phase": phase, "time": time.time(), **details}
        )
        atomic_json(self.manifest_path(manifest["experiment_id"]), manifest)
        return manifest

    def _require(self, manifest, phases):
        if manifest["phase"] not in phases:
            raise ValueError(
                "Invalid operator transition from " + manifest["phase"]
            )
        if (
            read_json(self.manifest_path(manifest["experiment_id"]))[
                "idempotency_key"
            ]
            != manifest["idempotency_key"]
        ):
            raise ValueError("Manifest identity mismatch")

    def _owned_state(self, manifest):
        state = systemd_state(manifest["unit"], self.runner)
        if state["state"] not in ("missing",):
            description = (state.get("detail") or {}).get("Description")
            if description != "siliconshield:" + manifest["idempotency_key"]:
                raise RuntimeError("Systemd unit belongs to another owner")
        return state

    def preflight(self, manifest):
        self._require(manifest, ("prepare", "preflight"))
        verify_laya(self.profile, manifest.get("laya_root"))
        self._probe_worker_entrypoint()
        self._probe_disk_space()
        self._probe_settings(manifest)
        if (
            hasattr(self.jobs, "gpu_available")
            and not self.jobs.gpu_available()
        ):
            raise RuntimeError(
                "GPU processes are active; do not disturb existing work"
            )
        trainer = (
            Path(self.profile["source"])
            / "siliconshield/training/laya/security_train.py"
        )
        fitting = {
            **self.profile.get("defaults", {}),
            **manifest["overrides"],
        }.get("limit_train_rows")
        required_inputs = [trainer, Path(self.profile["data"]) / "train.jsonl"]
        if not fitting:
            required_inputs.append(Path(self.profile["data"]) / "dev.jsonl")
        for required in required_inputs:
            if not required.is_file():
                raise ValueError(
                    "Preflight missing required input: " + str(required)
                )
        manifest["input_hashes"] = execution_inputs(
            self.profile, manifest["overrides"]
        )
        if manifest["input_hashes"] != manifest["execution_inputs"]:
            raise ValueError("Execution inputs changed after preparation")
        return self._save(manifest, "preflight")

    def _probe_worker_entrypoint(self):
        """Ensure the job python can import the unit entrypoint.

        Units launch ``owned_job`` with the snapshot as their working
        directory, so a snapshot holding only the trainer fails after
        submission. Probe that import here instead.
        """
        import subprocess

        from siliconshield.research_loop.execution.owned_job import (
            root_cause_from_text,
        )

        probe = subprocess.run(
            [
                self.profile["python"],
                "-c",
                "import siliconshield.research_loop.execution.owned_job",
            ],
            cwd=self.profile["source"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if probe.returncode != 0:
            cause = (
                root_cause_from_text(probe.stderr or "")
                or "exit_code=%d" % probe.returncode
            )
            raise ValueError(
                "Job entrypoint not importable from source snapshot: " + cause
            )

    def _probe_disk_space(self):
        """Refuse submission when the runs disk cannot hold the run.

        Resumable optimizer state, the selected export, and eval
        snapshots each approach parent-export size, so twice that
        size must be free. A run that dies serializing weights
        wastes its GPU allocation and leaves a partial export.
        """
        import shutil

        runs = Path(self.profile["runs"])
        if not runs.is_dir():
            raise ValueError("Runs directory is missing: " + str(runs))
        parent = sum(
            path.stat().st_size
            for path in Path(self.profile["model"]).rglob("*")
            if path.is_file()
        )
        free = shutil.disk_usage(runs).free
        if free < 2 * parent:
            raise ValueError(
                "Runs disk holds %.1f GiB free, below twice the "
                "parent export size (%.1f GiB)"
                % (free / 1024**3, 2 * parent / 1024**3)
            )

    def _probe_settings(self, manifest):
        """Dry-run the frozen trainer command on CPU before submit.

        Flag combinations such as a hold schedule longer than the
        update budget fail during training startup. The trainer
        validates its settings and exits before loading data, so
        this probe catches those defects with no GPU cost. It
        invokes the trainer directly: settings validation needs
        neither the distributed launcher nor a GPU.
        """
        import os
        import subprocess

        from siliconshield.research_loop.execution.owned_job import (
            root_cause_from_text,
        )

        command, _ = command_for(
            self.profile,
            manifest["experiment_id"],
            manifest["overrides"],
        )
        trainer = str(
            Path(self.profile["source"])
            / "siliconshield/training/laya/security_train.py"
        )
        argv = [
            self.profile["python"],
            trainer,
            *command[command.index(trainer) + 1 :],
            "--dry-run",
        ]
        env = dict(os.environ)
        root = manifest.get("laya_root") or self.profile.get("laya_root")
        if root:
            env["LAYA_ROOT"] = str(root)
            env["PYTHONPATH"] = (
                str(root) + os.pathsep + env.get("PYTHONPATH", "")
            )
        probe = subprocess.run(
            argv,
            cwd=self.profile["source"],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            check=False,
        )
        if probe.returncode != 0:
            cause = (
                root_cause_from_text(probe.stderr or "")
                or "exit_code=%d" % probe.returncode
            )
            raise ValueError("Trainer settings rejected: " + cause)

    def submit(self, manifest, action="train"):
        self._require(
            manifest,
            (
                "preflight",
                "submit",
                "observe",
                "diagnose",
                "intervene",
                "finalize",
            ),
        )
        for name, expected in manifest.get("input_hashes", {}).items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError("Prepared input changed: " + name)
        if not manifest.get("input_hashes"):
            raise ValueError("Preflight input identity is missing")
        resources = self.reservation.path.parent
        resources.mkdir(parents=True, exist_ok=True)
        with (resources / ".submission.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self._owned_state(manifest)
            if state["state"] == "running":
                return self._save(
                    manifest, "submit", deduplicated=True, systemd="running"
                )
            execution = self.jobs.status(manifest["experiment_id"]).get(
                "execution.json", {}
            )
            if action == "train" and manifest["attempts"]:
                return self._save(
                    manifest, "observe", systemd=state, deduplicated=True
                )
            owner_path = resources / "owner.json"
            if owner_path.exists():
                owner = read_json(owner_path)
                other_state = systemd_state(owner["unit"], self.runner)
                desc = (other_state.get("detail") or {}).get("Description")
                # A missing (garbage-collected) unit cannot be an active owner; its
                # stale ownership is settled by the terminal-receipt check below.
                if other_state["state"] == "running" or (
                    other_state["state"] != "missing"
                    and desc
                    and desc != "siliconshield:" + owner["identity"]
                ):
                    raise RuntimeError(
                        "GPU resources are owned by an active or foreign run"
                    )
                terminal = Path(owner["output"]) / "execution.json"
                if (
                    not terminal.is_file()
                    or read_json(terminal).get("identity") != owner["identity"]
                ):
                    raise RuntimeError(
                        "Stale GPU ownership needs a matching terminal receipt"
                    )
            if (
                action != "train"
                and manifest["attempts"] >= 1 + manifest["max_retries"]
            ):
                raise ValueError("Retry budget exhausted")
            if (
                execution
                and execution.get("identity") != manifest["idempotency_key"]
            ):
                raise ValueError("Execution receipt identity mismatch")
            from siliconshield.research_loop.execution import study_contract

            study = study_contract.verify_job(
                self.profile, manifest["experiment_id"], manifest["overrides"]
            )
            if study != manifest.get("study"):
                raise ValueError("Study binding changed before submission")
            if study:
                study_manifest = study_contract.load(self.profile)
                base = Path(study["path"]).parent
                study_contract.verify_receipts(
                    study_manifest,
                    read_json(base / "fitting-probe.json"),
                    read_json(base / "resource-check.json"),
                )
            study_contract.reserve_budget(self.profile, manifest)
            manifest["attempts"] += 1
            manifest["unit"] = (
                "laya-glimmer-job-"
                + manifest["experiment_id"]
                + "-attempt-"
                + str(manifest["attempts"])
            )
            self._save(manifest, "submit", action=action)
            atomic_json(
                owner_path,
                {
                    "unit": manifest["unit"],
                    "identity": manifest["idempotency_key"],
                    "output": manifest["output"],
                },
            )
            try:
                result = self.jobs.launch_owned(
                    manifest,
                    self.manifest_path(manifest["experiment_id"]),
                    resources,
                    action,
                )
            except Exception:
                # Keep the reservation if systemd may have accepted the job.
                state = self._owned_state(manifest)
                if state["state"] == "missing":
                    owner_path.unlink(missing_ok=True)
                raise
            return self._save(manifest, "submit", job=result)

    def observe(self, manifest):
        self._require(
            manifest, ("submit", "observe", "diagnose", "intervene", "finalize")
        )
        state = self._owned_state(manifest)
        status = self.jobs.status(manifest["experiment_id"])
        return self._save(
            manifest,
            "observe",
            systemd=state,
            has_report=self._report_valid(manifest, status),
        )

    def _report_valid(self, manifest, status):
        from siliconshield.research_loop.episodes.laya_results import (
            adapt_report,
        )

        report = status.get("report.json")
        execution = status.get("execution.json", {})
        if (
            execution.get("identity") != manifest["idempotency_key"]
            or execution.get("phase") != "completed"
        ):
            return False
        if (
            not isinstance(report, dict)
            or report.get("run_id") != manifest["experiment_id"]
        ):
            return False
        try:
            if report.get("evaluation_role") == "train_fitting":
                if (
                    report.get("test_evaluated") is not False
                    or report.get("final_evaluation") is not False
                    or set(report["signature"]["data"]) != {"train"}
                    or report["signature"].get("evaluation_role")
                    != "train_fitting"
                    or report["progress"]["last_eval_step"]
                    != report["progress"]["step"]
                ):
                    return False
                for key in ("initial_train", "train", "final_train"):
                    if (
                        report[key]["n"] <= 0
                        or not 0 <= report[key]["accuracy"] <= 1
                        or not math.isfinite(report[key]["nll"])
                    ):
                        return False
            else:
                adapt_report(report)
            return (
                Path(manifest["output"]) / "report.json"
            ).stat().st_mtime >= execution["started"]
        except (ValueError, KeyError, OSError):
            return False

    def diagnose(self, manifest):
        self._require(manifest, ("observe",))
        observation = manifest["history"][-1]
        state = observation.get("systemd", {}).get("state")
        if observation.get("has_report"):
            diagnosis = "healthy_complete"
        elif state == "running":
            diagnosis = "healthy_running"
        elif state == "failed":
            diagnosis = "execution_failure"
        elif state == "missing":
            diagnosis = "missing_unit"
        else:
            diagnosis = "unresolved"
        return self._save(manifest, "diagnose", diagnosis=diagnosis)

    def intervene(self, manifest, action, **params):
        self._require(manifest, ("diagnose", "intervene", "observe"))
        state = self._owned_state(manifest)
        if action not in BOUNDED_INTERVENTIONS:
            raise ValueError("Intervention is not allowlisted: " + str(action))
        budget = manifest.get("intervention_budget", {})
        used = [
            i
            for i in manifest.get("interventions", [])
            if i["action"] == action
        ]
        if action == "adjust_lr":
            if len(used) >= budget.get("max_lr_adjustments", 0):
                raise ValueError("Learning-rate intervention budget exhausted")
            low, high = budget.get("lr_bounds", [0, 0])
            lr = params.get("lr")
            if (
                isinstance(lr, bool)
                or not isinstance(lr, (int, float))
                or not math.isfinite(lr)
                or not low <= lr <= high
            ):
                raise ValueError("Learning rate outside pre-registered bounds")
        if action in ("stop", "adjust_lr"):
            if state["state"] != "running":
                raise ValueError(
                    "This intervention requires an owned running job"
                )
            request = {
                "run_id": manifest["experiment_id"],
                "sequence": len(manifest["interventions"]) + 1,
                "action": action,
                **params,
            }
            atomic_json(
                Path(manifest["output"]) / "operator-control.json", request
            )
        if action in ("resume", "finalize_retry"):
            if state["state"] == "running":
                raise ValueError("Cannot resume or finalize a running job")
            self._validate_resume(
                manifest, params.get("checkpoint_dir", manifest["output"])
            )
            self.submit(manifest, action=action)
        manifest.setdefault("interventions", []).append(
            {"action": action, "params": params, "time": time.time()}
        )
        return self._save(manifest, "intervene", action=action)

    def _validate_resume(self, manifest, checkpoint_dir):
        if manifest.get("test_touched"):
            raise ValueError(
                "Test-touched run lineage may not resume into further training"
            )
        if not checkpoint_dir:
            raise ValueError("Resume requires an explicit checkpoint directory")
        root = Path(checkpoint_dir).resolve()
        if root != Path(manifest["output"]).resolve():
            raise ValueError(
                "Checkpoint must belong to the registered output directory"
            )
        if (root / "test_touch.json").exists():
            raise ValueError("Test-touched checkpoint lineage")
        import torch

        try:
            checkpoint = torch.load(
                root / "latest.pt", map_location="cpu", weights_only=False
            )
            configuration = read_json(root / "configuration.json")
            signature = checkpoint["signature"]
            if (
                signature != configuration["signature"]
                or signature["run_id"] != manifest["experiment_id"]
            ):
                raise ValueError("Incompatible checkpoint signature")
            if not set(signature["data"]) <= {"train", "dev"}:
                raise ValueError("Test-touched checkpoint signature")
            for split, expected in signature["data"].items():
                if (
                    hashlib.sha256(
                        (
                            Path(self.profile["data"]) / (split + ".jsonl")
                        ).read_bytes()
                    ).hexdigest()
                    != expected
                ):
                    raise ValueError("Checkpoint data changed")
            if (
                not checkpoint["model"]
                or not checkpoint["optimizer"]["param_groups"]
            ):
                raise ValueError("Incomplete model or optimizer")
            progress = checkpoint["progress"]
            if any(
                type(progress[k]) is not int or progress[k] < 0
                for k in ("step", "epoch", "offset")
            ):
                raise ValueError("Invalid data position")
            rng = checkpoint["rng"]
            if len(rng) != signature["world"] or any(
                not {"torch", "cuda", "numpy", "python"} <= set(r) for r in rng
            ):
                raise ValueError("Missing rank RNG state")
            for state in rng:
                if any(
                    not isinstance(state[k], torch.Tensor)
                    or state[k].dtype != torch.uint8
                    or state[k].ndim != 1
                    or not state[k].numel()
                    for k in ("torch", "cuda")
                ):
                    raise ValueError("Invalid rank RNG representation")
                if not isinstance(state["python"], tuple) or not isinstance(
                    state["numpy"], tuple
                ):
                    raise ValueError("Invalid Python/NumPy RNG state")
        except (OSError, KeyError, TypeError, RuntimeError) as exc:
            raise ValueError("No valid LAYA resumable checkpoint") from exc
        return {"step": progress["step"], "signature": signature}

    def finalize(self, manifest):
        self._require(
            manifest, ("observe", "diagnose", "intervene", "finalize")
        )
        status = self.jobs.status(manifest["experiment_id"])
        if not self._report_valid(manifest, status):
            return self._save(
                manifest,
                "finalize",
                finalized=False,
                detail="No trainer report yet; finalization may be retried without new updates",
            )
        return self._save(manifest, "finalize", finalized=True)

    def report(self, manifest):
        self._require(manifest, ("finalize", "report"))
        status = self.jobs.status(manifest["experiment_id"])
        terminal = (
            "complete" if self._report_valid(manifest, status) else "incomplete"
        )
        manifest = self._save(manifest, "report", terminal=terminal)
        manifest["terminal"] = terminal
        atomic_json(self.manifest_path(manifest["experiment_id"]), manifest)
        return manifest

    def abandon(self, manifest):
        """Terminalize a dead attempt that left no usable receipt.

        A unit that dies before its worker writes anything leaves no
        execution receipt, so failure detection and reservation
        release cannot proceed. Abandon records the terminal failure
        under the attempt identity and marks the experiment failed.
        Refuses live jobs, valid reports, and attempts that could
        still resume from checkpoint state.
        """
        self._require(
            manifest,
            ("submit", "observe", "diagnose", "intervene", "finalize"),
        )
        state = self._owned_state(manifest)
        if state["state"] == "running":
            raise ValueError("Cannot abandon a running job")
        status = self.jobs.status(manifest["experiment_id"])
        if self._report_valid(manifest, status):
            raise ValueError("Cannot abandon a run with a valid report")
        if (
            manifest["attempts"] < 1 + manifest["max_retries"]
            and (Path(manifest["output"]) / "latest.pt").exists()
        ):
            raise ValueError("Retry budget remains with resumable state")
        output = Path(manifest["output"])
        output.mkdir(parents=True, exist_ok=True)
        receipt = status.get("execution.json", {})
        if receipt.get("identity") != manifest[
            "idempotency_key"
        ] or receipt.get("phase") not in ("failed", "completed"):
            atomic_json(
                output / "execution.json",
                {
                    "identity": manifest["idempotency_key"],
                    "unit": manifest["unit"],
                    "action": receipt.get("action", "train"),
                    "phase": "failed",
                    "failure": "abandoned: unit %s with no usable receipt"
                    % state["state"],
                    "finished": time.time(),
                },
            )
        manifest = self._save(
            manifest, "report", terminal="failed", unit_state=state["state"]
        )
        manifest["terminal"] = "failed"
        atomic_json(self.manifest_path(manifest["experiment_id"]), manifest)
        return manifest
