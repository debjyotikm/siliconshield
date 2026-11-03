"""Browser-facing orchestration over the maintained worker and study APIs."""

import copy
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, cast

from siliconshield.app.laya_assessor import checkpoint_manifest
from siliconshield.experts.arbitration import arbitrate_assessment
from siliconshield.experts.contracts import validate_result
from siliconshield.experts.service import InferenceService
from siliconshield.experts.worker import WorkerPool
from siliconshield.research_loop.episodes.contracts import atomic_json
from siliconshield.research_loop.execution.operator import Operator
from siliconshield.web_demo.glimmer_client import LocalGlimmerClient
from siliconshield.web_demo.toy_data import create_dataset

LABELS = [
    "satisfies_requirement",
    "violates_requirement",
    "insufficient_context",
]


def read_json(path: Path, fallback: Any = None) -> Any:
    """Read an optional report which may not yet have been published."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return fallback


def run_command(
    argv: list[str], timeout: int = 20, env: dict[str, str] | None = None
) -> str:
    """Run a bounded, administrator-configured command without a shell."""
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            (result.stderr or result.stdout or "Command failed")[-3000:]
        )
    return result.stdout


def gpu_status() -> list[dict[str, str]]:
    """Read GPU status, never loading a model or stopping a process."""
    text = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    return [
        dict(
            zip(
                ["index", "name", "memory_used", "memory_total", "utilization"],
                [x.strip() for x in row.split(",")],
            )
        )
        for row in text.strip().splitlines()
    ]


def state_text(body: dict[str, Any]) -> str:
    """Preserve supplied RTL and refuse oversized inputs before inference."""
    fields = {}
    for field, limit in [
        ("rtl", 50000),
        ("requirement", 3000),
        ("assumptions", 3000),
    ]:
        value = body.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(field + " is required.")
        if len(value) > limit:
            raise ValueError(
                field
                + " exceeds the demo input limit. Submit a self-contained region."
            )
        fields[field] = value.strip()
    return (
        "REQUIREMENT\n"
        + fields["requirement"]
        + "\n\nASSUMPTIONS\n"
        + fields["assumptions"]
        + "\n\nRTL\n"
        + fields["rtl"]
        + "\n"
    )


class Demo:
    """Own this session's workers and retain actual execution receipts."""

    def __init__(
        self, config: dict[str, Any], *, health_poll: bool = True
    ) -> None:
        self.config = config
        self.state = Path(config["state_dir"])
        self.state.mkdir(parents=True, exist_ok=True, mode=448)
        self.entries = {entry["id"]: entry for entry in config["entries"]}
        self.pools: dict[str, WorkerPool] = {}
        self.loaded: dict[str, dict[str, Any]] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.training: dict[str, Any] | None = None
        self.preparing_local = False
        self.health: dict[str, Any] = {
            "glimmer": {"status": "not configured"},
            "gpus": [],
        }
        self.stopping = False
        pointer = self.state / "active-training.json"
        if pointer.exists():
            saved = json.loads(pointer.read_text())
            self._validate_training_pointer(saved)
            self.training = saved
        if health_poll:
            threading.Thread(target=self.health_loop, daemon=True).start()

    def _validate_training_pointer(self, record: dict[str, Any]) -> None:
        """Refuse corrupt or foreign restart pointers before accessing jobs."""
        if not isinstance(record, dict) or not self.config["training_root"]:
            raise ValueError(
                "Existing training pointer requires its original output directory."
            )
        run_id = record.get("id", "")
        if not isinstance(run_id, str) or not re.fullmatch(
            "\\d{8}-\\d{6}-[a-f0-9]{6}", run_id
        ):
            raise ValueError(
                "Malformed training pointer: inspect the saved session."
            )
        expected = Path(self.config["training_root"]).resolve() / run_id
        if (
            Path(record.get("directory", "")).resolve() != expected
            or record.get("unit") != "siliconshield-web-toy-" + run_id
        ):
            raise ValueError(
                "Training pointer does not belong to this configured session."
            )

    def inference_device(self, value: str | None = None) -> str:
        """Select only CPU or the explicitly configured CUDA device."""
        gpu = self.config["gpu"]
        selected = (
            ("cpu" if gpu is None else f"cuda:{gpu}")
            if value is None
            else value
        )
        allowed = ["cpu"] + ([] if gpu is None else [f"cuda:{gpu}"])
        if not isinstance(selected, str) or selected not in allowed:
            raise ValueError("Select CPU or the configured GPU.")
        return selected

    def device_name(self, device: str) -> str:
        """Name the server-side execution device."""
        return "CPU" if device == "cpu" else "GPU " + device.split(":")[1]

    def glimmer_health(self) -> dict[str, Any]:
        """Check an explicitly configured local service without generation."""
        cfg = self.config["glimmer"]
        if not cfg:
            return {"status": "not configured"}
        client = LocalGlimmerClient(**cfg, timeout=8)
        reply = client.models()
        if cfg["model"] not in [model["id"] for model in reply["data"]]:
            raise RuntimeError("Configured Glimmer model is not served.")
        return {
            "status": "ready",
            "model": cfg["model"],
            "transport": "SSH" if cfg.get("ssh_host") else "Loopback HTTP",
        }

    def check_storage(self) -> Path:
        """Keep toy checkpoints on the configured external output volume."""
        root = Path(self.config["training_root"])
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < 64 * 1024**3:
            raise RuntimeError(
                "Toy training requires at least 64 GiB free on its output volume."
            )
        if shutil.disk_usage(self.state).free < 5 * 1024**3:
            raise RuntimeError(
                "State filesystem has less than 5 GiB headroom. Training was not launched."
            )
        return root

    def health_loop(self) -> None:
        """Poll configured services without dispatching inference or training."""
        while not self.stopping:
            try:
                self.health["gpus"] = (
                    gpu_status() if self.config["gpu"] is not None else []
                )
                self.health.pop("gpu_error", None)
            except Exception as exc:
                self.health["gpu_error"] = str(exc)
            try:
                self.health["glimmer"] = self.glimmer_health()
            except Exception as exc:
                self.health["glimmer"] = {
                    "status": "unavailable",
                    "error": str(exc),
                }
            for _ in range(30):
                if self.stopping:
                    return
                time.sleep(1)

    def close(self) -> None:
        """Close only the inference workers owned by this application."""
        self.stopping = True
        with self.inference_lock:
            for pool in self.pools.values():
                pool.close()
            self.pools.clear()
            self.loaded.clear()

    def training_active(self) -> bool:
        """Reconcile controller receipts and independently owned jobs."""
        if not self.training:
            return False
        folder = Path(self.training["directory"])
        terminal = read_json(folder / "controller-result.json")
        if terminal:
            return self.owned_job_running(folder)
        if self.training.get("status") in ["complete", "error", "stopped"]:
            return self.owned_job_running(folder)
        if (
            self.training.get("status") in ["preparing", "starting"]
            and self.preparing_local
        ):
            return True
        try:
            state = run_command(
                [
                    "systemctl",
                    "--user",
                    "show",
                    self.training["unit"],
                    "--property=ActiveState",
                    "--value",
                ],
                timeout=5,
            ).strip()
        except Exception:
            return True
        if state in ["active", "activating", "reloading"]:
            self.training["status"] = "running"
            return True
        if self.owned_job_running(folder):
            return True
        self.training["status"] = "error"
        self.training["error"] = (
            "Controller exited without a completion receipt. Inspect the saved run before retrying."
        )
        atomic_json(self.state / "active-training.json", self.training)
        return False

    def owned_job_running(self, folder: Path) -> bool:
        """Keep a live child job visible even after controller failure."""
        for path in (folder / "runs").glob("*/run-manifest.json"):
            manifest = read_json(path, {})
            try:
                state = run_command(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        manifest["unit"],
                        "--property=ActiveState",
                        "--value",
                    ],
                    timeout=5,
                ).strip()
                if state in ["active", "activating"]:
                    return True
            except Exception:
                return True
        return False

    def submit_task(self, kind: str, fn: Any) -> dict[str, Any]:
        """Run one bounded background request and retain its real result."""
        job_id = uuid.uuid4().hex
        with self.lock:
            if sum((j["status"] == "running" for j in self.jobs.values())) >= 8:
                raise ValueError(
                    "Too many active requests. Wait for the current work."
                )
            self.jobs[job_id] = {
                "id": job_id,
                "kind": kind,
                "status": "running",
                "started": time.time(),
                "messages": [],
            }

        def log(message: str) -> None:
            with self.lock:
                self.jobs[job_id]["messages"].append(
                    {"time": time.time(), "text": message}
                )

        def worker() -> None:
            try:
                value = fn(log)
                with self.lock:
                    self.jobs[job_id].update(
                        status="complete", result=value, finished=time.time()
                    )
            except Exception as exc:
                traceback.print_exc()
                with self.lock:
                    self.jobs[job_id].update(
                        status="error", error=str(exc), finished=time.time()
                    )
            atomic_json(
                self.state / f"request-{job_id}.json", self.jobs[job_id]
            )

        threading.Thread(target=worker, daemon=True).start()
        return {"job_id": job_id}

    def ensure_model(
        self, model: str, log: Any, device: str | None = None
    ) -> dict[str, Any]:
        """Verify and load an immutable model on the requested device."""
        if self.stopping:
            raise ValueError("This demo session is stopping.")
        if model not in self.entries:
            raise ValueError("Unknown decision model.")
        device = self.inference_device(device)
        if self.training_active():
            raise ValueError(
                "Toy training owns this demo session. Wait for closeout before loading models."
            )
        if model in self.loaded and self.loaded[model].get("device") == device:
            return self.loaded[model]
        if device.startswith("cuda:"):
            processes = run_command(
                [
                    "nvidia-smi",
                    f"--id={self.config['gpu']}",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ]
            ).split()
            owned = {
                str(p.pid)
                for pool in self.pools.values()
                for p in pool._workers.values()
            }
            if set(processes) - owned:
                raise RuntimeError(
                    "Demo GPU has an unrelated compute process. No process was interrupted."
                )
        if model in self.pools:
            log(f"Unloading {model} before changing inference device.")
            self.pools.pop(model).close()
        self.loaded.pop(model, None)
        entry = copy.deepcopy(self.entries[model])
        entry["device"] = device
        env = {
            "PYTHONPATH": self.config["source"],
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        log(
            f"Loading {model} on {self.device_name(device)}. Verifying checkpoint and runtime."
        )
        pool = WorkerPool(extra_env=env)
        try:
            pool.load(entry)
            ready = pool.ready(model)
            if ready.get("device") != device:
                raise RuntimeError(
                    "Worker device differs from the requested device."
                )
        except Exception:
            pool.close()
            raise
        self.pools[model] = pool
        self.loaded[model] = ready
        log(f"{model} loaded on {self.device_name(device)}.")
        return ready

    def load(
        self, model: str, log: Any, device: str | None = None
    ) -> dict[str, Any]:
        """Serialize worker loading against inference and training launch."""
        with self.inference_lock:
            return self.ensure_model(model, log, device)

    def unload(self, model: str) -> dict[str, Any]:
        """Release only this session's selected model worker."""
        with self.inference_lock:
            if model in self.pools:
                self.pools.pop(model).close()
            self.loaded.pop(model, None)
        return {"status": "unloaded"}

    def assess(self, body: dict[str, Any], log: Any) -> dict[str, Any]:
        """Score the supplied text through the real identity-checked worker."""
        text = state_text(body)
        model = body.get("model")
        if not isinstance(model, str):
            raise ValueError("A model ID is required.")
        device = self.inference_device(body.get("device"))
        with self.inference_lock:
            self.ensure_model(model, log, device)
            log(
                "Assessing the RTL against the supplied requirement and assumptions."
            )
            started = time.monotonic()
            try:
                result = cast(
                    dict[str, Any], self.pools[model].analyze(model, text)
                )
            except Exception:
                self.pools.pop(model).close()
                self.loaded.pop(model, None)
                raise
            result["scores"] = validate_result(result)["scores"]
            result["wall_seconds"] = time.monotonic() - started
            result["input"] = {
                k: body[k] for k in ["rtl", "requirement", "assumptions"]
            }
            log("Decision received. All three class scores are available.")
            return result

    def route(self, body: dict[str, Any], log: Any) -> dict[str, Any]:
        """Use the capability router and optionally interpret its evidence."""
        text = state_text(body)
        scope = body.get("scope")
        if not isinstance(scope, str) or scope not in [
            "fifo",
            "counter",
            "arbitration",
            "ot-access-control",
        ]:
            raise ValueError("Select a measured routing domain.")
        if not self.config["registry"]:
            raise ValueError("Register decision models before routing.")
        service = InferenceService(self.config["registry"])
        with self.inference_lock:
            if self.training_active():
                raise ValueError("Wait for the toy training closeout.")

            def consult(model: str, state: str) -> dict[str, Any]:
                self.ensure_model(
                    model, log, service.model_entry(model)["device"]
                )
                return cast(
                    dict[str, Any], self.pools[model].analyze(model, state)
                )

            log("Routing by measured domain coverage: " + scope + ".")
            assessment = service.assess([scope], text, consult)
        result: dict[str, Any] = {
            "assessment": assessment,
            "input": {
                k: body[k] for k in ["rtl", "requirement", "assumptions"]
            },
        }
        if body.get("arbitrate") is True and (not self.config["glimmer"]):
            result["arbitration_error"] = (
                "Configure local Glimmer to request arbitration."
            )
            return result
        if body.get("arbitrate") is True:
            log(
                "Local Muse Glimmer is reviewing the identity-bound specialist evidence."
            )
            try:
                result["arbitration"] = arbitrate_assessment(
                    assessment,
                    text,
                    body["requirement"],
                    profiles=service.profiles_for(service.expert_ids),
                    assumptions=body["assumptions"],
                    client_factory=lambda: LocalGlimmerClient(
                        **self.config["glimmer"],
                        max_tokens=1536,
                        timeout=180,
                        max_calls=1,
                        audit_dir=self.state / "arbitration",
                        decode_arguments=True,
                    ),
                )
                log("Muse Glimmer returned a validated arbitration response.")
            except Exception as exc:
                result["arbitration_error"] = str(exc)
                log(
                    "Arbitration failed. Specialist evidence remains available, without an agent verdict."
                )
        result["input"] = {
            k: body[k] for k in ["rtl", "requirement", "assumptions"]
        }
        return result

    def prepare_training(self, log: Any) -> dict[str, Any]:
        """Launch one fixed operability study after explicit preflight."""
        if not self.config["training_enabled"]:
            raise ValueError(
                "Toy training is disabled. Configure training.enabled explicitly."
            )
        with self.inference_lock:
            if self.training_active():
                raise ValueError("A demo training run is already active.")
            for pool in self.pools.values():
                pool.close()
            self.pools.clear()
            self.loaded.clear()
            log(
                "Released this demo’s inference workers. Checking GPU and Glimmer availability."
            )
            if run_command(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ]
            ).strip():
                raise RuntimeError(
                    "A GPU workload is active. The training preflight refuses to disturb it."
                )
            self.glimmer_health()
            training_root = self.check_storage()
            parent = self.entries["contract-1"]
            if (
                checkpoint_manifest(parent["checkpoint"])["files"]
                != parent["expected_files"]
            ):
                raise ValueError(
                    "Training parent files differ from the registered export."
                )
            run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            folder = training_root / run_id
            folder.mkdir(parents=True)
            unit = "siliconshield-web-toy-" + run_id
            self.preparing_local = True
            self.training = {
                "id": run_id,
                "directory": str(folder),
                "unit": unit,
                "status": "preparing",
                "started": time.time(),
            }
            atomic_json(self.state / "active-training.json", self.training)
        try:
            log(
                "Constructing 18 TRAIN and 6 development decisions in complete three-label groups."
            )
            summary = create_dataset(folder / "data")
            atomic_json(folder / "dataset-summary.json", summary)
            atomic_json(
                folder / "recipe.json", {"source_mass": {"toy-synthetic": 1.0}}
            )
            updates = self.config.get("toy_updates", 3)
            schedule = {
                "max_updates": updates,
                "min_updates": updates,
                "warmup": 2,
                "hold_updates": 0,
                "patience": 9,
                "eval_every_updates": 3,
                "lr_schedule": "constant",
                "lr_endpoint": 1.0,
            }
            profile = {
                "python": self.config["training_python"],
                "source": self.config["source"],
                "laya_root": self.config["training_runtime"],
                "model": self.entries["contract-1"]["checkpoint"],
                "data": str(folder / "data"),
                "runs": str(folder / "runs"),
                "gpus": [self.config["gpu"]],
                "recipe": str(folder / "recipe.json"),
                "resource_root": self.config["resource_root"],
                "max_retries": 0,
                "timeout_seconds": 600,
                "defaults": {
                    **schedule,
                    "continue_training": True,
                    "freeze_encoder": True,
                    "adapter_lr": 1e-05,
                    "head_lr": 1e-05,
                    "micro_batch": 3,
                    "accumulation": 2,
                    "seed": 7,
                    "sampling": "grouped",
                    "pools": str(folder / "data/pools.json"),
                    "question_contract": "siliconshield-requirement-context-v1",
                    "no_diagnostic_weights": True,
                },
            }
            atomic_json(folder / "profile.json", profile)
            campaign: dict[str, Any] = {
                "profile": str(folder / "profile.json"),
                "output": str(folder / "campaign"),
                "experiment_prefix": "webtoy-" + run_id,
                "timeout_seconds": 1500,
                "min_fit_gain": 0.0,
                "operator": {
                    **self.config["glimmer"],
                    "max_cycles": 6,
                    "poll_seconds": 5,
                    "intervention_budget": {},
                },
                "fitting_common": {
                    **schedule,
                    "no_development_in_fitting": True,
                },
                "arms": [
                    {"name": "adapter", "overrides": {"limit_train_rows": 18}}
                ],
                "development_overrides": schedule,
            }
            atomic_json(folder / "campaign.json", campaign)
            log(
                "Checking contract bytes, split isolation, sampler exposure, and reporting settings."
            )
            env = {
                **os.environ,
                "PYTHONPATH": self.config["source"],
                "LAYA_ROOT": self.config["training_runtime"],
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            command = [
                self.config["training_python"],
                "-m",
                "siliconshield.training.laya.validate_study_inputs",
                "--data",
                str(folder / "data"),
                "--pools",
                str(folder / "data/pools.json"),
                "--recipe",
                str(folder / "recipe.json"),
                "--question-contract",
                "siliconshield-requirement-context-v1",
                "--seed",
                "7",
                "--global-batch",
                "6",
                "--world",
                "1",
                "--max-updates",
                str(updates),
                "--output-receipt",
                str(folder / "input-validation.json"),
            ]
            output = run_command(command, timeout=120, env=env)
            (folder / "validation.log").write_text(output)
            receipt = read_json(folder / "input-validation.json", {})
            if receipt.get("result") != "pass":
                raise RuntimeError("Input validation did not pass.")
            if (folder / "cancel.request").exists():
                self.training["status"] = "stopped"
                atomic_json(self.state / "active-training.json", self.training)
                return self.training
            log(
                "Input preflight passed. Starting the bounded operator controller."
            )
            argv = [
                "systemd-run",
                "--user",
                "--quiet",
                "--unit=" + unit,
                "--property=Type=exec",
                "--property=RuntimeMaxSec=1800",
                "--property=WorkingDirectory=" + str(folder),
                "--property=StandardOutput=append:"
                + str(folder / "controller.log"),
                "--property=StandardError=append:"
                + str(folder / "controller.log"),
                "--setenv=PYTHONDONTWRITEBYTECODE=1",
                "--setenv=PYTHONPATH=" + self.config["source"],
                self.config["python"],
                "-m",
                "siliconshield.web_demo.toy_runner",
                str(folder / "campaign.json"),
            ]
            run_command(argv, timeout=30)
            self.training["status"] = "running"
            atomic_json(self.state / "active-training.json", self.training)
            log(
                "Training controller is running. The dashboard follows actual job receipts and events."
            )
            return self.training
        except Exception as exc:
            self.training.update(status="error", error=str(exc))
            atomic_json(self.state / "active-training.json", self.training)
            raise
        finally:
            self.preparing_local = False

    def stop_training(self) -> dict[str, Any]:
        """Request a checkpoint-boundary stop for this demo's owned jobs."""
        if not self.training:
            return {
                "status": "idle",
                "enabled": self.config["training_enabled"],
            }
        folder = Path(self.training["directory"])
        (folder / "cancel.request").touch()
        if not (folder / "profile.json").exists():
            return {
                "status": "stop_requested",
                "messages": ["Preparation will stop before launch."],
            }
        op = Operator(str(folder / "profile.json"))
        messages = []
        for file in (folder / "runs").glob("*/run-manifest.json"):
            manifest = read_json(file, {})
            if manifest.get("phase") in [
                "submit",
                "observe",
                "diagnose",
                "intervene",
            ]:
                try:
                    if manifest["phase"] == "submit":
                        manifest = op.observe(manifest)
                    op.intervene(manifest, "stop")
                    messages.append(
                        "Checkpoint-boundary stop requested for "
                        + manifest["experiment_id"]
                    )
                except Exception as exc:
                    messages.append(str(exc))
        return {"status": "stop_requested", "messages": messages}

    def training_status(self) -> dict[str, Any]:
        """Read receipts without changing training decisions or metrics."""
        if not self.training:
            return {
                "status": "idle",
                "enabled": self.config["training_enabled"],
            }
        active = self.training_active()
        value = copy.deepcopy(self.training)
        folder = Path(value["directory"])
        terminal = read_json(folder / "controller-result.json")
        campaign = read_json(folder / "campaign/campaign-status.json", {})
        value["updates_per_phase"] = (
            read_json(folder / "profile.json", {})
            .get("defaults", {})
            .get("max_updates", self.config.get("toy_updates", 3))
        )
        value["active"] = active
        value["phase"] = campaign.get("phase", "preparing")
        value["dataset"] = read_json(folder / "dataset-summary.json", {})
        value["validation"] = read_json(folder / "input-validation.json")
        value["jobs"] = []
        value["operator_cycles"] = []
        if terminal and (not active):
            value["status"] = terminal["status"]
            value["error"] = terminal.get("error")
            value["reason"] = terminal.get("reason")
        if (folder / "cancel.request").exists():
            value["stop_requested"] = True
        for manifest_path in sorted(
            (folder / "runs").glob("*/run-manifest.json"),
            key=lambda p: ("-fit-" not in p.parent.name, p.parent.name),
        ):
            manifest = read_json(manifest_path, {})
            path = manifest_path.parent
            events = []
            try:
                for line in (path / "events.jsonl").read_text().splitlines():
                    try:
                        row = json.loads(line)
                        event = row.get("event", "")
                        if event == "train":
                            events.append(
                                {
                                    k: row.get(k)
                                    for k in [
                                        "event",
                                        "step",
                                        "loss",
                                        "step_seconds",
                                        "time",
                                    ]
                                }
                            )
                        elif "development" in event or event in [
                            "evaluation",
                            "eval",
                            "train_fitting",
                        ]:
                            metrics = row.get(
                                "metrics", row.get("development", {})
                            )
                            events.append(
                                {
                                    "event": event,
                                    "step": row.get("step", 0),
                                    "accuracy": metrics.get("accuracy"),
                                    "nll": metrics.get("nll"),
                                    "false_safe_count": metrics.get(
                                        "false_safe_count"
                                    ),
                                }
                            )
                    except (ValueError, TypeError):
                        pass
            except OSError:
                pass
            report = read_json(path / "report.json", {})
            value["jobs"].append(
                {
                    "name": path.name,
                    "phase": manifest.get("phase"),
                    "terminal": manifest.get("terminal"),
                    "events": events,
                    "report": report,
                    "execution": read_json(path / "execution.json", {}),
                    "directory": str(path),
                }
            )
        for path in sorted(
            (folder / "campaign").glob("*/operator-status.json"),
            key=lambda p: ("-fit-" not in p.parent.name, p.parent.name),
        ):
            record = read_json(path, {})
            for cycle in record.get("cycles", []):
                value["operator_cycles"].append(
                    {"experiment": record.get("experiment"), **cycle}
                )
        value["campaign"] = campaign
        log = folder / "controller.log"
        if log.exists():
            value["log"] = log.read_text(errors="replace")[-12000:]
        return value

    def status(self) -> dict[str, Any]:
        """Report actual model readiness, profiles, and configured services."""
        service = (
            InferenceService(self.config["registry"])
            if self.config["registry"]
            else None
        )
        return {
            "models": [
                {
                    "id": key,
                    "display_name": entry["display_name"],
                    "device": self.loaded.get(key, {}).get(
                        "device", entry["device"]
                    ),
                    "default_device": entry["device"],
                    "loaded": key in self.loaded,
                    "ready": self.loaded.get(key),
                    "profile": service.profiles_for([key]).get(key)
                    if service
                    else None,
                }
                for key, entry in self.entries.items()
            ],
            "health": copy.deepcopy(self.health),
            "training": self.training_status(),
            "training_enabled": self.config["training_enabled"],
            "glimmer_configured": bool(self.config["glimmer"]),
        }
