"""Prepare or submit three-GPU SiliconShield-RTL jobs via systemd."""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    digest,
    read_json,
)

FLAGS = {
    "encoder_lr": "--encoder-lr",
    "head_lr": "--head-lr",
    "lr_endpoint": "--lr-endpoint",
    "max_updates": "--max-updates",
    "micro_batch": "--micro-batch",
    "eval_batch_size": "--eval-batch-size",
    "accumulation": "--accumulation",
    "seed": "--seed",
    "adapter_lr": "--adapter-lr",
    "adapter_scale": "--adapter-scale",
    "max_len": "--max-len",
    "head_max_len": "--head-max-len",
    "limit_train_rows": "--limit-train-rows",
    "eval_every_updates": "--eval-every-updates",
    "min_updates": "--min-updates",
    "patience": "--patience",
    "warmup": "--warmup",
    "hold_updates": "--hold-updates",
    "report_bootstrap": "--report-bootstrap",
    "epochs": "--epochs",
    "checkpoint_interval": "--checkpoint-interval",
    "pair_loss_weight": "--pair-loss-weight",
    "pair_loss_margin": "--pair-loss-margin",
    "encoder_trainable_blocks": "--encoder-trainable-blocks",
}
# store_true flags: bare when True, omitted when False.
BOOL_FLAGS = {
    "freeze_encoder": "--freeze-encoder",
    "disable_adapters": "--disable-adapters",
    "no_development_in_fitting": "--no-development-in-fitting",
    "continue_training": "--continue-training",
    "no_diagnostic_weights": "--no-diagnostic-weights",
    "triplet_shards": "--triplet-shards",
    "plateau_requires_full_exposure": "--plateau-requires-full-exposure",
}
# Local file paths carried into the job so a continuation runs the registered
# development-view and study configuration instead of silently dropping them.
PATH_FLAGS = {
    "development_views": "--development-views",
    "new_study_manifest": "--new-study-manifest",
    "pools": "--pools",
    "review_ledger": "--review-ledger",
    "review_packets": "--review-packets",
}
# Enumerated scheduling choices carried into the job.
CHOICE_FLAGS = {
    "lr_schedule": ("--lr-schedule", ("constant", "cosine", "hold-decay")),
    "sampling": ("--sampling", ("weighted", "grouped")),
    "question_contract": (
        "--question-contract",
        (
            "siliconshield-requirement-context-v0",
            "siliconshield-requirement-context-v1",
        ),
    ),
}


def preservation_teacher_files(profile: dict) -> list[Path]:
    """Return explicit teacher artifacts that must be bound before launch.

    The default teacher is already covered by the initialization checkpoint.
    A separate consolidation teacher is a real export, not merely a pathname
    in the recipe. Hash its complete file tree in both preparation paths.
    """
    if not profile.get("recipe"):
        return []
    recipe = read_json(profile["recipe"])
    if not isinstance(recipe, dict):
        raise ValueError("Training recipe must be a JSON object")
    config = recipe.get("replay_preservation")
    if not isinstance(config, dict) or "teacher_checkpoint" not in config:
        return []
    value = config["teacher_checkpoint"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Preservation teacher checkpoint must name a path")
    root = Path(value)
    for name in ("model.safetensors", "rl_agent_config.json"):
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Incomplete preservation teacher export: " + name)
    return sorted(path for path in root.rglob("*") if path.is_file())


def command_for(profile, experiment_id, overrides):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", experiment_id):
        raise ValueError("Experiment ID must be a simple stable identifier")
    if (
        set(overrides)
        - set(FLAGS)
        - set(BOOL_FLAGS)
        - set(PATH_FLAGS)
        - set(CHOICE_FLAGS)
    ):
        raise ValueError("Unknown training override")
    values = {**profile.get("defaults", {}), **overrides}
    for key, value in values.items():
        if key in BOOL_FLAGS:
            if not isinstance(value, bool):
                raise ValueError("Flag setting must be boolean: " + key)
            continue
        if key in PATH_FLAGS:
            if not isinstance(value, str) or not value:
                raise ValueError(
                    "Path setting must be a non-empty path: " + key
                )
            continue
        if key in CHOICE_FLAGS:
            _, allowed = CHOICE_FLAGS[key]
            if value not in allowed:
                raise ValueError(
                    "Choice setting %s must be one of %s"
                    % (key, sorted(allowed))
                )
            continue
        if (
            key not in FLAGS
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError("Invalid training setting")
        if (
            key.endswith("_lr")
            or key == "adapter_scale"
            or key in ("pair_loss_weight", "pair_loss_margin")
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    "Learning rate/scale must be finite and nonnegative"
                )
        elif key == "lr_endpoint":
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 < value <= 1.0
            ):
                raise ValueError("LR endpoint must be within (0, 1]")
        elif type(value) is not int or value < (
            0
            if key
            in (
                "seed",
                "limit_train_rows",
                "min_updates",
                "eval_every_updates",
                "eval_batch_size",
                "warmup",
                "hold_updates",
                "report_bootstrap",
            )
            else 1
        ):
            raise ValueError("Expected a positive integer training setting")
    if not values.get("max_updates"):
        raise ValueError(
            "Specify an optimizer-update budget for the experiment"
        )
    gpus = profile.get("gpus", [0, 1, 2])
    if len(gpus) == 1:
        (device,) = gpus
        if isinstance(device, bool) or type(device) is not int or device < 0:
            raise ValueError("Single-GPU profiles need one device index")
    elif gpus != [0, 1, 2]:
        raise ValueError("GPU profiles are one device or all three devices")
    trainer = (
        Path(profile["source"])
        / "siliconshield/training/laya/security_train.py"
    )
    output = Path(profile["runs"]) / experiment_id
    command = [
        profile["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=%d" % len(gpus),
        str(trainer),
        "--model",
        profile["model"],
        "--data",
        profile["data"],
        "--output",
        str(output),
        "--run-id",
        experiment_id,
        "--no-final-evaluation",
    ]
    if profile.get("reporting_contract"):
        command += ["--reporting-contract", profile["reporting_contract"]]
    for key in sorted(values):
        if key in BOOL_FLAGS:
            if values[key]:
                command += [BOOL_FLAGS[key]]
            continue
        if key in PATH_FLAGS:
            command += [PATH_FLAGS[key], values[key]]
            continue
        if key in CHOICE_FLAGS:
            command += [CHOICE_FLAGS[key][0], values[key]]
            continue
        command += [FLAGS[key], str(values[key])]
    return command, output


def laya_root_for(profile):
    """Explicit LAYA checkout for the job: profile ``laya_root`` wins, then $LAYA_ROOT.

    Validated here because a systemd user service does not inherit the
    submitting shell's environment; submit() forwards the result explicitly.
    """
    root = profile.get("laya_root") or os.environ.get("LAYA_ROOT")
    if root and not (Path(root) / "laya").is_dir():
        raise ValueError("LAYA root has no laya/ package: " + str(root))
    return root


def verify_laya(profile, laya_root):
    """Ensure the selected Python can import laya, with or without a checkout."""
    env = dict(os.environ)
    if laya_root:
        env["LAYA_ROOT"] = str(laya_root)
        env["PYTHONPATH"] = (
            str(laya_root) + os.pathsep + env.get("PYTHONPATH", "")
        )
    probe = subprocess.run(
        [profile["python"], "-c", "import laya; print(laya.__file__)"],
        capture_output=True,
        text=True,
        env=env,
    )
    if probe.returncode != 0:
        raise ValueError(
            'The job Python cannot import laya; set "laya_root" in the profile '
            "to a LAYA checkout or install a pinned laya in that environment"
        )
    return probe.stdout.strip()


def service_command(profile, experiment_id, command, laya_root=None):
    """Full systemd-run argv; carries CUDA_VISIBLE_DEVICES and LAYA_ROOT explicitly."""
    argv = [
        "systemd-run",
        "--user",
        "--unit=laya-glimmer-job-" + experiment_id,
        "--property=WorkingDirectory=" + profile["source"],
        "--setenv=CUDA_VISIBLE_DEVICES="
        + ",".join(map(str, profile.get("gpus", [0, 1, 2]))),
    ]
    if laya_root:
        argv.append("--setenv=LAYA_ROOT=" + str(laya_root))
    if profile.get("recipe"):
        argv.append(
            "--setenv=SILICONSHIELD_SECURITY_RECIPE=" + str(profile["recipe"])
        )
    return argv + ["--", *command]


class SecurityJobs:
    def __init__(self, profile_path):
        self.profile = read_json(profile_path)
        self.root = Path(self.profile["runs"])
        self.root.mkdir(parents=True, exist_ok=True)

    def submit(self, experiment_id, overrides=None, *, launch=False):
        if launch:
            from siliconshield.research_loop.execution.operator import Operator

            operator = Operator(None, jobs=self)
            manifest = operator.prepare(experiment_id, overrides)
            if manifest["phase"] == "prepare":
                manifest = operator.preflight(manifest)
            return operator.submit(manifest)
        command, output = command_for(
            self.profile, experiment_id, overrides or {}
        )
        trainer = (
            Path(self.profile["source"])
            / "siliconshield/training/laya/security_train.py"
        )
        inputs = [
            trainer,
            Path(self.profile["data"]) / "train.jsonl",
            Path(self.profile["data"]) / "dev.jsonl",
        ]
        for path in inputs:
            if not path.is_file():
                raise ValueError("Missing trainer or input: " + str(path))
        for key in PATH_FLAGS:
            configured = {
                **(self.profile.get("defaults", {})),
                **(overrides or {}),
            }.get(key)
            if configured is not None and not Path(configured).is_file():
                raise ValueError("Missing %s file: %s" % (key, configured))
        laya_root = laya_root_for(self.profile)
        bound = list(inputs)
        bound.extend(preservation_teacher_files(self.profile))
        for key in ("recipe", "reporting_contract"):
            if self.profile.get(key):
                bound.append(Path(self.profile[key]))
        configured_paths = {
            **(self.profile.get("defaults", {})),
            **(overrides or {}),
        }
        for key in PATH_FLAGS:
            if configured_paths.get(key) is not None:
                bound.append(Path(configured_paths[key]))
        receipt = {
            "command": command,
            "laya_root": laya_root,
            "service_command": service_command(
                self.profile, experiment_id, command, laya_root
            ),
            "source_and_data": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in bound
            },
        }
        identity = digest(receipt)
        if not launch:
            return {"phase": "prepared", "identity": identity, **receipt}

    def status(self, experiment_id):
        _, output = command_for(self.profile, experiment_id, {})
        return {
            name: read_json(output / name)
            for name in (
                "job.json",
                "execution.json",
                "status.json",
                "report.json",
                "training_report.json",
            )
            if (output / name).exists()
        }

    def launch_owned(self, manifest, manifest_path, resources, action):
        """Launch only the registered command, in a wrapper that owns its process group."""
        unit = manifest["unit"]
        subprocess.run(
            ["systemctl", "--user", "reset-failed", unit],
            capture_output=True,
            text=True,
        )
        worker = [
            self.profile["python"],
            "-m",
            "siliconshield.research_loop.execution.owned_job",
            "--manifest",
            str(manifest_path),
            "--resource-root",
            str(resources),
            "--action",
            action,
        ]
        argv = service_command(
            self.profile,
            manifest["experiment_id"],
            worker,
            manifest.get("laya_root"),
        )
        argv = [
            "--unit=" + unit if arg.startswith("--unit=") else arg
            for arg in argv
        ]
        argv[3:3] = [
            "--property=Description=siliconshield:"
            + manifest["idempotency_key"],
            "--property=KillMode=mixed",
            "--property=TimeoutStopSec=40",
        ]
        subprocess.run(
            argv, check=True, capture_output=True, text=True, timeout=30
        )
        result = {
            "phase": "submitted",
            "identity": manifest["idempotency_key"],
            "unit": unit,
            "action": action,
            "command": manifest["command"],
        }
        atomic_json(Path(manifest["output"]) / "job.json", result)
        return result

    def gpu_available(self):
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return not result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        help="Local JSON file with paths, GPU selection, and training settings",
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Unique job name using lowercase letters, digits, and hyphens",
    )
    parser.add_argument(
        "--overrides",
        default="{}",
        help="JSON object of training settings that override the profile",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Submit the job; without this flag, print the command and input hashes",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            SecurityJobs(args.profile).submit(
                args.experiment_id,
                json.loads(args.overrides),
                launch=args.launch,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
