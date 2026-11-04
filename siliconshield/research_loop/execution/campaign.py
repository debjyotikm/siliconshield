"""Wait for Glimmer training, then run controller evaluation and Harness2 trials."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    read_json,
)


def follow(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".glimmer-campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "A Glimmer campaign watcher already owns this workspace"
            ) from exc
        return _follow(args)


def _inspect_record(stdout):
    """Parse one docker-inspect record or refuse the container state."""
    try:
        record = json.loads(stdout)[0]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Cannot read the evaluation container record"
        ) from exc
    if not isinstance(record, dict):
        raise RuntimeError("Cannot read the evaluation container record")
    for key in ("Config", "State"):
        if not isinstance(record.get(key), dict):
            raise RuntimeError("Cannot read the evaluation container record")
    labels = record["Config"].get("Labels", {})
    if not isinstance(labels, dict):
        raise RuntimeError("Cannot read the evaluation container record")
    return record


def _follow(args):
    root = Path(args.root).resolve()
    status = root / "campaign-status.json"

    def event(phase, **details):
        atomic_json(status, {"phase": phase, "time": time.time(), **details})

    event("waiting_for_training", container=args.training_container)
    finished = subprocess.check_output(
        ["docker", "wait", args.training_container], text=True
    ).strip()
    if finished != "0":
        raise RuntimeError("Training container exited with code " + finished)
    run = root / "runs" / args.run
    state = read_json(run / "status.json")
    if not isinstance(state, dict):
        raise RuntimeError("Training status is unreadable")
    if state.get("phase") != "training_complete":
        raise RuntimeError(
            "Training exited without a complete training result: "
            + str(state.get("phase"))
        )
    event("training_complete", training=state)
    container = args.evaluation_container
    existing = subprocess.run(
        ["docker", "inspect", container], capture_output=True, text=True
    )
    if existing.returncode == 0:
        record = _inspect_record(existing.stdout)
        if (
            record["Config"].get("Labels", {}).get("laya.glimmer.run")
            != args.run
        ):
            raise RuntimeError(
                "Evaluation container name is owned by another run"
            )
        if (
            not record["State"].get("Running")
            and record["State"].get("ExitCode") != 0
        ):
            raise RuntimeError(
                "Existing evaluation failed; inspect its saved evidence before retrying"
            )
    else:
        image = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", args.image],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not image.startswith("sha256:"):
            raise RuntimeError("Cannot resolve the evaluation image identity")
        (root / "runtime").mkdir(exist_ok=True)
        key = getattr(args, "dataset_key", None) or os.environ.get(
            "SILICONSHIELD_DATASET_KEY"
        )
        key_args = []
        manifest = run.parent.parent / "data" / args.run / "manifest.json"
        if manifest.exists():
            record = read_json(manifest)
            if not isinstance(record, dict):
                raise ValueError("Dataset manifest is unreadable")
            if record.get("signature") and not key:
                raise ValueError(
                    "Authenticated dataset requires --dataset-key for evaluation"
                )
        if key:
            try:
                key = Path(key).resolve(strict=True)
            except OSError as exc:
                raise ValueError(
                    "Dataset key file is missing: " + str(key)
                ) from exc
            key_args = [
                "-v",
                f"{key}:/workspace/dataset-key:ro",
                "-e",
                "SILICONSHIELD_DATASET_KEY=/workspace/dataset-key",
            ]
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            container,
            "--label",
            "laya.glimmer.run=" + args.run,
            "--gpus",
            "all",
            "--network",
            "none",
            "--shm-size=8g",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "PYTHONPATH=/workspace/source:/workspace/dependencies/harness2",
            "-e",
            "HF_HOME=/workspace/cache",
            "-e",
            "HF_HUB_DISABLE_PROGRESS_BARS=1",
            "-e",
            "TOKENIZERS_PARALLELISM=false",
            "-e",
            "OMP_NUM_THREADS=8",
            "-e",
            "SILICONSHIELD_GPU_LOCK=/workspace/runtime/glimmer-gpu.lock",
            "-e",
            "SILICONSHIELD_IMAGE_DIGEST=" + image,
            *key_args,
            "-v",
            f"{root}:/workspace",
            "-v",
            f"{root}/source:/workspace/source:ro",
            "-v",
            f"{root}/models:/workspace/models:ro",
            "--entrypoint",
            "python3",
            image,
            "-u",
            "-m",
            "siliconshield.research_loop.controller.evaluate",
            "--model",
            "/workspace/models/muse-glimmer-30b",
            "--adapter",
            f"/workspace/runs/{args.run}/best/adapter",
            "--episodes",
            f"/workspace/data/{args.run}/dev.jsonl",
            "--output",
            f"/workspace/runs/{args.run}/generation-replay",
            "--examples",
            str(args.examples),
            "--harness2-output",
            f"/workspace/runs/{args.run}/harness2-replay",
        ]
        subprocess.run(command, check=True)
    event("evaluating_controller", container=container)
    finished = subprocess.check_output(
        ["docker", "wait", container], text=True
    ).strip()
    if finished != "0":
        raise RuntimeError("Evaluation container exited with code " + finished)
    report = read_json(run / "generation-replay/report.json")
    harness = read_json(run / "harness2-replay/report.json")
    event(
        "complete",
        controller_comparison=report,
        harness2=harness,
        automatic_laya_training_started=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Workspace containing source, models, data, and runs directories",
    )
    parser.add_argument(
        "--run", required=True, help="Run directory name under runs/ and data/"
    )
    parser.add_argument(
        "--training-container",
        required=True,
        help="Existing training container to wait for",
    )
    parser.add_argument(
        "--evaluation-container",
        required=True,
        help="Name for the evaluation container",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Container image with the Glimmer evaluation dependencies",
    )
    parser.add_argument(
        "--dataset-key", help="Local key for the authenticated episode manifest"
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Maximum development episodes per model",
    )
    args = parser.parse_args()
    try:
        follow(args)
    except Exception as exc:
        # Terminal receipt: record any failure before re-raising so an
        # interrupted campaign is distinguishable from a missing one.
        atomic_json(
            Path(args.root) / "campaign-status.json",
            {
                "phase": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "time": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()
