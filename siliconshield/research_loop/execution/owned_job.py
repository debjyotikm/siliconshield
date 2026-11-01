"""Hold GPU ownership for a single registered subprocess and write its terminal receipt."""

import argparse
import fcntl
import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path

from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    read_json,
)


def run(manifest_path, resource_root, action):
    manifest = read_json(manifest_path)
    output = Path(manifest["output"])
    command = list(manifest["command"])
    if action in ("resume", "finalize_retry"):
        command.append("--resume")
    if action == "finalize_retry":
        command.append("--finalize-development-only")
    resources = Path(resource_root)
    resources.mkdir(parents=True, exist_ok=True)
    with (resources / ".runtime.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = manifest["idempotency_key"]
        started = time.time()
        receipt = {
            "identity": identity,
            "unit": manifest["unit"],
            "action": action,
            "started": started,
            "phase": "running",
            "command": command,
        }
        atomic_json(output / "execution.json", receipt)
        try:
            if manifest.get("study"):
                from siliconshield.research_loop.execution import study_contract

                bound = study_contract.verify_job(
                    manifest["study_profile"],
                    manifest["experiment_id"],
                    manifest["overrides"],
                )
                if bound != manifest["study"]:
                    raise ValueError("Owned worker study identity mismatch")
            for name, expected in manifest.get("input_hashes", {}).items():
                if (
                    hashlib.sha256(Path(name).read_bytes()).hexdigest()
                    != expected
                ):
                    raise ValueError(
                        "Prepared input changed before execution: " + name
                    )
        except (OSError, ValueError) as exc:
            receipt.update(
                phase="failed",
                exit_code=1,
                failure=str(exc),
                finished=time.time(),
            )
            atomic_json(output / "execution.json", receipt)
            return 1
        log = output / ("execution-%d.log" % time.time_ns())
        with log.open("wb") as stream:
            env = dict(
                os.environ,
                SILICONSHIELD_EXECUTION_MANIFEST=str(
                    Path(manifest_path).resolve()
                ),
            )
            try:
                child = subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
            except OSError as exc:
                receipt.update(
                    phase="failed",
                    exit_code=1,
                    failure=str(exc),
                    finished=time.time(),
                )
                atomic_json(output / "execution.json", receipt)
                return 1

            def terminate(signum, frame):
                if child.poll() is None:
                    os.killpg(child.pid, signum)

            old = {
                sig: signal.signal(sig, terminate)
                for sig in (signal.SIGTERM, signal.SIGINT)
            }
            failure = None
            try:
                while child.poll() is None:
                    if time.time() - started > manifest["timeout_seconds"]:
                        failure = "timeout"
                    elif log.stat().st_size > manifest["output_quota_bytes"]:
                        failure = "output_quota"
                    if failure:
                        terminate(signal.SIGTERM, None)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            terminate(signal.SIGKILL, None)
                        break
                    time.sleep(0.5)
                code = child.wait()
            finally:
                for sig, previous in old.items():
                    signal.signal(sig, previous)
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        receipt.update(
            phase="completed" if code == 0 and not failure else "failed",
            exit_code=code,
            failure=failure,
            finished=time.time(),
            log=str(log),
        )
        atomic_json(output / "execution.json", receipt)
        return 0 if receipt["phase"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--resource-root", required=True)
    parser.add_argument(
        "--action", choices=("train", "resume", "finalize_retry"), required=True
    )
    args = parser.parse_args()
    raise SystemExit(run(args.manifest, args.resource_root, args.action))


if __name__ == "__main__":
    main()
