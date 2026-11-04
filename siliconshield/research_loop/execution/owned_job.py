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


def root_cause_from_text(text):
    """Return the most informative traceback summary in text, if any.

    Launchers wrap the real error (torch elastic prints its own
    summary after the child traceback), so callers get the
    exception line itself: the first non-indented line after a
    traceback marker. Traceback frames are always indented.
    Summaries without a message lose to earlier ones with one.
    """
    marker = "Traceback (most recent call last):"
    blocks = text.split(marker)
    if len(blocks) < 2:
        return _cli_error(text)
    summaries = []
    for block in blocks[1:]:
        for line in block.splitlines():
            if line and not line[0].isspace():
                summaries.append(line.strip()[:300])
                break
    for summary in reversed(summaries):
        _, colon, tail = summary.partition(":")
        if not colon or tail.strip():
            return summary
    if summaries:
        return _cli_error(text) or summaries[-1]
    return None


def _cli_error(text):
    """Return the last command-line error line, if any.

    Usage errors (argparse rejects a flag combination) exit
    without a traceback. The `prog.py: error: ...` line is the
    actionable summary.
    """
    matches = [
        line.strip()[:300]
        for line in text.splitlines()
        if ": error: " in line and not line[0].isspace()
    ]
    return matches[-1] if matches else None


def root_cause(log):
    """Return the most informative traceback summary from a log file."""
    try:
        text = Path(log).read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None
    return root_cause_from_text(text)


def run(manifest_path, resource_root, action):
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Execution manifest must be an object")
    command = manifest.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) for part in command)
    ):
        raise ValueError("Execution manifest needs a command list")
    command = list(command)
    for key in ("output", "idempotency_key", "unit"):
        if not manifest.get(key) or not isinstance(manifest[key], str):
            raise ValueError("Execution manifest needs " + key)
    for key in ("timeout_seconds", "output_quota_bytes"):
        value = manifest.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError("Execution manifest needs " + key)
    hashes = manifest.get("input_hashes", {})
    if not isinstance(hashes, dict) or any(
        not isinstance(name, str) or not isinstance(expected, str)
        for name, expected in hashes.items()
    ):
        raise ValueError("Execution manifest input hashes are malformed")
    output = Path(manifest["output"])
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
                    try:
                        os.killpg(child.pid, signum)
                    except (ProcessLookupError, PermissionError):
                        pass

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
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    child.wait()
        if failure is None and code != 0:
            failure = root_cause(log) or "exit_code=%d" % code
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
