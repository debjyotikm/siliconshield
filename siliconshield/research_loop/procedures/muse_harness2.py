"""Learn procedure guidance with Muse and Harness2, outside the GPU campaign."""

import argparse
import fcntl
import json
import time
from pathlib import Path
from typing import Any

from siliconshield.research_loop.episodes.contracts import atomic_json, digest
from siliconshield.research_loop.procedures.campaign_learning import (
    load_request,
)
from siliconshield.research_loop.procedures.muse_client import MuseClient


def process_queue(
    queue: Path,
    checks: list[dict[str, Any]],
    client: Any,
    output: Path,
    bundle: Path,
) -> list[dict[str, Any]]:
    """Process each queued observation once; failures preserve prior guidance.

    Use a fresh output directory to retry failed trials. This prevents a bad
    input or endpoint outage from consuming requests in an endless retry loop.
    """
    for name, value in (
        ("queue", queue),
        ("output", output),
        ("bundle", bundle),
    ):
        if not isinstance(value, (str, Path)):
            raise ValueError("Procedure queue needs a %s path" % name)
    queue, output, bundle = Path(queue), Path(output), Path(bundle)
    if not isinstance(checks, list) or any(
        not isinstance(item, dict) for item in checks
    ):
        raise ValueError("Procedure queue needs a list of check rows")
    from siliconshield.research_loop.procedures.muse_trials import learn

    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(queue.glob("*.json")):
        identifier = path.stem
        receipt = output / "processed" / (identifier + ".json")
        if receipt.exists():
            continue
        # Never reuse interrupted trial directories or mixed trajectories.
        root = output / "attempts" / identifier
        attempt = 1
        while (root / str(attempt)).exists():
            attempt += 1
        try:
            report = learn(
                [load_request(path)],
                checks,
                client,
                root / str(attempt),
                bundle,
            )
            record: dict[str, Any] = {
                "request": identifier,
                "status": "complete",
                "report": report,
            }
        except Exception as exc:
            # A rejected sidecar trial cannot interrupt a training campaign.
            record = {
                "request": identifier,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        atomic_json(receipt, record)
        records.append(record)
    return records


def main() -> None:
    """Run a signed replay population or watch a supervisor's private queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--episodes",
        type=Path,
        help="Signed train episodes used for procedure proposals",
    )
    source.add_argument(
        "--queue",
        type=Path,
        help="Completed development observations queued by the supervisor",
    )
    parser.add_argument(
        "--checks",
        type=Path,
        required=True,
        help="Signed dev episodes for independent qualification (never final test)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New private sidecar output directory, outside Git and campaign source",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Private published guidance file",
    )
    parser.add_argument("--muse-bin", type=Path, required=True)
    parser.add_argument("--model", default="muse-spark-1.3-contributor")
    parser.add_argument(
        "--effort", default="high", choices=("low", "medium", "high", "xhigh")
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument(
        "--once",
        action="store_true",
        help="With --queue, process pending requests once and exit",
    )
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be positive")

    # Optional Harness2 is imported only after argument parsing, so --help
    # works from a wheel installed without the research extras.
    from siliconshield.research_loop.adapters.glimmer_training import load_rows
    from siliconshield.research_loop.procedures.muse_trials import (
        implementation_identity,
        learn,
    )

    checks = load_rows(
        args.checks, allowed_splits=("dev",), require_signed=True
    )
    rows = (
        load_rows(args.episodes, allowed_splits=("train",), require_signed=True)
        if args.episodes
        else None
    )
    args.output.mkdir(parents=True, exist_ok=True)
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    with (
        (args.output / "worker.lock").open("a") as lock,
        args.bundle.with_suffix(args.bundle.suffix + ".lock").open(
            "a"
        ) as publication_lock,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(publication_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client = MuseClient(
            args.muse_bin,
            args.output / "calls",
            args.model,
            args.effort,
            args.timeout,
        )
        identity: dict[str, Any] = {
            "client": client.execution_identity,
            "checks": digest(checks),
            "code": implementation_identity(),
        }
        identity_file = args.output / "identity.json"
        if (
            identity_file.exists()
            and json.loads(identity_file.read_text()) != identity
        ):
            raise ValueError(
                "Sidecar model, code, or qualification population changed; use a fresh output"
            )
        atomic_json(identity_file, identity)
        if rows is not None:
            report = learn(
                rows, checks, client, args.output / "trial", args.bundle
            )
            print(json.dumps(report, indent=2))
        else:
            while True:
                # Recheck split hashes/signatures before each proposal batch.
                current = load_rows(
                    args.checks, allowed_splits=("dev",), require_signed=True
                )
                if (
                    digest(current) != identity["checks"]
                    or implementation_identity() != identity["code"]
                ):
                    raise ValueError(
                        "Sidecar inputs changed; use a frozen source and fresh output"
                    )
                records = process_queue(
                    args.queue, current, client, args.output, args.bundle
                )
                print(json.dumps({"requests": records}), flush=True)
                if args.once:
                    break
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
