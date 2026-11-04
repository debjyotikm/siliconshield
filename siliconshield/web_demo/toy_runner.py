"""Durable controller for the authored, bounded browser training fixture."""

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, cast

from siliconshield.research_loop.episodes.contracts import atomic_json
from siliconshield.research_loop.execution.glimmer_operator import run_operator
from siliconshield.research_loop.execution.operator import Operator
from siliconshield.research_loop.execution.strategy import run_campaign
from siliconshield.web_demo.glimmer_client import LocalGlimmerClient


class Cancelled(RuntimeError):
    """Stop the bounded strategy after settling its owned job."""


def max_calls_for_cycles(cycles: Any) -> int:
    """Return the Glimmer call budget for a cycle count.

    Raises ValueError for non-integer or out-of-range counts so
    a malformed campaign file fails before any worker spawns.
    """
    if isinstance(cycles, bool) or not isinstance(cycles, int):
        raise ValueError("Study max_cycles must be an integer")
    if not 1 <= cycles <= 50:
        raise ValueError("Study max_cycles must be 1 through 50")
    return 12 * int(cycles)


def execute(campaign_path: Path) -> None:
    """Reconcile the real operator and preserve its final status on disk."""
    if not isinstance(campaign_path, (str, Path)):
        raise ValueError("Toy campaign path must be a path")
    campaign_path = Path(campaign_path)
    folder = campaign_path.resolve().parent
    campaign = json.loads(campaign_path.read_text())
    if not isinstance(campaign, dict) or not isinstance(
        campaign.get("operator"), dict
    ):
        raise ValueError("Toy campaign needs an operator object")
    cfg = campaign["operator"]
    transport = {}
    for key in ("endpoint", "model", "ssh_host"):
        if key in cfg:
            if not isinstance(cfg[key], str):
                raise ValueError("Toy operator transport needs text " + key)
            transport[key] = cfg[key]

    def run(spec: dict[str, Any]) -> dict[str, Any]:
        if (folder / "cancel.request").exists():
            raise Cancelled("Stopped before the next phase.")
        operator = Operator(spec["profile"])

        def pause(seconds: float) -> None:
            if (folder / "cancel.request").exists():
                path = operator.manifest_path(spec["experiment_id"])
                if path.exists():
                    manifest = json.loads(path.read_text())
                    if manifest.get("phase") in [
                        "submit",
                        "observe",
                        "diagnose",
                        "intervene",
                    ]:
                        if manifest["phase"] == "submit":
                            manifest = operator.observe(manifest)
                        operator.intervene(manifest, "stop")
            time.sleep(min(seconds, 5))

        client = LocalGlimmerClient(
            **transport,
            max_tokens=2048,
            timeout=180,
            max_calls=max_calls_for_cycles(spec.get("max_cycles", 6)),
            audit_dir=folder / "glimmer-calls",
        )
        result = run_operator(
            spec, client=client, operator=operator, sleep=pause
        )
        if (folder / "cancel.request").exists() and result["manifest"].get(
            "terminal"
        ):
            raise Cancelled(
                "Owned job settled after stop request. No next phase launched."
            )
        return cast(dict[str, Any], result)

    try:
        result = run_campaign(campaign, run=run)
        atomic_json(
            folder / "controller-result.json",
            {"status": "complete", "result": result, "finished": time.time()},
        )
    except Cancelled as exc:
        atomic_json(
            folder / "controller-result.json",
            {"status": "stopped", "reason": str(exc), "finished": time.time()},
        )
    except Exception as exc:
        # A boundary failure is durable evidence, never a successful study result.
        atomic_json(
            folder / "controller-result.json",
            {"status": "error", "error": str(exc), "finished": time.time()},
        )
        traceback.print_exc()
        raise


def main() -> None:
    """Run only the campaign file submitted by the browser backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    execute(args.campaign)


if __name__ == "__main__":
    main()
