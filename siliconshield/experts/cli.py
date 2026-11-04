"""Command line for multi-expert assessment (Qt-free, no Torch).

``python -m siliconshield.experts assess`` routes one scope through
the registry, consults every qualified expert over persistent
subprocess workers, and prints the assessment as JSON. Optional
Muse arbitration and finding assembly stay in this CLI: the GUI
demo never owns router work.
"""

import argparse
import json
import sys
from subprocess import TimeoutExpired
from typing import Any

from siliconshield.experts import arbitration, findings
from siliconshield.experts.service import InferenceService
from siliconshield.experts.worker import WorkerPool


def build_parser() -> argparse.ArgumentParser:
    """Build the experts CLI parser (also used for --help checks)."""
    parser = argparse.ArgumentParser(
        prog="siliconshield.experts",
        description="Route and consult RTL security experts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    assess = sub.add_parser("assess", help="Assess one state text.")
    assess.add_argument("--registry", required=True, help="Registry path.")
    assess.add_argument(
        "--scope",
        required=True,
        help="Comma-separated mechanism tags for routing.",
    )
    assess.add_argument(
        "--state-file", required=True, help="File with state text."
    )
    assess.add_argument(
        "--arbitrate",
        action="store_true",
        help="Arbitrate expert evidence with the muse CLI.",
    )
    assess.add_argument("--requirement", default="", help="Requirement.")
    assess.add_argument("--issue", default="", help="Suspected issue.")
    assess.add_argument("--relevance", default="", help="Issue relevance.")
    assess.add_argument(
        "--rtl", action="append", default=[], help="RTL location (repeat)."
    )
    return parser


def run_assess(
    args: argparse.Namespace,
    pool_factory: Any = WorkerPool,
) -> dict[str, Any]:
    """Run one assessment; return the JSON-serializable report."""
    try:
        with open(args.state_file, encoding="utf-8") as handle:
            state = handle.read()
    except OSError as exc:
        raise ValueError("Cannot read state file: %s" % exc)
    scope = [tag.strip() for tag in args.scope.split(",") if tag.strip()]
    if not scope:
        raise ValueError("Scope must name at least one mechanism tag.")
    # Usage errors must fail before the consult, never after it.
    if args.arbitrate and not args.requirement.strip():
        raise ValueError("--arbitrate needs --requirement.")
    if bool(args.issue.strip()) != bool(args.relevance.strip()):
        raise ValueError("--issue and --relevance are required together.")
    if (args.issue.strip() or args.relevance.strip()) and not args.arbitrate:
        raise ValueError("--issue and --relevance need --arbitrate.")
    service = InferenceService(args.registry)
    report: dict[str, Any] = {
        "scope": scope,
        "broken_experts": service.broken_experts,
        "unprofiled_experts": service.unprofiled_experts,
    }
    with pool_factory() as pool:
        loaded: set[str] = set()

        def transport(expert_id: str, judged_state: str) -> Any:
            if expert_id not in loaded:
                pool.load(service.model_entry(expert_id))
                loaded.add(expert_id)
            return pool.transport()(expert_id, judged_state)

        assessment = service.assess(scope, state, transport)
    report["assessment"] = assessment
    if args.arbitrate:
        try:
            verdict = arbitration.arbitrate_assessment(
                dict(assessment),
                state,
                args.requirement,
                service.profiles_for(assessment["routing"]["experts"]),
            )
        except (
            ValueError,
            RuntimeError,
            TimeoutError,
            TimeoutExpired,
            OSError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            report["arbitration_error"] = str(exc)
            return report
        report["arbitration"] = verdict
        if args.issue.strip() and args.relevance.strip():
            report["finding"] = findings.build_finding(
                args.issue,
                args.relevance,
                args.requirement,
                list(args.rtl),
                dict(assessment),
                verdict,
            )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; return the process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "assess":
        try:
            report = run_assess(args)
        except (
            ValueError,
            RuntimeError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            print("experts: %s" % exc, file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
