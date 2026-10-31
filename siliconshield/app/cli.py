"""CLI: run one authored demo case or a supplied RTL + requirement.

Examples:
  python -m siliconshield.app --list
  python -m siliconshield.app --case unauthorized-write --output /tmp/investigation
  python -m siliconshield.app --rtl design.sv --requirement "..." --output /tmp/investigation
"""

import argparse
import json
from pathlib import Path

from siliconshield.app.demo_cases import CASES
from siliconshield.app.workflow import investigate, report_assessment


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="List authored demo cases"
    )
    parser.add_argument(
        "--case", choices=sorted(CASES), help="Authored demo case name"
    )
    parser.add_argument(
        "--rtl",
        type=Path,
        help="SystemVerilog file (module topmodule) for a custom case",
    )
    parser.add_argument(
        "--requirement", help="Explicit security requirement for --rtl"
    )
    parser.add_argument(
        "--assumptions",
        default="Two-state sampled-clock RTL; only the stated "
        "requirement is assessed.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("investigation-output")
    )
    parser.add_argument(
        "--muse-endpoint", help="Base Muse chat endpoint for patch proposals"
    )
    parser.add_argument(
        "--muse-model", help="Model identity served at --muse-endpoint"
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.list:
        for name, case in sorted(CASES.items()):
            print("%-26s %s" % (name, case["title"]))
        return
    if args.case:
        case = CASES[args.case]
    elif args.rtl and args.requirement:
        # No demo testbench is inherited: without a check explicitly bound to
        # the supplied requirement and interface, it is reported unverified.
        case = {
            "name": "custom",
            "title": "Custom RTL investigation",
            "rtl": args.rtl.read_text(),
            "requirement": args.requirement,
            "assumptions": args.assumptions,
            "protected_signals": [],
            "control_signals": [],
            "checks": [],
            "patch": [],
        }
    else:
        raise SystemExit(
            "Provide --case, or --rtl with --requirement (see --list)"
        )
    proposer = None
    if args.muse_endpoint:
        if not args.muse_model:
            raise SystemExit("--muse-model is required with --muse-endpoint")
        from siliconshield.agents.muse_proposer import make_muse_proposer

        proposer = make_muse_proposer(args.muse_endpoint, args.muse_model)
    report = investigate(case, args.output, proposer=proposer)
    label = report_assessment(report).get("label")
    print(
        json.dumps(
            {
                "conclusion": report["conclusion"]["status"],
                "security_label": label,
                # Deprecated alias of the same label; stdout keeps it
                # until consumers migrate to security_label.
                "laya": label,
                "report": str(args.output / "report.json"),
                "summary": str(args.output / "report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
