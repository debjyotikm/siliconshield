"""Route direct assessments to Muse API specialists behind skills.

This is the same specialist flow as ``glimmer_specialists`` (pinned
skill per registry expert, digest-checked on every call, one parsed
verdict with self-reported confidence), with one change: the
transport calls a Muse API model through the local ``muse`` CLI
instead of the served Glimmer endpoint.

Provider-managed models have no locally verifiable weights, so this
path deliberately stays outside the profile-bound result protocol:
responses carry the checked skill digest and an explicit
provider-managed identity, never a checkpoint hash or borrowed
capability claims from another engine's measurements. Scope routing
and Spark arbitration both need profile-bound evidence, so this CLI
consults one named expert per call and refuses ``--scope`` and
``--arbitrate`` with a pointer to the Glimmer path.
"""

import argparse
import http.client
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from siliconshield.experts.identity import text_digest
from siliconshield.experts.service import InferenceService
from siliconshield.research_loop.procedures.glimmer_specialists import (
    GlimmerTransport,
    _load_state,
)
from siliconshield.research_loop.procedures.muse_client import MuseClient

WEIGHT_IDENTITY = "provider-managed; not locally verifiable"


def build_parser() -> argparse.ArgumentParser:
    """Build the Muse specialists CLI parser."""
    parser = argparse.ArgumentParser(
        description="Consult Muse API specialists behind skills.",
    )
    parser.add_argument("--registry", required=True, help="Registry path.")
    parser.add_argument(
        "--expert", required=True, help="Consult exactly this expert id."
    )
    parser.add_argument("--state-file", default="", help="File with state.")
    parser.add_argument(
        "--design", action="append", default=[], help="RTL file (repeat)."
    )
    parser.add_argument("--requirement", default="", help="Requirement.")
    parser.add_argument("--assumptions", default="", help="Assumptions.")
    parser.add_argument(
        "--model",
        default="muse-spark-1.3",
        help="Muse model id (needs account access).",
    )
    parser.add_argument(
        "--effort", default="low", help="Muse reasoning effort."
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Model call timeout."
    )
    parser.add_argument("--output", required=True, help="Receipt dir.")
    parser.add_argument(
        "--brief", action="store_true", help="Print label, scores, time."
    )
    return parser


def format_brief(report: dict[str, Any]) -> str:
    """Render one specialist verdict without local absolute paths."""
    response = report["responses"][0]
    lines = [
        "%s via %s: %s"
        % (response["expert_id"], report["muse_model"], response["label"])
    ]
    for name in (
        "satisfies_requirement",
        "violates_requirement",
        "insufficient_context",
    ):
        lines.append("  %s: %.4f" % (name, response["scores"][name]))
    lines.append("inference_seconds: %.2f" % report["inference_seconds"])
    lines.append("weights: " + WEIGHT_IDENTITY)
    return "\n".join(lines)


def run_muse_specialists(
    args: argparse.Namespace,
    client_factory: Any = None,
) -> dict[str, Any]:
    """Consult one expert through a Muse API model; return the report."""
    state = _load_state(args)
    expert = args.expert.strip()
    if not expert:
        raise ValueError("Consult exactly one --expert id.")
    if not args.model.strip():
        raise ValueError("A Muse model id is required.")
    service = InferenceService(args.registry)
    entries = {
        model_id: service.model_entry(model_id)
        for model_id in service.expert_ids
    }
    if expert not in entries:
        raise ValueError(
            "Unknown expert %r; registered: %s."
            % (expert, ", ".join(sorted(entries)) or "none")
        )
    if entries[expert].get("kind", "local-worker") != "glimmer":
        raise ValueError(
            "Refusing to route %r through Muse: its registry "
            "entry is not a skill-backed specialist." % (expert,)
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if client_factory is None:
        executable = shutil.which("muse")
        if executable is None:
            raise ValueError("`muse` CLI not found on PATH.")
        client = MuseClient(
            executable=Path(executable),
            audit_dir=output / "audit",
            model=args.model.strip(),
            effort=args.effort,
            timeout=args.timeout,
        )
    else:
        client = client_factory()
    # The shared transport parses the verdict and scores. Its
    # serving fingerprint is internal-only here: provider-managed
    # weights have no digest, so the reported envelope below
    # carries the checked skill and an explicit provider-managed
    # identity instead of any checkpoint hash.
    transport = GlimmerTransport(
        client,
        args.model.strip(),
        "muse-cli",
        "0" * 64,
        entries,
        device_type="muse-cli",
    )
    started = time.monotonic()
    try:
        payload = transport(expert, state)
    finally:
        seconds = time.monotonic() - started
    # The transport raised unless the live skill matched the pinned
    # digest, so the pinned digest below names the checked skill.
    response = {
        "expert_id": expert,
        "muse_model": args.model.strip(),
        "skill_sha256": entries[expert]["expected_files"]["SKILL.md"],
        "skill_checked": True,
        "weight_identity": WEIGHT_IDENTITY,
        "device_type": "muse-cli",
        "label": payload["label"],
        "scores": payload["probabilities"],
        "state_sha256": text_digest(state),
    }
    report = {
        "expert": expert,
        "muse_model": args.model.strip(),
        "inference_seconds": seconds,
        "responses": [response],
        "errors": [],
    }
    (output / "receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; return the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        report = run_muse_specialists(args)
    except (
        ValueError,
        RuntimeError,
        OSError,
        KeyError,
        IndexError,
        TypeError,
        subprocess.SubprocessError,
        http.client.HTTPException,
    ) as exc:
        print("muse-specialists: %s" % exc, file=sys.stderr)
        return 2
    if args.brief:
        print(format_brief(report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
