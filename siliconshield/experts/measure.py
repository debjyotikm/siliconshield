"""Measure Glimmer specialists into capability profiles.

Runs benchmark rows through the served Glimmer endpoint behind each
expert's digest-checked skill, then writes one capability profile per
measured expert plus a measurement receipt. Profiles carry full
routing evidence bound to the live serving identity, so tiers rate
only while the same serving setup (weights, model name, access path,
token budget, contract) answers later assessments; anything else
stays honestly unrated.
"""

import argparse
import copy
import datetime
import json
import sys
from pathlib import Path
from typing import Any, cast

from siliconshield.benchmark.dataset import load_benchmark
from siliconshield.benchmark.runner import BenchmarkRunner
from siliconshield.experts.capabilities import (
    CapabilityProfile,
    validate_capability_profile,
)
from siliconshield.experts.registry import ModelEntry, load_registry
from siliconshield.experts.router import _area_tier
from siliconshield.research_loop.controller.client import ChatClient
from siliconshield.research_loop.procedures.glimmer_specialists import (
    GlimmerTransport,
)
from siliconshield.schemas import EXPERT_CAPABILITY_SCHEMA


def build_parser() -> argparse.ArgumentParser:
    """Build the measurement CLI parser."""
    parser = argparse.ArgumentParser(
        description="Measure Glimmer specialists into profiles.",
    )
    parser.add_argument("--registry", required=True, help="Registry path.")
    parser.add_argument(
        "--experts", default="", help="Comma-separated ids (default: all)."
    )
    parser.add_argument("--endpoint", required=True, help="Chat endpoint.")
    parser.add_argument("--ssh-host", default=None, help="SSH host.")
    parser.add_argument("--model", required=True, help="Served model.")
    parser.add_argument(
        "--glimmer-fingerprint", required=True, help="Serving digest."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Completion budget."
    )
    parser.add_argument("--timeout", type=int, default=300, help="Call cap.")
    parser.add_argument(
        "--dataset", required=True, help="hf:// source or local path."
    )
    parser.add_argument(
        "--areas", default="", help="Comma-separated areas (default: all)."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Score first N rows."
    )
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    return parser


def _device_type(ssh_host: str | None) -> str:
    if ssh_host and ssh_host.strip():
        return "ssh:%s" % ssh_host.strip()
    return "remote"


def _build_profile(
    expert_id: str,
    entry: ModelEntry,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    identity: dict[str, Any],
    qualification: str,
) -> CapabilityProfile:
    """Assemble one validated profile from a benchmark report."""
    overall = report["overall"]
    areas: dict[str, dict[str, int]] = {}
    for area, split in sorted(report["by_area"].items()):
        # Count violations from the executed slice, not the whole
        # dataset: a limited run must not claim violation rows it
        # never scored.
        tested_violations = sum(
            entry["total"]
            for entry in split["classes"]
            if entry["label"] == "violates_requirement"
        )
        areas[area] = {
            "correct": split["correct"],
            "total": split["rows"],
            "violations": tested_violations,
            "false_safe": split["false_safe"],
            "errors": split["errors"],
        }
    digest = entry["expected_files"]["model.safetensors"]
    profile: dict[str, Any] = {
        "schema": EXPERT_CAPABILITY_SCHEMA,
        "id": expert_id,
        "release": entry.get("release") or "glimmer-specialist-v1",
        "checkpoint_sha256": digest,
        "question_contract": entry.get("question_contract")
        or "glimmer-specialist-v1",
        "strengths": [],
        "weaknesses": [],
        "supported_mechanisms": [],
        "known_failures": [],
        "qualification_ref": qualification,
        "measurements": [
            {
                "eval": "glimmer-measure:%s:%d"
                % (
                    report["benchmark_sha256"][:12],
                    overall["rows"],
                ),
                "correct": overall["correct"],
                "total": overall["rows"],
            }
        ],
        "routing_evidence": {
            "execution_identity": copy.deepcopy(identity),
            "benchmark_sha256": report["benchmark_sha256"],
            "benchmark_source": report["benchmark_source"],
            "areas": areas,
        },
    }
    tiers = {
        area: _area_tier(cast(CapabilityProfile, profile), area)
        for area in areas
    }
    supported = sorted(
        area
        for area, tier in tiers.items()
        if tier in ("strong", "partial", "weak")
    )
    if not supported:
        raise ValueError(
            "Expert %r measured nothing ratable (tiers: %s); a profile "
            "with nothing demonstrated qualifies for nothing. Rerun "
            "with violation rows and clean executions." % (expert_id, tiers)
        )
    profile["supported_mechanisms"] = supported
    for area in supported:
        counts = areas[area]
        profile["strengths"].append(
            "%d/%d correct on %s (tier %s, %d false-safe)"
            % (
                counts["correct"],
                counts["total"],
                area,
                tiers[area],
                counts["false_safe"],
            )
        )
    for area in sorted(set(areas) - set(supported)):
        profile["weaknesses"].append(
            "%s unrated in this sample (tier %s)" % (area, tiers[area])
        )
    profile["weaknesses"].append(
        "Small-sample smoke run; self-reported confidence is uncalibrated."
    )
    failures = [
        "%s: violation judged %s" % (row_id, report["predictions"].get(row_id))
        for row_id in sorted(report["predictions"])
        if _row_label(rows, row_id) == "violates_requirement"
        and report["predictions"].get(row_id) != "violates_requirement"
    ]
    for item in report["row_errors"]:
        failures.append(
            "%s: execution error (%s)" % (item["row_id"], item["detail"][:120])
        )
    profile["known_failures"] = failures or [
        "No measured failures in this %d-row sample." % overall["rows"]
    ]
    return validate_capability_profile(profile)


def _row_label(rows: list[dict[str, Any]], row_id: str) -> str | None:
    for row in rows:
        if row["id"] == row_id:
            label = row["expected"]["security"]
            return label if isinstance(label, str) else None
    return None


def run_measure(
    args: argparse.Namespace,
    client_factory: Any = None,
) -> dict[str, Any]:
    """Measure experts; write profiles and a receipt; return summary."""
    try:
        data = load_benchmark(args.dataset)
        entries = {entry["id"]: entry for entry in load_registry(args.registry)}
    except ValueError as exc:
        raise ValueError("measure: %s" % exc) from exc
    wanted = [name.strip() for name in args.experts.split(",") if name.strip()]
    missing = [name for name in wanted if name not in entries]
    if missing:
        raise ValueError("measure: unknown experts: %s" % ",".join(missing))
    selected = (
        [entries[name] for name in wanted] if wanted else list(entries.values())
    )
    if not selected:
        raise ValueError("measure: registry has no models.")
    for entry in selected:
        if entry.get("kind", "local-worker") != "glimmer":
            raise ValueError(
                "measure: expert %r is not a Glimmer specialist; "
                "this path measures served Glimmer skills only."
                % (entry["id"],)
            )
        pinned = entry["expected_files"].get("model.safetensors")
        if pinned != args.glimmer_fingerprint:
            raise ValueError(
                "measure: expert %r pins weights %s but the run "
                "targets %s; measure the pinned serving or update "
                "the registry first."
                % (
                    entry["id"],
                    (pinned or "none")[:16],
                    args.glimmer_fingerprint[:16],
                )
            )
    out_dir = Path(args.out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            "measure: cannot create %s: %s" % (out_dir, exc)
        ) from exc
    areas = [name.strip() for name in args.areas.split(",") if name.strip()]
    runner = BenchmarkRunner(data)
    profiles: dict[str, Any] = {}
    for entry in selected:
        expert_id = entry["id"]
        if client_factory is None:
            client = ChatClient(
                args.endpoint,
                args.model,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                ssh_host=args.ssh_host,
            )
        else:
            client = client_factory()
        transport = GlimmerTransport(
            client,
            args.model,
            args.endpoint,
            args.glimmer_fingerprint,
            {expert_id: entry},
            max_tokens=args.max_tokens,
            device_type=_device_type(args.ssh_host),
        )
        captured: dict[str, Any] = {}

        def consult(state: str) -> Any:
            payload = transport(expert_id, state)
            captured.setdefault(
                "identity", copy.deepcopy(payload["execution_identity"])
            )
            return {
                "label": payload["label"],
                "probabilities": payload["probabilities"],
                "coverage": payload["coverage"],
            }

        def progress(done: int, total: int) -> None:
            print("%s: %d/%d" % (expert_id, done, total), file=sys.stderr)

        report = runner.run_model(
            expert_id,
            args.glimmer_fingerprint,
            consult,
            areas=areas or None,
            limit=args.limit,
            progress=progress,
        )
        if "identity" not in captured:
            raise ValueError(
                "measure: expert %r produced no successful consult; "
                "refusing to profile errors." % (expert_id,)
            )
        overall = report["overall"]
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d"
        )
        qualification = (
            "Measured %s %s: %d/%d over %s; model %s; benchmark "
            "%s@%s; %d errors."
            % (
                args.model,
                stamp,
                overall["correct"],
                overall["rows"],
                ",".join(sorted(report["by_area"])),
                args.model,
                report["benchmark_source"],
                report["benchmark_sha256"][:12],
                len(report["row_errors"]),
            )
        )
        profile = _build_profile(
            expert_id,
            entry,
            dict(report),
            data["rows"],
            captured["identity"],
            qualification,
        )
        profile_path = out_dir / ("%s.profile.json" % expert_id)
        # The validator strips the schema key on normalization; the
        # written file must carry it again or later loads refuse it.
        serialized = {"schema": EXPERT_CAPABILITY_SCHEMA, **profile}
        profile_path.write_text(
            json.dumps(serialized, indent=2, sort_keys=True)
        )
        report_path = out_dir / ("%s.measure.json" % expert_id)
        report_path.write_text(
            json.dumps(dict(report), indent=2, sort_keys=True)
        )
        profiles[expert_id] = {
            "profile": str(profile_path),
            "correct": overall["correct"],
            "total": overall["rows"],
        }
        print(
            "%s: wrote %s (%d/%d)"
            % (
                expert_id,
                profile_path.name,
                overall["correct"],
                overall["rows"],
            )
        )
    summary = {
        "experts": profiles,
        "benchmark_sha256": data["rows_sha256"],
        "benchmark_source": data["source"],
    }
    (out_dir / "receipt.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; return the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        run_measure(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print("%s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
