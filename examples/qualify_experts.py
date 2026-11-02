"""Measure registry models and write identity-bound routing profiles.

This example runs real inference. It never trains, assigns reliability tiers,
or changes an existing registry. Run --help before choosing a device/population.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from siliconshield.benchmark.dataset import load_benchmark
from siliconshield.benchmark.runner import BenchmarkRunner
from siliconshield.experts.capabilities import validate_capability_profile
from siliconshield.experts.registry import load_registry
from siliconshield.experts.worker import WorkerPool
from siliconshield.schemas import EXPERT_CAPABILITY_SCHEMA


def measured_profile(
    entry: dict[str, Any],
    ready: dict[str, Any],
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    """Build a profile from one worker's observed identity and measurements."""
    if report["row_errors"]:
        raise ValueError("Resolve inference errors before publishing a profile")
    identity = ready["execution_identity"]
    if identity["checkpoint_sha256"] != report["checkpoint_sha256"]:
        raise ValueError("Measurement and worker checkpoint differ")
    areas = {}
    for name, counts in report["by_area"].items():
        violations = next(
            row["total"]
            for row in counts["classes"]
            if row["label"] == "violates_requirement"
        )
        areas[name] = {
            "correct": counts["correct"],
            "total": counts["rows"],
            "violations": violations,
            "false_safe": counts["false_safe"],
            "errors": counts["errors"],
        }
    profile = {
        "schema": EXPERT_CAPABILITY_SCHEMA,
        "id": entry["id"],
        "release": entry.get("release", entry["id"]),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "question_contract": identity["question_contract"],
        "strengths": [
            f"Measured {name}: {c['correct']}/{c['total']} correct."
            for name, c in areas.items()
        ],
        "weaknesses": [
            "Development measurements do not establish independent qualification.",
            "Coverage is limited to the areas and execution identity recorded here.",
        ],
        "supported_mechanisms": list(areas),
        "known_failures": [
            f"{name}: {c['false_safe']}/{c['violations']} false-safe violations."
            for name, c in areas.items()
        ],
        "qualification_ref": str(report_path),
        "measurements": [
            {"eval": name, "correct": c["correct"], "total": c["total"]}
            for name, c in areas.items()
        ],
        "routing_evidence": {
            "benchmark_source": report["benchmark_source"],
            "benchmark_sha256": report["benchmark_sha256"],
            "execution_identity": identity,
            "areas": areas,
        },
    }
    validate_capability_profile(profile)
    return profile


def write_json(path: Path, value: Any) -> None:
    """Write a new example artifact without replacing earlier evidence."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> None:
    """Score models sequentially and publish a separate measured registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--areas", required=True, help="Comma-separated areas")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = load_registry(args.registry)
    data = load_benchmark(args.dataset)
    runner = BenchmarkRunner(data)
    areas = args.areas.split(",")
    if not set(areas) <= set(runner.areas):
        raise ValueError("Requested area is absent from the population")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    measured = []
    for index, entry in enumerate(entries):
        # Use numeric filenames rather than interpreting registry IDs as paths.
        prefix = f"expert-{index:02d}"
        report_path = output / f"{prefix}.measurement.json"
        with WorkerPool() as pool:
            pool.load(entry)
            ready = pool.ready(entry["id"])
            write_json(output / f"{prefix}.ready.json", ready)
            report = runner.run_model(
                entry["id"],
                entry["expected_files"]["model.safetensors"],
                lambda state: pool.analyze(entry["id"], state),
                areas=areas,
            )
        write_json(report_path, report)
        profile_path = output / f"{prefix}.profile.json"
        profile = measured_profile(entry, ready, report, report_path)
        write_json(profile_path, profile)
        measured.append({**entry, "profile": str(profile_path)})
    original = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    result_path = output / "models.local.json"
    write_json(result_path, {**original, "models": measured})
    load_registry(result_path)
    print(result_path)


if __name__ == "__main__":
    main()
