"""Command line for the RTL benchmark (Qt-free, no Torch).

``python -m siliconshield.benchmark run`` scores registry models on
the pinned public benchmark through persistent worker subprocesses
and writes one report per model plus a head-to-head comparison.
Any registry entry is benchmarkable: future checkpoints need no
code changes, only `profile`/`release` fields for routing.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from siliconshield.benchmark import metrics
from siliconshield.benchmark.dataset import default_source, load_benchmark
from siliconshield.benchmark.runner import BenchmarkRunner
from siliconshield.experts.registry import ModelEntry, load_registry
from siliconshield.experts.service import InferenceService
from siliconshield.experts.worker import WorkerPool


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark parser (also used for --help checks)."""
    parser = argparse.ArgumentParser(
        prog="siliconshield.benchmark",
        description="Score decision models on the RTL benchmark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    areas = sub.add_parser("areas", help="List benchmark areas.")
    areas.add_argument(
        "--dataset",
        default=default_source(),
        help="hf:// source or local benchmark path.",
    )
    run = sub.add_parser("run", help="Benchmark registry models.")
    run.add_argument(
        "--dataset",
        default=default_source(),
        help="hf:// source or local benchmark path.",
    )
    run.add_argument("--registry", required=True, help="Registry path.")
    run.add_argument(
        "--models",
        default="",
        help="Comma-separated model ids (default: every entry).",
    )
    run.add_argument(
        "--areas", default="", help="Comma-separated areas (default: all)."
    )
    run.add_argument(
        "--limit", type=int, default=None, help="Score only the first N rows."
    )
    run.add_argument("--out-dir", required=True, help="Report directory.")
    router_eval = sub.add_parser(
        "router-eval", help="Evaluate routing on sampled examples."
    )
    router_eval.add_argument(
        "--dataset",
        default=default_source(),
        help="hf:// source or local benchmark path.",
    )
    router_eval.add_argument("--registry", required=True, help="Registry path.")
    router_eval.add_argument(
        "--per-area",
        type=int,
        default=5,
        help="Triplets sampled per area (0 takes every triplet).",
    )
    router_eval.add_argument("--seed", type=int, default=7, help="Sample seed.")
    router_eval.add_argument(
        "--arbitrate-per-area",
        type=int,
        default=0,
        help="Triplets per area to arbitrate via Muse (slow).",
    )
    router_eval.add_argument("--out", required=True, help="Report path.")
    return parser


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def run_areas(args: argparse.Namespace) -> int:
    """List benchmark areas; return the exit code."""
    try:
        data = load_benchmark(args.dataset)
    except ValueError as exc:
        print("benchmark: %s" % exc, file=sys.stderr)
        return 2
    runner = BenchmarkRunner(data)
    print("source: %s" % data["source"])
    print("rows: %d (%s)" % (len(data["rows"]), data["rows_sha256"][:16]))
    for area in runner.areas:
        print(area)
    return 0


def run_benchmark(
    args: argparse.Namespace,
    pool_factory: Any = WorkerPool,
) -> int:
    """Benchmark the selected models; return the exit code."""
    try:
        data = load_benchmark(args.dataset)
        entries = {entry["id"]: entry for entry in load_registry(args.registry)}
    except ValueError as exc:
        print("benchmark: %s" % exc, file=sys.stderr)
        return 2
    wanted = [name.strip() for name in args.models.split(",") if name.strip()]
    selected: list[ModelEntry] = [
        entries[name] for name in wanted if name in entries
    ]
    missing = [name for name in wanted if name not in entries]
    if missing:
        print(
            "benchmark: unknown models: %s" % ",".join(missing), file=sys.stderr
        )
        return 2
    if not wanted:
        selected = list(entries.values())
    if not selected:
        print("benchmark: registry has no models.", file=sys.stderr)
        return 2
    areas = [name.strip() for name in args.areas.split(",") if name.strip()]
    out_dir = Path(args.out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            "benchmark: cannot create %s: %s" % (out_dir, exc), file=sys.stderr
        )
        return 2
    runner = BenchmarkRunner(data)
    reports: list[metrics.ModelReport] = []
    for entry in selected:
        model_id = entry["id"]
        digest = entry["expected_files"].get("model.safetensors", "")
        with pool_factory() as pool:
            try:
                pool.load(entry)
            except RuntimeError as exc:
                print("benchmark: %s: %s" % (model_id, exc), file=sys.stderr)
                return 2

            def consult(
                state: str,
                _pool: Any = pool,
                _id: str = model_id,
                _entry: Any = entry,
            ) -> Any:
                if _id not in _pool:
                    _pool.load(_entry)
                return _pool.analyze(_id, state)

            def progress(done: int, total: int) -> None:
                print("%s: %d/%d" % (model_id, done, total), file=sys.stderr)

            try:
                report = runner.run_model(
                    model_id,
                    digest,
                    consult,
                    areas=areas or None,
                    limit=args.limit,
                    progress=progress,
                )
            except ValueError as exc:
                print("benchmark: %s" % exc, file=sys.stderr)
                return 2
        try:
            _write(out_dir / ("%s.json" % model_id), dict(report))
        except OSError as exc:
            print(
                "benchmark: cannot write %s: %s" % (out_dir, exc),
                file=sys.stderr,
            )
            return 2
        reports.append(report)
        overall = report["overall"]
        print(
            "%s: %d/%d fs=%d trip=%d/%d errors=%d"
            % (
                model_id,
                overall["correct"],
                overall["rows"],
                overall["false_safe"],
                overall["triplets_correct"],
                overall["triplets_total"],
                len(report["row_errors"]),
            )
        )
    try:
        _write(out_dir / "comparison.json", metrics.compare(reports))
    except OSError as exc:
        print(
            "benchmark: cannot write %s: %s" % (out_dir, exc), file=sys.stderr
        )
        return 2
    return 0


def run_router_eval(
    args: argparse.Namespace, pool_factory: Any = WorkerPool
) -> int:
    """Sample examples, route, consult, and score; return exit code."""
    from siliconshield.benchmark.router_eval import evaluate_routing
    from siliconshield.experts.arbitration import arbitrate_assessment

    try:
        data = load_benchmark(args.dataset)
        service = InferenceService(args.registry)
    except ValueError as exc:
        print("benchmark: %s" % exc, file=sys.stderr)
        return 2
    if not service.expert_ids:
        print("benchmark: registry has no profiled experts.", file=sys.stderr)
        return 2
    with pool_factory() as pool:
        loaded: set[str] = set()

        def consult(expert_id: str, state: str) -> Any:
            # Load failures are never cached: a transient first-row
            # failure must not poison every later row.
            if expert_id not in loaded or expert_id not in pool:
                pool.load(service.model_entry(expert_id))
                loaded.add(expert_id)
            return pool.analyze(expert_id, state)

        def progress(done: int, total: int) -> None:
            if done == 1 or done == total or done % 100 == 0:
                print("router-eval: %d/%d" % (done, total), file=sys.stderr)

        try:
            report = evaluate_routing(
                data,
                service,
                consult,
                per_area=args.per_area,
                seed=args.seed,
                arbitrate_per_area=args.arbitrate_per_area,
                arbitrate=arbitrate_assessment
                if args.arbitrate_per_area > 0
                else None,
                progress=progress,
            )
        except ValueError as exc:
            print("benchmark: %s" % exc, file=sys.stderr)
            return 2
    try:
        Path(args.out).write_text(
            json.dumps(report, indent=1, sort_keys=True) + "\n"
        )
    except OSError as exc:
        print(
            "benchmark: cannot write %s: %s" % (args.out, exc), file=sys.stderr
        )
        return 2
    for area in sorted(report["by_area"]):
        entry = report["by_area"][area]
        print(
            "%s: rows=%d thin=%d agree=%d %s"
            % (
                area,
                entry["rows"],
                entry["thin_rows"],
                entry["agree_rows"],
                json.dumps(entry["expert_correct"], sort_keys=True),
            )
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; return the process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "areas":
        return run_areas(args)
    if args.command == "run":
        return run_benchmark(args)
    if args.command == "router-eval":
        return run_router_eval(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
