"""Synthetic tests for the area benchmark (no network, no GPU)."""

import json
from subprocess import TimeoutExpired
from types import SimpleNamespace

import pytest

from siliconshield.benchmark import metrics
from siliconshield.benchmark.dataset import validate_rows
from siliconshield.benchmark.router_eval import (
    evaluate_routing,
    sample_triplets,
)

LABELS = [
    "satisfies_requirement",
    "violates_requirement",
    "insufficient_context",
]


def _row(row_id, area, label, pair):
    return {
        "id": row_id,
        "state": "synthetic state %s" % row_id,
        "questions": {"security": {"type": "synthetic"}},
        "expected": {"security": label},
        "benchmark_area": area,
        "benchmark_source": "synthetic",
        "pair_id": pair,
        "mechanism_id": "synthetic-mech",
    }


def _rows():
    return [
        _row("a0", "fifo", LABELS[0], "trip-a"),
        _row("a1", "fifo", LABELS[1], "trip-a"),
        _row("a2", "fifo", LABELS[2], "trip-a"),
        _row("b0", "counter", LABELS[0], "trip-b"),
        _row("b1", "counter", LABELS[1], "trip-b"),
        _row("b2", "counter", LABELS[2], "trip-b"),
    ]


def _payload(label):
    return {
        "label": label,
        "probabilities": {
            name: (0.7 if name == label else 0.15) for name in LABELS
        },
        "coverage": {
            "truncated": False,
            "state_tokens_dropped": 0,
            "truncated_questions": [],
        },
    }


def _eval_rows():
    rows = []
    for area in ("fifo", "counter"):
        for trip in range(2):
            for pos, label in enumerate(LABELS):
                rows.append(
                    _row(
                        "%s-%d-%d" % (area, trip, pos),
                        area,
                        label,
                        "%s-trip-%d" % (area, trip),
                    )
                )
    return rows


def test_cli_run_end_to_end(tmp_path):
    from siliconshield.benchmark.cli import build_parser, run_benchmark
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    bench = tmp_path / "bench.jsonl"
    bench.write_text("\n".join(json.dumps(row) for row in _rows()))
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    laya = tmp_path / "laya"
    (laya / "laya").mkdir(parents=True)
    (laya / "laya" / "__init__.py").write_text("")
    registry = tmp_path / "models.json"
    registry.write_text(
        json.dumps(
            {
                "schema": GUI_MODEL_REGISTRY_SCHEMA,
                "models": [
                    {
                        "id": "expert-a",
                        "display_name": "Expert A",
                        "checkpoint": str(tmp_path),
                        "python": str(tmp_path / "python"),
                        "laya_root": str(laya),
                        "expected_files": {"model": "cd" * 32},
                        "device": "cpu",
                    }
                ],
            }
        )
    )

    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def __contains__(self, expert_id):
            return True

        def load(self, entry):
            assert entry["id"] == "expert-a"

        def analyze(self, expert_id, state):
            assert expert_id == "expert-a"
            return _payload(LABELS[2])

    args = build_parser().parse_args(
        [
            "run",
            "--dataset",
            str(bench),
            "--registry",
            str(registry),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert run_benchmark(args, pool_factory=FakePool) == 0
    report = json.loads((tmp_path / "out" / "expert-a.json").read_text())
    assert report["overall"]["correct"] == 2
    compared = json.loads((tmp_path / "out" / "comparison.json").read_text())
    assert compared["models"][0]["model_id"] == "expert-a"


def test_metrics_counts_and_triplets():
    rows = _rows()
    labels = {
        "a0": LABELS[0],
        "a1": LABELS[0],
        "a2": LABELS[2],
        "b0": LABELS[1],
        "b1": LABELS[1],
        "b2": LABELS[2],
    }
    probs = {
        row_id: {
            name: (0.7 if name == labels[row_id] else 0.15) for name in LABELS
        }
        for row_id in labels
    }
    overall, areas, mechs = metrics.score_predictions(rows, labels, probs)
    assert overall["correct"] == 4
    assert overall["false_safe"] == 1
    assert overall["false_violation"] == 1
    assert overall["triplets_correct"] == 0
    assert overall["triplets_total"] == 2
    assert areas["fifo"]["triplets_total"] == 1
    assert mechs["synthetic-mech"]["rows"] == 6
    assert overall["nll"] > 0.0
    with pytest.raises(ValueError):
        metrics.score_predictions(rows, {"a0": "nope"}, probs)


def _eval_service():
    assessment = {
        "routing": {
            "experts": ["expert-a"],
            "tiers": {"expert-a": "strong"},
            "thin_evidence": False,
        },
        "responses": [{"expert_id": "expert-a", "label": LABELS[0]}],
        "errors": [],
    }
    return SimpleNamespace(
        expert_ids=["expert-a"],
        route=lambda scope: assessment["routing"],
        assess=lambda scope, state, consult: dict(assessment),
        profiles_for=lambda ids: {name: {"id": name} for name in ids},
    )


def test_arbitration_timeout_and_oserror_stay_per_row():
    service = _eval_service()
    failures = [TimeoutExpired("muse", 5), OSError("synthetic pipe error")]
    state = {"count": 0}

    def arbitrate(assessment, state_text, requirement, profiles):
        failure = failures[state["count"] % len(failures)]
        state["count"] += 1
        raise failure

    rows = _eval_rows()
    for row in rows:
        row["state"] = (
            "REQUIREMENT\nsynthetic requirement\nRTL\nassign q = d;\n"
        )
    data = {"rows": rows, "source": "synthetic", "rows_sha256": "ab" * 32}
    report = evaluate_routing(
        data,
        service,
        lambda expert_id, text: _payload(LABELS[0]),
        0,
        7,
        arbitrate_per_area=99,
        arbitrate=arbitrate,
    )
    assert len(report["arbitration"]) == len(rows)
    assert all(
        entry["verdict"]["conclusion"] == "unresolved"
        for entry in report["arbitration"]
    )
    assert all(
        "arbitration_error" in entry["verdict"]
        for entry in report["arbitration"]
    )
    details = [
        entry["verdict"]["arbitration_error"] for entry in report["arbitration"]
    ]
    assert any("TimeoutExpired" in detail for detail in details)
    assert any(detail.startswith("OSError") for detail in details)


def test_split_triplets_rejected_at_load_and_sample():
    rows = _rows()
    rows[2]["benchmark_area"] = "counter"
    with pytest.raises(ValueError, match="span benchmark areas"):
        validate_rows(rows)
    with pytest.raises(ValueError, match="not a triplet"):
        sample_triplets(rows, 0, 7)


def _router_eval_inputs(tmp_path):
    from siliconshield.schemas import (
        EXPERT_CAPABILITY_SCHEMA,
        GUI_MODEL_REGISTRY_SCHEMA,
    )

    bench = tmp_path / "bench.jsonl"
    bench.write_text("\n".join(json.dumps(row) for row in _eval_rows()))
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    laya = tmp_path / "laya"
    (laya / "laya").mkdir(parents=True)
    (laya / "laya" / "__init__.py").write_text("")
    profile = {
        "schema": EXPERT_CAPABILITY_SCHEMA,
        "id": "expert-a",
        "release": "synthetic-1",
        "question_contract": "siliconshield-requirement-context-v1",
        "checkpoint_sha256": "ab" * 32,
        "qualification_ref": "synthetic run",
        "supported_mechanisms": ["fifo", "counter"],
        "strengths": ["synthetic"],
        "weaknesses": ["synthetic"],
        "known_failures": ["synthetic"],
        "measurements": [{"eval": "synthetic", "correct": 1, "total": 2}],
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    registry = tmp_path / "models.json"
    registry.write_text(
        json.dumps(
            {
                "schema": GUI_MODEL_REGISTRY_SCHEMA,
                "models": [
                    {
                        "id": "expert-a",
                        "display_name": "Expert A",
                        "checkpoint": str(tmp_path),
                        "python": str(python),
                        "laya_root": str(laya),
                        "expected_files": {"model.safetensors": "ab" * 32},
                        "device": "cpu",
                        "profile": str(profile_path),
                        "release": "synthetic-1",
                        "question_contract": (
                            "siliconshield-requirement-context-v1"
                        ),
                        "qualification": "synthetic run",
                    }
                ],
            }
        )
    )
    return bench, registry


def test_router_eval_reloads_after_worker_discard(tmp_path):
    from siliconshield.benchmark.cli import build_parser, run_router_eval

    bench, registry = _router_eval_inputs(tmp_path)

    class FlakyPool:
        loads = 0

        def __init__(self):
            self.ready = set()
            self.failed_once = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def __contains__(self, expert_id):
            return expert_id in self.ready

        def load(self, entry):
            FlakyPool.loads += 1
            self.ready.add(entry["id"])

        def analyze(self, expert_id, state):
            if expert_id not in self.ready:
                raise RuntimeError("No validated worker ready")
            if not self.failed_once:
                self.failed_once = True
                self.ready.discard(expert_id)
                raise RuntimeError("synthetic transient")
            return _payload(LABELS[0])

    FlakyPool.loads = 0
    args = build_parser().parse_args(
        [
            "router-eval",
            "--dataset",
            str(bench),
            "--registry",
            str(registry),
            "--per-area",
            "1",
            "--out",
            str(tmp_path / "report.json"),
        ]
    )
    assert run_router_eval(args, pool_factory=FlakyPool) == 0
    assert FlakyPool.loads == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["by_area"]["fifo"]["rows"] == 3
    assert report["by_area"]["counter"]["rows"] == 3


def test_cli_maps_unreadable_dataset_to_exit_2(tmp_path, capsys):
    from siliconshield.benchmark.cli import build_parser, run_areas

    for missing in ("", str(tmp_path / "missing.jsonl")):
        args = build_parser().parse_args(["areas", "--dataset", missing])
        assert run_areas(args) == 2
        assert "benchmark:" in capsys.readouterr().err
    args = build_parser().parse_args(["areas", "--dataset", str(tmp_path)])
    assert run_areas(args) == 2
    assert "benchmark:" in capsys.readouterr().err


def test_non_object_rows_rejected_at_load(tmp_path, capsys):
    from siliconshield.benchmark.cli import build_parser, run_areas

    bench = tmp_path / "bench.jsonl"
    bench.write_text("123\n")
    args = build_parser().parse_args(["areas", "--dataset", str(bench)])
    assert run_areas(args) == 2
    assert "not an object" in capsys.readouterr().err
    bench.write_text("not json\n")
    assert run_areas(args) == 2
    assert "invalid JSON" in capsys.readouterr().err


def test_blank_lines_skipped_before_validation(tmp_path):
    from siliconshield.benchmark.dataset import load_benchmark

    bench = tmp_path / "bench.jsonl"
    bench.write_text("\n".join(json.dumps(row) for row in _rows()) + "\n\n")
    assert len(load_benchmark(str(bench))["rows"]) == 6


def test_json_array_file_loads_same_rows(tmp_path):
    from siliconshield.benchmark.dataset import load_benchmark

    bench = tmp_path / "bench.example.json"
    bench.write_text(json.dumps(_rows(), indent=2))
    assert len(load_benchmark(str(bench))["rows"]) == 6
    bad = tmp_path / "bad.example.json"
    bad.write_text(json.dumps([{"id": "x"}]))
    with pytest.raises(ValueError, match="missing keys"):
        load_benchmark(str(bad))


def test_cli_maps_unwritable_report_to_exit_2(tmp_path, capsys):
    from siliconshield.benchmark.cli import build_parser, run_benchmark
    from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

    bench = tmp_path / "bench.jsonl"
    bench.write_text("\n".join(json.dumps(row) for row in _rows()))
    out = tmp_path / "out"
    out.mkdir()
    (out / "expert-a.json").mkdir()
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    laya = tmp_path / "laya"
    (laya / "laya").mkdir(parents=True)
    (laya / "laya" / "__init__.py").write_text("")
    registry = tmp_path / "models.json"
    registry.write_text(
        json.dumps(
            {
                "schema": GUI_MODEL_REGISTRY_SCHEMA,
                "models": [
                    {
                        "id": "expert-a",
                        "display_name": "Expert A",
                        "checkpoint": str(tmp_path),
                        "python": str(python),
                        "laya_root": str(laya),
                        "expected_files": {"model": "cd" * 32},
                        "device": "cpu",
                    }
                ],
            }
        )
    )

    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def __contains__(self, expert_id):
            return True

        def load(self, entry):
            return None

        def analyze(self, expert_id, state):
            return _payload(LABELS[2])

    args = build_parser().parse_args(
        [
            "run",
            "--dataset",
            str(bench),
            "--registry",
            str(registry),
            "--out-dir",
            str(out),
        ]
    )
    assert run_benchmark(args, pool_factory=FakePool) == 2
    assert "cannot write" in capsys.readouterr().err


def test_non_string_mechanism_rejected_at_load(tmp_path, capsys):
    from siliconshield.benchmark.cli import build_parser, run_areas

    rows = _rows()
    rows[0]["mechanism_id"] = ["fifo"]
    bench = tmp_path / "bench.jsonl"
    bench.write_text("\n".join(json.dumps(row) for row in rows))
    args = build_parser().parse_args(["areas", "--dataset", str(bench)])
    assert run_areas(args) == 2
    assert "mechanism id" in capsys.readouterr().err


def test_router_eval_retries_transient_load_failure(tmp_path):
    from siliconshield.benchmark.cli import build_parser, run_router_eval

    bench, registry = _router_eval_inputs(tmp_path)

    class OnceFailingPool:
        loads = 0

        def __init__(self):
            self.ready = set()
            self.failed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def __contains__(self, expert_id):
            return expert_id in self.ready

        def load(self, entry):
            OnceFailingPool.loads += 1
            if not self.failed:
                self.failed = True
                raise RuntimeError("synthetic transient load")
            self.ready.add(entry["id"])

        def analyze(self, expert_id, state):
            return _payload(LABELS[0])

    OnceFailingPool.loads = 0
    args = build_parser().parse_args(
        [
            "router-eval",
            "--dataset",
            str(bench),
            "--registry",
            str(registry),
            "--per-area",
            "1",
            "--out",
            str(tmp_path / "report.json"),
        ]
    )
    assert run_router_eval(args, pool_factory=OnceFailingPool) == 0
    # One failed attempt plus one successful retry: a cached failure
    # would load only once and poison every row instead.
    assert OnceFailingPool.loads == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["by_area"]["fifo"]["rows"] == 3
    assert report["by_area"]["counter"]["rows"] == 3


def _runner_data(rows):
    return {"rows": rows, "source": "synthetic", "rows_sha256": "ab" * 32}


def test_non_object_consult_becomes_a_row_error():
    from siliconshield.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(_runner_data(_rows()))
    report = runner.run_model("synthetic", "ab" * 32, lambda state: None)
    assert len(report["row_errors"]) == 6
    assert report["overall"]["rows"] == 6


def test_benchmark_area_is_type_checked():
    rows = _rows()
    rows[0]["benchmark_area"] = None
    with pytest.raises(ValueError, match="benchmark area"):
        validate_rows(rows)
    rows = _rows()
    rows[0]["benchmark_area"] = ["fifo"]
    with pytest.raises(ValueError, match="benchmark area"):
        validate_rows(rows)


def test_short_limit_scores_a_partial_group():
    from siliconshield.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(_runner_data(_rows()))
    report = runner.run_model(
        "synthetic",
        "ab" * 32,
        lambda state: _payload("satisfies_requirement"),
        limit=2,
    )
    assert report["overall"]["rows"] == 2
    assert report["overall"]["triplets_incomplete"] == 1


def test_all_error_split_reports_null_nll():
    import json

    rows = _rows()
    labels = {row["id"]: LABELS[0] for row in rows}
    probs = {row["id"]: {name: 0.34 for name in LABELS} for row in rows}
    errors = {row["id"]: "consult: synthetic" for row in rows}
    overall, areas, _ = metrics.score_predictions(rows, labels, probs, errors)
    assert overall["nll"] is None
    assert areas["fifo"]["nll"] is None
    json.dumps({"overall": overall, "areas": areas}, allow_nan=False)


def test_empty_selection_is_a_clean_error():
    from siliconshield.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(_runner_data(_rows()))
    with pytest.raises(ValueError, match="selection is empty"):
        runner.run_model(
            "synthetic",
            "ab" * 32,
            lambda state: _payload("satisfies_requirement"),
            limit=0,
        )


def test_negative_limit_is_rejected():
    from siliconshield.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(_runner_data(_rows()))
    with pytest.raises(ValueError, match="cannot be negative"):
        runner.run_model(
            "synthetic",
            "ab" * 32,
            lambda state: _payload("satisfies_requirement"),
            limit=-1,
        )


def test_per_mechanism_triplets_are_counted():
    rows = _rows()
    labels = {row["id"]: row["expected"]["security"] for row in rows}
    probabilities = {
        row["id"]: {
            name: (0.9 if name == row["expected"]["security"] else 0.05)
            for name in LABELS
        }
        for row in rows
    }
    _, _, by_mech = metrics.score_predictions(rows, labels, probabilities)
    assert by_mech["synthetic-mech"]["triplets_total"] == 2
    assert by_mech["synthetic-mech"]["triplets_correct"] == 2


def test_runner_rejects_malformed_arguments():
    from siliconshield.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(_runner_data(_rows()))
    with pytest.raises(ValueError, match="must be callable"):
        runner.run_model("m", "c", 5)
    with pytest.raises(ValueError, match="must be callable"):
        runner.run_model("m", "c", lambda state: None, progress=5)
    with pytest.raises(ValueError, match="must be an integer"):
        runner.run_model("m", "c", lambda state: None, limit="x")
    with pytest.raises(ValueError, match="must be an integer"):
        runner.run_model("m", "c", lambda state: None, limit=True)


def test_sampler_rejects_malformed_arguments():
    with pytest.raises(ValueError, match="list of row objects"):
        sample_triplets(5, 0, 7)
    with pytest.raises(ValueError, match="per_area must be an integer"):
        sample_triplets(_rows(), "x", 7)
    with pytest.raises(ValueError, match="must be seedable"):
        sample_triplets(_rows(), 0, [7])


def test_router_eval_rejects_malformed_arguments():
    from siliconshield.benchmark.router_eval import evaluate_routing

    data = _runner_data(_rows())
    with pytest.raises(ValueError, match="arbitrate must be callable"):
        evaluate_routing(data, None, None, 0, 1, arbitrate=5)
    with pytest.raises(ValueError, match="progress must be callable"):
        evaluate_routing(data, None, None, 0, 1, progress=5)
    with pytest.raises(ValueError, match="arbitrate count"):
        evaluate_routing(data, None, None, 0, 1, arbitrate_per_area="x")


def test_compare_rejects_malformed_reports():
    with pytest.raises(ValueError, match="at least one report"):
        metrics.compare(5)
    with pytest.raises(ValueError, match="report objects"):
        metrics.compare([5])
    with pytest.raises(ValueError, match="missing 'overall'"):
        metrics.compare(
            [{"benchmark_sha256": "d", "model_id": "m", "by_area": {}}]
        )
