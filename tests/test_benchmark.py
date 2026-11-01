"""Synthetic tests for the area benchmark (no network, no GPU)."""

import json

import pytest

from siliconshield.benchmark import metrics
from siliconshield.benchmark.dataset import load_benchmark, validate_rows
from siliconshield.benchmark.runner import BenchmarkRunner

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


def test_validate_rows_accepts_and_rejects(tmp_path):
    assert validate_rows(_rows())
    bad = _rows()
    del bad[0]["state"]
    with pytest.raises(ValueError):
        validate_rows(bad)
    dup = _rows()
    dup[1]["id"] = "a0"
    with pytest.raises(ValueError):
        validate_rows(dup)
    broken = _rows()[:5]
    with pytest.raises(ValueError):
        validate_rows(broken)
    path = tmp_path / "bench.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in _rows()))
    data = load_benchmark(str(path))
    assert len(data["rows"]) == 6
    assert data["source"].startswith("local:")
    with pytest.raises(ValueError):
        load_benchmark("hf://:bad")


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


def test_failed_members_fail_triplets_and_keep_support():
    rows = _rows()
    labels = {row["id"]: row["expected"]["security"] for row in rows}
    probs = {
        row_id: {
            name: (0.9 if name == labels[row_id] else 0.05) for name in LABELS
        }
        for row_id in labels
    }
    one_failed, _, _ = metrics.score_predictions(
        rows, labels, probs, {"a2": "worker failed"}
    )
    assert one_failed["correct"] == 5
    assert one_failed["errors"] == 1
    assert one_failed["triplets_correct"] == 1
    assert one_failed["triplets_total"] == 2
    assert one_failed["triplets_failed"] == 1
    assert one_failed["triplets_incomplete"] == 0
    ctx = [
        entry for entry in one_failed["classes"] if entry["label"] == LABELS[2]
    ][0]
    assert ctx == {"label": LABELS[2], "correct": 1, "total": 2}
    all_failed, _, _ = metrics.score_predictions(
        rows, {}, {}, {row["id"]: "down" for row in rows}
    )
    assert all_failed["correct"] == 0
    assert all_failed["triplets_correct"] == 0
    assert all_failed["triplets_total"] == 2
    assert all_failed["triplets_failed"] == 2


def test_partial_triplet_counts_incomplete():
    rows = _rows()[:4]
    labels = {row["id"]: row["expected"]["security"] for row in rows}
    probs = {
        row_id: {
            name: (0.9 if name == labels[row_id] else 0.05) for name in LABELS
        }
        for row_id in labels
    }
    overall, _, _ = metrics.score_predictions(rows, labels, probs)
    assert overall["triplets_total"] == 2
    assert overall["triplets_correct"] == 1
    assert overall["triplets_incomplete"] == 1


def test_compare_requires_same_benchmark():
    base = {
        "model_id": "m1",
        "checkpoint_sha256": "ab" * 32,
        "benchmark_source": "local:x",
        "benchmark_sha256": "cd" * 32,
        "overall": {"correct": 1, "rows": 2, "false_safe": 0},
        "by_area": {"fifo": {"correct": 1, "rows": 2, "false_safe": 0}},
        "by_mechanism": {},
        "row_errors": [],
    }
    other = dict(base, model_id="m2")
    compared = metrics.compare([base, other])
    assert compared["by_area"]["fifo"]["m1"]["correct"] == 1
    with pytest.raises(ValueError):
        metrics.compare([base, dict(other, benchmark_sha256="ee" * 32)])
    with pytest.raises(ValueError):
        metrics.compare([base, base])


def test_runner_collects_errors_and_reports():
    data = {
        "rows": _rows(),
        "source": "local:synthetic",
        "rows_sha256": "ff" * 32,
    }
    runner = BenchmarkRunner(data)
    assert runner.areas == ["counter", "fifo"]

    def consult(state):
        if state.endswith("a1"):
            raise RuntimeError("worker down")
        if state.endswith("b0"):
            return {"label": "bogus"}
        return _payload(LABELS[0])

    seen = []
    report = runner.run_model(
        "synthetic-model",
        "ab" * 32,
        consult,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert report["overall"]["rows"] == 6
    assert report["overall"]["errors"] == 2
    assert len(report["row_errors"]) == 2
    assert seen[-1] == (6, 6)
    with pytest.raises(ValueError):
        runner.run_model("m", "ab" * 32, consult, areas=["nope"])
    counter_only = runner.run_model("m", "ab" * 32, consult, areas=["counter"])
    assert counter_only["overall"]["rows"] == 3
    limited = runner.run_model("m", "ab" * 32, consult, limit=4)
    assert limited["overall"]["rows"] == 3
    assert limited["overall"]["triplets_total"] == 1
    assert limited["overall"]["triplets_incomplete"] == 0


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


def test_sample_triplets_is_seeded_and_complete():
    from siliconshield.benchmark.router_eval import sample_triplets

    first = sample_triplets(_eval_rows(), 1, 7)
    again = sample_triplets(_eval_rows(), 1, 7)
    assert [row["id"] for row in first] == [row["id"] for row in again]
    assert len(first) == 6
    everything = sample_triplets(_eval_rows(), 0, 7)
    assert len(everything) == 12
    with pytest.raises(ValueError):
        sample_triplets(_eval_rows(), 3, 7)


def test_evaluate_routing_scores_and_arbitrates():
    from siliconshield.benchmark.router_eval import evaluate_routing

    data = {
        "rows": _eval_rows(),
        "source": "local:synthetic",
        "rows_sha256": "ff" * 32,
    }

    class FakeService:
        def assess(self, scope, state, transport):
            assert scope[0] in ("fifo", "counter")
            return {
                "routing": {
                    "experts": ["expert-a"],
                    "tiers": {"expert-a": "strong"},
                    "thin_evidence": False,
                },
                "responses": [
                    {
                        "expert_id": "expert-a",
                        "label": transport("expert-a", state)["label"],
                    }
                ],
                "errors": [],
            }

        def profiles_for(self, expert_ids):
            return {expert_id: {} for expert_id in expert_ids}

    def transport(expert_id, state):
        return _payload(LABELS[0])

    def arbitrate(assessment, state, requirement, profiles):
        assert requirement
        assert assessment["responses"]
        return {"conclusion": "satisfies_requirement"}

    report = evaluate_routing(
        data,
        FakeService(),
        transport,
        per_area=1,
        seed=7,
        arbitrate_per_area=1,
        arbitrate=arbitrate,
    )
    assert set(report["by_area"]) == {"fifo", "counter"}
    assert report["by_area"]["fifo"]["rows"] == 3
    assert report["by_area"]["fifo"]["expert_correct"]["expert-a"] == {
        "correct": 1,
        "total": 3,
    }
    assert len(report["arbitration"]) == 6


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
