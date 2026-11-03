"""Synthetic tests for the area benchmark (no network, no GPU)."""

import json

import pytest

from siliconshield.benchmark import metrics

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
