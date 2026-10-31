"""Synthetic path: real security-trainer report schema -> episode -> replay -> grade. No GPU."""

import copy
import json

import pytest

from siliconshield.research_loop.controller.replay import execute, grade
from siliconshield.research_loop.episodes.dataset import build
from siliconshield.research_loop.episodes.laya_results import adapt_report


def trainer_report(i):
    return {
        "schema": 2,
        "run_id": f"synthetic-security-{i}",
        "test_evaluated": False,
        "final_evaluation": False,
        "calibration_fitted": False,
        "development": {
            "n": 100,
            "accuracy": 0.80 + i / 1000,
            "nll": 0.60,
            "macro_f1": 0.7,
            "false_safe_count": 1,
            "false_safe_rate": 0.02,
            "worst_family_false_safe_rate": 0.05,
            "min_class_recall": 0.6,
            "selection_cost": 0.4,
            "both_pair_members_correct": 0.75,
            "complete_pairs": 40,
            "degeneracy_count": 0,
            "class_support": [40, 40, 20],
            "confusion": [[38, 1, 1], [1, 38, 1], [2, 2, 16]],
        },
        "initial_development": {"n": 100, "accuracy": 0.70, "nll": 0.80},
        "selection_key": [0, 0, 0, 0.4, 0.6],
        "development_acceptance": {
            "accepted": True,
            "reasons": [],
            "thresholds": {},
        },
        "progress": {"step": 120, "stop_reason": "complete"},
        "stop_reason": "complete",
        "selected_checkpoint": {
            "generation": "epoch-000001",
            "epoch": 1,
            "step": 120,
            "weights_sha256": "a" * 64,
        },
        "signature": {
            "data": {"train": f"train-hash-{i}", "dev": f"dev-hash-{i}"},
            "initial_checkpoint": {"model.safetensors": "b" * 64},
            "code_manifest": {"combined_sha256": "c" * 64},
        },
    }


def test_adapter_preserves_security_metrics_and_provenance():
    summary, provenance, _ = adapt_report(
        trainer_report(0), source_sha256="d" * 64
    )
    assert summary["initial"] == {
        "decisions": 100,
        "agreement": 0.70,
        "nll": 0.80,
    }
    assert summary["selected"]["agreement"] == pytest.approx(0.80)
    assert summary["security"]["false_safe_rate"] == 0.02
    assert summary["selected_checkpoint"]["weights_sha256"] == "a" * 64
    assert provenance["adapter"] == "siliconshield-laya-development-result-v1"
    assert provenance["trainer_schema"] == 2


def test_adapter_rejects_test_evaluated_or_missing_initial():
    report = trainer_report(0)
    report["test_evaluated"] = True
    with pytest.raises(ValueError, match="final/test"):
        adapt_report(report)
    report = trainer_report(0)
    report["initial_development"] = None
    with pytest.raises(ValueError, match="initial"):
        adapt_report(report)


class TraceClient:
    """Replays the authored trace stored in the episode itself; never calls a service."""

    model = "fixture-laya-v1"
    endpoint = "http://fixture.invalid"

    def __init__(self, row):
        self.row = row
        self.calls = 0

    def complete(self, messages, tools=()):
        self.calls += 1
        return copy.deepcopy(self.row["messages"][len(messages)])


def test_trainer_report_through_episode_to_verified_replay(tmp_path):
    runs = tmp_path / "runs"
    for i in range(5):
        directory = runs / f"experiment-{i}"
        directory.mkdir(parents=True)
        (directory / "report.json").write_text(json.dumps(trainer_report(i)))
    manifest = build(runs, tmp_path / "episodes")
    assert sum(manifest["rows"].values()) == 5
    row = None
    for split in ("train", "dev", "test"):
        lines = (
            (tmp_path / "episodes" / (split + ".jsonl"))
            .read_text()
            .splitlines()
        )
        if lines:
            row = json.loads(lines[0])
            break
    assert row["summary"]["security"]["worst_family_false_safe_rate"] == 0.05
    delivered = execute(row, TraceClient(row), tmp_path / "replay")
    result = grade(row, delivered)
    assert result["execution_complete"] and result["action_correct"]
    assert result["evidence_correct"] and result["analysis_correct"], result
    # Same action with fabricated structured numbers must not verify.
    forged = copy.deepcopy(delivered)
    forged["decision"] = {
        **forged["decision"],
        "agreement_delta_pp": -4.2,
        "nll_delta": 1.5,
    }
    forged_result = grade(row, forged)
    assert (
        forged_result["action_correct"]
        and not forged_result["analysis_correct"]
    )
