"""Episode pipeline, identity and splits, and transfer episode tests: synthetic fixtures only."""

import copy
import hashlib
import json
import tempfile

import pytest

from siliconshield.research_loop.controller.replay import execute, grade
from siliconshield.research_loop.episodes.dataset import (
    build,
    verify_split_membership,
)
from siliconshield.research_loop.episodes.laya_results import adapt_report
from siliconshield.research_loop.episodes.transfer_episodes import (
    CAMPAIGN_FAMILY,
    SCENARIOS,
    build_transfer,
    load_transfer,
)
from siliconshield.training.glimmer.identity import (
    snapshot_manifest,
    verify_snapshot,
)


# ---- Merged from test_laya_episode_pipeline.py ----
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


# ---- Merged from test_identity_and_splits.py ----
def make_snapshot(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "muse_glimmer"}')
    (tmp_path / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "Auto"}'
    )
    (tmp_path / "model.safetensors").write_bytes(b"weights-v1")
    return tmp_path


def test_snapshot_fingerprint_verifies_and_detects_change(tmp_path):
    make_snapshot(tmp_path)
    manifest = verify_snapshot(tmp_path)
    assert (
        verify_snapshot(tmp_path, manifest["fingerprint"])["fingerprint"]
        == manifest["fingerprint"]
    )
    (tmp_path / "model.safetensors").write_bytes(b"weights-v2")
    # Full hashing covers weight shards, so a shard change changes the fingerprint.
    assert (
        snapshot_manifest(tmp_path, full=True)["fingerprint"]
        != manifest["fingerprint"]
    )
    before = snapshot_manifest(tmp_path, full=True)["fingerprint"]
    (tmp_path / "config.json").write_text('{"model_type": "other"}')
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_snapshot(tmp_path, before, full=True)


def test_missing_snapshot_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        verify_snapshot(tmp_path / "nope")
    with pytest.raises(ValueError, match="no safetensors"):
        verify_snapshot(tmp_path)


def test_split_membership_enforced_against_manifest(tmp_path):
    rows = [{"group_id": "g-dev"}]
    path = tmp_path / "dev.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "groups": {"g-train": "train", "g-dev": "dev", "g-test": "test"},
        "files": {"dev": checksum},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert verify_split_membership(rows, path) == "dev"
    renamed = tmp_path / "renamed.jsonl"
    renamed.write_bytes(path.read_bytes())
    assert verify_split_membership(rows, renamed) == "dev"
    with pytest.raises(ValueError, match="not permitted"):
        verify_split_membership(rows, renamed, allowed_splits=("train",))
    path.write_text(json.dumps({"group_id": "g-dev", "tampered": True}) + "\n")
    with pytest.raises(ValueError, match="split hash"):
        verify_split_membership(rows, path)
    with pytest.raises(ValueError, match="cannot be verified"):
        verify_split_membership(
            rows, path, manifest_path=tmp_path / "absent.json"
        )


def test_default_identity_detects_changed_weights(tmp_path):
    make_snapshot(tmp_path)
    identity = verify_snapshot(tmp_path)["fingerprint"]
    (tmp_path / "model.safetensors").write_bytes(b"changed-weights")
    with pytest.raises(ValueError, match="mismatch"):
        verify_snapshot(tmp_path, identity)


def test_hmac_manifest_rejects_tampering(tmp_path, monkeypatch):
    from siliconshield.research_loop.episodes.dataset import write_manifest

    key = tmp_path / "key"
    key.write_bytes(b"fixture-key-for-contract-test")
    monkeypatch.setenv("SILICONSHIELD_DATASET_KEY", str(key))
    for split in ("train", "dev", "test"):
        (tmp_path / (split + ".jsonl")).write_text(
            json.dumps({"group_id": split}) + "\n"
        )
    write_manifest(
        tmp_path, {"groups": {s: s for s in ("train", "dev", "test")}}
    )
    rows = [{"group_id": "dev"}]
    verify_split_membership(rows, tmp_path / "dev.jsonl")
    monkeypatch.delenv("SILICONSHIELD_DATASET_KEY")
    with pytest.raises(ValueError, match="requires SILICONSHIELD_DATASET_KEY"):
        verify_split_membership(rows, tmp_path / "dev.jsonl")
    monkeypatch.setenv("SILICONSHIELD_DATASET_KEY", str(key))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["groups"]["dev"] = "train"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="signature"):
        verify_split_membership(rows, tmp_path / "dev.jsonl")


def test_training_rejects_unsigned_splits_before_model_loading(
    tmp_path, monkeypatch
):
    from siliconshield.research_loop.episodes.operator_episodes import build
    from siliconshield.training.glimmer.train import training_rows

    monkeypatch.delenv("SILICONSHIELD_DATASET_KEY", raising=False)
    build(tmp_path / "data")
    with pytest.raises(ValueError, match="requires a signed dataset"):
        training_rows(tmp_path / "data")
    # Re-signing the authored fixture allows the same rows to enter training.
    from siliconshield.research_loop.episodes.dataset import write_manifest

    key = tmp_path / "key"
    key.write_bytes(b"fixture-key")
    monkeypatch.setenv("SILICONSHIELD_DATASET_KEY", str(key))
    manifest = json.loads((tmp_path / "data/manifest.json").read_text())
    write_manifest(tmp_path / "data", manifest)
    train, dev = training_rows(tmp_path / "data")
    assert train and dev
    assert not {r["group_id"] for r in train} & {r["group_id"] for r in dev}


def test_training_code_identity_gate_precedes_gpu_acquisition(monkeypatch):
    from siliconshield.research_loop.execution import gpu_lease
    from siliconshield.training.glimmer import identity, train

    monkeypatch.setattr(identity, "code_identity", lambda: "a" * 64)
    monkeypatch.setattr(
        gpu_lease,
        "acquire_gpu",
        lambda: pytest.fail("GPU acquisition preceded identity check"),
    )
    monkeypatch.delenv("SILICONSHIELD_EXPECTED_CODE_IDENTITY", raising=False)
    with pytest.raises(ValueError, match="EXPECTED_CODE_IDENTITY"):
        train.train(None)
    monkeypatch.setenv("SILICONSHIELD_EXPECTED_CODE_IDENTITY", "b" * 64)
    with pytest.raises(ValueError, match="Launch code identity mismatch"):
        train.train(None)
    monkeypatch.setenv("SILICONSHIELD_EXPECTED_CODE_IDENTITY", "a" * 64)
    assert identity.verify_code_identity() == "a" * 64


def test_training_releases_gpu_lease_on_failure(tmp_path, monkeypatch):
    from siliconshield.research_loop.execution import gpu_lease
    from siliconshield.training.glimmer import identity, train

    monkeypatch.setattr(identity, "code_identity", lambda: "a" * 64)
    monkeypatch.setenv("SILICONSHIELD_EXPECTED_CODE_IDENTITY", "a" * 64)
    monkeypatch.setenv("SILICONSHIELD_GPU_LOCK", str(tmp_path / "gpu.lock"))
    handles = []
    acquire = gpu_lease.acquire_gpu

    def record_acquire():
        handle = acquire()
        handles.append(handle)
        return handle

    def fail(args, verified):
        assert verified == "a" * 64
        assert not handles[0].closed
        with pytest.raises(RuntimeError, match="already owned"):
            acquire()
        raise ValueError("authored failure")

    monkeypatch.setattr(gpu_lease, "acquire_gpu", record_acquire)
    monkeypatch.setattr(train, "_train", fail)
    with pytest.raises(ValueError, match="authored failure"):
        train.train(None)
    assert handles[0].closed
    with acquire() as handle:
        assert not handle.closed


# ---- Merged from test_transfer_episodes.py ----
def _fixture_client(row):
    class TraceClient:
        model = "transfer-fixture"
        endpoint = "http://fixture.invalid"

        def complete(self, messages, tools=()):
            return copy.deepcopy(row["messages"][len(messages)])

    return TraceClient()


def test_transfer_episodes_form_one_reviewed_campaign(tmp_path):
    from siliconshield.schemas import TRANSFER_EPISODES_SCHEMA

    manifest = build_transfer(tmp_path / "transfer")
    assert manifest["rows"] == 4
    assert manifest["schema"] == TRANSFER_EPISODES_SCHEMA
    assert manifest["campaign_family"] == CAMPAIGN_FAMILY
    assert manifest["development_only"] is True
    rows = load_transfer(tmp_path / "transfer/transfer.jsonl")
    assert (
        {r["group_id"] for r in rows}
        == {CAMPAIGN_FAMILY}
        == set(manifest["lineages"])
    )
    for row in rows:
        assert row["verification"].get("campaign_family") == CAMPAIGN_FAMILY


def test_transfer_episodes_verify_with_supported_reasons(tmp_path):
    from siliconshield.research_loop.controller.replay import execute, grade

    build_transfer(tmp_path / "transfer")
    for row in load_transfer(tmp_path / "transfer/transfer.jsonl"):
        with tempfile.TemporaryDirectory() as attempt:
            delivered = execute(row, _fixture_client(row), attempt)
        result = grade(row, delivered)
        assert result["action_correct"], row["verification"]["scenario"]
        assert result["explanation_correct"], (
            row["verification"]["scenario"],
            result.get("explanation_issues"),
        )
        assert result["verified"], row["verification"]["scenario"]


def test_transfer_loader_rejects_other_lineages(tmp_path):
    from siliconshield.research_loop.episodes.operator_episodes import (
        episode_for,
    )

    scenario = SCENARIOS[0]
    row = episode_for(
        scenario[0],
        scenario[1],
        scenario[2],
        scenario[3],
        scenario[4],
        "other-lineage",
        0,
    )
    path = tmp_path / "other.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="campaign lineages"):
        load_transfer(path)


def test_transfer_episodes_carry_decision_point_evidence(tmp_path):
    build_transfer(tmp_path / "transfer")
    rows = load_transfer(tmp_path / "transfer/transfer.jsonl")
    by_scenario = {r["verification"]["scenario"]: r for r in rows}
    stop = by_scenario["transfer_regression_stop"]["summary"]
    assert stop["acceptance"] == {
        "accepted": False,
        "reasons": ["false_safe_budget_exceeded"],
        "thresholds": {"max_worst_family_false_safe_rate": 0.1},
    }
    assert stop["selected_step"] == 2
    assert stop["schedule"]["plateau_lr_reductions"] == []
    plateau = by_scenario["transfer_plateau_no_intervention"]["summary"]
    assert plateau["schedule"]["stale"] == 2
    assert plateau["schedule"]["lr_scale"] == 1.0


def test_external_catalog_controls_observations_and_family(tmp_path):
    from siliconshield.research_loop.episodes.transfer_episodes import (
        load_catalog,
    )

    catalog = load_catalog()
    catalog["campaign_family"] = "external-fixture"
    catalog["scenarios"][0]["state"]["catalog_marker"] = "supplied-observation"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    manifest = build_transfer(tmp_path / "output", path)
    rows = load_transfer(tmp_path / "output/transfer.jsonl")
    assert manifest["lineages"] == ["external-fixture"]
    assert rows[0]["summary"]["catalog_marker"] == "supplied-observation"
    with pytest.raises(ValueError, match="campaign lineages"):
        load_transfer(
            tmp_path / "output/transfer.jsonl", allowed_prefix="another-"
        )


def test_invalid_catalog_rejected_before_creating_output(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"schema": "unknown"}))
    with pytest.raises(ValueError, match="catalog schema"):
        build_transfer(tmp_path / "output", path)
    assert not (tmp_path / "output").exists()
