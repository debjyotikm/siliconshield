"""Snapshot identity verification and dataset split-membership enforcement."""

import hashlib
import json

import pytest

from siliconshield.research_loop.episodes.dataset import verify_split_membership
from siliconshield.training.glimmer.identity import (
    snapshot_manifest,
    verify_snapshot,
)


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
