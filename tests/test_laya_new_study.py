"""An evaluated ancestor can initialize only an explicitly registered fresh study."""

import json
from pathlib import Path

import pytest

from siliconshield.training.laya.development_views import (
    attach_views,
    selection,
)
from siliconshield.training.laya.new_study import (
    file_hash,
    load_readonly_checkpoint,
    verify_new_study,
)


def fixture(tmp_path):
    model, data = tmp_path / "parent", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    (model / "weights.bin").write_bytes(b"fixture weights")
    old = tmp_path / "retired.jsonl"
    old.write_text(json.dumps({"id": "old-case", "state": "old input"}) + "\n")
    hashes = {}
    for split in ("train", "dev", "calibration", "test"):
        path = data / (split + ".jsonl")
        path.write_text(
            json.dumps({"id": split, "state": split + " example"}) + "\n"
        )
        hashes[split] = file_hash(path)
    manifest = {
        "schema": "siliconshield-new-study-v1",
        "run_id": "example",
        "retire_parent_evaluations": True,
        "parent_checkpoint": {"weights.bin": file_hash(model / "weights.bin")},
        "dataset": hashes,
        "retired_evaluations": [
            {"path": str(old), "sha256": file_hash(old), "role": "prior_test"}
        ],
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(manifest))
    return path, model, data, tmp_path / "run", manifest


def test_sdk_metadata_repair_leaves_parent_unchanged(tmp_path):
    model = tmp_path / "model"
    (model / "tokenizer").mkdir(parents=True)
    config = model / "tokenizer/tokenizer_config.json"
    config.write_text('{"tokenizer_class": "OldName"}')
    weights = model / "model.safetensors"
    weights.write_bytes(b"fixture weights")
    original = config.read_bytes()

    def loader(path):
        staged = Path(path)
        assert (staged / "model.safetensors").is_symlink()
        assert (staged / "model.safetensors").read_bytes() == b"fixture weights"
        (staged / "tokenizer/tokenizer_config.json").write_text(
            '{"tokenizer_class": "RepairedName"}'
        )
        return json.loads(
            (staged / "tokenizer/tokenizer_config.json").read_text()
        )

    loaded = load_readonly_checkpoint(model, loader)
    assert loaded["tokenizer_class"] == "RepairedName"
    assert config.read_bytes() == original
    assert weights.read_bytes() == b"fixture weights"


def test_explicit_child_preserves_parent_identity(tmp_path):
    path, model, data, output, _ = fixture(tmp_path)
    before = (model / "weights.bin").read_bytes()
    provenance = verify_new_study(path, model, data, output, "example")
    assert provenance["retire_parent_evaluations"]
    assert (model / "weights.bin").read_bytes() == before
    with pytest.raises(ValueError, match="overwrite"):
        verify_new_study(path, model, data, model / "child", "example")
    with pytest.raises(ValueError, match="overwrite"):
        verify_new_study(path, model, data, model.parent, "example")
    (model / "weights.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="parent checkpoint"):
        verify_new_study(path, model, data, output, "example")


def test_reused_evaluation_rejected_even_when_new_hash_registered(tmp_path):
    path, model, data, output, manifest = fixture(tmp_path)
    (data / "train.jsonl").write_text(
        json.dumps({"id": "renamed", "state": "old input"}) + "\n"
    )
    manifest["dataset"]["train"] = file_hash(data / "train.jsonl")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Retired evaluation row"):
        verify_new_study(path, model, data, output, "example")


def test_view_selection_does_not_hide_transfer_failure_with_retention():
    config = {
        "views": {"transfer": ["new"], "retention": ["old"]},
        "selection_view": "transfer",
        "retention_view": "retention",
        "max_retention_accuracy_drop": 0.01,
    }
    rows = [{"source": {"name": "new"}}, {"source": {"name": "old"}}]
    result = attach_views(
        {},
        rows,
        [[0.5], [0.99]],
        config,
        lambda rows, values: {"accuracy": values[0][0]},
    )
    assert selection(result, config, 0.99, lambda m: (-m["accuracy"],)) == (
        0,
        -0.5,
    )
    result["development_views"]["transfer"]["accuracy"] = 0.9
    result["development_views"]["retention"]["accuracy"] = 0.9
    assert selection(result, config, 0.99, lambda m: (-m["accuracy"],)) == (
        1,
        -0.9,
    )
    config["views"]["transfer"].append("old")
    with pytest.raises(ValueError, match="partition"):
        attach_views({}, rows, [[0.5], [0.99]], config, lambda *_: {})


def test_study_rejects_retired_lineages_and_cross_split_variants(tmp_path):
    path, model, data, output, manifest = fixture(tmp_path)
    for split in ("train", "dev"):
        target = data / (split + ".jsonl")
        row = json.loads(target.read_text())
        row["split_group_id"] = "shared-parent"
        target.write_text(json.dumps(row) + "\n")
        manifest["dataset"][split] = file_hash(target)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="lineage crosses"):
        verify_new_study(path, model, data, output, "example")
    retired = tmp_path / "retired.jsonl"
    retired.write_text(
        json.dumps(
            {"id": "old", "state": "old", "split_group_id": "shared-parent"}
        )
        + "\n"
    )
    manifest["retired_evaluations"][0]["sha256"] = file_hash(retired)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Retired evaluation"):
        verify_new_study(path, model, data, output, "example")


@pytest.mark.parametrize("mutation", ["parent", "train", "retired"])
def test_study_detects_changed_inputs(tmp_path, mutation):
    path, model, data, output, _ = fixture(tmp_path)
    target = {
        "parent": model / "weights.bin",
        "train": data / "train.jsonl",
        "retired": tmp_path / "retired.jsonl",
    }[mutation]
    with target.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_new_study(path, model, data, output, "example")
