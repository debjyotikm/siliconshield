"""Validated GUI model registry: synthetic fixtures only."""

import json
from pathlib import Path

import pytest

from siliconshield.gui import registry
from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

VALID_DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


def _write_registry(
    tmp_path: Path, models: list[dict], schema: str = GUI_MODEL_REGISTRY_SCHEMA
) -> Path:
    path = tmp_path / "models.local.json"
    path.write_text(
        json.dumps({"schema": schema, "models": models}), encoding="utf-8"
    )
    return path


def _valid_model(tmp_path: Path, model_id: str = "fixture") -> dict:
    checkpoint = tmp_path / ("checkpoint-%s" % model_id)
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    python = tmp_path / ("python-%s" % model_id)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    laya_root = tmp_path / ("laya-%s" % model_id)
    (laya_root / "laya").mkdir(parents=True)
    (laya_root / "laya" / "__init__.py").write_text("")
    return {
        "id": model_id,
        "display_name": "Fixture %s" % model_id,
        "checkpoint": str(checkpoint),
        "python": str(python),
        "laya_root": str(laya_root),
        "expected_files": {"model.safetensors": VALID_DIGEST},
        "device": "cuda:0",
    }


def test_load_valid_registry_resolves_absolute(tmp_path: Path) -> None:
    entry = _valid_model(tmp_path)
    path = _write_registry(tmp_path, [entry])
    loaded = registry.load_registry(path)
    assert len(loaded) == 1
    assert loaded[0]["id"] == "fixture"
    assert Path(loaded[0]["checkpoint"]).is_absolute()
    assert loaded[0]["expected_files"] == {"model.safetensors": VALID_DIGEST}


def test_relative_paths_resolve_against_registry(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    entry = _valid_model(sub)
    # Rewrite paths relative to the registry file location.
    entry["checkpoint"] = "checkpoint-fixture"
    entry["python"] = "python-fixture"
    entry["laya_root"] = "laya-fixture"
    path = _write_registry(sub, [entry])
    loaded = registry.load_registry(path)
    assert loaded[0]["checkpoint"] == str(sub / "checkpoint-fixture")


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    first = _valid_model(tmp_path, "fixture")
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = _valid_model(second_dir, "fixture")
    path = _write_registry(tmp_path, [first, second])
    with pytest.raises(ValueError, match="duplicate"):
        registry.load_registry(path)


def test_missing_checkpoint_is_rejected(tmp_path: Path) -> None:
    entry = _valid_model(tmp_path)
    entry["checkpoint"] = str(tmp_path / "absent")
    path = _write_registry(tmp_path, [entry])
    with pytest.raises(ValueError, match="checkpoint"):
        registry.load_registry(path)


def test_non_executable_interpreter_is_rejected(tmp_path: Path) -> None:
    entry = _valid_model(tmp_path)
    missing = tmp_path / "absent-python"
    entry["python"] = str(missing)
    path = _write_registry(tmp_path, [entry])
    with pytest.raises(ValueError, match="interpreter"):
        registry.load_registry(path)


def test_missing_laya_package_is_rejected(tmp_path: Path) -> None:
    entry = _valid_model(tmp_path)
    entry["laya_root"] = str(tmp_path / "absent-laya")
    path = _write_registry(tmp_path, [entry])
    with pytest.raises(ValueError, match="LAYA runtime"):
        registry.load_registry(path)


def test_malformed_digest_is_rejected(tmp_path: Path) -> None:
    entry = _valid_model(tmp_path)
    entry["expected_files"] = {"model.safetensors": "short"}
    path = _write_registry(tmp_path, [entry])
    with pytest.raises(ValueError, match="digest"):
        registry.load_registry(path)


def test_cpu_device_is_accepted(tmp_path: Path) -> None:
    entry = _valid_model(tmp_path)
    entry["device"] = "cpu"
    path = _write_registry(tmp_path, [entry])
    loaded = registry.load_registry(path)
    assert loaded[0]["device"] == "cpu"


def test_invalid_device_is_rejected(tmp_path: Path) -> None:
    entry = _valid_model(tmp_path)
    entry["device"] = "tpu"
    path = _write_registry(tmp_path, [entry])
    with pytest.raises(ValueError, match="cpu or cuda"):
        registry.load_registry(path)


def test_wrong_schema_is_rejected(tmp_path: Path) -> None:
    entry = _valid_model(tmp_path)
    path = _write_registry(tmp_path, [entry], schema="other-schema-v1")
    with pytest.raises(ValueError, match="schema"):
        registry.load_registry(path)


def test_empty_model_list_is_rejected(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, [])
    with pytest.raises(ValueError, match="model list"):
        registry.load_registry(path)


def test_expected_files_are_preserved(tmp_path: Path) -> None:
    entry = _valid_model(tmp_path)
    entry["expected_files"] = {
        "model.safetensors": VALID_DIGEST,
        "tokenizer/tokenizer.json": OTHER_DIGEST,
    }
    path = _write_registry(tmp_path, [entry])
    loaded = registry.load_registry(path)
    assert loaded[0]["expected_files"] == entry["expected_files"]
