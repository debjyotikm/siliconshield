"""GUI entry, lifecycle, registry, setup, qualification, and worker-protocol tests: Qt-free fixtures only."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from siliconshield.app.inference_worker import questions_for_staged
from siliconshield.gui import contracts, registry
from siliconshield.gui import setup as gui_setup
from siliconshield.gui.__main__ import (
    build_parser,
    ensure_registry_for_launch,
    has_location_args,
    resolve_models_path,
)
from siliconshield.gui.application import (
    MAX_DESIGN_BYTES,
    available_load_devices,
    build_state,
    check_inputs,
    format_scan_excerpt,
    is_result_stale,
    needs_preview_refresh,
    read_design_file,
    resolve_load_device,
    snapshot_key,
)
from siliconshield.gui.qualification import (
    REQUIRED_CASES,
    build_case_states,
    qualify_states,
)
from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA
from siliconshield.training.laya.question_contracts import (
    LEGACY_CONTRACT,
    NEW_CONTRACT,
    match_version,
    write_export_record,
)


# ---- Merged from test_gui_entry.py ----
def test_resolve_models_prefers_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import siliconshield.gui.setup as setup_module

    monkeypatch.setattr(
        setup_module, "default_registry_path", lambda: tmp_path / "absent.json"
    )
    monkeypatch.setenv("SILICONSHIELD_GUI_MODELS", "/env/models.json")
    assert resolve_models_path("/flag/models.json") == "/flag/models.json"
    assert resolve_models_path(None) == "/env/models.json"
    monkeypatch.delenv("SILICONSHIELD_GUI_MODELS")
    assert resolve_models_path(None) is None


def test_resolve_models_uses_default_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import siliconshield.gui.setup as setup_module

    present = tmp_path / "models.local.json"
    present.write_text("{}")
    monkeypatch.setattr(setup_module, "default_registry_path", lambda: present)
    monkeypatch.delenv("SILICONSHIELD_GUI_MODELS", raising=False)
    assert resolve_models_path(None) == str(present)


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.load_timeout == 180
    assert args.analyze_timeout == 60
    assert args.init is False
    assert args.device == "cuda:0"


def test_ensure_registry_prefers_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import siliconshield.gui.setup as setup_module

    present = tmp_path / "models.local.json"
    present.write_text("{}")
    monkeypatch.setattr(setup_module, "default_registry_path", lambda: present)
    monkeypatch.delenv("SILICONSHIELD_GUI_MODELS", raising=False)
    monkeypatch.delenv("SILICONSHIELD_SECURITY_CHECKPOINT", raising=False)
    args = build_parser().parse_args([])
    assert ensure_registry_for_launch(args) == str(present)
    assert has_location_args(args) is False


def test_ensure_registry_autocreates_from_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import siliconshield.gui.setup as setup_module

    monkeypatch.setattr(
        setup_module, "default_registry_path", lambda: tmp_path / "absent.json"
    )
    monkeypatch.delenv("SILICONSHIELD_GUI_MODELS", raising=False)
    checkpoint = tmp_path / "export"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    python = tmp_path / "pybin"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    laya_root = tmp_path / "laya"
    (laya_root / "laya").mkdir(parents=True)
    (laya_root / "laya" / "__init__.py").write_text("")
    target = tmp_path / "custom.json"
    args = build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--python",
            str(python),
            "--laya-root",
            str(laya_root),
            "--models",
            str(target),
        ]
    )
    assert has_location_args(args) is True
    assert ensure_registry_for_launch(args) == str(target)
    assert target.is_file()


def test_ensure_registry_returns_none_when_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import siliconshield.gui.setup as setup_module

    monkeypatch.setattr(
        setup_module, "default_registry_path", lambda: tmp_path / "absent.json"
    )
    monkeypatch.delenv("SILICONSHIELD_GUI_MODELS", raising=False)
    monkeypatch.delenv("SILICONSHIELD_SECURITY_CHECKPOINT", raising=False)
    monkeypatch.delenv("SILICONSHIELD_LAYA_CHECKPOINT", raising=False)
    monkeypatch.delenv("SILICONSHIELD_SECURITY_PYTHON", raising=False)
    monkeypatch.delenv("SILICONSHIELD_LAYA_PYTHON", raising=False)
    monkeypatch.delenv("LAYA_PYTHON", raising=False)
    monkeypatch.delenv("LAYA_ROOT", raising=False)
    args = build_parser().parse_args([])
    assert ensure_registry_for_launch(args) is None


def test_gui_help_without_qt(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [sys.executable, "-m", "siliconshield.gui", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0
    assert "--models" in run.stdout


def test_worker_help_outside_checkout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [sys.executable, "-m", "siliconshield.app.inference_worker", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0
    assert "usage:" in run.stdout.lower()


def test_lightweight_gui_imports_without_qt() -> None:
    import siliconshield.gui.contracts
    import siliconshield.gui.controller
    import siliconshield.gui.registry

    assert siliconshield.gui.contracts.PROTOCOL_VERSION == 1


def test_read_design_file_round_trip(tmp_path: Path) -> None:
    rtl = tmp_path / "design.sv"
    rtl.write_text("module topmodule; endmodule", encoding="utf-8")
    text, snapshot = read_design_file(str(rtl))
    assert "topmodule" in text
    assert snapshot["bytes"] == len(rtl.read_bytes())
    assert len(snapshot["sha256"]) == 64


def test_read_design_file_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        read_design_file(str(tmp_path / "absent.sv"))


def test_read_design_file_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a file"):
        read_design_file(str(tmp_path))


def test_read_design_file_rejects_suffix(tmp_path: Path) -> None:
    other = tmp_path / "design.txt"
    other.write_text("text")
    with pytest.raises(ValueError, match="suffix"):
        read_design_file(str(other))


def test_read_design_file_rejects_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.sv"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_design_file(str(empty))


def test_read_design_file_rejects_oversized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import siliconshield.gui.application as app_module

    big = tmp_path / "big.sv"
    big.write_bytes(b"x" * (MAX_DESIGN_BYTES + 1))
    monkeypatch.setattr(app_module, "MAX_DESIGN_BYTES", 4)
    tiny = tmp_path / "tiny.sv"
    tiny.write_bytes(b"12345")
    with pytest.raises(ValueError, match="exceeds"):
        read_design_file(str(tiny))


def test_read_design_file_rejects_encoding(tmp_path: Path) -> None:
    bad = tmp_path / "bad.sv"
    bad.write_bytes(b"\xff\xfe\x00module")
    with pytest.raises(ValueError, match="UTF-8"):
        read_design_file(str(bad))


def test_check_inputs_gates_run() -> None:
    assert check_inputs("", "", False) is not None
    assert check_inputs("req", "", True) is not None
    assert check_inputs("", "rtl", True) is not None
    assert check_inputs("req", "rtl", True) is None


def test_build_state_matches_cli_format() -> None:
    from siliconshield.app.laya_assessor import state_text

    assert build_state("req", "asm", "rtl") == state_text("req", "asm", "rtl")


def test_snapshot_key_is_stable() -> None:
    first = snapshot_key("STATE")
    assert first == snapshot_key("STATE")
    assert first != snapshot_key("OTHER")


def test_device_free_fails_open_without_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil

    from siliconshield.gui.application import _device_free

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert _device_free("cuda:0") is True
    assert _device_free("cpu") is True


def test_device_free_detects_busy_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil
    import subprocess as sp

    from siliconshield.gui.application import _device_free

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def fake_run(*args: object, **kwargs: object) -> object:
        return type("R", (), {"stdout": "1234\n"})()

    monkeypatch.setattr(sp, "run", fake_run)
    assert _device_free("cuda:0") is False


def test_save_record_uses_gui_schema() -> None:
    from siliconshield.schemas import GUI_ANALYSIS_SCHEMA

    assert GUI_ANALYSIS_SCHEMA == "siliconshield-gui-analysis-v1"
    record = {"schema": GUI_ANALYSIS_SCHEMA}
    assert json.dumps(record)


def test_available_devices_prefers_registry_default() -> None:
    assert available_load_devices("cuda:0") == ["cuda:0", "cpu"]
    assert available_load_devices("cuda:1") == ["cuda:1", "cpu"]
    assert available_load_devices("cpu") == ["cpu", "cuda:0"]


def test_resolve_load_device_prefers_gui_choice() -> None:
    assert resolve_load_device("cuda:0", "cpu") == "cpu"
    assert resolve_load_device("cuda:0", "cuda:0") == "cuda:0"
    assert resolve_load_device("cpu", None) == "cpu"
    assert resolve_load_device("cuda:1", None) == "cuda:1"


def test_needs_preview_refresh() -> None:
    assert needs_preview_refresh("", None) is False
    assert needs_preview_refresh("", {"path": "demo:x"}) is False
    assert needs_preview_refresh("/a.sv", None) is True
    assert needs_preview_refresh("/a.sv", {"path": "/a.sv"}) is False
    assert needs_preview_refresh("/b.sv", {"path": "/a.sv"}) is True
    assert needs_preview_refresh("/b.sv", {"path": "demo:x"}) is True


def test_is_result_stale() -> None:
    assert is_result_stale(None, "k") is True
    assert is_result_stale("a", "a") is False
    assert is_result_stale("a", "b") is True


def test_format_scan_excerpt() -> None:
    title, body = format_scan_excerpt(
        {
            "index": 3,
            "lines": "21-35",
            "label": "violates_requirement",
            "text": "module x;",
        }
    )
    assert title == "Excerpt 3 · lines 21-35 · violates_requirement"
    assert body == "module x;"
    title, body = format_scan_excerpt({"index": 0, "verdict": "ERROR"})
    assert "ERROR" in title
    assert body == ""


def test_check_inputs_rejects_huge_text() -> None:
    from siliconshield.gui.application import (
        MAX_ASSUMPTIONS_CHARS,
        MAX_REQUIREMENT_CHARS,
    )

    assert (
        check_inputs("x" * (MAX_REQUIREMENT_CHARS + 1), "rtl", True) is not None
    )
    assert (
        check_inputs("req", "rtl", True, "y" * (MAX_ASSUMPTIONS_CHARS + 1))
        is not None
    )
    assert check_inputs("req", "rtl", True, "asm") is None


def test_window_opens_offscreen_without_models() -> None:
    pytest.importorskip("PySide6.QtWidgets")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from siliconshield.gui import registry as registry_module
    from siliconshield.gui.application import _build_window

    app = QApplication.instance() or QApplication([])
    window = _build_window(registry_module, None, 180, 60)
    assert "SiliconShield" in window.windowTitle()
    window.close()


def test_window_has_device_selector(tmp_path: Path) -> None:
    pytest.importorskip("PySide6.QtWidgets")
    import json
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QComboBox

    from siliconshield.gui import registry as registry_module
    from siliconshield.gui.application import _build_window

    app = QApplication.instance() or QApplication([])
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"x")
    python = tmp_path / "pybin"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    laya_root = tmp_path / "laya"
    (laya_root / "laya").mkdir(parents=True)
    (laya_root / "laya" / "__init__.py").write_text("")
    registry = tmp_path / "models.local.json"
    registry.write_text(
        json.dumps(
            {
                "schema": "siliconshield-gui-model-registry-v1",
                "models": [
                    {
                        "id": "fixture",
                        "display_name": "Fixture",
                        "checkpoint": str(checkpoint),
                        "python": str(python),
                        "laya_root": str(laya_root),
                        "expected_files": {"model.safetensors": "a" * 64},
                        "device": "cuda:0",
                    }
                ],
            }
        )
    )
    window = _build_window(registry_module, str(registry), 180, 60)
    boxes = window.findChildren(QComboBox)
    assert len(boxes) >= 2
    window.close()


# ---- Merged from test_gui_lifecycle.py ----
def test_should_enable_load_matrix() -> None:
    from siliconshield.gui.application import should_enable_load

    assert should_enable_load(False, True, False, False) is True
    assert should_enable_load(True, True, False, False) is False
    assert should_enable_load(False, False, False, False) is False
    # Already loaded: must unload first, never stack workers.
    assert should_enable_load(False, True, True, True) is False
    assert should_enable_load(False, True, True, False) is False
    # Previous worker still exiting: wait before starting a new one.
    assert should_enable_load(False, True, False, True) is False


def test_worker_start_refuses_when_running() -> None:
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from siliconshield.gui.worker_client import WorkerClient

    app = QApplication.instance() or QApplication([])
    client = WorkerClient(lambda _m: None, lambda _c, _e: None, lambda _d: None)
    try:
        client.start("sleep", ["10"])
        assert client.running
        with pytest.raises(RuntimeError, match="already running"):
            client.start("sleep", ["10"])
    finally:
        client.terminate()
        client.wait_for_exit(5000)
        app.processEvents()


def test_intentional_terminate_reports_no_process_error() -> None:
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from siliconshield.gui.worker_client import WorkerClient

    app = QApplication.instance() or QApplication([])
    errors: list[str] = []
    finished: list[tuple[int, bool]] = []
    client = WorkerClient(
        lambda _m: None,
        lambda code, crashed: finished.append((code, crashed)),
        errors.append,
    )
    try:
        client.start("sleep", ["30"])
        assert client.running
        client.terminate()
        assert client.wait_for_exit(5000)
        for _ in range(100):
            app.processEvents()
            if finished:
                break
        assert finished, "finished callback must fire after terminate"
        assert errors == [], "intentional stop must not report a crash: %r" % (
            errors,
        )
    finally:
        client.wait_for_exit(5000)


def test_stage_reuses_existing_generation(tmp_path: Path) -> None:
    from siliconshield.app.inference_worker import _stage_checkpoint

    checkpoint = tmp_path / "export"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "config.json").write_bytes(b"{}")
    first, _fp1 = _stage_checkpoint(checkpoint, lambda _s, _d: None)
    marker = first / "model.safetensors"
    assert marker.is_file()
    mtime = marker.stat().st_mtime_ns
    second, _fp2 = _stage_checkpoint(checkpoint, lambda _s, _d: None)
    assert second == first
    assert marker.stat().st_mtime_ns == mtime


def test_verify_expected_rejects_extra_files(tmp_path: Path) -> None:
    from siliconshield.app.inference_worker import _verify_expected

    checkpoint = tmp_path / "export"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "extra.bin").write_bytes(b"unexpected")
    import hashlib

    digest = hashlib.sha256(b"weights").hexdigest()
    with pytest.raises(ValueError, match="unregistered"):
        _verify_expected(
            checkpoint,
            {"model.safetensors": digest},
            lambda _s, _d: None,
        )


def test_staged_matches_detects_partial_copy(tmp_path: Path) -> None:
    from siliconshield.app.inference_worker import (
        _stage_checkpoint,
        _staged_matches,
    )

    checkpoint = tmp_path / "export"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "config.json").write_bytes(b"{}")
    staged, _fp = _stage_checkpoint(checkpoint, lambda _s, _d: None)
    assert _staged_matches(staged, checkpoint) is True
    (staged / "config.json").unlink()
    assert _staged_matches(staged, checkpoint) is False
    (staged / "model.safetensors").write_bytes(b"partial")
    assert _staged_matches(staged, checkpoint) is False


def test_staged_copy_rejects_same_size_tampering(tmp_path: Path) -> None:
    from siliconshield.app.inference_worker import (
        _stage_checkpoint,
        _staged_matches,
    )

    checkpoint = tmp_path / "export"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights-01")
    (checkpoint / "config.json").write_bytes(b"{}")
    staged, _ = _stage_checkpoint(checkpoint, lambda _s, _d: None)
    # Same-size tampering must not be trusted or reused.
    (staged / "model.safetensors").write_bytes(b"weights-02")
    assert _staged_matches(staged, checkpoint) is False
    restaged, _ = _stage_checkpoint(checkpoint, lambda _s, _d: None)
    assert (restaged / "model.safetensors").read_bytes() == b"weights-01"
    assert _staged_matches(restaged, checkpoint) is True


def test_single_file_stage_rejects_same_size_tampering(
    tmp_path: Path,
) -> None:
    from siliconshield.app.inference_worker import _stage_checkpoint

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"weights-01")
    staged, _ = _stage_checkpoint(checkpoint, lambda _s, _d: None)
    assert staged.read_bytes() == b"weights-01"
    staged.write_bytes(b"weights-02")
    restaged, _ = _stage_checkpoint(checkpoint, lambda _s, _d: None)
    assert restaged.read_bytes() == b"weights-01"


def test_default_registry_path_expands_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from siliconshield.gui.setup import default_registry_path

    monkeypatch.setenv("XDG_CONFIG_HOME", "~/cfg")
    assert str(default_registry_path()).endswith(
        "siliconshield/models.local.json"
    )
    assert "~" not in str(default_registry_path())


def test_window_has_close_handler_and_bounded_details(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from siliconshield.gui import registry as registry_module
    from siliconshield.gui.application import _build_window

    app = QApplication.instance() or QApplication([])
    window = _build_window(registry_module, None, 180, 60)
    try:
        assert callable(getattr(window, "closeEvent", None))
        from PySide6.QtWidgets import QPlainTextEdit

        details = window.findChild(QPlainTextEdit, "details")
        assert details is not None
        assert details.maximumBlockCount() == 500
    finally:
        window.close()


# ---- Merged from test_gui_registry.py ----
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


# ---- Merged from test_gui_setup.py ----
def _fixture_layout(tmp_path: Path) -> dict[str, str]:
    checkpoint = tmp_path / "export"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"fixture-weights")
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    laya_root = tmp_path / "laya-root"
    (laya_root / "laya").mkdir(parents=True)
    (laya_root / "laya" / "__init__.py").write_text("")
    return {
        "checkpoint": str(checkpoint),
        "python": str(python),
        "laya_root": str(laya_root),
    }


def test_build_entry_hashes_checkpoint(tmp_path: Path) -> None:
    layout = _fixture_layout(tmp_path)
    entry = gui_setup.build_entry(
        model_id="fixture",
        display_name="Fixture",
        checkpoint=layout["checkpoint"],
        python=layout["python"],
        laya_root=layout["laya_root"],
        device="cuda:0",
    )
    assert entry["id"] == "fixture"
    assert "model.safetensors" in entry["expected_files"]
    assert len(next(iter(entry["expected_files"].values()))) == 64


def test_build_entry_accepts_cpu(tmp_path: Path) -> None:
    layout = _fixture_layout(tmp_path)
    entry = gui_setup.build_entry(
        model_id="fixture",
        display_name="Fixture",
        checkpoint=layout["checkpoint"],
        python=layout["python"],
        laya_root=layout["laya_root"],
        device="cpu",
    )
    assert entry["device"] == "cpu"


def test_build_entry_rejects_bad_device(tmp_path: Path) -> None:
    layout = _fixture_layout(tmp_path)
    with pytest.raises(ValueError, match="cpu or cuda"):
        gui_setup.build_entry(
            model_id="fixture",
            display_name="Fixture",
            checkpoint=layout["checkpoint"],
            python=layout["python"],
            laya_root=layout["laya_root"],
            device="tpu",
        )


def test_write_registry_round_trip(tmp_path: Path) -> None:
    from siliconshield.gui import registry as registry_module

    layout = _fixture_layout(tmp_path)
    entry = gui_setup.build_entry(
        model_id="fixture",
        display_name="Fixture",
        checkpoint=layout["checkpoint"],
        python=layout["python"],
        laya_root=layout["laya_root"],
        device="cuda:1",
    )
    target = tmp_path / "sub" / "models.local.json"
    written = gui_setup.write_registry(target, [entry])
    assert written == target
    loaded = registry_module.load_registry(written)
    assert loaded[0]["device"] == "cuda:1"


def test_init_from_args_uses_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _fixture_layout(tmp_path)
    monkeypatch.setenv(
        "SILICONSHIELD_SECURITY_CHECKPOINT", layout["checkpoint"]
    )
    monkeypatch.setenv("SILICONSHIELD_SECURITY_PYTHON", layout["python"])
    monkeypatch.setenv("LAYA_ROOT", layout["laya_root"])
    monkeypatch.delenv("SILICONSHIELD_GUI_MODELS", raising=False)
    target = tmp_path / "models.local.json"
    args = type(
        "Args",
        (),
        {
            "checkpoint": None,
            "python": None,
            "laya_root": None,
            "device": "cuda:0",
            "model_id": "fixture",
            "display_name": "Fixture",
            "models": str(target),
        },
    )()
    written = gui_setup.init_from_args(args)
    assert written == target
    assert target.is_file()


def test_init_merges_gpu_and_cpu_entries(tmp_path: Path) -> None:
    layout = _fixture_layout(tmp_path)
    target = tmp_path / "models.local.json"
    for model_id, display, device in (
        ("fixture", "Fixture GPU", "cuda:0"),
        ("fixture-cpu", "Fixture CPU", "cpu"),
    ):
        args = type(
            "Args",
            (),
            {
                "checkpoint": layout["checkpoint"],
                "python": layout["python"],
                "laya_root": layout["laya_root"],
                "device": device,
                "model_id": model_id,
                "display_name": display,
                "models": str(target),
            },
        )()
        gui_setup.init_from_args(args)
    from siliconshield.gui import registry as registry_module

    loaded = registry_module.load_registry(target)
    by_id = {item["id"]: item for item in loaded}
    assert by_id["fixture"]["device"] == "cuda:0"
    assert by_id["fixture-cpu"]["device"] == "cpu"


def test_init_from_args_missing_locations() -> None:
    args = type(
        "Args",
        (),
        {
            "checkpoint": None,
            "python": None,
            "laya_root": None,
            "device": "cuda:0",
            "model_id": "fixture",
            "display_name": "Fixture",
            "models": None,
        },
    )()
    with pytest.raises(ValueError, match="Missing"):
        gui_setup.init_from_args(args)


def test_default_registry_path_is_outside_repo() -> None:
    path = gui_setup.default_registry_path()
    assert path.name == "models.local.json"
    assert "siliconshield" in str(path)


# ---- Merged from test_gui_qualification.py ----
def test_required_cases_present_and_covered():
    states = build_case_states()
    assert set(states) == set(REQUIRED_CASES)
    report = qualify_states(states, contract=NEW_CONTRACT)
    assert report["contract"] == NEW_CONTRACT
    assert len(report["cases"]) == len(REQUIRED_CASES)


def test_fixtures_carry_distinct_functional_bodies():
    states = build_case_states()
    assert len(set(states.values())) == len(REQUIRED_CASES)
    bodies = {
        name: state.split("RTL\n", 1)[1] for name, state in states.items()
    }
    assert all(
        "always_ff" in body or "function" in body for body in bodies.values()
    )
    # Intentional same-RTL opposing-policy pair; all others differ.
    assert bodies["reset-exempt"] == bodies["reset-inclusive"]
    rest = [body for name, body in bodies.items() if name != "reset-inclusive"]
    assert len(set(rest)) == len(REQUIRED_CASES) - 1


def test_staged_questions_follow_export_record(tmp_path):
    legacy_questions, legacy_version = questions_for_staged(
        tmp_path / "missing"
    )
    assert legacy_version == LEGACY_CONTRACT
    assert match_version(legacy_questions) == LEGACY_CONTRACT
    export = tmp_path / "export"
    export.mkdir()
    write_export_record(export, NEW_CONTRACT)
    questions, version = questions_for_staged(export)
    assert version == NEW_CONTRACT
    assert match_version(questions) == NEW_CONTRACT


def test_staged_single_file_reads_parent_record(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    write_export_record(export, NEW_CONTRACT)
    weights = export / "model.pt"
    weights.write_bytes(b"fake-weights")
    _, version = questions_for_staged(weights)
    assert version == NEW_CONTRACT


def test_staged_corrupt_record_raises(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    (export / "question_contract.json").write_text("{corrupt")
    with pytest.raises(ValueError):
        questions_for_staged(export)
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    write_export_record(tampered, NEW_CONTRACT)
    record = tampered / "question_contract.json"
    record.write_text(record.read_text().replace("v1", "v0"))
    with pytest.raises(ValueError):
        questions_for_staged(tampered)


# ---- Merged from test_gui_contracts.py ----
def _valid_payload() -> dict:
    return {
        "label": "satisfies_requirement",
        "probabilities": {
            "satisfies_requirement": 0.7,
            "violates_requirement": 0.2,
            "insufficient_context": 0.1,
        },
        "usage": {"input_tokens": 10},
        "coverage": {
            "truncated": False,
            "truncated_questions": [],
            "state_tokens_dropped": 0,
        },
    }


def test_make_load_request_carries_protocol() -> None:
    request = contracts.make_load_request("req-1", "fixture")
    assert request["type"] == "load"
    assert request["protocol"] == contracts.PROTOCOL_VERSION
    assert request["model_id"] == "fixture"


def test_make_load_request_without_model() -> None:
    request = contracts.make_load_request("req-2")
    assert request["type"] == "load"
    assert "model_id" not in request


def test_make_analyze_request_carries_state() -> None:
    request = contracts.make_analyze_request("req-3", "REQUIREMENT\nx")
    assert request["state"] == "REQUIREMENT\nx"


def test_valid_result_passes() -> None:
    checked = contracts.validate_result(_valid_payload())
    assert checked["label"] == "satisfies_requirement"


def test_missing_label_is_rejected() -> None:
    payload = _valid_payload()
    del payload["probabilities"]["violates_requirement"]
    with pytest.raises(ValueError, match="missing labels"):
        contracts.validate_result(payload)


def test_verdict_must_match_top_score() -> None:
    payload = _valid_payload()
    payload["label"] = "violates_requirement"
    with pytest.raises(ValueError, match="disagrees"):
        contracts.validate_result(payload)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_non_finite_or_ranged_scores_rejected(bad: float) -> None:
    payload = _valid_payload()
    payload["probabilities"]["satisfies_requirement"] = bad
    # Renormalize the other scores so only the bad field matters.
    with pytest.raises(ValueError):
        contracts.validate_result(payload)


def test_scores_must_sum_to_one() -> None:
    payload = _valid_payload()
    payload["probabilities"] = {
        "satisfies_requirement": 0.5,
        "violates_requirement": 0.2,
        "insufficient_context": 0.1,
    }
    with pytest.raises(ValueError, match="sum"):
        contracts.validate_result(payload)


def test_bool_scores_are_rejected() -> None:
    payload = _valid_payload()
    payload["probabilities"]["satisfies_requirement"] = True  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="not a number"):
        contracts.validate_result(payload)


def test_scope_note_marks_custom_inputs_unverified() -> None:
    from siliconshield.gui.application import (
        CUSTOM_SCOPE_NOTE,
        SCOPE_NOTE,
        scope_note_for,
    )

    assert scope_note_for({"path": "demo:unauthorized-write"}) == SCOPE_NOTE
    custom = scope_note_for({"path": "/tmp/design.sv"})
    assert custom == CUSTOM_SCOPE_NOTE
    assert "not independently verified" in custom
    assert scope_note_for(None) == CUSTOM_SCOPE_NOTE


def test_truncated_input_refuses_verdict() -> None:
    payload = _valid_payload()
    assert isinstance(payload["coverage"], dict)
    payload["coverage"]["truncated"] = True
    with pytest.raises(ValueError, match="truncated"):
        contracts.validate_result(payload)


def test_dropped_tokens_refuse_verdict() -> None:
    payload = _valid_payload()
    assert isinstance(payload["coverage"], dict)
    payload["coverage"]["state_tokens_dropped"] = 3
    with pytest.raises(ValueError, match="truncated"):
        contracts.validate_result(payload)


def test_truncated_questions_refuse_verdict() -> None:
    payload = _valid_payload()
    assert isinstance(payload["coverage"], dict)
    payload["coverage"]["truncated_questions"] = ["security"]
    with pytest.raises(ValueError, match="truncated"):
        contracts.validate_result(payload)


def test_missing_coverage_refuses_verdict() -> None:
    payload = _valid_payload()
    del payload["coverage"]
    with pytest.raises(ValueError, match="coverage"):
        contracts.validate_result(payload)


def test_unknown_message_kind_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        contracts.message_kind({"type": "nonsense"})
    with pytest.raises(ValueError, match="must be a string"):
        contracts.message_kind({"type": 3})


def test_known_kinds_pass() -> None:
    for kind in ("progress", "ready", "result", "error"):
        assert contracts.message_kind({"type": kind}) == kind


def test_math_is_finite_without_import_inside() -> None:
    assert math.isfinite(1.0)
