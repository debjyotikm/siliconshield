"""GUI entry point and file helpers: lightweight, no GPU."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

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
