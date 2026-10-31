"""Assessment invalidation: stale results must never look current."""

import json
import os
from pathlib import Path

import pytest


def _build_window(tmp_path: Path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from siliconshield.gui import registry as registry_module
    from siliconshield.gui.application import _build_window

    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "model.safetensors").write_bytes(b"x")
    python = tmp_path / "pybin"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    laya_root = tmp_path / "laya"
    (laya_root / "laya").mkdir(parents=True, exist_ok=True)
    (laya_root / "laya" / "__init__.py").write_text("")
    second = tmp_path / "ckpt2"
    second.mkdir(exist_ok=True)
    (second / "model.safetensors").write_bytes(b"y")
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
                    },
                    {
                        "id": "fixture-cpu",
                        "display_name": "Fixture CPU",
                        "checkpoint": str(second),
                        "python": str(python),
                        "laya_root": str(laya_root),
                        "expected_files": {"model.safetensors": "b" * 64},
                        "device": "cpu",
                    },
                ],
            }
        )
    )
    app = QApplication.instance() or QApplication([])
    window = _build_window(registry_module, str(registry), 180, 60)
    return app, window


def _widgets(window):
    from PySide6.QtWidgets import QComboBox, QLabel, QPlainTextEdit, QPushButton

    def one(cls, name):
        found = window.findChild(cls, name)
        assert found is not None, name
        return found

    return {
        "demo_box": one(QComboBox, "demo_box"),
        "demo_load": one(QPushButton, "demo_load_btn"),
        "req": one(QPlainTextEdit, "req_edit"),
        "assume": one(QPlainTextEdit, "assume_edit"),
        "preview": one(QPlainTextEdit, "rtl_preview"),
        "assessment": one(QLabel, "assessment_label"),
        "scores": one(QLabel, "scores_label"),
        "meta": one(QLabel, "meta_label"),
        "save": one(QPushButton, "save_btn"),
        "run": one(QPushButton, "run_btn"),
        "model_box": one(QComboBox, "model_box"),
        "device_box": one(QComboBox, "device_box"),
    }


def _fake_old_result(widgets) -> None:
    widgets["assessment"].setText("Assessment: Violates requirement")
    widgets["scores"].setText(
        "Satisfies requirement 0.100 | Violates requirement 0.800 | "
        "Insufficient context 0.100"
    )
    widgets["meta"].setText(
        "Analysis time: 1.0s  Device: cuda:0  Model: Fixture"
    )
    widgets["save"].setEnabled(True)


def test_loading_new_demo_clears_scores_and_meta(tmp_path: Path) -> None:
    pytest.importorskip("PySide6.QtWidgets")
    _app, window = _build_window(tmp_path)
    try:
        widgets = _widgets(window)
        widgets["demo_box"].setCurrentIndex(0)
        widgets["demo_load"].click()
        first_preview = widgets["preview"].toPlainText()
        assert first_preview.strip()
        _fake_old_result(widgets)
        assert widgets["save"].isEnabled()
        last_index = widgets["demo_box"].count() - 1
        widgets["demo_box"].setCurrentIndex(last_index)
        widgets["demo_load"].click()
        assert widgets["preview"].toPlainText().strip()
        assert (
            widgets["preview"].toPlainText() != first_preview
            or widgets["demo_box"].count() == 1
        )
        assert widgets["assessment"].text() == "Assessment: --"
        assert "0.800" not in widgets["scores"].text()
        assert "--" in widgets["scores"].text()
        assert widgets["meta"].text().startswith("Analysis time: --")
        assert not widgets["save"].isEnabled()
    finally:
        window.close()


def test_editing_requirement_clears_scores(tmp_path: Path) -> None:
    pytest.importorskip("PySide6.QtWidgets")
    _app, window = _build_window(tmp_path)
    try:
        widgets = _widgets(window)
        widgets["demo_box"].setCurrentIndex(0)
        widgets["demo_load"].click()
        _fake_old_result(widgets)
        widgets["req"].setPlainText(widgets["req"].toPlainText() + " more")
        assert widgets["assessment"].text().startswith("Assessment: --")
        assert "0.800" not in widgets["scores"].text()
        assert not widgets["save"].isEnabled()
    finally:
        window.close()


def test_browsing_new_file_clears_scores(tmp_path: Path) -> None:
    pytest.importorskip("PySide6.QtWidgets")
    _app, window = _build_window(tmp_path)
    try:
        widgets = _widgets(window)
        widgets["demo_box"].setCurrentIndex(0)
        widgets["demo_load"].click()
        _fake_old_result(widgets)
        first = tmp_path / "first.sv"
        first.write_text("module first; endmodule")
        second = tmp_path / "second.sv"
        second.write_text("module second; endmodule")
        from PySide6.QtWidgets import QLineEdit

        file_edit = window.findChild(QLineEdit, "file_edit")
        assert file_edit is not None
        file_edit.setText(str(first))
        # Simulate the preview path directly through the demo loader
        # then switch files: browsing must invalidate.
        file_edit.setText(str(second))
        # Drive the real preview handler via editingFinished signal.
        file_edit.editingFinished.emit()
        assert widgets["assessment"].text().startswith("Assessment: --")
        assert "0.800" not in widgets["scores"].text()
        assert not widgets["save"].isEnabled()
    finally:
        window.close()


def test_switching_model_or_device_clears_scores(tmp_path: Path) -> None:
    pytest.importorskip("PySide6.QtWidgets")
    _app, window = _build_window(tmp_path)
    try:
        widgets = _widgets(window)
        widgets["demo_box"].setCurrentIndex(0)
        widgets["demo_load"].click()
        _fake_old_result(widgets)
        # Simulate a loaded model by switching selection: any model or
        # device change after a result must invalidate the display.
        initial_model = widgets["model_box"].currentIndex()
        widgets["model_box"].setCurrentIndex(1 - initial_model)
        assert widgets["assessment"].text().startswith("Assessment: --")
        assert "0.800" not in widgets["scores"].text()
        assert not widgets["save"].isEnabled()
        _fake_old_result(widgets)
        initial_device = widgets["device_box"].currentIndex()
        widgets["device_box"].setCurrentIndex(1 - initial_device)
        assert widgets["assessment"].text().startswith("Assessment: --")
        assert "0.800" not in widgets["scores"].text()
    finally:
        window.close()
