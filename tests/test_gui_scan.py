"""Scan helpers: deterministic windows, validation, counting."""

import pytest

from siliconshield.gui.scan import (
    ScanQueue,
    count_summary,
    describe_excerpt_scope,
    split_scan_excerpts,
    split_scan_windows,
    validate_scan_selection,
)


def test_split_windows_overlap_and_bounds() -> None:
    text = "\n".join("line %d" % i for i in range(200))
    windows = split_scan_windows(text, window_lines=60, overlap_lines=15)
    assert len(windows) == 5
    assert windows[0]["start_line"] == 1
    assert windows[1]["start_line"] == 46
    assert windows[-1]["end_line"] == 200
    assert all(not w["truncated"] for w in windows)


def test_split_windows_marks_overlong() -> None:
    text = "x" * 100 + "\n" + "y\n" * 5
    windows = split_scan_windows(text, window_lines=10, max_chars=50)
    assert windows[0]["truncated"] is True
    assert windows[0]["text"].endswith("[window truncated]...\n")


def test_split_windows_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        split_scan_windows("a\n", window_lines=0)
    with pytest.raises(ValueError):
        split_scan_windows("a\n", window_lines=10, overlap_lines=10)
    with pytest.raises(ValueError):
        split_scan_windows("", window_lines=10)


def test_split_excerpts_prefers_complete_modules() -> None:
    text = (
        "// module ignored_comment;\n"
        "module alpha;\n"
        '  string note = "endmodule is text";\n'
        "endmodule\n"
        "/* module ignored_block; endmodule */\n"
        "module beta;\n"
        "endmodule\n"
    )
    excerpts = split_scan_excerpts(text)
    assert [excerpt["module"] for excerpt in excerpts] == ["alpha", "beta"]
    assert [excerpt["start_line"] for excerpt in excerpts] == [2, 6]
    assert all(excerpt["kind"] == "module" for excerpt in excerpts)
    assert all(excerpt["complete"] is True for excerpt in excerpts)
    assert excerpts[0]["text"].startswith("module alpha")
    assert excerpts[0]["text"].endswith("endmodule\n")


def test_split_excerpts_falls_back_for_unclosed_module() -> None:
    text = "module unfinished;\nlogic state;\n"
    excerpts = split_scan_excerpts(text, window_lines=2, overlap_lines=0)
    assert all(excerpt["kind"] == "line_window" for excerpt in excerpts)
    assert all(excerpt["complete"] is False for excerpt in excerpts)


def test_split_excerpts_bounds_oversized_module() -> None:
    text = "module large;\n" + ("logic state;\n" * 8) + "endmodule\n"
    excerpts = split_scan_excerpts(
        text,
        window_lines=4,
        overlap_lines=1,
        max_chars=45,
    )
    assert len(excerpts) > 1
    assert all(excerpt["kind"] == "module_window" for excerpt in excerpts)
    assert all(excerpt["module"] == "large" for excerpt in excerpts)
    assert all(excerpt["complete"] is False for excerpt in excerpts)
    assert excerpts[0]["start_line"] == 1


def test_describe_excerpt_scope_distinguishes_fallback() -> None:
    assert (
        describe_excerpt_scope({"complete": True, "module": "control"})
        == "complete module control"
    )
    assert (
        describe_excerpt_scope({"complete": False, "module": "control"})
        == "incomplete line window in module control"
    )
    assert describe_excerpt_scope({"complete": False}) == (
        "incomplete line window"
    )


def test_validate_selection_strict() -> None:
    good = [
        {"index": 2, "reason": "writes cfg"},
        {"index": 0, "reason": "reset"},
    ]
    assert [e["index"] for e in validate_scan_selection(good, 3)] == [0, 2]
    for bad in (
        [],
        [{"index": 9, "reason": "x"}],
        [{"index": 0, "reason": " "}],
        [{"index": 0, "reason": "a"}, {"index": 0, "reason": "b"}],
        [{"index": True, "reason": "x"}],
        ["nope"],
    ):
        with pytest.raises(ValueError):
            validate_scan_selection(bad, 3)


def test_scan_queue_counts_deterministically() -> None:
    queue = ScanQueue([0, 1, 2])
    assert queue.remaining == 3
    assert queue.advance("violates_requirement") == 1
    assert queue.advance("satisfies_requirement") == 2
    assert queue.advance("violates_requirement") is None
    assert queue.judged == 3
    assert queue.violations == 2
    assert queue.advance("violates_requirement") is None
    assert "2 of 3" in count_summary(queue, 0)
    assert "1 errors" in count_summary(queue, 1)
    assert "Potentially flagged" in count_summary(queue, 0)
    assert "not independently verified" in count_summary(queue, 0)
    assert "Violations:" not in count_summary(queue, 0)


def test_muse_select_uses_tool_call() -> None:
    from siliconshield.gui import muse_scan

    calls: list[dict] = []

    class FakeClient:
        def complete(self, messages, tools=()):
            calls.append({"messages": messages, "tools": tools})
            assert tools[0]["function"]["name"] == "scan.select_excerpts"
            return {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "scan.select_excerpts",
                            "arguments": {
                                "excerpts": [
                                    {"index": 1, "reason": "dbg path"},
                                ]
                            },
                        },
                    }
                ]
            }

    windows = [
        {"index": 0, "start_line": 1, "end_line": 10, "text": "a"},
        {"index": 1, "start_line": 11, "end_line": 20, "text": "b"},
    ]
    selected = muse_scan.select_excerpts(
        windows, "req", "asm", client_factory=lambda: FakeClient()
    )
    assert selected == [{"index": 1, "reason": "dbg path"}]
    assert calls and calls[0]["messages"][0]["role"] == "system"


def test_muse_select_rejects_missing_call() -> None:
    from siliconshield.gui import muse_scan

    class SilentClient:
        def complete(self, messages, tools=()):
            return {"tool_calls": []}

    with pytest.raises(ValueError, match="no scan.select_excerpts"):
        muse_scan.select_excerpts(
            [{"index": 0, "start_line": 1, "end_line": 2, "text": "a"}],
            "req",
            "asm",
            client_factory=lambda: SilentClient(),
        )


def test_muse_audit_dir_is_outside_repo(tmp_path) -> None:
    from siliconshield.gui.muse_scan import default_muse_audit_dir

    path = default_muse_audit_dir()
    assert path.name == "muse-audit"
    assert "siliconshield-gui" in str(path)


def test_read_scan_source_bounds(tmp_path) -> None:
    from siliconshield.gui.scan import read_scan_source

    good = tmp_path / "big.sv"
    good.write_text("module a; endmodule\n")
    text, snap = read_scan_source(str(good))
    assert "module a" in text
    assert snap["bytes"] == len(good.read_bytes())
    bad = tmp_path / "big.txt"
    bad.write_text("x")
    import pytest as _pytest

    with _pytest.raises(ValueError, match="suffix"):
        read_scan_source(str(bad))
    empty = tmp_path / "empty.sv"
    empty.write_text("")
    with _pytest.raises(ValueError, match="empty"):
        read_scan_source(str(empty))


def test_scan_tab_exists_offscreen(tmp_path) -> None:
    import pytest as _pytest

    _pytest.importorskip("PySide6.QtWidgets")
    import json
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QTabWidget

    from siliconshield.gui import registry as registry_module
    from siliconshield.gui.application import _build_window

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
    app = QApplication.instance() or QApplication([])
    window = _build_window(registry_module, str(registry), 180, 60)
    try:
        tabs = window.findChild(QTabWidget, "analysis_tabs")
        assert tabs is not None
        assert tabs.count() == 2
        assert tabs.tabText(0) == "Single file"
        assert tabs.tabText(1) == "Large file scan"
        from PySide6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QHeaderView,
            QPushButton,
            QSplitter,
            QTableWidget,
        )

        splitter = window.findChild(QSplitter, "main_splitter")
        assert splitter is not None
        assert splitter.count() == 2
        scan_splitter = window.findChild(QSplitter, "scan_splitter")
        assert scan_splitter is not None
        assert scan_splitter.count() == 3
        assert window.minimumWidth() >= 900
        assert window.findChild(QPushButton, "scan_btn") is not None
        assert window.findChild(QPushButton, "scan_stop_btn") is not None
        assert window.findChild(QTableWidget, "scan_table") is not None
        assert window.findChild(QPushButton, "save_scan_btn") is not None
        assert window.findChild(QCheckBox, "muse_check") is not None
        assert window.findChild(QComboBox, "scan_demo_box") is not None
        # Scan requires a loaded model: disabled while unloaded.
        assert not window.findChild(QPushButton, "scan_btn").isEnabled()
        table = window.findChild(QTableWidget, "scan_table")
        assert table is not None
        header = table.horizontalHeader()
        assert header.sectionResizeMode(0) == QHeaderView.Interactive
        assert header.stretchLastSection() is True
        assert table.columnWidth(2) >= 150
    finally:
        window.close()


def test_scan_demo_loads_requirement_offscreen(tmp_path) -> None:
    import pytest as _pytest

    _pytest.importorskip("PySide6.QtWidgets")
    import json
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QLabel,
        QPlainTextEdit,
        QPushButton,
    )

    from siliconshield.gui import registry as registry_module
    from siliconshield.gui.application import _build_window

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
                        "device": "cpu",
                    }
                ],
            }
        )
    )
    app = QApplication.instance() or QApplication([])
    window = _build_window(registry_module, str(registry), 180, 60)
    try:
        demo_box = window.findChild(QComboBox, "scan_demo_box")
        assert demo_box is not None
        assert demo_box.count() == 1
        assert demo_box.itemData(0) == "peripheral-config-bank"
        window.findChild(QPushButton, "scan_demo_load_btn").click()
        req = window.findChild(QPlainTextEdit, "scan_req_edit")
        assert req is not None
        assert "cfg_reg" in req.toPlainText()
        status = window.findChild(QLabel, "scan_status_label")
        assert status is not None
        assert "Peripheral configuration bank" in status.text()
        # Scan demos live on their own tab: the single-file verdict is untouched.
        assessment = window.findChild(QLabel, "assessment_label")
        assert assessment is not None
        assert assessment.text() == "Assessment: --"
    finally:
        window.close()


def test_reloading_scan_demo_clears_table(tmp_path) -> None:
    import pytest as _pytest

    _pytest.importorskip("PySide6.QtWidgets")
    import json
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QLabel,
        QPushButton,
        QTableWidget,
    )

    from siliconshield.gui import registry as registry_module
    from siliconshield.gui.application import _build_window

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
                        "device": "cpu",
                    }
                ],
            }
        )
    )
    app = QApplication.instance() or QApplication([])
    window = _build_window(registry_module, str(registry), 180, 60)
    try:
        demo_box = window.findChild(QComboBox, "scan_demo_box")
        assert demo_box is not None
        demo_box.setCurrentIndex(0)
        window.findChild(QPushButton, "scan_demo_load_btn").click()
        table = window.findChild(QTableWidget, "scan_table")
        assert table is not None
        # Simulate a finished scan's stale rows.
        table.setRowCount(2)
        summary = window.findChild(QLabel, "scan_summary_label")
        assert summary is not None
        summary.setText("Potentially flagged: 1 of 2 excerpts (0 errors).")
        save_btn = window.findChild(QPushButton, "save_scan_btn")
        assert save_btn is not None
        save_btn.setEnabled(True)
        # Reloading the demo must clear the tabular list.
        window.findChild(QPushButton, "scan_demo_load_btn").click()
        assert table.rowCount() == 0
        assert summary.text() == "Potentially flagged: --"
        assert not save_btn.isEnabled()
    finally:
        window.close()


def test_scan_row_click_shows_excerpt(tmp_path) -> None:
    import pytest as _pytest

    _pytest.importorskip("PySide6.QtWidgets")
    import json
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QLabel,
        QPlainTextEdit,
        QPushButton,
        QTableWidget,
    )

    from siliconshield.gui import registry as registry_module
    from siliconshield.gui.application import _build_window

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
                        "device": "cpu",
                    }
                ],
            }
        )
    )
    app = QApplication.instance() or QApplication([])
    window = _build_window(registry_module, str(registry), 180, 60)
    try:
        demo_box = window.findChild(QComboBox, "scan_demo_box")
        assert demo_box is not None
        demo_box.setCurrentIndex(0)
        window.findChild(QPushButton, "scan_demo_load_btn").click()
        preview = window.findChild(QPlainTextEdit, "scan_excerpt_preview")
        assert preview is not None
        assert preview.placeholderText() != ""
        # No rows yet: clicking the empty table must not crash or fill.
        table = window.findChild(QTableWidget, "scan_table")
        assert table is not None
        assert table.rowCount() == 0
        excerpt_label = window.findChild(QLabel, "scan_excerpt_label")
        assert excerpt_label is not None
        assert excerpt_label.text() == "Selected excerpt: --"
    finally:
        window.close()
