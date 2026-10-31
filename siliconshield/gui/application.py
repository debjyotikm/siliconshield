"""Qt window for SiliconShield RTL security analysis.

The window owns no model state directly: a Qt-free
:class:`siliconshield.gui.controller.Controller` drives loading and
analysis over an owned worker process. File reads are bounded to keep
the event loop responsive; checkpoint hashing and inference happen in
the worker process.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from siliconshield.app.laya_assessor import state_text

WINDOW_TITLE = "SiliconShield — RTL Security Analysis"
SCOPE_NOTE = "Model assessment of the supplied RTL and requirement. No verification tool was run."
CUSTOM_SCOPE_NOTE = (
    "Model assessment only. This custom input has no "
    "requirement-bound check, so the verdict is not independently "
    "verified and must not be reported as a confirmed finding."
)


def scope_note_for(snapshot: dict[str, Any] | None) -> str:
    """Return the honesty banner for a loaded design snapshot.

    Authored demos at least have requirement-bound checks in the CLI
    investigation workflow; custom files do not, so they carry the
    stronger warning.
    """
    path = (snapshot or {}).get("path", "")
    if isinstance(path, str) and path.startswith("demo:"):
        return SCOPE_NOTE
    return CUSTOM_SCOPE_NOTE


MAX_DESIGN_BYTES = 1024 * 1024
MAX_REQUIREMENT_CHARS = 20000
MAX_ASSUMPTIONS_CHARS = 20000
SUPPORTED_SUFFIXES = (".v", ".sv")

LABELS = (
    "satisfies_requirement",
    "violates_requirement",
    "insufficient_context",
)
LABEL_TITLES = {
    "satisfies_requirement": "Satisfies requirement",
    "violates_requirement": "Violates requirement",
    "insufficient_context": "Insufficient context",
}


def read_design_file(path_str: str) -> tuple[str, dict[str, Any]]:
    """Read one Verilog file with a bounded, explicit snapshot.

    Returns the source text and a snapshot record with sha256, bytes,
    and path. Raises ValueError with an actionable message for
    missing files, directories, unreadable content, unsupported
    suffixes, empty content, oversized inputs, or bad encoding.
    """
    if not path_str.strip():
        raise ValueError("Choose a design file to preview.")
    path = Path(path_str).expanduser()
    if not path.exists():
        raise ValueError("Design file does not exist: %s" % path)
    if not path.is_file():
        raise ValueError("Design path is not a file: %s" % path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            "Unsupported design suffix %r; use .v or .sv." % path.suffix
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError("Cannot read design file: %s" % exc)
    if len(raw) == 0:
        raise ValueError("Design file is empty: %s" % path)
    if len(raw) > MAX_DESIGN_BYTES:
        raise ValueError(
            "Design file exceeds %d bytes; supply a smaller "
            "self-contained module." % MAX_DESIGN_BYTES
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Design file is not UTF-8: %s" % exc)
    if not text.strip():
        raise ValueError("Design file holds no Verilog text.")
    snapshot = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    return text, snapshot


def build_state(requirement: str, assumptions: str, rtl: str) -> str:
    """Build the exact model input format shared with the CLI worker."""
    return str(state_text(requirement, assumptions, rtl))


def snapshot_key(state: str) -> str:
    """Return a stable key binding a shown result to its inputs."""
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def check_inputs(
    requirement: str,
    rtl_text: str,
    models_ready: bool,
    assumptions: str = "",
) -> str | None:
    """Return None when Run is allowed, else a human reason."""
    if not models_ready:
        return "Configure a model registry to enable analysis."
    if not requirement.strip():
        return "Enter a security requirement."
    if len(requirement) > MAX_REQUIREMENT_CHARS:
        return (
            "Requirement exceeds %d characters; shorten it."
            % MAX_REQUIREMENT_CHARS
        )
    if len(assumptions) > MAX_ASSUMPTIONS_CHARS:
        return (
            "Assumptions exceed %d characters; shorten them."
            % MAX_ASSUMPTIONS_CHARS
        )
    if not rtl_text.strip():
        return "Choose a design file with Verilog text."
    return None


def available_load_devices(registry_device: str) -> list[str]:
    """Return GUI device options with the registry default first.

    Offers the registry device plus the other family (cpu vs cuda:0)
    so one registry entry can load on either target without editing
    files. No GPU probing happens here.
    """
    if registry_device == "cpu":
        return ["cpu", "cuda:0"]
    if registry_device.startswith("cuda:"):
        return [registry_device, "cpu"]
    return ["cuda:0", "cpu"]


def resolve_load_device(registry_device: str, gui_device: str | None) -> str:
    """Return the device to load on, preferring the GUI selection."""
    if gui_device == "cpu" or (
        gui_device is not None and gui_device.startswith("cuda:")
    ):
        return gui_device
    return registry_device


def two_line_height(edit: Any) -> int:
    """Return a fixed widget height showing about two text lines.

    Keeps requirement/assumption inputs compact so result tables keep
    the room. Falls back to a small default when metrics fail.
    """
    try:
        spacing = int(edit.fontMetrics().lineSpacing())
        margin = int(edit.document().documentMargin())
        frame = int(edit.frameWidth())
        return spacing * 2 + margin * 2 + frame * 2 + 4
    except RuntimeError:
        return 52


def format_scan_excerpt(entry: dict[str, Any]) -> tuple[str, str]:
    """Format a scan row's excerpt title and body for the preview pane."""
    verdict = str(entry.get("label", entry.get("verdict", "--")))
    title = "Excerpt %s · lines %s · %s" % (
        entry.get("index", "?"),
        entry.get("lines", "--"),
        verdict,
    )
    return title, str(entry.get("text", ""))


def needs_preview_refresh(pending_path: str, snapshot: dict | None) -> bool:
    """Return True when the typed path differs from the shown preview."""
    pending = pending_path.strip()
    if not pending:
        return False
    if not snapshot:
        return True
    return pending != str(snapshot.get("path", ""))


def is_result_stale(current_key: str | None, frozen_key: str | None) -> bool:
    """Return True when inputs changed after analysis started."""
    if current_key is None or frozen_key is None:
        return True
    return current_key != frozen_key


def should_enable_load(
    busy: bool,
    has_models: bool,
    already_loaded: bool,
    worker_running: bool,
) -> bool:
    """Return True when Load may start a new worker.

    Load is disabled while busy, with no registry, when the selected
    model and device already match the loaded worker, or while a
    previous worker is still exiting (callers must unload first to
    avoid orphaned GPU processes).
    """
    return not busy and has_models and not already_loaded and not worker_running


def run_window(
    models_path: str | None,
    load_timeout_seconds: int = 180,
    analyze_timeout_seconds: int = 60,
) -> None:
    """Open the desktop window; blocks until the user quits.

    Args:
        models_path: Local registry file, or None to open with a
            configuration message and Run disabled.
        load_timeout_seconds: Bounded wait for model load.
        analyze_timeout_seconds: Bounded wait for one analysis.
    """
    # Local imports keep module import Qt-free for tooling.
    from PySide6.QtWidgets import QApplication

    from siliconshield.gui import registry as registry_module

    app = QApplication([])
    window = _build_window(
        registry_module,
        models_path,
        load_timeout_seconds,
        analyze_timeout_seconds,
    )
    window.show()
    app.exec()


def _build_window(
    registry_module: Any,
    models_path: str | None,
    load_timeout_seconds: int,
    analyze_timeout_seconds: int,
) -> Any:
    """Create the main window after QApplication exists."""
    # Local Qt imports keep this module importable without Qt.
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    from siliconshield.gui.controller import (
        ANALYZING,
        ERROR,
        LOADING,
        READY,
        UNLOADED,
        UNLOADING,
        Controller,
    )

    models: list[dict[str, Any]] = []
    models_error: str | None = None
    if models_path is None:
        models_error = (
            "No model registry configured. Relaunch with "
            "python -m siliconshield.gui --checkpoint CHECKPOINT "
            "--python PYTHON --laya-root LAYA_ROOT."
        )
    else:
        try:
            loaded = registry_module.load_registry(models_path)
            models = [dict(entry) for entry in loaded]
        except ValueError as exc:
            models_error = str(exc)

    class _GuiWindow(QMainWindow):
        """Main window with owned-worker cleanup on close."""

        def closeEvent(self, event: Any) -> None:  # noqa: N802
            _shutdown_owned_worker()
            event.accept()

    win = _GuiWindow()
    win.setWindowTitle(WINDOW_TITLE)
    win.resize(1100, 800)
    win.setMinimumSize(900, 650)
    root = QWidget()
    win.setCentralWidget(root)
    layout = QVBoxLayout(root)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)

    top_row = QHBoxLayout()
    top_row.setSpacing(8)
    model_label = QLabel("Decision model:")
    model_box = QComboBox()
    model_box.setObjectName("model_box")
    for entry in models:
        model_box.addItem(str(entry.get("display_name")), entry.get("id"))
    if models:
        model_box.setCurrentIndex(0)
    device_label = QLabel("Device:")
    device_box = QComboBox()
    device_box.setObjectName("device_box")
    device_box.setToolTip("CPU needs no GPU; GPU uses CUDA.")
    load_btn = QPushButton("Load")
    load_btn.setObjectName("load_btn")
    unload_btn = QPushButton("Unload")
    unload_btn.setObjectName("unload_btn")
    unload_btn.setEnabled(False)
    top_row.addWidget(model_label)
    top_row.addWidget(model_box, 1)
    top_row.addWidget(device_label)
    top_row.addWidget(device_box)
    top_row.addWidget(load_btn)
    top_row.addWidget(unload_btn)
    layout.addLayout(top_row)

    status_label = QLabel("Status: Not loaded")
    status_label.setObjectName("status_label")
    if models_error:
        status_label.setText("Status: %s" % models_error)
    layout.addWidget(status_label)

    tabs = QTabWidget()
    tabs.setObjectName("analysis_tabs")
    tabs.setTabsClosable(False)

    single_tab = QWidget()
    single_tab.setObjectName("single_tab")
    single_layout = QVBoxLayout(single_tab)
    single_layout.setContentsMargins(12, 12, 12, 12)
    single_layout.setSpacing(8)

    file_row = QHBoxLayout()
    file_row.setSpacing(8)
    file_label = QLabel("Design file:")
    file_edit = QLineEdit()
    file_edit.setObjectName("file_edit")
    file_edit.setPlaceholderText("/path/to/design.sv")
    browse_btn = QPushButton("Browse")
    browse_btn.setObjectName("browse_btn")
    file_row.addWidget(file_label)
    file_row.addWidget(file_edit, 1)
    file_row.addWidget(browse_btn)
    single_layout.addLayout(file_row)

    demo_row = QHBoxLayout()
    demo_row.setSpacing(8)
    demo_label = QLabel("Demo:")
    demo_box = QComboBox()
    demo_box.setObjectName("demo_box")
    demo_box.setToolTip("Authored synthetic cases (never training data).")
    demo_load_btn = QPushButton("Load Demo")
    demo_load_btn.setObjectName("demo_load_btn")
    demo_row.addWidget(demo_label)
    demo_row.addWidget(demo_box, 1)
    demo_row.addWidget(demo_load_btn)
    single_layout.addLayout(demo_row)

    rtl_preview = QPlainTextEdit()
    rtl_preview.setObjectName("rtl_preview")
    rtl_preview.setReadOnly(True)
    mono = QFont("Monospace")
    mono.setStyleHint(QFont.TypeWriter)
    rtl_preview.setFont(mono)
    rtl_preview.setPlaceholderText("Read-only source preview.")
    rtl_header = QLabel("RTL preview (single file, no includes):")
    rtl_header_font = QFont(rtl_header.font())
    rtl_header_font.setBold(True)
    rtl_header.setFont(rtl_header_font)
    single_layout.addWidget(rtl_header)
    single_layout.addWidget(rtl_preview, 1)

    req_header = QLabel("Security requirement:")
    req_header_font = QFont(req_header.font())
    req_header_font.setBold(True)
    req_header.setFont(req_header_font)
    single_layout.addWidget(req_header)
    req_edit = QPlainTextEdit()
    req_edit.setObjectName("req_edit")
    req_edit.setPlaceholderText("What must this design satisfy?")
    req_edit.setFixedHeight(two_line_height(req_edit))
    single_layout.addWidget(req_edit)

    assume_header = QLabel("Assumptions (optional):")
    assume_header_font = QFont(assume_header.font())
    assume_header_font.setBold(True)
    assume_header.setFont(assume_header_font)
    single_layout.addWidget(assume_header)
    assume_edit = QPlainTextEdit()
    assume_edit.setObjectName("assume_edit")
    assume_edit.setPlaceholderText(
        "Reset, clock, or operating assumptions supplied by you."
    )
    assume_edit.setFixedHeight(two_line_height(assume_edit))
    single_layout.addWidget(assume_edit)

    run_row = QHBoxLayout()
    run_row.setSpacing(8)
    run_btn = QPushButton("Run Analysis")
    run_btn.setObjectName("run_btn")
    run_btn.setEnabled(False)
    run_btn.setDefault(True)
    run_btn.setAutoDefault(True)
    cancel_btn = QPushButton("Cancel")
    cancel_btn.setObjectName("cancel_btn")
    cancel_btn.setEnabled(False)
    run_row.addWidget(run_btn)
    run_row.addWidget(cancel_btn)
    run_row.addStretch(1)
    single_layout.addLayout(run_row)

    assessment_label = QLabel("Assessment: --")
    assessment_label.setObjectName("assessment_label")
    assessment_font = QFont(assessment_label.font())
    assessment_font.setBold(True)
    assessment_label.setFont(assessment_font)
    single_layout.addWidget(assessment_label)
    scores_label = QLabel(
        "Satisfies requirement -- | Violates requirement -- | "
        "Insufficient context --"
    )
    scores_label.setObjectName("scores_label")
    scores_label.setWordWrap(True)
    single_layout.addWidget(scores_label)
    meta_label = QLabel("Analysis time: --  Device: --  Model: --")
    meta_label.setObjectName("meta_label")
    single_layout.addWidget(meta_label)
    tabs.addTab(single_tab, "Single file")

    scan_tab = QWidget()
    scan_tab.setObjectName("scan_tab")
    scan_layout = QVBoxLayout(scan_tab)
    scan_layout.setContentsMargins(12, 12, 12, 12)
    scan_layout.setSpacing(8)

    scan_inputs_pane = QWidget()
    scan_inputs_pane.setObjectName("scan_inputs_pane")
    scan_inputs_layout = QVBoxLayout(scan_inputs_pane)
    scan_inputs_layout.setContentsMargins(0, 0, 0, 0)
    scan_inputs_layout.setSpacing(8)

    scan_demo_row = QHBoxLayout()
    scan_demo_row.setSpacing(8)
    scan_demo_label = QLabel("Demo:")
    scan_demo_box = QComboBox()
    scan_demo_box.setObjectName("scan_demo_box")
    scan_demo_box.setToolTip(
        "Authored synthetic large files (never training data)."
    )
    scan_demo_load_btn = QPushButton("Load Demo")
    scan_demo_load_btn.setObjectName("scan_demo_load_btn")
    scan_demo_row.addWidget(scan_demo_label)
    scan_demo_row.addWidget(scan_demo_box, 1)
    scan_demo_row.addWidget(scan_demo_load_btn)
    scan_inputs_layout.addLayout(scan_demo_row)

    scan_file_row = QHBoxLayout()
    scan_file_row.setSpacing(8)
    scan_file_label = QLabel("Large design file:")
    scan_file_edit = QLineEdit()
    scan_file_edit.setObjectName("scan_file_edit")
    scan_file_edit.setPlaceholderText("/path/to/big-design.sv")
    scan_browse_btn = QPushButton("Browse")
    scan_browse_btn.setObjectName("scan_browse_btn")
    scan_file_row.addWidget(scan_file_label)
    scan_file_row.addWidget(scan_file_edit, 1)
    scan_file_row.addWidget(scan_browse_btn)
    scan_inputs_layout.addLayout(scan_file_row)

    scan_inputs_layout.addWidget(QLabel("Security requirement:"))
    scan_req_edit = QPlainTextEdit()
    scan_req_edit.setObjectName("scan_req_edit")
    scan_req_edit.setPlaceholderText("What must this design satisfy?")
    scan_req_edit.setFixedHeight(two_line_height(scan_req_edit))
    scan_inputs_layout.addWidget(scan_req_edit)

    scan_inputs_layout.addWidget(QLabel("Assumptions (optional):"))
    scan_assume_edit = QPlainTextEdit()
    scan_assume_edit.setObjectName("scan_assume_edit")
    scan_assume_edit.setPlaceholderText(
        "Reset, clock, or operating assumptions supplied by you."
    )
    scan_assume_edit.setFixedHeight(two_line_height(scan_assume_edit))
    scan_inputs_layout.addWidget(scan_assume_edit)

    scan_options_row = QHBoxLayout()
    muse_check = QCheckBox("Use Muse to select excerpts")
    muse_check.setObjectName("muse_check")
    muse_check.setToolTip(
        "Muse Spark 1.3 Contributor picks relevant excerpts; "
        "unchecked means complete modules where available, with "
        "bounded line-window fallback."
    )
    muse_check.setChecked(True)
    scan_options_row.addWidget(muse_check)
    scan_options_row.addStretch(1)
    scan_inputs_layout.addLayout(scan_options_row)

    scan_btn_row = QHBoxLayout()
    scan_btn_row.setSpacing(8)
    scan_btn = QPushButton("Scan File")
    scan_btn.setObjectName("scan_btn")
    scan_btn.setEnabled(False)
    scan_btn.setDefault(True)
    scan_btn.setAutoDefault(True)
    scan_stop_btn = QPushButton("Stop")
    scan_stop_btn.setObjectName("scan_stop_btn")
    scan_stop_btn.setEnabled(False)
    scan_btn_row.addWidget(scan_btn)
    scan_btn_row.addWidget(scan_stop_btn)
    scan_btn_row.addStretch(1)
    scan_inputs_layout.addLayout(scan_btn_row)

    scan_progress = QProgressBar()
    scan_progress.setObjectName("scan_progress")
    scan_progress.setRange(0, 1)
    scan_progress.setValue(0)
    scan_inputs_layout.addWidget(scan_progress)
    scan_status_label = QLabel("No scan yet.")
    scan_status_label.setObjectName("scan_status_label")
    scan_status_label.setWordWrap(True)
    scan_inputs_layout.addWidget(scan_status_label)

    scan_splitter = QSplitter(Qt.Vertical)
    scan_splitter.setObjectName("scan_splitter")
    scan_splitter.setChildrenCollapsible(False)
    scan_layout.addWidget(scan_splitter, 1)
    scan_splitter.addWidget(scan_inputs_pane)
    scan_splitter.setStretchFactor(0, 1)

    scan_table = QTableWidget()
    scan_table.setObjectName("scan_table")
    scan_table.setColumnCount(9)
    scan_table.setHorizontalHeaderLabels(
        [
            "Excerpt",
            "Lines",
            "Verdict",
            "satisfies",
            "violates",
            "insuff.",
            "Time",
            "Muse reason",
            "Verified",
        ]
    )
    scan_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    scan_header = scan_table.horizontalHeader()
    scan_header.setSectionResizeMode(QHeaderView.Interactive)
    scan_header.setStretchLastSection(True)
    for _col, _width in (
        (0, 70),
        (1, 90),
        (2, 170),
        (3, 85),
        (4, 85),
        (5, 85),
        (6, 70),
        (8, 110),
    ):
        scan_table.setColumnWidth(_col, _width)
    scan_splitter.addWidget(scan_table)
    scan_excerpt_pane = QWidget()
    scan_excerpt_pane.setObjectName("scan_excerpt_pane")
    scan_excerpt_layout = QVBoxLayout(scan_excerpt_pane)
    scan_excerpt_layout.setContentsMargins(0, 0, 0, 0)
    scan_excerpt_label = QLabel("Selected excerpt: --")
    scan_excerpt_label.setObjectName("scan_excerpt_label")
    scan_excerpt_layout.addWidget(scan_excerpt_label)
    scan_excerpt_preview = QPlainTextEdit()
    scan_excerpt_preview.setObjectName("scan_excerpt_preview")
    scan_excerpt_preview.setReadOnly(True)
    scan_excerpt_preview.setPlaceholderText(
        "Click a table row to view its design excerpt."
    )
    scan_excerpt_preview.setFont(mono)
    scan_excerpt_layout.addWidget(scan_excerpt_preview, 1)
    scan_splitter.addWidget(scan_excerpt_pane)
    scan_splitter.setStretchFactor(0, 1)
    scan_splitter.setStretchFactor(1, 2)
    scan_splitter.setStretchFactor(2, 1)
    scan_splitter.setSizes([260, 340, 160])

    scan_summary_label = QLabel("Potentially flagged: --")
    scan_summary_label.setObjectName("scan_summary_label")
    scan_layout.addWidget(scan_summary_label)

    save_scan_btn = QPushButton("Save Scan")
    save_scan_btn.setObjectName("save_scan_btn")
    save_scan_btn.setEnabled(False)
    scan_layout.addWidget(save_scan_btn)
    tabs.addTab(scan_tab, "Large file scan")

    main_splitter = QSplitter(Qt.Vertical)
    main_splitter.setObjectName("main_splitter")
    main_splitter.setChildrenCollapsible(False)
    layout.addWidget(main_splitter, 1)
    main_splitter.addWidget(tabs)

    details_pane = QWidget()
    details_pane.setObjectName("details_pane")
    details_layout = QVBoxLayout(details_pane)
    details_layout.setContentsMargins(0, 0, 0, 0)
    details = QPlainTextEdit()
    details.setObjectName("details")
    details.setReadOnly(True)
    details.setPlaceholderText("Details: hashes, runtime, usage, errors.")
    details.setMaximumBlockCount(500)
    details_layout.addWidget(QLabel("Details:"))
    details_layout.addWidget(details, 1)

    bottom_row = QHBoxLayout()
    save_btn = QPushButton("Save Result")
    save_btn.setObjectName("save_btn")
    save_btn.setEnabled(False)
    bottom_row.addWidget(save_btn)
    scope_label = QLabel(SCOPE_NOTE)
    scope_label.setWordWrap(True)
    bottom_row.addWidget(scope_label, 1)
    details_layout.addLayout(bottom_row)
    main_splitter.addWidget(details_pane)
    main_splitter.setStretchFactor(0, 3)
    main_splitter.setStretchFactor(1, 1)

    state: dict[str, Any] = {
        "rtl_text": "",
        "rtl_snapshot": None,
        "state_text": "",
        "state_key": None,
        "last_result": None,
        "last_result_key": None,
        "load_started": 0.0,
        "analyze_started": 0.0,
        "loaded_entry": None,
        "pending_entry": None,
        "session_id": None,
        "load_timer": None,
        "analyze_timer": None,
    }
    scan: dict[str, Any] = {
        "active": False,
        "muse_running": False,
        "file_text": "",
        "file_snapshot": None,
        "requirement": "",
        "assumptions": "",
        "windows": [],
        "selected": [],
        "queue": None,
        "current": None,
        "rows": [],
        "errors": 0,
        "violations": 0,
        "muse_used": False,
        "run_id": 0,
        "excerpt_started": 0.0,
        "done": False,
        "muse_timer": None,
        "muse_done": None,
        "muse_outcome": None,
    }

    holders: dict[str, Any] = {"client": None}
    controller_holder: dict[str, Any] = {"controller": None}

    def _noop_state(_name: str) -> None:
        return None

    def _noop_progress(_stage: str, _detail: str) -> None:
        return None

    def _noop_result(_result: dict[str, Any]) -> None:
        return None

    def _noop_error(_message: str, _stage: str) -> None:
        return None

    controller = Controller(
        on_state=_noop_state,
        on_progress=_noop_progress,
        on_result=_noop_result,
        on_error=_noop_error,
        load_timeout_seconds=load_timeout_seconds,
        analyze_timeout_seconds=analyze_timeout_seconds,
    )
    controller_holder["controller"] = controller

    def set_status(text: str) -> None:
        status_label.setText("Status: %s" % text)

    def _reset_assessment_display(reason: str | None = None) -> None:
        suffix = " (%s)" % reason if reason else ""
        assessment_label.setText("Assessment: --%s" % suffix)
        scores_label.setText(
            "Satisfies requirement -- | Violates requirement -- | "
            "Insufficient context --"
        )
        meta_label.setText("Analysis time: --  Device: --  Model: --")

    def _invalidate_result(reason: str | None) -> None:
        state["last_result"] = None
        state["last_result_key"] = None
        controller.clear_result()
        save_btn.setEnabled(False)
        _reset_assessment_display(reason)

    def refresh_run_enabled() -> None:
        requirement = req_edit.toPlainText()
        rtl_text = state["rtl_text"]
        reason = check_inputs(
            requirement,
            rtl_text,
            bool(models) and not models_error,
            assume_edit.toPlainText(),
        )
        ready_ok = controller.state == READY
        loaded = state["loaded_entry"]
        matched = (
            loaded is not None
            and _selected_id() == loaded.get("id")
            and _selected_device() == loaded.get("device")
        )
        stale = controller.has_stale_result(_current_key())
        if (stale or not matched) and state["last_result"] is not None:
            _invalidate_result("inputs changed" if stale else "select and load")
        busy = controller.state in (LOADING, ANALYZING, UNLOADING)
        already_loaded = ready_ok and matched
        client = holders.get("client")
        worker_running = bool(
            client is not None and getattr(client, "running", False) is True
        )
        load_btn.setEnabled(
            should_enable_load(
                busy, bool(models), already_loaded, worker_running
            )
        )
        run_btn.setEnabled(
            reason is None and ready_ok and matched and not scan["active"]
        )
        if reason is not None and controller.state == READY and matched:
            set_status(reason)
        scan_reason = check_inputs(
            scan_req_edit.toPlainText(),
            scan["file_text"],
            bool(models) and not models_error,
            scan_assume_edit.toPlainText(),
        )
        scan_btn.setEnabled(
            scan_reason is None
            and ready_ok
            and matched
            and not busy
            and not scan["active"]
        )
        scan_stop_btn.setEnabled(bool(scan["active"]))

    def _selected_id() -> str | None:
        data = model_box.currentData()
        return str(data) if data else None

    def _selected_entry() -> dict[str, Any] | None:
        selected = _selected_id()
        if selected is None:
            return None
        return next(
            (dict(item) for item in models if item.get("id") == selected),
            None,
        )

    def _selected_device() -> str | None:
        data = device_box.currentData()
        if isinstance(data, str) and data:
            return data
        text = device_box.currentText()
        return text.strip() or None

    def _refresh_devices() -> None:
        entry = _selected_entry()
        registry_device = (
            str(entry.get("device", "cuda:0")) if entry else "cuda:0"
        )
        options = available_load_devices(registry_device)
        previous = _selected_device()
        device_box.blockSignals(True)
        try:
            device_box.clear()
            for option in options:
                label = (
                    "CPU (no GPU)" if option == "cpu" else "GPU (%s)" % option
                )
                device_box.addItem(label, option)
            if previous in options:
                device_box.setCurrentIndex(options.index(previous))
            else:
                device_box.setCurrentIndex(0)
        finally:
            device_box.blockSignals(False)

    def _current_key() -> str | None:
        requirement = req_edit.toPlainText()
        rtl_text = state["rtl_text"]
        assumptions = assume_edit.toPlainText()
        if not requirement.strip() or not rtl_text.strip():
            return None
        return snapshot_key(build_state(requirement, assumptions, rtl_text))

    def _stop_timer(key: str) -> None:
        timer = state.get(key)
        try:
            if timer is not None:
                timer.stop()
        except RuntimeError:
            pass
        state[key] = None

    def on_state(name: str) -> None:
        busy = name in (LOADING, ANALYZING, UNLOADING)
        unload_btn.setEnabled(name in (READY, ANALYZING, LOADING, ERROR))
        cancel_btn.setEnabled(name in (LOADING, ANALYZING))
        model_box.setEnabled(not busy)
        device_box.setEnabled(not busy)
        if name == READY:
            pending = state["pending_entry"]
            if pending is not None:
                state["loaded_entry"] = pending
                state["pending_entry"] = None
            _stop_timer("load_timer")
            started = state.get("load_started") or 0.0
            if started:
                elapsed_load = time.monotonic() - float(started)
                set_status(
                    "Ready on %s in %.1fs." % (_device_text(), elapsed_load)
                )
                _append_details(
                    "loaded %s on %s in %.1fs"
                    % (
                        (state["loaded_entry"] or {}).get("id", "--"),
                        _device_text(),
                        elapsed_load,
                    )
                )
            else:
                set_status("Ready on %s." % _device_text())
        elif name == LOADING:
            set_status("Loading model …")
        elif name == ANALYZING:
            set_status("Analyzing …")
        elif name == ERROR:
            state["pending_entry"] = None
            _stop_timer("load_timer")
            _stop_timer("analyze_timer")
        elif name in (UNLOADING, UNLOADED):
            _stop_timer("load_timer")
            _stop_timer("analyze_timer")
        refresh_run_enabled()

    def _device_text() -> str:
        entry = state["loaded_entry"]
        if entry is None:
            return "--"
        return str(entry.get("device", "--"))

    def _append_details(line: str) -> None:
        details.appendPlainText(line)

    def on_progress(stage: str, detail: str) -> None:
        set_status("%s — %s" % (stage or "Working", detail or ""))
        _append_details("progress %s: %s" % (stage, detail))

    def on_result(result: dict[str, Any]) -> None:
        frozen_key = state["state_key"]
        if isinstance(frozen_key, str) and frozen_key.startswith("scan-"):
            _on_scan_result(result)
            return
        if scan["active"]:
            _append_details("result ignored: scan superseded it")
            return
        current = _current_key()
        if is_result_stale(current, frozen_key):
            details.appendPlainText("result ignored: inputs changed")
            _invalidate_result("inputs changed")
            set_status("Result ignored; inputs changed during analysis.")
            refresh_run_enabled()
            return
        elapsed = time.monotonic() - state["analyze_started"]
        scores = result.get("scores") or {}
        label = str(result.get("label"))
        title = LABEL_TITLES.get(label, label)
        assessment_label.setText("Assessment: %s" % title)
        parts = []
        for name in LABELS:
            value = scores.get(name)
            if isinstance(value, (int, float)):
                parts.append("%s %.3f" % (LABEL_TITLES[name], float(value)))
            else:
                parts.append("%s --" % LABEL_TITLES[name])
        scores_label.setText(" | ".join(parts))
        entry = state["loaded_entry"] or {}
        meta_label.setText(
            "Analysis time: %.1fs  Device: %s  Model: %s"
            % (elapsed, _device_text(), entry.get("display_name", "--"))
        )
        state["last_result"] = result
        state["last_result_key"] = state["state_key"]
        state["session_id"] = state.get("session_id")
        _append_details("result %s" % json.dumps(result)[:2000])
        _stop_timer("analyze_timer")
        save_btn.setEnabled(True)
        refresh_run_enabled()

    def on_error(message: str, stage: str) -> None:
        frozen_key = state["state_key"]
        if (
            scan["active"]
            and isinstance(frozen_key, str)
            and frozen_key.startswith("scan-")
        ):
            _append_details("scan excerpt error [%s] %s" % (stage, message))
            scan["errors"] += 1
            _scan_record_current_row(error="%s: %s" % (stage, message))
            _finish_scan(
                "Scan stopped on excerpt %s; reload the model to retry."
                % str(scan.get("current")),
                aborted=True,
            )
            state["loaded_entry"] = None
            state["pending_entry"] = None
            _invalidate_result(None)
            _stop_timer("load_timer")
            _stop_timer("analyze_timer")
            client = holders.get("client")
            try:
                if client is not None and client.running:
                    client.terminate()
            except RuntimeError:
                pass
            refresh_run_enabled()
            return
        set_status("Error (%s): %s" % (stage, message))
        _append_details("error [%s] %s" % (stage, message))
        state["loaded_entry"] = None
        state["pending_entry"] = None
        _invalidate_result(None)
        _stop_timer("load_timer")
        _stop_timer("analyze_timer")
        client = holders.get("client")
        try:
            if client is not None and client.running:
                client.terminate()
        except RuntimeError:
            pass
        refresh_run_enabled()

    controller.set_callbacks(
        on_state=on_state,
        on_progress=on_progress,
        on_result=on_result,
        on_error=on_error,
    )

    def do_browse() -> None:
        picked, _ = QFileDialog.getOpenFileName(
            win, "Choose design file", "", "Verilog (*.v *.sv)"
        )
        if picked:
            file_edit.setText(picked)
            do_preview()

    def _reset_scan_results() -> None:
        scan["rows"] = []
        scan["errors"] = 0
        scan["violations"] = 0
        scan["current"] = None
        scan["queue"] = None
        scan["selected"] = []
        scan["done"] = False
        scan["muse_used"] = False
        scan_table.setRowCount(0)
        scan_summary_label.setText("Potentially flagged: --")
        scan_excerpt_label.setText("Selected excerpt: --")
        scan_excerpt_preview.clear()
        save_scan_btn.setEnabled(False)

    def do_show_scan_excerpt(row: int, _column: int = 0) -> None:
        """Show the selected row's design excerpt (read-only)."""
        rows = scan.get("rows") or []
        if row < 0 or row >= len(rows):
            return
        entry = rows[row]
        if not isinstance(entry, dict):
            return
        title, body = format_scan_excerpt(entry)
        scan_excerpt_label.setText(title)
        scan_excerpt_preview.setPlainText(body)

    def do_scan_browse() -> None:
        picked, _ = QFileDialog.getOpenFileName(
            win, "Choose large design file", "", "Verilog (*.v *.sv)"
        )
        if picked:
            if scan["active"]:
                do_scan_stop()
            scan_file_edit.setText(picked)
            _reset_scan_results()
            refresh_run_enabled()

    def do_preview() -> None:
        try:
            text, snap = read_design_file(file_edit.text())
        except ValueError as exc:
            rtl_preview.setPlainText("")
            state["rtl_text"] = ""
            state["rtl_snapshot"] = None
            _invalidate_result(None)
            set_status(str(exc))
            refresh_run_enabled()
            return
        state["rtl_text"] = text
        state["rtl_snapshot"] = snap
        rtl_preview.setPlainText(text)
        scope_label.setText(scope_note_for(snap))
        _invalidate_result(None)
        set_status("Design loaded (%d bytes)." % snap["bytes"])
        refresh_run_enabled()

    def do_load() -> None:
        from siliconshield.gui.worker_client import WorkerClient

        selected = _selected_id()
        if selected is None:
            set_status("Select a model before loading.")
            return
        entry = _selected_entry()
        if entry is None:
            set_status("Selected model is not in the registry.")
            return
        registry_device = str(entry.get("device", "cuda:0"))
        load_device = resolve_load_device(registry_device, _selected_device())
        old_client = holders.get("client")
        try:
            old_running = bool(old_client is not None and old_client.running)
        except RuntimeError:
            old_running = False
        if old_running:
            set_status("Releasing previous worker; unload, then retry.")
            _append_details("load refused: previous worker still running")
            return
        effective = dict(entry)
        effective["device"] = load_device
        if not _device_free(load_device):
            set_status(
                "GPU busy — unload the current model or finish the "
                "other job, then retry."
            )
            _append_details(
                "load refused on %s: device busy, no worker started."
                % load_device
            )
            return
        controller.select_model(selected)
        client = WorkerClient(on_result, on_finished, on_error_cb, win)
        holders["client"] = client
        controller.attach(client)
        program = str(effective.get("python", ""))
        args = ["-m", "siliconshield.app.inference_worker"]
        env = dict(os.environ)
        laya_root = str(effective.get("laya_root", ""))
        if laya_root:
            env["LAYA_ROOT"] = laya_root
            env["PYTHONPATH"] = (
                laya_root + os.pathsep + env.get("PYTHONPATH", "")
            )
        try:
            client.start(program, args, env=env)
        except (OSError, RuntimeError) as exc:
            holders["client"] = None
            on_error("Cannot start worker: %s" % exc, "start")
            return
        state["load_started"] = time.monotonic()
        state["loaded_entry"] = None
        state["pending_entry"] = effective
        _invalidate_result("loading")
        _append_details("loading %s on %s" % (effective.get("id"), load_device))
        if not controller.load(effective):
            state["pending_entry"] = None
            holders["client"] = None
            try:
                client.terminate()
            except RuntimeError:
                pass
            return
        _stop_timer("load_timer")
        timer = QTimer(win)
        timer.setSingleShot(True)

        def _fire() -> None:
            controller.on_load_timeout()

        timer.timeout.connect(_fire)
        timer.start(load_timeout_seconds * 1000)
        state["load_timer"] = timer

    def on_finished(exit_code: int, crashed: bool) -> None:
        controller.on_transport_finished(crashed, "exit code %d" % exit_code)
        state["loaded_entry"] = None
        state["pending_entry"] = None
        holders["client"] = None
        _invalidate_result(None)
        _stop_timer("load_timer")
        _stop_timer("analyze_timer")
        refresh_run_enabled()

    def on_error_cb(detail: str) -> None:
        controller.on_transport_error(detail)

    def do_unload() -> None:
        if scan["active"]:
            do_scan_stop()
        controller.unload()

    def _reset_scan_display(reason: str | None = None) -> None:
        scan_table.setRowCount(0)
        suffix = " (%s)" % reason if reason else ""
        scan_summary_label.setText("Potentially flagged: --%s" % suffix)
        if reason is not None:
            scan_status_label.setText(reason)

    def do_run() -> None:
        if needs_preview_refresh(file_edit.text(), state["rtl_snapshot"]):
            do_preview()
            if not state["rtl_text"]:
                return
        requirement = req_edit.toPlainText()
        assumptions = assume_edit.toPlainText()
        rtl_text = state["rtl_text"]
        reason = check_inputs(
            requirement,
            rtl_text,
            bool(models) and not models_error,
            assumptions,
        )
        if reason is not None:
            set_status(reason)
            return
        frozen = build_state(requirement, assumptions, rtl_text)
        key = snapshot_key(frozen)
        inputs = {
            "requirement": requirement,
            "assumptions": assumptions,
            "rtl": rtl_text,
            "state": frozen,
        }
        state["analyze_started"] = time.monotonic()
        _invalidate_result(None)
        state["state_text"] = frozen
        state["state_key"] = key
        if not controller.analyze(inputs, key):
            set_status("Run is not available in the current state.")
            return
        set_status("Analyzing …")
        _stop_timer("analyze_timer")
        timer = QTimer(win)
        timer.setSingleShot(True)

        def _fire() -> None:
            controller.on_analyze_timeout()

        timer.timeout.connect(_fire)
        timer.start(analyze_timeout_seconds * 1000)
        state["analyze_timer"] = timer

    def do_cancel() -> None:
        controller.cancel()

    def _scan_append_row(
        excerpt_no: int,
        lines: str,
        verdict: str,
        scores: dict[str, Any],
        seconds: float | None,
        reason: str,
        verified: str = "Not verified",
    ) -> None:
        row = scan_table.rowCount()
        scan_table.insertRow(row)

        def _item(text: str) -> Any:
            from PySide6.QtWidgets import QTableWidgetItem as _Item

            return _Item(text)

        def _score(name: str) -> str:
            value = scores.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return "--"
            return "%.3f" % float(value)

        elapsed = "--" if seconds is None else "%.1fs" % seconds
        scan_table.setItem(row, 0, _item(str(excerpt_no)))
        scan_table.setItem(row, 1, _item(lines))
        scan_table.setItem(row, 2, _item(verdict))
        scan_table.setItem(row, 3, _item(_score("satisfies_requirement")))
        scan_table.setItem(row, 4, _item(_score("violates_requirement")))
        scan_table.setItem(row, 5, _item(_score("insufficient_context")))
        scan_table.setItem(row, 6, _item(elapsed))
        scan_table.setItem(row, 7, _item(reason))
        scan_table.setItem(row, 8, _item(verified))

    def _scan_record_current_row(error: str) -> None:
        index = scan.get("current")
        windows = scan.get("windows") or []
        window = next((w for w in windows if w.get("index") == index), None)
        lines = "--"
        if isinstance(window, dict):
            lines = "%s-%s" % (window.get("start_line"), window.get("end_line"))
        reason = ""
        for entry in scan.get("selected") or []:
            if entry.get("index") == index:
                reason = str(entry.get("reason", ""))
                break
        text = ""
        if isinstance(window, dict):
            text = str(window.get("text", ""))
        _scan_append_row(
            len(scan["rows"]),
            lines,
            "ERROR",
            {},
            None,
            error or reason,
            "Not run",
        )
        scan["rows"].append(
            {
                "index": index,
                "lines": lines,
                "label": "ERROR",
                "text": text,
                "kind": window.get("kind")
                if isinstance(window, dict)
                else None,
                "module": (
                    window.get("module") if isinstance(window, dict) else None
                ),
                "complete": (
                    bool(window.get("complete"))
                    if isinstance(window, dict)
                    else False
                ),
                "reason": reason,
                "error": error,
            }
        )

    def _scan_update_summary() -> None:
        queue = scan.get("queue")
        if queue is None:
            return
        scan_summary_label.setText(
            "Potentially flagged: %d of %d excerpts (%d errors, muse=%s; model verdicts only)."
            % (
                queue.violations,
                queue.judged,
                scan.get("errors", 0),
                "used" if scan.get("muse_used") else "skipped",
            )
        )
        scan_progress.setRange(0, max(1, len(scan.get("selected") or [])))
        scan_progress.setValue(queue.judged)

    def _finish_scan(message: str, aborted: bool = False) -> None:
        scan["active"] = False
        scan["muse_running"] = False
        _stop_muse_timer()
        _stop_timer("analyze_timer")
        scan_progress.setRange(0, 1)
        scan_progress.setValue(0 if aborted else 1)
        if aborted:
            scan_status_label.setText(message)
            _append_details("scan aborted: %s" % message)
        else:
            queue = scan.get("queue")
            errors = int(scan.get("errors", 0))
            if queue is not None:
                from siliconshield.gui.scan import count_summary

                summary = count_summary(queue, errors)
                scan_summary_label.setText(
                    summary
                    + " muse=%s."
                    % ("used" if scan.get("muse_used") else "skipped")
                )
            scan_status_label.setText(message)
            _append_details("scan finished: %s" % message)
        save_scan_btn.setEnabled(bool(scan.get("rows")))
        refresh_run_enabled()

    def _scan_next() -> None:
        if not scan["active"]:
            return
        queue = scan.get("queue")
        if queue is None:
            _finish_scan("Scan has no excerpts.", aborted=True)
            return
        if queue.remaining == 0:
            scan["done"] = True
            _finish_scan(
                "Scan complete: %d violations of %d excerpts."
                % (queue.violations, queue.judged)
            )
            return
        windows = {w["index"]: w for w in scan.get("windows") or []}
        index = queue._pending[0]
        window = windows.get(index)
        if window is None:
            _finish_scan("Scan lost excerpt %s." % index, aborted=True)
            return
        frozen = build_state(
            scan["requirement"], scan["assumptions"], str(window["text"])
        )
        key = "scan-%d-%d" % (scan["run_id"], index)
        inputs = {
            "requirement": scan["requirement"],
            "assumptions": scan["assumptions"],
            "rtl": str(window["text"]),
            "state": frozen,
        }
        scan["current"] = index
        scan["excerpt_started"] = time.monotonic()
        state["state_key"] = key
        scan_status_label.setText(
            "Judging excerpt %d of %d (lines %s-%s)…"
            % (
                queue.judged + 1,
                len(scan.get("selected") or []),
                window.get("start_line"),
                window.get("end_line"),
            )
        )
        if not controller.analyze(inputs, key):
            _finish_scan(
                "Scan cannot judge now; reload the model and retry.",
                aborted=True,
            )
            return
        _stop_timer("analyze_timer")
        timer = QTimer(win)
        timer.setSingleShot(True)

        def _fire() -> None:
            controller.on_analyze_timeout()

        timer.timeout.connect(_fire)
        timer.start(analyze_timeout_seconds * 1000)
        state["analyze_timer"] = timer

    def _on_scan_result(result: dict[str, Any]) -> None:
        if not scan["active"]:
            _append_details("scan result ignored: scan stopped")
            return
        index = scan.get("current")
        windows = {w["index"]: w for w in scan.get("windows") or []}
        window = windows.get(index)
        if window is None:
            _finish_scan("Scan lost excerpt %s." % index, aborted=True)
            return
        elapsed = time.monotonic() - float(
            scan.get("excerpt_started") or time.monotonic()
        )
        scores = result.get("scores") or {}
        label = str(result.get("label"))
        reason = ""
        for entry in scan.get("selected") or []:
            if entry.get("index") == index:
                reason = str(entry.get("reason", ""))
                break
        lines = "%s-%s" % (window.get("start_line"), window.get("end_line"))
        _scan_append_row(
            len(scan["rows"]), lines, label, scores, elapsed, reason
        )
        scan["rows"].append(
            {
                "index": index,
                "lines": lines,
                "label": label,
                "text": str(window.get("text", "")),
                "kind": window.get("kind"),
                "module": window.get("module"),
                "complete": bool(window.get("complete")),
                "probabilities": dict(scores),
                "seconds": elapsed,
                "reason": reason,
            }
        )
        queue = scan.get("queue")
        advanced = queue.advance(label) if queue is not None else None
        controller.clear_result()
        _scan_update_summary()
        _stop_timer("analyze_timer")
        if advanced is None:
            scan["done"] = True
            _finish_scan(
                "Scan complete: %d potentially flagged of %d excerpts "
                "(model verdicts only; not independently verified)."
                % (
                    queue.violations if queue else 0,
                    queue.judged if queue else 0,
                )
            )
        else:
            _scan_next()

    def _on_muse_failed(detail: str) -> None:
        if not scan["active"]:
            return
        scan["muse_running"] = False
        scan_progress.setRange(0, 1)
        scan_progress.setValue(0)
        _append_details(
            "Muse selection failed (%s); using all excerpts." % detail
        )
        scan_status_label.setText(
            "Muse selection failed; judging all %d excerpts."
            % len(scan["windows"])
        )
        from siliconshield.gui.scan import (
            SCAN_MAX_EXCERPTS,
            ScanQueue,
            describe_excerpt_scope,
        )

        scan["selected"] = [
            {
                "index": w["index"],
                "reason": "%s; Muse selection failed"
                % describe_excerpt_scope(w),
            }
            for w in scan["windows"][:SCAN_MAX_EXCERPTS]
        ]
        scan["muse_used"] = False
        scan["queue"] = ScanQueue([e["index"] for e in scan["selected"]])
        scan_progress.setRange(0, max(1, len(scan["selected"])))
        scan_progress.setValue(0)
        _scan_next()

    def _on_muse_selected(selected: list[dict[str, Any]]) -> None:
        if not scan["active"]:
            return
        scan["muse_running"] = False
        try:
            from siliconshield.gui.scan import (
                SCAN_MAX_EXCERPTS,
                ScanQueue,
                describe_excerpt_scope,
                validate_scan_selection,
            )

            cleaned = validate_scan_selection(selected, len(scan["windows"]))
        except ValueError as exc:
            _on_muse_failed(str(exc))
            return
        windows_by_index = {
            window["index"]: window for window in scan["windows"]
        }
        enriched = [
            {
                "index": entry["index"],
                "reason": "%s; %s"
                % (
                    describe_excerpt_scope(windows_by_index[entry["index"]]),
                    entry["reason"],
                ),
            }
            for entry in cleaned
        ]
        capped = enriched[:SCAN_MAX_EXCERPTS]
        if len(cleaned) > len(capped):
            _append_details(
                "Muse selected %d windows; capped to %d."
                % (len(cleaned), len(capped))
            )
        scan["selected"] = capped
        scan["muse_used"] = True
        scan_status_label.setText(
            "Muse selected %d of %d excerpts; judging…"
            % (len(capped), len(scan["windows"]))
        )
        _append_details(
            "Muse selected %d/%d excerpts."
            % (len(capped), len(scan["windows"]))
        )
        scan["queue"] = ScanQueue([e["index"] for e in capped])
        scan_progress.setRange(0, max(1, len(capped)))
        scan_progress.setValue(0)
        _scan_next()

    def do_scan_demo() -> None:
        from siliconshield.gui.demos import get_scan_demo

        demo_id = scan_demo_box.currentData()
        if not isinstance(demo_id, str) or not demo_id:
            demo_id = str(scan_demo_box.currentText()).strip()
        try:
            demo = get_scan_demo(str(demo_id))
        except ValueError as exc:
            scan_status_label.setText(str(exc))
            return
        scan_req_edit.blockSignals(True)
        scan_assume_edit.blockSignals(True)
        try:
            scan_req_edit.setPlainText(str(demo["requirement"]))
            scan_assume_edit.setPlainText(str(demo["assumptions"]))
        finally:
            scan_req_edit.blockSignals(False)
            scan_assume_edit.blockSignals(False)
        if scan["active"]:
            do_scan_stop()
        rtl_text = str(demo["rtl"])
        scan_file_edit.setText("demo:%s" % demo["id"])
        scan["file_text"] = rtl_text
        scan["file_snapshot"] = {
            "path": "demo:%s" % demo["id"],
            "sha256": hashlib.sha256(rtl_text.encode("utf-8")).hexdigest(),
            "bytes": len(rtl_text.encode("utf-8")),
        }
        _reset_scan_results()
        _invalidate_result(None)
        scan_status_label.setText("Demo loaded: %s." % demo["title"])
        _append_details("scan demo loaded: %s" % demo["id"])
        refresh_run_enabled()

    def _stop_muse_timer() -> None:
        timer = scan.get("muse_timer")
        scan["muse_timer"] = None
        try:
            if timer is not None:
                timer.stop()
        except RuntimeError:
            pass

    def _check_muse_result() -> None:
        if not scan["active"] or not scan["muse_running"]:
            return
        done = scan.get("muse_done")
        try:
            finished = bool(done is not None and done.is_set())
        except RuntimeError:
            return
        if not finished:
            return
        _stop_muse_timer()
        scan["muse_running"] = False
        outcome = scan.get("muse_outcome")
        if not isinstance(outcome, dict) or not outcome.get("ok", False):
            detail = "Muse selection failed"
            if isinstance(outcome, dict) and outcome.get("detail"):
                detail = str(outcome["detail"])
            _on_muse_failed(detail)
            return
        selected = outcome.get("selected", [])
        if not isinstance(selected, list):
            _on_muse_failed("Muse returned a malformed selection.")
            return
        _on_muse_selected(selected)

    def do_scan() -> None:
        from siliconshield.gui.scan import (
            SCAN_MAX_EXCERPTS,
            ScanQueue,
            describe_excerpt_scope,
            read_scan_source,
            split_scan_excerpts,
        )

        if scan["active"]:
            return
        pending_path = scan_file_edit.text().strip()
        if pending_path.startswith("demo:"):
            if (
                not scan.get("file_text")
                or not isinstance(scan.get("file_snapshot"), dict)
                or scan["file_snapshot"].get("path") != pending_path
            ):
                scan_status_label.setText("Reload the scan demo.")
                return
            scan_text = scan["file_text"]
            scan_snap = scan["file_snapshot"]
        else:
            try:
                scan_text, scan_snap = read_scan_source(scan_file_edit.text())
            except ValueError as exc:
                scan_status_label.setText(str(exc))
                return
        requirement = scan_req_edit.toPlainText()
        assumptions = scan_assume_edit.toPlainText()
        reason = check_inputs(
            requirement,
            scan_text,
            bool(models) and not models_error,
            assumptions,
        )
        if reason is not None:
            scan_status_label.setText(reason)
            return
        try:
            windows = split_scan_excerpts(scan_text)
        except ValueError as exc:
            scan_status_label.setText(str(exc))
            return
        scan["run_id"] += 1
        scan["file_text"] = scan_text
        scan["file_snapshot"] = scan_snap
        scan["requirement"] = requirement
        scan["assumptions"] = assumptions
        scan["windows"] = windows
        scan["selected"] = []
        scan["rows"] = []
        scan["errors"] = 0
        scan["violations"] = 0
        scan["current"] = None
        scan["done"] = False
        scan["muse_used"] = False
        scan_table.setRowCount(0)
        scan_summary_label.setText("Potentially flagged: --")
        save_scan_btn.setEnabled(False)
        state["last_result"] = None
        controller.clear_result()
        _invalidate_result(None)
        scan["active"] = True
        refresh_run_enabled()
        use_muse = muse_check.isChecked()
        if not use_muse:
            scan["selected"] = [
                {
                    "index": w["index"],
                    "reason": "%s; Muse selection off"
                    % describe_excerpt_scope(w),
                }
                for w in windows[:SCAN_MAX_EXCERPTS]
            ]
            scan["queue"] = ScanQueue([e["index"] for e in scan["selected"]])
            scan_status_label.setText(
                "Judging %d deterministic excerpts…" % len(scan["selected"])
            )
            _append_details(
                "scan started: %d deterministic excerpts"
                % len(scan["selected"])
            )
            scan_progress.setRange(0, max(1, len(scan["selected"])))
            scan_progress.setValue(0)
            _scan_next()
            return
        scan["muse_running"] = True
        scan_progress.setRange(0, 0)
        scan_status_label.setText("Muse is selecting relevant excerpts…")
        _append_details("scan started: asking Muse to select excerpts")
        refresh_run_enabled()
        import threading as _threading

        muse_done: Any = _threading.Event()
        muse_outcome: dict[str, Any] = {}
        scan["muse_done"] = muse_done
        scan["muse_outcome"] = muse_outcome
        scan["muse_timer"] = None

        def _muse_run() -> None:
            try:
                from siliconshield.gui.muse_scan import (
                    default_muse_audit_dir,
                    select_excerpts,
                )

                selected = select_excerpts(
                    windows,
                    requirement,
                    assumptions,
                    audit_dir=default_muse_audit_dir(),
                )
                muse_outcome["ok"] = True
                muse_outcome["selected"] = selected
            except Exception as exc:
                muse_outcome["ok"] = False
                muse_outcome["detail"] = "%s: %s" % (type(exc).__name__, exc)
            finally:
                muse_done.set()

        muse_thread = _threading.Thread(target=_muse_run, daemon=True)
        muse_thread.start()
        muse_timer = QTimer(win)
        muse_timer.setInterval(200)
        muse_timer.timeout.connect(_check_muse_result)
        scan["muse_timer"] = muse_timer
        muse_timer.start()

    def do_scan_stop() -> None:
        if not scan["active"]:
            return
        scan["active"] = False
        scan["muse_running"] = False
        _stop_muse_timer()
        _stop_timer("analyze_timer")
        scan_progress.setRange(0, 1)
        scan_progress.setValue(0)
        scan_status_label.setText("Scan stopped; model stays loaded.")
        _append_details("scan stopped by user; worker stays loaded")
        refresh_run_enabled()

    def do_save_scan() -> None:
        from PySide6.QtWidgets import QFileDialog as SaveDialog

        if not scan.get("rows"):
            return
        picked, _ = SaveDialog.getSaveFileName(
            win, "Save scan", "gui-scan.json", "JSON (*.json)"
        )
        if not picked:
            return
        from siliconshield.schemas import GUI_SCAN_SCHEMA

        queue = scan.get("queue")
        record = {
            "schema": GUI_SCAN_SCHEMA,
            "scope": scope_note_for(scan.get("file_snapshot")),
            "warning": "This export contains the analyzed RTL and "
            "requirement; share it only intentionally.",
            "inputs": {
                "requirement": scan.get("requirement", ""),
                "assumptions": scan.get("assumptions", ""),
                "file_snapshot": scan.get("file_snapshot"),
                "windows": len(scan.get("windows") or []),
                "selected": scan.get("selected"),
                "muse_used": bool(scan.get("muse_used")),
            },
            "results": scan.get("rows"),
            "violations": queue.violations if queue is not None else 0,
            "judged": queue.judged if queue is not None else 0,
            "errors": int(scan.get("errors", 0)),
            "device": _device_text(),
            "model": state["loaded_entry"],
        }
        try:
            Path(picked).write_text(
                json.dumps(record, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            set_status("Scan saved to %s." % picked)
        except (OSError, ValueError) as exc:
            set_status("Cannot save scan: %s" % exc)

    def do_save() -> None:
        from PySide6.QtWidgets import QFileDialog as SaveDialog

        result = state["last_result"]
        if result is None:
            return
        picked, _ = SaveDialog.getSaveFileName(
            win, "Save result", "gui-result.json", "JSON (*.json)"
        )
        if not picked:
            return
        record = {
            "schema": "siliconshield-gui-analysis-v1",
            "scope": scope_note_for(state["rtl_snapshot"]),
            "warning": "This export contains the analyzed RTL and "
            "requirement; share it only intentionally.",
            "inputs": {
                "requirement": req_edit.toPlainText(),
                "assumptions": assume_edit.toPlainText(),
                "rtl": state["rtl_text"],
                "rtl_snapshot": state["rtl_snapshot"],
                "state_key": state["state_key"],
            },
            "result": result,
            "device": _device_text(),
            "model": state["loaded_entry"],
        }
        try:
            Path(picked).write_text(
                json.dumps(record, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            set_status("Result saved to %s." % picked)
        except (OSError, ValueError) as exc:
            set_status("Cannot save result: %s" % exc)

    def on_text_changed() -> None:
        _invalidate_result("inputs changed")
        refresh_run_enabled()

    def on_model_changed(_index: int) -> None:
        if controller.state in (LOADING, ANALYZING):
            return
        _refresh_devices()
        controller.select_model(_selected_id())
        _invalidate_result("select and load")
        _reset_scan_results()
        refresh_run_enabled()

    def on_device_changed(_index: int) -> None:
        if controller.state in (LOADING, ANALYZING):
            return
        _invalidate_result("select and load")
        _reset_scan_results()
        refresh_run_enabled()

    def do_load_demo() -> None:
        from siliconshield.gui.demos import get_demo

        demo_id = demo_box.currentData()
        if not isinstance(demo_id, str) or not demo_id:
            demo_id = str(demo_box.currentText()).strip()
        try:
            demo = get_demo(str(demo_id))
        except ValueError as exc:
            set_status(str(exc))
            return
        req_edit.blockSignals(True)
        assume_edit.blockSignals(True)
        try:
            file_edit.setText("")
            req_edit.setPlainText(str(demo["requirement"]))
            assume_edit.setPlainText(str(demo["assumptions"]))
        finally:
            req_edit.blockSignals(False)
            assume_edit.blockSignals(False)
        rtl_text = str(demo["rtl"])
        rtl_preview.setPlainText(rtl_text)
        state["rtl_text"] = rtl_text
        state["rtl_snapshot"] = {
            "path": "demo:%s" % demo["id"],
            "sha256": hashlib.sha256(rtl_text.encode("utf-8")).hexdigest(),
            "bytes": len(rtl_text.encode("utf-8")),
        }
        scope_label.setText(scope_note_for(state["rtl_snapshot"]))
        _invalidate_result(None)
        set_status("Demo loaded: %s." % demo["title"])
        refresh_run_enabled()

    from siliconshield.gui.demos import list_demos, list_scan_demos

    for demo in list_demos():
        demo_box.addItem(str(demo["title"]), demo["id"])
    for demo in list_scan_demos():
        scan_demo_box.addItem(str(demo["title"]), demo["id"])
    _refresh_devices()
    try:
        import shutil as _shutil

        _muse_found = _shutil.which("muse") is not None
    except OSError:
        _muse_found = False
    muse_check.setChecked(_muse_found)
    muse_check.setEnabled(_muse_found)
    if not _muse_found:
        muse_check.setToolTip("`muse` CLI not found; deterministic windows.")
    browse_btn.clicked.connect(do_browse)
    demo_load_btn.clicked.connect(do_load_demo)
    scan_demo_load_btn.clicked.connect(do_scan_demo)
    file_edit.editingFinished.connect(do_preview)
    load_btn.clicked.connect(do_load)
    unload_btn.clicked.connect(do_unload)
    run_btn.clicked.connect(do_run)
    cancel_btn.clicked.connect(do_cancel)
    save_btn.clicked.connect(do_save)
    req_edit.textChanged.connect(on_text_changed)
    assume_edit.textChanged.connect(on_text_changed)
    model_box.currentIndexChanged.connect(on_model_changed)
    device_box.currentIndexChanged.connect(on_device_changed)
    scan_browse_btn.clicked.connect(do_scan_browse)
    scan_btn.clicked.connect(do_scan)
    scan_stop_btn.clicked.connect(do_scan_stop)
    save_scan_btn.clicked.connect(do_save_scan)
    scan_table.cellClicked.connect(do_show_scan_excerpt)
    scan_table.currentCellChanged.connect(
        lambda cur_row, _c, _p_row, _p_col: do_show_scan_excerpt(cur_row)
    )

    controller.select_model(_selected_id())
    refresh_run_enabled()

    def _shutdown_owned_worker() -> None:
        # A running Muse selection is a daemon thread that never touches
        # Qt; drop our references and let it finish alone.
        scan["active"] = False
        scan["muse_running"] = False
        scan["muse_done"] = None
        scan["muse_outcome"] = None
        _stop_muse_timer()
        try:
            controller.shutdown()
        except RuntimeError:
            pass
        client = holders.get("client")
        try:
            if client is not None and client.running:
                client.terminate()
                client.wait_for_exit(5000)
        except RuntimeError:
            pass

    # The Qt signal connections retain the slots, which retain the
    # controller, timers, and input snapshots for the window lifetime.
    _ = (state, controller_holder, holders)
    return win


def _device_free(device: str) -> bool:
    """Return False when nvidia-smi shows the device is in use.

    Missing tooling fails open for CPU/fixture runs; the worker still
    verifies actual CUDA placement at load time.
    """
    import shutil
    import subprocess

    if not device.startswith("cuda:"):
        return True
    index = device.split(":", 1)[1]
    if not index.isdigit():
        return True
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return True
    try:
        run = subprocess.run(
            [
                smi,
                "--query-compute-apps=pid",
                "--format=csv,noheader",
                "-i",
                index,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return not run.stdout.strip()
