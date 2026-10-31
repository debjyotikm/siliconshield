"""Worker lifecycle and staging guards: no GPU needed."""

import os
from pathlib import Path

import pytest


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
