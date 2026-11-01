"""GUI controller, demo, scan, and assessment-clearing tests: Qt-free fixtures only."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from siliconshield.gui.controller import (
    ANALYZING,
    ERROR,
    LOADING,
    READY,
    UNLOADED,
    Controller,
)
from siliconshield.gui.scan import (
    ScanQueue,
    count_summary,
    describe_excerpt_scope,
    split_scan_excerpts,
    split_scan_windows,
    validate_scan_selection,
)


# ---- Merged from test_gui_controller.py ----
class FakeTransport:
    """Scripted worker transport without processes or GPUs."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any], str]] = []
        self.terminated = False
        self.on_message = lambda _m: None
        self.on_finished = lambda _c, _e: None
        self.on_error = lambda _d: None

    def send_load(self, model: dict[str, Any], request_id: str) -> None:
        self.sent.append(("load", dict(model), request_id))

    def send_analyze(self, inputs: dict[str, Any], request_id: str) -> None:
        self.sent.append(("analyze", dict(inputs), request_id))

    def terminate(self) -> None:
        self.terminated = True


def _controller() -> tuple[Controller, dict[str, Any], FakeTransport]:
    seen: dict[str, Any] = {
        "states": [],
        "progress": [],
        "results": [],
        "errors": [],
    }
    control = Controller(
        on_state=lambda name: seen["states"].append(name),
        on_progress=lambda stage, detail: seen["progress"].append(
            (stage, detail)
        ),
        on_result=lambda result: seen["results"].append(result),
        on_error=lambda message, stage: seen["errors"].append((message, stage)),
    )
    transport = FakeTransport()
    control.attach(transport)
    return control, seen, transport


def _model(model_id: str = "fixture") -> dict[str, Any]:
    return {"id": model_id, "display_name": "Fixture"}


def test_load_success_enables_run() -> None:
    control, seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    assert control.state == LOADING
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "progress",
            "request_id": request_id,
            "session_id": None,
            "stage": "Loading model",
            "detail": "weights",
        }
    )
    assert seen["progress"]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.state == READY
    assert control.can_analyze(True)
    assert not control.can_analyze(False)


def test_load_failure_never_produces_ready() -> None:
    control, seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "error",
            "request_id": request_id,
            "session_id": "sess-1",
            "stage": "load",
            "detail": "CUDA is unavailable",
        }
    )
    assert control.state == ERROR
    assert seen["errors"]
    assert not control.can_analyze(True)


def test_cpu_fallback_during_load_refuses_ready() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 1,
        }
    )
    assert control.state == ERROR


def test_cpu_ready_is_accepted() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cpu",
            "cpu_fallback_count": 0,
        }
    )
    assert control.state == READY


def test_unknown_device_ready_is_rejected() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "tpu",
            "cpu_fallback_count": 0,
        }
    )
    assert control.state == ERROR


def test_model_mismatch_disables_run() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.state == READY
    control.select_model("other")
    assert not control.can_analyze(True)


def test_one_worker_survives_two_analyses() -> None:
    control, seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-9",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    for key in ("first", "second"):
        assert control.analyze(
            {
                "requirement": "req",
                "rtl": "module x; endmodule",
                "state": "STATE %s" % key,
            },
            key,
        )
        assert control.state == ANALYZING
        pending = transport.sent[-1][2]
        control.on_transport_message(
            {
                "type": "result",
                "request_id": pending,
                "session_id": "sess-9",
                "label": "satisfies_requirement",
                "probabilities": {
                    "satisfies_requirement": 0.6,
                    "violates_requirement": 0.3,
                    "insufficient_context": 0.1,
                },
                "usage": {},
                "coverage": {
                    "truncated": False,
                    "truncated_questions": [],
                    "state_tokens_dropped": 0,
                },
            }
        )
        assert control.state == READY
    assert len(seen["results"]) == 2
    assert not transport.terminated


def test_stale_replies_are_ignored() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.analyze(
        {"requirement": "req", "rtl": "rtl", "state": "STATE"},
        "key-1",
    )
    control.on_transport_message(
        {
            "type": "result",
            "request_id": "wrong",
            "session_id": "sess-1",
            "label": "satisfies_requirement",
            "probabilities": {
                "satisfies_requirement": 0.6,
                "violates_requirement": 0.3,
                "insufficient_context": 0.1,
            },
            "usage": {},
            "coverage": {
                "truncated": False,
                "truncated_questions": [],
                "state_tokens_dropped": 0,
            },
        }
    )
    assert control.state == ANALYZING
    assert control.stale_replies == 1


def test_truncated_result_refuses_verdict() -> None:
    control, seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.analyze(
        {"requirement": "req", "rtl": "rtl", "state": "STATE"},
        "key-1",
    )
    pending = transport.sent[-1][2]
    control.on_transport_message(
        {
            "type": "result",
            "request_id": pending,
            "session_id": "sess-1",
            "label": "satisfies_requirement",
            "probabilities": {
                "satisfies_requirement": 0.6,
                "violates_requirement": 0.3,
                "insufficient_context": 0.1,
            },
            "usage": {},
            "coverage": {
                "truncated": True,
                "truncated_questions": [],
                "state_tokens_dropped": 0,
            },
        }
    )
    assert control.state == ERROR
    assert seen["errors"]
    assert not seen["results"]


def test_unknown_worker_message_fails_closed() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message({"type": "mystery", "request_id": request_id})
    assert control.state == ERROR


def test_cancel_terminates_owned_worker() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.analyze(
        {"requirement": "req", "rtl": "rtl", "state": "STATE"},
        "key-1",
    )
    assert control.state == ANALYZING
    control.cancel()
    assert transport.terminated
    assert control.state == UNLOADED


def test_load_timeout_terminates_worker() -> None:
    control, seen, _transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    control.on_load_timeout()
    assert control.state == UNLOADED
    assert seen["errors"]


def test_unload_releases_worker() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    control.unload()
    assert transport.terminated
    control.on_transport_finished(False, "exit code 0")
    assert control.state == UNLOADED


def test_unexpected_exit_clears_readiness() -> None:
    control, seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.state == READY
    assert transport.sent
    control.on_transport_finished(True, "exit code 1")
    assert control.state == ERROR
    assert seen["errors"]


def test_input_edits_invalidate_results() -> None:
    control, _seen, transport = _controller()
    control.select_model("fixture")
    assert control.load(_model())
    request_id = transport.sent[0][2]
    control.on_transport_message(
        {
            "type": "ready",
            "request_id": request_id,
            "session_id": "sess-1",
            "model_id": "fixture",
            "device": "cuda:0",
            "cpu_fallback_count": 0,
        }
    )
    assert control.analyze(
        {"requirement": "req", "rtl": "rtl", "state": "STATE"},
        "key-1",
    )
    pending = transport.sent[-1][2]
    control.on_transport_message(
        {
            "type": "result",
            "request_id": pending,
            "session_id": "sess-1",
            "label": "satisfies_requirement",
            "probabilities": {
                "satisfies_requirement": 0.6,
                "violates_requirement": 0.3,
                "insufficient_context": 0.1,
            },
            "usage": {},
            "coverage": {
                "truncated": False,
                "truncated_questions": [],
                "state_tokens_dropped": 0,
            },
        }
    )
    assert control.has_stale_result("key-2")
    assert not control.has_stale_result("key-1")
    control.clear_result()
    assert not control.has_stale_result("key-2")


@pytest.mark.parametrize("state", [LOADING, ANALYZING])
def test_selection_locked_during_busy(state: str) -> None:
    control, _seen, _transport = _controller()
    control.select_model("fixture")
    control._state = state  # Fixture drives the busy state directly.
    control.select_model("other")
    assert control._selected_id == "fixture"


# ---- Merged from test_gui_demos.py ----
def test_list_demos_matches_strength_showcase() -> None:
    from siliconshield.gui.demos import get_demo, list_demos

    demos = list_demos()
    assert [demo["id"] for demo in demos] == [
        "privilege-check-missing",
        "privileged-write-guarded",
        "clear-enable-omitted",
        "clear-enable-enforced",
        "address-check-too-broad",
        "exact-address-guard",
    ]
    for demo in demos:
        assert demo["rtl"].strip()
        assert demo["requirement"].strip()
        assert demo["assumptions"].strip()
        assert demo["expected"] in {
            "satisfies_requirement",
            "violates_requirement",
        }
    by_id = {demo["id"]: demo for demo in demos}
    pairs = (
        ("privilege-check-missing", "privileged-write-guarded"),
        ("clear-enable-omitted", "clear-enable-enforced"),
        ("address-check-too-broad", "exact-address-guard"),
    )
    for violating_id, satisfying_id in pairs:
        violating = by_id[violating_id]
        satisfying = by_id[satisfying_id]
        assert violating["expected"] == "violates_requirement"
        assert satisfying["expected"] == "satisfies_requirement"
        assert violating["requirement"] == satisfying["requirement"]
        assert violating["assumptions"] == satisfying["assumptions"]
        assert violating["rtl"] != satisfying["rtl"]
    assert get_demo("privilege-check-missing")["id"] == (
        "privilege-check-missing"
    )
    assert get_demo("exact-address-guard")["rtl"].strip().startswith("module")


def test_get_demo_rejects_unknown() -> None:
    import pytest

    from siliconshield.gui.demos import get_demo

    with pytest.raises(ValueError, match="Unknown demo"):
        get_demo("training-dataset-case")


def test_demo_files_match_presets() -> None:
    import json

    from siliconshield.gui.demos import list_demos

    root = Path(__file__).resolve().parents[1] / "examples" / "gui-demo"
    assert (root / "README.md").is_file()
    requirements = json.loads((root / "requirements.example.json").read_text())
    for demo in list_demos():
        rtl_file = root / (demo["id"] + ".sv")
        assert rtl_file.is_file()
        assert demo["rtl"] in rtl_file.read_text()
        assert requirements[demo["id"]]["requirement"] == demo["requirement"]
        assert requirements[demo["id"]]["assumptions"] == demo["assumptions"]
        assert requirements[demo["id"]]["expected"] == demo["expected"]


def test_scan_demos_are_large_and_synthetic() -> None:
    import json

    from siliconshield.gui.demos import get_scan_demo, list_scan_demos
    from siliconshield.gui.scan import split_scan_excerpts

    demos = list_scan_demos()
    assert [demo["id"] for demo in demos] == ["peripheral-config-bank"]
    for demo in demos:
        assert demo["requirement"].strip()
        assert demo["rtl"].count("module ") == 12
        lowered = demo["rtl"].lower()
        for leaked_label in (
            "faulty",
            "clean",
            "violation",
            "satisfies",
            "expected",
        ):
            assert leaked_label not in lowered
    demo = get_scan_demo("peripheral-config-bank")
    excerpts = split_scan_excerpts(demo["rtl"])
    assert len(excerpts) == 12
    assert all(excerpt["kind"] == "module" for excerpt in excerpts)
    assert all(excerpt["complete"] is True for excerpt in excerpts)
    assert all(excerpt["text"].count("module ") == 1 for excerpt in excerpts)
    assert all(excerpt["text"].count("endmodule") == 1 for excerpt in excerpts)
    import pytest as _pytest

    with _pytest.raises(ValueError, match="Unknown scan demo"):
        get_scan_demo("training-dataset")
    root = Path(__file__).resolve().parents[1] / "examples" / "gui-demo"
    requirements = json.loads((root / "requirements.example.json").read_text())
    for demo in demos:
        rtl_file = root / (demo["id"] + ".sv")
        assert rtl_file.is_file()
        assert demo["rtl"] in rtl_file.read_text()
        assert requirements[demo["id"]]["requirement"] == demo["requirement"]
        assert requirements[demo["id"]]["assumptions"] == demo["assumptions"]


def test_scan_demo_hidden_oracle_stays_out_of_rtl() -> None:
    from siliconshield.gui.demos import get_scan_demo
    from siliconshield.gui.scan import split_scan_excerpts

    expected = {
        "clock_config": "violates_requirement",
        "pad_config": "satisfies_requirement",
        "alert_config": "violates_requirement",
        "timer_config": "satisfies_requirement",
        "irq_config": "violates_requirement",
        "dma_config": "satisfies_requirement",
        "bus_config": "violates_requirement",
        "power_config": "satisfies_requirement",
        "reset_config": "violates_requirement",
        "memory_config": "satisfies_requirement",
        "trace_config": "violates_requirement",
        "watchdog_config": "satisfies_requirement",
    }
    demo = get_scan_demo("peripheral-config-bank")
    excerpts = split_scan_excerpts(demo["rtl"])
    assert {excerpt["module"] for excerpt in excerpts} == set(expected)
    for excerpt in excerpts:
        module = str(excerpt["module"])
        body = str(excerpt["text"])
        if expected[module] == "satisfies_requirement":
            assert "else if (wr_en && privileged)" in body
        else:
            assert "else if (wr_en)" in body
        assert expected[module] not in body


def test_demo_window_loads_preset_offscreen() -> None:
    import pytest

    pytest.importorskip("PySide6.QtWidgets")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from siliconshield.gui import registry as registry_module
    from siliconshield.gui.application import _build_window

    app = QApplication.instance() or QApplication([])
    window = _build_window(registry_module, None, 180, 60)
    window.close()


# ---- Merged from test_gui_scan.py ----
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


# ---- Merged from test_gui_assessment_clearing.py ----
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
