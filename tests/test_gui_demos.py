"""GUI demo presets: synthetic fixtures only, never training data."""

from pathlib import Path


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
