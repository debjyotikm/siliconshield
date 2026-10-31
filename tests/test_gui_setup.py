"""GUI --init setup: synthetic fixtures only, no GPU."""

from pathlib import Path

import pytest

from siliconshield.gui import setup as gui_setup


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
