"""Synthetic tests for Muse API specialists behind skills."""

import json
from pathlib import Path

import pytest
from test_glimmer_specialists import VERDICT_TEXT, _glimmer_registry

from siliconshield.research_loop.procedures import muse_specialists


class _StubMuse:
    """Canned Muse CLI completions for specialist tests."""

    def __init__(self, text):
        self.text = text
        self.calls = []

    def complete(self, messages, tools=()):
        self.calls.append((messages, tools))
        return {"content": self.text}


def _args(tmp_path, **overrides):
    parser = muse_specialists.build_parser()
    design = tmp_path / "demo.sv"
    design.write_text("module topmodule; endmodule\n")
    values = {
        "--registry": _glimmer_registry(tmp_path),
        "--expert": "glimmer-a",
        "--design": str(design),
        "--requirement": "Hold the count.",
        "--assumptions": "Two-state RTL.",
        "--model": "synthetic-muse",
        "--output": str(tmp_path / "receipt"),
    }
    values.update(overrides)
    argv = []
    for key, value in values.items():
        argv.extend([key, str(value)])
    return parser.parse_args(argv), values["--registry"]


def test_direct_verdict_receipt_and_brief(tmp_path):
    args, registry_path = _args(tmp_path)
    chat = _StubMuse(VERDICT_TEXT)
    report = muse_specialists.run_muse_specialists(
        args, client_factory=lambda: chat
    )
    assert report["muse_model"] == "synthetic-muse"
    response = report["responses"][0]
    assert response["label"] == "violates_requirement"
    assert response["scores"]["violates_requirement"] == 0.8
    registry = json.loads(Path(registry_path).read_text())
    pinned = [
        model["expected_files"]["SKILL.md"]
        for model in registry["models"]
        if model["id"] == "glimmer-a"
    ][0]
    assert response["skill_sha256"] == pinned
    assert response["skill_checked"] is True
    assert response["weight_identity"] == muse_specialists.WEIGHT_IDENTITY
    assert "checkpoint_sha256" not in response
    assert report["errors"] == []
    receipt = json.loads((tmp_path / "receipt" / "receipt.json").read_text())
    assert receipt["responses"][0]["label"] == "violates_requirement"
    assert receipt["muse_model"] == "synthetic-muse"


def test_main_brief_wiring_without_live_calls(tmp_path, capsys, monkeypatch):
    def _fake_run(args):
        return {
            "muse_model": "synthetic-muse",
            "inference_seconds": 1.5,
            "responses": [
                {
                    "expert_id": "glimmer-a",
                    "label": "satisfies_requirement",
                    "scores": {
                        "satisfies_requirement": 0.9,
                        "violates_requirement": 0.05,
                        "insufficient_context": 0.05,
                    },
                }
            ],
        }

    monkeypatch.setattr(muse_specialists, "run_muse_specialists", _fake_run)
    assert (
        muse_specialists.main(
            [
                "--registry",
                "registry.json",
                "--expert",
                "glimmer-a",
                "--design",
                "demo.sv",
                "--requirement",
                "Hold the count.",
                "--model",
                "synthetic-muse",
                "--output",
                str(tmp_path),
                "--brief",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "glimmer-a via synthetic-muse: satisfies_requirement" in out


def test_brief_printer_hides_local_paths():
    text = muse_specialists.format_brief(
        {
            "muse_model": "synthetic-muse",
            "inference_seconds": 1.5,
            "responses": [
                {
                    "expert_id": "glimmer-a",
                    "label": "satisfies_requirement",
                    "scores": {
                        "satisfies_requirement": 0.9,
                        "violates_requirement": 0.05,
                        "insufficient_context": 0.05,
                    },
                }
            ],
        }
    )
    assert "glimmer-a via synthetic-muse: satisfies_requirement" in text
    assert "satisfies_requirement: 0.9000" in text
    assert "provider-managed" in text
    assert "/tmp" not in text


def test_unknown_expert_and_tampered_skill(tmp_path):
    args, _registry_path = _args(tmp_path, **{"--expert": "nobody"})
    chat = _StubMuse(VERDICT_TEXT)
    with pytest.raises(ValueError, match="Unknown expert"):
        muse_specialists.run_muse_specialists(args, client_factory=lambda: chat)
    assert chat.calls == []
    second = tmp_path / "second"
    second.mkdir()
    args, _registry_path = _args(second)
    (second / "glimmer-a" / "SKILL.md").write_text("tampered skill")
    with pytest.raises(ValueError, match="skill digest mismatch"):
        muse_specialists.run_muse_specialists(args, client_factory=lambda: chat)


def test_missing_muse_cli_is_a_clean_error(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    args, _registry_path = _args(tmp_path)
    with pytest.raises(ValueError, match="muse.*CLI not found"):
        muse_specialists.run_muse_specialists(args)


def test_scope_and_arbitrate_are_not_offered():
    parser = muse_specialists.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--registry", "r", "--expert", "e", "--scope", "x"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--registry", "r", "--expert", "e", "--arbitrate"])
