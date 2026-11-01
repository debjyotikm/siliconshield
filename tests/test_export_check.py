"""The release checker must reject artifacts that directory allowlists used to accept."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "export_check", ROOT / "tools" / "export_check.py"
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

REVIEWED = checker.release_files()


def issues(name, content=b"authored example\n"):
    return checker.check(name, content, (), REVIEWED)


def test_reviewed_public_files_pass():
    assert issues("siliconshield/training/laya/build_security_data.py") == []
    assert (
        issues(
            "examples/security_cases.example.json", b'{"train_dev_cases": []}'
        )
        == []
    )


def test_unreviewed_and_artifact_files_are_rejected():
    assert any(
        "reviewed release list" in i for i in issues("docs/private-notes.md")
    )
    assert any(
        "artifact" in i
        for i in issues("siliconshield/results.csv", b"a,b\n1,2\n")
    )
    assert any(
        "only allowed as an .example." in i
        for i in issues("siliconshield/report.json", b"{}")
    )
    assert any(
        "only allowed as an .example." in i
        for i in issues("configs/runtime.json", b"{}")
    )


def test_reviewed_image_asset_passes_and_others_still_fail():
    banner = "docs/assets/siliconshield-banner.png"
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    assert issues(banner, png) == []
    assert any(
        "binary/non-UTF-8" in i for i in issues("docs/assets/other.png", png)
    )
    assert any(
        "unexpected format" in i for i in issues(banner, b"not image bytes\n")
    )
    assert any(
        "unexpected large" in i
        for i in issues(banner, png + b"\x00" * 512 * 1024)
    )


def test_private_content_patterns_are_rejected():
    # Fixtures are assembled at run time so this test file itself stays clean.
    path = ("/ho" + "me/someone/lab").encode()
    uuid = ("123e4567-" + "e89b-12d3-a456-426614174000").encode()
    assert any(
        "workstation" in i for i in issues("README.md", b"see " + path + b"\n")
    )
    assert any(
        "UUID" in i
        for i in issues("tests/test_x.py", b'id = "' + uuid + b'"\n')
    )
    assert any(
        "experiment-result" in i
        for i in issues("docs/notes.md", b'"training_report": {"loss": 1}\n')
    )
