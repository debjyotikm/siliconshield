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
