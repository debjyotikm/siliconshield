"""Workflow contracts and Verilator integration on authored RTL cases."""

import json
import shutil
import textwrap
from pathlib import Path

import pytest

from siliconshield.app.demo_cases import CASES
from siliconshield.app.workflow import (
    investigate,
)

requires_verilator = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="verilator not installed"
)


class FixtureAssessor:
    name = "fixture"

    def __init__(self, label):
        self.label = label

    def assess(self, requirement, assumptions, rtl):
        return {
            "assessor": self.name,
            "status": "assessed",
            "label": self.label,
            "probabilities": None,
        }


def run_case(name, tmp_path, label="violates_requirement"):
    report = investigate(
        CASES[name], tmp_path / name, assessor=FixtureAssessor(label)
    )
    saved = json.loads((tmp_path / name / "report.json").read_text())
    assert saved["schema"] == "siliconshield-investigation-v1"
    assert (tmp_path / name / "report.md").exists()
    return report


@requires_verilator
def test_unauthorized_write_is_demonstrated_and_fixed(tmp_path):
    report = run_case("unauthorized-write", tmp_path)
    assert report["conclusion"]["status"] == "demonstrated_violation_fixed"
    assert report["conclusion"]["repair_accepted"] is True
    original = {c["check"]: c for c in report["checks"]["original"]}
    assert original["unprivileged_write_blocked"]["outcome"] == "fail"
    assert original["unprivileged_write_blocked"]["counterexample"]
    assert all(c["outcome"] == "pass" for c in report["checks"]["patched"])
    assert (
        "wr_en && privileged"
        in (tmp_path / "unauthorized-write/candidate/topmodule.sv").read_text()
    )


@pytest.fixture
def fixture_laya_runtime(tmp_path: Path) -> Path:
    """Supply a CPU-only loader to the actual assessment subprocess.

    The loader reads authored bytes on every invocation and changes its
    verdict with those bytes, exposing stale identity across worker loads.
    """
    root = tmp_path / "fixture-runtime"
    package = root / "laya"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "agent.py").write_text(
        textwrap.dedent("""\
            from pathlib import Path

            class FixtureAgent:
                def __init__(self, checkpoint):
                    artifact = Path(checkpoint)
                    if artifact.is_dir():
                        artifact = artifact / "weights.bin"
                    self.label = (
                        "satisfies_requirement"
                        if artifact.read_bytes() == b"generation-A"
                        else "violates_requirement"
                    )

                def predict_batch(self, states, questions, sort_by_length):
                    labels = (
                        "satisfies_requirement",
                        "violates_requirement",
                        "insufficient_context",
                    )
                    values = {key: float(key == self.label) for key in labels}
                    return [{
                        "answers": {"security": {"probabilities": values}},
                        "usage": {},
                    }]

            def load(checkpoint):
                return FixtureAgent(checkpoint)
            """)
    )
    return root
