"""Workflow contracts and Verilator integration on authored RTL cases."""

import hashlib
import json
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from siliconshield.app import laya_assessor
from siliconshield.app.demo_cases import CASES
from siliconshield.app.workflow import (
    apply_patch,
    investigate,
    relevant_statements,
)
from siliconshield.research_loop.episodes.contracts import digest

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


@requires_verilator
def test_lock_bypass_second_path_is_demonstrated_and_fixed(tmp_path):
    report = run_case("lock-bypass-second-path", tmp_path)
    assert report["conclusion"]["status"] == "demonstrated_violation_fixed"
    assert report["checks"]["original"][0]["outcome"] == "fail"


@requires_verilator
def test_legitimate_access_is_not_flagged(tmp_path):
    report = run_case(
        "legitimate-access", tmp_path, label="satisfies_requirement"
    )
    assert report["conclusion"]["status"] == "no_violation_found"


@requires_verilator
def test_missing_context_stays_unresolved(tmp_path):
    report = run_case("missing-context", tmp_path, label="insufficient_context")
    assert report["conclusion"]["status"] == "insufficient_context"
    assert report["conclusion"]["requirement_verified"] is False
    assert report["checks"]["original"][0]["outcome"] == "compile_error"


def test_relevant_statements_identify_unguarded_write():
    case = CASES["unauthorized-write"]
    findings = relevant_statements(
        case["rtl"], case["protected_signals"], case["control_signals"]
    )
    writes = [f for f in findings if "wdata" in f["text"]]
    assert writes and writes[0]["guard"]["mentions_controls"] == ["wr_en"]


def test_patch_anchor_must_exist():
    with pytest.raises(ValueError, match="Patch anchor not found"):
        apply_patch("module x; endmodule", [("absent", "replacement")])


def test_failure_beats_pass_and_exit_code_is_honored():
    from siliconshield.app.workflow import classify_markers

    fail_then_pass = {
        "exit_code": 0,
        "timed_out": False,
        "stdout": ["CHECK_FAIL a", "CHECK_PASS b"],
    }
    assert classify_markers(fail_then_pass)[0] == "fail"
    pass_then_crash = {"exit_code": 1, "stdout": ["CHECK_PASS a"]}
    assert classify_markers(pass_then_crash)[0] == "error"
    assert (
        classify_markers(
            {"exit_code": 0, "timed_out": False, "stdout": ["CHECK_PASS a"]}
        )[0]
        == "pass"
    )
    assert (
        classify_markers(
            {"exit_code": 0, "timed_out": False, "stdout": ["no marker"]}
        )[0]
        == "inconclusive"
    )


def test_abnormal_execution_is_error_not_failure():
    from siliconshield.app.workflow import classify_markers

    timed_out_with_fail = {
        "exit_code": None,
        "timed_out": True,
        "stdout": ["CHECK_FAIL a"],
    }
    label, evidence = classify_markers(timed_out_with_fail)
    assert label == "error"
    assert evidence == []
    crashed_with_fail = {
        "exit_code": 1,
        "timed_out": False,
        "stdout": ["CHECK_FAIL a"],
    }
    label, evidence = classify_markers(crashed_with_fail)
    assert label == "error"
    assert evidence == []
    missing_exit_with_fail = {"stdout": ["CHECK_FAIL a"]}
    assert classify_markers(missing_exit_with_fail)[0] == "error"


def test_violation_survives_unrelated_compile_failure():
    from siliconshield.app.workflow import _conclude

    original = [
        {"check": "req", "kind": "requirement", "outcome": "fail"},
        {"check": "reg", "kind": "regression", "outcome": "compile_error"},
    ]
    conclusion = _conclude(
        {}, original, [], {"label": None, "status": "unavailable"}
    )
    assert conclusion["status"] == "demonstrated_violation_unfixed"
    assert conclusion["incomplete_checks"] == ["reg"]


def test_custom_case_without_bound_checks_is_unverified(tmp_path):
    case = {
        "name": "custom",
        "title": "Custom",
        "rtl": CASES["legitimate-access"]["rtl"],
        "requirement": "cfg_reg must remain zero after reset, including during privileged writes.",
        "assumptions": "",
        "protected_signals": [],
        "control_signals": [],
        "checks": [],
        "patch": [],
    }
    report = investigate(
        case,
        tmp_path / "custom",
        assessor=FixtureAssessor("satisfies_requirement"),
    )
    assert report["conclusion"]["status"] == "requirement_not_verified"
    assert report["conclusion"]["requirement_verified"] is False


@requires_verilator
def test_relative_output_path_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = investigate(
        CASES["legitimate-access"],
        "relative-out",
        assessor=FixtureAssessor("satisfies_requirement"),
    )
    assert report["conclusion"]["status"] == "no_violation_found"
    assert (tmp_path / "relative-out/report.json").exists()


@requires_verilator
def test_proposer_is_called_on_violation_without_authored_answers(tmp_path):
    case = dict(CASES["unauthorized-write"])
    case["patch"] = []  # no authored solution available to leak
    seen = {}

    def proposer(context):
        seen.update(context)
        return [
            (
                "else if (wr_en) cfg_reg <= wdata;",
                "else if (wr_en && privileged) cfg_reg <= wdata;",
            )
        ]

    report = investigate(
        case,
        tmp_path / "proposed",
        assessor=FixtureAssessor("violates_requirement"),
        proposer=proposer,
    )
    assert report["conclusion"]["status"] == "demonstrated_violation_fixed"
    assert (
        "patch" not in seen
        and "expected_original" not in seen
        and "expected_patched" not in seen
    )
    assert seen["laya_assessment"]["label"] == "violates_requirement"
    assert seen["security_assessment"] == seen["laya_assessment"]
    assert seen["checks"][0]["outcome"] in ("fail", "pass")


def test_truncated_laya_assessment_is_rejected(tmp_path, monkeypatch):
    import subprocess as sp

    from siliconshield.app.laya_assessor import ExternalLayaAssessor

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.safetensors").write_text("fixture-weights")
    payload = {
        "label": "satisfies_requirement",
        "probabilities": {
            "satisfies_requirement": 0.9,
            "violates_requirement": 0.05,
            "insufficient_context": 0.05,
        },
        "usage": {
            "input_tokens": 5000,
            "state_tokens": 3000,
            "state_tokens_dropped": 1100,
            "truncated": True,
            "truncated_questions": [],
        },
    }
    monkeypatch.setattr(
        sp,
        "run",
        lambda *a, **k: type(
            "R",
            (),
            {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""},
        )(),
    )
    assessor = ExternalLayaAssessor("/usr/bin/python3", checkpoint)
    result = assessor.assess("req", "assumptions", "rtl")
    assert result["status"] == "truncated" and result["label"] is None
    assert result["coverage"]["state_tokens_dropped"] == 1100
    assert result["model_id"] == "siliconshield-rtl"
    assert result["runtime"] == "laya"
    assert result["checkpoint"] == str(checkpoint.resolve())
    assert result["checkpoint_fingerprint"]["files"] == {
        "weights.safetensors": hashlib.sha256(b"fixture-weights").hexdigest()
    }


@requires_verilator
def test_patch_that_breaks_legitimate_writes_is_rejected(tmp_path):
    def proposer(context):
        return {
            "replacements": [
                {
                    "old": "else if (wr_en) cfg_reg <= wdata;",
                    "new": "else if (wr_en && privileged) cfg_reg <= 8'h00;",
                }
            ]
        }

    case = dict(CASES["unauthorized-write"], patch=[])
    report = investigate(
        case,
        tmp_path,
        assessor=FixtureAssessor("violates_requirement"),
        proposer=proposer,
    )
    conclusion = report["conclusion"]
    assert conclusion["status"] == "repair_rejected_regression"
    assert conclusion["requirement_repaired"] is True
    assert conclusion["repair_accepted"] is False
    assert conclusion["regression_original"] == "pass"
    assert conclusion["regression_patched"] == "fail"
    assert "Repair accepted: no" in (tmp_path / "report.md").read_text()
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["conclusion"] == conclusion


@pytest.mark.parametrize(
    "outcome", ["compile_error", "error", "inconclusive", None]
)
def test_incomplete_candidate_regression_prevents_acceptance(outcome):
    from siliconshield.app.workflow import _conclude

    original = [
        {"check": "req", "kind": "requirement", "outcome": "fail"},
        {"check": "reg", "kind": "regression", "outcome": "pass"},
    ]
    patched = [{"check": "req", "kind": "requirement", "outcome": "pass"}]
    if outcome is not None:
        patched.append(
            {"check": "reg", "kind": "regression", "outcome": outcome}
        )
    conclusion = _conclude({}, original, patched, {})
    assert conclusion["requirement_repaired"] is True
    assert conclusion["repair_accepted"] is False
    assert conclusion["status"] == "repair_verification_incomplete"


def test_missing_candidate_requirement_cannot_be_accepted():
    from siliconshield.app.workflow import _conclude

    original = [
        {"check": "req-a", "kind": "requirement", "outcome": "fail"},
        {"check": "req-b", "kind": "requirement", "outcome": "pass"},
    ]
    patched = [{"check": "req-a", "kind": "requirement", "outcome": "pass"}]
    conclusion = _conclude({}, original, patched, {})
    assert conclusion["requirement_repaired"] is False
    assert conclusion["repair_accepted"] is False


@pytest.mark.parametrize(
    "kind,outcome,verified",
    [
        ("requirement", "compile_error", False),
        ("requirement", "error", False),
        ("requirement", "inconclusive", False),
        ("regression", "pass", False),
        ("requirement", "pass", True),
        ("requirement", "fail", True),
    ],
)
def test_verification_flag_requires_conclusive_requirement_checks(
    kind, outcome, verified
):
    from siliconshield.app.workflow import _conclude

    conclusion = _conclude(
        {}, [{"check": "check", "kind": kind, "outcome": outcome}], [], {}
    )
    assert conclusion["requirement_verified"] is verified
    if kind == "regression":
        assert conclusion["status"] == "requirement_not_verified"


def test_partial_requirement_evidence_keeps_violation_without_claiming_completion():
    from siliconshield.app.workflow import _conclude

    original = [
        {"check": "req-a", "kind": "requirement", "outcome": "fail"},
        {"check": "req-b", "kind": "requirement", "outcome": "compile_error"},
    ]
    conclusion = _conclude({}, original, [], {})
    assert conclusion["status"] == "demonstrated_violation_unfixed"
    assert conclusion["requirement_verified"] is False


@pytest.mark.parametrize(
    "replacement",
    [
        ("", "inserted"),
        (None, "new"),
        ("gate", None),
        ("gate", 1),
        ("gate",),
        "ab",
    ],
)
def test_invalid_replacement_pairs_are_rejected(replacement):
    with pytest.raises(ValueError):
        apply_patch("gate", [replacement])


@pytest.mark.parametrize("rtl,anchor", [("gate gate", "gate"), ("aaa", "aa")])
def test_ambiguous_anchors_include_overlapping_matches(rtl, anchor):
    with pytest.raises(ValueError, match="ambiguous"):
        apply_patch(rtl, [(anchor, "changed")])


def test_unique_anchor_can_delete_text():
    patched, applied = apply_patch("before remove after", [("remove ", "")])
    assert patched == "before after"
    assert applied == [{"old": "remove ", "new": ""}]


@pytest.mark.parametrize(
    "proposal",
    [
        None,
        {},
        [],
        {"replacements": "ab"},
        {"replacements": [{"old": "gate"}]},
        {"rtl": ""},
        {"rtl": "module topmodule; endmodule", "replacements": []},
    ],
)
def test_malformed_proposal_shapes_are_rejected(proposal):
    from siliconshield.app.workflow import _normalize_proposal

    with pytest.raises(ValueError):
        _normalize_proposal(proposal, "gate")


@pytest.mark.parametrize("wrapped", [False, True])
def test_full_text_proposals_have_explicit_provenance(wrapped):
    from siliconshield.app.workflow import _normalize_proposal

    replacement = "module different; endmodule"
    proposal = {"rtl": replacement} if wrapped else replacement
    patched, applied = _normalize_proposal(
        proposal, "module original; endmodule"
    )
    assert patched == replacement
    assert applied == [{"full_text": True}]


def test_invalid_muse_proposal_is_reported_without_authored_fallback(
    tmp_path, monkeypatch
):
    from siliconshield.agents.muse_proposer import make_muse_proposer
    from siliconshield.app import workflow

    calls = []

    class Client:
        def complete(self, messages, tools=()):
            assert len(messages) == 2 and not tools
            return {
                "content": '{"replacements": [{"old": "", "new": "injected"}]}'
            }

    def check(rtl, definition, *args):
        calls.append(definition["name"])
        return {
            "check": definition["name"],
            "kind": definition["kind"],
            "outcome": "fail"
            if definition["kind"] == "requirement"
            else "pass",
        }

    monkeypatch.setattr(workflow, "run_verilator_check", check)
    monkeypatch.setattr(workflow, "tool_versions", lambda: {})
    proposer = make_muse_proposer("fixture", "base-muse", client=Client())
    report = investigate(
        CASES["unauthorized-write"],
        tmp_path,
        assessor=FixtureAssessor("violates_requirement"),
        proposer=proposer,
    )
    assert report["conclusion"]["repair_accepted"] is False
    assert "nonempty" in report["conclusion"]["patch_error"]
    assert report["checks"]["patched"] == []
    assert len(calls) == len(CASES["unauthorized-write"]["checks"])
    assert not (tmp_path / "candidate").exists()


@pytest.mark.parametrize("as_bytes", [False, True])
def test_timeout_is_error_without_counterexample_and_logs(
    tmp_path, monkeypatch, as_bytes
):
    import subprocess

    from siliconshield.app import workflow

    stdout = "CHECK_FAIL concrete violation\n" + "later output\n" * 10
    stderr = "simulation diagnostic\n"
    outputs = iter(
        [
            subprocess.CompletedProcess([], 0, "build complete\n", ""),
            subprocess.TimeoutExpired(
                "sim",
                120,
                output=stdout.encode() if as_bytes else stdout,
                stderr=stderr.encode() if as_bytes else stderr,
            ),
        ]
    )

    def run(*args, **kwargs):
        result = next(outputs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(subprocess, "run", run)
    check = {
        "name": "req",
        "kind": "requirement",
        "testbench": "module tb; endmodule",
    }
    result = workflow.run_verilator_check(
        "module topmodule; endmodule", check, tmp_path, "timeout"
    )
    # Partial CHECK_FAIL lines from a timed-out run are evidence of
    # an incomplete tool execution, not of a demonstrated violation.
    assert result["outcome"] == "error"
    assert result["counterexample"] == []
    assert result["run"]["timed_out"] is True
    assert result["run"]["exit_code"] is None
    assert "TimeoutExpired" in result["run"]["error"]
    assert result["run"]["stdout_tail"] == ["later output"] * 8
    assert result["run"]["stderr"] == ["simulation diagnostic"]
    log = (tmp_path / "timeout/sim.log").read_text()
    assert (
        "CHECK_FAIL concrete violation" in log
        and "simulation diagnostic" in log
    )
    conclusion = workflow._conclude({}, [result], [], {})
    assert conclusion["status"] == "inconclusive"
    assert conclusion["requirement_verified"] is False
    assert conclusion["incomplete_checks"] == ["req"]


def test_timeout_without_output_is_recorded_as_incomplete(monkeypatch):
    import subprocess

    from siliconshield.app.workflow import _run, classify_markers

    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("sim", 120)

    monkeypatch.setattr(subprocess, "run", run)
    result = _run(["sim"], "/tmp")
    assert result["stdout"] == [] and result["stderr"] == []
    assert result["timed_out"] is True
    assert classify_markers(result)[0] == "error"


def test_report_carries_security_assessment_with_legacy_alias(tmp_path):
    case = {
        "name": "custom",
        "title": "Custom",
        "rtl": "module topmodule; endmodule",
        "requirement": "Nothing must happen.",
        "assumptions": "",
        "protected_signals": [],
        "control_signals": [],
        "checks": [],
        "patch": [],
    }
    report = investigate(
        case,
        tmp_path / "custom",
        assessor=FixtureAssessor("satisfies_requirement"),
    )
    assert report["security_assessment"] == report["laya_assessment"]
    assert report["security_assessment"]["label"] == ("satisfies_requirement")
    conclusion = report["conclusion"]
    assert conclusion["security_label"] == conclusion["laya_label"]
    assert conclusion["security_status"] == conclusion["laya_status"]
    heading = (tmp_path / "custom/report.md").read_text()
    assert "## Security model assessment" in heading
    assert "## LAYA assessment" not in heading


def test_report_assessment_reads_legacy_reports():
    from siliconshield.app.workflow import report_assessment

    legacy = {"label": "violates_requirement"}
    assert report_assessment({"laya_assessment": legacy}) == legacy
    both = {
        "security_assessment": {"label": "satisfies_requirement"},
        "laya_assessment": legacy,
    }
    assert report_assessment(both)["label"] == "satisfies_requirement"
    assert report_assessment({}) == {}


def test_assessor_names_share_one_backend():
    from siliconshield.app.laya_assessor import (
        ExternalLayaAssessor,
        LayaBackend,
        SecurityAssessor,
    )

    assert SecurityAssessor is LayaBackend
    assert ExternalLayaAssessor is LayaBackend


def test_from_environment_prefers_security_vars(monkeypatch):
    from siliconshield.app.laya_assessor import SecurityAssessor

    for var in (
        "SILICONSHIELD_SECURITY_PYTHON",
        "SILICONSHIELD_SECURITY_CHECKPOINT",
        "SILICONSHIELD_LAYA_PYTHON",
        "SILICONSHIELD_LAYA_CHECKPOINT",
    ):
        monkeypatch.delenv(var, raising=False)
    assert SecurityAssessor.from_environment() is None
    monkeypatch.setenv("SILICONSHIELD_LAYA_PYTHON", "/legacy/python")
    monkeypatch.setenv("SILICONSHIELD_LAYA_CHECKPOINT", "/legacy/checkpoint")
    legacy = SecurityAssessor.from_environment()
    assert legacy.python == "/legacy/python"
    monkeypatch.setenv("SILICONSHIELD_SECURITY_PYTHON", "/new/python")
    monkeypatch.setenv("SILICONSHIELD_SECURITY_CHECKPOINT", "/new/checkpoint")
    current = SecurityAssessor.from_environment()
    assert current.python == "/new/python"
    assert str(current.checkpoint) == "/new/checkpoint"


def test_unavailable_assessor_carries_model_identity():
    from siliconshield.app.laya_assessor import UnavailableAssessor

    result = UnavailableAssessor().assess("req", "assumptions", "rtl")
    assert result["status"] == "unavailable"
    assert result["model_id"] == "siliconshield-rtl"
    assert result["runtime"] == "laya"
    assert result["checkpoint"] is None
    assert result["checkpoint_fingerprint"] is None
    assert result["checkpoint_error"] is None
    assert "SILICONSHIELD_SECURITY_PYTHON" in result["detail"]


def test_backend_error_carries_model_identity(tmp_path, monkeypatch):
    import subprocess as sp

    from siliconshield.app.laya_assessor import SecurityAssessor

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.safetensors").write_text("fixture-weights")
    monkeypatch.setattr(
        sp,
        "run",
        lambda *a, **k: type(
            "R",
            (),
            {"returncode": 1, "stdout": "", "stderr": "boom"},
        )(),
    )
    result = SecurityAssessor("/usr/bin/python3", checkpoint).assess(
        "req", "assumptions", "rtl"
    )
    assert result["status"] == "error"
    assert result["model_id"] == "siliconshield-rtl"
    assert result["runtime"] == "laya"
    assert result["checkpoint"] == str(checkpoint.resolve())
    assert result["checkpoint_error"] is None
    assert "laya_worker.py" in result["detail"]


def test_missing_checkpoint_refuses_without_spawning_worker(
    tmp_path, monkeypatch
):
    import subprocess as sp

    from siliconshield.app.laya_assessor import SecurityAssessor

    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        raise AssertionError("worker must not spawn")

    monkeypatch.setattr(sp, "run", run)
    result = SecurityAssessor("/usr/bin/python3", tmp_path / "absent").assess(
        "req", "assumptions", "rtl"
    )
    assert result["status"] == "error"
    assert result["checkpoint_fingerprint"] is None
    assert "does not exist" in result["checkpoint_error"]
    assert calls == []


def test_file_checkpoint_fingerprint_covers_its_bytes(tmp_path, monkeypatch):
    import subprocess as sp

    from siliconshield.app.laya_assessor import SecurityAssessor

    weights = tmp_path / "model.safetensors"
    weights.write_text("fixture-weights")
    monkeypatch.setattr(
        sp,
        "run",
        lambda *a, **k: type(
            "R",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps(
                    {
                        "label": "satisfies_requirement",
                        "probabilities": None,
                        "usage": {},
                    }
                ),
                "stderr": "",
            },
        )(),
    )
    assessor = SecurityAssessor("/usr/bin/python3", weights)
    first = assessor.assess("req", "assumptions", "rtl")
    second = assessor.assess("req", "assumptions", "rtl")
    assert first["status"] == "assessed"
    assert first["checkpoint_fingerprint"]["files"] == {
        "model.safetensors": hashlib.sha256(b"fixture-weights").hexdigest()
    }
    assert first["checkpoint_fingerprint"] == second["checkpoint_fingerprint"]


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


@pytest.mark.parametrize("directory", [False, True])
def test_replaced_checkpoint_tracks_each_worker_load(
    tmp_path: Path, fixture_laya_runtime: Path, directory: bool
) -> None:
    checkpoint = tmp_path / "checkpoint"
    if directory:
        checkpoint.mkdir()
    weights = checkpoint / "weights.bin" if directory else checkpoint
    weights.write_bytes(b"generation-A")
    assessor = laya_assessor.SecurityAssessor(
        sys.executable, checkpoint, laya_root=str(fixture_laya_runtime)
    )
    first = assessor.assess("req", "assumptions", "rtl")
    weights.write_bytes(b"generation-B")
    second = assessor.assess("req", "assumptions", "rtl")

    for result, content, label in (
        (first, b"generation-A", "satisfies_requirement"),
        (second, b"generation-B", "violates_requirement"),
    ):
        assert result["status"] == "assessed", result
        assert result["label"] == label
        files = {weights.name: hashlib.sha256(content).hexdigest()}
        assert result["checkpoint_fingerprint"] == {
            "files": files,
            "fingerprint": digest(files),
        }
        assert result["checkpoint"] == str(checkpoint.resolve())
        assert result["checkpoint_error"] is None
    assert first["checkpoint_fingerprint"] != second["checkpoint_fingerprint"]


def test_retargeted_checkpoint_symlink_tracks_each_worker_load(
    tmp_path: Path, fixture_laya_runtime: Path
) -> None:
    first_checkpoint = tmp_path / "generation-a"
    second_checkpoint = tmp_path / "generation-b"
    for checkpoint, content in (
        (first_checkpoint, b"generation-A"),
        (second_checkpoint, b"generation-B"),
    ):
        checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(content)
    link = tmp_path / "current"
    link.symlink_to(first_checkpoint, target_is_directory=True)
    assessor = laya_assessor.SecurityAssessor(
        sys.executable, link, laya_root=str(fixture_laya_runtime)
    )
    first = assessor.assess("req", "assumptions", "rtl")
    link.unlink()
    link.symlink_to(second_checkpoint, target_is_directory=True)
    second = assessor.assess("req", "assumptions", "rtl")
    assert first["status"] == second["status"] == "assessed"
    assert first["label"] == "satisfies_requirement"
    assert second["label"] == "violates_requirement"
    assert first["checkpoint"] == str(first_checkpoint)
    assert second["checkpoint"] == str(second_checkpoint)
    assert first["checkpoint_fingerprint"] != second["checkpoint_fingerprint"]
    assert second["checkpoint_fingerprint"] == (
        laya_assessor.checkpoint_manifest(second_checkpoint)
    )


def test_symlink_retarget_at_spawn_keeps_the_resolved_checkpoint(
    tmp_path: Path, fixture_laya_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_weights = tmp_path / "generation-a.bin"
    second_weights = tmp_path / "generation-b.bin"
    first_weights.write_bytes(b"generation-A")
    second_weights.write_bytes(b"generation-B")
    link = tmp_path / "current"
    link.symlink_to(first_weights)
    run_worker = laya_assessor.subprocess.run

    def retarget_and_run(*args, **kwargs):
        assert json.loads(kwargs["input"])["checkpoint"] == str(first_weights)
        link.unlink()
        link.symlink_to(second_weights)
        return run_worker(*args, **kwargs)

    monkeypatch.setattr(laya_assessor.subprocess, "run", retarget_and_run)
    result = laya_assessor.SecurityAssessor(
        sys.executable, link, laya_root=str(fixture_laya_runtime)
    ).assess("req", "assumptions", "rtl")
    assert result["status"] == "assessed", result
    assert result["label"] == "satisfies_requirement"
    assert result["checkpoint"] == str(first_weights)
    assert result["checkpoint_fingerprint"] == (
        laya_assessor.checkpoint_manifest(first_weights)
    )


@pytest.mark.parametrize("remove", [False, True])
def test_checkpoint_change_during_worker_execution_refuses_judgment(
    tmp_path: Path,
    fixture_laya_runtime: Path,
    monkeypatch: pytest.MonkeyPatch,
    remove: bool,
) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"generation-A")
    run_worker = laya_assessor.subprocess.run

    def change_and_run(*args, **kwargs):
        if remove:
            weights.unlink()
        else:
            weights.write_bytes(b"generation-B")
        return run_worker(*args, **kwargs)

    monkeypatch.setattr(laya_assessor.subprocess, "run", change_and_run)
    result = laya_assessor.SecurityAssessor(
        sys.executable, weights, laya_root=str(fixture_laya_runtime)
    ).assess("req", "assumptions", "rtl")
    assert result["status"] == "error"
    assert result["label"] is None
    assert result["probabilities"] is None
    assert result["checkpoint_fingerprint"] is None
    expected = "does not exist" if remove else "changed during assessment"
    assert expected in result["checkpoint_error"]


def test_previously_missing_checkpoint_can_be_assessed_after_creation(
    tmp_path: Path, fixture_laya_runtime: Path
) -> None:
    weights = tmp_path / "weights.bin"
    assessor = laya_assessor.SecurityAssessor(
        sys.executable, weights, laya_root=str(fixture_laya_runtime)
    )
    first = assessor.assess("req", "assumptions", "rtl")
    assert first["status"] == "error"
    weights.write_bytes(b"generation-A")
    second = assessor.assess("req", "assumptions", "rtl")
    assert second["status"] == "assessed", second
    assert second["label"] == "satisfies_requirement"
    assert second["checkpoint_error"] is None


def test_deleted_checkpoint_refuses_before_another_worker_spawn(
    tmp_path: Path, fixture_laya_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"generation-A")
    assessor = laya_assessor.SecurityAssessor(
        sys.executable, weights, laya_root=str(fixture_laya_runtime)
    )
    assert assessor.assess("req", "assumptions", "rtl")["status"] == "assessed"
    weights.unlink()

    def refuse_spawn(*args, **kwargs):
        pytest.fail("Deleted checkpoint must be refused before worker spawn")

    monkeypatch.setattr(laya_assessor.subprocess, "run", refuse_spawn)
    result = assessor.assess("req", "assumptions", "rtl")
    assert result["status"] == "error"
    assert result["checkpoint_fingerprint"] is None
    assert "does not exist" in result["checkpoint_error"]


def test_cli_reports_security_label_with_deprecated_alias(
    tmp_path, monkeypatch, capsys
):
    import sys

    from siliconshield.app import cli

    for var in (
        "SILICONSHIELD_SECURITY_PYTHON",
        "SILICONSHIELD_SECURITY_CHECKPOINT",
        "SILICONSHIELD_LAYA_PYTHON",
        "SILICONSHIELD_LAYA_CHECKPOINT",
    ):
        monkeypatch.delenv(var, raising=False)
    rtl = tmp_path / "design.sv"
    rtl.write_text("module topmodule; endmodule")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "app",
            "--rtl",
            str(rtl),
            "--requirement",
            "Nothing must happen.",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    cli.main()
    shown = json.loads(capsys.readouterr().out)
    assert shown["conclusion"] == "requirement_not_verified"
    assert shown["security_label"] is None
    assert shown["laya"] == shown["security_label"]
