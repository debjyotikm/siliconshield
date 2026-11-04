"""One RTL security investigation, end to end.

RTL + requirement -> security-model assessment -> relevant-statement
investigation -> executable check (Verilator simulation) ->
candidate patch in a separate copy -> same checks rerun -> evidence report.

Conclusions distinguish a demonstrated violation (counterexample executed),
simulation finding no counterexample (not a proof), and unresolved checks.
A repair is accepted only after all configured checks pass.
"""

import difflib
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

from siliconshield.app.laya_assessor import (
    SecurityAssessor,
    UnavailableAssessor,
)
from siliconshield.research_loop.episodes.contracts import atomic_json
from siliconshield.schemas import INVESTIGATION_SCHEMA

TOOL_TIMEOUT = 300


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _output_lines(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return (value or "").splitlines()


def _run(command, cwd, timeout=TOOL_TIMEOUT):
    """Keep full output, including partial output on timeout; tails are for display.

    The child inherits only an allowlisted environment and runs under
    bounded resources, since checked RTL may be untrusted.
    """
    from siliconshield.research_loop.execution.sandbox import (
        limited_preexec,
        restricted_env,
    )

    try:
        run = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=restricted_env(),
            preexec_fn=limited_preexec(timeout),
        )
        stdout, stderr = _output_lines(run.stdout), _output_lines(run.stderr)
        return {
            "command": command,
            "exit_code": run.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_tail": stdout[-8:],
            "stderr_tail": stderr[-8:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _output_lines(exc.stdout), _output_lines(exc.stderr)
        return {
            "command": command,
            "exit_code": None,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_tail": stdout[-8:],
            "stderr_tail": stderr[-8:],
            "timed_out": True,
            "error": "%s: %s" % (type(exc).__name__, exc),
        }
    except OSError as exc:
        line = "%s: %s" % (type(exc).__name__, exc)
        return {
            "command": command,
            "exit_code": None,
            "stdout": [],
            "stderr": [line],
            "stdout_tail": [],
            "stderr_tail": [line],
            "timed_out": False,
            "error": line,
        }


def tool_versions():
    versions = {}
    for name in ("verilator", "yosys"):
        path = shutil.which(name)
        if path:
            run = _run([path, "--version"], cwd="/", timeout=30)
            versions[name] = (
                run["stdout_tail"] or run["stderr_tail"] or ["unknown"]
            )[0]
        else:
            versions[name] = None
    return versions


def relevant_statements(rtl, protected, controls):
    """Line-numbered assignments to protected signals and their guarding conditions.

    Matches run over the whole text, so an assignment split across
    lines still reads as one write; only whitespace may separate a
    signal from its operator, never another statement. Guard lookup
    starts from the write's first line.
    """
    lines = rtl.splitlines()
    findings = []
    for signal in protected:
        pattern = re.compile(
            r"\b%s\b\s*(?:\[[^\]]*\]\s*)*(<=|=(?!=|>))" % re.escape(signal)
        )
        for match in pattern.finditer(rtl):
            first = rtl.count("\n", 0, match.start()) + 1
            last = rtl.count("\n", 0, match.end(1)) + 1
            line_start = rtl.rfind("\n", 0, match.start()) + 1
            head = rtl[line_start : match.start(1)]
            if match[1] == "<=" and re.search(
                r"=(?!=|>)", re.sub(r"\([^()]*\)", " ", head)
            ):
                # A `<=` after a plain `=` in the same statement
                # compares; only assignments write the signal.
                continue
            index = first - 1
            guard = None
            candidates = (
                [index] + list(range(index - 1, max(-1, index - 4), -1))
                if re.search(r"\b(if|else if)\b", head)
                else list(range(index - 1, max(-1, index - 4), -1))
            )
            for back in candidates:
                if re.search(r"\b(if|else if)\b", lines[back]):
                    guard = {
                        "line": back + 1,
                        "text": lines[back].strip(),
                        "mentions_controls": [
                            c for c in controls if c in lines[back]
                        ],
                    }
                    break
            text = " ".join(line.strip() for line in lines[first - 1 : last])
            findings.append(
                {
                    "signal": signal,
                    "line": first,
                    "text": text,
                    "guard": guard,
                }
            )
    findings.sort(key=lambda finding: (finding["line"], finding["signal"]))
    return findings


def classify_markers(run):
    """Unambiguous verdict for ONE check: abnormal execution beats markers.

    A timeout, missing exit code, or nonzero exit means the tool did
    not complete, so even CHECK_FAIL lines in partial output read as
    an error without counterexample evidence — never as a confirmed
    security failure. Only a clean exit (code 0, no timeout) with
    CHECK_FAIL lines is a failure; failure still beats success.
    Evaluated over the complete output, never a tail.
    """
    if run.get("timed_out") or run.get("exit_code") is None:
        return "error", []
    if run.get("exit_code") != 0:
        return "error", []
    fails = [line for line in run.get("stdout", []) if "CHECK_FAIL" in line]
    passes = [line for line in run.get("stdout", []) if "CHECK_PASS" in line]
    if fails:
        return "fail", fails
    if passes:
        return "pass", []
    return "inconclusive", []


def run_verilator_check(rtl, check, workdir, label):
    """Compile RTL + the testbench bound to THIS check's requirement."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError("Check label is not a safe name: %r" % (label,))
    root = Path(workdir).resolve()
    folder = (root / label).resolve()
    if folder != root and root not in folder.parents:
        raise ValueError("Check label escapes its workdir: %r" % (label,))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "dut.sv").write_text(rtl)
    (folder / "tb.sv").write_text(check["testbench"])
    build = _run(
        [
            "verilator",
            "--binary",
            "--timing",
            "-j",
            "4",
            "--top-module",
            "tb",
            "-Mdir",
            str(folder / "obj"),
            str(folder / "dut.sv"),
            str(folder / "tb.sv"),
            "-o",
            "sim",
        ],
        cwd=folder,
    )
    (folder / "build.log").write_text(
        "\n".join(build["stdout"] + build["stderr"]) + "\n"
    )
    record = {
        "check": check["name"],
        "kind": check["kind"],
        "tool": "verilator",
        "requirement": check.get("requirement"),
        "build": build,
        "build_log": str(folder / "build.log"),
    }
    if build["exit_code"] != 0:
        text = " ".join(build["stdout"] + build["stderr"])
        record["outcome"] = "compile_error"
        record["missing_context"] = bool(
            re.search(
                r"Can't find definition|undeclared|not declared", text, re.I
            )
        )
        return record
    run = _run([str(folder / "obj" / "sim")], cwd=folder, timeout=120)
    (folder / "sim.log").write_text(
        "\n".join(run["stdout"] + run["stderr"]) + "\n"
    )
    record["run"] = run
    record["sim_log"] = str(folder / "sim.log")
    record["outcome"], record["counterexample"] = classify_markers(run)
    return record


def apply_patch(rtl, replacements):
    """Apply string replacements whose nonempty anchors each have one exact match."""
    if not isinstance(replacements, (list, tuple)) or not replacements:
        raise ValueError("Patch requires a nonempty list of replacement pairs")
    patched = rtl
    applied = []
    for item in replacements:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each replacement must contain old and new text")
        old, new = item
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError(
                "Patch anchors must be nonempty strings and replacements must be strings"
            )
        start = patched.find(old)
        if start < 0:
            raise ValueError("Patch anchor not found in RTL: " + old)
        if patched.find(old, start + 1) >= 0:
            raise ValueError(
                "Patch anchor is ambiguous; supply unique surrounding RTL: "
                + old
            )
        patched = patched.replace(old, new, 1)
        applied.append({"old": old, "new": new})
    return patched, applied


def _normalize_proposal(result, rtl):
    """Validate replacement pairs, or accept explicit full RTL for separate verification."""
    if isinstance(result, str) or isinstance(result, dict) and "rtl" in result:
        text = result if isinstance(result, str) else result["rtl"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("A full-text proposal must contain nonempty RTL")
        if isinstance(result, dict) and "replacements" in result:
            raise ValueError("Choose full RTL or replacements, not both")
        return text, [{"full_text": True}]
    pairs = result.get("replacements") if isinstance(result, dict) else result
    if not isinstance(pairs, (list, tuple)) or not pairs:
        raise ValueError("Proposal requires a nonempty list of replacements")
    normalized = []
    for item in pairs:
        if isinstance(item, dict):
            if set(item) != {"old", "new"}:
                raise ValueError(
                    "Each replacement object must contain only old and new text"
                )
            item = (item["old"], item["new"])
        normalized.append(item)
    return apply_patch(
        rtl, normalized
    )  # validates every anchor before application


def investigate(case, output, assessor=None, proposer=None):
    """Run the full investigation for one case dict; write report.json/report.md."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A rerun must not present the previous run's checks, candidate,
    # or reports as its own evidence: clear generated paths first.
    for generated in ("checks", "candidate", "report.json", "report.md"):
        target = output / generated
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.is_file() or target.is_symlink():
            target.unlink()
    rtl, requirement = case["rtl"], case["requirement"]
    assumptions = case.get("assumptions", "")
    assessor = (
        assessor or SecurityAssessor.from_environment() or UnavailableAssessor()
    )
    assessment = assessor.assess(requirement, assumptions, rtl)
    findings = relevant_statements(
        rtl, case.get("protected_signals", []), case.get("control_signals", [])
    )
    versions = tool_versions()
    # Every check is explicitly bound to this case's requirement and interface.
    checks = [
        {**check, "requirement": requirement}
        for check in case.get("checks", [])
    ]
    original = [
        run_verilator_check(
            rtl, check, output / "checks", "original-" + check["name"]
        )
        for check in checks
    ]
    patched_rtl, patch_record, patched, patch_error = None, [], [], None
    if _kind_verdict(original, "requirement") == "violation":
        # Model-facing context: no authored patch, no expected verdicts.
        context = {
            "case": case["name"],
            "title": case.get("title"),
            "rtl": rtl,
            "requirement": requirement,
            "assumptions": assumptions,
            "security_assessment": assessment,
            # Legacy alias of the same assessment.
            "laya_assessment": assessment,
            "relevant_statements": findings,
            "checks": [
                {
                    "check": c["check"],
                    "kind": c["kind"],
                    "outcome": c["outcome"],
                    "counterexample": c.get("counterexample", []),
                }
                for c in original
            ],
        }
        try:
            if proposer is not None:
                patched_rtl, patch_record = _normalize_proposal(
                    proposer(context), rtl
                )
                proposer_name = getattr(
                    proposer, "__name__", type(proposer).__name__
                )
            elif case.get("patch"):
                patched_rtl, patch_record = apply_patch(rtl, case["patch"])
                proposer_name = "authored-case-patch"
            else:
                proposer_name = None
        except (
            Exception
        ) as exc:  # invalid anchors or proposer failure: report, never guess
            patch_error = "%s: %s" % (type(exc).__name__, exc)
            proposer_name = (
                getattr(proposer, "__name__", type(proposer).__name__)
                if proposer is not None
                else "authored-case-patch"
            )
        if patched_rtl is not None:
            (output / "candidate").mkdir(exist_ok=True)
            (output / "candidate/topmodule.sv").write_text(patched_rtl)
            patched = [
                run_verilator_check(
                    patched_rtl,
                    check,
                    output / "checks",
                    "patched-" + check["name"],
                )
                for check in checks
            ]
    else:
        proposer_name = (
            getattr(proposer, "__name__", type(proposer).__name__)
            if proposer
            else None
        )
    conclusion = _conclude(case, original, patched, assessment)
    if patch_error:
        conclusion["patch_error"] = patch_error
    report = {
        "schema": INVESTIGATION_SCHEMA,
        "case": case["name"],
        "title": case.get("title"),
        "inputs": {
            "rtl_sha256": _sha(rtl),
            "requirement": requirement,
            "assumptions": assumptions,
            "rtl": rtl,
        },
        "tool_versions": versions,
        "security_assessment": assessment,
        # Legacy alias of the same assessment.
        "laya_assessment": assessment,
        "relevant_statements": findings,
        "checks": {"original": original, "patched": patched},
        "patch": {
            "applied": patch_record,
            "diff": "".join(
                difflib.unified_diff(
                    rtl.splitlines(True),
                    (patched_rtl or "").splitlines(True),
                    fromfile="original",
                    tofile="candidate",
                )
            )
            if patched_rtl
            else None,
            "proposer": proposer_name,
        },
        "conclusion": conclusion,
    }
    atomic_json(output / "report.json", report)
    _atomic_text(output / "report.md", _markdown(report))
    return report


def _atomic_text(path: Path, text: str) -> None:
    """Write text atomically so a crash never leaves a half report."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _kind_verdict(checks, kind):
    selected = [c for c in checks if c["kind"] == kind]
    if not selected:
        return "not_run"
    if any(c["outcome"] == "fail" for c in selected):
        return "violation" if kind == "requirement" else "fail"
    if all(c["outcome"] == "pass" for c in selected):
        return "pass"
    return "inconclusive"


def _conclude(case, original, patched, assessment):
    req_before = _kind_verdict(original, "requirement")
    reg_before = _kind_verdict(original, "regression")
    req_after = _kind_verdict(patched, "requirement") if patched else "not_run"
    reg_after = _kind_verdict(patched, "regression") if patched else "not_run"
    incomplete = [
        c["check"]
        for stage in (original, patched)
        for c in stage
        if c["outcome"] in ("compile_error", "error", "inconclusive")
        or c.get("run", {}).get("timed_out")
    ]
    required = {(c["kind"], c["name"]) for c in case.get("checks", [])}
    required.update((c["kind"], c["check"]) for c in original)
    required_requirements = {key for key in required if key[0] == "requirement"}
    verified_original = {
        (c["kind"], c["check"])
        for c in original
        if c["outcome"] in ("pass", "fail")
        and not c.get("run", {}).get("timed_out")
    }
    requirement_verified = (
        bool(required_requirements)
        and required_requirements <= verified_original
    )
    passed_candidate = {
        (c["kind"], c["check"])
        for c in patched
        if c["outcome"] == "pass" and not c.get("run", {}).get("timed_out")
    }
    requirement_repaired = (
        req_before == "violation"
        and req_after == "pass"
        and required_requirements <= passed_candidate
    )
    repair_accepted = (
        requirement_repaired
        and required <= passed_candidate
        and all(c["outcome"] == "pass" for c in patched)
    )
    missing = any(
        c.get("missing_context") for c in original if c["kind"] == "requirement"
    )
    if req_before == "not_run":
        status = "requirement_not_verified"
    elif req_before == "violation":
        # A demonstrated counterexample stands even if another check is incomplete.
        if repair_accepted:
            status = "demonstrated_violation_fixed"
        elif requirement_repaired:
            status = (
                "repair_rejected_regression"
                if reg_after == "fail"
                else "repair_verification_incomplete"
            )
        else:
            status = "demonstrated_violation_unfixed"
    elif missing or (
        assessment.get("label") == "insufficient_context"
        and req_before == "inconclusive"
    ):
        status = "insufficient_context"
    elif req_before == "pass" and reg_before in ("pass", "not_run"):
        status = "no_violation_found"
    elif reg_before == "fail":
        status = "regression_failed"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "requirement_verified": requirement_verified,
        "requirement_repaired": requirement_repaired,
        "repair_accepted": repair_accepted,
        "original_verdict": req_before,
        "patched_verdict": req_after,
        "regression_original": reg_before,
        "regression_patched": reg_after,
        "incomplete_checks": incomplete,
        "basis": "Executable Verilator checks bound to the stated requirement; "
        "simulation passes are not formal proofs.",
        "security_label": assessment.get("label"),
        "security_status": assessment.get("status"),
        "laya_label": assessment.get("label"),
        "laya_status": assessment.get("status"),
    }


def report_assessment(report):
    """Security assessment, accepting reports written before the rename."""
    return (
        report.get("security_assessment") or report.get("laya_assessment") or {}
    )


def _markdown(report):
    c = report["conclusion"]
    assessment = report_assessment(report)
    lines = [
        "# SiliconShield investigation: %s" % report["title"],
        "",
        "**Conclusion:** %s (original: %s, patched: %s)"
        % (c["status"], c["original_verdict"], c["patched_verdict"]),
        "",
        "Repair accepted: %s. Regression checks: original=%s, candidate=%s."
        % (
            "yes" if c["repair_accepted"] else "no",
            c["regression_original"],
            c["regression_patched"],
        ),
        "",
        "## Requirement",
        "",
        report["inputs"]["requirement"],
        "",
        "## Security model assessment",
        "",
        "- Status: %s; label: %s"
        % (assessment.get("status"), assessment.get("label")),
        "- Probabilities: %s" % assessment.get("probabilities"),
        "",
        "## Relevant statements",
        "",
    ]
    for finding in report["relevant_statements"]:
        lines.append(
            "- Line %d: `%s` (guard: %s)"
            % (
                finding["line"],
                finding["text"],
                finding["guard"]["text"] if finding["guard"] else "none found",
            )
        )
    lines += ["", "## Checks", ""]
    for stage in ("original", "patched"):
        for check in report["checks"][stage]:
            lines.append(
                "- [%s] %s: %s" % (stage, check["check"], check["outcome"])
            )
            for evidence in check.get("counterexample", []):
                lines.append("  - counterexample: `%s`" % evidence)
    if report["patch"]["diff"]:
        lines += [
            "",
            "## Candidate patch",
            "",
            "```diff",
            report["patch"]["diff"],
            "```",
        ]
    lines += [
        "",
        "## Basis and limits",
        "",
        c["basis"],
        "Tool versions: %s" % report["tool_versions"],
        "Full commands, exit codes, and outputs are in report.json.",
    ]
    return "\n".join(lines) + "\n"
