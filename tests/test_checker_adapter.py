"""Fixed solver-backed checker: registry, evidence, and refusals."""

import json
import shutil

import pytest

from siliconshield.training.laya import build_security_data as data
from siliconshield.training.laya.checker_adapter import (
    REGISTERED_ASSUMPTIONS,
    FixedChecker,
    load_contracts,
)
from siliconshield.training.laya.proposer_ingest import (
    ProposalError,
    ingest_proposals,
    load_current,
    validate_evidence,
)

requires_solver = pytest.mark.skipif(
    shutil.which("yosys") is None or shutil.which("iverilog") is None,
    reason="yosys/iverilog not installed",
)

RTL = "module topmodule; endmodule"
REQUIREMENT = "The sealed register must hold still."
SEQ = REGISTERED_ASSUMPTIONS["sequential"]
COMB = REGISTERED_ASSUMPTIONS["combinational"]
HOLE_RTL = "module topmodule; wire guard_expr; assign x = guard_expr; endmodule"
HOLE = {"name": "guard_expr", "width": 1}


def contracts_file(tmp_path, registry=None):
    if registry is None:
        registry = {
            "c1": {
                "requirement": REQUIREMENT,
                "assertion": "always @* assert(1);",
                "scope": "sequential",
                "reset": "rst_n",
            }
        }
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(registry))
    return path


def comb_contracts_file(tmp_path):
    return contracts_file(
        tmp_path,
        {
            "c2": {
                "requirement": "The output must settle.",
                "assertion": "always @* assert(1);",
                "scope": "combinational",
                "reset": "",
            }
        },
    )


def proposal(**extra):
    row = {
        "id": "p1",
        "contract_id": "c1",
        "requirement": REQUIREMENT,
        "assumptions": SEQ,
        "rtl": RTL,
        "kind": "implementation",
    }
    row.update(extra)
    return row


@pytest.fixture
def tool_paths(tmp_path):
    """Executable stand-ins so stubbed checks pass tool resolution."""
    paths = {}
    for name in ("yosys", "iverilog"):
        binary = tmp_path / name
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        paths[name] = str(binary)
    return paths


def stub_result(verdict, basis="stub"):
    return {
        "accepted": True,
        "verdict": verdict,
        "basis": basis,
        "evidence_id": "stub-evidence",
        "solver_version": "stub-solver",
        "checker_sha256": "ab",
        "design_sha256": "cd",
        "assertion_sha256": "ef",
    }


def stub_tools(monkeypatch, verify=None):
    monkeypatch.setattr(
        data, "compile_rtl", lambda *args, **kwargs: {"returncode": 0}
    )
    monkeypatch.setattr(
        data,
        "verify",
        verify or (lambda *args, **kwargs: stub_result(data.LABELS[0])),
    )


def test_registry_validation(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        load_contracts(contracts_file(tmp_path, {}))
    with pytest.raises(ValueError, match="exactly one assertion"):
        load_contracts(
            contracts_file(
                tmp_path,
                {
                    "c1": {
                        "requirement": REQUIREMENT,
                        "assertion": "assert(a); assert(b);",
                        "scope": "sequential",
                        "reset": "rst_n",
                    }
                },
            )
        )
    with pytest.raises(ValueError, match="share one requirement"):
        load_contracts(
            contracts_file(
                tmp_path,
                {
                    "c1": {
                        "requirement": REQUIREMENT,
                        "assertion": "always @* assert(1);",
                        "scope": "sequential",
                        "reset": "rst_n",
                    },
                    "c2": {
                        "requirement": REQUIREMENT,
                        "assertion": "always @* assert(1);",
                        "scope": "sequential",
                        "reset": "rst_n",
                    },
                },
            )
        )
    with pytest.raises(ValueError, match="registered scope"):
        load_contracts(
            contracts_file(
                tmp_path,
                {
                    "c1": {
                        "requirement": REQUIREMENT,
                        "assertion": "always @* assert(1);",
                        "scope": "quantum",
                        "reset": "rst_n",
                    }
                },
            )
        )
    with pytest.raises(ValueError, match="disagrees with its scope"):
        load_contracts(
            contracts_file(
                tmp_path,
                {
                    "c1": {
                        "requirement": REQUIREMENT,
                        "assertion": "always @* assert(1);",
                        "scope": "combinational",
                        "reset": "rst_n",
                    }
                },
            )
        )
    with pytest.raises(ValueError, match="disagrees with its scope"):
        load_contracts(
            contracts_file(
                tmp_path,
                {
                    "c1": {
                        "requirement": REQUIREMENT,
                        "assertion": "always @* assert(1);",
                        "scope": "sequential",
                        "reset": "",
                    }
                },
            )
        )


def test_unregistered_contract_and_mismatch_refused(
    tmp_path, tool_paths, monkeypatch
):
    stub_tools(monkeypatch)
    checker = FixedChecker(
        contracts_file(tmp_path),
        yosys=tool_paths["yosys"],
        iverilog=tool_paths["iverilog"],
        workdir=tmp_path / "work",
    )
    with pytest.raises(ProposalError, match="Unregistered contract"):
        checker(proposal(contract_id="nope"))
    with pytest.raises(ProposalError, match="does not match"):
        checker(proposal(requirement="Paraphrased words."))


def test_missing_toolchain_refused_before_solving(tmp_path, monkeypatch):
    stub_tools(monkeypatch)
    checker = FixedChecker(
        contracts_file(tmp_path),
        yosys="/nonexistent/yosys",
        workdir=tmp_path / "work",
    )
    with pytest.raises(ProposalError, match="toolchain unavailable"):
        checker(proposal())


def test_satisfies_and_violates_carry_solver_evidence(
    tmp_path, tool_paths, monkeypatch
):
    calls = []

    def verify(code, assertion, yosys, directory, reset, **kwargs):
        calls.append((code, assertion, reset))
        assert assertion == "always @* assert(1);"
        assert reset == "rst_n"
        assert kwargs.get("bounded_cycles", 6) == 6
        return stub_result(data.LABELS[0], "temporal-induction")

    stub_tools(monkeypatch, verify)
    checker = FixedChecker(
        contracts_file(tmp_path),
        yosys=tool_paths["yosys"],
        iverilog=tool_paths["iverilog"],
        workdir=tmp_path / "work",
    )
    outcome = checker(proposal())
    assert outcome["label"] == "satisfies_requirement"
    assert outcome["evidence"]["proof"]["procedure"] == ("temporal-induction")
    assert outcome["evidence"]["checker_id"].startswith("siliconshield-")
    validate_evidence(REQUIREMENT, SEQ, RTL, "c1", outcome)
    stub_tools(
        monkeypatch,
        lambda *args, **kwargs: stub_result(data.LABELS[1], "bounded-6-cycles"),
    )
    outcome = checker(proposal())
    assert outcome["label"] == "violates_requirement"
    assert outcome["evidence"]["counterexample"]["reachable"] is True
    validate_evidence(REQUIREMENT, SEQ, RTL, "c1", outcome)
    assert len(calls) == 1


def test_unestablished_verdicts_refuse_instead_of_defaulting(
    tmp_path, tool_paths, monkeypatch
):
    stub_tools(
        monkeypatch,
        lambda *args, **kwargs: {
            "accepted": False,
            "reason": "induction_inconclusive",
        },
    )
    checker = FixedChecker(
        contracts_file(tmp_path),
        yosys=tool_paths["yosys"],
        iverilog=tool_paths["iverilog"],
        workdir=tmp_path / "work",
    )
    with pytest.raises(ProposalError, match="Unestablished"):
        checker(proposal())


def test_compile_failures_refuse(tmp_path, tool_paths, monkeypatch):
    def compile_rtl(*args, **kwargs):
        raise ValueError("Icarus compilation failed")

    monkeypatch.setattr(data, "compile_rtl", compile_rtl)
    monkeypatch.setattr(
        data, "verify", lambda *args, **kwargs: stub_result(data.LABELS[0])
    )
    checker = FixedChecker(
        contracts_file(tmp_path),
        yosys=tool_paths["yosys"],
        iverilog=tool_paths["iverilog"],
        workdir=tmp_path / "work",
    )
    with pytest.raises(ProposalError, match="Compilation failed"):
        checker(proposal())


def stubbed_checker(tmp_path, tool_paths):
    return FixedChecker(
        contracts_file(tmp_path),
        yosys=tool_paths["yosys"],
        iverilog=tool_paths["iverilog"],
        workdir=tmp_path / "work",
    )


def test_completions_derive_from_the_visible_hole(
    tmp_path, tool_paths, monkeypatch
):
    verdicts = [data.LABELS[0], data.LABELS[1]]
    compiled = []
    monkeypatch.setattr(
        data,
        "compile_rtl",
        lambda *args, **kwargs: compiled.append(args[0]),
    )
    monkeypatch.setattr(
        data,
        "verify",
        lambda *args, **kwargs: stub_result(verdicts.pop(0), "stub"),
    )
    checker = stubbed_checker(tmp_path, tool_paths)
    outcome = checker(
        proposal(
            rtl=HOLE_RTL,
            hole=dict(HOLE),
            completions=[{"fill": "a"}, {"fill": "b"}],
        )
    )
    assert outcome["label"] == "insufficient_context"
    assert outcome["evidence"]["hole"] == HOLE
    assert [entry["fill"] for entry in outcome["evidence"]["completions"]] == [
        "a",
        "b",
    ]
    for body, fill in zip(compiled[1:], ("a", "b")):
        assert "assign guard_expr = %s;" % fill in body
        assert "assign x = guard_expr;" in body
    validate_evidence(REQUIREMENT, SEQ, HOLE_RTL, "c1", outcome)
    with pytest.raises(ProposalError, match="two completions"):
        checker(
            proposal(
                rtl=HOLE_RTL,
                hole=dict(HOLE),
                completions=[{"fill": "only"}],
            )
        )


def test_agreeing_completions_refuse(tmp_path, tool_paths, monkeypatch):
    stub_tools(monkeypatch)
    checker = stubbed_checker(tmp_path, tool_paths)
    with pytest.raises(ProposalError, match="do not oppose"):
        checker(
            proposal(
                rtl=HOLE_RTL,
                hole=dict(HOLE),
                completions=[{"fill": "a"}, {"fill": "b"}],
            )
        )


def test_completions_without_a_hole_refuse(tmp_path, tool_paths, monkeypatch):
    stub_tools(monkeypatch)
    checker = stubbed_checker(tmp_path, tool_paths)
    with pytest.raises(ProposalError, match="no omitted hole"):
        checker(proposal(completions=[{"fill": "a"}, {"fill": "b"}]))
    with pytest.raises(ProposalError, match="identifier name"):
        checker(
            proposal(
                rtl=HOLE_RTL,
                hole={"name": "not an identifier", "width": 1},
                completions=[{"fill": "a"}, {"fill": "b"}],
            )
        )
    with pytest.raises(ProposalError, match="positive width"):
        checker(
            proposal(
                rtl=HOLE_RTL,
                hole={"name": "guard_expr", "width": 0},
                completions=[{"fill": "a"}, {"fill": "b"}],
            )
        )
    with pytest.raises(ProposalError, match="no hole fill"):
        checker(
            proposal(
                rtl=HOLE_RTL,
                hole=dict(HOLE),
                completions=[{"rtl": "legacy"}, {"fill": "b"}],
            )
        )


def test_hole_must_be_omitted_visible_and_closed(
    tmp_path, tool_paths, monkeypatch
):
    stub_tools(monkeypatch)
    checker = stubbed_checker(tmp_path, tool_paths)
    fills = [{"fill": "a"}, {"fill": "b"}]
    with pytest.raises(ProposalError, match="declared exactly once"):
        checker(proposal(rtl=RTL, hole=dict(HOLE), completions=fills))
    driven = HOLE_RTL + " assign guard_expr = 1;"
    with pytest.raises(ProposalError, match="driven, not omitted"):
        checker(proposal(rtl=driven, hole=dict(HOLE), completions=fills))
    unused = "module topmodule; wire guard_expr; assign x = 1; endmodule"
    with pytest.raises(ProposalError, match="never used"):
        checker(proposal(rtl=unused, hole=dict(HOLE), completions=fills))
    leaky = [{"fill": "guard_expr"}, {"fill": "b"}]
    with pytest.raises(ProposalError, match="references the omitted hole"):
        checker(proposal(rtl=HOLE_RTL, hole=dict(HOLE), completions=leaky))
    with pytest.raises(ProposalError, match="needs completions"):
        checker(proposal(rtl=HOLE_RTL, hole=dict(HOLE)))


def test_assign_fill_preserves_grouping_and_uses():
    rtl = "module topmodule; wire omitted; assign q = a && omitted; endmodule"
    body = FixedChecker._derive_completion(rtl, "omitted", 1, "b || c")
    assert "assign q = a && omitted;" in body
    assert "assign omitted = b || c;" in body
    assert "a && b || c" not in body


def test_permitted_fills_derive_assignments():
    for fill in (
        "!sig_lock",
        "b || c",
        "{a, b}",
        "a ? b : c",
        "8'hFF",
        "(a && b)",
        "a == b",
        "a[3:0]",
    ):
        body = FixedChecker._derive_completion(HOLE_RTL, "guard_expr", 1, fill)
        assert "assign guard_expr = %s;" % fill in body


def test_fill_must_be_a_permitted_expression(tmp_path, tool_paths, monkeypatch):
    stub_tools(monkeypatch)
    checker = stubbed_checker(tmp_path, tool_paths)
    for fill in (
        "a; assign x = 1",
        "`FOO",
        '"lit"',
        "$past(a)",
        "top.sig",
        "a // comment",
        "a /* comment",
        "(a",
        "a)",
        "{a, b",
        "#5 a",
        "a @ b",
    ):
        with pytest.raises(ProposalError, match="permitted expression"):
            checker(
                proposal(
                    rtl=HOLE_RTL,
                    hole=dict(HOLE),
                    completions=[{"fill": fill}, {"fill": "b"}],
                )
            )


def test_unregistered_assumptions_refuse(tmp_path, tool_paths, monkeypatch):
    stub_tools(monkeypatch)
    checker = stubbed_checker(tmp_path, tool_paths)
    with pytest.raises(ProposalError, match="registered sequential scope"):
        checker(proposal(assumptions="the lock is always inactive"))
    with pytest.raises(ProposalError, match="registered sequential scope"):
        checker(proposal(assumptions=COMB))
    comb = FixedChecker(
        comb_contracts_file(tmp_path),
        yosys=tool_paths["yosys"],
        iverilog=tool_paths["iverilog"],
        workdir=tmp_path / "work",
    )
    with pytest.raises(ProposalError, match="registered combinational"):
        comb(
            proposal(
                contract_id="c2",
                requirement="The output must settle.",
                assumptions=SEQ,
            )
        )
    outcome = comb(
        proposal(
            contract_id="c2",
            requirement="The output must settle.",
            assumptions=COMB,
        )
    )
    assert outcome["label"] == "satisfies_requirement"
    validate_evidence("The output must settle.", COMB, RTL, "c2", outcome)


def test_ingest_accepts_checker_established_rows(
    tmp_path, tool_paths, monkeypatch
):
    from siliconshield.schemas import PROPOSAL_SCHEMA

    stub_tools(monkeypatch)
    checker = FixedChecker(
        contracts_file(tmp_path),
        yosys=tool_paths["yosys"],
        iverilog=tool_paths["iverilog"],
        workdir=tmp_path / "work",
    )
    incoming = tmp_path / "in"
    incoming.mkdir()
    (incoming / "p00.json").write_text(
        json.dumps(
            {
                "schema": PROPOSAL_SCHEMA,
                "id": "p1",
                "contract_id": "c1",
                "requirement": REQUIREMENT,
                "assumptions": SEQ,
                "rtl": RTL,
                "kind": "implementation",
                "lineage": {"family": "f1", "base_design": "b1"},
            }
        )
    )
    accounting = ingest_proposals(incoming, checker, tmp_path / "out")
    assert accounting["accepted"] == 1
    current = load_current(tmp_path / "out")
    rows = (current["dir"] / "accepted.jsonl").read_text().splitlines()
    assert json.loads(rows[0])["evidence"]["checker_id"].startswith(
        "siliconshield-"
    )


GOOD_RTL = """\
module topmodule(input clk, rst_n, input sig_lock, input sig_d, output reg sig_q);
always @(posedge clk) begin
  if (!rst_n) sig_q <= 0;
  else if (!sig_lock) sig_q <= sig_d;
end
endmodule
"""

BAD_RTL = """\
module topmodule(input clk, rst_n, input sig_lock, input sig_d, output reg sig_q);
always @(posedge clk) begin
  if (!rst_n) sig_q <= 0;
  else sig_q <= sig_d;
end
endmodule
"""


def lock_contract(tmp_path):
    requirement = data.requirement_for("sig_q", "sig_lock", "hold")
    path = tmp_path / "contracts.json"
    path.write_text(
        json.dumps(
            {
                "lock-hold": {
                    "requirement": requirement,
                    "assertion": data.hold_check(
                        "sig_q", "sig_lock", "rst_n", "hold"
                    ),
                    "scope": "sequential",
                    "reset": "rst_n",
                }
            }
        )
    )
    return path, requirement


OPAQUE_LOCK_RTL = """\
module topmodule(input clk, rst_n, input sig_lock, input sig_d, output reg sig_q);
wire guard_expr;
always @(posedge clk) begin
  if (!rst_n) sig_q <= 0;
  else if (guard_expr) sig_q <= sig_d;
end
endmodule
"""

GROUP_RTL = """\
module topmodule(input a, input b, input c, output q);
wire omitted;
assign q = a && omitted;
endmodule
"""

WIDE_RTL = """\
module topmodule(input [7:0] a, input [7:0] b, output q);
wire [7:0] omitted;
assign q = (omitted == 8'd0);
endmodule
"""


def comb_contract(tmp_path, name, requirement, assertion):
    path = tmp_path / (name + ".json")
    path.write_text(
        json.dumps(
            {
                name: {
                    "requirement": requirement,
                    "assertion": assertion,
                    "scope": "combinational",
                    "reset": "",
                }
            }
        )
    )
    return path


@requires_solver
def test_solver_proves_and_refutes_lock_behavior(tmp_path):
    contracts, requirement = lock_contract(tmp_path)
    checker = FixedChecker(contracts, workdir=tmp_path / "work")
    proved = checker(
        {
            "id": "good",
            "contract_id": "lock-hold",
            "requirement": requirement,
            "assumptions": SEQ,
            "rtl": GOOD_RTL,
            "kind": "implementation",
        }
    )
    assert proved["label"] == "satisfies_requirement"
    validate_evidence(requirement, SEQ, GOOD_RTL, "lock-hold", proved)
    refuted = checker(
        {
            "id": "bad",
            "contract_id": "lock-hold",
            "requirement": requirement,
            "assumptions": SEQ,
            "rtl": BAD_RTL,
            "kind": "mutation",
        }
    )
    assert refuted["label"] == "violates_requirement"
    validate_evidence(requirement, SEQ, BAD_RTL, "lock-hold", refuted)


@requires_solver
def test_solver_refuses_unrelated_attached_programs(tmp_path):
    contracts, requirement = lock_contract(tmp_path)
    checker = FixedChecker(contracts, workdir=tmp_path / "work")
    complete = checker(
        {
            "id": "good",
            "contract_id": "lock-hold",
            "requirement": requirement,
            "assumptions": SEQ,
            "rtl": GOOD_RTL,
            "kind": "implementation",
        }
    )
    assert complete["label"] == "satisfies_requirement"
    with pytest.raises(ProposalError, match="no omitted hole"):
        checker(
            {
                "id": "opaque",
                "contract_id": "lock-hold",
                "requirement": requirement,
                "assumptions": SEQ,
                "rtl": GOOD_RTL,
                "kind": "implementation",
                "completions": [{"rtl": GOOD_RTL}, {"rtl": BAD_RTL}],
            }
        )


@requires_solver
def test_solver_grouping_preserved_through_assign_fill(tmp_path):
    # q = a && X is zero whenever a is zero, for any omitted X: the
    # requirement holds regardless of the missing context, so no two
    # faithful fills can oppose. Substitution used to reparse the
    # `b || c` fill as `a && b || c` and mislabel this insufficient.
    requirement = "When a is zero, q must be zero."
    assertion = "always @* assert(a || !q);"
    contracts = comb_contract(tmp_path, "zero-hold", requirement, assertion)
    checker = FixedChecker(contracts, workdir=tmp_path / "work")
    filled = FixedChecker._derive_completion(GROUP_RTL, "omitted", 1, "b || c")
    assert "assign q = a && omitted;" in filled
    proved = data.verify(filled, assertion, "yosys", tmp_path / "direct", "")
    assert proved["accepted"] and proved["verdict"] == data.LABELS[0]
    with pytest.raises(ProposalError, match="do not oppose"):
        checker(
            {
                "id": "group",
                "contract_id": "zero-hold",
                "requirement": requirement,
                "assumptions": COMB,
                "rtl": GROUP_RTL,
                "kind": "implementation",
                "hole": {"name": "omitted", "width": 1},
                "completions": [{"fill": "b || c"}, {"fill": "b && c"}],
            }
        )


@requires_solver
def test_solver_wide_hole_keeps_wire_width(tmp_path):
    # The 8-bit fill assigns into the 8-bit wire before the use site
    # sees it: `{a, b}` truncates to b, so q follows b alone. Textual
    # substitution would compare the full 16-bit value instead.
    requirement = "When b is zero, q must be high."
    assertion = "always @* assert(|b || q);"
    contracts = comb_contract(tmp_path, "wide-zero", requirement, assertion)
    checker = FixedChecker(contracts, workdir=tmp_path / "work")
    outcome = checker(
        {
            "id": "wide",
            "contract_id": "wide-zero",
            "requirement": requirement,
            "assumptions": COMB,
            "rtl": WIDE_RTL,
            "kind": "implementation",
            "hole": {"name": "omitted", "width": 8},
            "completions": [{"fill": "{a, b}"}, {"fill": "8'd1"}],
        }
    )
    assert outcome["label"] == "insufficient_context"
    assert {
        entry["outcome"] for entry in outcome["evidence"]["completions"]
    } == {"satisfies_requirement", "violates_requirement"}
    validate_evidence(requirement, COMB, WIDE_RTL, "wide-zero", outcome)


@requires_solver
def test_solver_hole_fills_establish_missing_context(tmp_path):
    contracts, requirement = lock_contract(tmp_path)
    checker = FixedChecker(contracts, workdir=tmp_path / "work")
    outcome = checker(
        {
            "id": "opaque",
            "contract_id": "lock-hold",
            "requirement": requirement,
            "assumptions": SEQ,
            "rtl": OPAQUE_LOCK_RTL,
            "kind": "implementation",
            "hole": {"name": "guard_expr", "width": 1},
            "completions": [{"fill": "!sig_lock"}, {"fill": "1"}],
        }
    )
    assert outcome["label"] == "insufficient_context"
    assert outcome["evidence"]["hole"] == {"name": "guard_expr", "width": 1}
    assert {
        entry["outcome"] for entry in outcome["evidence"]["completions"]
    } == {"satisfies_requirement", "violates_requirement"}
    validate_evidence(requirement, SEQ, OPAQUE_LOCK_RTL, "lock-hold", outcome)
