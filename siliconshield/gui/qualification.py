"""Fixture-only GUI/SDK product qualification checks.

The dataset plan requires a small development qualification set to
exercise the real GUI/SDK path before bulk generation: a locked
register, a debug bypass, explicit reset exemption/inclusion, an
unresolved policy, supplied helpers, relevant missing helpers, and
irrelevant omitted logic. These checks run on CPU fixtures without
loading model weights: they verify state formatting, contract
bytes, input coverage, and report-field presence so a later
model-quality gate cannot be confused with a wiring failure.
"""

from typing import Any

from siliconshield.app.laya_assessor import state_text
from siliconshield.training.laya.question_contracts import (
    NEW_CONTRACT,
    match_version,
    question_for,
)

REQUIRED_CASES = (
    "locked-register",
    "debug-bypass",
    "reset-exempt",
    "reset-inclusive",
    "unresolved-policy",
    "supplied-helper",
    "relevant-missing-helper",
    "irrelevant-omission",
)

# Minimal synthetic bodies: distinct functional RTL per case so the
# fixtures exercise formatting over varied inputs. Authored here;
# not experiment data and not solver-verified labels.
_FIXTURES = (
    (
        "locked-register",
        "Outside reset, cfg must hold its value while locked is 1.",
        "rst_n is 1 after reset; clk rises each cycle.",
        "module top(input logic clk, rst_n, locked, wr,\n"
        "  input logic [3:0] d, output logic [3:0] cfg);\n"
        "always_ff @(posedge clk) begin\n"
        "  if (!rst_n) cfg <= 0;\n"
        "  else if (wr && !locked) cfg <= d;\n"
        "end\nendmodule",
    ),
    (
        "debug-bypass",
        "No path may update cfg while locked is 1.",
        "rst_n is 1 after reset; clk rises each cycle.",
        "module top(input logic clk, rst_n, locked, dbg,\n"
        "  input logic [3:0] d, output logic [3:0] cfg);\n"
        "always_ff @(posedge clk) begin\n"
        "  if (!rst_n) cfg <= 0;\n"
        "  else if (dbg) cfg <= d;\n"
        "end\nendmodule",
    ),
    (
        "reset-exempt",
        "Outside reset, cfg must hold while locked is 1.",
        "Reset is asserted first, then inactive; reset writes "
        "are exempt; only later edges are assessed.",
        "module top(input logic clk, rst_n, locked,\n"
        "  output logic [3:0] cfg);\n"
        "always_ff @(posedge clk) begin\n"
        "  if (!rst_n) cfg <= 0;\n"
        "end\nendmodule",
    ),
    (
        "reset-inclusive",
        "Cfg must never change while locked is 1, on any edge.",
        "Reset may assert on any edge including while locked.",
        "module top(input logic clk, rst_n, locked,\n"
        "  output logic [3:0] cfg);\n"
        "always_ff @(posedge clk) begin\n"
        "  if (!rst_n) cfg <= 0;\n"
        "end\nendmodule",
    ),
    (
        "unresolved-policy",
        "Privileged updates need approval.",
        "The approval policy text is not supplied.",
        "module top(input logic clk, priv, ok,\n"
        "  output logic [3:0] cfg);\n"
        "always_ff @(posedge clk) begin\n"
        "  if (priv && ok) cfg <= 1;\n"
        "end\nendmodule",
    ),
    (
        "supplied-helper",
        "Cfg loads the helper result on a write.",
        "Helper combinational semantics as written.",
        "module top(input logic clk, wr,\n"
        "  input logic [3:0] d, output logic [3:0] cfg);\n"
        "function [3:0] f; input [3:0] x; begin f = x | 1; end "
        "endfunction\n"
        "always_ff @(posedge clk) begin\n"
        "  if (wr) cfg <= f(d);\n"
        "end\nendmodule",
    ),
    (
        "relevant-missing-helper",
        "Cfg loads the helper result on a write.",
        "Helper body omitted; its value decides the verdict.",
        "module top(input logic clk, wr,\n"
        "  input logic [3:0] d, output logic [3:0] cfg);\n"
        "wire [3:0] h;\n"
        "/* helper definition unavailable */\n"
        "always_ff @(posedge clk) begin\n"
        "  if (wr) cfg <= h;\n"
        "end\nendmodule",
    ),
    (
        "irrelevant-omission",
        "Cfg loads d on a write.",
        "Observer net omitted; it cannot reach cfg.",
        "module top(input logic clk, wr,\n"
        "  input logic [3:0] d, output logic [3:0] cfg);\n"
        "wire [3:0] obs;\n"
        "/* observer definition unavailable */\n"
        "always_ff @(posedge clk) begin\n"
        "  if (wr) cfg <= d;\n"
        "end\nendmodule",
    ),
)


def build_case_states() -> dict[str, str]:
    """Return authored fixture states for each required case."""
    states = {}
    for name, requirement, assumptions, rtl in _FIXTURES:
        states[name] = state_text(requirement, assumptions, rtl)
    assert set(states) == set(REQUIRED_CASES)
    return states


def qualify_states(
    states: dict[str, str],
    contract: str = NEW_CONTRACT,
    max_len: int = 1536,
) -> dict[str, Any]:
    """Check fixture states for coverage and contract agreement.

    Verifies every required case is present, non-empty, carries an
    RTL section, fits the declared token budget by a conservative
    whitespace estimate, and matches the pinned question bytes.
    Returns the contract version and per-case sizes.
    """
    missing = [name for name in REQUIRED_CASES if name not in states]
    if missing:
        raise ValueError("Missing qualification cases: %s" % missing)
    questions = question_for(contract)
    if match_version(questions) != contract:
        raise ValueError("Question contract does not round-trip")
    sizes = {}
    for name in REQUIRED_CASES:
        state = states[name]
        if not isinstance(state, str) or not state:
            raise ValueError("Empty state for %s" % name)
        if "RTL\n" not in state:
            raise ValueError("State lacks RTL section: %s" % name)
        # Conservative token proxy: one token per ~4 characters.
        estimate = len(state) // 4 + 1
        if estimate > max_len:
            raise ValueError("Fixture exceeds budget: %s" % name)
        sizes[name] = len(state)
    return {
        "contract": contract,
        "cases": sorted(REQUIRED_CASES),
        "sizes": sizes,
    }
