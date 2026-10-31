"""Fixed solver-backed checker for RTL proposals.

``FixedChecker`` implements the ingestion checker protocol with real
evidence: every candidate compiles under Icarus and its contract
assertion runs through Yosys SAT. The assertion always comes from a
registered contract, never from the proposal: the proposal names a
contract id, the checker loads the exact requirement text, assertion,
reset, and scope, and refuses paraphrased requirements, unregistered
contracts, unregistered assumptions, and unestablished verdicts.

Labels are established, never defaulted. A proved assertion yields
satisfies; a reachable counterexample yields violates; anything the
solver cannot establish refuses the proposal instead of degrading to
missing context. Missing context is itself a positive claim: the
proposal names one explicitly omitted undriven wire, and each
completion fills that hole with an alternative expression. The
checker derives every completion by keeping the declared wire and
all its uses and driving the wire from one validated continuous
assignment, so each fill evaluates with exactly the grouping,
width, and signedness it was written with; textual substitution
into use sites would reparse the fill and change the logic. Each
completion must compile and verify, with opposing established
verdicts. Unrelated programs attached as completions are refused,
never relabeled.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from siliconshield.training.laya import build_security_data as data
from siliconshield.training.laya.proposer_ingest import ProposalError

CHECKER_ID = "siliconshield-fixed-checker-v1"

_FILL_ALLOWED = re.compile(r"[A-Za-z0-9_'\s()\[\]{}+\-*/%<>&|^~!?:,=]+")

REGISTERED_ASSUMPTIONS = {
    "sequential": data.assumptions({"reset": "rst_n", "kind": "sequential"}),
    "combinational": data.assumptions({"reset": "", "kind": "comb_zero"}),
}


def _bound_value(contract: dict[str, Any], key: str, default: int) -> int:
    """Return a declared solver bound, defaulting when absent."""
    value = contract.get(key, default)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > 64
    ):
        raise ValueError("Contract bound %s must be 1..64" % key)
    return value


def load_contracts(contracts_path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate the registered contract registry.

    Each contract carries an exact requirement text, one SystemVerilog
    assertion, a verification scope, and a reset name. Sequential
    contracts verify at reset-conditioned clock edges; combinational
    contracts carry an empty reset and settle unconditionally. The
    scope selects the registered assumptions text a proposal must
    repeat exactly. Requirements must be unique so a requirement
    identifies its contract without ambiguity. Optional
    ``bound_cycles`` and ``induction_steps`` declare the
    counterexample horizon and induction depth for that property.
    """
    registry = json.loads(Path(contracts_path).read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or not registry:
        raise ValueError("Contract registry must be a non-empty object")
    seen = set()
    for contract_id, contract in registry.items():
        for key in ("requirement", "assertion", "scope"):
            if not contract.get(key) or not isinstance(contract[key], str):
                raise ValueError("Contract %s needs %s" % (contract_id, key))
        if contract["scope"] not in REGISTERED_ASSUMPTIONS:
            raise ValueError(
                "Contract needs a registered scope: " + contract_id
            )
        reset = contract.get("reset")
        if not isinstance(reset, str):
            raise ValueError("Contract %s needs reset" % contract_id)
        if bool(reset) != (contract["scope"] == "sequential"):
            raise ValueError(
                "Contract reset disagrees with its scope: " + contract_id
            )
        if contract["assertion"].count("assert(") != 1:
            raise ValueError(
                "Contract needs exactly one assertion: " + contract_id
            )
        if contract["requirement"] in seen:
            raise ValueError("Two contracts share one requirement text")
        seen.add(contract["requirement"])
        contract["bound_cycles"] = _bound_value(contract, "bound_cycles", 6)
        contract["induction_steps"] = _bound_value(
            contract, "induction_steps", 12
        )
    return registry


class FixedChecker:
    """Compile plus Yosys-SAT verification behind the checker protocol."""

    def __init__(
        self,
        contracts_path: Path,
        *,
        yosys: str = "yosys",
        iverilog: str = "iverilog",
        workdir: Path,
    ) -> None:
        self.contracts = load_contracts(contracts_path)
        self.yosys = yosys
        self.iverilog = iverilog
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)

    def _tools(self) -> tuple[str, str]:
        """Resolve solver binaries; refuse when the toolchain is absent."""
        yosys_path = shutil.which(self.yosys) or self.yosys
        iverilog_path = shutil.which(self.iverilog) or self.iverilog
        for name, path in (
            ("yosys", yosys_path),
            ("iverilog", iverilog_path),
        ):
            if not Path(path).is_file():
                raise ProposalError("Solver toolchain unavailable: " + name)
        return yosys_path, iverilog_path

    def _compile(self, rtl: str, evidence_id: str, iverilog: str) -> None:
        """Compile one RTL body; refuse proposals that do not build."""
        try:
            data.compile_rtl(rtl, evidence_id, iverilog, self.workdir)
        except (ValueError, OSError) as exc:
            raise ProposalError("Compilation failed: %s" % exc) from exc

    def _verify(
        self,
        rtl: str,
        assertion: str,
        reset: str,
        yosys: str,
        bound_cycles: int = 6,
        induction_steps: int = 12,
    ) -> dict[str, Any]:
        """Run the solver; refuse anything it cannot establish."""
        try:
            result = data.verify(
                rtl,
                assertion,
                yosys,
                self.workdir,
                reset,
                bounded_cycles=bound_cycles,
                induction_steps=induction_steps,
            )
        except (ValueError, OSError) as exc:
            raise ProposalError("Solver failure: %s" % exc) from exc
        if not result.get("accepted"):
            raise ProposalError(
                "Unestablished verdict: %s"
                % result.get("reason", "solver_undecided")
            )
        return result

    def _base_evidence(
        self,
        contract_id: str,
        rtl: str,
        result: dict[str, Any],
        assumptions: str,
        contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evidence shared by every established outcome."""
        evidence: dict[str, Any] = {
            "checker_id": CHECKER_ID,
            "contract_id": contract_id,
            "rtl_sha256": hashlib.sha256(rtl.encode()).hexdigest(),
            "assumptions_sha256": hashlib.sha256(
                assumptions.encode()
            ).hexdigest(),
            "solver_version": result.get("solver_version"),
            "checker_sha256": result.get("checker_sha256"),
            "evidence_id": result.get("evidence_id"),
            "design_sha256": result.get("design_sha256"),
            "assertion_sha256": result.get("assertion_sha256"),
        }
        for key in ("bounded_cycles", "induction_steps"):
            value = result.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                evidence[key] = value
        if contract is not None:
            evidence["bound_cycles"] = contract.get("bound_cycles", 6)
            evidence["induction_steps"] = contract.get("induction_steps", 12)
        return evidence

    def check(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Establish one proposal's label with solver evidence."""
        yosys, iverilog = self._tools()
        contract_id = proposal.get("contract_id", "")
        contract = self.contracts.get(contract_id)
        if contract is None:
            raise ProposalError("Unregistered contract: " + str(contract_id))
        if proposal.get("requirement") != contract["requirement"]:
            raise ProposalError(
                "Requirement does not match the registered contract"
            )
        assumptions = proposal.get("assumptions")
        expected = REGISTERED_ASSUMPTIONS[contract["scope"]]
        if assumptions != expected:
            raise ProposalError(
                "Assumptions are not the registered %s scope"
                % contract["scope"]
            )
        rtl = proposal["rtl"]
        self._compile(rtl, "proposal-%s" % proposal["id"], iverilog)
        completions = proposal.get("completions") or []
        hole = proposal.get("hole")
        if completions:
            return self._check_completions(
                proposal,
                contract,
                completions,
                yosys=yosys,
                iverilog=iverilog,
            )
        if hole:
            raise ProposalError("An omitted hole needs completions")
        result = self._verify(
            rtl,
            contract["assertion"],
            contract["reset"],
            yosys,
            bound_cycles=contract.get("bound_cycles", 6),
            induction_steps=contract.get("induction_steps", 12),
        )
        evidence = self._base_evidence(
            contract_id, rtl, result, assumptions, contract
        )
        if result["verdict"] == data.LABELS[0]:
            evidence["proof"] = {
                "procedure": result.get("basis", "yosys_sat"),
                "antecedent": result.get("antecedent", "unconditional"),
            }
            return {"label": "satisfies_requirement", "evidence": evidence}
        evidence["counterexample"] = {
            "reachable": True,
            "trace": {
                "evidence_id": result.get("evidence_id"),
                "basis": result.get("basis"),
            },
            "counterexample_sha256": self._counterexample_hash(result),
        }
        return {"label": "violates_requirement", "evidence": evidence}

    @staticmethod
    def _hole(proposal: dict[str, Any]) -> tuple[str, int]:
        """Require one explicitly omitted hole declaration."""
        hole = proposal.get("hole")
        if not isinstance(hole, dict):
            raise ProposalError("Completions name no omitted hole")
        name = hole.get("name")
        width = hole.get("width")
        if (
            not name
            or not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_]\w*", name) is None
        ):
            raise ProposalError("Omitted hole needs an identifier name")
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ProposalError("Omitted hole needs a positive width")
        return name, width

    @staticmethod
    def _hole_visible_once(rtl: str, name: str, width: int) -> None:
        """Require the hole undriven, used, and unambiguous upstream."""
        declaration = data.hole_declaration(width, name)
        if rtl.count(declaration) != 1:
            raise ProposalError("Omitted hole is not declared exactly once")
        exposed = rtl.replace(declaration, "", 1)
        escaped = re.escape(name)
        if re.search(r"assign\s+%s\b" % escaped, exposed):
            raise ProposalError("Omitted hole is driven, not omitted")
        if re.search(
            r"[A-Za-z0-9_$]%s|%s[A-Za-z0-9_$]" % (escaped, escaped),
            exposed,
        ):
            raise ProposalError("Hole name collides with another identifier")
        if re.search(r"\b%s\b" % escaped, exposed) is None:
            raise ProposalError("Omitted hole is never used")

    @staticmethod
    def _balanced(fill: str) -> bool:
        """Check that a fill's delimiters nest correctly."""
        pairs = {")": "(", "]": "[", "}": "{"}
        stack = []
        for char in fill:
            if char in "([{":
                stack.append(char)
            elif char in pairs:
                if not stack or stack.pop() != pairs[char]:
                    return False
        return not stack

    @staticmethod
    def _checked_fill(fill: object, hole: str) -> None:
        """Require a side-effect-free combinational expression fill.

        Only expression characters may appear: no statement
        terminators, directives, strings, system calls, hierarchical
        references, delays, or comments that could escape the
        assignment. The fill must not reference the hole itself,
        which would close a combinational loop.
        """
        if not isinstance(fill, str) or not fill:
            raise ProposalError("Completion names no hole fill")
        if (
            _FILL_ALLOWED.fullmatch(fill) is None
            or "//" in fill
            or "/*" in fill
            or not FixedChecker._balanced(fill)
        ):
            raise ProposalError("Fill is not a permitted expression")
        if re.search(r"\b%s\b" % re.escape(hole), fill):
            raise ProposalError("Fill references the omitted hole")

    @staticmethod
    def _derive_completion(
        rtl: str, name: str, width: int, fill: object
    ) -> str:
        """Drive the preserved wire from one validated assignment.

        The declaration and every use stay exactly as shown; the
        fill evaluates once, at the wire's declared width, instead
        of being reparsed at each use site.
        """
        FixedChecker._checked_fill(fill, name)
        declaration = data.hole_declaration(width, name)
        if rtl.count(declaration) != 1:
            raise ProposalError("Omitted hole is not declared exactly once")
        driven = "%s\nassign %s = %s;" % (declaration, name, fill)
        return rtl.replace(declaration, driven, 1)

    def _check_completions(
        self,
        proposal: dict[str, Any],
        contract: dict[str, Any],
        completions: list[dict[str, Any]],
        *,
        yosys: str,
        iverilog: str,
    ) -> dict[str, Any]:
        """Fill the preserved wire once per completion and oppose.

        Each completion keeps the declared wire and every use,
        driven from one validated continuous assignment, then must
        compile, verify, and oppose. Programs that are not hole
        fills of this fragment cannot reach the solver.
        """
        if len(completions) < 2:
            raise ProposalError("Missing context needs two completions")
        name, width = self._hole(proposal)
        self._hole_visible_once(proposal["rtl"], name, width)
        checked = []
        first_result: dict[str, Any] = {}
        for index, completion in enumerate(completions):
            if not isinstance(completion, dict):
                raise ProposalError("Completion is malformed")
            fill = completion.get("fill")
            body = self._derive_completion(proposal["rtl"], name, width, fill)
            self._compile(
                body,
                "proposal-%s-completion-%d" % (proposal["id"], index),
                iverilog,
            )
            result = self._verify(
                body,
                contract["assertion"],
                contract["reset"],
                yosys,
                bound_cycles=contract.get("bound_cycles", 6),
                induction_steps=contract.get("induction_steps", 12),
            )
            if not first_result:
                first_result = result
            outcome = (
                "satisfies_requirement"
                if result["verdict"] == data.LABELS[0]
                else "violates_requirement"
            )
            checked.append(
                {
                    "rtl_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "fill": fill,
                    "outcome": outcome,
                    "evidence_id": result.get("evidence_id"),
                    "basis": result.get("basis"),
                }
            )
        outcomes = {entry["outcome"] for entry in checked}
        if len(outcomes) < 2:
            raise ProposalError("Completions do not oppose")
        evidence = self._base_evidence(
            proposal["contract_id"],
            proposal["rtl"],
            first_result,
            proposal["assumptions"],
            contract,
        )
        evidence["hole"] = {"name": name, "width": width}
        evidence["completions"] = checked
        return {"label": "insufficient_context", "evidence": evidence}

    def _counterexample_hash(self, result: dict[str, Any]) -> Optional[str]:
        """Hash the dumped counterexample model when the solver left one."""
        model = (
            self.workdir
            / str(result.get("evidence_id"))
            / "counterexample.json"
        )
        if not model.is_file():
            return None
        digest = hashlib.sha256()
        with model.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def __call__(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Accept proposals through the ingestion checker protocol."""
        return self.check(proposal)
