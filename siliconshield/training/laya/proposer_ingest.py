"""Ingest Muse RTL proposals through fixed-checker validation.

Each proposal file carries one candidate: a registered contract id,
a requirement, execution assumptions, RTL, and established lineage.
The proposer's claimed verdict is never the accepted label: every
proposal runs through an injected fixed checker and only
checker-established rows are accepted, with rejections and reasons
retained for diagnosis.

The checker obeys a strict evidence contract. Production checkers
compose the existing builder stages: ``compile_rtl`` for compilation,
``verify`` (Yosys SAT over the contract assertion) for proofs and
counterexamples, and the completion machinery for missing-context
alternatives. Fixture checkers implement the same outcome shape.

Accepted evidence per label::

    satisfies: proof.procedure names the establishing procedure.
    violates: counterexample carries reachable=true plus a trace.
    insufficient: hole names the omitted wire, and completions holds
        at least two fills with opposing outcomes.

Every outcome echoes the checked RTL bytes, the verified
assumptions, and the answering contract id.

Family independence is never inferred from proposal names: lineage
must name an established family and base design, and only those
families count. Ingestion publishes one complete generation per
successful pass into a new ``gen-N/`` directory; a single
``current.json`` pointer switches to it atomically once every file
is written. A failed or interrupted pass leaves the pointer — and
therefore every reader-visible output — untouched. Orphaned
``gen-N/`` directories from crashed passes are ignored; only the
pointed generation is ever read.
"""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from siliconshield.research_loop.episodes.contracts import atomic_json
from siliconshield.schemas import PROPOSAL_SCHEMA

KINDS = ("implementation", "mutation", "rewrite")
VERDICTS = (
    "satisfies_requirement",
    "violates_requirement",
    "insufficient_context",
)

Checker = Callable[[dict[str, Any]], dict[str, Any]]


class ProposalError(ValueError):
    """A proposal the ingestion pass refused."""


def _read_proposals(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Load and schema-check every proposal file in a directory."""
    proposals = []
    for candidate in sorted(Path(path).glob("*.json")):
        try:
            proposal = json.loads(candidate.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ProposalError(
                "%s is not valid JSON: %s" % (candidate.name, exc)
            ) from exc
        if proposal.get("schema") != PROPOSAL_SCHEMA:
            raise ProposalError(
                "%s has an unknown proposal schema" % candidate.name
            )
        for key in (
            "id",
            "contract_id",
            "requirement",
            "assumptions",
            "rtl",
            "kind",
        ):
            if not proposal.get(key):
                raise ProposalError("%s is missing %s" % (candidate.name, key))
        if proposal["kind"] not in KINDS:
            raise ProposalError("%s has an unknown kind" % candidate.name)
        proposals.append((candidate.name, proposal))
    return proposals


def validate_evidence(
    requirement: str,
    assumptions: str,
    rtl: str,
    contract_id: str,
    outcome: dict[str, Any],
) -> dict[str, Any]:
    """Validate one checker outcome against the evidence contract.

    Requires the checker identity, the contract echo, an RTL hash
    matching the checked bytes, an assumptions hash matching the
    verified scope, and label-specific proof: a named procedure, a
    reachable counterexample with a trace, or a named omitted hole
    with at least two fills carrying opposing outcomes. Returns the
    validated evidence.
    """
    label = outcome.get("label")
    if label not in VERDICTS:
        raise ProposalError("Checker returned no verdict")
    evidence = outcome.get("evidence")
    if not isinstance(evidence, dict):
        raise ProposalError("Checker returned no evidence")
    if not evidence.get("checker_id") or not isinstance(
        evidence["checker_id"], str
    ):
        raise ProposalError("Evidence lacks a checker identity")
    if evidence.get("contract_id") != contract_id:
        raise ProposalError("Evidence answers a different contract")
    rtl_hash = hashlib.sha256(rtl.encode()).hexdigest()
    if evidence.get("rtl_sha256") != rtl_hash:
        raise ProposalError("Evidence covers different RTL bytes")
    if not isinstance(assumptions, str):
        raise ProposalError("Proposal assumptions are malformed")
    assumptions_hash = hashlib.sha256(assumptions.encode()).hexdigest()
    if evidence.get("assumptions_sha256") != assumptions_hash:
        raise ProposalError("Evidence covers different assumptions")
    for key in ("bounded_cycles", "bound_cycles", "induction_steps"):
        if key in evidence:
            value = evidence[key]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                or value > 64
            ):
                raise ProposalError("Evidence bound %s is invalid" % key)
    if label == "satisfies_requirement":
        proof = evidence.get("proof", {})
        if not proof.get("procedure") or not isinstance(
            proof["procedure"], str
        ):
            raise ProposalError("Satisfaction lacks a named procedure")
    elif label == "violates_requirement":
        counterexample = evidence.get("counterexample", {})
        if counterexample.get("reachable") is not True:
            raise ProposalError("Counterexample is not reachable")
        if not counterexample.get("trace"):
            raise ProposalError("Counterexample lacks a trace")
    else:
        hole = evidence.get("hole")
        if not isinstance(hole, dict):
            raise ProposalError("Missing context names no omitted hole")
        name, width = hole.get("name"), hole.get("width")
        if not name or not isinstance(name, str):
            raise ProposalError("Missing context names no omitted hole")
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ProposalError("Missing context names no omitted hole")
        completions = evidence.get("completions", [])
        if not isinstance(completions, list) or len(completions) < 2:
            raise ProposalError("Missing context needs two completions")
        outcomes = set()
        for completion in completions:
            if not isinstance(completion, dict):
                raise ProposalError("Completion is malformed")
            if not completion.get("outcome"):
                raise ProposalError("Completion lacks an outcome")
            if not completion.get("fill") or not isinstance(
                completion["fill"], str
            ):
                raise ProposalError("Completion names no hole fill")
            outcomes.add(completion["outcome"])
        if len(outcomes) < 2:
            raise ProposalError("Completions do not oppose")
    _ = requirement
    return evidence


def _dedup_key(proposal: dict[str, Any]) -> str:
    """Identity covering requirement, assumptions, RTL, kind, contract."""
    return hashlib.sha256(
        "\0".join(
            (
                proposal["requirement"],
                proposal["assumptions"],
                proposal["rtl"],
                proposal["kind"],
                proposal["contract_id"],
            )
        ).encode()
    ).hexdigest()


def _sha_bytes(payload: bytes) -> str:
    """Return the SHA-256 hex digest of in-memory bytes."""
    return hashlib.sha256(payload).hexdigest()


def _write_gen_file(path: Path, text: str) -> None:
    """Write one file inside an unpublished generation directory."""
    path.write_text(text, encoding="utf-8")


def _next_generation(output: Path) -> int:
    """Allocate a generation number past the pointer and orphans."""
    numbers = [0]
    pointer = output / "current.json"
    if pointer.is_file():
        try:
            record = json.loads(pointer.read_text(encoding="utf-8"))
            numbers.append(int(record["generation"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ProposalError(
                "Current generation pointer is unreadable"
            ) from exc
    for child in output.glob("gen-*"):
        if not child.is_dir():
            continue
        try:
            numbers.append(int(child.name.split("-", 1)[1]))
        except (ValueError, IndexError):
            continue
    return max(numbers) + 1


def _publish_generation(
    output: Path,
    accepted: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    accounting: dict[str, Any],
    set_hash: str,
) -> dict[str, Any]:
    """Write a complete generation, then switch the pointer to it.

    Generation files land in a fresh directory no reader can see;
    only after every file is written does ``current.json`` move.
    An interruption at any point leaves the previous generation
    current.
    """
    generation = _next_generation(output)
    gen_dir = output / ("gen-%d" % generation)
    gen_dir.mkdir(parents=True, exist_ok=False)
    accepted_text = "".join(json.dumps(row) + "\n" for row in accepted)
    _write_gen_file(gen_dir / "accepted.jsonl", accepted_text)
    rejections_text = json.dumps(rejections, indent=2) + "\n"
    _write_gen_file(gen_dir / "rejections.json", rejections_text)
    accounting_text = json.dumps(accounting, indent=2) + "\n"
    _write_gen_file(gen_dir / "accounting.json", accounting_text)
    outputs = {
        "accepted.jsonl": _sha_bytes(accepted_text.encode()),
        "rejections.json": _sha_bytes(rejections_text.encode()),
        "accounting.json": _sha_bytes(accounting_text.encode()),
    }
    receipt = {
        "status": "complete",
        "generation": generation,
        "proposal_set_sha256": set_hash,
        "accepted": len(accepted),
        "outputs": outputs,
        "time": time.time(),
    }
    receipt_text = json.dumps(receipt, indent=2) + "\n"
    _write_gen_file(gen_dir / "receipt.json", receipt_text)
    atomic_json(
        output / "current.json",
        {
            "generation": generation,
            "dir": gen_dir.name,
            "receipt_sha256": _sha_bytes(receipt_text.encode()),
        },
    )
    return accounting


def load_current(output: Path) -> dict[str, Any]:
    """Return the current generation's directory and records.

    Follows ``current.json`` and re-hashes the receipt and every
    output it names; any missing file, unreadable JSON, or hash
    mismatch refuses instead of returning torn or tampered rows.
    """
    try:
        record = json.loads(
            (Path(output) / "current.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ProposalError("No complete published generation") from exc
    if not isinstance(record, dict) or not re.fullmatch(
        r"gen-[1-9][0-9]*", record.get("dir", "")
    ):
        raise ProposalError("Generation pointer names no generation")
    gen_dir = Path(output) / record["dir"]
    try:
        receipt_bytes = (gen_dir / "receipt.json").read_bytes()
        receipt = json.loads(receipt_bytes)
        accounting = json.loads(
            (gen_dir / "accounting.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ProposalError("No complete published generation") from exc
    if not isinstance(receipt, dict) or not isinstance(accounting, dict):
        raise ProposalError("No complete published generation")
    if record.get("generation") != receipt.get("generation") or _sha_bytes(
        receipt_bytes
    ) != record.get("receipt_sha256"):
        raise ProposalError("Published generation failed verification")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise ProposalError("Published generation failed verification")
    for name, expected in outputs.items():
        try:
            actual = hashlib.sha256((gen_dir / name).read_bytes())
        except OSError as exc:
            raise ProposalError("Published output is missing: " + name) from exc
        if actual.hexdigest() != expected:
            raise ProposalError(
                "Published output changed under its receipt: " + name
            )
    return {
        "generation": record["generation"],
        "dir": gen_dir,
        "receipt": receipt,
        "accounting": accounting,
    }


def ingest_proposals(
    path: Path, checker: Checker, output: Path
) -> dict[str, Any]:
    """Validate proposals and publish one complete output generation.

    ``checker`` maps one proposal dict to a ``{"label": ...,
    "evidence": ...}`` outcome obeying the evidence contract; the
    production checker resolves the proposal's contract and runs the
    solver over its RTL and completions. Returns the accounting
    summary. Outputs publish as a new ``gen-N/`` directory holding
    ``accepted.jsonl``, ``rejections.json``, ``accounting.json``,
    and ``receipt.json``; ``current.json`` switches to it only once
    every file is written. A malformed proposal file records its
    failure under ``failures/`` and leaves any previous generation
    untouched.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        proposals = _read_proposals(path)
    except ProposalError as exc:
        failures = output / "failures"
        failures.mkdir(exist_ok=True)
        atomic_json(
            failures / ("attempt-%d.json" % time.time_ns()),
            {"fatal": str(exc), "time": time.time()},
        )
        raise
    accepted: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen: set[str] = set()
    families: set[str] = set()
    for name, proposal in proposals:
        key = _dedup_key(proposal)
        if key in seen:
            rejections.append(
                {"id": proposal["id"], "reason": "duplicate", "file": name}
            )
            continue
        seen.add(key)
        lineage = proposal.get("lineage", {})
        family = lineage.get("family")
        base_design = lineage.get("base_design")
        if not family or not base_design:
            rejections.append(
                {
                    "id": proposal["id"],
                    "reason": "no established lineage",
                    "file": name,
                }
            )
            continue
        try:
            outcome = checker(proposal)
            evidence = validate_evidence(
                proposal["requirement"],
                proposal["assumptions"],
                proposal["rtl"],
                proposal["contract_id"],
                outcome,
            )
        except ProposalError as exc:
            rejections.append(
                {"id": proposal["id"], "reason": str(exc), "file": name}
            )
            continue
        families.add(family)
        accepted.append(
            {
                "id": proposal["id"],
                "contract_id": proposal["contract_id"],
                "requirement": proposal["requirement"],
                "assumptions": proposal["assumptions"],
                "rtl": proposal["rtl"],
                "kind": proposal["kind"],
                "label": outcome["label"],
                "evidence": evidence,
                "lineage": lineage,
            }
        )
    duplicates = sum(1 for item in rejections if item["reason"] == "duplicate")
    set_hash = hashlib.sha256(
        json.dumps(sorted(seen), sort_keys=True).encode()
    ).hexdigest()
    accounting = {
        "status": "complete",
        "proposal_set_sha256": set_hash,
        "proposed": len(proposals),
        "accepted": len(accepted),
        "rejected": len(rejections),
        "duplicates": duplicates,
        "accepted_families": len(families),
    }
    return _publish_generation(
        output, accepted, rejections, accounting, set_hash
    )
