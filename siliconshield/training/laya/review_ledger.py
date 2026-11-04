"""Independent review ledger for English-to-assertion mapping.

Each contract template, ambiguity policy, reset exception, and new
extraction mechanism needs an independent review before its rows
admit training. The reviewer reads the English requirement, scope,
and source context without seeing the expected label, records a
property interpretation, and compares it with the executable
assertion. Disagreements block the template, not the whole
dataset. Model-assisted review is recorded as such, never as
human review.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from siliconshield.schemas import REVIEW_LEDGER_SCHEMA as REVIEW_SCHEMA

REVIEWER_TYPES = ("human", "independent-model")
VERDICTS = ("agree", "disagree")
APPROVED_STATUS = "resolved-and-approved"


def _section(state: str, name: str, stop: str) -> str:
    """Extract a state section by header (mirrors the row renderer)."""
    start = state.find(name + "\n")
    if start < 0:
        raise ValueError("Row state lacks section: " + name)
    start += len(name) + 1
    end = state.find(stop, start)
    return state[start : end if end >= 0 else len(state)]


def requirement_of(row: dict[str, Any]) -> str:
    """Return the requirement text from rendered row bytes."""
    if not isinstance(row, dict):
        raise ValueError("Row must be an object")
    state = row.get("state")
    if not isinstance(state, str):
        raise ValueError("Row lacks state: " + repr(row.get("id")))
    return _section(state, "REQUIREMENT", "\nASSUMPTIONS")


def assumptions_of(row: dict[str, Any]) -> str:
    """Return the assumptions text from rendered row bytes."""
    if not isinstance(row, dict):
        raise ValueError("Row must be an object")
    state = row.get("state")
    if not isinstance(state, str):
        raise ValueError("Row lacks state: " + repr(row.get("id")))
    return _section(state, "ASSUMPTIONS", "\nRTL")


def _digest(record: dict[str, Any]) -> str:
    """Return the canonical SHA-256 of a review record."""
    return hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()


def validate_review(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one template review; refuse incomplete records.

    Requires template, requirement, assertion, and assumption
    hashes, a reviewer type, an interpretation, and a verdict. A
    disagreement additionally needs a resolution, an explicit
    resolved-and-approved status, and non-empty evidence
    references; prose alone never approves a template. Returns
    the validated record.
    """
    if not isinstance(record, dict):
        raise ValueError("Review record must be an object")
    for key in (
        "template_id",
        "requirement_sha256",
        "assertion_sha256",
        "assumptions_sha256",
        "reviewer_type",
        "interpretation",
        "verdict",
    ):
        if not record.get(key) or not isinstance(record[key], str):
            raise ValueError("Review record needs %s" % key)
    if record["reviewer_type"] not in REVIEWER_TYPES:
        raise ValueError("Unknown reviewer type")
    if record["verdict"] not in VERDICTS:
        raise ValueError("Unknown review verdict")
    if "resolution" in record and not isinstance(record["resolution"], str):
        raise ValueError("Review resolution must be text")
    if (
        record["verdict"] == "disagree"
        and not record.get("resolution", "").strip()
    ):
        raise ValueError("Disagreement needs a resolution")
    if record.get("resolution", "").strip():
        # Agree-with-resolution carries the same burden as a
        # disagreement: prose alone never approves a template.
        if record.get("status") != APPROVED_STATUS:
            raise ValueError("Resolution needs status " + APPROVED_STATUS)
        refs = record.get("evidence_refs")
        if (
            not isinstance(refs, list)
            or not refs
            or any(not isinstance(r, str) or not r.strip() for r in refs)
        ):
            raise ValueError("Resolution needs evidence_refs")
    if record.get("schema", REVIEW_SCHEMA) != REVIEW_SCHEMA:
        raise ValueError("Unknown review schema")
    return record


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def validate_ledger(path: Path) -> list[dict[str, Any]]:
    """Load and validate every review in a JSONL ledger."""
    if not isinstance(path, (str, Path)):
        raise ValueError("Review ledger path must be a path")
    records = []
    for lineno, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ValueError("Review line %d is not JSON" % lineno) from exc
        records.append(validate_review(record))
    if not records:
        raise ValueError("Review ledger has no records")
    seen = set()
    for record in records:
        if record["template_id"] in seen:
            raise ValueError(
                "Duplicate template review: %s" % record["template_id"]
            )
        seen.add(record["template_id"])
    return records


def ledger_digest(records: list[dict[str, Any]]) -> str:
    """Return the canonical digest over a validated ledger."""
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ValueError("Ledger digest needs a list of review objects")
    return hashlib.sha256(
        json.dumps(records, sort_keys=True).encode()
    ).hexdigest()


def template_ids(records: list[dict[str, Any]]) -> set[str]:
    """Return the set of reviewed template ids."""
    if not isinstance(records, list) or any(
        not isinstance(record, dict)
        or not isinstance(record.get("template_id"), str)
        for record in records
    ):
        raise ValueError("Template ids need a list of review objects")
    return {record["template_id"] for record in records}


def _approved(record: dict[str, Any]) -> bool:
    """Return whether a ledger record admits training rows."""
    if not isinstance(record, dict):
        raise ValueError("Approval check needs a review object")
    resolution = record.get("resolution", "")
    if resolution and not isinstance(resolution, str):
        raise ValueError("Review resolution must be text")
    if isinstance(resolution, str) and resolution.strip():
        refs = record.get("evidence_refs")
        return bool(
            record.get("status") == APPROVED_STATUS
            and isinstance(refs, list)
            and refs
        )
    return bool(record["verdict"] == "agree")


def require_template_coverage(
    rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> None:
    """Require every row to match an approved reviewed template.

    Rows carry ``template_id``; the requirement and assumption
    hashes are recomputed from the row's rendered bytes, never
    trusted from row fields. The ledger must contain that id
    with an ``agree`` verdict (or a resolved-and-approved
    disagreement with evidence) AND identical hashes; a
    matching id with different reviewed bytes refuses
    training, since the review did not cover those bytes.
    """
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) for row in rows
    ):
        raise ValueError("Template coverage needs a list of row objects")
    if not isinstance(records, list) or any(
        not isinstance(record, dict)
        or not isinstance(record.get("template_id"), str)
        for record in records
    ):
        raise ValueError("Template coverage needs a list of review objects")
    reviewed = {
        record["template_id"]: record for record in records if _approved(record)
    }
    for row in rows:
        template = row.get("template_id")
        if not isinstance(template, str) or not template:
            raise ValueError("Row lacks template_id: " + repr(row.get("id")))
        record = reviewed.get(template)
        if record is None:
            raise ValueError("Template lacks agreed review: %s" % template)
        actual = {
            "requirement_sha256": _sha(requirement_of(row)),
            "assumptions_sha256": _sha(assumptions_of(row)),
        }
        for key, seen in actual.items():
            if seen != record[key]:
                raise ValueError(
                    "Row %s %s differs from reviewed template %s"
                    % (repr(row.get("id")), key, template)
                )
        if row.get("assertion_sha256") != record["assertion_sha256"]:
            raise ValueError(
                "Row %s assertion_sha256 differs from reviewed "
                "template %s" % (repr(row.get("id")), template)
            )


def packet_assertion_sha(assertions: Any) -> str:
    """Return the canonical hash of a packet's assertion blocks."""
    try:
        blob = json.dumps(assertions, sort_keys=True).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("Packet assertions must be JSON-serializable") from exc
    return hashlib.sha256(blob).hexdigest()


def require_packet_binding(
    records: list[dict[str, Any]], packets: dict[str, Any]
) -> None:
    """Bind ledger entries to actual review-packet bytes.

    Recomputes each entry's ``assertion_sha256`` from the bound
    packet's ``assertions`` blocks and refuses any mismatch. Row
    and ledger hashes alone cannot establish this: the packet
    file is the reviewed artifact, so the check reads it. Every
    approved template must have a packet; unknown packets are
    ignored.
    """
    if not isinstance(packets, dict) or not packets:
        raise ValueError("Packet binding needs a packet mapping")
    if not isinstance(records, list) or any(
        not isinstance(record, dict)
        or not isinstance(record.get("template_id"), str)
        for record in records
    ):
        raise ValueError("Packet binding needs a list of review objects")
    for record in records:
        if not _approved(record):
            continue
        template = record["template_id"]
        packet = packets.get(template)
        if not isinstance(packet, dict) or "assertions" not in packet:
            raise ValueError(
                "Reviewed template lacks a bound packet: %s" % template
            )
        seen = packet_assertion_sha(packet["assertions"])
        if seen != record["assertion_sha256"]:
            raise ValueError(
                "Packet assertions for %s differ from reviewed "
                "ledger entry" % template
            )
