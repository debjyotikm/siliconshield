"""Review ledger validation and template coverage."""

import hashlib
import json

import pytest

from siliconshield.training.laya.review_ledger import (
    ledger_digest,
    packet_assertion_sha,
    require_packet_binding,
    require_template_coverage,
    validate_ledger,
    validate_review,
)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _record(template="t1", verdict="agree"):
    record = {
        "schema": "siliconshield-review-ledger-v1",
        "template_id": template,
        "requirement_sha256": _sha("req text"),
        "assertion_sha256": "b" * 64,
        "assumptions_sha256": _sha("asm text"),
        "reviewer_type": "human",
        "interpretation": "holds when locked",
        "verdict": verdict,
    }
    if verdict == "disagree":
        record["resolution"] = "corrected packet approved"
        record["status"] = "resolved-and-approved"
        record["evidence_refs"] = ["review/packet-t1.json"]
    return record


def test_validate_review_accepts_human_and_model():
    assert validate_review(_record())["template_id"] == "t1"
    model = _record(template="t2")
    model["reviewer_type"] = "independent-model"
    assert validate_review(model)["verdict"] == "agree"
    bad = _record()
    bad["verdict"] = "disagree"
    with pytest.raises(ValueError, match="resolution"):
        validate_review(bad)


def test_validate_review_refuses_unapproved_disagreement():
    pending = _record(verdict="disagree")
    del pending["status"]
    with pytest.raises(ValueError, match="resolved-and-approved"):
        validate_review(pending)
    pending = _record(verdict="disagree")
    pending["evidence_refs"] = []
    with pytest.raises(ValueError, match="evidence_refs"):
        validate_review(pending)
    pending = _record(verdict="disagree")
    pending["resolution"] = "still unresolved, needs another review"
    del pending["status"]
    with pytest.raises(ValueError, match="resolved-and-approved"):
        validate_review(pending)


def _row(
    row_id="r1", template="t1", requirement="req text", assumptions="asm text"
):
    state = "REQUIREMENT\n%s\nASSUMPTIONS\n%s\nRTL\nassign q = a;\n" % (
        requirement,
        assumptions,
    )
    return {
        "id": row_id,
        "template_id": template,
        "state": state,
        "requirement_sha256": _sha("req text"),
        "assertion_sha256": "b" * 64,
        "assumptions_sha256": _sha("asm text"),
    }


def test_ledger_requires_agreed_templates(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        json.dumps(_record("t1")) + "\n" + json.dumps(_record("t2")) + "\n"
    )
    records = validate_ledger(path)
    assert ledger_digest(records)
    require_template_coverage([_row()], records)
    with pytest.raises(ValueError, match="lacks agreed review"):
        require_template_coverage([_row("r2", "t-missing")], records)


def test_coverage_binds_reviewed_hashes(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(_record("t1")) + "\n")
    records = validate_ledger(path)
    with pytest.raises(ValueError, match="differs from reviewed template"):
        require_template_coverage([_row(requirement="changed")], records)


def test_coverage_recomputes_bytes_not_row_claims(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(_record("t1")) + "\n")
    records = validate_ledger(path)
    row = _row(requirement="changed")
    # Self-declared hash still claims the old bytes: must fail anyway.
    row["requirement_sha256"] = _sha("req text")
    with pytest.raises(ValueError, match="differs from reviewed template"):
        require_template_coverage([row], records)


def test_coverage_admits_approved_disagreement(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(_record("t1", "disagree")) + "\n")
    records = validate_ledger(path)
    require_template_coverage([_row()], records)


def test_agree_with_bare_resolution_is_refused():
    record = _record()
    record["resolution"] = "looks fine after a second glance"
    with pytest.raises(ValueError, match="resolved-and-approved"):
        validate_review(record)
    record["status"] = "resolved-and-approved"
    with pytest.raises(ValueError, match="evidence_refs"):
        validate_review(record)
    record["evidence_refs"] = ["review/packet-t1.json"]
    assert validate_review(record)["verdict"] == "agree"


def test_packet_binding_recomputes_from_packet_bytes(tmp_path):
    assertions = [{"evidence_id": "aid", "assertions": ["assert(q);"]}]
    record = _record()
    record["assertion_sha256"] = packet_assertion_sha(assertions)
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(record) + "\n")
    records = validate_ledger(path)
    packets = {"t1": {"assertions": assertions}}
    require_packet_binding(records, packets)
    stale = {"t1": {"assertions": [{"assertions": ["assert(other);"]}]}}
    with pytest.raises(ValueError, match="differ from reviewed"):
        require_packet_binding(records, stale)
    with pytest.raises(ValueError, match="lacks a bound packet"):
        require_packet_binding(records, {"t2": {"assertions": []}})
