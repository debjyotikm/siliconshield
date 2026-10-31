"""Proposal ingestion: evidence contract and atomic generations."""

import hashlib
import json

import pytest

from siliconshield.schemas import PROPOSAL_SCHEMA
from siliconshield.training.laya import proposer_ingest
from siliconshield.training.laya.proposer_ingest import (
    ProposalError,
    ingest_proposals,
    load_current,
    validate_evidence,
)


def proposal(pid, rtl="module topmodule; endmodule", **extra):
    row = {
        "schema": PROPOSAL_SCHEMA,
        "id": pid,
        "contract_id": "contract-" + pid,
        "requirement": "req " + pid,
        "assumptions": "two-state",
        "rtl": rtl,
        "kind": "implementation",
        "claimed_verdict": "satisfies_requirement",
        "lineage": {"family": "fam-" + pid, "base_design": "base-" + pid},
    }
    row.update(extra)
    return row


def evidence_for(proposal, label="violates_requirement"):
    base = {
        "checker_id": "fixture-checker-v1",
        "contract_id": proposal["contract_id"],
        "rtl_sha256": hashlib.sha256(proposal["rtl"].encode()).hexdigest(),
        "assumptions_sha256": hashlib.sha256(
            proposal["assumptions"].encode()
        ).hexdigest(),
    }
    if label == "satisfies_requirement":
        base["proof"] = {"procedure": "yosys-sat"}
    elif label == "violates_requirement":
        base["counterexample"] = {"reachable": True, "trace": ["step"]}
    else:
        base["hole"] = {"name": "condition_expr", "width": 1}
        base["completions"] = [
            {"outcome": "satisfies_requirement", "fill": "a"},
            {"outcome": "violates_requirement", "fill": "b"},
        ]
    return {"label": label, "evidence": base}


def write_proposals(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        (path / ("p%02d.json" % index)).write_text(json.dumps(row))


def test_accepted_rows_carry_checker_labels_not_claims(tmp_path):
    first, second = proposal("a"), proposal("b", rtl="module other; endmodule")
    write_proposals(tmp_path / "in", [first, second])
    outcomes = {
        "req a": evidence_for(first),
        "req b": evidence_for(second, "satisfies_requirement"),
    }
    accounting = ingest_proposals(
        tmp_path / "in",
        lambda proposal: outcomes[proposal["requirement"]],
        tmp_path / "out",
    )
    assert accounting["accepted"] == 2
    assert accounting["accepted_families"] == 2
    assert accounting["status"] == "complete"
    current = load_current(tmp_path / "out")
    assert current["generation"] == 1
    rows = [
        json.loads(line)
        for line in (current["dir"] / "accepted.jsonl").read_text().splitlines()
    ]
    assert [row["label"] for row in rows] == [
        "violates_requirement",
        "satisfies_requirement",
    ]
    assert all("claimed_verdict" not in row for row in rows)
    receipt = current["receipt"]
    assert receipt["status"] == "complete" and receipt["accepted"] == 2
    assert set(receipt["outputs"]) == {
        "accepted.jsonl",
        "rejections.json",
        "accounting.json",
    }


def test_checker_refusals_and_duplicates_are_retained(tmp_path):
    first = proposal("a", rtl="good")
    second = proposal("b", rtl="bad")
    third = proposal("c", rtl="good", requirement="req a")

    def checker(proposal):
        if proposal["rtl"] == "bad":
            raise ProposalError("no counterexample")
        return evidence_for(first)

    write_proposals(tmp_path / "in", [first, second, third])
    # Same requirement, assumptions, RTL, kind, and contract as "a":
    # a duplicate, not new coverage.
    third["contract_id"] = first["contract_id"]
    (tmp_path / "in/p02.json").write_text(json.dumps(third))
    accounting = ingest_proposals(tmp_path / "in", checker, tmp_path / "out")
    assert accounting["proposed"] == 3
    assert accounting["accepted"] == 1
    assert accounting["rejected"] == 2
    assert accounting["duplicates"] == 1
    assert accounting["accepted_families"] == 1
    current = load_current(tmp_path / "out")
    reasons = {
        row["id"]: row["reason"]
        for row in json.loads((current["dir"] / "rejections.json").read_text())
    }
    assert reasons == {"b": "no counterexample", "c": "duplicate"}


def test_changed_assumptions_are_not_duplicates(tmp_path):
    first = proposal("a", rtl="shared")
    second = proposal(
        "b", rtl="shared", requirement="req a", assumptions="gated-reset"
    )
    second["contract_id"] = first["contract_id"]
    write_proposals(tmp_path / "in", [first, second])
    seen = []

    def checker(proposal):
        seen.append(proposal["assumptions"])
        return evidence_for(proposal)

    accounting = ingest_proposals(tmp_path / "in", checker, tmp_path / "out")
    assert accounting["accepted"] == 2
    assert seen == ["two-state", "gated-reset"]


def test_missing_lineage_is_rejected_not_counted(tmp_path):
    row = proposal("a")
    del row["lineage"]
    write_proposals(tmp_path / "in", [row])
    accounting = ingest_proposals(
        tmp_path / "in",
        lambda proposal: evidence_for(proposal),
        tmp_path / "out",
    )
    assert accounting["accepted"] == 0
    assert accounting["accepted_families"] == 0
    current = load_current(tmp_path / "out")
    reasons = json.loads((current["dir"] / "rejections.json").read_text())
    assert reasons[0]["reason"] == "no established lineage"


def accepted_ids(current):
    rows = (current["dir"] / "accepted.jsonl").read_text().splitlines()
    return [json.loads(line)["id"] for line in rows]


def test_malformed_retry_preserves_the_last_generation(tmp_path):
    first = proposal("a")
    write_proposals(tmp_path / "in", [first])
    ingest_proposals(
        tmp_path / "in",
        lambda proposal: evidence_for(first),
        tmp_path / "out",
    )
    pointer = (tmp_path / "out/current.json").read_bytes()
    (tmp_path / "in/broken.json").write_text("{not json")
    with pytest.raises(ProposalError, match="not valid JSON"):
        ingest_proposals(tmp_path / "in", lambda proposal: {}, tmp_path / "out")
    assert (tmp_path / "out/current.json").read_bytes() == pointer
    current = load_current(tmp_path / "out")
    assert current["generation"] == 1
    assert accepted_ids(current) == ["a"]
    failures = list((tmp_path / "out/failures").glob("attempt-*.json"))
    assert len(failures) == 1


def test_interrupted_publish_preserves_the_last_generation(
    tmp_path, monkeypatch
):
    first = proposal("a")
    write_proposals(tmp_path / "in", [first])
    ingest_proposals(
        tmp_path / "in",
        lambda proposal: evidence_for(first),
        tmp_path / "out",
    )
    pointer = (tmp_path / "out/current.json").read_bytes()
    second = proposal("b", rtl="module new; endmodule")
    write_proposals(tmp_path / "in2", [second])
    real_write = proposer_ingest._write_gen_file
    calls = {"count": 0}

    def flaky(path, text):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("injected crash")
        return real_write(path, text)

    monkeypatch.setattr(proposer_ingest, "_write_gen_file", flaky)
    with pytest.raises(RuntimeError, match="injected crash"):
        ingest_proposals(
            tmp_path / "in2",
            lambda proposal: evidence_for(second),
            tmp_path / "out",
        )
    assert (tmp_path / "out/current.json").read_bytes() == pointer
    current = load_current(tmp_path / "out")
    assert current["generation"] == 1
    assert accepted_ids(current) == ["a"]
    monkeypatch.setattr(proposer_ingest, "_write_gen_file", real_write)
    accounting = ingest_proposals(
        tmp_path / "in2",
        lambda proposal: evidence_for(second),
        tmp_path / "out",
    )
    assert accounting["accepted"] == 1
    current = load_current(tmp_path / "out")
    assert current["generation"] == 3
    assert accepted_ids(current) == ["b"]


def test_pointer_crash_preserves_the_last_generation(tmp_path, monkeypatch):
    first = proposal("a")
    write_proposals(tmp_path / "in", [first])
    ingest_proposals(
        tmp_path / "in",
        lambda proposal: evidence_for(first),
        tmp_path / "out",
    )
    pointer = (tmp_path / "out/current.json").read_bytes()
    second = proposal("b", rtl="module new; endmodule")
    write_proposals(tmp_path / "in2", [second])

    def flaky(path, value):
        raise RuntimeError("injected crash")

    real_atomic = proposer_ingest.atomic_json
    monkeypatch.setattr(proposer_ingest, "atomic_json", flaky)
    with pytest.raises(RuntimeError, match="injected crash"):
        ingest_proposals(
            tmp_path / "in2",
            lambda proposal: evidence_for(second),
            tmp_path / "out",
        )
    # The complete but unpublished generation is an orphan: readers
    # still see the previous generation with its original rows.
    assert (tmp_path / "out/gen-2/accepted.jsonl").is_file()
    assert (tmp_path / "out/current.json").read_bytes() == pointer
    current = load_current(tmp_path / "out")
    assert accepted_ids(current) == ["a"]
    monkeypatch.setattr(proposer_ingest, "atomic_json", real_atomic)
    ingest_proposals(
        tmp_path / "in2",
        lambda proposal: evidence_for(second),
        tmp_path / "out",
    )
    current = load_current(tmp_path / "out")
    assert current["generation"] == 3
    assert accepted_ids(current) == ["b"]


def test_load_current_refuses_tampered_outputs(tmp_path):
    first = proposal("a")
    write_proposals(tmp_path / "in", [first])
    ingest_proposals(
        tmp_path / "in",
        lambda proposal: evidence_for(first),
        tmp_path / "out",
    )
    current = load_current(tmp_path / "out")
    accepted_path = current["dir"] / "accepted.jsonl"
    before = accepted_path.read_bytes()
    with accepted_path.open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ProposalError, match="changed under its receipt"):
        load_current(tmp_path / "out")
    accepted_path.write_bytes(before)
    assert accepted_ids(load_current(tmp_path / "out")) == ["a"]
    receipt_path = current["dir"] / "receipt.json"
    receipt_before = receipt_path.read_bytes()
    receipt = json.loads(receipt_before)
    receipt["accepted"] = 99
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ProposalError, match="failed verification"):
        load_current(tmp_path / "out")
    receipt_path.write_bytes(receipt_before)
    assert accepted_ids(load_current(tmp_path / "out")) == ["a"]


def test_evidence_contract_rejects_weak_outcomes():
    row = proposal("a")
    good = evidence_for(row)
    assert (
        validate_evidence(
            row["requirement"],
            row["assumptions"],
            row["rtl"],
            row["contract_id"],
            good,
        )["checker_id"]
        == "fixture-checker-v1"
    )
    bad = evidence_for(row)
    bad["evidence"]["rtl_sha256"] = "0" * 64
    with pytest.raises(ProposalError, match="different RTL"):
        validate_evidence(
            row["requirement"],
            row["assumptions"],
            row["rtl"],
            row["contract_id"],
            bad,
        )
    bad = evidence_for(row, "satisfies_requirement")
    del bad["evidence"]["proof"]
    with pytest.raises(ProposalError, match="named procedure"):
        validate_evidence(
            row["requirement"],
            row["assumptions"],
            row["rtl"],
            row["contract_id"],
            bad,
        )
    bad = evidence_for(row, "insufficient_context")
    bad["evidence"]["completions"] = bad["evidence"]["completions"][:1]
    with pytest.raises(ProposalError, match="two completions"):
        validate_evidence(
            row["requirement"],
            row["assumptions"],
            row["rtl"],
            row["contract_id"],
            bad,
        )
    bad = evidence_for(row)
    bad["evidence"]["checker_id"] = ""
    with pytest.raises(ProposalError, match="checker identity"):
        validate_evidence(
            row["requirement"],
            row["assumptions"],
            row["rtl"],
            row["contract_id"],
            bad,
        )
    bad = evidence_for(row)
    bad["evidence"]["assumptions_sha256"] = "0" * 64
    with pytest.raises(ProposalError, match="different assumptions"):
        validate_evidence(
            row["requirement"],
            row["assumptions"],
            row["rtl"],
            row["contract_id"],
            bad,
        )
    bad = evidence_for(row, "insufficient_context")
    del bad["evidence"]["hole"]
    with pytest.raises(ProposalError, match="no omitted hole"):
        validate_evidence(
            row["requirement"],
            row["assumptions"],
            row["rtl"],
            row["contract_id"],
            bad,
        )
    bad = evidence_for(row, "insufficient_context")
    del bad["evidence"]["completions"][0]["fill"]
    with pytest.raises(ProposalError, match="no hole fill"):
        validate_evidence(
            row["requirement"],
            row["assumptions"],
            row["rtl"],
            row["contract_id"],
            bad,
        )
