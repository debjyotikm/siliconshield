"""Dataset versions, source and question contracts, proposer ingest, ledger, replay, and audit tests: synthetic fixtures."""

import hashlib
import json
import shutil
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

from siliconshield.app.laya_worker import QUESTION as WORKER_QUESTION
from siliconshield.schemas import PROPOSAL_SCHEMA, REPLAY_PRESERVATION_SCHEMA
from siliconshield.training.laya import build_security_data as data
from siliconshield.training.laya import proposer_ingest
from siliconshield.training.laya import question_contracts as contracts
from siliconshield.training.laya.build_security_data import solver_commands
from siliconshield.training.laya.build_source_security import (
    checked_rows,
    exact_antecedent,
    population_counts,
)
from siliconshield.training.laya.checker_adapter import load_contracts
from siliconshield.training.laya.coverage_audit import (
    quotas_for_round,
    simulate_coverage,
    triplet_integrity,
)
from siliconshield.training.laya.dataset_versions import (
    accept_version,
    assign_pools,
    audit_partition_overlap,
    bind_pools,
    is_accepted,
    pool_assignment,
    rtl_overlap_key,
    verify_pools,
    verify_version,
    write_version,
)
from siliconshield.training.laya.inventory import feasibility, summarize
from siliconshield.training.laya.proposer_ingest import (
    ProposalError,
    ingest_proposals,
    load_current,
    validate_evidence,
)
from siliconshield.training.laya.replay_preservation import (
    preservation_diagnostics,
    preservation_loss,
    teacher_checkpoint,
    validate_preservation,
)
from siliconshield.training.laya.review_ledger import (
    ledger_digest,
    packet_assertion_sha,
    require_packet_binding,
    require_template_coverage,
    validate_ledger,
    validate_review,
)
from siliconshield.training.laya.source_contracts import candidates


# ---- Merged from test_dataset_versions.py ----
def rows_file(path, ids=("a", "b", "c", "d")):
    path.write_text("".join(json.dumps({"id": i, "x": 1}) + "\n" for i in ids))
    return path


def audit_for(manifest):
    from siliconshield.schemas import DATASET_AUDIT_SCHEMA

    return {
        "schema": DATASET_AUDIT_SCHEMA,
        "splits": {
            role: entry["sha256"] for role, entry in manifest["splits"].items()
        },
        "cross_split_exact_rtl_overlap": 0,
        "retired_rtl_overlap": 0,
        "truncated": 0,
        "lineage_overlap": 0,
        "canonical_overlap": 0,
        "max_group_share": 0.05,
        "max_mechanism_share": 0.15,
        "review_ledger_sha256": "ab" * 32,
        "reviewed_templates": 1,
        "release_class": "narrow-pilot",
    }


def test_version_pins_and_detects_drift(tmp_path):
    train = rows_file(tmp_path / "train.jsonl")
    manifest = write_version(tmp_path / "v", "v1", {"train": train})
    assert manifest["splits"]["train"]["rows"] == 4
    assert verify_version(tmp_path / "v", "v1") == manifest
    with pytest.raises(ValueError, match="exists"):
        write_version(tmp_path / "v", "v1", {"train": train})
    train.write_text("{}\n")
    with pytest.raises(ValueError, match="drift"):
        verify_version(tmp_path / "v", "v1")


def test_version_rejects_empty_and_shared_splits(tmp_path):
    train = rows_file(tmp_path / "train.jsonl")
    with pytest.raises(ValueError, match="at least one split"):
        write_version(tmp_path / "v", "empty", {})
    with pytest.raises(ValueError, match="share one split"):
        write_version(tmp_path / "v", "shared", {"train": train, "dev": train})
    rows_file(tmp_path / "blank.jsonl", ids=())
    with pytest.raises(ValueError, match="no rows"):
        write_version(
            tmp_path / "v", "blank", {"train": tmp_path / "blank.jsonl"}
        )


def test_acceptance_is_separate_and_audited(tmp_path):
    train = rows_file(tmp_path / "train.jsonl")
    manifest = write_version(tmp_path / "v", "v1", {"train": train})
    assert is_accepted(tmp_path / "v", "v1") is False
    bad = audit_for(manifest)
    bad["retired_rtl_overlap"] = 3
    with pytest.raises(ValueError, match="blocks training"):
        accept_version(tmp_path / "v", "v1", bad)
    legacy = {
        "splits": {
            role: entry["sha256"] for role, entry in manifest["splits"].items()
        },
        "cross_split_exact_rtl_overlap": 0,
        "retired_rtl_overlap": 0,
        "truncated": 0,
    }
    with pytest.raises(ValueError, match="audit schema"):
        accept_version(tmp_path / "v", "v1", legacy)
    recorded = accept_version(tmp_path / "v", "v1", audit_for(manifest))
    assert is_accepted(tmp_path / "v", "v1") is True
    assert accept_version(tmp_path / "v", "v1", audit_for(manifest)) == recorded
    changed = audit_for(manifest)
    changed["truncated"] = 0
    changed["note"] = "new"
    with pytest.raises(ValueError, match="new audit"):
        accept_version(tmp_path / "v", "v1", changed)


def test_strict_audit_requires_review_ledger_binding(tmp_path):
    train = rows_file(tmp_path / "train.jsonl")
    manifest = write_version(tmp_path / "v", "v1", {"train": train})
    for key, value in (
        ("review_ledger_sha256", "not-a-digest"),
        ("review_ledger_sha256", None),
        ("reviewed_templates", 0),
        ("reviewed_templates", "many"),
    ):
        bad = audit_for(manifest)
        bad[key] = value
        with pytest.raises(ValueError, match="SHA-256|positive"):
            accept_version(tmp_path / "v", "v1", bad)


def test_legacy_marker_stays_readable(tmp_path):
    import hashlib

    from siliconshield.research_loop.episodes.contracts import atomic_json
    from siliconshield.schemas import DATASET_ACCEPTANCE_SCHEMA

    train = rows_file(tmp_path / "train.jsonl")
    manifest = write_version(tmp_path / "v", "v1", {"train": train})
    legacy = {
        "splits": {
            role: entry["sha256"] for role, entry in manifest["splits"].items()
        },
        "cross_split_exact_rtl_overlap": 0,
        "retired_rtl_overlap": 0,
        "truncated": 0,
    }
    manifest_path = tmp_path / "v" / "versions" / "v1" / "manifest.json"
    atomic_json(
        tmp_path / "v" / "versions" / "v1" / "accepted.json",
        {
            "schema": DATASET_ACCEPTANCE_SCHEMA,
            "version": "v1",
            "manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "audit": legacy,
            "audit_sha256": hashlib.sha256(
                json.dumps(legacy, sort_keys=True).encode()
            ).hexdigest(),
        },
    )
    assert is_accepted(tmp_path / "v", "v1") is True


def accepted_marker(tmp_path):
    train = rows_file(tmp_path / "train.jsonl")
    manifest = write_version(tmp_path / "v", "v1", {"train": train})
    accept_version(tmp_path / "v", "v1", audit_for(manifest))
    assert is_accepted(tmp_path / "v", "v1") is True
    return tmp_path / "v/versions/v1/accepted.json"


def rewrite_marker(path, mutate):
    record = json.loads(path.read_text())
    mutate(record)
    path.write_text(json.dumps(record))


def test_acceptance_rejects_an_empty_record(tmp_path):
    marker = accepted_marker(tmp_path)
    marker.write_text("")
    assert is_accepted(tmp_path / "v", "v1") is False


def test_acceptance_rejects_a_truncated_record(tmp_path):
    marker = accepted_marker(tmp_path)
    marker.write_text('{"schema": "siliconshield-dataset-accept')
    assert is_accepted(tmp_path / "v", "v1") is False


def test_acceptance_rejects_a_record_for_another_version(tmp_path):
    marker = accepted_marker(tmp_path)
    rewrite_marker(marker, lambda r: r.update(version="v2"))
    assert is_accepted(tmp_path / "v", "v1") is False


def test_acceptance_rejects_a_detached_manifest(tmp_path):
    marker = accepted_marker(tmp_path)
    rewrite_marker(marker, lambda r: r.update(manifest_sha256="0" * 64))
    assert is_accepted(tmp_path / "v", "v1") is False


def test_acceptance_rejects_a_tampered_audit(tmp_path):
    marker = accepted_marker(tmp_path)
    rewrite_marker(marker, lambda r: r["audit"].update(retired_rtl_overlap=5))
    assert is_accepted(tmp_path / "v", "v1") is False


def test_acceptance_lapses_when_data_drifts(tmp_path):
    marker = accepted_marker(tmp_path)
    assert marker.is_file()
    (tmp_path / "train.jsonl").write_text("{}\n")
    assert is_accepted(tmp_path / "v", "v1") is False


def test_pools_partition_rows_exactly(tmp_path):
    rows = rows_file(tmp_path / "train.jsonl")
    manifest = assign_pools(
        rows,
        {
            "new": {"rows": ["a", "b", "c"], "weight": 0.5},
            "old": {"rows": ["d"], "weight": 0.5},
        },
        tmp_path / "pools.json",
    )
    assert manifest["pools"]["new"]["weight"] == 0.5
    assert verify_pools(rows, tmp_path / "pools.json") == manifest
    with pytest.raises(ValueError, match="two pools"):
        assign_pools(
            rows,
            {
                "new": {"rows": ["a", "b"], "weight": 1.0},
                "old": {"rows": ["b", "c", "d"], "weight": 1.0},
            },
            tmp_path / "dup.json",
        )
    with pytest.raises(ValueError, match="exactly"):
        assign_pools(
            rows, {"new": {"rows": ["a"], "weight": 1.0}}, tmp_path / "x.json"
        )
    with pytest.raises(ValueError, match="positive"):
        assign_pools(
            rows,
            {"new": {"rows": ["a", "b", "c", "d"], "weight": 0.0}},
            tmp_path / "y.json",
        )


def test_pool_binding_pins_weights_into_the_version(tmp_path):
    rows = rows_file(tmp_path / "train.jsonl")
    write_version(tmp_path / "v", "v1", {"train": rows})
    pools = tmp_path / "pools.json"
    assign_pools(
        rows,
        {
            "new": {"rows": ["a", "b", "c"], "weight": 0.5},
            "old": {"rows": ["d"], "weight": 0.5},
        },
        pools,
    )
    bind_pools(tmp_path / "v", "v1", pools)
    with pytest.raises(ValueError, match="already binds"):
        bind_pools(tmp_path / "v", "v1", pools)
    pools.write_text(
        json.dumps(
            {
                "rows": str(rows.resolve()),
                "rows_sha256": "0" * 64,
                "pools": {
                    "new": {"rows": ["a", "b", "c"], "weight": 999.0},
                    "old": {"rows": ["d"], "weight": 0.5},
                },
            }
        )
    )
    with pytest.raises(ValueError, match="pools changed"):
        verify_version(tmp_path / "v", "v1")


def test_verify_pools_is_read_only(tmp_path):
    rows = rows_file(tmp_path / "train.jsonl")
    pools = tmp_path / "pools.json"
    assign_pools(
        rows,
        {
            "new": {"rows": ["a", "b"], "weight": 1.0},
            "old": {"rows": ["c", "d"], "weight": 2.0},
        },
        pools,
    )
    before = pools.read_bytes()
    verify_pools(rows, pools)
    assert pools.read_bytes() == before


def test_pool_assignment_maps_indices_for_sampling(tmp_path):
    rows = rows_file(tmp_path / "train.jsonl")
    pools = tmp_path / "pools.json"
    assign_pools(
        rows,
        {
            "new": {"rows": ["a", "b"], "weight": 3.0},
            "old": {"rows": ["c", "d"], "weight": 1.0},
        },
        pools,
    )
    loaded = [json.loads(line) for line in rows.read_text().splitlines()]
    assignment, weights, modes = pool_assignment(pools, loaded)
    assert assignment == {0: "new", 1: "new", 2: "old", 3: "old"}
    assert weights == {"new": 3.0, "old": 1.0}
    assert modes == {"new": "cycle", "old": "cycle"}
    loaded[0] = {"id": "renamed"}
    with pytest.raises(ValueError, match="outside every pool"):
        pool_assignment(pools, loaded)


def state_row(row_id, rtl):
    return {"id": row_id, "state": "Requirement\nok\nRTL\n" + rtl + "\n"}


def state_file(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_overlap_key_ignores_whitespace_only():
    first = state_row("a", "assign q = a  &&  b;")
    second = state_row("b", "assign\tq=a&&b;")
    assert rtl_overlap_key(first) == rtl_overlap_key(second)
    assert rtl_overlap_key(state_row("c", "assign q = a;")) != (
        rtl_overlap_key(first)
    )
    with pytest.raises(ValueError, match="RTL section"):
        rtl_overlap_key({"id": "d", "state": "no rtl here"})


def test_audit_counts_cross_split_and_retired_only(tmp_path):
    train = state_file(
        tmp_path / "train.jsonl",
        [
            state_row("t1", "assign q = a;"),
            state_row("t2", "assign  q  =  a ;"),
            state_row("t3", "assign q = b;"),
        ],
    )
    dev = state_file(tmp_path / "dev.jsonl", [state_row("d1", "assign q=c;")])
    retired = state_file(
        tmp_path / "old.jsonl", [state_row("r1", "assign q = z;")]
    )
    clean = audit_partition_overlap({"train": train, "dev": dev}, [retired])
    assert clean == {
        "cross_split_exact_rtl_overlap": 0,
        "retired_rtl_overlap": 0,
    }
    leaked = state_file(
        tmp_path / "dev2.jsonl", [state_row("d2", "assign q=a;")]
    )
    stale = state_file(
        tmp_path / "old2.jsonl", [state_row("r2", "assign q=b;")]
    )
    dirty = audit_partition_overlap(
        {"train": train, "dev": leaked}, [retired, stale]
    )
    assert dirty == {
        "cross_split_exact_rtl_overlap": 1,
        "retired_rtl_overlap": 1,
    }
    broken = state_file(tmp_path / "bad.jsonl", [{"id": "x"}])
    with pytest.raises(ValueError, match="RTL section"):
        audit_partition_overlap({"train": broken, "dev": dev}, [])


# ---- Merged from test_source_contracts.py ----
RTL = """module topmodule(input clk, rst_n, hw_en, sw_en, authorized,
input [3:0] hw_data, sw_data, output reg [3:0] value);
always @(posedge clk) begin
if (!rst_n) value <= 0;
else if (hw_en) value <= hw_data;
else if (sw_en && authorized) value <= sw_data;
end
endmodule"""
SOURCE = {
    "name": "authored",
    "case": "example",
    "family": "example",
    "topic": "priority",
}


def test_population_counts_distinguish_files_records_specializations_and_lineage():
    def row(source, record, design, group, rtl):
        return {
            "source": {"name": source, "path": "register.v", "case": record},
            "base_design_id": design,
            "split_group_id": group,
            "state": "RTL\n" + rtl,
        }

    rows = [
        row("authored-a", "record-a", "width-small", "group-a", "small"),
        row("authored-a", "record-a", "width-large", "group-a", "large"),
        row("authored-a", "record-b", "width-large", "group-a", "large"),
        row("authored-b", "record-a", "width-small", "group-b", "other"),
    ]
    assert population_counts(rows) == {
        "decision_rows": 4,
        "source_files": 2,
        "source_records": 3,
        "base_design_ids": 3,
        "split_groups": 2,
        "unique_rtl_bodies": 3,
    }


def extract():
    pytest.importorskip("pyverilog")
    result, error = candidates(RTL, SOURCE)
    assert error is None
    return result


def test_sampled_antecedent_preserves_past_and_reset():
    assertion = data.hold_check("value", "!authorized", "rst_n")
    control = exact_antecedent(assertion)
    assert "$past(!authorized)" in control
    assert "$past(rst_n)" in control
    assert "assert(value ==" not in control
    assert "assert(1'b0);" in control
    assert (
        exact_antecedent("if (a) assert(q == $past(d)); done = 1;")
        == "if (a) assert(1'b0); done = 1;"
    )


def test_hardware_override_not_discarded_from_hold_contract():
    result = extract()
    holds = [
        c for c in result if c["kind"] == "hold" and c["policy"] == "original"
    ]
    assert holds
    for case in holds:
        assert "hw_en" in case["assertion"] and "sw_en" in case["assertion"]
        assert "authorized" in case["assertion"]
    updates = [c for c in result if c["kind"] == "update"]
    assert any("hw_data" in c["assertion"] for c in updates)
    assert any("sw_data" in c["assertion"] for c in updates)


def test_counterfactuals_and_missing_expressions_are_exact():
    result = extract()
    assert {c["policy"] for c in result} == {"original", "counterfactual"}
    for case in result:
        for variant in data.VARIANTS:
            rendered = data.render_variant(case, variant)
            assert rendered["good"] != rendered["bad"]
            assert rendered["good"] != rendered["opaque"]
            # render_variant checks both completions after identifier rewriting.
            assert rendered["relevant_hole"] in rendered["opaque"]


def test_unsupported_reset_is_not_silently_guessed():
    pytest.importorskip("pyverilog")
    result, error = candidates(RTL.replace("rst_n", "reset_custom"), SOURCE)
    assert not result and error == "requires_supported_clock_and_reset"


def test_additional_clock_or_async_reset_requires_another_model():
    pytest.importorskip("pyverilog")
    extra_clock = RTL.replace(
        "endmodule", "always @(posedge other_clk) value <= 0; endmodule"
    )
    result, error = candidates(extra_clock, SOURCE)
    assert not result and error == "unsupported_additional_clock_or_async_reset"
    extra_reset = RTL.replace(
        "posedge clk)", "posedge clk or negedge warm_reset)"
    )
    result, error = candidates(extra_reset, SOURCE)
    assert not result and error == "unsupported_additional_clock_or_async_reset"


@pytest.mark.skipif(
    not (shutil.which("yosys") and shutil.which("iverilog")),
    reason="requires Yosys and Icarus",
)
def test_solver_proves_both_policies_and_compiles_opaque_rows(tmp_path):
    result = extract()
    pair = [
        c
        for c in result
        if c["kind"] == "update" and "hw_data" in c["assertion"]
    ][:2]
    assert len(pair) == 2
    for case in pair:
        case["split"] = "train"
    # Repeated callers share content-addressed proof and compilation directories.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda c: checked_rows(
                    c, shutil.which("yosys"), shutil.which("iverilog"), tmp_path
                ),
                pair + pair,
            )
        )
    for rows in results:
        assert len(rows) == 9
        assert {r["expected"]["security"] for r in rows} == set(data.LABELS)
        assert all(r["validation"]["compile_evidence"] for r in rows)
        assert all(r["validation"]["exact_antecedent_evidence"] for r in rows)
    for variant in data.VARIANTS:
        left = next(
            r
            for r in results[0]
            if r["variant"] == variant
            and r["expected"]["security"] == data.LABELS[0]
        )
        right = next(
            r
            for r in results[1]
            if r["variant"] == variant
            and r["expected"]["security"] == data.LABELS[1]
        )
        assert (
            left["state"].split("RTL\n")[1] == right["state"].split("RTL\n")[1]
        )
        assert (
            left["state"].split("RTL\n")[0] != right["state"].split("RTL\n")[0]
        )
        assert left["pair_id"] != right["pair_id"]


# ---- Merged from test_question_contracts.py ----
class QuestionContractsTest(unittest.TestCase):
    def test_legacy_matches_worker_bytes(self):
        worker = WORKER_QUESTION["security"]
        built = contracts.question_for(contracts.LEGACY_CONTRACT)
        self.assertEqual(
            built["security"]["instructions"], worker["instructions"]
        )
        self.assertEqual(built["security"]["criteria"], worker["criteria"])

    def test_registry_round_trip(self):
        for version in contracts.known_contracts():
            questions = contracts.question_for(version)
            self.assertEqual(contracts.match_version(questions), version)
            self.assertEqual(len(contracts.digest_for(version)), 64)

    def test_unknown_version_refused(self):
        with self.assertRaises(ValueError):
            contracts.question_for("no-such-contract")
        with self.assertRaises(ValueError):
            contracts.digest_for("no-such-contract")

    def test_foreign_questions_refused(self):
        with self.assertRaises(ValueError):
            contracts.match_version(
                {
                    "security": {
                        "instructions": "something else",
                        "criteria": list(contracts.LABELS),
                    }
                }
            )
        with self.assertRaises(ValueError):
            contracts.match_version({})

    def test_validate_rows(self):
        rows = [
            {
                "id": "r%d" % i,
                "questions": contracts.question_for(contracts.NEW_CONTRACT),
            }
            for i in range(3)
        ]
        self.assertEqual(
            contracts.validate_rows(rows, contracts.NEW_CONTRACT),
            contracts.NEW_CONTRACT,
        )
        mixed = rows + [
            {
                "id": "old",
                "questions": contracts.question_for(contracts.LEGACY_CONTRACT),
            }
        ]
        with self.assertRaises(ValueError):
            contracts.validate_rows(mixed, contracts.NEW_CONTRACT)

    def test_export_record_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            record = contracts.write_export_record(tmp, contracts.NEW_CONTRACT)
            self.assertEqual(record["version"], contracts.NEW_CONTRACT)
            read = contracts.read_export_record(tmp)
            self.assertEqual(read["version"], contracts.NEW_CONTRACT)
            self.assertFalse(read["legacy_default"])

    def test_legacy_default_for_unrecorded_export(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            read = contracts.read_export_record(tmp)
            self.assertEqual(read["version"], contracts.LEGACY_CONTRACT)
            self.assertTrue(read["legacy_default"])

    def test_tampered_record_refused(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            contracts.write_export_record(tmp, contracts.NEW_CONTRACT)
            path = Path(tmp) / "question_contract.json"
            record = json.loads(path.read_text())
            record["instructions"] = "tampered"
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                contracts.read_export_record(tmp)


if __name__ == "__main__":
    unittest.main()


# ---- Merged from test_proposer_ingest.py ----
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


# ---- Merged from test_review_ledger.py (aliased: _row) ----
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


def _ledger_row(
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
    require_template_coverage([_ledger_row()], records)
    with pytest.raises(ValueError, match="lacks agreed review"):
        require_template_coverage([_ledger_row("r2", "t-missing")], records)


def test_coverage_binds_reviewed_hashes(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(_record("t1")) + "\n")
    records = validate_ledger(path)
    with pytest.raises(ValueError, match="differs from reviewed template"):
        require_template_coverage([_ledger_row(requirement="changed")], records)


def test_coverage_recomputes_bytes_not_row_claims(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(_record("t1")) + "\n")
    records = validate_ledger(path)
    row = _ledger_row(requirement="changed")
    # Self-declared hash still claims the old bytes: must fail anyway.
    row["requirement_sha256"] = _sha("req text")
    with pytest.raises(ValueError, match="differs from reviewed template"):
        require_template_coverage([row], records)


def test_coverage_admits_approved_disagreement(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(_record("t1", "disagree")) + "\n")
    records = validate_ledger(path)
    require_template_coverage([_ledger_row()], records)


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


# ---- Merged from test_replay_preservation.py ----
def config():
    return {
        "version": REPLAY_PRESERVATION_SCHEMA,
        "weight": 1.0,
        "temperature": 2.0,
        "pools": ["retained"],
    }


@pytest.mark.parametrize("invalid", [True, 0, -1, float("nan")])
def test_invalid_teacher_weight_refuses(invalid):
    value = config()
    value["weight"] = invalid
    with pytest.raises(ValueError, match="positive weight"):
        validate_preservation(value)


def test_teacher_configuration_is_optional_and_copied():
    assert validate_preservation(None) is None
    value = config()
    result = validate_preservation(value)
    result["pools"].append("additional")
    assert value["pools"] == ["retained"]


def test_consolidation_teacher_is_independent_of_student_initialization():
    value = config()
    assert teacher_checkpoint(value, "newly-fitted") == "newly-fitted"
    value["teacher_checkpoint"] = "previously-accepted"
    assert teacher_checkpoint(value, "newly-fitted") == "previously-accepted"


@pytest.mark.parametrize("invalid", [None, True, "", "  "])
def test_invalid_explicit_teacher_path_refuses(invalid):
    value = config()
    value["teacher_checkpoint"] = invalid
    with pytest.raises(ValueError, match="teacher checkpoint"):
        validate_preservation(value)


def test_only_correct_teacher_on_eligible_rows_has_gradient():
    torch = pytest.importorskip("torch")
    teacher = torch.tensor(
        [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
        requires_grad=True,
    )
    student = torch.zeros(3, 3, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]] * 3)
    losses = preservation_loss(
        student, teacher, target, torch.tensor([True, False, True]), 2.0
    )
    assert losses[0] > 0
    assert losses[1:].tolist() == [0, 0]
    losses.sum().backward()
    assert teacher.grad is None
    assert student.grad[0, 0] < 0
    assert student.grad[1:].count_nonzero() == 0


def test_identical_logits_have_zero_loss_and_option_permutation_is_invariant():
    torch = pytest.importorskip("torch")
    teacher = torch.tensor([[4.0, 0.0, -2.0]])
    student = torch.tensor([[2.0, 1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([True])
    assert preservation_loss(teacher, teacher, target, mask, 2.0).item() == 0
    loss = preservation_loss(student, teacher, target, mask, 2.0)
    order = [2, 0, 1]
    changed = preservation_loss(
        student[:, order], teacher[:, order], target[:, order], mask, 2.0
    )
    torch.testing.assert_close(loss, changed)


def test_partial_accumulation_matches_a_single_weighted_batch():
    torch = pytest.importorskip("torch")
    student = torch.tensor([[1.0, 0.0, 0.0]] * 3, requires_grad=True)
    teacher = torch.tensor([[4.0, 0.0, 0.0]] * 3)
    target = torch.tensor([[1.0, 0.0, 0.0]] * 3)
    mask = torch.ones(3, dtype=torch.bool)
    validity = torch.tensor([1.0, 1.0, 0.0])
    full = preservation_loss(student, teacher, target, mask, 2.0) * validity
    full.sum().div(2).backward()
    expected = student.grad.clone()
    student.grad = None
    for lo, hi in [(0, 2), (2, 3)]:
        part = preservation_loss(
            student[lo:hi], teacher[lo:hi], target[lo:hi], mask[lo:hi], 2.0
        )
        (part * validity[lo:hi]).sum().div(2).backward()
    torch.testing.assert_close(student.grad, expected)


def test_diagnostics_mirror_the_loss_mask():
    torch = pytest.importorskip("torch")
    student = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    teacher = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    target = torch.tensor([[1.0, 0.0, 0.0]] * 5)
    eligible = torch.tensor([True, True, True, False, True])
    valid = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    result = preservation_diagnostics(student, teacher, target, eligible, valid)
    assert result == {
        "pool_rows": 3,
        "teacher_match_rows": 2,
        "student_teacher_agree_rows": 1,
    }
    penalty = preservation_loss(student, teacher, target, eligible, 2.0)
    nonzero = {index for index, value in enumerate(penalty) if value > 0}
    assert nonzero == {0, 1, 4}


# ---- Merged from test_inventory.py (aliased: _row) ----
def _inventory_row(origin, group, mechanism="locking"):
    return {
        "id": "%s-%s" % (origin, group),
        "origin": origin,
        "split_group_id": group,
        "mechanism_id": mechanism,
    }


def test_narrow_pilot_when_inventory_short():
    rows = [_inventory_row("auth", "g%d" % i) for i in range(10)]
    report = summarize(rows)
    assert report["train_groups"] == 10
    verdict = feasibility(
        {**report, "dev_groups": 2},
        {"train_groups": 80, "dev_groups": 24},
    )
    assert verdict["broad_release"] is False
    assert verdict["release_class"] == "narrow-pilot"


def test_broad_release_when_targets_met():
    rows = [_inventory_row("o%d" % (i % 5), "g%d" % i) for i in range(90)]
    report = summarize(rows)
    verdict = feasibility(
        {
            **report,
            "dev_groups": 30,
            "calibration_groups": 12,
            "test_groups": 24,
        },
        {"train_groups": 80, "dev_groups": 24},
    )
    assert verdict["broad_release"] is True


def test_partial_report_is_narrow_pilot():
    rows = [_inventory_row("o%d" % (i % 5), "g%d" % i) for i in range(90)]
    report = summarize(rows)
    verdict = feasibility({**report, "dev_groups": 30})
    assert verdict["broad_release"] is False
    assert verdict["release_class"] == "narrow-pilot"


# ---- Merged from test_coverage_audit.py ----
def make_rows():
    rows = []
    for pair in range(4):
        for member, label in enumerate(
            (
                "satisfies_requirement",
                "violates_requirement",
                "insufficient_context",
            )
        ):
            rows.append(
                {
                    "id": "r%d-%d" % (pair, member),
                    "pair_id": "p%d" % pair,
                    "expected": {"security": label},
                }
            )
    return rows


def test_triplet_split_and_broken():
    rows = make_rows()
    assignment = {i: ("a" if i % 3 else "b") for i in range(len(rows))}
    report = triplet_integrity(rows, assignment, set(range(len(rows))))
    assert report["triplets"] == 4
    assert report["split"] == ["p0", "p1", "p2", "p3"]
    assert report["broken"] == []
    clean = {i: "a" for i in range(len(rows))}
    report = triplet_integrity(rows, clean, set(range(len(rows) - 1)))
    assert report["split"] == []
    assert report["broken"] == ["p3"]


def test_quota_schedule_inherits_and_validates():
    schedule = {"1": {"new": 24, "replay": 24}, "3": {"new": 36, "replay": 12}}
    assert quotas_for_round(schedule, 1) == {"new": 24, "replay": 24}
    assert quotas_for_round(schedule, 2) == {"new": 24, "replay": 24}
    assert quotas_for_round(schedule, 4) == {"new": 36, "replay": 12}
    for bad in ({}, {"1": {}}, {"1": {"new": 0}}, {"x": {"new": 1}}):
        try:
            quotas_for_round(bad, 1)
        except ValueError:
            continue
        raise AssertionError("accepted bad schedule: %r" % (bad,))
    try:
        quotas_for_round(schedule, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("round before first entry accepted")


def test_simulation_mirrors_sampler_calls():
    rows = make_rows()
    assignment = {i: ("new" if i < 6 else "replay") for i in range(len(rows))}
    calls = []

    def fake_sampler(
        rows_arg,
        world,
        rank,
        seed,
        epoch,
        batch,
        pool_plan=None,
        _queue_state=None,
    ):
        calls.append((rank, epoch, seed, batch, _queue_state is not None))
        _queue_state["epochs"] = _queue_state.get("epochs", 0) + 1
        base = rank * 4
        return [base + (epoch + k) % 4 for k in range(4)]

    report = simulate_coverage(
        rows,
        (
            assignment,
            {"new": 0.5, "replay": 0.5},
            {"new": "flat", "replay": "flat"},
        ),
        77,
        4,
        world=3,
        global_batch=6,
        per_rank=2,
        grouped_epoch_indices=fake_sampler,
    )
    assert report["updates"] == 4
    assert report["epochs"] == 2
    assert sum(report["seen"].values()) == 24
    # Per-rank carried state: each rank saw distinct epoch sequence.
    by_rank = Counter(rank for rank, _, _, _, _ in calls)
    assert by_rank == {0: 2, 1: 2, 2: 2}, by_rank
    assert all(carried for _, _, _, _, carried in calls)
    assert all(sum(q.values()) == 6 for q in report["quotas"])


if __name__ == "__main__":
    test_triplet_split_and_broken()
    test_quota_schedule_inherits_and_validates()
    test_simulation_mirrors_sampler_calls()
    print("coverage_audit tests: 3 passed")


# ---- Merged from test_verification_bounds.py ----
def test_solver_commands_honor_declared_bound():
    bounded, induction = solver_commands("assert(1);", "rst_n")
    assert "sat -seq 6" in bounded
    assert "maxsteps 12" in induction
    custom_bounded, custom_induction = solver_commands(
        "assert(1);", "rst_n", bounded_cycles=10, induction_steps=20
    )
    assert "sat -seq 10" in custom_bounded
    assert "maxsteps 20" in custom_induction
    with pytest.raises(ValueError, match="bounded_cycles"):
        solver_commands("assert(1);", "rst_n", bounded_cycles=0)


def test_contract_bounds_default_and_validate(tmp_path):
    registry = tmp_path / "contracts.json"
    import json

    registry.write_text(
        json.dumps(
            {
                "c1": {
                    "requirement": "Hold when locked.",
                    "assertion": "assert(q);",
                    "scope": "sequential",
                    "reset": "rst_n",
                },
                "c2": {
                    "requirement": "Hold when locked deeply.",
                    "assertion": "assert(q);",
                    "scope": "sequential",
                    "reset": "rst_n",
                    "bound_cycles": 10,
                    "induction_steps": 20,
                },
            }
        )
    )
    loaded = load_contracts(registry)
    assert loaded["c1"]["bound_cycles"] == 6
    assert loaded["c2"]["bound_cycles"] == 10
