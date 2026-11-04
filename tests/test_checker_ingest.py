"""Checker, ingest, and dataset-builder input-shape tests: synthetic fixtures."""

import json

from siliconshield.schemas import PROPOSAL_SCHEMA
from siliconshield.training.laya import build_security_data as data
from siliconshield.training.laya import checker_adapter as checker
from siliconshield.training.laya import proposer_ingest as ingest
from siliconshield.training.laya.proposer_ingest import ProposalError


def _contracts_file(tmp_path):
    requirement = data.requirement_for("q", "sel", "comb_zero")
    registry = {
        "c1": {
            "requirement": requirement,
            "assertion": data.hold_check("q", "sel", kind="comb_zero"),
            "scope": "combinational",
            "reset": "",
        }
    }
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(registry))
    return path, requirement


def _checker(tmp_path):
    path, requirement = _contracts_file(tmp_path)
    instance = checker.FixedChecker(path, workdir=tmp_path / "work")
    proposal = {
        "id": "t",
        "contract_id": "c1",
        "requirement": requirement,
        "assumptions": checker.REGISTERED_ASSUMPTIONS["combinational"],
        "rtl": "module topmodule(input wire sel, output wire q);\n"
        "assign q = sel ? 1'b0 : 1'b1;\nendmodule\n",
    }
    return instance, proposal


def test_contract_registry_rejects_malformed_entries(tmp_path):
    import pytest

    requirement = data.requirement_for("q", "sel", "comb_zero")
    good = {
        "requirement": requirement,
        "assertion": data.hold_check("q", "sel", kind="comb_zero"),
        "scope": "combinational",
        "reset": "",
    }
    for registry in ({"c1": ["not", "a", "dict"]}, {"": dict(good)}):
        path = tmp_path / "contracts.json"
        path.write_text(json.dumps(registry))
        with pytest.raises(ValueError):
            checker.load_contracts(path)


def test_proposal_reader_rejects_malformed_files(tmp_path):
    import pytest

    folder = tmp_path / "proposals"
    folder.mkdir()
    (folder / "a.json").write_text("[1, 2]")
    with pytest.raises(ProposalError):
        ingest._read_proposals(folder)
    (folder / "a.json").write_text(
        json.dumps(
            {
                "schema": PROPOSAL_SCHEMA,
                "id": "p",
                "contract_id": "c1",
                "requirement": "R",
                "assumptions": "A",
                "rtl": 5,
                "kind": "complete",
            }
        )
    )
    with pytest.raises(ProposalError):
        ingest._read_proposals(folder)


def test_checker_rejects_malformed_proposals_without_solvers(tmp_path):
    import pytest

    instance, proposal = _checker(tmp_path)
    bad = dict(proposal, rtl=5)
    with pytest.raises(ProposalError, match="RTL text"):
        instance.check(bad)
    bad = {k: v for k, v in proposal.items() if k != "id"}
    with pytest.raises(ProposalError, match="evidence id"):
        instance.check(bad)
    bad = dict(proposal, completions=5)
    with pytest.raises(ProposalError, match="Completions must be a list"):
        instance.check(bad)
    with pytest.raises(ProposalError, match="must be an object"):
        instance.check(["not", "a", "dict"])


def test_evidence_validation_rejects_malformed_outcomes():
    import hashlib

    import pytest

    rtl = "module topmodule(input wire sel, output wire q);\nendmodule\n"
    assumptions = checker.REGISTERED_ASSUMPTIONS["combinational"]
    digests = {
        "rtl_sha256": hashlib.sha256(rtl.encode()).hexdigest(),
        "assumptions_sha256": hashlib.sha256(assumptions.encode()).hexdigest(),
    }
    with pytest.raises(ProposalError):
        ingest.validate_evidence("R", assumptions, rtl, "c1", ["x"])
    with pytest.raises(ProposalError):
        ingest.validate_evidence("R", assumptions, 5, "c1", {})
    base = {
        "label": "satisfies_requirement",
        "evidence": dict(
            {
                "checker_id": "c",
                "contract_id": "c1",
                "proof": "not-a-dict",
            },
            **digests,
        ),
    }
    with pytest.raises(ProposalError):
        ingest.validate_evidence("R", assumptions, rtl, "c1", base)
    base["label"] = "violates_requirement"
    base["evidence"] = dict(base["evidence"], counterexample="not-a-dict")
    with pytest.raises(ProposalError):
        ingest.validate_evidence("R", assumptions, rtl, "c1", base)
    base["label"] = "insufficient_context"
    base["evidence"] = dict(
        base["evidence"],
        hole={"name": "h", "width": 1},
        completions=[
            {"outcome": {"unhashable": True}, "fill": "a"},
            {"outcome": "violates_requirement", "fill": "b"},
        ],
    )
    with pytest.raises(ProposalError):
        ingest.validate_evidence("R", assumptions, rtl, "c1", base)


def test_compile_rejects_escaping_evidence_ids(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="escapes its folder"):
        data.compile_rtl("code", "proposal-../../evil", "iverilog", tmp_path)


def test_materialize_rejects_malformed_catalogs(tmp_path):
    import pytest

    from siliconshield.training.laya import build_source_security as build

    for cases in (
        {"c1": {"uid": "x"}},
        [{"split": "train"}],
        [{"uid": "x"}],
        ["hello"],
    ):
        with pytest.raises(ValueError):
            build.materialize(
                cases, tmp_path / "out", tmp_path / "v", "yosys", "iverilog"
            )


def test_validate_dataset_rejects_empty_splits():
    import pytest

    with pytest.raises(ValueError, match="no splits"):
        data.validate_dataset({})


def _version_rows(tmp_path):
    rows = tmp_path / "train.jsonl"
    rows.write_text(
        '{"id": "a", "state": "RTL\\nwire x;"}\n'
        '{"id": "b", "state": "RTL\\nwire y;"}\n'
    )
    return rows


def test_write_version_rejects_malformed_inputs(tmp_path):
    import pytest

    from siliconshield.training.laya import dataset_versions as versions

    rows = _version_rows(tmp_path)
    with pytest.raises(ValueError, match="escapes its folder"):
        versions.write_version(tmp_path, "../evil", {"train": str(rows)})
    with pytest.raises(ValueError, match="escapes its folder"):
        versions.write_version(tmp_path, 5, {"train": str(rows)})
    with pytest.raises(ValueError, match="at least one split"):
        versions.write_version(tmp_path, "v", ["train"])
    with pytest.raises(ValueError, match="role name"):
        versions.write_version(tmp_path, "v", {7: str(rows)})
    with pytest.raises(ValueError, match="needs a path"):
        versions.write_version(tmp_path, "v", {"train": 5})
    assert not (tmp_path / "evil").exists()


def test_verify_version_rejects_malformed_manifests(tmp_path):
    import pytest

    from siliconshield.schemas import DATASET_VERSION_SCHEMA
    from siliconshield.training.laya import dataset_versions as versions

    rows = _version_rows(tmp_path)
    versions.write_version(tmp_path, "v", {"train": str(rows)})
    manifest_path = tmp_path / "versions" / "v" / "manifest.json"
    original = manifest_path.read_text()
    try:
        for text in (
            "[1, 2]",
            json.dumps({"schema": DATASET_VERSION_SCHEMA}),
            json.dumps({"schema": DATASET_VERSION_SCHEMA, "splits": []}),
            json.dumps(
                {
                    "schema": DATASET_VERSION_SCHEMA,
                    "splits": {"train": {"path": str(rows)}},
                }
            ),
        ):
            manifest_path.write_text(text)
            with pytest.raises(ValueError):
                versions.verify_version(tmp_path, "v")
    finally:
        manifest_path.write_text(original)


def test_pool_helpers_reject_malformed_shapes(tmp_path):
    import pytest

    from siliconshield.training.laya import dataset_versions as versions

    rows = _version_rows(tmp_path)
    with pytest.raises(ValueError, match="must be an object"):
        versions._check_partition(["a"], ["x"])
    with pytest.raises(ValueError, match="spec must be an object"):
        versions._check_partition(["a"], {"p": 5})
    with pytest.raises(ValueError, match="string list"):
        versions._check_partition(
            ["a"], {"p": {"weight": 1.0, "rows": [{"x": 1}]}}
        )
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a"}\n[1]\n')
    with pytest.raises(ValueError, match="not an object"):
        versions._row_ids(bad)
    bad.write_text('{"noid": 1}\n')
    with pytest.raises(ValueError, match="string id"):
        versions._row_ids(bad)
    manifest = tmp_path / "pools.json"
    manifest.write_text("[1]")
    with pytest.raises(ValueError, match="must be an object"):
        versions.pool_assignment(manifest, [])
    manifest.write_text(
        json.dumps({"pools": {"p": {"weight": 1.0, "rows": ["a"]}}})
    )
    with pytest.raises(ValueError, match="must be an object"):
        versions.pool_assignment(manifest, ["x"])


def test_version_pool_lifecycle_still_accepts(tmp_path):
    import hashlib

    from siliconshield.schemas import (
        DATASET_AUDIT_SCHEMA,
    )
    from siliconshield.training.laya import dataset_versions as versions

    rows = _version_rows(tmp_path)
    versions.write_version(tmp_path, "v", {"train": str(rows)})
    pools_path = tmp_path / "pools.json"
    versions.assign_pools(
        rows, {"p": {"weight": 1.0, "rows": ["a", "b"]}}, pools_path
    )
    versions.bind_pools(tmp_path, "v", pools_path)
    versions.verify_version(tmp_path, "v")
    digest = hashlib.sha256(rows.read_bytes()).hexdigest()
    audit = {
        "schema": DATASET_AUDIT_SCHEMA,
        "splits": {"train": digest},
        "cross_split_exact_rtl_overlap": 0,
        "retired_rtl_overlap": 0,
        "truncated": 0,
        "lineage_overlap": 0,
        "canonical_overlap": 0,
        "max_group_share": 0.01,
        "max_mechanism_share": 0.05,
        "release_class": "narrow-pilot",
        "review_ledger_sha256": hashlib.sha256(b"ledger").hexdigest(),
        "reviewed_templates": 1,
    }
    versions.accept_version(tmp_path, "v", audit)
    assert versions.is_accepted(tmp_path, "v")


def test_overlap_auditors_reject_malformed_inputs(tmp_path):
    import pytest

    from siliconshield.training.laya import dataset_versions as versions

    dev = tmp_path / "dev.jsonl"
    dev.write_text(
        '{"id": "a", "state": "X\\nRTL\\nwire x;", "split_group_id": "g"}\n'
    )
    with pytest.raises(ValueError, match="needs role"):
        versions.audit_partition_overlap({"dev": dev}, [])
    with pytest.raises(ValueError, match="needs role"):
        versions.audit_lineage_overlap({"dev": dev})
    with pytest.raises(ValueError, match="needs role"):
        versions.audit_canonical_overlap({"dev": dev})
    with pytest.raises(ValueError, match="split map"):
        versions.audit_split_groups(["dev"])
    train = tmp_path / "train.jsonl"
    train.write_text(
        '{"id": "a", "state": "X\\nRTL\\nwire x;",'
        ' "split_group_id": "g", "mechanism_id": {"m": 1}}\n'
    )
    with pytest.raises(ValueError, match="mechanism is malformed"):
        versions.audit_concentration(train)
