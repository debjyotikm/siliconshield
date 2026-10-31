"""Lineage, canonical, and concentration gates."""

import json

from siliconshield.training.laya.dataset_versions import (
    accept_version,
    audit_canonical_overlap,
    audit_concentration,
    audit_lineage_overlap,
    audit_split_groups,
    canonical_rtl_key,
    write_version,
)


def _row(row_id, rtl, group, mechanism="locking"):
    return {
        "id": row_id,
        "state": "REQUIREMENT\nr\nASSUMPTIONS\na\nRTL\n%s\n" % rtl,
        "split_group_id": group,
        "dependency_group_id": group,
        "mechanism_id": mechanism,
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_lineage_and_canonical_overlap_refuse(tmp_path):
    train = _write(
        tmp_path / "train.jsonl",
        [_row("t1", "assign q = a;", "g1")],
    )
    dev = _write(
        tmp_path / "dev.jsonl",
        [_row("d1", "assign q = a;", "g1")],
    )
    splits = {"train": train, "dev": dev}
    assert audit_lineage_overlap(splits)["lineage_overlap"] == 1
    assert audit_canonical_overlap(splits)["canonical_overlap"] == 1
    assert audit_split_groups(splits) == {"train": 1, "dev": 1}


def test_concentration_caps_enforced(tmp_path):
    rows = [_row("t%d" % i, "assign q%d = a;" % i, "big") for i in range(10)]
    train = _write(tmp_path / "train.jsonl", rows)
    report = audit_concentration(train)
    assert report["max_group_share"] == 1.0
    manifest = write_version(tmp_path / "v", "v1", {"train": str(train)})
    audit = _accept_audit(manifest, {"max_group_share": 1.0})
    try:
        accept_version(tmp_path / "v", "v1", audit)
    except ValueError as exc:
        assert "max_group_share" in str(exc)
    else:
        raise AssertionError("concentration cap not enforced")


def _accept_audit(manifest, extra):
    from siliconshield.schemas import DATASET_AUDIT_SCHEMA

    audit = {
        "schema": DATASET_AUDIT_SCHEMA,
        "splits": {
            "train": manifest["splits"]["train"]["sha256"],
        },
        "cross_split_exact_rtl_overlap": 0,
        "retired_rtl_overlap": 0,
        "truncated": 0,
        "lineage_overlap": 0,
        "canonical_overlap": 0,
        "max_group_share": 0.05,
        "max_mechanism_share": 0.15,
        "review_ledger_sha256": "ab" * 32,
        "reviewed_templates": 2,
        "release_class": "narrow-pilot",
    }
    audit.update(extra)
    return audit


def test_acceptance_records_release_class(tmp_path):
    rows = [
        _row("t%d" % i, "assign q%d = a;" % i, "g%d" % i) for i in range(10)
    ]
    train = _write(tmp_path / "train.jsonl", rows)
    manifest = write_version(tmp_path / "v", "v1", {"train": str(train)})
    audit = _accept_audit(manifest, {"train_groups": 10})
    recorded = accept_version(tmp_path / "v", "v1", audit)
    assert recorded["audit"]["release_class"] == "narrow-pilot"
    # Re-accepting with the original audit is idempotent.
    again = accept_version(tmp_path / "v", "v1", dict(audit))
    assert again["audit_sha256"] == recorded["audit_sha256"]


def test_acceptance_refuses_false_broad_claim(tmp_path):
    rows = [
        _row("t%d" % i, "assign q%d = a;" % i, "g%d" % i) for i in range(10)
    ]
    train = _write(tmp_path / "train.jsonl", rows)
    manifest = write_version(tmp_path / "v", "v2", {"train": str(train)})
    audit = _accept_audit(
        manifest, {"train_groups": 10, "release_class": "broad"}
    )
    try:
        accept_version(tmp_path / "v", "v2", audit)
    except ValueError as exc:
        assert "broad release" in str(exc)
    else:
        raise AssertionError("false broad claim not refused")


def test_canonical_key_ignores_consistent_renames():
    base = _row("t1", "assign q = a && b;", "g1")
    renamed = _row("t2", "assign sig_x = sig_y && sig_z;", "g2")
    assert canonical_rtl_key(base) == canonical_rtl_key(renamed)


def test_canonical_key_keeps_operators_widths_literals():
    base = _row("t1", "assign q = a && b;", "g1")
    swapped = _row("t2", "assign q = a || b;", "g2")
    assert canonical_rtl_key(base) != canonical_rtl_key(swapped)
    narrow = _row("t3", "assign q = 4'd0;", "g3")
    wide = _row("t4", "assign q = 8'd0;", "g4")
    assert canonical_rtl_key(narrow) != canonical_rtl_key(wide)
    commented = _row("t5", "// note\nassign q=a&&b;", "g5")
    assert canonical_rtl_key(base) == canonical_rtl_key(commented)
