"""Dataset versions: pinning, acceptance, pools, and drift refusal."""

import json

import pytest

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
