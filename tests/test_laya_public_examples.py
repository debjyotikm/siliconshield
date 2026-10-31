"""Public authored examples: builders must work without any local experiment manifests."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "siliconshield" / "training" / "laya"


def test_example_catalog_shape_and_split_ownership():
    from siliconshield.training.laya import build_security_data as data

    catalog = data.load_case_catalog()
    assert catalog["train_dev_cases"] and catalog["outer_cases"]
    assert all(
        case[0].startswith("authored-example-")
        for case in catalog["train_dev_cases"] + catalog["outer_cases"]
    )
    data.apply_catalog(catalog)
    assert data.split_for(catalog["dev_ids"][0]) == "dev"
    assert data.split_for("unknown-case") == "train"


def test_local_catalog_manifest_roundtrip(tmp_path):
    from siliconshield.training.laya import build_security_data as data

    manifest = tmp_path / "cases.json"
    manifest.write_text(
        json.dumps(
            {
                "train_dev_cases": [
                    ["local-1", "sig", "guard", "!guard", "guard"]
                ],
                "outer_cases": [],
                "dev_ids": [],
                "calibration_ids": [],
                "test_ids": ["local-1"],
                "mutation_occurrences": {"local-1": [0, 1]},
                "expected_counts": None,
                "manifest_seed": 7,
            }
        )
    )
    catalog = data.load_case_catalog(manifest)
    data.apply_catalog(catalog)
    assert data.split_for("local-1") == "test"
    assert data.MUTATION_OCCURRENCES == {"local-1": (0, 1)}


def test_authored_external_case_is_synthetic():
    from siliconshield.training.laya import (
        build_security_external as external,
    )

    cases = external.build_cases(Path("/nonexistent-sources"))
    assert len(cases) == 1
    assert cases[0]["source"]["name"] == "authored-synthetic"
    assert cases[0]["completion_checks"]["good_exact"]


def test_authored_semantic_scenario_is_illustrative():
    from siliconshield.training.laya import expand_security_data as expand

    case = expand.scenario(0, 0)
    assert case["mechanism_id"] == "privilege_lock"
    assert case["source"]["name"] == "authored-semantic"
    assert "POLICY" in case["code"]


def test_authored_opentitan_fallback_and_no_embedded_cases():
    from siliconshield.training.laya import build_security_data as data

    cases = list(data.opentitan_cases(Path("/nonexistent-sources")))
    assert (
        len(cases) == 1 and cases[0]["source"]["name"] == "authored-synthetic"
    )
    text = (TRAINING / "build_security_data.py").read_text()
    # Loader machinery may reference provider paths, but case definitions must not be embedded.
    assert "Software has priority" not in text


def test_guard_shuffle_seed_controls_deterministic_ordering():
    from siliconshield.training.laya import augment_security_guards as guards

    keys = ["pair-%02d" % i for i in range(24)]
    # Arbitrary fixture seeds; determinism and seed-sensitivity are the behavior under test.
    first = guards.shuffled_order(keys, 12345)
    assert first == guards.shuffled_order(keys, 12345)
    assert sorted(first) == keys
    assert guards.shuffled_order(keys, 6789) != first
    assert "--seed" in (TRAINING / "augment_security_guards.py").read_text()


def test_public_source_has_no_embedded_experiment_catalog():
    text = (TRAINING / "build_security_data.py").read_text()
    external = (TRAINING / "build_security_external.py").read_text()
    assert (
        "TRAIN_DEV_CASES = [" not in text.replace(" ", "")
        or "authored-example" in text
    )
    assert "hackdac:" not in external and "Hack@DAC" not in external
    assert "unspecified in supplied source excerpts" not in external


def test_held_out_splits_copy_byte_for_byte_or_fail(tmp_path, monkeypatch):
    from siliconshield.training.laya import augment_security_policies as augment

    base, output = tmp_path / "base", tmp_path / "output"
    base.mkdir()
    output.mkdir()
    (base / "train.jsonl").write_text('{"n": 2}\n')
    (base / "dev.jsonl").write_text('{"n": 1}\n')
    (base / "test.jsonl").write_text('{"n": 3}\n')
    held_out = augment.copy_held_out_splits(base, output)
    assert set(held_out) == {"dev", "test"}
    for split in ("dev", "test"):
        assert (output / (split + ".jsonl")).read_bytes() == (
            base / (split + ".jsonl")
        ).read_bytes()
        assert (
            held_out[split]["base_sha256"] == held_out[split]["output_sha256"]
        )
    # Missing splits are skipped, not fabricated.
    assert "calibration" not in held_out and "external" not in held_out
    # A divergent copy is rejected, never silently kept.
    (tmp_path / "output2").mkdir()
    monkeypatch.setattr(
        shutil,
        "copyfile",
        lambda source, target: Path(target).write_text("tampered\n"),
    )
    with pytest.raises(ValueError, match="held-out split dev changed"):
        augment.copy_held_out_splits(base, tmp_path / "output2")


SUPPORTED_ENTRY_POINTS = (
    "siliconshield.training.laya.augment_security_policies",
    "siliconshield.training.laya.augment_security_guards",
    "siliconshield.training.laya.expand_security_data",
    "siliconshield.training.laya.build_security_external",
)


def test_supported_entry_points_run_by_module():
    for dotted in SUPPORTED_ENTRY_POINTS:
        run = subprocess.run(
            [sys.executable, "-m", dotted, "--help"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert "usage:" in run.stdout.lower()
