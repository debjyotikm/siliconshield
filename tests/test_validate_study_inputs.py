"""Study-input validator tests: synthetic fixtures only."""

import copy
import json

import pytest

torch = pytest.importorskip("torch")
laya = pytest.importorskip("laya")

from siliconshield.training.laya import security_train as train
from siliconshield.training.laya import validate_study_inputs as validator
from siliconshield.training.laya.question_contracts import (
    LEGACY_CONTRACT,
    question_for,
)

QUESTIONS = question_for(LEGACY_CONTRACT)


def toy_row(split, label, index, source="toy", design=None):
    """Author one synthetic decision row with unique inputs."""
    design = design or split + "-design"
    state = (
        "REQUIREMENT\n%s-decision-%d\nASSUMPTIONS\nnone\n"
        "RTL\nmodule topmodule; wire value_%d; endmodule"
        % (split, index, index)
    )
    return {
        "id": "%s-%d" % (split, index),
        "family_id": split + "-family",
        "split_group_id": "%s-group-%d" % (split, index // 3),
        "split": split,
        "base_design_id": design,
        "mechanism_id": "toy-mechanism",
        "pair_id": "%s-%d" % (split, index // 3),
        "state": state,
        "questions": copy.deepcopy(QUESTIONS),
        "expected": {"security": label},
        "validation": {"accepted": True},
        "source": {"name": source},
    }


def write_study(path, triplets=1, source="toy"):
    """Write a tiny balanced train/dev study dataset."""
    for split in ("train", "dev"):
        rows = []
        for triplet in range(triplets):
            base = triplet * 3
            for offset, label in enumerate(train.LABELS):
                rows.append(
                    toy_row(
                        split,
                        label,
                        base + offset,
                        source=source,
                        design="%s-design-%d" % (split, triplet),
                    )
                )
        (path / (split + ".jsonl")).write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    return path


def write_pools(path):
    """Pool every train row into one flat pool."""
    rows = [
        json.loads(line)
        for line in (path / "train.jsonl").read_text().splitlines()
    ]
    manifest = {
        "pools": {
            "toy": {
                "rows": [row["id"] for row in rows],
                "weight": 1.0,
                "mode": "flat",
            }
        }
    }
    pools = path / "pools.json"
    pools.write_text(json.dumps(manifest))
    return pools


def write_recipe(path, source="toy"):
    """Declare study source mass in a recipe file."""
    recipe = path / "recipe.json"
    recipe.write_text(json.dumps({"source_mass": {source: 1.0}}))
    return recipe


def test_validator_passes_toy_study_and_writes_receipt(tmp_path):
    data = write_study(tmp_path)
    pools = write_pools(tmp_path)
    recipe = write_recipe(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    assert (
        validator.main(
            [
                "--data",
                str(data),
                "--pools",
                str(pools),
                "--recipe",
                str(recipe),
                "--seed",
                "7",
                "--global-batch",
                "3",
                "--max-updates",
                "9",
                "--output-receipt",
                str(receipt_path),
            ]
        )
        == 0
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["result"] == "pass"
    assert receipt["counts"] == {"train": 3, "dev": 3}
    assert receipt["trained_rows"] == 3
    assert receipt["coverage"]["updates_to_cover"] == 1
    assert receipt["budget_covers_data"] is True
    assert receipt["token_screen"]["status"] == "skipped"


def test_validator_rejects_undeclared_source_mass(tmp_path):
    data = write_study(tmp_path)
    pools = write_pools(tmp_path)
    before = dict(train.SOURCE_MASS)
    with pytest.raises(ValueError, match="no declared source mass"):
        validator.validate(
            str(data),
            pools=str(pools),
            seed=7,
            global_batch=3,
        )
    assert train.SOURCE_MASS == before


def test_validator_rejects_incomplete_triplet(tmp_path):
    data = write_study(tmp_path)
    lines = (data / "train.jsonl").read_text().splitlines()
    (data / "train.jsonl").write_text(lines[0] + "\n" + lines[1] + "\n")
    with pytest.raises(ValueError, match="incomplete decision triplet"):
        validator.validate(
            str(data),
            recipe=str(write_recipe(tmp_path)),
            seed=7,
            global_batch=3,
        )


def test_validator_rejects_pool_gap(tmp_path):
    data = write_study(tmp_path)
    pools = write_pools(tmp_path)
    manifest = json.loads(pools.read_text())
    manifest["pools"]["toy"]["rows"] = manifest["pools"]["toy"]["rows"][:2]
    pools.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Row outside every pool"):
        validator.validate(
            str(data),
            pools=str(pools),
            recipe=str(write_recipe(tmp_path)),
            seed=7,
            global_batch=3,
        )


def test_validator_subset_selection_keeps_designs_whole(tmp_path):
    data = write_study(tmp_path, triplets=2)
    pools = write_pools(tmp_path)
    receipt = validator.validate(
        str(data),
        pools=str(pools),
        recipe=str(write_recipe(tmp_path)),
        seed=7,
        global_batch=3,
        limit_train_rows=3,
    )
    assert receipt["trained_rows"] == 3
    assert receipt["coverage"]["triplets"] == 1
