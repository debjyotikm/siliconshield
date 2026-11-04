"""Study-to-execution regressions use synthetic files and never launch jobs."""

import hashlib
import json
import sys

import pytest

from siliconshield.research_loop.execution import study_contract
from siliconshield.research_loop.execution.security_jobs import command_for
from siliconshield.training.laya import decision_study as ds
from tests import test_decision_study as fixture


@pytest.fixture
def tree(tmp_path):
    return fixture.study_tree.__wrapped__(tmp_path)


def bound_study(tree):
    spec = fixture.single_view_spec(tree)
    base = tree["code"].parent
    views = base / "views.json"
    views.write_text(
        json.dumps(
            {
                "views": spec["evaluation"]["views"],
                "selection_view": "solo",
                "single_view_study": True,
                "score_triplets": True,
                "checkpoint_objective": spec["evaluation"][
                    "selection_objective"
                ],
            }
        )
    )
    spec["training"]["settings"]["development_views"] = str(views)
    spec["training"]["recipe"] = {
        "training": dict(spec["training"]["settings"])
    }
    recipe = base / "recipe.json"
    recipe.write_text(json.dumps(spec["training"]["recipe"]))
    profile = {
        "model": str(tree["parent"]),
        "data": str(base),
        "source": str(tree["code"]),
        "laya_root": str(tree["runtime"]),
        "python": sys.executable,
        "runs": str(base / "runs"),
        "gpus": [0],
        "recipe": str(recipe),
        "defaults": dict(spec["training"]["settings"]),
        "timeout_seconds": 60,
        "image_digest": spec["code"]["image_digest"],
    }
    command, output = command_for(profile, "demo-r1", {})
    spec["launch_command"] = [
        "{experiment_id}"
        if part == "demo-r1"
        else "{output}"
        if part == str(output)
        else part
        for part in command
    ]
    manifest = ds.freeze_study_manifest(spec)
    path = base / "study.json"
    path.write_text(json.dumps(manifest))
    profile.update(
        decision_study_manifest=str(path),
        decision_study_sha256=manifest["manifest_sha256"],
    )
    return spec, manifest, profile


def receipts(manifest, base):
    evidence = base / "measurement.json"
    evidence.write_text('{"measurement": "synthetic"}')
    common = {
        "study_sha256": manifest["manifest_sha256"],
        "parent_sha256": manifest["parent"]["weights_sha256"],
        "train_sha256": manifest["dataset"]["splits"]["train"]["sha256"],
        "artifacts": {
            str(evidence): hashlib.sha256(evidence.read_bytes()).hexdigest()
        },
    }
    probe = {
        **common,
        "trainable_groups": manifest["training"]["trainable_groups"],
        "movement": {"encoder": 0, "adapter": 0.1, "head": 0.2},
        "train_fit": {"correct_before": 1, "correct_after": 2, "rows": 3},
    }
    resources = {
        **common,
        "requirement": {"gpus": 1, "memory_gb": 2, "hours": 0.1},
        "available": {"gpus": 1, "memory_gb": 4, "hours": 1},
    }
    return probe, resources


def test_overrides_cannot_bypass_study(tree):
    _, _, profile = bound_study(tree)
    with pytest.raises(ValueError, match="settings"):
        study_contract.verify_job(profile, "demo-r1", {"adapter_lr": 9e-2})


def test_study_contract_rejects_malformed_shapes(tree):
    spec, manifest, profile = bound_study(tree)
    probe, resources = receipts(manifest, tree["code"].parent)
    with pytest.raises(ValueError, match="must be an object"):
        study_contract.load(["x"])
    with pytest.raises(ValueError, match="manifest path"):
        study_contract.load({"decision_study_manifest": 5})
    with pytest.raises(ValueError, match="overrides must be an object"):
        study_contract.verify_job(profile, "e", ["x"])
    with pytest.raises(ValueError, match="must be an object"):
        study_contract.verify_job(["x"], "e", {})
    bad = {k: v for k, v in profile.items() if k != "model"}
    with pytest.raises(ValueError, match="needs model"):
        study_contract.verify_job(bad, "e", {})
    with pytest.raises(ValueError, match="must be objects"):
        study_contract.verify_receipts(manifest, ["x"], resources)
    bad_resources = {k: v for k, v in resources.items() if k != "requirement"}
    with pytest.raises(ValueError, match="measured requirements"):
        study_contract.verify_receipts(manifest, probe, bad_resources)
    job = {
        "idempotency_key": "k",
        "attempts": 0,
        "timeout_seconds": 60,
        "experiment_id": "r1",
    }
    with pytest.raises(ValueError, match="must be an object"):
        study_contract.reserve_budget(profile, ["x"])
    bad_job = {k: v for k, v in job.items() if k != "attempts"}
    with pytest.raises(ValueError, match="attempt count"):
        study_contract.reserve_budget(profile, bad_job)
    base = tree["code"].parent
    (base / "budget.json").write_text("[1, 2]")
    with pytest.raises(ValueError, match="unreadable"):
        study_contract.reserve_budget(profile, dict(job, idempotency_key="k"))
