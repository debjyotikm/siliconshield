"""Study-to-execution regressions use synthetic files and never launch jobs."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from siliconshield.research_loop.execution import campaign_supervisor as sup
from siliconshield.research_loop.execution import study_contract
from siliconshield.research_loop.execution.security_jobs import command_for
from siliconshield.training.laya import dataset_versions
from siliconshield.training.laya import decision_study as ds
from tests import test_campaign_supervisor as campaign_fixture
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


def test_unaccepted_replacement_rows_refused(tree, tmp_path):
    spec = fixture.spec_for(tree)
    train = tmp_path / "unaccepted.jsonl"
    train.write_text(
        tree["train"].read_text().replace("q <= 0;", "q <= secret;")
    )
    pools = tmp_path / "unaccepted-pools.json"
    dataset_versions.assign_pools(
        train, {"main": {"rows": ["a1", "a2", "a3"], "weight": 1}}, pools
    )
    spec["dataset"]["splits"]["train"]["path"] = str(train)
    spec["dataset"]["pools"] = str(pools)
    with pytest.raises(ValueError, match="accepted split"):
        ds.freeze_study_manifest(spec)


@pytest.mark.parametrize(
    "part", ["contract", "runtime", "tokenizer", "foundation"]
)
def test_identity_drift_refused(tree, part):
    manifest = ds.freeze_study_manifest(fixture.spec_for(tree))
    path = {
        "contract": tree["contract"],
        "runtime": tree["runtime"] / "runtime.py",
        "tokenizer": tree["parent"] / "tokenizer/tokenizer.json",
        "foundation": tree["foundation"],
    }[part]
    path.write_text("changed")
    with pytest.raises(ValueError):
        ds.verify_study_manifest(manifest)


def test_new_model_requires_actual_foundation(tree):
    spec = fixture.spec_for(tree, mode="create_new")
    del spec["parent"]["foundation"]
    with pytest.raises(ValueError, match="foundation"):
        ds.freeze_study_manifest(spec)


def test_unbuilt_evaluation_is_explicit_and_never_has_an_invented_hash(tree):
    spec = fixture.spec_for(tree)
    spec["dataset"]["splits"]["test"] = {
        "status": "not_materialized",
        "reason": "Training-only infrastructure trial",
    }
    manifest = ds.freeze_study_manifest(spec)
    reserved = manifest["dataset"]["splits"]["test"]
    assert reserved["verified"] is False and "sha256" not in reserved
    spec["dataset"]["splits"]["test"]["sha256"] = "ab" * 32
    with pytest.raises(ValueError, match="cannot claim dataset hashes"):
        ds.freeze_study_manifest(spec)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "/another-parent"),
        ("data", "/another-dataset"),
        ("source", "/another-source"),
        ("laya_root", "/another-runtime"),
        ("image_digest", "sha256:" + "ff" * 32),
        ("gpus", [0, 1, 2]),
    ],
)
def test_effective_job_drift_refused(tree, field, value):
    _, manifest, profile = bound_study(tree)
    assert (
        study_contract.verify_job(profile, "demo-r1", {})["sha256"]
        == manifest["manifest_sha256"]
    )
    profile[field] = value
    with pytest.raises((ValueError, OSError)):
        study_contract.verify_job(profile, "demo-r1", {})


def test_overrides_cannot_bypass_study(tree):
    _, _, profile = bound_study(tree)
    with pytest.raises(ValueError, match="settings"):
        study_contract.verify_job(profile, "demo-r1", {"adapter_lr": 9e-2})


def test_contract_argv_cannot_be_replaced(tree):
    spec, _, profile = bound_study(tree)
    spec["launch_command"] = ["unreviewed-script"]
    manifest = ds.freeze_study_manifest(spec)
    Path(profile["decision_study_manifest"]).write_text(json.dumps(manifest))
    profile["decision_study_sha256"] = manifest["manifest_sha256"]
    with pytest.raises(ValueError, match="Launch command"):
        study_contract.verify_job(profile, "demo-r1", {})


def test_qualification_is_bound_and_rechecks_artifacts(tree, tmp_path):
    _, manifest, _ = bound_study(tree)
    probe, resources = receipts(manifest, tmp_path)
    study_contract.verify_receipts(manifest, probe, resources)
    bad = dict(probe, trainable_groups=["head"])
    with pytest.raises(ValueError, match="trainable groups"):
        study_contract.verify_receipts(manifest, bad, resources)
    bad = dict(probe, study_sha256="ff" * 32)
    with pytest.raises(ValueError, match="another study"):
        study_contract.verify_receipts(manifest, bad, resources)
    (tmp_path / "measurement.json").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        study_contract.verify_receipts(manifest, probe, resources)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
def test_nonfinite_probe_and_resources_refused(bad):
    with pytest.raises(ValueError):
        ds.qualify_fitting_probe(
            {
                "trainable_groups": ["adapter"],
                "movement": {"adapter": bad},
                "train_fit": {
                    "correct_before": 0,
                    "correct_after": 1,
                    "rows": 3,
                },
            }
        )
    with pytest.raises(ValueError):
        ds.check_resources(
            {"gpus": 1, "memory_gb": bad, "hours": 1},
            {"gpus": 1, "memory_gb": 10, "hours": 1},
        )


def test_budget_reservation_is_idempotent_and_retry_bounded(tree):
    spec, _, profile = bound_study(tree)
    spec["budget"]["max_hours"] = 0.03
    manifest = ds.freeze_study_manifest(spec)
    Path(profile["decision_study_manifest"]).write_text(json.dumps(manifest))
    profile["decision_study_sha256"] = manifest["manifest_sha256"]
    job = {
        "idempotency_key": "identity",
        "attempts": 0,
        "timeout_seconds": 60,
        "experiment_id": "demo-r1",
    }
    study_contract.reserve_budget(profile, job)
    study_contract.reserve_budget(profile, job)
    job["attempts"] = 1
    with pytest.raises(ValueError, match="time budget"):
        study_contract.reserve_budget(profile, job)
    job.update(attempts=0, experiment_id="demo-r2", idempotency_key="other")
    with pytest.raises(ValueError, match="round budget"):
        study_contract.reserve_budget(profile, job)


def test_contradictory_prediction_counts_refused(tree):
    rows = [json.loads(line) for line in tree["dev"].read_text().splitlines()]
    report = fixture.dev_report([1, 2, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3)
    with pytest.raises(ValueError, match="confusion disagrees"):
        ds.report_counts(report, rows)


def test_missing_family_and_unknown_parity_block_acceptance(tree):
    manifest = ds.freeze_study_manifest(fixture.spec_for(tree))
    manifest["evaluation"]["objectives"]["families"] = {
        "target": {"f": {"max_false_safe": 0}}
    }
    good = fixture.counts(9, 9, [3, 3, 3], [3, 3, 3], 0, 3)
    args = (
        manifest,
        {"fit": {"correct": 3, "rows": 3, "meets_probe_criterion": True}},
        {"target": good, "keep": good},
        fixture.full_exposure_record(),
        {"complete": True, "report_present": True},
    )
    with pytest.raises(ValueError, match="families"):
        ds.assess_attempt(*args)
    good["families"] = {"f": dict(good, false_safe=1)}
    evidence = ds.assess_attempt(*args)
    assert evidence["target_met"] is False
    assert ds.diagnose(evidence)["category"] == "execution_defect"


def test_recommendation_from_analysis_and_plan_stays_parked(tree, tmp_path):
    home = tmp_path / "home"
    fixture.write_campaign(home, tree["parent"])
    fixture.write_run_records(home, fixture.full_exposure_record())
    manifest = ds.freeze_study_manifest(fixture.spec_for(tree))
    fixture.write_study(home, manifest)
    action = {
        "type": "recommend_new_model",
        "study_id": "demo",
        "domain": "new",
        "dataset_plan": "independent rows",
        "eval_plan": "frozen views",
    }
    record = campaign_fixture.analysis_record(next_action=action)
    record["diagnosis"] = {
        "category": "poor_generalization",
        "reasons": ["measured gap"],
        "next": "new model",
        "evidence": {
            "job_complete": True,
            "exposure_complete": True,
            "sdk_parity_ok": True,
            "train_fit_ok": True,
        },
    }
    (home / "campaign/rounds/r1/analysis.json").write_text(json.dumps(record))
    handoff = sup.load_handoff(home)
    result = sup._analyze_event(
        home,
        {"repo": str(tmp_path), "study_id": "demo"},
        handoff,
        "r1",
        campaign_fixture.FakeAgent(),
    )
    assert result == "analyzed"
    sup.save_handoff(home, handoff)
    assert sup.load_handoff(home)["mode"] == "parked"
    handoff = sup.default_handoff()

    def propose(prompt):
        (home / "campaign/supervisor/plan-proposal.json").write_text(
            json.dumps(action)
        )

    result = sup._plan_event(
        home,
        {"repo": str(tmp_path), "study_id": "demo"},
        handoff,
        campaign_fixture.FakeAgent().on("plan", propose),
    )
    assert result == "planned_recommendation"
    sup.save_handoff(home, handoff)
    assert sup.load_handoff(home)["mode"] == "parked"
    record["diagnosis"]["category"] = "execution_defect"
    (home / "campaign/rounds/r1/analysis.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="qualified attempt"):
        sup.recommend_new_model(home, "demo", action)


def test_shared_gpu_uses_process_identity_and_memory(tmp_path):
    from siliconshield.research_loop.execution.security_jobs import (
        process_identity,
        qualified_coexistence,
    )

    proc = tmp_path / "123"
    proc.mkdir()
    (proc / "stat").write_text(
        "123 (resident server) " + " ".join(["S"] + ["0"] * 18 + ["42"])
    )
    (proc / "cmdline").write_bytes(b"resident\0serve")
    (tmp_path / "meminfo").write_text("MemAvailable: 2048 kB\n")
    sharing = {
        "processes": {"123": process_identity("123", tmp_path)},
        "min_available_bytes": 1024 * 1024,
    }
    assert qualified_coexistence(["123"], sharing, tmp_path)
    assert not qualified_coexistence(["456"], sharing, tmp_path)
    sharing["min_available_bytes"] = 4096 * 1024
    assert not qualified_coexistence(["123"], sharing, tmp_path)
    sharing["min_available_bytes"] = 1024
    (proc / "stat").write_text(
        "123 (resident server) " + " ".join(["S"] + ["0"] * 18 + ["43"])
    )
    assert not qualified_coexistence(["123"], sharing, tmp_path)


def test_multiview_assessment_recomputes_selected_predictions(tree, tmp_path):
    rows = []
    for view in ("target", "keep"):
        for index, label in enumerate(ds.LABELS):
            row = fixture.make_row(view + str(index), label=label, pair=view)
            row["mechanism_id"] = "counter"
            rows.append(row)
    tree["dev"].write_text("".join(json.dumps(r) + "\n" for r in rows))
    tree["store"] = tmp_path / "fresh-store"
    fixture.accept_fixture(
        tree["store"], tree["train"], tree["dev"], tree["ledger"]
    )
    spec = fixture.spec_for(tree)
    baseline = fixture.counts(3, 3, [1, 1, 1], [1, 1, 1], 0, 1)
    limits = {"rows": 3, "min_correct": 2, "max_false_safe": 0}
    evaluation = spec["evaluation"]
    evaluation["views"] = {
        view: {"row_ids": [r["id"] for r in rows if r["pair_id"] == view]}
        for view in ("target", "keep")
    }
    evaluation["objectives"] = {
        "baseline": {"target": baseline, "keep": baseline},
        "targets": {"target": limits},
        "retention": {"keep": limits},
        "families": {"target": {"counter": {"max_false_safe": 0}}},
    }
    evaluation["selection_objective"].update(
        protected={"target": {}, "keep": limits},
        targets=limits,
    )
    manifest = ds.freeze_study_manifest(spec)
    per_view = fixture.dev_report(
        [0, 1, 2], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3
    )
    aggregate = dict(
        per_view,
        n=6,
        predictions=[0, 1, 2, 0, 1, 2],
        confusion=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        complete_pairs=2,
        development_views={"target": per_view, "keep": per_view},
    )
    report = dict(per_view, development=aggregate)
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps({"delivered_exposure": fixture.full_exposure_record()})
        + "\n"
    )
    evidence = ds.assess_round(
        manifest,
        report,
        events,
        {"correct": 3, "rows": 3, "meets_probe_criterion": True},
        sdk_parity_ok=True,
        expected_run_id="demo-r1",
        checkpoint_sha256="ab" * 32,
    )
    assert evidence["target_met"] and evidence["retention_ok"]
    aggregate["development_views"]["keep"] = dict(
        per_view, predictions=[1, 1, 2]
    )
    with pytest.raises(ValueError, match="View predictions disagree"):
        ds.assess_round(
            manifest,
            report,
            events,
            {},
            sdk_parity_ok=True,
            expected_run_id="demo-r1",
            checkpoint_sha256="ab" * 32,
        )


def test_unequal_trainer_and_study_gates_refused(tree):
    spec = fixture.spec_for(tree)
    spec["evaluation"]["selection_objective"]["protected"]["keep"][
        "max_false_safe"
    ] = 3
    with pytest.raises(ValueError, match="protection differs"):
        ds.freeze_study_manifest(spec)


def test_missing_trainer_view_fields_refused_before_launch(tree):
    _, _, profile = bound_study(tree)
    path = Path(profile["defaults"]["development_views"])
    config = json.loads(path.read_text())
    del config["selection_view"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="selection_view"):
        study_contract.verify_job(profile, "demo-r1", {})
