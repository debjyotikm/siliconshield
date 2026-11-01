"""Frozen decision-study manifests, attempt diagnosis, and handoff."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

from siliconshield.research_loop.episodes.contracts import digest
from siliconshield.research_loop.execution import campaign_supervisor as sup
from siliconshield.schemas import (
    CAMPAIGN_ROUND_SCHEMA,
    DATASET_AUDIT_SCHEMA,
    LAYA_CAMPAIGN_SCHEMA,
    PROCEDURE_BUNDLE_SCHEMA,
)
from siliconshield.training.laya import dataset_versions, decision_study
from siliconshield.training.laya.question_contracts import write_export_record


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def make_row(row_id, template="t-demo", label=None, pair=None):
    req = "Keep the demo register at reset while idle."
    asm = "Reset is active low. Idle means no request."
    state = "REQUIREMENT\n%s\nASSUMPTIONS\n%s\nRTL\nq <= 0;" % (req, asm)
    row = {
        "id": row_id,
        "state": state,
        "template_id": template,
        "assertion_sha256": sha_text("assert-demo"),
    }
    if label is not None:
        row["expected"] = {"security": label}
    if pair is not None:
        row["pair_id"] = pair
    return row


def write_rows(path, ids):
    Path(path).write_text(
        "\n".join(json.dumps(make_row(i)) for i in ids) + "\n"
    )


def write_ledger(path):
    row = make_row("probe")
    req = row["state"].split("\nASSUMPTIONS")[0].split("REQUIREMENT\n")[1]
    asm = row["state"].split("\nASSUMPTIONS\n")[1].split("\nRTL")[0]
    record = {
        "template_id": "t-demo",
        "requirement_sha256": sha_text(req),
        "assertion_sha256": sha_text("assert-demo"),
        "assumptions_sha256": sha_text(asm),
        "reviewer_type": "human",
        "interpretation": "hold-at-reset property",
        "verdict": "agree",
    }
    Path(path).write_text(json.dumps(record) + "\n")
    return record


def accept_fixture(root, train, dev, ledger):
    manifest = dataset_versions.write_version(
        root, "v1", {"train": str(train), "dev": str(dev)}
    )
    audit = {
        "schema": DATASET_AUDIT_SCHEMA,
        "splits": {
            role: entry["sha256"] for role, entry in manifest["splits"].items()
        },
        "cross_split_exact_rtl_overlap": 0,
        "retired_rtl_overlap": 0,
        "truncated": 0,
        "lineage_overlap": 0,
        "canonical_overlap": 0,
        "max_group_share": 0.01,
        "max_mechanism_share": 0.05,
        "release_class": "narrow-pilot",
        "review_ledger_sha256": hashlib.sha256(
            Path(ledger).read_bytes()
        ).hexdigest(),
        "reviewed_templates": 1,
    }
    dataset_versions.accept_version(root, "v1", audit)


def counts(rows, correct, classes, support, false_safe, triplets):
    return {
        "rows": rows,
        "correct": correct,
        "class_correct": classes,
        "class_support": support,
        "false_safe": false_safe,
        "false_violation": 0,
        "unnecessary_abstention": 0,
        "triplets_correct": triplets,
    }


@pytest.fixture
def study_tree(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    header = json.dumps(
        {
            "encoder.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        }
    ).encode()
    (parent / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * 4
    )
    write_export_record(parent, "siliconshield-requirement-context-v1")
    (parent / "tokenizer").mkdir()
    (parent / "tokenizer" / "tokenizer.json").write_text("{}")
    (parent / "config.json").write_text("{}")
    code = tmp_path / "code"
    code.mkdir()
    (code / "trainer.py").write_text("train")
    contract_root = tmp_path / "contract"
    contract_root.mkdir()
    write_export_record(contract_root, "siliconshield-requirement-context-v1")
    contract = contract_root / "question_contract.json"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "runtime.py").write_text("fixture")
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    write_rows(train, ["a1", "a2", "a3"])
    dev.write_text(
        "\n".join(
            json.dumps(
                make_row(
                    "d-%s" % variant,
                    label=label,
                    pair="pd1",
                )
            )
            for variant, label in (
                ("sat", "satisfies_requirement"),
                ("vio", "violates_requirement"),
                ("ctx", "insufficient_context"),
            )
        )
        + "\n"
    )
    ledger = tmp_path / "ledger.jsonl"
    write_ledger(ledger)
    store = tmp_path / "datasets"
    accept_fixture(store, train, dev, ledger)
    pools = tmp_path / "pools.json"
    dataset_versions.assign_pools(
        train, {"main": {"rows": ["a1", "a2", "a3"], "weight": 1}}, pools
    )
    weights = hashlib.sha256(
        (parent / "model.safetensors").read_bytes()
    ).hexdigest()
    foundation = tmp_path / "foundation.json"
    foundation.write_text(
        json.dumps(
            {
                "checkpoint": "laya-fixture",
                "revision": "r1",
                "files": decision_study._hash_tree(parent),
            }
        )
    )
    return {
        "parent": parent,
        "runtime": runtime,
        "foundation": foundation,
        "code": code,
        "contract": contract,
        "train": train,
        "dev": dev,
        "ledger": ledger,
        "store": store,
        "pools": pools,
        "weights": weights,
    }


def spec_for(tree, mode="improve_existing"):
    baseline = {
        "target": counts(9, 5, [2, 2, 1], [3, 3, 3], 1, 1),
        "keep": counts(9, 8, [3, 3, 2], [3, 3, 3], 0, 2),
    }
    spec = {
        "study_id": "demo",
        "mode": mode,
        "parent": {
            "path": str(tree["parent"]),
            "weights_sha256": tree["weights"],
            "adapters_present": False,
            "question_contract": "siliconshield-requirement-context-v1",
            "foundation": {
                "reference": str(tree["foundation"]),
                "sha256": hashlib.sha256(
                    tree["foundation"].read_bytes()
                ).hexdigest(),
            },
        },
        "dataset": {
            "root": str(tree["store"]),
            "version": "v1",
            "splits": {
                "train": {"path": str(tree["train"])},
                "dev": {"path": str(tree["dev"])},
                "qualification": {"sha256": "ab" * 32},
                "test": {"sha256": "cd" * 32},
            },
            "pools": str(tree["pools"]),
            "review_ledger": str(tree["ledger"]),
        },
        "code": {
            "path": str(tree["code"]),
            "runtime": {
                "name": "laya",
                "revision": "r1",
                "path": str(tree["runtime"]),
            },
            "python": sys.executable,
            "image_digest": "sha256:" + "ab" * 32,
            "tokenizer_path": str(tree["parent"] / "tokenizer"),
            "tokenizer_sha256": "ef" * 32,
            "question_contract": "siliconshield-requirement-context-v1",
            "question_contract_path": str(tree["contract"]),
        },
        "training": {
            "trainables": {"mode": "adapters_only"},
            "learning_rates": {"adapter": 1e-4, "head": 1e-4},
            "schedule": "cosine",
            "seed": 7,
            "batch": {
                "micro": 2,
                "accumulation": 2,
                "devices": 1,
                "global": 4,
            },
            "max_updates": 10,
            "eval_every_updates": 5,
            "hardware": {"devices": "1xCPU", "precision": "fp32"},
            "fitting_probe": "probe-receipt",
            "resource_check": "resource-receipt",
        },
        "evaluation": {
            "views": {"target": ["dev"], "keep": ["dev"]},
            "selection_objective": {"version": "protected-views-v2-triplets"},
            "selection_policy": {"research": {"floor": 1}},
            "objectives": {
                "baseline": baseline,
                "targets": {
                    "target": {
                        "rows": 9,
                        "min_correct": 7,
                        "max_false_safe": 0,
                        "min_class_correct": [2, 2, 2],
                    }
                },
                "retention": {
                    "keep": {
                        "rows": 9,
                        "min_correct": 8,
                        "max_false_safe": 0,
                    }
                },
            },
        },
        "budget": {"max_rounds": 1, "max_hours": 2.0},
        "launch_command": ["python", "train.py"],
    }
    settings = {
        "adapter_lr": 1e-4,
        "head_lr": 1e-4,
        "micro_batch": 2,
        "accumulation": 2,
        "seed": 7,
        "lr_schedule": "cosine",
        "max_updates": 10,
        "eval_every_updates": 5,
        "freeze_encoder": True,
        "question_contract": "siliconshield-requirement-context-v1",
        "plateau_requires_full_exposure": True,
        "sampling": "grouped",
        "pools": str(tree["pools"]),
        "review_ledger": str(tree["ledger"]),
    }
    spec["evaluation"]["selection_objective"].update(
        protected={
            "target": {},
            "keep": dict(spec["evaluation"]["objectives"]["retention"]["keep"]),
        },
        view="target",
        targets=dict(spec["evaluation"]["objectives"]["targets"]["target"]),
    )
    spec["training"]["settings"] = settings
    spec["training"]["recipe"] = {"training": dict(settings)}
    return spec


def test_freeze_and_verify_round_trip(study_tree):
    manifest = decision_study.freeze_study_manifest(spec_for(study_tree))
    assert manifest["mode"] == "improve_existing"
    assert manifest["dataset"]["splits"]["qualification"] == {
        "sha256": "ab" * 32,
        "verified": False,
    }
    assert decision_study.verify_study_manifest(manifest) == {
        "study_id": "demo",
        "verified": True,
    }


def test_create_new_modes(study_tree):
    manifest = decision_study.freeze_study_manifest(
        spec_for(study_tree, mode="create_new")
    )
    assert manifest["mode"] == "create_new"
    bad = spec_for(study_tree, mode="create_new")
    bad["parent"]["adapters_present"] = True
    with pytest.raises(ValueError, match="adapter"):
        decision_study.freeze_study_manifest(bad)


def test_verify_detects_parent_drift(study_tree):
    manifest = decision_study.freeze_study_manifest(spec_for(study_tree))
    (study_tree["parent"] / "config.json").write_text('{"x": 1}')
    with pytest.raises(ValueError, match="Parent export"):
        decision_study.verify_study_manifest(manifest)


def test_freeze_refuses_unaccepted_version(study_tree, tmp_path):
    raw = tmp_path / "raw"
    manifest = dataset_versions.write_version(
        raw,
        "v9",
        {
            "train": str(study_tree["train"]),
            "dev": str(study_tree["dev"]),
        },
    )
    assert manifest["version"] == "v9"
    spec = spec_for(study_tree)
    spec["dataset"]["root"] = str(raw)
    spec["dataset"]["version"] = "v9"
    with pytest.raises(ValueError, match="not accepted"):
        decision_study.freeze_study_manifest(spec)


def test_freeze_refuses_review_gap(study_tree, tmp_path):
    train = tmp_path / "gap-train.jsonl"
    write_rows(train, ["g1"])
    lines = train.read_text().splitlines()
    row = json.loads(lines[0])
    row["template_id"] = "t-unreviewed"
    train.write_text(json.dumps(row) + "\n")
    pools = tmp_path / "gap-pools.json"
    dataset_versions.assign_pools(
        train, {"main": {"rows": ["g1"], "weight": 1}}, pools
    )
    spec = spec_for(study_tree)
    spec["dataset"]["splits"]["train"]["path"] = str(train)
    spec["dataset"]["pools"] = str(pools)
    with pytest.raises(ValueError, match="accepted split"):
        decision_study.freeze_study_manifest(spec)


def test_freeze_refuses_bad_batch(study_tree):
    spec = spec_for(study_tree)
    spec["training"]["batch"]["global"] = 5
    with pytest.raises(ValueError, match="Batch geometry"):
        decision_study.freeze_study_manifest(spec)


def test_probe_and_resources():
    passed = decision_study.qualify_fitting_probe(
        {
            "trainable_groups": ["adapter", "head"],
            "movement": {"adapter": 0.5, "head": 0.2, "encoder": 0},
            "train_fit": {"correct_before": 3, "correct_after": 5, "rows": 9},
        }
    )
    assert passed == {
        "qualified": True,
        "trainable_groups": ["adapter", "head"],
        "gain": 2,
    }
    with pytest.raises(ValueError, match="frozen group"):
        decision_study.qualify_fitting_probe(
            {
                "trainable_groups": ["adapter"],
                "movement": {"adapter": 0.5, "encoder": 0.1},
                "train_fit": {
                    "correct_before": 3,
                    "correct_after": 5,
                    "rows": 9,
                },
            }
        )
    with pytest.raises(ValueError, match="did not improve"):
        decision_study.qualify_fitting_probe(
            {
                "trainable_groups": ["adapter"],
                "movement": {"adapter": 0.5},
                "train_fit": {
                    "correct_before": 5,
                    "correct_after": 5,
                    "rows": 9,
                },
            }
        )
    assert decision_study.check_resources(
        {"gpus": 1, "memory_gb": 8, "hours": 1},
        {"gpus": 2, "memory_gb": 16, "hours": 4},
    ) == {"fits": True}
    with pytest.raises(ValueError, match="only 16 is available"):
        decision_study.check_resources(
            {"gpus": 1, "memory_gb": 32, "hours": 1},
            {"gpus": 2, "memory_gb": 16, "hours": 4},
        )


def full_exposure(rows=3):
    return {
        "unique_rows": rows,
        "total_rows": rows,
        "broken_triplets": [],
        "split_triplets": [],
    }


def test_assess_and_diagnose_taxonomy(study_tree):
    manifest = decision_study.freeze_study_manifest(spec_for(study_tree))
    good_fit = {"correct": 3, "rows": 3, "meets_probe_criterion": True}
    bad_fit = {"correct": 1, "rows": 3, "meets_probe_criterion": False}
    target_met = counts(9, 8, [3, 3, 2], [3, 3, 3], 0, 2)
    target_missed = counts(9, 5, [2, 2, 1], [3, 3, 3], 1, 1)
    keep_ok = counts(9, 9, [3, 3, 3], [3, 3, 3], 0, 3)
    keep_bad = counts(9, 6, [2, 2, 2], [3, 3, 3], 1, 1)
    job_ok = {"complete": True, "report_present": True, "sdk_parity_ok": True}
    job_dead = {"complete": False, "report_present": False}

    def assess(fit, dev, job, exposure):
        return decision_study.assess_attempt(
            manifest, {"fit": fit}, dev, exposure, job
        )

    assert (
        decision_study.diagnose(
            assess(
                good_fit,
                {"target": target_met, "keep": keep_ok},
                job_dead,
                full_exposure(),
            )
        )["category"]
        == "execution_defect"
    )
    thin = dict(full_exposure(), unique_rows=1)
    assert (
        decision_study.diagnose(
            assess(
                good_fit, {"target": target_met, "keep": keep_ok}, job_ok, thin
            )
        )["category"]
        == "insufficient_exposure"
    )
    assert (
        decision_study.diagnose(
            assess(
                bad_fit,
                {"target": target_missed, "keep": keep_bad},
                job_ok,
                full_exposure(),
            )
        )["category"]
        == "failure_to_fit"
    )
    assert (
        decision_study.diagnose(
            assess(
                good_fit,
                {"target": target_met, "keep": keep_bad},
                job_ok,
                full_exposure(),
            )
        )["category"]
        == "retention_tradeoff"
    )
    assert (
        decision_study.diagnose(
            assess(
                good_fit,
                {"target": target_missed, "keep": keep_ok},
                job_ok,
                full_exposure(),
            )
        )["category"]
        == "poor_generalization"
    )
    labeled = assess(
        good_fit,
        {"target": target_missed, "keep": keep_ok},
        job_ok,
        full_exposure(),
    )
    assert (
        decision_study.diagnose({**labeled, "label_audit_ok": False})[
            "category"
        ]
        == "poor_labels"
    )
    with pytest.raises(ValueError, match="nothing to diagnose"):
        decision_study.diagnose(
            assess(
                good_fit,
                {"target": target_met, "keep": keep_ok},
                job_ok,
                full_exposure(),
            )
        )


def write_campaign(home, export, policy=None, with_round=True):
    campaign = home / "campaign"
    campaign.mkdir(parents=True, exist_ok=True)
    if with_round:
        (campaign / "rounds" / "r1").mkdir(parents=True)
        (campaign / "rounds" / "r1" / "round.json").write_text(
            json.dumps({"schema": CAMPAIGN_ROUND_SCHEMA, "phase": "done"})
        )
    (campaign / "incumbent.json").write_text(
        json.dumps({"export": str(export)})
    )
    profile = home / "profile.json"
    runs = home / "runs"
    runs.mkdir(exist_ok=True)
    profile.write_text(json.dumps({"runs": str(runs)}))
    (campaign / "campaign.json").write_text(
        json.dumps(
            {
                "schema": LAYA_CAMPAIGN_SCHEMA,
                "campaign_id": "demo",
                "selection": (
                    {"research": {"floor": 1}} if policy is None else policy
                ),
                "operator_profile": str(profile),
            }
        )
    )
    return campaign


def write_run_records(home, exposure):
    run = home / "runs" / "demo-r1"
    run.mkdir(parents=True, exist_ok=True)
    (run / "report.json").write_text(json.dumps({"run_id": "demo-r1"}))
    (run / "events.jsonl").write_text(
        json.dumps({"event": "development", "delivered_exposure": exposure})
        + "\n"
    )


def full_exposure_record():
    return {
        "unique_rows": 3,
        "total_rows": 3,
        "broken_triplets": [],
        "split_triplets": [],
    }


def write_study(home, manifest):
    path = home / "campaign" / "studies" / "demo" / "study.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))
    rounds = home / "campaign" / "rounds"
    for receipt in rounds.glob("*/round.json"):
        value = json.loads(receipt.read_text())
        value["study"] = {
            "study_id": manifest["study_id"],
            "sha256": manifest["manifest_sha256"],
        }
        receipt.write_text(json.dumps(value))
    return path


def test_recommend_new_model_success(study_tree, tmp_path):
    home = tmp_path / "home"
    write_campaign(home, study_tree["parent"])
    write_run_records(home, full_exposure_record())
    manifest = decision_study.freeze_study_manifest(spec_for(study_tree))
    write_study(home, manifest)
    (home / "campaign" / "rounds" / "r1" / "analysis.json").write_text(
        json.dumps(
            {
                "diagnosis": {
                    "category": "poor_generalization",
                    "evidence": {
                        "job_complete": True,
                        "exposure_complete": True,
                        "sdk_parity_ok": True,
                        "train_fit_ok": True,
                    },
                }
            }
        )
    )
    record = sup.recommend_new_model(
        home,
        "demo",
        {
            "domain": "temporal specialists",
            "dataset_plan": "independent fifo families",
            "eval_plan": "fresh group-disjoint dev",
        },
    )
    assert record["qualified_rounds"] == ["r1"]
    assert record["preserved_incumbent"] == str(study_tree["parent"])
    assert (
        home / "campaign" / "studies" / "demo" / "new-model-proposal.json"
    ).is_file()
    handoff = json.loads(
        (home / "campaign" / "supervisor" / "handoff.json").read_text()
    )
    assert handoff["mode"] == "parked"
    assert "awaits owner approval" in handoff["parked_reason"]


def test_recommend_refuses_unqualified_history(study_tree, tmp_path):
    home = tmp_path / "home"
    write_campaign(home, study_tree["parent"])
    manifest = decision_study.freeze_study_manifest(spec_for(study_tree))
    write_study(home, manifest)
    (home / "campaign" / "rounds" / "r1" / "analysis.json").write_text(
        json.dumps(
            {
                "diagnosis": {
                    "category": "execution_defect",
                    "evidence": {
                        "job_complete": False,
                        "exposure_complete": False,
                    },
                }
            }
        )
    )
    with pytest.raises(ValueError, match="No qualified attempt"):
        sup.recommend_new_model(
            home,
            "demo",
            {
                "domain": "x",
                "dataset_plan": "y",
                "eval_plan": "z",
            },
        )


def test_recommend_refuses_exposure_overclaim(study_tree, tmp_path):
    home = tmp_path / "home"
    write_campaign(home, study_tree["parent"])
    thin = dict(full_exposure_record(), unique_rows=1)
    thin["broken_triplets"] = ["p1"]
    write_run_records(home, thin)
    manifest = decision_study.freeze_study_manifest(spec_for(study_tree))
    write_study(home, manifest)
    (home / "campaign" / "rounds" / "r1" / "analysis.json").write_text(
        json.dumps(
            {
                "diagnosis": {
                    "category": "poor_generalization",
                    "evidence": {
                        "job_complete": True,
                        "exposure_complete": True,
                        "sdk_parity_ok": True,
                        "train_fit_ok": True,
                    },
                }
            }
        )
    )
    with pytest.raises(ValueError, match="draws deny"):
        sup.recommend_new_model(
            home,
            "demo",
            {"domain": "x", "dataset_plan": "y", "eval_plan": "z"},
        )


def test_recommend_refuses_open_budget(study_tree, tmp_path):
    home = tmp_path / "home"
    write_campaign(home, study_tree["parent"])
    spec = spec_for(study_tree)
    spec["budget"]["max_rounds"] = 3
    write_study(home, decision_study.freeze_study_manifest(spec))
    with pytest.raises(ValueError, match="not exhausted"):
        sup.recommend_new_model(
            home, "demo", {"domain": "x", "dataset_plan": "y", "eval_plan": "z"}
        )


def test_recommend_action_validation():
    action = sup.check_proposal_next_action(
        {
            "type": "recommend_new_model",
            "study_id": "demo",
            "domain": "temporal",
            "dataset_plan": "new families",
            "eval_plan": "fresh dev",
        }
    )
    assert action["type"] == "recommend_new_model"
    with pytest.raises(ValueError, match="eval_plan"):
        sup.check_proposal_next_action(
            {
                "type": "recommend_new_model",
                "study_id": "demo",
                "domain": "temporal",
                "dataset_plan": "new families",
            }
        )


def test_commit_prep_enforces_study_manifest(tmp_path):
    home = tmp_path / "home"
    (home / "campaign").mkdir(parents=True)
    handoff = {
        "pending_proposal": {
            "round_id": "r1",
            "hypothesis": "h",
            "selection": None,
        }
    }
    prep = {"round_id": "r1", "hypothesis": "h", "selection": None}
    with pytest.raises(sup.PrepCommitError, match="[Ss]tudy manifest"):
        sup.commit_prep(home, handoff, prep, {"study_id": "demo"})


def commit_harness(home):
    handoff = {
        "pending_proposal": {
            "round_id": "r1",
            "hypothesis": "h",
            "selection": None,
        }
    }
    prep = {"round_id": "r1", "hypothesis": "h", "selection": None}
    return handoff, prep


def test_commit_prep_refuses_selection_drift(study_tree, tmp_path):
    home = tmp_path / "home"
    write_campaign(
        home, study_tree["parent"], policy={"research": {}}, with_round=False
    )
    write_study(
        home, decision_study.freeze_study_manifest(spec_for(study_tree))
    )
    handoff, prep = commit_harness(home)
    with pytest.raises(sup.PrepCommitError, match="drifted"):
        sup.commit_prep(home, handoff, prep, {"study_id": "demo"})


def test_commit_prep_requires_first_commit_receipts(study_tree, tmp_path):
    home = tmp_path / "home"
    write_campaign(home, study_tree["parent"], with_round=False)
    write_study(
        home, decision_study.freeze_study_manifest(spec_for(study_tree))
    )
    handoff, prep = commit_harness(home)
    with pytest.raises(sup.PrepCommitError, match="fitting probe"):
        sup.commit_prep(home, handoff, prep, {"study_id": "demo"})
    base = home / "campaign" / "studies" / "demo"
    (base / "fitting-probe.json").write_text(
        json.dumps(
            {
                "trainable_groups": ["adapter"],
                "movement": {"adapter": 0.3},
                "train_fit": {
                    "correct_before": 1,
                    "correct_after": 2,
                    "rows": 3,
                },
            }
        )
    )
    (base / "resource-check.json").write_text(
        json.dumps(
            {
                "requirement": {"gpus": 1, "memory_gb": 8, "hours": 1},
                "available": {"gpus": 1, "memory_gb": 16, "hours": 2},
            }
        )
    )
    # Unbound receipts cannot qualify this study.
    prep["manifest_path"] = str(tmp_path / "missing.json")
    with pytest.raises(sup.PrepCommitError, match="another study"):
        sup.commit_prep(home, handoff, prep, {"study_id": "demo"})


def write_bundle(path, model="spark"):
    grade = {
        "execution_complete": True,
        "action_correct": True,
        "evidence_correct": True,
        "explanation_correct": True,
        "final_consistent": True,
        "verified": True,
    }
    body = {
        "schema": PROCEDURE_BUNDLE_SCHEMA,
        "scope": "development_procedure_replay",
        "identity": {"model": model},
        "guidance": "Prefer tiny diffs with receipts.",
        "baseline": {"t1": dict(grade)},
        "candidate": {"t1": dict(grade)},
    }
    path = Path(path)
    path.write_text(json.dumps({**body, "digest": digest(body)}, indent=1))
    return path


def test_procedure_block_optional_and_verified(study_tree, tmp_path):
    plain = decision_study.freeze_study_manifest(spec_for(study_tree))
    assert plain["procedure"] == {"enabled": False}
    bundle = write_bundle(tmp_path / "bundle.json")
    queue = tmp_path / "queue"
    queue.mkdir()
    spec = spec_for(study_tree)
    spec["procedure"] = {
        "model": "spark",
        "bundle": str(bundle),
        "queue": str(queue),
    }
    manifest = decision_study.freeze_study_manifest(spec)
    assert manifest["procedure"]["enabled"] is True
    assert manifest["procedure"]["model"] == "spark"
    assert decision_study.verify_study_manifest(manifest)["verified"] is True
    bundle.write_text(bundle.read_text().replace("tiny", "small"))
    with pytest.raises(ValueError, match="[Bb]undle"):
        decision_study.verify_study_manifest(manifest)


def test_procedure_block_refuses_wrong_model(study_tree, tmp_path):
    bundle = write_bundle(tmp_path / "bundle.json", model="other")
    queue = tmp_path / "queue"
    queue.mkdir()
    spec = spec_for(study_tree)
    spec["procedure"] = {
        "model": "spark",
        "bundle": str(bundle),
        "queue": str(queue),
    }
    with pytest.raises(ValueError, match="[Bb]undle"):
        decision_study.freeze_study_manifest(spec)


def single_view_spec(study_tree):
    spec = spec_for(study_tree)
    baseline = {
        "solo": counts(3, 3, [1, 1, 1], [1, 1, 1], 0, 1),
    }
    spec["evaluation"]["objectives"] = {
        "baseline": baseline,
        "targets": {
            "solo": {
                "rows": 3,
                "min_correct": 3,
                "max_false_safe": 0,
                "min_class_correct": [1, 1, 1],
            }
        },
        "retention": {
            "solo": {"rows": 3, "min_correct": 2, "max_false_safe": 0}
        },
    }
    spec["evaluation"]["views"] = {
        "solo": {"row_ids": ["d-sat", "d-vio", "d-ctx"]}
    }
    spec["evaluation"]["selection_objective"].update(
        view="solo",
        protected=spec["evaluation"]["objectives"]["retention"],
        targets=spec["evaluation"]["objectives"]["targets"]["solo"],
    )
    return spec


def dev_report(predictions, confusion, correct):
    return {
        "run_id": "demo-r1",
        "n": 3,
        "accuracy": correct / 3,
        "confusion": confusion,
        "false_safe_count": confusion[1][0],
        "complete_pairs": 1,
        "both_pair_members_correct": 1.0 if correct == 3 else 0.0,
        "nll": 0.1,
        "predictions": predictions,
        "stop_reason": "update_budget",
        "development": None,
        "test_evaluated": False,
        "calibration_fitted": False,
        "final_evaluation": False,
        "selected_checkpoint": {"weights_sha256": "ab" * 32},
    }


def test_assess_round_from_trainer_records(study_tree, tmp_path):
    manifest = decision_study.freeze_study_manifest(
        single_view_spec(study_tree)
    )
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "event": "development",
                "delivered_exposure": {
                    "unique_rows": 3,
                    "total_rows": 3,
                    "broken_triplets": [],
                    "split_triplets": [],
                },
            }
        )
        + "\n"
    )
    report = dev_report([1, 1, 2], [[0, 1, 0], [0, 1, 0], [0, 0, 1]], 2)
    report["development"] = {k: report[k] for k in report if k != "development"}
    evidence = decision_study.assess_round(
        manifest,
        report,
        events,
        {"correct": 3, "rows": 3, "meets_probe_criterion": True},
        sdk_parity_ok=True,
        expected_run_id="demo-r1",
        checkpoint_sha256="ab" * 32,
    )
    assert evidence["target_met"] is False
    assert evidence["retention_ok"] is True
    assert evidence["exposure_complete"] is True
    verdict = decision_study.diagnose(evidence)
    assert verdict["category"] == "poor_generalization"


def test_assess_round_requires_bound_checkpoint(study_tree, tmp_path):
    manifest = decision_study.freeze_study_manifest(spec_for(study_tree))
    with pytest.raises(ValueError, match="checkpoint"):
        decision_study.assess_round(
            manifest,
            {"run_id": "x"},
            tmp_path / "events.jsonl",
            {"correct": 1, "rows": 1, "meets_probe_criterion": True},
            expected_run_id="x",
            checkpoint_sha256="ab" * 32,
        )


def test_report_counts_tolerates_null_pairs(study_tree):
    dev_rows = [
        json.loads(line)
        for line in (study_tree["dev"]).read_text().splitlines()
    ]
    counts = decision_study.report_counts(
        {
            "n": 3,
            "accuracy": 1 / 3,
            "confusion": [[0, 0, 1], [0, 1, 0], [0, 1, 0]],
            "false_safe_count": 0,
            "complete_pairs": None,
            "both_pair_members_correct": 0.0,
            "nll": 1.1,
            "predictions": [2, 1, 1],
        },
        dev_rows,
    )
    assert counts["pairs"] == 1
    assert counts["correct"] == 1
    assert counts["triplets_correct"] == 0


def test_read_terminal_exposure_requires_tracking(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps({"event": "train"}) + "\n")
    with pytest.raises(ValueError, match="no delivered exposure"):
        decision_study.read_terminal_exposure(events)


def test_read_terminal_exposure_nested_metrics(tmp_path):
    exposure = {
        "unique_rows": 3,
        "total_rows": 3,
        "broken_triplets": [],
        "split_triplets": [],
    }
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "event": "development",
                "metrics": {
                    "accuracy": 1.0,
                    "delivered_exposure": exposure,
                },
            }
        )
        + "\n"
    )
    assert decision_study.read_terminal_exposure(events) == exposure


def analysis_record(round_id="r1", **extra):
    record = {
        "schema": sup.ANALYSIS_SCHEMA,
        "round_id": round_id,
        "verdict_summary": "v",
        "what_failed": {"metrics": {}, "artifacts": []},
        "explanation": "e",
        "proposed_change": "c",
        "predicted_improvement": "p",
        "possible_regressions": "r",
        "falsifier": "f",
        "next_action": {"type": "stop", "reason": "done"},
    }
    record.update(extra)
    return record


def test_check_analysis_validates_diagnosis():
    checked = sup.check_analysis(
        analysis_record(
            diagnosis={
                "category": "poor_labels",
                "evidence": {
                    "job_complete": True,
                    "exposure_complete": True,
                    "sdk_parity_ok": True,
                    "train_fit_ok": True,
                },
                "reasons": ["audit flags"],
                "next": "repair data",
            }
        ),
        "r1",
    )
    assert checked["diagnosis"]["category"] == "poor_labels"
    assert sup.check_analysis(analysis_record(), "r1")["diagnosis"] is None
    with pytest.raises(ValueError, match="no known cause"):
        sup.check_analysis(
            analysis_record(
                diagnosis={
                    "category": "bad_luck",
                    "evidence": {
                        "job_complete": True,
                        "exposure_complete": True,
                        "sdk_parity_ok": True,
                        "train_fit_ok": True,
                    },
                    "reasons": ["x"],
                    "next": "y",
                }
            ),
            "r1",
        )


def test_study_exhausted_guard(study_tree, tmp_path):
    home = tmp_path / "home"
    write_campaign(home, study_tree["parent"])
    write_study(
        home, decision_study.freeze_study_manifest(spec_for(study_tree))
    )
    assert sup.study_exhausted(home, "demo") is True
    spec = spec_for(study_tree)
    spec["budget"]["max_rounds"] = 3
    write_study(home, decision_study.freeze_study_manifest(spec))
    assert sup.study_exhausted(home, "demo") is False
