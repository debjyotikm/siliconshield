"""Check arithmetic, negation, intervention parameters, and curriculum boundaries."""

import copy
import json

import pytest

from siliconshield.research_loop.controller.replay import _failure_claims, grade
from siliconshield.research_loop.episodes.contracts import final_payload
from siliconshield.research_loop.episodes.operator_assessment import (
    assessment,
    assessment_issues,
    parameter_issues,
)
from siliconshield.research_loop.episodes.operator_curriculum import (
    build,
    episode,
    fixture_verify,
)


def spec():
    return {
        "scenario": "illustrative_recovery",
        "family": "authored-family",
        "index": 0,
        "split": "train",
        "action": "resume",
        "reason": "Resume the validated owned checkpoint.",
        "state": {
            "phase": "paused",
            "systemd": "inactive",
            "checkpoint_valid": True,
            "execution_failed": False,
            "selected_step": 10,
            "schedule": {"plateau_lr_reductions": []},
            "development_history": {
                "entries": [
                    {
                        "population_id": "dev",
                        "metrics": {"n": 100, "accuracy": 0.8, "nll": 0.4},
                    },
                    {
                        "population_id": "dev",
                        "metrics": {"n": 100, "accuracy": 0.9, "nll": 0.3},
                    },
                ]
            },
        },
        "resources": {"gpus_free": 3, "owned_by_this_run": False},
    }


@pytest.mark.parametrize(
    "text,asserts_failure",
    [
        ("This is a learning failure, not an execution failure.", False),
        ("There was no execution failure.", False),
        (
            "This execution failure does not establish unseen-family generalization.",
            True,
        ),
        ("The job crashed. No final-test claim is supported.", True),
    ],
)
def test_failure_negation_is_scoped(text, asserts_failure):
    assert bool(_failure_claims(text)) is asserts_failure


def test_assessment_checks_metric_binding_and_unknowns():
    state = spec()["state"]
    facts = assessment(state)
    assert facts["comparisons"]["development"]["accuracy_delta_pp"] == 10.0
    assert facts["comparisons"]["development"]["nll_delta"] == -0.1
    assert facts["acceptance"] == "unknown"
    assert facts["comparisons"]["development"]["false_safe_delta_pp"] is None
    assert assessment_issues(state, facts) == []
    for value in (float("nan"), float("inf"), True, -10.0):
        wrong = copy.deepcopy(facts)
        wrong["comparisons"]["development"]["accuracy_delta_pp"] = value
        assert assessment_issues(state, wrong)
    state["development_history"]["entries"][-1]["population_id"] = "another"
    assert assessment(state)["comparisons"] == {}
    assert assessment_issues(state, facts)
    for entry in state["development_history"]["entries"]:
        entry.pop("population_id")
    assert assessment(state)["comparisons"] == {}


def test_proposed_parameters_must_match_owned_state():
    row = episode(spec())
    assert parameter_issues(
        row, {"action": "resume", "checkpoint_dir": "/wrong"}
    )
    assert not parameter_issues(row, row["target"])
    row["summary"]["lr_bounds"] = [1e-6, 1e-4]
    row["target"]["lr"] = 2e-5
    assert parameter_issues(row, {"action": "adjust_lr", "lr": 9e-5})
    assert not parameter_issues(row, {"action": "adjust_lr", "lr": 2e-5})


def test_fixture_uses_real_dispatch_and_structured_evidence():
    row = episode(spec())
    fixture_verify(row)
    decision = {
        k: v for k, v in row["target"].items() if k != "require_assessment"
    }
    decision["run_id"] = row["summary"]["run_id"]
    delivered = {
        "decision": decision,
        "read_evidence": True,
        "compared_metrics": True,
        "final": json.dumps(final_payload("resume", run_id=decision["run_id"])),
    }
    assert grade(row, delivered)["verified"]
    decision["assessment"]["selected_step"] = 999
    result = grade(row, delivered)
    assert (
        result["action_correct"]
        and not result["evidence_correct"]
        and not result["verified"]
    )


def test_catalog_family_cannot_cross_split(tmp_path, monkeypatch):
    key = tmp_path / "key"
    key.write_bytes(b"fixture-signing-key")
    monkeypatch.setenv("SILICONSHIELD_DATASET_KEY", str(key))
    train, dev = spec(), spec()
    dev.update(index=1, split="dev")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(json.dumps(train) + "\n" + json.dumps(dev) + "\n")
    with pytest.raises(ValueError, match="crosses dataset splits"):
        build(catalog, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
