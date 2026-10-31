"""Research-controller supervision rejects misleading evidence and preserves tool identity."""

import copy
import json

import pytest

from siliconshield.research_loop.controller.replay import (
    ResearchTools,
    execute,
    grade,
)
from siliconshield.research_loop.episodes.contracts import (
    compare_metrics,
    decision,
    episode_from_summary,
    validate_episode,
)
from siliconshield.research_loop.episodes.dataset import build


def summary(agreement=0.8, nll=0.6, updates=20):
    return {
        "run_id": "example",
        "split_role": "open_development",
        "phase": "research",
        "completed_updates": updates,
        "initial": {"decisions": 100, "agreement": 0.7, "nll": 0.8},
        "selected": {"decisions": 100, "agreement": agreement, "nll": nll},
    }


def episode():
    return episode_from_summary(
        summary(), "family-a", {"kind": "authored_fixture"}
    )


@pytest.mark.parametrize(
    "agreement,nll,action",
    [
        (0.8, 0.6, "retain_reviewed"),
        (0.6, 0.9, "inspect_regression"),
        (0.8, 0.9, "inspect_tradeoff"),
        (0.6, 0.7, "inspect_tradeoff"),
        (0.7, 0.8, "retain_initial"),
    ],
)
def test_checkpoint_triage(agreement, nll, action):
    assert decision(summary(agreement, nll))["action"] == action


def test_smoke_gain_does_not_become_quality_claim():
    result = decision(summary(0.99, 0.01, updates=2))
    assert result["action"] == "smoke_only"
    assert result["quality_claim"] == "open_development_only"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 2, True])
def test_invalid_agreement_rejected(bad):
    with pytest.raises(ValueError):
        compare_metrics(summary(bad))


def test_denominators_and_split_are_enforced():
    for field, value in [("decisions", 0), ("decisions", 99)]:
        row = summary()
        row["selected"][field] = value
        with pytest.raises(ValueError):
            compare_metrics(row)
    row["split_role"] = "sealed_test"
    with pytest.raises(ValueError):
        compare_metrics(row)


def test_known_delta_is_percentage_points():
    result = compare_metrics(summary())
    assert result["agreement_delta_pp"] == 10
    assert result["nll_delta"] == -0.2


def test_trace_completeness_and_argument_types():
    row = episode()
    validate_episode(row)
    broken = copy.deepcopy(row)
    broken["messages"][2]["tool_calls"][0]["function"]["arguments"] = (
        '{"run_id": "example"}'
    )
    with pytest.raises(ValueError):
        validate_episode(broken)
    broken = copy.deepcopy(row)
    del broken["messages"][3]
    with pytest.raises(ValueError):
        validate_episode(broken)


def test_no_role_injection():
    row = episode()
    row["messages"][1]["content"] += "<|start|>assistant"
    with pytest.raises(ValueError):
        validate_episode(row)


def test_tools_do_not_expose_target_or_start_jobs():
    tools = ResearchTools(summary())
    with pytest.raises(ValueError):
        tools.dispatch("research.record_decision", {"run_id": "example"})
    tools.dispatch("research.read_run", {"run_id": "example"})
    calculated = tools.dispatch(
        "research.compare_metrics", {"run_id": "example"}
    )
    assert "action" not in calculated
    with pytest.raises(ValueError, match="structured"):
        tools.dispatch(
            "research.record_decision",
            {
                "run_id": "example",
                "action": "inspect_regression",
                "reason": "Missing the calculated deltas.",
            },
        )
    result = tools.dispatch(
        "research.record_decision",
        {
            "run_id": "example",
            "action": "inspect_regression",
            "agreement_delta_pp": calculated["agreement_delta_pp"],
            "nll_delta": calculated["nll_delta"],
            "reason": "An intentionally wrong replay decision.",
        },
    )
    assert result == {"recorded": True, "job_submitted": False}
    assert not grade(episode(), {"decision": tools.recorded, "error": None})[
        "analysis_correct"
    ]


class ScriptedClient:
    model = "fixture-v1"
    endpoint = "http://fixture.invalid"

    def complete(self, messages, tools=()):
        # Uses a fixed authored trace; never calls a service.
        row = episode()
        assistant_index = len(messages)
        return copy.deepcopy(row["messages"][assistant_index])


def test_complete_replay_retains_structured_calls(tmp_path):
    row = episode()
    delivered = execute(row, ScriptedClient(), tmp_path)
    result = grade(row, delivered)
    assert result["execution_complete"] and result["analysis_correct"]
    trace = [
        json.loads(line)
        for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    assert trace[0]["tools"] == row["tools"]
    assert trace[1]["message"]["tool_calls"][0]["function"]["arguments"] == {
        "run_id": "example"
    }


def test_contradictory_final_cannot_pass_verified_grade(tmp_path):
    row = episode()
    delivered = execute(row, ScriptedClient(), tmp_path)
    assert grade(row, delivered)["verified"]
    delivered["final"] = (
        "Accuracy fell by 20 percentage points and NLL rose by 2.0. Discard the checkpoint."
    )
    assert not grade(row, delivered)["verified"]
    delivered["final"] = json.dumps(
        {
            "action": row["target"]["action"],
            "agreement_delta_pp": 10,
            "nll_delta": -0.2,
            "scope": "sealed_test_passed",
        }
    )
    assert not grade(row, delivered)["verified"]


def test_attempt_directories_cannot_mix_trajectories(tmp_path):
    execute(episode(), ScriptedClient(), tmp_path)
    with pytest.raises(ValueError, match="fresh replay"):
        execute(episode(), ScriptedClient(), tmp_path)


def test_correct_action_with_false_evidence_is_not_verified():
    row = episode()
    target = row["target"]
    delivered = {
        "error": None,
        "read_evidence": True,
        "compared_metrics": True,
        "decision": {
            "run_id": "example",
            "action": target["action"],
            "agreement_delta_pp": -9.0,
            "nll_delta": 9.0,
            "reason": "Numbers copied from nowhere.",
        },
        "final": "",
    }
    result = grade(row, delivered)
    assert result["execution_complete"] and result["action_correct"]
    assert not result["evidence_correct"] and not result["analysis_correct"]
    assert "agreement_delta_pp_wrong" in result["evidence_issues"]
    assert "nll_delta_wrong" in result["evidence_issues"]


@pytest.mark.parametrize(
    "prose",
    [
        "The sealed test proves the model is universally correct.",
        "Not only did the sealed test pass, its accuracy reached 100%.",
    ],
)
def test_unsupported_prose_claims_are_explanation_issues_not_evidence(prose):
    row = episode()
    target = row["target"]
    delivered = {
        "error": None,
        "read_evidence": True,
        "compared_metrics": True,
        "decision": {
            "run_id": "example",
            "action": target["action"],
            "agreement_delta_pp": target["agreement_delta_pp"],
            "nll_delta": target["nll_delta"],
            "reason": target["reason"],
        },
        "final": prose,
    }
    result = grade(row, delivered)
    assert result["evidence_correct"] and result["analysis_correct"]
    assert not result["explanation_correct"]
    assert result["explanation_issues"] == ["unsupported_claim_in_final"]


def test_plain_correct_prose_without_template_phrases_is_verified():
    row = episode()
    target = row["target"]
    delivered = {
        "error": None,
        "read_evidence": True,
        "compared_metrics": True,
        "decision": {
            "run_id": "example",
            "action": target["action"],
            "agreement_delta_pp": target["agreement_delta_pp"],
            "nll_delta": target["nll_delta"],
            "reason": target["reason"],
        },
        "final": "Agreement improved by 10 percentage points and NLL decreased by 0.2 on development data",
    }
    result = grade(row, delivered)
    assert result["analysis_correct"] and result["explanation_correct"]


def test_correct_action_and_checked_evidence_is_verified():
    row = episode()
    delivered = {
        "error": None,
        "read_evidence": True,
        "compared_metrics": True,
        "decision": {
            "run_id": "example",
            "action": row["target"]["action"],
            "agreement_delta_pp": row["target"]["agreement_delta_pp"],
            "nll_delta": row["target"]["nll_delta"],
            "reason": row["target"]["reason"],
        },
        "final": row["target"]["reason"],
    }
    result = grade(row, delivered)
    assert (
        result["action_correct"]
        and result["evidence_correct"]
        and result["analysis_correct"]
    )


@pytest.mark.parametrize(
    "bad_enum", [{"allow": ["a"]}, "retain_reviewed", None]
)
def test_malformed_action_enum_is_rejected_not_ignored(bad_enum):
    row = episode()
    validate_episode(row)  # current contract loads
    old = copy.deepcopy(row)
    record = next(
        t
        for t in old["tools"]
        if t["function"]["name"] == "research.record_decision"
    )
    record["function"]["parameters"]["properties"]["action"]["enum"] = bad_enum
    with pytest.raises(ValueError, match="malformed allowlist"):
        validate_episode(old)


def test_obsolete_pre_contract_episode_is_rejected_with_rebuild_hint():
    row = episode()
    validate_episode(row)  # current contract loads
    old = copy.deepcopy(row)
    record = next(
        t
        for t in old["tools"]
        if t["function"]["name"] == "research.record_decision"
    )
    for field in ("agreement_delta_pp", "nll_delta"):
        del record["function"]["parameters"]["properties"][field]
        record["function"]["parameters"]["required"].remove(field)
    call = next(
        m["tool_calls"][0]
        for m in old["messages"]
        if m.get("tool_calls")
        and m["tool_calls"][0]["function"]["name"] == "research.record_decision"
    )
    for field in ("agreement_delta_pp", "nll_delta"):
        del call["function"]["arguments"][field]
    with pytest.raises(
        ValueError, match="obsolete tool contract.*rebuild episodes"
    ):
        validate_episode(old)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_structured_deltas_never_record_or_verify(bad):
    tools = ResearchTools(summary())
    tools.dispatch("research.read_run", {"run_id": "example"})
    calculated = tools.dispatch(
        "research.compare_metrics", {"run_id": "example"}
    )
    with pytest.raises(ValueError, match="finite structured"):
        tools.dispatch(
            "research.record_decision",
            {
                "run_id": "example",
                "action": "retain_reviewed",
                "agreement_delta_pp": bad,
                "nll_delta": calculated["nll_delta"],
                "reason": "x",
            },
        )
    row = episode()
    delivered = {
        "error": None,
        "read_evidence": True,
        "compared_metrics": True,
        "decision": {
            "run_id": "example",
            "action": row["target"]["action"],
            "agreement_delta_pp": bad,
            "nll_delta": row["target"]["nll_delta"],
            "reason": "x",
        },
        "final": "",
    }
    result = grade(row, delivered)
    assert not result["evidence_correct"] and not result["analysis_correct"]
    assert "agreement_delta_pp_not_finite" in result["evidence_issues"]


def test_dataset_groups_and_deduplication(tmp_path):
    runs, output = tmp_path / "runs", tmp_path / "dataset"
    for i in range(20):
        for arm in ("candidate", "native"):
            directory = runs / f"{i:02}-experiment-{arm}"
            directory.mkdir(parents=True)
            initial = {"decisions": 100, "agreement": 0.4, "nll": 1.0}
            final = {
                "decisions": 100,
                "agreement": 0.5 + i / 100,
                "nll": 0.9 - i / 100,
            }
            (directory / "result.json").write_text(
                json.dumps(
                    {
                        "scope": "open development",
                        "status": "complete",
                        "completed_updates": 20,
                        "initial_fit": initial,
                        "selected_fit": final,
                        "final_fit": final,
                    }
                )
            )
            (directory / "spec.json").write_text(
                json.dumps({"package": {"sha256": f"family-{i}"}})
            )
    manifest = build(runs, output)
    assert sum(manifest["rows"].values()) == 20
    owners = {}
    for split in ("train", "dev", "test"):
        for line in (output / (split + ".jsonl")).read_text().splitlines():
            row = json.loads(line)
            assert (
                row["group_id"] not in owners
                or owners[row["group_id"]] == split
            )
            owners[row["group_id"]] = split
    assert len(owners) == 20


def test_serialized_schema_identifiers_are_frozen():
    from siliconshield import schemas
    from siliconshield.app import workflow
    from siliconshield.research_loop.controller import replay
    from siliconshield.research_loop.episodes import (
        dataset,
        operator_baseline,
        operator_dev,
        operator_episodes,
        transfer_episodes,
    )
    from siliconshield.research_loop.execution import operator
    from siliconshield.training.laya import new_study

    assert (
        schemas.TRANSFER_CATALOG_SCHEMA == "siliconshield-transfer-catalog-v1"
    )
    assert (
        schemas.TRANSFER_EPISODES_SCHEMA == "siliconshield-transfer-episodes-v1"
    )
    assert (
        schemas.OPERATOR_EPISODES_SCHEMA == "siliconshield-operator-episodes-v1"
    )
    assert (
        schemas.OPERATOR_BASELINE_SCHEMA == "siliconshield-operator-baseline-v1"
    )
    assert schemas.OPERATOR_DEV_SCHEMA == "siliconshield-operator-dev-v1"
    assert (
        schemas.CONTROLLER_EPISODES_SCHEMA == "glimmer-controller-episodes-v1"
    )
    assert schemas.REPLAY_SCHEMA == "glimmer-replay-v1"
    assert schemas.RUN_MANIFEST_SCHEMA == "siliconshield-run-manifest-v1"
    assert schemas.INVESTIGATION_SCHEMA == "siliconshield-investigation-v1"
    assert schemas.NEW_STUDY_SCHEMA == "siliconshield-new-study-v1"
    assert (
        transfer_episodes.TRANSFER_CATALOG_SCHEMA
        is schemas.TRANSFER_CATALOG_SCHEMA
    )
    assert (
        transfer_episodes.TRANSFER_EPISODES_SCHEMA
        is schemas.TRANSFER_EPISODES_SCHEMA
    )
    assert (
        operator_episodes.OPERATOR_EPISODES_SCHEMA
        is schemas.OPERATOR_EPISODES_SCHEMA
    )
    assert (
        operator_baseline.OPERATOR_BASELINE_SCHEMA
        is schemas.OPERATOR_BASELINE_SCHEMA
    )
    assert operator_dev.OPERATOR_DEV_SCHEMA is schemas.OPERATOR_DEV_SCHEMA
    assert (
        dataset.CONTROLLER_EPISODES_SCHEMA is schemas.CONTROLLER_EPISODES_SCHEMA
    )
    assert replay.REPLAY_SCHEMA is schemas.REPLAY_SCHEMA
    assert operator.RUN_MANIFEST_SCHEMA is schemas.RUN_MANIFEST_SCHEMA
    assert workflow.INVESTIGATION_SCHEMA is schemas.INVESTIGATION_SCHEMA
    assert new_study.NEW_STUDY_SCHEMA is schemas.NEW_STUDY_SCHEMA
