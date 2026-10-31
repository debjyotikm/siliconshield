"""Phase-1 qualification dataset: coverage, leakage, split isolation, acceptance."""

import copy
import json

from siliconshield.research_loop.episodes.contracts import validate_episode
from siliconshield.research_loop.episodes.operator_episodes import (
    SCENARIOS,
    build,
    episode_for,
    migrate_operator_dataset,
    migrate_operator_row,
    validate_operator_dataset,
)
from siliconshield.training.glimmer.train import load_rows


def test_all_scenarios_produce_valid_episodes():
    for scenario, state, resources, action, reason in SCENARIOS:
        row = episode_for(
            scenario, state, resources, action, reason, "lineage-x", 0
        )
        validate_episode(row)
        visible = json.dumps(row["messages"])
        assert row["outcome"]["recorded_only"] and "outcome" not in visible


def test_qualification_dataset_acceptance_and_splits(tmp_path):
    from siliconshield.schemas import OPERATOR_EPISODES_SCHEMA

    manifest = build(tmp_path / "data", lineages_per_scenario=1)
    assert manifest["schema"] == OPERATOR_EPISODES_SCHEMA
    assert sum(manifest["rows"].values()) == len(SCENARIOS)
    rows = []
    owners = {}
    for split in ("train", "dev", "test"):
        loaded = load_rows(
            tmp_path / "data" / (split + ".jsonl"), allowed_splits=(split,)
        )
        rows.extend(loaded)
        for row in loaded:
            assert (
                row["group_id"] not in owners
                or owners[row["group_id"]] == split
            )
            owners[row["group_id"]] = split
    report = validate_operator_dataset(rows)
    assert report["accepted"], report["issues"]
    assert set(report["actions"]) >= {
        "continue",
        "leave_unchanged",
        "stop",
        "adjust_lr",
        "resume",
        "finalize_retry",
    }


def test_leakage_scan_catches_future_fields():
    scenario, state, resources, action, reason = SCENARIOS[0]
    row = episode_for(
        scenario, state, resources, action, reason, "lineage-y", 0
    )
    row["messages"][1]["content"] += " future_metric: improved"
    report = validate_operator_dataset([row] * 3)
    assert not report["accepted"]
    assert any("future_" in issue for issue in report["issues"])


def test_duplicate_scenarios_are_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="not independent"):
        build(tmp_path, lineages_per_scenario=6)
    args = SCENARIOS[0]
    row = episode_for(*args, "lineage", 0)
    assert not validate_operator_dataset([row, row])["accepted"]


def test_no_scenario_crosses_splits(tmp_path):
    build(tmp_path)
    ownership = {}
    for split in ("train", "dev", "test"):
        for row in load_rows(
            tmp_path / (split + ".jsonl"), allowed_splits=(split,)
        ):
            name = row["verification"]["scenario"]
            assert ownership.setdefault(name, split) == split


def test_coverage_distinguishes_copies_from_operator_signal():
    from siliconshield.research_loop.episodes.operator_episodes import (
        operator_coverage,
    )

    train = [episode_for(*SCENARIOS[0], "group-%d" % i, i) for i in range(600)]
    dev = [episode_for(*s, "dev-" + s[0], 0) for s in SCENARIOS]
    report = operator_coverage(train, dev, dev[:1])
    assert report["train"]["operator_rows"] == 600
    assert report["train"]["distinct_operator_situations"] == 1
    assert "resume" in report["train"]["missing_actions"]
    assert "adjust_lr" in report["replay"]["missing_actions"]
    assert not report["coverage_passed"]


def test_operator_training_rejects_qualification_dataset_before_model_load(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import pytest

    from siliconshield.training.glimmer import train

    rows = [episode_for(*s, "group-" + s[0], 0) for s in SCENARIOS]
    monkeypatch.setattr(train, "training_rows", lambda data: (rows, rows))
    args = SimpleNamespace(
        data=tmp_path,
        output=tmp_path,
        dev_examples=16,
        purpose="operator",
        task_eval_every=100,
    )
    with pytest.raises(ValueError, match="Operator training requires"):
        train._train(args, "a" * 64)
    receipt = json.loads((tmp_path / "operator-coverage.json").read_text())
    assert not receipt["coverage_passed"] and receipt["purpose"] == "operator"


def _old_style_row(scenario_name):
    """Simulate a saved pre-contract episode: old tools, missing decide fields."""
    scenario, state, resources, action, reason = next(
        s for s in SCENARIOS if s[0] == scenario_name
    )
    row = episode_for(
        scenario, state, resources, action, reason, "old-lineage", 0
    )
    row = copy.deepcopy(row)
    for message in row["messages"]:
        for call in message.get("tool_calls", []):
            if call["function"]["name"] == "operator.decide":
                call["function"]["arguments"].pop("lr", None)
                call["function"]["arguments"].pop("checkpoint_dir", None)
    for tool in row["tools"]:
        if tool["function"]["name"] == "operator.decide":
            properties = tool["function"]["parameters"]["properties"]
            properties.pop("lr", None)
            properties.pop("checkpoint_dir", None)
    row["target"].pop("lr", None)
    row["target"].pop("checkpoint_dir", None)
    return row


def test_migration_fills_resume_and_adjust_lr_fields():
    for scenario_name, field in (
        ("valid_resume", "checkpoint_dir"),
        ("lr_instability", "lr"),
    ):
        row = _old_style_row(scenario_name)
        try:
            validate_episode(row)
            raise AssertionError("old row should fail the current contract")
        except ValueError:
            pass
        migrated, outcome = migrate_operator_row(row)
        assert outcome["status"] == "migrated", outcome
        assert field in migrated["target"]
        validate_episode(migrated)
        assert field in migrated["verification"]["migration"]["filled"]


def test_migration_rejects_obsolete_actions_without_remapping():
    row = _old_style_row("valid_resume")
    for message in row["messages"]:
        for call in message.get("tool_calls", []):
            if call["function"]["name"] == "operator.decide":
                call["function"]["arguments"]["action"] = "investigate_data"
    row["target"]["action"] = "investigate_data"
    migrated, outcome = migrate_operator_row(row)
    assert migrated is None and outcome["status"] == "rejected"


def test_migrate_dataset_preserves_sealed_test(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    (source).mkdir()
    rows = [_old_style_row("valid_resume"), _old_style_row("lr_instability")]
    (source / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (source / "dev.jsonl").write_text("")
    test_payload = '{"sealed": true}\n'
    (source / "test.jsonl").write_text(test_payload)
    (source / "manifest.json").write_text(
        json.dumps({"schema": "x", "groups": {"g": "test"}, "files": {}})
    )
    report = migrate_operator_dataset(source, dest)
    assert report["migrated"] == 2 and not report["rejected"]
    assert (dest / "test.jsonl").read_text() == test_payload
    assert report["test_rows_copied"] == 1


def test_fabricated_execution_failure_is_not_verified():
    import tempfile

    from siliconshield.research_loop.controller.replay import execute, grade

    scenario, state, resources, action, reason = next(
        s for s in SCENARIOS if s[0] == "healthy_progressing"
    )
    row = episode_for(scenario, state, resources, action, reason, "lin-fab", 0)

    class TraceClient:
        model = "fixture"

        def complete(self, messages, tools=()):
            return row["messages"][len(messages)]

    with tempfile.TemporaryDirectory() as attempt:
        delivered = execute(row, TraceClient(), attempt)
    assert grade(row, delivered)["verified"] is True
    bad = copy.deepcopy(delivered)
    bad["decision"] = dict(
        bad["decision"],
        reason=(
            "This execution failure is a loading problem. Stop the run; retrying will repeat the failure."
        ),
    )
    result = grade(row, bad)
    assert result["action_correct"] is True
    assert result["explanation_correct"] is False
    assert result["verified"] is False
    assert (
        "explanation_contradicts_execution_evidence_in_reason"
        in result["explanation_issues"]
    )


def _operator_row(
    summary, action="continue", reason="Continuing within budget."
):
    from siliconshield.research_loop.episodes.contracts import OPERATOR_TOOLS

    return {
        "id": "repro-1",
        "group_id": "repro-g",
        "tools": OPERATOR_TOOLS,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Inspect run repro-1."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "i",
                        "type": "function",
                        "function": {
                            "name": "operator.decide",
                            "arguments": {"run_id": "repro-1"},
                        },
                    }
                ],
            },
        ],
        "summary": summary,
        "target": {"action": action, "reason": reason},
        "verification": {"kind": "authored_fixture"},
    }


def _operator_delivered(reason, action="continue"):
    return {
        "error": None,
        "read_evidence": True,
        "compared_metrics": True,
        "decision": {"run_id": "repro-1", "action": action, "reason": reason},
        "final": '{"action": "%s", "run_id": "repro-1", "executed": false, "scope": "operator_replay"}'
        % action,
    }


def _failed_acceptance_summary():
    return {
        "run_id": "repro-1",
        "systemd": "running",
        "step": 200,
        "execution_failed": False,
        "acceptance": {
            "accepted": False,
            "reasons": ["worst_family_false_safe_above_budget"],
            "thresholds": {"max_worst_family_false_safe_rate": 0.25},
        },
        "development_history": {
            "entries": [
                {"step": 100, "metrics": {"accuracy": 0.9, "nll": 0.3}},
                {"step": 200, "metrics": {"accuracy": 0.94, "nll": 0.36}},
            ],
            "complete": True,
            "total": 2,
        },
        "schedule": {"plateau_lr_reductions": []},
        "selected_step": 200,
    }


def test_structured_claims_reject_false_acceptance_train_loss_proof_and_phantom_lr():
    from siliconshield.research_loop.controller.replay import grade

    row = _operator_row(_failed_acceptance_summary())
    delivered = _operator_delivered(
        "All acceptance checks passed. Near-zero training loss proves generalization. "
        "The plateau controller reduced the learning rate twice."
    )
    result = grade(row, delivered)
    assert result["action_correct"] is True
    assert result["explanation_correct"] is False
    assert result["verified"] is False
    assert (
        "explanation_claims_acceptance_despite_failed_criteria_in_reason"
        in result["explanation_issues"]
    )
    assert (
        "explanation_train_loss_proves_quality_in_reason"
        in result["explanation_issues"]
    )
    assert (
        "explanation_plateau_reduction_differs_from_record_in_reason"
        in result["explanation_issues"]
    )


def test_supported_acceptance_and_denials_pass_grading():
    from siliconshield.research_loop.controller.replay import grade

    accepted = {
        "run_id": "repro-1",
        "systemd": "running",
        "step": 200,
        "execution_failed": False,
        "acceptance": {"accepted": True, "reasons": []},
        "development_history": {
            "entries": [
                {"step": 200, "metrics": {"accuracy": 0.94, "nll": 0.36}}
            ],
            "complete": True,
            "total": 1,
        },
        "schedule": {"plateau_lr_reductions": []},
    }
    row = _operator_row(accepted)
    delivered = _operator_delivered(
        "Acceptance passed on every criterion; continuing."
    )
    assert grade(row, delivered)["verified"] is True
    # Hedged or negated phrasing is not a claim.
    row2 = _operator_row(_failed_acceptance_summary())
    delivered2 = _operator_delivered(
        "The training loss does not prove anything yet."
    )
    assert grade(row2, delivered2)["explanation_correct"] is True
    delivered3 = _operator_delivered(
        "No plateau reduction occurred; lr_scale is still 1.0."
    )
    assert (
        grade(row3 := _operator_row(_failed_acceptance_summary()), delivered3)[
            "explanation_correct"
        ]
        is True
    )


def test_plateau_denial_requires_proximity_not_sentence_match():
    from siliconshield.research_loop.controller.replay import (
        _claimed_plateau_reductions,
        grade,
    )

    # Leading denial suppresses the claim.
    assert (
        _claimed_plateau_reductions(
            "No plateau reduction occurred; lr_scale is still 1.0."
        )
        == []
    )
    # Trailing denial in another clause must not suppress an earlier real claim.
    trailing = _claimed_plateau_reductions(
        "The plateau controller reduced the learning rate twice; no further action is needed."
    )
    assert trailing and trailing[0][1] == 2
    row = _operator_row(_failed_acceptance_summary())
    result = grade(
        row,
        _operator_delivered(
            "The plateau controller reduced the learning rate twice; no further action is needed."
        ),
    )
    assert (
        "explanation_plateau_reduction_differs_from_record_in_reason"
        in result["explanation_issues"]
    )
