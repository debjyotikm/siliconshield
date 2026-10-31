"""Regression checks + independent dev: correct decisions verify, unsupported do not."""

import json

import pytest

from siliconshield.research_loop.episodes.operator_baseline import (
    build_baseline,
    evaluate_baseline,
    load_baseline,
)


def _fixture_client(rows):
    indexed = {row["summary"]["run_id"]: row for row in rows}

    class FixtureClient:
        model = "baseline-fixture"
        endpoint = "http://fixture.invalid"

        def complete(self, messages, tools=()):
            import re

            match = re.search(r"Inspect run (\S+)", messages[1]["content"])
            row = indexed[match.group(1)]
            return json.loads(json.dumps(row["messages"][len(messages)]))

    return FixtureClient()


def test_baseline_covers_all_six_actions_with_held_out_lineages(tmp_path):
    from siliconshield.schemas import OPERATOR_BASELINE_SCHEMA

    manifest = build_baseline(tmp_path / "base")
    assert manifest["schema"] == OPERATOR_BASELINE_SCHEMA
    assert (
        manifest["rows"] == 9
        and manifest["never_train"]
        and manifest["regression_checks"]
    )
    assert not manifest["held_out"]
    assert set(manifest["actions"]) >= {
        "continue",
        "leave_unchanged",
        "stop",
        "adjust_lr",
        "resume",
        "finalize_retry",
    }
    rows = load_baseline(tmp_path / "base/baseline.jsonl")
    assert all(r["group_id"].startswith("baseline-heldout-") for r in rows)
    # Overlap guard: shared history is regression-check only, never training.
    assert manifest["regression_check_subset"] == ["regression_stop_supported"]
    assert manifest["overlap"]["regression_stop_supported"] == [
        "dpo-stop-grounding-v1",
        "dpo-baseline-stop-supported",
    ]


def test_baseline_grading_separates_action_from_explanation(tmp_path):
    build_baseline(tmp_path / "base")
    rows = load_baseline(tmp_path / "base/baseline.jsonl")
    summary = evaluate_baseline(
        _fixture_client(rows),
        tmp_path / "base/baseline.jsonl",
        tmp_path / "eval",
    )
    assert summary["action_correct"] == 9 and summary["evidence_correct"] == 9
    assert summary["verified"] == 6
    by_scenario = {r["scenario"]: r for r in summary["results"]}
    assert by_scenario["regression_stop_supported"]["verified"] is True
    assert (
        by_scenario["healthy_continue_fabricated_failure"]["verified"] is False
    )
    assert (
        "explanation_contradicts_execution_evidence_in_reason"
        in by_scenario["healthy_continue_fabricated_failure"][
            "explanation_issues"
        ]
    )
    assert (
        "explanation_metric_not_observed_in_reason"
        in by_scenario["stop_invented_numbers"]["explanation_issues"]
    )
    assert (
        by_scenario["stop_with_unsupported_failure_excuse"]["verified"] is False
    )


def test_baseline_rejects_non_held_out_rows(tmp_path):
    from siliconshield.research_loop.episodes.operator_episodes import (
        SCENARIOS,
        episode_for,
    )

    scenario = SCENARIOS[0]
    row = episode_for(
        scenario[0],
        scenario[1],
        scenario[2],
        scenario[3],
        scenario[4],
        "train-x",
        0,
    )
    path = tmp_path / "other.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="held-out"):
        load_baseline(path)


def test_shared_history_marked_regression_only():
    from siliconshield.training.glimmer.dpo_pairs import (
        REGRESSION_ONLY_IDS,
        corrected_stop_pair,
        is_regression_only,
        same_action_pair,
        training_pairs,
        validate_pair,
    )

    assert REGRESSION_ONLY_IDS == {
        "dpo-stop-grounding-v1",
        "dpo-baseline-stop-supported",
    }
    overlap = corrected_stop_pair()
    assert validate_pair(overlap) == []
    assert is_regression_only(overlap) is True
    assert (
        overlap["provenance"]["overlap"] == "baseline regression_stop_supported"
    )
    # Same ID via same_action_pair is also regression-only by construction.
    from siliconshield.research_loop.episodes.operator_baseline import (
        BASELINE_SCENARIOS,
    )

    sc = next(
        s for s in BASELINE_SCENARIOS if s[0] == "regression_stop_supported"
    )
    state = dict(sc[1], run_id="run-baseline")
    res = dict(sc[2], run_id="run-baseline")
    dup = same_action_pair(
        "run-baseline",
        state,
        res,
        "stop",
        "Development accuracy fell from 82.41% at step 202 to 80.22% at step 302, while NLL "
        "increased from 0.0422 to 0.0533. Stop under the configured regression rule and preserve "
        "the selected checkpoint. No execution failure was observed.",
        "The execution layer is failing, so stop the run before more damage is done.",
        "dpo-baseline-stop-supported",
        "baseline regression_stop_supported",
    )
    assert validate_pair(dup) == []
    assert is_regression_only(dup) is True
    assert len(training_pairs([overlap, dup])) == 0
