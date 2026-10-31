"""Synthetic transfer episodes retain campaign lineage and decision-point evidence."""

import copy
import json
import tempfile

import pytest

from siliconshield.research_loop.episodes.transfer_episodes import (
    CAMPAIGN_FAMILY,
    SCENARIOS,
    build_transfer,
    load_transfer,
)


def _fixture_client(row):
    class TraceClient:
        model = "transfer-fixture"
        endpoint = "http://fixture.invalid"

        def complete(self, messages, tools=()):
            return copy.deepcopy(row["messages"][len(messages)])

    return TraceClient()


def test_transfer_episodes_form_one_reviewed_campaign(tmp_path):
    from siliconshield.schemas import TRANSFER_EPISODES_SCHEMA

    manifest = build_transfer(tmp_path / "transfer")
    assert manifest["rows"] == 4
    assert manifest["schema"] == TRANSFER_EPISODES_SCHEMA
    assert manifest["campaign_family"] == CAMPAIGN_FAMILY
    assert manifest["development_only"] is True
    rows = load_transfer(tmp_path / "transfer/transfer.jsonl")
    assert (
        {r["group_id"] for r in rows}
        == {CAMPAIGN_FAMILY}
        == set(manifest["lineages"])
    )
    for row in rows:
        assert row["verification"].get("campaign_family") == CAMPAIGN_FAMILY


def test_transfer_episodes_verify_with_supported_reasons(tmp_path):
    from siliconshield.research_loop.controller.replay import execute, grade

    build_transfer(tmp_path / "transfer")
    for row in load_transfer(tmp_path / "transfer/transfer.jsonl"):
        with tempfile.TemporaryDirectory() as attempt:
            delivered = execute(row, _fixture_client(row), attempt)
        result = grade(row, delivered)
        assert result["action_correct"], row["verification"]["scenario"]
        assert result["explanation_correct"], (
            row["verification"]["scenario"],
            result.get("explanation_issues"),
        )
        assert result["verified"], row["verification"]["scenario"]


def test_transfer_loader_rejects_other_lineages(tmp_path):
    from siliconshield.research_loop.episodes.operator_episodes import (
        episode_for,
    )

    scenario = SCENARIOS[0]
    row = episode_for(
        scenario[0],
        scenario[1],
        scenario[2],
        scenario[3],
        scenario[4],
        "other-lineage",
        0,
    )
    path = tmp_path / "other.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="campaign lineages"):
        load_transfer(path)


def test_transfer_episodes_carry_decision_point_evidence(tmp_path):
    build_transfer(tmp_path / "transfer")
    rows = load_transfer(tmp_path / "transfer/transfer.jsonl")
    by_scenario = {r["verification"]["scenario"]: r for r in rows}
    stop = by_scenario["transfer_regression_stop"]["summary"]
    assert stop["acceptance"] == {
        "accepted": False,
        "reasons": ["false_safe_budget_exceeded"],
        "thresholds": {"max_worst_family_false_safe_rate": 0.1},
    }
    assert stop["selected_step"] == 2
    assert stop["schedule"]["plateau_lr_reductions"] == []
    plateau = by_scenario["transfer_plateau_no_intervention"]["summary"]
    assert plateau["schedule"]["stale"] == 2
    assert plateau["schedule"]["lr_scale"] == 1.0


def test_external_catalog_controls_observations_and_family(tmp_path):
    from siliconshield.research_loop.episodes.transfer_episodes import (
        load_catalog,
    )

    catalog = load_catalog()
    catalog["campaign_family"] = "external-fixture"
    catalog["scenarios"][0]["state"]["catalog_marker"] = "supplied-observation"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    manifest = build_transfer(tmp_path / "output", path)
    rows = load_transfer(tmp_path / "output/transfer.jsonl")
    assert manifest["lineages"] == ["external-fixture"]
    assert rows[0]["summary"]["catalog_marker"] == "supplied-observation"
    with pytest.raises(ValueError, match="campaign lineages"):
        load_transfer(
            tmp_path / "output/transfer.jsonl", allowed_prefix="another-"
        )


def test_invalid_catalog_rejected_before_creating_output(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"schema": "unknown"}))
    with pytest.raises(ValueError, match="catalog schema"):
        build_transfer(tmp_path / "output", path)
    assert not (tmp_path / "output").exists()
