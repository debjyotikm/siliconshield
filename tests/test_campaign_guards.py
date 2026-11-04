"""Campaign policy, round, and study-guard tests: synthetic fixtures."""

import copy
import json

from siliconshield.research_loop.execution import (
    campaign_supervisor as supervisor,
)
from siliconshield.research_loop.execution import laya_campaign as campaign
from siliconshield.research_loop.execution.campaign_supervisor import (
    PrepCommitError,
)
from siliconshield.schemas import (
    CAMPAIGN_ROUND_SCHEMA,
    LAYA_CAMPAIGN_SCHEMA,
)


def _policy():
    return {
        "research": {
            "min_retention_accuracy": 0.9,
            "max_retention_false_safe_increase": 0,
            "require_no_missing_context_regression": True,
            "require_no_pair_regression": True,
            "min_new_coverage_gain_pp": 0.0,
        },
        "deployment": {
            "max_worst_family_false_safe_rate": 0.1,
            "min_class_recall": 0.5,
            "min_pair_accuracy": 0.5,
        },
    }


def test_policy_checker_rejects_malformed_policies():
    import pytest

    campaign._check_policy(_policy())
    for bad in (None, ["x"], {"research": 5, "deployment": {}}):
        with pytest.raises(ValueError, match="must be an object"):
            campaign._check_policy(bad)
    bad = _policy()
    bad["research"]["min_retention_accuracy"] = "high"
    with pytest.raises(ValueError, match="retention floor"):
        campaign._check_policy(bad)
    bad = _policy()
    bad["research"]["require_no_pair_regression"] = "yes"
    with pytest.raises(ValueError, match="require_no_pair_regression"):
        campaign._check_policy(bad)
    bad = _policy()
    bad["deployment"]["min_class_recall"] = [1]
    with pytest.raises(ValueError, match="min_class_recall"):
        campaign._check_policy(bad)


def test_relaxes_fails_closed_on_uncomparable_policies():
    base = _policy()
    assert campaign.selection_relaxes(base, copy.deepcopy(base)) is False
    assert campaign.selection_relaxes(None, base) is True
    bad = copy.deepcopy(base)
    bad["research"]["min_retention_accuracy"] = "high"
    assert campaign.selection_relaxes(base, bad) is True


def test_round_accessors_reject_malformed_receipts(tmp_path):
    import pytest

    root = tmp_path / "campaign"
    (root / "rounds" / "r1").mkdir(parents=True)
    (root / "rounds" / "r1" / "round.json").write_text(
        json.dumps({"schema": CAMPAIGN_ROUND_SCHEMA})
    )
    with pytest.raises(ValueError, match="needs a phase"):
        campaign.round_phase(root, "r1")
    with pytest.raises(ValueError, match="Round id"):
        campaign.read_round(root, 5)


def test_selection_update_refuses_malformed_state(tmp_path):
    import pytest

    root = tmp_path / "campaign"
    (root / "rounds" / "r1").mkdir(parents=True)
    (root / "campaign.json").write_text(
        json.dumps({"schema": LAYA_CAMPAIGN_SCHEMA, "selection": _policy()})
    )
    (root / "rounds" / "r1" / "round.json").write_text(
        json.dumps({"schema": CAMPAIGN_ROUND_SCHEMA})
    )
    with pytest.raises(ValueError, match="needs a phase"):
        campaign.update_campaign_selection(root, _policy(), "test")
    (root / "campaign.json").write_text(
        json.dumps({"schema": LAYA_CAMPAIGN_SCHEMA})
    )
    with pytest.raises(ValueError, match="malformed"):
        campaign.update_campaign_selection(root, _policy(), "test")


def test_study_round_budget_guard_rejects_malformed_files(tmp_path):
    import pytest

    home = tmp_path / "home"
    study = home / "campaign" / "studies" / "s"
    study.mkdir(parents=True)
    (study / "study.json").write_text("{}")
    with pytest.raises(ValueError, match="round budget"):
        supervisor._study_max_rounds(home, "s")
    with pytest.raises(ValueError, match="missing"):
        supervisor._study_max_rounds(home, "nope")


def test_study_manifest_rejects_non_canonical_payloads():
    import pytest

    from siliconshield.training.laya import decision_study as studies

    with pytest.raises(ValueError, match="JSON-canonical"):
        studies.verify_study_manifest(
            {"schema": studies.STUDY_MANIFEST_SCHEMA, "x": {1, 2}}
        )


def test_agent_launch_rejects_malformed_config():
    import pytest

    config = {"muse_bin": "muse", "model": "m", "repo": "/tmp"}
    argv = supervisor.agent_argv(config, "analyze", "prompt")
    assert argv[:4] == ["muse", "exec", "--model", "m"]
    with pytest.raises(ValueError, match="must be an object"):
        supervisor.agent_argv(["x"], "analyze", "p")
    with pytest.raises(ValueError, match="needs muse_bin"):
        supervisor.agent_argv({"model": "m", "repo": "r"}, "analyze", "p")
    with pytest.raises(ValueError, match="Unknown agent session kind"):
        supervisor.agent_argv(config, "dance", "p")
    with pytest.raises(ValueError, match="must be objects"):
        supervisor.agent_argv(
            dict(config, max_model_steps=["x"]), "analyze", "p"
        )
    with pytest.raises(ValueError, match="string list"):
        supervisor.agent_argv(dict(config, extra_args="--x"), "analyze", "p")
    with pytest.raises(ValueError, match="must be a string"):
        supervisor.agent_argv(config, "analyze", 5)


def test_agent_session_rejects_malformed_timeouts(tmp_path):
    import pytest

    config = {"muse_bin": "muse", "model": "m", "repo": "/tmp"}
    with pytest.raises(ValueError, match="must be a number"):
        supervisor.run_muse_session(
            dict(config, session_timeout_s={"analyze": "x"}),
            "analyze",
            "p",
            tmp_path,
        )
    with pytest.raises(ValueError, match="must be an object"):
        supervisor.run_muse_session(
            dict(config, session_timeout_s=["x"]), "analyze", "p", tmp_path
        )
    with pytest.raises(ValueError, match="must be positive"):
        supervisor.run_muse_session(
            dict(config, session_timeout_s={"analyze": 0}),
            "analyze",
            "p",
            tmp_path,
        )


def test_transcript_skips_non_string_session_ids():
    import json

    lines = [
        {"stream": {"kind": "session", "id": 5}, "payload": {}},
        {
            "stream": {"kind": "other"},
            "payload": {"terminal": "done", "text": "hi"},
            "payload_type": "run.terminal.done",
        },
    ]
    parsed = supervisor.parse_exec_jsonl(
        "\n".join(json.dumps(line) for line in lines)
    )
    assert parsed["session"] is None
    assert parsed["terminal"] == "done"


def _commit_fixture(tmp_path, phase="done"):
    home = tmp_path / "home"
    rounds = home / "campaign" / "rounds" / "r0"
    rounds.mkdir(parents=True)
    receipt = {"schema": CAMPAIGN_ROUND_SCHEMA}
    if phase is not None:
        receipt["phase"] = phase
    (rounds / "round.json").write_text(json.dumps(receipt))
    handoff = {
        "pending_proposal": {
            "round_id": "r1",
            "hypothesis": "h",
            "selection": None,
        }
    }
    prep = {
        "round_id": "r1",
        "hypothesis": "h",
        "selection": None,
        "dataset_version": "v",
        "manifest_path": str(tmp_path / "store" / "x" / "y" / "m.json"),
        "study_path": str(tmp_path / "study.json"),
        "qualification": "q",
    }
    return home, handoff, prep


def test_commit_prep_refuses_unreadable_state(tmp_path):
    import pytest

    home, handoff, prep = _commit_fixture(tmp_path, phase=None)
    with pytest.raises(PrepCommitError, match="unreadable"):
        supervisor.commit_prep(home, handoff, prep, {})


def test_commit_prep_rejects_malformed_config(tmp_path):
    import pytest

    home, handoff, prep = _commit_fixture(tmp_path)
    with pytest.raises(PrepCommitError, match="must be an object"):
        supervisor.commit_prep(home, handoff, prep, ["x"])
    with pytest.raises(PrepCommitError, match="needs audit_extra_splits"):
        supervisor.commit_prep(home, handoff, prep, {"audit_extra_splits": "x"})


def _campaign_root(tmp_path):
    root = tmp_path / "campaign"
    (root / "rounds" / "r1").mkdir(parents=True)
    (root / "campaign.json").write_text(
        json.dumps(
            {
                "schema": LAYA_CAMPAIGN_SCHEMA,
                "recipe": {"path": "x", "sha256": "y"},
            }
        )
    )
    return root


def test_prepare_round_rejects_malformed_inputs(tmp_path):
    import pytest

    root = _campaign_root(tmp_path)
    with pytest.raises(ValueError, match="stable identifier"):
        campaign.prepare_round(root, 5, "v", "h")
    with pytest.raises(ValueError, match="stable identifier"):
        campaign.prepare_round(root, "../x", "v", "h")
    with pytest.raises(ValueError, match="recorded hypothesis"):
        campaign.prepare_round(root, "r2", "v", 5)


def test_advance_rejects_malformed_transitions(tmp_path):
    import pytest

    root = _campaign_root(tmp_path)
    (root / "rounds" / "r1" / "round.json").write_text(
        json.dumps({"schema": CAMPAIGN_ROUND_SCHEMA})
    )
    with pytest.raises(ValueError, match="unknown phase"):
        campaign.advance(root, "r1", "failed")
    with pytest.raises(ValueError, match="string ids"):
        campaign.advance(root, 5, "failed")
    (root / "rounds" / "r1" / "round.json").write_text(
        json.dumps(
            {
                "schema": CAMPAIGN_ROUND_SCHEMA,
                "phase": "observe",
                "history": "x",
            }
        )
    )
    with pytest.raises(ValueError, match="history is malformed"):
        campaign.advance(root, "r1", "failed")


def test_recipe_update_rejects_malformed_inputs(tmp_path):
    import pytest

    root = _campaign_root(tmp_path)
    with pytest.raises(ValueError, match="path and hash"):
        campaign.update_campaign_recipe(root, 5, "ab" * 32, "test")
    with pytest.raises(ValueError, match="file is missing"):
        campaign.update_campaign_recipe(
            root, str(tmp_path / "no.json"), "ab" * 32, "test"
        )


def test_leaderboard_rejects_malformed_candidates(tmp_path):
    import pytest

    root = _campaign_root(tmp_path)
    (root / "rounds" / "r1" / "round.json").write_text(
        json.dumps(
            {
                "schema": CAMPAIGN_ROUND_SCHEMA,
                "phase": "done",
                "candidate": ["x"],
            }
        )
    )
    with pytest.raises(ValueError, match="malformed"):
        campaign.record_leaderboard_entry(root, "r1")


def test_candidate_attach_rejects_malformed_exports(tmp_path):
    import pytest

    root = _campaign_root(tmp_path)
    (root / "rounds" / "r1" / "round.json").write_text(
        json.dumps({"schema": CAMPAIGN_ROUND_SCHEMA, "phase": "observe"})
    )
    with pytest.raises(ValueError, match="must be a path"):
        campaign.record_candidate(root, "r1", 5, {}, {})


def _handoff_file(tmp_path, **overrides):
    from siliconshield.research_loop.execution.campaign_supervisor import (
        HANDOFF_SCHEMA,
    )

    home = tmp_path / "home"
    path = home / "campaign" / "supervisor" / "handoff.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(supervisor.default_handoff(), **overrides)
    record["schema"] = HANDOFF_SCHEMA
    path.write_text(json.dumps(record))
    return home


def test_load_handoff_rejects_malformed_fields(tmp_path):
    import pytest

    home = _handoff_file(tmp_path)
    assert supervisor.load_handoff(home)["mode"] == "observe"
    for field, value in (
        ("prep_attempts", "x"),
        ("plan_attempts", -1),
        ("consecutive_agent_failures", True),
        ("observed_terminal", "r1"),
        ("mode", 5),
        ("pending_proposal", ["x"]),
        ("last_analysis", ["x"]),
        ("parked_reason", 5),
    ):
        home = _handoff_file(tmp_path, **{field: value})
        with pytest.raises(ValueError):
            supervisor.load_handoff(home)


def test_status_helpers_tolerate_malformed_records(tmp_path):
    assert supervisor._summarize_counts(["x"]) == "(no candidate)"
    assert (
        supervisor._summarize_counts({"new_coverage": ["x"]})
        == "(no candidate)"
    )
    home = tmp_path / "home"
    (home / "campaign").mkdir(parents=True)
    handoff = dict(
        supervisor.default_handoff(),
        active_round="r1",
        last_analysis={"round_id": "r0", "next_action": "prepare_round"},
    )
    status = supervisor.build_status(home, handoff, {"r1": "done"}, "done")
    assert status["candidate"] is None
    assert status["latest_results"] == "(no candidate)"


def test_agent_event_records_launcher_failures(tmp_path):
    home = tmp_path / "home"
    (home / "supervisor").mkdir(parents=True)
    result = supervisor._run_agent_event(
        home, {"repo": "/tmp"}, "analyze", "p", "r1", lambda *a: ["x"]
    )
    assert result["terminal"] == "launcher_error"
    result = supervisor._run_agent_event(
        home, {}, "analyze", "p", "r1", lambda *a: {}
    )
    assert result["terminal"] == "launcher_error"


def test_supervise_once_rejects_malformed_config(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="must be an object"):
        supervisor.supervise_once(
            tmp_path,
            ["x"],
            operator_factory=lambda p: None,
            evaluator=lambda *a: {},
        )


def test_register_dataset_rejects_malformed_inputs(tmp_path):
    import pytest

    root = tmp_path / "campaign"
    root.mkdir()
    (root / "datasets.json").write_text(json.dumps({"versions": {}}))
    with pytest.raises(ValueError, match="version and hash"):
        campaign.register_dataset(root, 5, "m", "ab" * 32)
    with pytest.raises(ValueError, match="manifest path"):
        campaign.register_dataset(root, "v", 5, "ab" * 32)
    with pytest.raises(ValueError, match="is missing"):
        campaign.register_dataset(
            root, "v", str(tmp_path / "no.json"), "ab" * 32
        )
    (root / "datasets.json").write_text(json.dumps({"versions": ["x"]}))
    manifest = tmp_path / "m.json"
    manifest.write_text("{}")
    digest = campaign._sha(manifest)
    with pytest.raises(ValueError, match="registry is malformed"):
        campaign.register_dataset(root, "v", str(manifest), digest)


def test_revalidate_inputs_rejects_malformed_blocks(tmp_path):
    import pytest

    root = tmp_path / "campaign"
    root.mkdir()
    (root / "datasets.json").write_text(json.dumps({"versions": {}}))
    with pytest.raises(ValueError, match="parent block"):
        campaign._revalidate_inputs(root, {"parent": ["x"]}, "v")
    with pytest.raises(ValueError, match="is missing"):
        campaign._revalidate_inputs(
            root,
            {"parent": {"export": str(tmp_path), "weights_sha256": "x"}},
            "v",
        )


def test_systemd_probe_tolerates_malformed_entries():
    assert campaign._systemd_active(["x"]) is None
    assert campaign._systemd_active([["x"], {}]) is None
    assert campaign._systemd_active([{"systemd": {"state": "running"}}]) is True
    assert campaign._systemd_active([{"systemd": {"state": "dead"}}]) is False


def test_decision_helpers_reject_malformed_records():
    import pytest

    with pytest.raises(ValueError, match="malformed"):
        campaign._require_unmoved_incumbent(["x"], {"export": "e"}, "r1")
    with pytest.raises(ValueError, match="unreadable"):
        campaign._require_retry_consistency(
            {"export": "e"}, {"candidate": ["x"]}, "r1"
        )
    with pytest.raises(ValueError, match="must be an object"):
        campaign._frozen_policy("/tmp", ["x"])
    with pytest.raises(ValueError, match="malformed"):
        campaign._revalidate_candidate("/tmp", "r1", ["x"])
    with pytest.raises(ValueError, match="receipt and profile"):
        campaign._bound_file(["x"], "recipe", "recipe")
    with pytest.raises(ValueError, match="lacks a bound"):
        campaign._bound_file({"recipe": 5}, "recipe", "recipe")
    with pytest.raises(ValueError, match="receipt and profile"):
        campaign._require_profile_bindings(["x"], {})


def test_run_round_rejects_malformed_setup(tmp_path):
    import pytest

    root = tmp_path / "campaign"
    root.mkdir()
    (root / "campaign.json").write_text(
        json.dumps({"schema": LAYA_CAMPAIGN_SCHEMA})
    )
    with pytest.raises(ValueError, match="operator profile"):
        campaign.run_round(
            root,
            "r1",
            operator_factory=lambda p: None,
            evaluator=lambda *a: {},
        )
    (root / "campaign.json").write_text(
        json.dumps(
            {
                "schema": LAYA_CAMPAIGN_SCHEMA,
                "campaign_id": "c",
                "operator_profile": "p",
            }
        )
    )
    with pytest.raises(ValueError, match="step budget"):
        campaign.run_round(
            root,
            "r1",
            operator_factory=lambda p: None,
            evaluator=lambda *a: {},
            max_steps=0,
        )


def test_close_round_rejects_phaseless_receipts(tmp_path):
    import pytest

    root = tmp_path / "campaign"
    (root / "rounds" / "r1").mkdir(parents=True)
    (root / "rounds" / "r1" / "round.json").write_text(
        json.dumps({"schema": CAMPAIGN_ROUND_SCHEMA})
    )
    with pytest.raises(ValueError, match="follows decision"):
        campaign.close_round(root, "r1", "summary")
