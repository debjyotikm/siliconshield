"""Supervisor: handoff, agent protocol, commit checks, passes."""

import hashlib
import json
import stat
from pathlib import Path

import pytest

from siliconshield.research_loop.episodes.contracts import digest
from siliconshield.research_loop.execution import campaign_supervisor as sup
from siliconshield.research_loop.execution import laya_campaign
from siliconshield.training.laya import dataset_versions

POP_SHA = "ab" * 32
INF_DIGEST = "cd" * 32


def write(path, text="fixture"):
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def policy():
    return {
        "research": {
            "min_retention_accuracy": 0.9,
            "max_retention_false_safe_increase": 0,
            "require_no_missing_context_regression": True,
            "require_no_pair_regression": True,
            "min_new_coverage_gain_pp": 5.0,
        },
        "deployment": {
            "max_worst_family_false_safe_rate": 0.2,
            "min_class_recall": 0.5,
            "min_pair_accuracy": 0.5,
        },
    }


def view(n, correct, false_safe, violations, split):
    return {
        "population": {
            "dataset_version": "d1",
            "split": split,
            "sha256": POP_SHA,
            "n": n,
        },
        "inference": {"digest": INF_DIGEST},
        "counts": {
            "n": n,
            "correct": correct,
            "false_safe": false_safe,
            "violations": violations,
            "mc_correct": violations,
            "mc_total": violations,
            "pairs_correct": violations,
            "pairs_total": violations,
            "classes": [
                {"correct": correct // 3, "total": violations},
                {"correct": correct // 3, "total": violations},
                {"correct": correct // 3, "total": violations},
            ],
        },
        "worst_family": {
            "id": "w",
            "false_safe": false_safe,
            "violations": violations,
        },
    }


def results():
    return {
        "retention": view(60, 60, 1, 20, "dev-retention"),
        "new_coverage": view(30, 24, 3, 10, "dev-transfer"),
    }


def state_row(row_id, rtl):
    return {"id": row_id, "state": "Requirement\nok\nRTL\n" + rtl + "\n"}


def state_file(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def make_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    campaign = home / "campaign"
    export = tmp_path / "parent"
    export.mkdir()
    weights = write(export / "model.safetensors")
    recipe = home / "recipe.json"
    training = {"max_updates": 10}
    recipe_sha = write(recipe, json.dumps({"training": training}))
    data = home / "dataset"
    data.mkdir()
    state_file(data / "train.jsonl", [state_row("t1", "assign q = a;")])
    state_file(data / "dev.jsonl", [state_row("d1", "assign q = b;")])
    state_file(data / "calibration.jsonl", [state_row("c1", "assign q=c;")])
    state_file(data / "test.jsonl", [state_row("e1", "assign q = d;")])
    profile = {
        "model": str(export),
        "data": str(data),
        "runs": str(home / "runs"),
        "recipe": str(recipe),
        "defaults": dict(training),
    }
    (home / "profile.json").write_text(json.dumps(profile))
    evaluated = results()
    receipt = tmp_path / "receipt.json"
    content = {
        "run": {"experiment_id": "genesis-eval", "round_id": None},
        "export_manifest": laya_campaign.export_manifest(export),
        "results_sha256": digest(evaluated),
    }
    receipt.write_text(json.dumps(content, sort_keys=True))
    laya_campaign.init_campaign(
        campaign,
        {
            "campaign_id": "demo",
            "parent": {"export": str(export), "weights_sha256": weights},
            "recipe": {"path": str(recipe), "sha256": recipe_sha},
            "selection": policy(),
            "operator_profile": str(home / "profile.json"),
            "incumbent": {
                "export": str(export),
                "results": evaluated,
                "receipt": {
                    "path": str(receipt),
                    "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                },
            },
        },
    )
    store = home / "datasets"
    manifest = dataset_versions.write_version(
        store,
        "d1",
        {"train": str(data / "train.jsonl"), "dev": str(data / "dev.jsonl")},
    )
    from siliconshield.schemas import DATASET_AUDIT_SCHEMA

    dataset_versions.accept_version(
        store,
        "d1",
        {
            "schema": DATASET_AUDIT_SCHEMA,
            "splits": {
                role: entry["sha256"]
                for role, entry in manifest["splits"].items()
            },
            "cross_split_exact_rtl_overlap": 0,
            "retired_rtl_overlap": 0,
            "truncated": 0,
            "lineage_overlap": 0,
            "canonical_overlap": 0,
            "max_group_share": 0.05,
            "max_mechanism_share": 0.15,
            "review_ledger_sha256": "ab" * 32,
            "reviewed_templates": 1,
            "release_class": "narrow-pilot",
        },
    )
    manifest_path = store / "versions/d1/manifest.json"
    laya_campaign.register_dataset(
        campaign,
        "d1",
        str(manifest_path),
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    return home


def study_for(home, run_id, retired_rows):
    campaign = home / "campaign"
    data = home / "dataset"
    retired = home / ("retired-%s.jsonl" % run_id)
    state_file(retired, retired_rows)
    parent = Path(laya_campaign.load_incumbent(campaign)["export"])
    manifest = {
        "schema": "siliconshield-new-study-v1",
        "run_id": run_id,
        "retire_parent_evaluations": True,
        "parent_checkpoint": {
            str(p.relative_to(parent)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(parent.rglob("*"))
            if p.is_file()
        },
        "dataset": {
            role: hashlib.sha256(
                (data / (role + ".jsonl")).read_bytes()
            ).hexdigest()
            for role in ("train", "dev", "calibration", "test")
        },
        "retired_evaluations": [
            {
                "path": str(retired),
                "sha256": hashlib.sha256(retired.read_bytes()).hexdigest(),
                "role": "previous_test",
            }
        ],
    }
    path = home / ("study-%s.json" % run_id)
    path.write_text(json.dumps(manifest))
    return path


def config_for(home):
    return {
        "muse_bin": "muse",
        "model": "muse-spark-1.3-contributor",
        "repo": str(home),
        "retired_paths": [],
        "audit_extra_splits": [
            str(home / "dataset/calibration.jsonl"),
            str(home / "dataset/test.jsonl"),
        ],
    }


class FakeAgent:
    """Scripted agent runner recording every session."""

    def __init__(self):
        self.calls = []
        self.handlers = {}
        self.result = {
            "session": "session-1",
            "terminal": "completed",
            "exit_code": 0,
            "text": "DONE",
            "usage": {},
            "timed_out": False,
        }

    def on(self, kind, handler):
        self.handlers[kind] = handler
        return self

    def __call__(self, config, kind, prompt, workdir):
        self.calls.append(kind)
        if kind in self.handlers:
            self.handlers[kind](prompt)
        return dict(self.result)


def analysis_record(round_id="r1", next_action=None):
    return {
        "schema": sup.ANALYSIS_SCHEMA,
        "round_id": round_id,
        "verdict_summary": "candidate promoted",
        "what_failed": {"metrics": {"gain_pp": 6.0}, "artifacts": ["r"]},
        "explanation": "why",
        "proposed_change": "more data",
        "predicted_improvement": "+1pp",
        "possible_regressions": "none expected",
        "falsifier": "no gain",
        "next_action": next_action
        or {
            "type": "prepare_round",
            "round_id": "r2",
            "hypothesis": "contrasts fix the family",
            "dataset_plan": "generate contrasts",
        },
    }


def test_handoff_defaults_and_roundtrip(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    handoff = sup.load_handoff(home)
    assert handoff["mode"] == "observe"
    assert handoff["observed_terminal"] == []
    handoff["mode"] = "prepare"
    sup.save_handoff(home, handoff)
    assert sup.load_handoff(home)["mode"] == "prepare"
    (home / "campaign" / "supervisor" / "handoff.json").write_text("{}")
    with pytest.raises(ValueError, match="schema"):
        sup.load_handoff(home)


def test_check_analysis_validates_schema(tmp_path):
    record = analysis_record()
    assert sup.check_analysis(record, "r1")["round_id"] == "r1"
    with pytest.raises(ValueError, match="schema"):
        sup.check_analysis(dict(record, schema="other"), "r1")
    with pytest.raises(ValueError, match="another round"):
        sup.check_analysis(record, "r2")
    broken = dict(record)
    del broken["falsifier"]
    with pytest.raises(ValueError, match="falsifier"):
        sup.check_analysis(broken, "r1")
    broken = dict(record, next_action={"type": "unknown"})
    with pytest.raises(ValueError, match="next-action"):
        sup.check_analysis(broken, "r1")
    stop = analysis_record(next_action={"type": "stop", "reason": "done here"})
    assert sup.check_analysis(stop, "r1")["next_action"]["type"] == "stop"


def test_check_prep_receipt_validates_schema(tmp_path):
    record = {
        "schema": sup.PREP_SCHEMA,
        "round_id": "r2",
        "dataset_version": "d1",
        "manifest_path": "/tmp/manifest.json",
        "recipe_path": None,
        "study_path": "/tmp/study.json",
        "hypothesis": "contrasts fix the family",
        "selection": None,
        "qualification": "/tmp/qual.json",
    }
    assert sup.check_prep_receipt(record, "r2")["dataset_version"] == "d1"
    with pytest.raises(ValueError, match="schema"):
        sup.check_prep_receipt(dict(record, schema="other"), "r2")
    with pytest.raises(ValueError, match="another round"):
        sup.check_prep_receipt(record, "r3")
    broken = dict(record)
    del broken["study_path"]
    with pytest.raises(ValueError, match="study_path"):
        sup.check_prep_receipt(broken, "r2")


def test_parse_exec_jsonl_extracts_terminal():
    transcript = "\n".join(
        [
            json.dumps(
                {
                    "stream": {"kind": "session", "id": "s-1"},
                    "payload_type": "event",
                    "payload": {},
                }
            ),
            json.dumps(
                {
                    "stream": {"kind": "exec.usage", "id": "s-1"},
                    "payload_type": "run.usage",
                    "payload": {
                        "calls": 2,
                        "models": [{"input_tokens": 10, "output_tokens": 5}],
                    },
                }
            ),
            json.dumps(
                {
                    "stream": {"kind": "session", "id": "s-1"},
                    "payload_type": "run.terminal.completed",
                    "payload": {"terminal": "completed", "text": "DONE"},
                }
            ),
        ]
    )
    parsed = sup.parse_exec_jsonl(transcript)
    assert parsed["session"] == "s-1"
    assert parsed["terminal"] == "completed"
    assert parsed["usage"] == {
        "calls": 2,
        "input_tokens": 10,
        "output_tokens": 5,
    }
    with pytest.raises(ValueError, match="terminal record"):
        sup.parse_exec_jsonl('{"stream": {}}')


def test_agent_argv_carries_model_and_workspace(tmp_path):
    argv = sup.agent_argv(config_for(tmp_path), "analyze", "do analysis")
    assert argv[:5] == [
        "muse",
        "exec",
        "--model",
        "muse-spark-1.3-contributor",
        "--json",
    ]
    assert "--disable-sandbox" not in argv
    assert "--workspace" in argv and argv[-1] == "do analysis"


def test_agent_argv_unsandboxed_only_when_explicit(tmp_path):
    config = config_for(tmp_path)
    config["allow_unsandboxed_agent"] = True
    assert "--disable-sandbox" in sup.agent_argv(
        config, "analyze", "do analysis"
    )


def test_run_muse_session_uses_stub_and_timeout(tmp_path):
    stub = tmp_path / "muse"
    stub.write_text(
        '#!/bin/sh\necho \'{"stream": {"kind": "session", "id": "s"},'
        ' "payload_type": "run.terminal.completed",'
        ' "payload": {"terminal": "completed", "text": "DONE"}}\'\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    config = config_for(tmp_path)
    config["muse_bin"] = str(stub)
    result = sup.run_muse_session(config, "plan", "hello", tmp_path)
    assert result["terminal"] == "completed"
    assert result["exit_code"] == 0
    sleeper = tmp_path / "sleepy"
    sleeper.write_text("#!/bin/sh\nsleep 30\n")
    sleeper.chmod(sleeper.stat().st_mode | stat.S_IEXEC)
    config["muse_bin"] = str(sleeper)
    config["session_timeout_s"] = {"plan": 1}
    slow = sup.run_muse_session(config, "plan", "hello", tmp_path)
    assert slow["timed_out"] is True
    assert slow["terminal"] == "timeout"


def test_prompt_notes_reach_agent_sessions(tmp_path, monkeypatch):
    home = make_home(tmp_path)
    monkeypatch.setattr(
        laya_campaign, "run_round", lambda *a, **k: {"phase": "observe"}
    )
    seen = {}

    def capture(config, kind, prompt, workdir):
        seen["prompt"] = prompt
        return {
            "session": "s",
            "terminal": "completed",
            "exit_code": 0,
            "text": "DONE",
            "usage": {},
            "timed_out": False,
        }

    (home / "campaign" / "supervisor").mkdir(parents=True, exist_ok=True)
    (home / "campaign" / "supervisor" / "plan-proposal.json").write_text(
        json.dumps({"type": "stop", "reason": "rest"})
    )
    config = config_for(home)
    config["prompt_notes"] = "Keep populations_for stable."
    sup.supervise_once(
        home,
        config,
        operator_factory=fake_operator_factory,
        evaluator=fake_evaluator,
        agent_runner=capture,
    )
    assert "Keep populations_for stable." in seen["prompt"]


def test_build_prompt_covers_kinds(tmp_path):
    context = {
        "home": "/h",
        "repo": "/r",
        "round_id": "r1",
        "phase": "done",
        "receipt": "/h/r.json",
        "incumbent": "/h/i.json",
        "report": "/h/report.json",
        "leaderboard": "/h/board.jsonl",
        "proposal": "/h/p.json",
        "handoff": "/h/h.json",
        "failure": "boom",
        "output_path": "/h/out.json",
    }
    for kind in ("analyze", "prepare", "recover", "plan"):
        prompt = sup.build_prompt(kind, context)
        assert "must not rewrite verification outcomes" in prompt
    assert "/h/out.json" in sup.build_prompt("analyze", context)
    with pytest.raises(ValueError, match="Unknown agent"):
        sup.build_prompt("nap", context)


def prep_receipt_for(home, round_id="r2", hypothesis=None):
    campaign = home / "campaign"
    manifest = campaign / "datasets.json"
    entry = json.loads(manifest.read_text())["versions"]["d1"]
    return {
        "schema": sup.PREP_SCHEMA,
        "round_id": round_id,
        "dataset_version": "d1",
        "manifest_path": entry["manifest"],
        "recipe_path": None,
        "study_path": str(
            study_for(home, "demo-" + round_id, [state_row("o1", "q=z;")])
        ),
        "hypothesis": hypothesis or "contrasts fix the family",
        "selection": None,
        "qualification": str(home / "qual.json"),
    }


def fake_operator_factory(path):
    raise AssertionError("no operator in supervisor tests")


def fake_evaluator(root, round_id, reported):
    raise AssertionError("no evaluator in supervisor tests")


def test_plan_then_prepare_then_commit(tmp_path, monkeypatch):
    home = make_home(tmp_path)
    campaign = home / "campaign"
    (home / "qual.json").write_text("{}")
    monkeypatch.setattr(
        laya_campaign, "run_round", lambda *a, **k: {"phase": "observe"}
    )
    agent = FakeAgent()

    def plan(prompt):
        (home / "campaign" / "supervisor" / "plan-proposal.json").write_text(
            json.dumps(
                {
                    "type": "prepare_round",
                    "round_id": "r2",
                    "hypothesis": "contrasts fix the family",
                    "dataset_plan": "generate contrasts",
                }
            )
        )

    def prepare(prompt):
        (campaign / "rounds" / "r2").mkdir(parents=True, exist_ok=True)
        (campaign / "rounds" / "r2" / "prep-receipt.json").write_text(
            json.dumps(prep_receipt_for(home))
        )

    agent.on("plan", plan).on("prepare", prepare)
    status = sup.supervise_once(
        home,
        config_for(home),
        operator_factory=fake_operator_factory,
        evaluator=fake_evaluator,
        agent_runner=agent,
    )
    assert agent.calls == ["plan"]
    assert status["action"] == "planned"
    assert status["next_action"].startswith("prepare round r2")
    status = sup.supervise_once(
        home,
        config_for(home),
        operator_factory=fake_operator_factory,
        evaluator=fake_evaluator,
        agent_runner=agent,
    )
    assert agent.calls == ["plan", "prepare"]
    assert status["action"] == "committed"
    assert status["rounds"] == {"r2": "prepare"}
    assert status["active_round"] == "r2"
    profile = json.loads((home / "profile.json").read_text())
    bound = laya_campaign.read_round(campaign, "r2")["bindings"]["round_recipe"]
    assert profile["recipe"] == bound["path"]
    assert "selection_reference" in json.loads(Path(bound["path"]).read_text())
    assert (
        (home / "STATUS.md")
        .read_text()
        .startswith("# Campaign supervisor status")
    )


def test_analyze_terminal_round_records_proposal(tmp_path, monkeypatch):
    home = make_home(tmp_path)
    campaign = home / "campaign"
    laya_campaign.prepare_round(campaign, "r1", "d1", "first")
    laya_campaign.advance(campaign, "r1", "failed", reason="fixture")
    monkeypatch.setattr(
        laya_campaign, "run_round", lambda *a, **k: {"phase": "failed"}
    )
    agent = FakeAgent()

    def analyze(prompt):
        (campaign / "rounds" / "r1" / "analysis.json").write_text(
            json.dumps(analysis_record())
        )

    agent.on("analyze", analyze)
    status = sup.supervise_once(
        home,
        config_for(home),
        operator_factory=fake_operator_factory,
        evaluator=fake_evaluator,
        agent_runner=agent,
    )
    assert agent.calls == ["analyze"]
    assert status["action"] == "analyzed"
    assert status["last_analysis"]["round_id"] == "r1"
    assert (campaign / "leaderboard.jsonl").is_file()
    handoff = sup.load_handoff(home)
    assert handoff["observed_terminal"] == ["r1"]
    assert handoff["pending_proposal"]["round_id"] == "r2"


def test_commit_refuses_defective_prep(tmp_path):
    home = make_home(tmp_path)
    campaign = home / "campaign"
    handoff = sup.default_handoff()
    handoff["pending_proposal"] = {
        "type": "prepare_round",
        "round_id": "r2",
        "hypothesis": "contrasts fix the family",
        "dataset_plan": "generate contrasts",
        "selection": None,
        "recipe_plan": None,
        "overrides": {},
    }
    handoff["proposal_round"] = "r2"
    config = config_for(home)
    receipt = prep_receipt_for(home, hypothesis="another idea")
    with pytest.raises(sup.PrepCommitError, match="hypothesis"):
        sup.commit_prep(home, handoff, receipt, config)
    laya_campaign.prepare_round(campaign, "r9", "d1", "blocking")
    with pytest.raises(sup.PrepCommitError, match="still active"):
        sup.commit_prep(home, handoff, prep_receipt_for(home), config)


def test_commit_refuses_overlap_and_unaccepted(tmp_path):
    home = make_home(tmp_path)
    data = home / "dataset"
    leaked = state_file(
        data / "train2.jsonl", [state_row("t9", "assign q = b;")]
    )
    store = home / "datasets"
    manifest = dataset_versions.write_version(
        store, "dx", {"train": str(leaked), "dev": str(data / "dev.jsonl")}
    )
    from siliconshield.schemas import DATASET_AUDIT_SCHEMA

    dataset_versions.accept_version(
        store,
        "dx",
        {
            "schema": DATASET_AUDIT_SCHEMA,
            "splits": {
                role: entry["sha256"]
                for role, entry in manifest["splits"].items()
            },
            "cross_split_exact_rtl_overlap": 0,
            "retired_rtl_overlap": 0,
            "truncated": 0,
            "lineage_overlap": 0,
            "canonical_overlap": 0,
            "max_group_share": 0.05,
            "max_mechanism_share": 0.15,
            "review_ledger_sha256": "ab" * 32,
            "reviewed_templates": 1,
            "release_class": "narrow-pilot",
        },
    )
    handoff = sup.default_handoff()
    handoff["pending_proposal"] = {
        "type": "prepare_round",
        "round_id": "r2",
        "hypothesis": "contrasts fix the family",
        "dataset_plan": "generate contrasts",
        "selection": None,
        "recipe_plan": None,
        "overrides": {},
    }
    handoff["proposal_round"] = "r2"
    receipt = prep_receipt_for(home)
    receipt["dataset_version"] = "dx"
    receipt["manifest_path"] = str(store / "versions/dx/manifest.json")
    with pytest.raises(sup.PrepCommitError, match="overlap"):
        sup.commit_prep(home, handoff, receipt, config_for(home))
    assert laya_campaign.list_rounds(home / "campaign") == []


def test_supervisor_parks_after_repeated_failures(tmp_path, monkeypatch):
    home = make_home(tmp_path)
    monkeypatch.setattr(
        laya_campaign, "run_round", lambda *a, **k: {"phase": "observe"}
    )
    agent = FakeAgent()
    agent.result = {
        "session": None,
        "terminal": "timeout",
        "exit_code": None,
        "text": "",
        "usage": {},
        "timed_out": True,
    }
    for _ in range(3):
        status = sup.supervise_once(
            home,
            config_for(home),
            operator_factory=fake_operator_factory,
            evaluator=fake_evaluator,
            agent_runner=agent,
        )
    assert status["action"] == "plan_failed"
    assert status["mode"] == "parked"
    assert "parked" in status["gpu_note"]
    calls = len(agent.calls)
    sup.supervise_once(
        home,
        config_for(home),
        operator_factory=fake_operator_factory,
        evaluator=fake_evaluator,
        agent_runner=agent,
    )
    assert len(agent.calls) == calls


def test_recover_event_runs_on_reconcile_error(tmp_path, monkeypatch):
    home = make_home(tmp_path)
    campaign = home / "campaign"
    laya_campaign.prepare_round(campaign, "r1", "d1", "first")

    def explode(*args, **kwargs):
        raise RuntimeError("gpu exploded")

    monkeypatch.setattr(laya_campaign, "run_round", explode)
    agent = FakeAgent()
    status = sup.supervise_once(
        home,
        config_for(home),
        operator_factory=fake_operator_factory,
        evaluator=fake_evaluator,
        agent_runner=agent,
    )
    assert agent.calls == ["recover"]
    assert status["action"] == "recovered"
    assert "gpu exploded" in status["recovery"]


def test_reference_results_thread_to_runner(tmp_path, monkeypatch):
    home = make_home(tmp_path)
    campaign = home / "campaign"
    laya_campaign.prepare_round(campaign, "r1", "d1", "first")
    (campaign / "permanent_reference.json").write_text(
        json.dumps({"results": results()})
    )
    seen = {}

    def capture(root, round_id, **kwargs):
        seen.update(kwargs)
        return {"phase": "observe"}

    monkeypatch.setattr(laya_campaign, "run_round", capture)
    sup.supervise_once(
        home,
        config_for(home),
        operator_factory=fake_operator_factory,
        evaluator=fake_evaluator,
        agent_runner=FakeAgent(),
    )
    assert seen["reference_results"]["retention"]["counts"]["n"] == 60


def test_gpu_note_names_preflight_waits(tmp_path, monkeypatch):
    home = make_home(tmp_path)
    campaign = home / "campaign"
    laya_campaign.prepare_round(campaign, "r1", "d1", "first")
    laya_campaign.advance(campaign, "r1", "validate_data")
    laya_campaign.advance(campaign, "r1", "preflight")
    monkeypatch.setattr(
        laya_campaign, "run_round", lambda *a, **k: {"phase": "preflight"}
    )
    status = sup.supervise_once(
        home,
        config_for(home),
        operator_factory=fake_operator_factory,
        evaluator=fake_evaluator,
        agent_runner=FakeAgent(),
    )
    assert status["gpu_note"] == "round r1 waiting for free GPUs"
    assert status["next_action"] == "drive round r1 (preflight)"
