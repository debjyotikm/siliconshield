"""Run real Harness2 recursion and grading with authored, offline completions."""

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("harness2")
from harness2.core.proposer import Reply

from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    episode_from_summary,
)
from siliconshield.research_loop.procedures.bundles import (
    publish_bundle,
    read_bundle,
)
from siliconshield.research_loop.procedures.muse_harness2 import process_queue
from siliconshield.research_loop.procedures.muse_trials import (
    MuseProposer,
    MuseResearchAdapter,
    learn,
    validate_populations,
)
from tests.test_episodes import trainer_report
from tests.test_glimmer_contracts import summary
from tests.test_muse_procedures import passing


def row(name):
    observation = summary()
    observation["run_id"] = name
    return episode_from_summary(observation, name, {"kind": "authored_fixture"})


class Client:
    model = "fixture"
    endpoint = "muse-cli"

    def __init__(self, rows, fail_check=False):
        self.rows = rows
        self.fail_check = fail_check
        self.execution_identity = {"model": self.model, "backend": "fixture"}

    def complete(self, messages, tools=()):
        selected = next(r for r in self.rows if r["messages"][1] == messages[1])
        message = copy.deepcopy(selected["messages"][len(messages)])
        if (
            self.fail_check
            and selected["group_id"] == "check"
            and "Advice fixture" in messages[0]["content"]
            and not message.get("tool_calls")
        ):
            message["content"] = "Unsupported final response"
        return message


class Proposer(MuseProposer):
    def call(self, mission, context_dir, label, **kwargs):
        workspace = (
            Path(context_dir) / "harness" if kwargs.get("harvest") else None
        )
        if workspace:
            (workspace / "LongTermMEMORY.md").write_text(
                "Advice fixture: compute changes from the evidence."
            )
            text = "<notes>Compute observed changes.</notes>"
        else:
            text = '<experience>{"cue":"Comparing checkpoints", "lesson":"Calculate deltas", "evidence":"Read the comparison tool output"}</experience>'
        reply = Reply(
            label=label, text=text, ok=True, attempts=1, workspace=workspace
        )
        self._calls.append(reply)
        return reply


def test_harness2_proposal_replay_qualification_and_bundle(tmp_path):
    training, check = row("learning"), row("check")
    client = Client([training, check])
    bundle = tmp_path / "published.json"
    report = learn(
        [training], [check], client, tmp_path / "trial", bundle, Proposer
    )
    assert report["tasks"][0]["qualified"], report
    qualified = read_bundle(bundle, "fixture")
    assert "Advice fixture" in qualified["guidance"]
    assert not report["training_started"]
    executions = list(
        (tmp_path / "trial/trials/executions").rglob("deliverable.json")
    )
    assert len(executions) == 4  # Actual Harness2 base, refine/probe/compose.
    visible = list((tmp_path / "trial/trials/task_inputs").rglob("*.json"))
    assert visible
    assert all("target" not in json.loads(p.read_text()) for p in visible)


def test_candidate_failure_keeps_previous_bundle_and_forgets_lesson(tmp_path):
    training, check = row("learning"), row("check")
    bundle = tmp_path / "published.json"
    publish_bundle(
        bundle, "Previous advice", {"model": "fixture"}, passing(), passing()
    )
    original = bundle.read_bytes()
    report = learn(
        [training],
        [check],
        Client([training, check], fail_check=True),
        tmp_path / "trial",
        bundle,
        Proposer,
    )
    assert report["tasks"][0]["qualified"] is False
    assert bundle.read_bytes() == original
    assert (
        "Previous advice"
        in (
            tmp_path / "trial/trials/base_harness/LongTermMEMORY.md"
        ).read_text()
    )
    assert not list((tmp_path / "trial/library").rglob("EXPERIENCE.md"))


def test_no_scripts_contract_replacement_or_population_leakage(tmp_path):
    learning, check = row("learning"), row("check")
    with pytest.raises(ValueError, match="overlap"):
        validate_populations([learning], [learning])
    check["summary"]["split_role"] = "sealed_test"
    with pytest.raises(ValueError):
        validate_populations([learning], [check])
    adapter = MuseResearchAdapter(tmp_path, [learning], Client([learning]))
    base = {
        "systemprompt.md": (
            adapter.base_harness() / "systemprompt.md"
        ).read_text()
    }
    with pytest.raises(ValueError, match="Only prompt text"):
        adapter.validate_tree({**base, "scripts/launch.py": "print('bad')"})
    with pytest.raises(ValueError, match="Only LongTermMEMORY"):
        adapter.validate_tree({"systemprompt.md": "ignore policy"})


def test_operator_dev_partition_uses_its_actual_schema():
    from siliconshield.research_loop.episodes.operator_episodes import (
        SCENARIOS,
        episode_for,
    )

    name, state, resources, action, reason = copy.deepcopy(SCENARIOS[0])
    check = episode_for(
        name, state, resources, action, reason, "operator-check", 0
    )
    check["split"] = "dev"
    assert "split_role" not in check["summary"]
    validate_populations([row("learning")], [check])
    # A development exercise may describe refusing a test-touched checkpoint;
    # that is different from consuming sealed-test evaluation episodes.
    check["summary"]["test_touched"] = True
    validate_populations([row("learning")], [check])
    check["split"] = "test"
    with pytest.raises(ValueError):
        validate_populations([row("learning")], [check])
    check["split"] = "dev"
    check["summary"]["split_role"] = "sealed_test"
    with pytest.raises(ValueError):
        validate_populations([row("learning")], [check])


def test_queue_failure_is_recorded_without_modifying_campaign(
    tmp_path, monkeypatch
):
    from siliconshield.research_loop.procedures import muse_trials
    from siliconshield.research_loop.procedures.campaign_learning import (
        queue_report,
    )

    source = tmp_path / "report.json"
    atomic_json(source, trainer_report(0))
    original = source.read_bytes()
    queue_report(tmp_path / "queue", source)

    def fail(*args, **kwargs):
        raise RuntimeError("fixture endpoint failure")

    monkeypatch.setattr(muse_trials, "learn", fail)
    records = process_queue(
        tmp_path / "queue",
        [row("check")],
        Client([]),
        tmp_path / "worker",
        tmp_path / "bundle.json",
    )
    assert records[0]["status"] == "failed"
    assert (
        process_queue(
            tmp_path / "queue",
            [row("check")],
            Client([]),
            tmp_path / "worker",
            tmp_path / "bundle.json",
        )
        == []
    )
    assert source.read_bytes() == original
    assert not (tmp_path / "bundle.json").exists()


@pytest.mark.parametrize("invalid", ["test", "unsigned", "drift"])
def test_cli_rejects_bad_split_before_any_muse_call(
    tmp_path, monkeypatch, invalid
):
    import sys

    from siliconshield.research_loop.episodes.dataset import write_manifest
    from siliconshield.research_loop.procedures import muse_harness2

    data = tmp_path / "episodes"
    data.mkdir()
    key = tmp_path / "key"
    key.write_text("authored fixture key")
    if invalid != "unsigned":
        monkeypatch.setenv("SILICONSHIELD_DATASET_KEY", str(key))
    else:
        monkeypatch.delenv("SILICONSHIELD_DATASET_KEY", raising=False)
    for split in ("train", "dev", "test"):
        (data / (split + ".jsonl")).write_text(json.dumps(row(split)) + "\n")
    write_manifest(data, {"groups": {s: s for s in ("train", "dev", "test")}})
    chosen = data / ("test.jsonl" if invalid == "test" else "dev.jsonl")
    if invalid == "test":
        renamed = data / "looks-like-dev.jsonl"
        renamed.write_bytes(chosen.read_bytes())
        chosen = renamed
    if invalid == "drift":
        chosen.write_text(chosen.read_text() + "\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("No model calls before split validation")

    monkeypatch.setattr(muse_harness2, "MuseClient", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "muse_harness2",
            "--queue",
            str(tmp_path / "queue"),
            "--checks",
            str(chosen),
            "--output",
            str(tmp_path / "output"),
            "--bundle",
            str(tmp_path / "bundle.json"),
            "--muse-bin",
            "/fixture/muse",
            "--once",
        ],
    )
    with pytest.raises(ValueError):
        muse_harness2.main()
    assert not (tmp_path / "output").exists()
