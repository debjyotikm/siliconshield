"""Procedure boundaries need no model, Harness2 installation, or live campaign."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from siliconshield.research_loop.episodes.contracts import (
    OPERATOR_TOOLS,
    atomic_json,
    digest,
)
from siliconshield.research_loop.execution import campaign_supervisor as sup
from siliconshield.research_loop.procedures.bundles import (
    GRADE_FIELDS,
    publish_bundle,
    qualifies,
    read_bundle,
)
from siliconshield.research_loop.procedures.campaign_learning import (
    load_request,
    queue_report,
)
from siliconshield.research_loop.procedures.muse_client import (
    MuseClient,
    parse_message,
    response_schema,
)
from tests.test_campaign_supervisor import (
    FakeAgent,
    analysis_record,
    config_for,
    fake_evaluator,
    fake_operator_factory,
    make_home,
)
from tests.test_episodes import trainer_report
from tests.test_glimmer_contracts import episode


def passing():
    return {"check": dict.fromkeys(GRADE_FIELDS, True)}


def bundle(path, model="fixture"):
    return publish_bundle(
        path,
        "Read measured evidence before deciding.",
        {"model": model},
        passing(),
        passing(),
    )


def test_bundle_refuses_failed_checks_and_tampering(tmp_path):
    path = tmp_path / "bundle.json"
    good = bundle(path)
    before = path.read_bytes()
    for field in GRADE_FIELDS:
        bad = passing()
        bad["check"][field] = False
        with pytest.raises(ValueError, match="qualification"):
            publish_bundle(
                path, "new advice", {"model": "fixture"}, passing(), bad
            )
        assert path.read_bytes() == before
    assert not qualifies({}, {})
    assert not qualifies(passing(), {"other": passing()["check"]})
    good["guidance"] = "tampered"
    atomic_json(path, good)
    with pytest.raises(ValueError, match="damaged"):
        read_bundle(path, "fixture")
    bundle(path)
    with pytest.raises(ValueError, match="another model"):
        read_bundle(path, "other-model")


def test_supervisor_applies_only_qualified_optional_guidance(tmp_path):
    path = tmp_path / "bundle.json"
    saved = bundle(path)
    config = {
        "repo": str(tmp_path),
        "model": "fixture",
        "procedure_bundle": str(path),
    }
    prompts = []

    def runner(config, kind, prompt, workdir):
        prompts.append(prompt)
        return {"terminal": "completed", "exit_code": 0, "session": "fixture"}

    for action in ("analyze", "prepare", "recover", "plan"):
        sup._run_agent_event(
            tmp_path, config, action, "Original task", "fixture", runner
        )
    assert all("does not change campaign policy" in text for text in prompts)
    assert all(saved["guidance"] in text for text in prompts)
    records = [
        json.loads(line)
        for line in (tmp_path / "campaign/supervisor/sessions.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(row["procedure"]["digest"] == saved["digest"] for row in records)
    path.write_text("corrupt")
    sup._run_agent_event(
        tmp_path, config, "analyze", "Original task", "fixture", runner
    )
    assert prompts[-1] == "Original task"
    records = [
        json.loads(line)
        for line in (tmp_path / "campaign/supervisor/sessions.jsonl")
        .read_text()
        .splitlines()
    ]
    assert records[-1]["procedure"]["status"] == "unavailable"
    del config["procedure_bundle"]
    sup._run_agent_event(
        tmp_path, config, "plan", "Original task", "fixture", runner
    )
    assert prompts[-1] == "Original task"


def test_queue_copies_only_development_and_is_idempotent(tmp_path):
    source = tmp_path / "report.json"
    atomic_json(source, trainer_report(0))
    original = source.read_bytes()
    request = queue_report(tmp_path / "queue", source)
    assert queue_report(tmp_path / "queue", source) == request
    assert source.read_bytes() == original
    row = load_request(request)
    assert row["summary"]["split_role"] == "open_development"
    bad = trainer_report(0)
    bad["test_evaluated"] = True
    atomic_json(source, bad)
    with pytest.raises(ValueError, match="final/test"):
        queue_report(tmp_path / "queue", source)
    assert len(list((tmp_path / "queue").glob("*.json"))) == 1
    data = json.loads(request.read_text())
    data["episode"]["summary"]["selected"]["agreement"] = 0.1
    atomic_json(request, data)
    with pytest.raises(ValueError, match="identity"):
        load_request(request)


def test_tool_free_muse_command_and_executable_drift(tmp_path):
    binary = tmp_path / "muse"
    binary.write_text("fixture")
    client = MuseClient(binary, tmp_path / "calls")
    argv = client.argv(tmp_path, tmp_path / "prompt.txt")
    assert argv[argv.index("--preset") + 1] == "empty"
    for flag in (
        "--disable-shell",
        "--disable-write",
        "--disable-web-tools",
        "--no-session-log",
    ):
        assert flag in argv
    assert "--disable-sandbox" not in argv and "--yolo" not in argv
    binary.write_text("changed")
    with pytest.raises(ValueError, match="executable changed"):
        client.complete([])


def test_cli_jsonl_completion_and_failed_terminal(tmp_path, monkeypatch):
    binary = tmp_path / "muse"
    binary.write_text("fixture")
    final = {"role": "assistant", "content": "Answer", "tool_calls": []}
    terminal = {
        "payload_type": "run.terminal.completed",
        "payload": {"terminal": "completed", "text": json.dumps(final)},
    }

    class Process:
        def __init__(self, argv, **kwargs):
            assert kwargs["start_new_session"]
            assert Path(kwargs["cwd"]).name == "workspace"
            kwargs["stdout"].write(json.dumps(terminal) + "\n")
            kwargs["stdout"].flush()

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", Process)
    client = MuseClient(binary, tmp_path / "calls")
    assert client.complete([]) == final
    terminal["payload"]["terminal"] = "failed"
    with pytest.raises(RuntimeError, match="failed"):
        client.complete([])
    assert len(list((tmp_path / "calls").glob("call-*/receipt.json"))) == 2


def test_response_parser_cannot_dispatch_native_or_unknown_tools():
    row = episode()
    message = copy.deepcopy(row["messages"][2])
    assert parse_message(json.dumps(message), row["tools"])["tool_calls"]
    message["tool_calls"][0]["function"]["name"] = "shell"
    with pytest.raises(ValueError, match="Unknown tool"):
        parse_message(json.dumps(message), row["tools"])
    with pytest.raises(ValueError):
        parse_message('{"role":"assistant","content":null,"tool_calls":[]}', [])


def test_malformed_bundle_cannot_sneak_in_with_valid_digest(tmp_path):
    path = tmp_path / "bundle.json"
    value = bundle(path)
    value["candidate"] = {}
    value["digest"] = digest({k: v for k, v in value.items() if k != "digest"})
    atomic_json(path, value)
    with pytest.raises(ValueError):
        read_bundle(path, "fixture")


@pytest.mark.parametrize(
    "malformed", [None, [], {"check": []}, {"check": None}]
)
def test_malformed_grades_are_rejected(malformed):
    assert not qualifies(passing(), malformed)
    assert not qualifies(malformed, passing())


@pytest.mark.parametrize("test_touched", [False, True])
def test_real_supervisor_analysis_to_private_queue(
    tmp_path, monkeypatch, test_touched
):
    home = make_home(tmp_path)
    campaign = home / "campaign"
    sup.laya_campaign.prepare_round(campaign, "r1", "d1", "fixture")
    sup.laya_campaign.advance(campaign, "r1", "failed", reason="fixture")
    monkeypatch.setattr(
        sup.laya_campaign, "run_round", lambda *a, **k: {"phase": "failed"}
    )
    report = trainer_report(0)
    report["test_evaluated"] = test_touched
    path = home / "runs/demo-r1/report.json"
    atomic_json(path, report)
    original = path.read_bytes()
    agent = FakeAgent()

    def analyze(prompt):
        atomic_json(campaign / "rounds/r1/analysis.json", analysis_record())

    agent.on("analyze", analyze)
    config = config_for(home)
    config["procedure_queue"] = str(tmp_path / "queue")
    status = sup.supervise_once(
        home,
        config,
        operator_factory=fake_operator_factory,
        evaluator=fake_evaluator,
        agent_runner=agent,
    )
    assert status["action"] == "analyzed"
    assert path.read_bytes() == original
    handoff = sup.load_handoff(home)
    assert handoff["procedure_learning"]["status"] == (
        "unavailable" if test_touched else "queued"
    )
    if not test_touched:
        assert len(list((tmp_path / "queue").glob("*.json"))) == 1


def test_schema_contains_only_advertised_replay_tools():
    tools = episode()["tools"]
    schema = response_schema(tools)
    choices = schema["properties"]["tool_calls"]["items"]["properties"][
        "function"
    ]["anyOf"]
    assert {c["properties"]["name"]["const"] for c in choices} == {
        t["function"]["name"] for t in tools
    }
    assert response_schema([])["properties"]["tool_calls"]["maxItems"] == 0


def test_operator_wire_schema_closes_objects_and_preserves_tool_contract():
    original = copy.deepcopy(OPERATOR_TOOLS)

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert set(node["required"]) == set(node["properties"])
                assert node["additionalProperties"] is False
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(response_schema(OPERATOR_TOOLS))
    assert OPERATOR_TOOLS == original


def test_operator_wire_decoding_preserves_numbers_and_removes_null_optionals():
    args = {
        "run_id": "fixture",
        "action": "continue",
        "reason": "Observed gain",
        "lr": None,
        "checkpoint_dir": None,
        "assessment": {
            "comparisons": [
                {
                    "view": "transfer",
                    "accuracy_delta_pp": 2.25,
                    "nll_delta": -0.1,
                    "false_safe_delta_pp": None,
                },
            ]
        },
    }
    raw = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "operator.decide",
                    "arguments": args,
                },
            },
        ],
    }
    message = parse_message(json.dumps(raw), OPERATOR_TOOLS)
    actual = message["tool_calls"][0]["function"]["arguments"]
    assert "lr" not in actual and "checkpoint_dir" not in actual
    assert actual["assessment"]["comparisons"] == {
        "transfer": {
            "accuracy_delta_pp": 2.25,
            "nll_delta": -0.1,
            "false_safe_delta_pp": None,
        },
    }
    args["assessment"]["comparisons"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        parse_message(json.dumps(raw), OPERATOR_TOOLS)


def test_structured_content_is_serialized_once_for_final_and_proposer():
    for content in (
        {"files": {"LongTermMEMORY.md": "Use evidence."}, "notes": "Fixture"},
        {
            "action": "continue",
            "run_id": "fixture",
            "executed": False,
            "scope": "operator_replay",
        },
    ):
        raw = {"role": "assistant", "content": content, "tool_calls": []}
        result = parse_message(json.dumps(raw), OPERATOR_TOOLS)
        assert json.loads(result["content"]) == content


def test_timeout_kills_only_own_process_group(tmp_path, monkeypatch):
    from siliconshield.research_loop.procedures import muse_client

    binary = tmp_path / "muse"
    binary.write_text("fixture")
    killed = []

    class Process:
        pid = 12345

        def __init__(self, *args, **kwargs):
            assert kwargs["start_new_session"]

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return -9

    monkeypatch.setattr(subprocess, "Popen", Process)
    monkeypatch.setattr(
        muse_client.os, "killpg", lambda pid, sig: killed.append(pid)
    )
    with pytest.raises(subprocess.TimeoutExpired):
        MuseClient(binary, tmp_path / "calls", timeout=1).complete([])
    assert killed == [Process.pid]
    receipt = next((tmp_path / "calls").glob("call-*/receipt.json"))
    assert json.loads(receipt.read_text())["completed"] is False
