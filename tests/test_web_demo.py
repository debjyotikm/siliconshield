"""Browser transport and real-router boundaries on synthetic inputs."""

import copy
import hashlib
import http.client
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from test_multi_expert import _rated, _result, _write_registry

from siliconshield.web_demo import backend, config, toy_runner
from siliconshield.web_demo.glimmer_client import LocalGlimmerClient
from siliconshield.web_demo.server import DemoServer
from siliconshield.web_demo.toy_data import create_dataset

BODY = {
    "model": "contract-1",
    "device": "cpu",
    "rtl": "module t; wire q; endmodule",
    "requirement": "Keep q low.",
    "assumptions": "Two-state logic.",
    "scope": "fifo",
}


def settings(tmp_path: Path, **values: Any) -> dict[str, Any]:
    path = tmp_path / "web.local.json"
    path.write_text(
        json.dumps({"state_dir": str(tmp_path / "state"), **values})
    )
    return config.load_config(path)


@pytest.fixture
def demo(tmp_path):
    app = backend.Demo(settings(tmp_path), health_poll=False)
    yield app
    app.close()


@pytest.fixture
def http_api(demo):
    server = DemoServer(("127.0.0.1", 0), demo)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, body=None, *, headers=None):
        conn = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        base = {
            "Content-Type": "application/json",
            "X-Demo-Token": server.token,
            "Origin": f"http://127.0.0.1:{server.server_port}",
        }
        base.update(headers or {})
        conn.request(
            "GET" if body is None else "POST",
            path,
            body=None if body is None else json.dumps(body),
            headers=base,
        )
        response = conn.getresponse()
        data = response.read()
        result = (
            json.loads(data)
            if response.getheader("Content-Type") == "application/json"
            else data.decode()
        )
        code = response.status
        conn.close()
        return code, result

    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


class FixturePool:
    """Replace only model execution, keeping real router and result validation."""

    seen = []
    truncated = False

    def __init__(self, extra_env=None):
        self.env = extra_env
        self._workers = {}

    def load(self, entry):
        self.entry = entry

    def ready(self, model):
        return {"device": self.entry["device"]}

    def analyze(self, model, state):
        self.seen.append((model, state, self.env))
        result = _result(state=state, model_id=model)
        result["coverage"]["truncated"] = self.truncated
        return result

    def close(self):
        pass


def add_models(demo, tmp_path, monkeypatch):
    from siliconshield.schemas import EXPERT_CAPABILITY_SCHEMA

    profile = _rated("contract-1", ["fifo"], {"fifo": 90})
    profile["schema"] = EXPERT_CAPABILITY_SCHEMA
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry = _write_registry(
        tmp_path,
        profile,
        {"id": "contract-1", "checkpoint": str(checkpoint)},
    )
    raw = json.loads(Path(registry).read_text())
    second = {
        **copy.deepcopy(raw["models"][0]),
        "id": "contract-2",
        "profile": "second.json",
    }
    (tmp_path / "second.json").write_text(
        json.dumps(
            {
                **_rated("contract-2", ["fifo"], {"fifo": 80}),
                "schema": EXPERT_CAPABILITY_SCHEMA,
            }
        )
    )
    raw["models"].append(second)
    Path(registry).write_text(json.dumps(raw))
    new = settings(tmp_path, registry=registry)
    demo.config = new
    demo.entries = {entry["id"]: entry for entry in new["entries"]}
    FixturePool.seen = []
    FixturePool.truncated = False
    monkeypatch.setattr(backend, "WorkerPool", FixturePool)


def wait_job(http_api, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        _, job = http_api("/api/jobs/" + job_id)
        if job["status"] != "running":
            return job
        time.sleep(0.01)
    pytest.fail("Fixture request did not terminate")


def test_import_and_empty_configuration_do_not_need_models_or_gpu(tmp_path):
    cfg = settings(tmp_path)
    assert cfg["entries"] == [] and cfg["gpu"] is None
    assert cfg["glimmer"] is None and cfg["training_enabled"] is False


@pytest.mark.parametrize(
    "values",
    [
        {"gpu": True},
        {"gpu": -1},
        {"port": 0},
        {"training": {"enabled": "false"}},
        {"source": "/unexpected"},
    ],
)
def test_configuration_refuses_malformed_authority(tmp_path, values):
    with pytest.raises(ValueError):
        settings(tmp_path, **values)


def test_relative_paths_are_relative_to_config_not_working_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir("/")
    cfg = settings(tmp_path, state_dir="session")
    assert cfg["state_dir"] == str(tmp_path / "session")


def test_http_bootstrap_and_assets_work_without_private_configuration(http_api):
    status, data = http_api("/api/bootstrap")
    assert status == 200 and data["default_device"] == "cpu"
    assert data["devices"] == [{"id": "cpu", "label": "CPU · server"}]
    assert (
        data["training_enabled"] is False
        and data["glimmer_configured"] is False
    )
    assert len(data["examples"]) == 7
    for name in ("/", "/app.js", "/style.css", "/favicon.svg"):
        code, text = http_api(name)
        assert code == 200 and text
    assert http_api("/api/status")[1]["models"] == []


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "foreign.invalid"},
        {"Origin": "https://foreign.invalid"},
        {"X-Demo-Token": ""},
    ],
)
def test_http_rejects_foreign_mutations_before_dispatch(
    http_api, demo, monkeypatch, headers
):
    monkeypatch.setattr(
        demo, "submit_task", lambda *args: pytest.fail("Forbidden dispatch")
    )
    code, _ = http_api(
        "/api/training/start", {"confirm": True}, headers=headers
    )
    assert code == 403


def test_disabled_training_and_browser_recipe_overrides_never_launch(
    http_api, demo, monkeypatch
):
    monkeypatch.setattr(
        demo, "prepare_training", lambda *args: pytest.fail("Training launched")
    )
    assert http_api("/api/training/start", {"confirm": True})[0] == 400
    demo.config["training_enabled"] = True
    assert (
        http_api("/api/training/start", {"confirm": True, "max_updates": 1000})[
            0
        ]
        == 400
    )


def test_cpu_assessment_uses_real_validation_and_preserves_input(
    http_api, demo, tmp_path, monkeypatch
):
    add_models(demo, tmp_path, monkeypatch)
    monkeypatch.setattr(
        backend,
        "run_command",
        lambda *args, **kwargs: pytest.fail(
            "CPU inference called a hardware command"
        ),
    )
    code, submitted = http_api("/api/assess", BODY)
    assert code == 202
    job = wait_job(http_api, submitted["job_id"])
    assert job["status"] == "complete"
    assert job["result"]["label"] == "satisfies_requirement"
    assert set(job["result"]["scores"]) == set(backend.LABELS)
    assert job["result"]["input"]["rtl"] == BODY["rtl"]
    assert FixturePool.seen[0][2]["CUDA_VISIBLE_DEVICES"] == ""
    assert FixturePool.seen[0][1] == backend.state_text(BODY)


def test_truncated_worker_result_is_error_without_verdict(
    http_api, demo, tmp_path, monkeypatch
):
    add_models(demo, tmp_path, monkeypatch)
    FixturePool.truncated = True
    _, submitted = http_api("/api/assess", BODY)
    job = wait_job(http_api, submitted["job_id"])
    assert job["status"] == "error" and "truncated" in job["error"]
    assert "result" not in job


def test_real_router_consults_both_workers_and_preserves_agent_error(
    demo, tmp_path, monkeypatch
):
    add_models(demo, tmp_path, monkeypatch)
    result = demo.route({**BODY, "arbitrate": True}, lambda text: None)
    assert len(result["assessment"]["responses"]) == 2
    assert not result["assessment"]["errors"]
    assert {row[0] for row in FixturePool.seen} == {"contract-1", "contract-2"}
    assert (
        "arbitration" not in result
        and "Configure local Glimmer" in result["arbitration_error"]
    )
    assert result["input"]["rtl"] == BODY["rtl"]


def test_foreign_gpu_process_blocks_loading(demo, monkeypatch):
    demo.config["gpu"] = 0
    demo.entries = {"contract-1": {}}
    monkeypatch.setattr(
        backend, "run_command", lambda *args, **kwargs: "12345\n"
    )
    with pytest.raises(RuntimeError, match="unrelated"):
        demo.load("contract-1", lambda text: None, "cuda:0")
    assert not demo.pools


def test_controller_failure_keeps_live_owned_job_active(
    demo, tmp_path, monkeypatch
):
    demo.training = {
        "directory": str(tmp_path),
        "status": "error",
        "unit": "owned",
    }
    (tmp_path / "controller-result.json").write_text('{"status":"error"}')
    monkeypatch.setattr(demo, "owned_job_running", lambda path: True)
    assert demo.training_active() is True
    monkeypatch.setattr(demo, "owned_job_running", lambda path: False)
    assert demo.training_active() is False


def test_corrupt_restart_pointer_refuses_instead_of_forgetting_job(tmp_path):
    cfg = settings(tmp_path)
    state = Path(cfg["state_dir"])
    state.mkdir()
    (state / "active-training.json").write_text("{")
    with pytest.raises(ValueError):
        backend.Demo(cfg, health_poll=False)


def test_toy_dataset_balances_complete_disjoint_groups(tmp_path):
    create_dataset(tmp_path / "toy")
    splits = {
        name: [
            json.loads(line)
            for line in (tmp_path / "toy" / f"{name}.jsonl")
            .read_text()
            .splitlines()
        ]
        for name in ("train", "dev")
    }
    assert [len(splits[name]) for name in ("train", "dev")] == [18, 6]
    assert not {r["split_group_id"] for r in splits["train"]} & {
        r["split_group_id"] for r in splits["dev"]
    }
    for rows in splits.values():
        for group in {r["pair_id"] for r in rows}:
            assert {
                r["expected"]["security"] for r in rows if r["pair_id"] == group
            } == set(backend.LABELS)
    with pytest.raises(FileExistsError):
        create_dataset(tmp_path / "toy")


def test_training_preparation_binds_real_fixture_and_owned_command(
    demo, tmp_path, monkeypatch
):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "weights").write_bytes(b"synthetic-parent")
    demo.entries = {
        "contract-1": {
            "checkpoint": str(parent),
            "expected_files": {
                "weights": hashlib.sha256(b"synthetic-parent").hexdigest()
            },
        }
    }
    demo.config.update(
        training_enabled=True,
        gpu=0,
        glimmer={
            "endpoint": "http://127.0.0.1:18400/v1/chat/completions",
            "model": "synthetic",
        },
        training_root=str(tmp_path / "runs"),
        training_python=sys.executable,
        training_runtime=str(tmp_path / "runtime"),
        resource_root=str(tmp_path / "leases"),
    )
    monkeypatch.setattr(demo, "glimmer_health", lambda: {"status": "ready"})
    monkeypatch.setattr(
        backend.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": 100 * 1024**3})(),
    )
    commands = []

    def command(argv, **kwargs):
        commands.append(argv)
        if "--output-receipt" in argv:
            Path(argv[argv.index("--output-receipt") + 1]).write_text(
                '{"result":"pass"}'
            )
        return ""

    monkeypatch.setattr(backend, "run_command", command)
    record = demo.prepare_training(lambda text: None)
    folder = Path(record["directory"])
    profile = json.loads((folder / "profile.json").read_text())
    assert profile["model"] == str(parent) and profile["gpus"] == [0]
    assert profile["defaults"]["max_updates"] == 3
    assert profile["defaults"]["freeze_encoder"] is True
    assert json.loads((folder / "recipe.json").read_text())["source_mass"] == {
        "toy-synthetic": 1.0
    }
    assert sum(argv[0] == "systemd-run" for argv in commands) == 1
    assert "siliconshield.web_demo.toy_runner" in commands[-1]
    demo._validate_training_pointer(record)


def test_training_rejects_changed_parent_before_submission(
    demo, tmp_path, monkeypatch
):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "weights").write_bytes(b"changed")
    demo.entries = {
        "contract-1": {
            "checkpoint": str(parent),
            "expected_files": {"weights": "a" * 64},
        }
    }
    demo.config.update(
        training_enabled=True, gpu=0, training_root=str(tmp_path / "runs")
    )
    monkeypatch.setattr(demo, "glimmer_health", lambda: {"status": "ready"})
    monkeypatch.setattr(demo, "check_storage", lambda: tmp_path / "runs")
    monkeypatch.setattr(
        backend,
        "run_command",
        lambda argv, **kwargs: ""
        if argv[0] == "nvidia-smi"
        else pytest.fail("Submission attempted"),
    )
    with pytest.raises(ValueError, match="registered export"):
        demo.prepare_training(lambda text: None)


def test_glimmer_native_tool_arguments_and_budgets(monkeypatch):
    observed = []

    def reply(self, url, payload=None):
        observed.append(payload)
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call",
                                "function": {
                                    "name": "experts.arbitrate",
                                    "arguments": '{"conclusion":"unresolved"}',
                                },
                            }
                        ]
                    },
                }
            ]
        }

    monkeypatch.setattr(LocalGlimmerClient, "_request", reply)
    client = LocalGlimmerClient(
        "http://127.0.0.1:18400/v1/chat/completions",
        "synthetic",
        max_calls=1,
        decode_arguments=True,
    )
    result = client.complete([], tools=[{"type": "function"}])
    assert result["tool_calls"][0]["function"]["arguments"] == {
        "conclusion": "unresolved"
    }
    assert observed[0]["chat_template_kwargs"] == {"reasoning_strength": "low"}
    with pytest.raises(RuntimeError, match="budget"):
        client.complete([])


def test_glimmer_incomplete_response_is_never_accepted(monkeypatch):
    monkeypatch.setattr(
        LocalGlimmerClient,
        "_request",
        lambda *args: {
            "choices": [
                {"finish_reason": "length", "message": {"content": "partial"}}
            ]
        },
    )
    client = LocalGlimmerClient(
        "http://127.0.0.1:18400/v1/chat/completions", "synthetic"
    )
    with pytest.raises(RuntimeError, match="budget"):
        client.complete([])


def test_toy_controller_stop_prevents_next_phase(tmp_path, monkeypatch):
    path = tmp_path / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "operator": {
                    "endpoint": "http://127.0.0.1:18400/v1/chat/completions",
                    "model": "synthetic",
                }
            }
        )
    )
    (tmp_path / "cancel.request").touch()
    monkeypatch.setattr(toy_runner, "run_campaign", lambda cfg, run: run({}))
    toy_runner.execute(path)
    assert (
        json.loads((tmp_path / "controller-result.json").read_text())["status"]
        == "stopped"
    )


def test_closed_session_cannot_spawn_more_workers(demo, tmp_path, monkeypatch):
    add_models(demo, tmp_path, monkeypatch)
    demo.load("contract-1", lambda _: None, "cpu")
    demo.close()
    assert not demo.pools
    assert not demo.loaded
    with pytest.raises(ValueError, match="stopping"):
        demo.load("contract-1", lambda _: None, "cpu")


def test_final_operator_acknowledgement_is_requested_from_glimmer(monkeypatch):
    client = LocalGlimmerClient(
        "http://127.0.0.1:8000/v1/chat/completions", "fixture"
    )
    captured = []
    reply = {
        "role": "assistant",
        "content": '{"action":"stop","run_id":"demo","executed":false,"scope":"operator_replay"}',
    }

    def request(url, payload):
        captured.append(payload)
        return {"choices": [{"finish_reason": "stop", "message": reply}]}

    monkeypatch.setattr(client, "_request", request)
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "decision",
                    "type": "function",
                    "function": {
                        "name": "operator.decide",
                        "arguments": '{"action":"stop","run_id":"demo"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "decision",
            "content": '{"recorded":true}',
        },
    ]
    original = copy.deepcopy(messages)
    assert client.complete(messages, tools=[{"type": "function"}]) == reply
    assert messages == original
    assert len(captured) == 1
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[0]["tool_choice"] == "none"
    assert captured[0]["messages"][-1]["role"] == "user"
