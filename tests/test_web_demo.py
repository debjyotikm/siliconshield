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

    def __contains__(self, model):
        return True

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


def test_route_evicts_dead_pool_and_recovers(demo, tmp_path, monkeypatch):
    add_models(demo, tmp_path, monkeypatch)
    calls = {"count": 0}
    real_analyze = FixturePool.analyze

    def flaky(self, model, state):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("synthetic worker death")
        return real_analyze(self, model, state)

    monkeypatch.setattr(FixturePool, "analyze", flaky)
    first = demo.route(dict(BODY), lambda text: None)
    assert len(first["assessment"]["errors"]) == 1
    assert "contract-1" not in demo.pools
    assert "contract-1" not in demo.loaded
    second = demo.route(dict(BODY), lambda text: None)
    assert len(second["assessment"]["responses"]) == 2
    assert not second["assessment"]["errors"]


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


def test_glimmer_malformed_responses_raise_value_errors(monkeypatch):
    bodies = [
        {},
        {"choices": []},
        {"choices": [{"message": "not-a-dict"}]},
        {"choices": [{"message": {"tool_calls": "not-a-list"}}]},
        {"choices": [{"message": {"tool_calls": [{"id": "call-1"}]}}]},
        [1, 2],
    ]
    calls = {"count": 0}

    def reply(self, url, payload=None):
        body = bodies[calls["count"]]
        calls["count"] += 1
        return body

    monkeypatch.setattr(LocalGlimmerClient, "_request", reply)
    client = LocalGlimmerClient(
        "http://127.0.0.1:18400/v1/chat/completions", "synthetic"
    )
    for _ in bodies:
        with pytest.raises(ValueError):
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


def test_state_text_rejects_smuggled_section_headers():
    from siliconshield.experts.identity import state_sections

    good = backend.state_text(dict(BODY))
    requirement, _, _ = state_sections(good)
    assert requirement == BODY["requirement"]
    for field in ("requirement", "assumptions"):
        for header in ("REQUIREMENT", "ASSUMPTIONS", "RTL"):
            body = dict(BODY)
            body[field] = "first line\n%s\nsecond line" % header
            with pytest.raises(ValueError, match="section header"):
                backend.state_text(body)


def test_glimmer_complete_rejects_malformed_messages(monkeypatch):
    def reply(self, url, payload=None):
        raise AssertionError("no network in this test")

    monkeypatch.setattr(LocalGlimmerClient, "_request", reply)
    client = LocalGlimmerClient(
        endpoint="http://127.0.0.1:1/v1/chat/completions",
        model="synthetic",
        max_calls=4,
        timeout=1,
    )
    with pytest.raises(ValueError, match="list of objects"):
        client.complete([{"role": "system"}, "nope"])
    with pytest.raises(ValueError, match="must be text"):
        client.complete([{"role": "system", "content": ["nope"]}])


def test_non_ascii_token_is_forbidden_not_dropped(http_api):
    code, result = http_api(
        "/api/assess", dict(BODY), headers={"X-Demo-Token": "tökén"}
    )
    assert code == 403
    assert "error" in result
    code, _ = http_api("/api/status", None)
    assert code == 200


def test_empty_transfer_encoding_is_rejected(http_api):
    code, _ = http_api(
        "/api/assess", dict(BODY), headers={"Transfer-Encoding": ""}
    )
    assert code == 400


def test_glimmer_config_needs_endpoint_and_model(tmp_path):
    with pytest.raises(ValueError, match="endpoint and model"):
        settings(tmp_path, glimmer={})
    with pytest.raises(ValueError, match="endpoint and model"):
        settings(
            tmp_path,
            glimmer={"endpoint": "http://127.0.0.1:1/chat/completions"},
        )


def test_glimmer_budgets_validated_at_construction():
    good = {
        "endpoint": "http://127.0.0.1:1/v1/chat/completions",
        "model": "synthetic",
    }
    with pytest.raises(ValueError, match="max_calls"):
        LocalGlimmerClient(**good, max_calls="6")
    with pytest.raises(ValueError, match="max_tokens"):
        LocalGlimmerClient(**good, max_tokens=0)
    with pytest.raises(ValueError, match="timeout"):
        LocalGlimmerClient(**good, timeout=-1)


def test_toy_study_rejects_non_integer_cycles():
    from siliconshield.web_demo import toy_runner

    with pytest.raises(ValueError, match="max_cycles"):
        toy_runner.max_calls_for_cycles("6")
    with pytest.raises(ValueError, match="max_cycles"):
        toy_runner.max_calls_for_cycles(0)
    assert toy_runner.max_calls_for_cycles(6) == 72


def _await_quiet(demo, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with demo.lock:
            running = [
                job
                for job in demo.jobs.values()
                if job.get("status") == "running"
            ]
            threads = [thread for thread in demo._threads if thread.is_alive()]
        if not running and not threads:
            return
        time.sleep(0.05)
    raise AssertionError("jobs did not finish")


def test_close_drains_running_jobs(demo):
    started = threading.Event()

    def slow(log):
        started.set()
        time.sleep(0.5)
        return {"ok": True}

    receipt = demo.submit_task("assess", slow)
    assert started.wait(timeout=10)
    demo.close()
    with demo.lock:
        job = demo.jobs[receipt["job_id"]]
    assert job["status"] == "complete"
    assert (demo.state / ("request-%s.json" % receipt["job_id"])).exists()


def test_terminal_jobs_evicted_from_memory_and_disk(demo):
    for _ in range(2):
        for _ in range(5):
            demo.submit_task("assess", lambda log: {"ok": True})
        _await_quiet(demo)
    with demo.lock:
        assert len(demo.jobs) <= 8
    assert len(list(demo.state.glob("request-*.json"))) <= 8


def test_status_read_never_rewrites_training_pointer(
    demo, tmp_path, monkeypatch
):
    folder = tmp_path / "run"
    (folder / "runs").mkdir(parents=True)
    pointer = demo.state / "active-training.json"
    saved = {
        "id": "synthetic-run",
        "directory": str(folder),
        "unit": "synthetic-unit",
        "status": "running",
    }
    pointer.write_text(json.dumps(saved))
    demo.training = dict(saved)
    before = pointer.read_bytes()
    monkeypatch.setattr(backend, "run_command", lambda *args, **kwargs: "")
    status = demo.training_status()
    assert status["active"] is False
    assert pointer.read_bytes() == before


def test_rejected_glimmer_input_keeps_its_budget(tmp_path):
    client = LocalGlimmerClient(
        "http://127.0.0.1:9/v1/chat/completions",
        "synthetic",
        max_calls=1,
        audit_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="list of objects"):
        client.complete([{"role": "system"}, "nope"])
    assert client.calls == 0


def test_route_scope_is_validated_synchronously(http_api):
    code, result = http_api("/api/route", {**BODY, "scope": "not-a-domain"})
    assert code == 400
    assert "measured routing domain" in result["error"]


def test_status_survives_a_missing_registry(demo, tmp_path):
    demo.config["registry"] = str(tmp_path / "missing.json")
    status = demo.status()
    assert status["registry_error"].startswith(
        ("ValueError", "OSError", "FileNotFoundError")
    )
    assert status["models"] == []


def test_device_change_unloads_and_logs(demo, monkeypatch):
    demo.entries = {"contract-1": {"device": "cpu"}}
    demo.config["gpu"] = 0
    monkeypatch.setattr(backend, "run_command", lambda *a, **k: "")
    monkeypatch.setattr(backend, "WorkerPool", FixturePool)
    lines = []
    demo.ensure_model("contract-1", lines.append, "cpu")
    demo.ensure_model("contract-1", lines.append, "cuda:0")
    assert any("Unloading contract-1" in line for line in lines)
    assert demo.loaded["contract-1"]["device"] == "cuda:0"


def test_close_does_not_wait_out_a_stuck_job(demo, monkeypatch):
    pool = FixturePool()
    calls = []
    pool.close = lambda: calls.append(1)
    demo.pools["contract-1"] = pool

    class StuckLock:
        def acquire(self, timeout=None):
            assert timeout == 30
            return False

        def release(self):  # pragma: no cover - never acquired
            raise AssertionError("not acquired")

    monkeypatch.setattr(demo, "inference_lock", StuckLock())
    demo.close()
    assert calls == [1]


def test_evicted_jobs_leave_tombstones(demo):
    first = demo.submit_task("assess", lambda log: {"ok": True})["job_id"]
    for _ in range(2):
        for _ in range(5):
            demo.submit_task("assess", lambda log: {"ok": True})
        _await_quiet(demo)
    assert demo.was_evicted(first) is True
    assert demo.was_evicted("0" * 32) is False


def test_evicted_job_polls_as_expired(demo, http_api):
    first = demo.submit_task("assess", lambda log: {"ok": True})["job_id"]
    for _ in range(2):
        for _ in range(5):
            demo.submit_task("assess", lambda log: {"ok": True})
        _await_quiet(demo)
    code, result = http_api(f"/api/jobs/{first}")
    assert code == 410
    assert "expired" in result["error"]
    code, _ = http_api("/api/jobs/" + "0" * 32)
    assert code == 404


def test_pointer_with_non_string_directory_is_a_value_error(tmp_path):
    cfg = settings(tmp_path)
    cfg["training_root"] = str(tmp_path / "runs")
    Path(cfg["state_dir"]).mkdir(parents=True, exist_ok=True)
    pointer = Path(cfg["state_dir"]) / "active-training.json"
    pointer.write_text(
        json.dumps(
            {
                "id": "20240101-000000-abcdef",
                "directory": 123,
                "unit": "siliconshield-web-toy-20240101-000000-abcdef",
            }
        )
    )
    with pytest.raises(ValueError, match="Malformed training pointer"):
        backend.Demo(cfg, health_poll=False)


def test_training_status_reports_only_the_log_tail(demo, tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "controller-result.json").write_text(
        json.dumps({"status": "complete"})
    )
    head = "H" * 8000
    tail = "T" * 12000
    (folder / "controller.log").write_text(head + tail)
    demo.training = {"directory": str(folder), "status": "complete"}
    value = demo.training_status()
    assert value["log"] == tail
    assert "H" not in value["log"]


def test_malformed_history_never_breaks_the_recorded_scan(monkeypatch):
    client = LocalGlimmerClient(
        endpoint="http://127.0.0.1:1/chat/completions",
        model="synthetic",
        max_calls=3,
    )
    seen = []

    def fake_request(endpoint, payload):
        seen.append(payload)
        return {"choices": [{"message": {"content": "done"}}]}

    monkeypatch.setattr(client, "_request", fake_request)
    histories = [
        [{"role": "assistant", "tool_calls": "operator.decide"}],
        [{"role": "assistant", "tool_calls": ["x", 1]}],
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "1", "function": "operator.decide"}],
            }
        ],
    ]
    for history in histories:
        message = client.complete(
            history + [{"role": "tool", "tool_call_id": "1", "content": "{}"}]
        )
        assert message["content"] == "done"
    assert len(seen) == 3


def test_stop_reports_without_experiment_id(demo, tmp_path, monkeypatch):
    import siliconshield.web_demo.backend as backend_mod

    folder = tmp_path / "run"
    run = folder / "runs" / "r1"
    run.mkdir(parents=True)
    (folder / "profile.json").write_text("{}")
    (run / "run-manifest.json").write_text('{"phase": "observe"}')
    demo.training = {"directory": str(folder), "status": "running"}

    class FakeOperator:
        def __init__(self, profile):
            self.stopped = []

        def observe(self, manifest):
            return manifest

        def intervene(self, manifest, action):
            self.stopped.append(action)

    fake = FakeOperator("x")
    monkeypatch.setattr(backend_mod, "Operator", lambda profile: fake)
    result = demo.stop_training()
    assert result["status"] == "stop_requested"
    assert fake.stopped == ["stop"]
    assert len(result["messages"]) == 1
    assert "r1" in result["messages"][0]


def test_backend_rejects_malformed_inputs(tmp_path):
    from siliconshield.web_demo import backend

    assert backend.read_json(5, "fallback") == "fallback"
    with pytest.raises(ValueError, match="argument list"):
        backend.run_command("echo hi")
    with pytest.raises(ValueError, match="timeout must be positive"):
        backend.run_command(["echo", "hi"], timeout="x")
    with pytest.raises(ValueError, match="must be an object"):
        backend.state_text(["x"])
    with pytest.raises(ValueError, match="must be an object"):
        backend.Demo(["x"], health_poll=False)
    with pytest.raises(ValueError, match="state directory"):
        backend.Demo({"entries": []}, health_poll=False)
    with pytest.raises(ValueError, match="entries with ids"):
        backend.Demo(
            {"state_dir": str(tmp_path / "s"), "entries": [{"x": 1}]},
            health_poll=False,
        )
    demo = backend.Demo(
        {"state_dir": str(tmp_path / "s"), "entries": []},
        health_poll=False,
    )
    with pytest.raises(ValueError, match="Unknown execution device"):
        demo.device_name("cuda")
    with pytest.raises(ValueError, match="Unknown execution device"):
        demo.device_name(5)
    assert demo.device_name("cpu") == "CPU"
    assert demo.device_name("cuda:2") == "GPU 2"
    assert demo.inference_device() == "cpu"
    assert demo.glimmer_health() == {"status": "not configured"}


def test_demo_config_accepts_string_paths(tmp_path):
    from siliconshield.web_demo import config as demo_config

    path = tmp_path / "demo.json"
    path.write_text("{}")
    assert demo_config.load_config(str(path))["port"] == 8765
    with pytest.raises(ValueError, match="must be a path"):
        demo_config.load_config(5)


def test_toy_runner_rejects_malformed_campaigns(tmp_path):
    from siliconshield.web_demo import toy_runner

    with pytest.raises(ValueError, match="must be a path"):
        toy_runner.execute(5)
    bad = tmp_path / "bad.json"
    bad.write_text("[1]")
    with pytest.raises(ValueError, match="operator object"):
        toy_runner.execute(bad)
    bad.write_text('{"x": 1}')
    with pytest.raises(ValueError, match="operator object"):
        toy_runner.execute(bad)
    bad.write_text('{"operator": {"endpoint": 5}}')
    with pytest.raises(ValueError, match="needs text endpoint"):
        toy_runner.execute(bad)


def test_training_status_survives_corrupt_receipts(tmp_path):
    import json

    from siliconshield.web_demo import backend

    root = tmp_path / "runs"
    run_id = "20261104-120000-abcdef"
    folder = root / run_id
    (folder / "runs" / "job-a").mkdir(parents=True)
    (folder / "campaign").mkdir(parents=True)
    (folder / "controller-result.json").write_text("[1, 2]")
    (folder / "campaign" / "campaign-status.json").write_text("[1]")
    (folder / "profile.json").write_text('{"defaults": [1]}')
    (folder / "runs" / "job-a" / "run-manifest.json").write_text("[1]")
    (folder / "runs" / "job-a" / "events.jsonl").write_text(
        '5\n{"event": "train", "step": 1}\n'
        '{"event": "evaluation", "metrics": [1]}\n'
    )
    (folder / "campaign" / "x").mkdir(parents=True)
    (folder / "campaign" / "x" / "operator-status.json").write_text(
        '{"experiment": "e", "cycles": [5, {"step": 1}]}'
    )
    state = tmp_path / "state"
    state.mkdir()
    pointer = {
        "id": run_id,
        "directory": str(folder),
        "unit": "siliconshield-web-toy-" + run_id,
        "status": "complete",
    }
    (state / "active-training.json").write_text(json.dumps(pointer))
    demo = backend.Demo(
        {
            "state_dir": str(state),
            "entries": [],
            "training_root": str(root),
            "training_enabled": False,
        },
        health_poll=False,
    )
    assert demo.training["id"] == run_id
    status = demo.training_status()
    assert status["active"] is False
    assert status["jobs"][0]["events"] == [
        {
            "event": "train",
            "step": 1,
            "loss": None,
            "step_seconds": None,
            "time": None,
        }
    ]
    assert status["operator_cycles"] == [{"experiment": "e", "step": 1}]
    assert status["campaign"] == {}
    stopped = demo.stop_training()
    assert stopped["status"] == "stop_requested"
    assert stopped["messages"][0].startswith("Unreadable launch profile: ")
    import sys

    (folder / "profile.json").write_text(
        json.dumps(
            {
                "python": sys.executable,
                "source": str(tmp_path),
                "model": str(tmp_path),
                "data": str(tmp_path),
                "runs": str(folder / "runs"),
                "defaults": {"max_updates": 10},
            }
        )
    )
    stopped = demo.stop_training()
    assert stopped["messages"] == ["Unreadable run manifest: run-manifest.json"]


def test_glimmer_client_rejects_malformed_arguments():
    from siliconshield.web_demo import glimmer_client

    endpoint = "http://127.0.0.1:1/chat/completions"
    with pytest.raises(ValueError, match="non-empty string"):
        glimmer_client.LocalGlimmerClient(5, "model")
    with pytest.raises(ValueError, match="audit directory"):
        glimmer_client.LocalGlimmerClient(endpoint, "model", audit_dir=5)
    client = glimmer_client.LocalGlimmerClient(endpoint, "model")
    with pytest.raises(ValueError, match="tools must be a list"):
        client.complete([], tools=5)


def test_toy_dataset_rejects_malformed_root(tmp_path):
    from siliconshield.web_demo import toy_data

    with pytest.raises(ValueError, match="must be a path"):
        toy_data.create_dataset(5)
    summary = toy_data.create_dataset(str(tmp_path / "toy"))
    assert summary["train"] == 18
