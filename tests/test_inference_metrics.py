"""Token-level timing must account for prompts and speculative blocks."""

import fcntl
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from siliconshield.research_loop.controller.client import run_agent
from siliconshield.research_loop.controller.inference_metrics import (
    GenerationStreamer,
    TokenTiming,
)
from siliconshield.research_loop.controller.serve_fast import ServerConfig


def test_prefill_is_separate_and_prompt_tokens_are_not_generated() -> None:
    times = iter([12.0, 12.5, 13.0])
    timer = TokenTiming(10.0)
    streamer = GenerationStreamer(timer, clock=lambda: next(times))
    streamer.put(SimpleNamespace(numel=lambda: 4096))
    for _ in range(3):
        streamer.put(SimpleNamespace(numel=lambda: 1))
    streamer.end()
    result = timer.summary(13.0)
    assert result["generated_tokens"] == 3
    assert result["first_token_seconds"] == 2.0
    assert result["generation_tokens_per_second"] == 1.0
    assert result["decode_tokens_per_second"] == 2.0


def test_speculative_first_block_is_excluded_from_decode_rate() -> None:
    timer = TokenTiming(0.0)
    timer.record(4, 2.0)
    timer.record(3, 2.5)
    assert timer.summary(2.5)["decode_tokens_per_second"] == 6.0
    assert timer.summary(2.5)["generated_tokens"] == 7


def test_no_decode_interval_is_not_reported_as_zero_speed() -> None:
    timer = TokenTiming(0.0)
    assert timer.summary(1.0)["decode_tokens_per_second"] is None
    timer.record(1, 2.0)
    assert timer.summary(2.0)["decode_tokens_per_second"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_timestamps_are_rejected(value: float) -> None:
    timer = TokenTiming(0.0)
    with pytest.raises(ValueError):
        timer.record(1, value)
    with pytest.raises(ValueError):
        timer.summary(value)


def test_usage_is_saved_without_becoming_model_visible(tmp_path) -> None:
    messages = [{"role": "user", "content": "Authored fixture"}]

    class Client:
        model = "fixture"
        last_completion_metrics = {
            "usage": {"prompt_tokens": 20, "completion_tokens": 4}
        }

        def complete(self, history, tools):
            assert history == messages
            return {"role": "assistant", "content": "Done"}

    path = tmp_path / "trajectory.jsonl"
    result = run_agent(Client(), messages, [], lambda *a: None, path)
    event = json.loads(path.read_text().splitlines()[-1])
    assert event["completion_metrics"]["usage"]["completion_tokens"] == 4
    assert event["completion_seconds"] >= 0
    assert result[-1] == {"role": "assistant", "content": "Done"}


def test_native_inference_keeps_bf16_and_explicit_adapter_names() -> None:
    config = ServerConfig(
        "/fixture/model", adapters=(("before", "/fixture/adapter with spaces"),)
    )
    command = config.command()
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "before=/fixture/adapter with spaces" in command
    assert "--speculative-config" not in command
    assert command[command.index("--safetensors-load-strategy") + 1] == "lazy"


def test_speculation_requires_explicit_draft_configuration() -> None:
    command = ServerConfig("/fixture/model", draft="/fixture/draft").command()
    value = json.loads(command[command.index("--speculative-config") + 1])
    assert value == {
        "model": "/fixture/draft",
        "method": "dflash",
        "num_speculative_tokens": 15,
    }


def test_draft_capacity_covers_all_sequences() -> None:
    config = ServerConfig(
        "fixture", draft="draft", max_sequences=8, max_batched_tokens=127
    )
    with pytest.raises(ValueError, match="every draft position"):
        config.command()
    command = ServerConfig(
        "fixture", draft="draft", max_sequences=8, max_batched_tokens=128
    ).command()
    assert command[command.index("--max-num-batched-tokens") + 1] == "128"


@pytest.mark.parametrize("fraction", [float("nan"), 0, 1, 2])
def test_native_inference_rejects_invalid_memory_fraction(fraction) -> None:
    with pytest.raises(ValueError):
        ServerConfig("fixture", memory_fraction=fraction).command()


def test_native_inference_rejects_ambiguous_adapter_identity() -> None:
    with pytest.raises(ValueError, match="unique"):
        ServerConfig(
            "fixture", adapters=(("same", "a"), ("same", "b"))
        ).command()


def test_native_server_inherits_the_exclusive_gpu_lease(
    tmp_path, monkeypatch
) -> None:
    from siliconshield.research_loop.controller import serve_fast

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("SILICONSHIELD_GPU_LOCK", str(lock))
    monkeypatch.setattr(
        serve_fast,
        "verify_snapshot",
        lambda *a, **k: {"fingerprint": "fixture"},
    )
    monkeypatch.setattr(sys, "argv", ["serve_fast", "--model", str(model)])

    def exec_probe(executable, command):
        assert executable == "vllm"
        with lock.open("a") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inherited = [
            int(p.name)
            for p in Path("/proc/self/fd").iterdir()
            if p.exists() and p.resolve() == lock
        ]
        assert len(inherited) == 1
        assert os.get_inheritable(inherited[0])
        raise FileNotFoundError("Authored missing-runtime fixture")

    monkeypatch.setattr(os, "execvp", exec_probe)
    with pytest.raises(FileNotFoundError):
        serve_fast.main()
    with lock.open("a") as released:
        fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
