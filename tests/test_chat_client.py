"""ChatClient auth, transport rules, and call budgets (no network)."""

import json
import urllib.request

import pytest

from siliconshield.research_loop.controller.client import ChatClient


def _ok_response(monkeypatch, seen):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "done"}}],
                    "usage": {"prompt_tokens": 1},
                }
            ).encode()

    def fake_urlopen(request, timeout=None):
        seen.append(dict(request.headers))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_bearer_header_sent_only_when_configured(monkeypatch):
    seen = []
    _ok_response(monkeypatch, seen)
    client = ChatClient(
        "http://127.0.0.1:18320/v1/chat/completions",
        "test-model",
        api_key="secret-token",
    )
    assert client.complete([{"role": "user", "content": "hi"}])
    assert seen[0].get("Authorization") == "Bearer secret-token"
    seen.clear()
    plain = ChatClient(
        "http://127.0.0.1:18320/v1/chat/completions", "test-model"
    )
    plain.complete([{"role": "user", "content": "hi"}])
    assert "Authorization" not in seen[0]


def test_api_key_env_resolves_without_logging(monkeypatch, tmp_path):
    monkeypatch.setenv("SILICONSHIELD_TEST_API_KEY", "env-token")
    seen = []
    _ok_response(monkeypatch, seen)
    client = ChatClient(
        "http://127.0.0.1:18320/v1/chat/completions",
        "test-model",
        api_key_env="SILICONSHIELD_TEST_API_KEY",
    )
    client.complete([{"role": "user", "content": "hi"}])
    assert seen[0].get("Authorization") == "Bearer env-token"


def test_plaintext_http_refused_off_loopback():
    with pytest.raises(ValueError, match="non-loopback"):
        ChatClient("http://models.example.com/v1", "test-model")
    # Loopback services and everything over https stay allowed.
    ChatClient("http://127.0.0.1:18320/v1", "test-model")
    ChatClient("https://models.example.com/v1", "test-model")


def test_call_budget_is_enforced(monkeypatch):
    seen = []
    _ok_response(monkeypatch, seen)
    client = ChatClient("http://127.0.0.1:1/v1", "test-model", max_calls=2)
    client.complete([{"role": "user", "content": "a"}])
    client.complete([{"role": "user", "content": "b"}])
    assert client.calls_made == 2
    with pytest.raises(RuntimeError, match="budget exhausted"):
        client.complete([{"role": "user", "content": "c"}])
    with pytest.raises(ValueError, match="positive integer"):
        ChatClient("http://127.0.0.1:1/v1", "m", max_calls=0)


def test_ssh_transport_carries_headers_off_the_command_line(
    monkeypatch,
):
    import subprocess

    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["input"] = kwargs.get("input", "")
        assert "secret-token" not in " ".join(argv)

        class Result:
            stdout = json.dumps({"choices": [{"message": {"content": "done"}}]})

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    client = ChatClient(
        "https://models.example.com/v1",
        "test-model",
        ssh_host="gpu-host",
        api_key="secret-token",
    )
    assert client.complete([{"role": "user", "content": "hi"}])
    envelope = json.loads(captured["input"])
    assert envelope["headers"]["Authorization"] == "Bearer secret-token"
    assert envelope["payload"]["model"] == "test-model"
