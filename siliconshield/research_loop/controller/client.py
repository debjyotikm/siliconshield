"""Call a chat endpoint and save the conversation and tool calls."""

import json
import os
import shlex
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path


def _is_loopback(url):
    """Return True for loopback-only HTTP destinations."""
    try:
        host = urllib.parse.urlsplit(url).hostname or ""
    except ValueError:
        raise ValueError("Chat endpoint is not a valid URL")
    return host in ("127.0.0.1", "::1", "localhost")


class ChatClient:
    """OpenAI-compatible chat client with optional auth and call budget.

    ``api_key`` (or the ``api_key_env`` environment variable) adds a
    Bearer token that is sent only over HTTPS, except to loopback
    addresses used by local services. Plaintext HTTP to any other
    host is refused. ``max_calls`` bounds how many completions one
    client may issue; the SSH transport carries credentials only
    inside the encrypted channel (never on a command line).
    """

    def __init__(
        self,
        endpoint,
        model,
        timeout=300,
        max_tokens=2048,
        adapter_path=None,
        checkpoint_fingerprint=None,
        ssh_host=None,
        api_key=None,
        api_key_env=None,
        max_calls=None,
    ):
        scheme = urllib.parse.urlsplit(endpoint).scheme
        if scheme not in ("http", "https"):
            raise ValueError("Chat endpoint must be http(s): " + endpoint[:64])
        if scheme == "http" and not _is_loopback(endpoint):
            raise ValueError(
                "Refusing plaintext HTTP to a non-loopback endpoint; "
                "use https or a loopback service"
            )
        if api_key is None and api_key_env:
            api_key = os.environ.get(api_key_env)
        if max_calls is not None and (
            isinstance(max_calls, bool)
            or not isinstance(max_calls, int)
            or max_calls < 1
        ):
            raise ValueError("max_calls must be a positive integer")
        self.endpoint, self.model = endpoint, model
        self.timeout, self.max_tokens = timeout, max_tokens
        # Identity of the weights actually served; names alone do not identify a checkpoint.
        self.adapter_path, self.checkpoint_fingerprint = (
            adapter_path,
            checkpoint_fingerprint,
        )
        self.ssh_host = ssh_host
        self.api_key = api_key
        self.max_calls = max_calls
        self.calls_made = 0
        self.last_completion_metrics = None

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        return headers

    def _check_budget(self):
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            raise RuntimeError(
                "Chat call budget exhausted (%d calls)" % self.max_calls
            )
        self.calls_made += 1

    def complete(self, messages, tools=()):
        self.last_completion_metrics = None
        self._check_budget()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers=self._headers(),
        )
        if self.ssh_host:
            # A fresh SSH channel per request avoids a long-lived forwarding
            # connection becoming an unobserved dependency of the operator.
            # Headers travel inside the encrypted channel via stdin, never
            # on a command line.
            script = (
                "import sys,json,urllib.request; "
                "env=json.load(sys.stdin); "
                "r=urllib.request.Request(sys.argv[1],"
                "data=json.dumps(env['payload']).encode(),"
                "headers=env['headers']); "
                "sys.stdout.buffer.write(urllib.request.urlopen(r,timeout=float(sys.argv[2])).read())"
            )
            command = "python3 -c %s %s %s" % (
                shlex.quote(script),
                shlex.quote(self.endpoint),
                self.timeout,
            )
            response = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    "-o",
                    "ServerAliveInterval=30",
                    self.ssh_host,
                    command,
                ],
                input=json.dumps(
                    {"headers": self._headers(), "payload": payload}
                ),
                text=True,
                capture_output=True,
                timeout=self.timeout + 30,
                check=True,
            )
            result = json.loads(response.stdout)
        else:
            with urllib.request.urlopen(
                request, timeout=self.timeout
            ) as response:
                result = json.load(response)
        choice = result["choices"][0]
        self.last_completion_metrics = {
            "usage": result.get("usage"),
            "timing": result.get("timing"),
        }
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "The model response was cut short; increase the token or context limit"
            )
        message = choice["message"]
        for call in message.get("tool_calls", []):
            args = call["function"]["arguments"]
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args)
            if not isinstance(call["function"]["arguments"], dict):
                raise ValueError("Tool arguments must be an object")
        if not message.get("content") and not message.get("tool_calls"):
            raise ValueError("Empty model completion")
        return message


def run_agent(client, messages, tools, dispatch, trace_path, max_turns=20):
    """Run one tool-use conversation, up to the configured turn limit."""
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    history = [dict(m) for m in messages]

    def record(value):
        with trace_path.open("a") as stream:
            stream.write(
                json.dumps({"time": time.time(), **value}, allow_nan=False)
                + "\n"
            )

    record(
        {
            "kind": "start",
            "messages": history,
            "tools": tools,
            "model": client.model,
        }
    )
    for _ in range(max_turns):
        started = time.perf_counter()
        message = client.complete(history, tools)
        history.append(message)
        record(
            {
                "kind": "message",
                "message": message,
                "completion_seconds": time.perf_counter() - started,
                "completion_metrics": getattr(
                    client, "last_completion_metrics", None
                ),
            }
        )
        calls = message.get("tool_calls", [])
        if not calls:
            return history
        for call in calls:
            try:
                result = dispatch(
                    call["function"]["name"], call["function"]["arguments"]
                )
            except (KeyError, ValueError, OSError) as exc:
                result = {"error": type(exc).__name__, "detail": str(exc)}
            tool_result = {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result, allow_nan=False),
            }
            history.append(tool_result)
            record({"kind": "message", "message": tool_result})
    raise RuntimeError(
        "Replay did not finish within its configured turn budget"
    )
