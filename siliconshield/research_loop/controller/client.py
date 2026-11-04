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
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("Chat endpoint must be a non-empty URL")
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
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("Chat timeout must be a positive number")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
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
        try:
            choice = result["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Chat response has no choices: %s" % exc) from exc
        if not isinstance(choice, dict):
            raise ValueError("Chat response choice must be an object")
        self.last_completion_metrics = {
            "usage": result.get("usage"),
            "timing": result.get("timing"),
        }
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "The model response was cut short; increase the token or context limit"
            )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Chat response has no message object")
        # Validate only: the message stays in wire format so the next
        # turn can echo it. Strict servers reject parsed arguments.
        calls = message.get("tool_calls", [])
        if calls and not isinstance(calls, list):
            raise ValueError("Model tool_calls must be a list")
        # An explicit null is the wire encoding of no calls.
        for call in calls or []:
            if not isinstance(call, dict):
                raise ValueError("Tool call must be an object")
            function = call.get("function")
            if not isinstance(function, dict):
                raise ValueError("Tool call must name a function")
            args = function.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    raise ValueError("Tool arguments must be an object")
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must be an object")
        if not message.get("content") and not message.get("tool_calls"):
            raise ValueError("Empty model completion")
        return message


def run_agent(client, messages, tools, dispatch, trace_path, max_turns=20):
    """Run one tool-use conversation, up to the configured turn limit.

    Malformed tool calls become tool errors the model can see and
    retry; failures that stop the loop are recorded in the trace
    before they propagate.
    """
    if (
        isinstance(max_turns, bool)
        or not isinstance(max_turns, int)
        or max_turns < 1
    ):
        raise ValueError("max_turns must be a positive integer")
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    history = [dict(m) for m in messages]

    def record(value):
        try:
            line = json.dumps({"time": time.time(), **value}, allow_nan=False)
        except (TypeError, ValueError):
            line = json.dumps(
                {
                    "time": time.time(),
                    "kind": value.get("kind"),
                    "unserializable": repr(value),
                },
                allow_nan=False,
            )
        with trace_path.open("a") as stream:
            stream.write(line + "\n")

    record(
        {
            "kind": "start",
            "messages": history,
            "tools": tools,
            "model": client.model,
            "checkpoint": {
                "adapter_path": getattr(client, "adapter_path", None),
                "checkpoint_fingerprint": getattr(
                    client, "checkpoint_fingerprint", None
                ),
            },
            "client_settings": {
                "timeout": getattr(client, "timeout", None),
                "max_tokens": getattr(client, "max_tokens", None),
                "max_calls": getattr(client, "max_calls", None),
            },
        }
    )
    try:
        for _ in range(max_turns):
            started = time.perf_counter()
            message = client.complete(history, tools)
            if not isinstance(message, dict):
                raise ValueError("Model completion must be an object")
            # Model output is always assistant content: a forged
            # role=tool message must never pair with a real tool
            # call or satisfy the evidence gate downstream.
            message = dict(message)
            message["role"] = "assistant"
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
            # Malformed calls are reported as a user message, never as a
            # tool result: a tool message must reference a real call id,
            # and strict OpenAI-compatible servers reject histories whose
            # tool messages reference unknown ids. The trajectory keeps
            # the original malformed message; only the live history is
            # rewritten so the next turn stays valid everywhere.
            assistant_index = len(history) - 1
            if not isinstance(calls, list):
                history[assistant_index] = {
                    "role": "user",
                    "content": (
                        "Tool error (ValueError): model tool_calls "
                        "must be a list; reply again."
                    ),
                }
                record({"kind": "message", "message": history[assistant_index]})
                continue
            valid_calls = [
                call
                for call in calls
                if isinstance(call, dict)
                and isinstance(call.get("id"), str)
                and call.get("id")
            ]
            if len(valid_calls) != len(calls):
                history[assistant_index] = {
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": valid_calls,
                }
                history.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool error (ValueError): every tool call "
                            "needs a string id; calls without one were "
                            "dropped. Reply again."
                        ),
                    }
                )
                record({"kind": "message", "message": history[-1]})
                calls = valid_calls
                if not calls:
                    continue
            for call in calls:
                call_id = call["id"]
                try:
                    if not isinstance(call, dict):
                        raise ValueError("Tool call must be an object")
                    function = call.get("function")
                    if not isinstance(function, dict):
                        raise ValueError("Tool call must name a function")
                    name = function.get("name")
                    if not isinstance(name, str) or not name:
                        raise ValueError("Tool call must name a function")
                    args = function.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            raise ValueError("Tool arguments must be an object")
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be an object")
                    result = dispatch(name, args)
                    content = json.dumps(result, allow_nan=False)
                except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                    content = json.dumps(
                        {"error": type(exc).__name__, "detail": str(exc)}
                    )
                tool_result = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }
                history.append(tool_result)
                record({"kind": "message", "message": tool_result})
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        record(
            {
                "kind": "error",
                "error": type(exc).__name__,
                "detail": str(exc),
            }
        )
        raise
    record(
        {
            "kind": "error",
            "error": "RuntimeError",
            "detail": "Turn budget exhausted without a final answer",
        }
    )
    raise RuntimeError("Agent did not finish within its configured turn budget")
