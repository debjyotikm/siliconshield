"""Bounded local Glimmer transport for arbitration and the toy operator."""

import copy
import json
import re
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse


class LocalGlimmerClient:
    """Use a loopback endpoint directly or through a configured SSH host."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        ssh_host: str | None = None,
        *,
        max_tokens: int = 2048,
        timeout: float = 180,
        max_calls: int = 8,
        audit_dir: Path | None = None,
        decode_arguments: bool = False,
    ) -> None:
        parsed = urlparse(endpoint)
        if (
            parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.scheme != "http"
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not parsed.path.endswith("/chat/completions")
        ):
            raise ValueError(
                "Glimmer endpoint must be loopback HTTP ending in /chat/completions"
            )
        if not isinstance(model, str) or not model.strip():
            raise ValueError("A served Glimmer model name is required")
        if ssh_host is not None and (
            not isinstance(ssh_host, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", ssh_host)
        ):
            raise ValueError(
                "ssh_host must be a local SSH configuration alias or user@host"
            )
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("timeout must be positive")
        if (
            isinstance(max_calls, bool)
            or not isinstance(max_calls, int)
            or max_calls < 1
        ):
            raise ValueError("max_calls must be a positive integer")
        self.endpoint, self.model, self.ssh_host = endpoint, model, ssh_host
        self.timeout, self.max_tokens, self.max_calls = (
            timeout,
            max_tokens,
            max_calls,
        )
        self.calls = 0
        self.last_completion_metrics: dict[str, Any] | None = None
        self.audit_dir = Path(audit_dir) if audit_dir else None
        self.decode_arguments = decode_arguments

    def _request(
        self, url: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute one bounded request without changing service configuration."""
        if self.ssh_host:
            script = (
                "import sys,json,urllib.request; p=json.load(sys.stdin); "
                "body=None if p is None else json.dumps(p).encode(); "
                'r=urllib.request.Request(sys.argv[1],data=body,headers={"Content-Type":"application/json"}); '
                "sys.stdout.buffer.write(urllib.request.urlopen(r,timeout=float(sys.argv[2])).read())"
            )
            command = (
                "python3 -c "
                + shlex.quote(script)
                + " "
                + shlex.quote(url)
                + " "
                + str(float(self.timeout))
            )
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=8",
                    self.ssh_host,
                    command,
                ],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.timeout + 15,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    "Local Glimmer request failed: " + result.stderr[-1200:]
                )
            return cast(dict[str, Any], json.loads(result.stdout))
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return cast(dict[str, Any], json.load(response))

    def models(self) -> dict[str, Any]:
        """Check the served roster without consuming a generation call."""
        return self._request(
            self.endpoint.removesuffix("/chat/completions") + "/models"
        )

    def complete(
        self, messages: list[dict[str, Any]], tools: Any = ()
    ) -> dict[str, Any]:
        """Return the real model response, refusing incomplete tool output."""
        if self.calls >= self.max_calls:
            raise RuntimeError("Local demo agent-call budget exhausted.")
        if not isinstance(messages, list) or any(
            not isinstance(item, dict) for item in messages
        ):
            raise ValueError("Chat messages must be a list of objects.")
        messages = copy.deepcopy(messages)
        if messages and messages[0].get("role") == "system":
            if not isinstance(messages[0].get("content"), str):
                raise ValueError("System message content must be text.")
            messages[0]["content"] = messages[0]["content"].replace(
                "Reasoning strength: high.", "Reasoning strength: low."
            )
            messages[0]["content"] += (
                "\nReasoning strength: low. Keep explanations concise, at most 100 words. Complete the required tool call."
            )
        # Rejected inputs never reach the service, so they must not
        # consume the fixed agent-call budget.
        self.calls += 1
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"reasoning_strength": "low"},
        }
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        # The final acknowledgement is still generated and validated, never fabricated.
        if messages and messages[-1].get("role") == "tool":
            last = messages[-1]
            decided = False
            for prior in messages[:-1]:
                calls = prior.get("tool_calls") or []
                if not isinstance(calls, list):
                    continue
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function")
                    if not isinstance(function, dict):
                        continue
                    if (
                        call.get("id") == last.get("tool_call_id")
                        and function.get("name") == "operator.decide"
                    ):
                        decided = True
                        break
                if decided:
                    break
            try:
                recorded = (
                    json.loads(last.get("content", "{}")).get("recorded")
                    is True
                )
            except (ValueError, AttributeError):
                recorded = False
            if decided and recorded:
                payload.update(
                    response_format={"type": "json_object"}, tool_choice="none"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "The decision is recorded. Finish with only a JSON object containing exactly action, run_id, executed and scope. "
                        "Copy action and run_id from the recorded operator.decide call. "
                        "Set executed to false and scope to operator_replay. Do not call tools. No Markdown or explanation.",
                    }
                )
        started = time.monotonic()
        data = self._request(self.endpoint, payload)
        if not isinstance(data, dict):
            raise ValueError("Glimmer response must be a JSON object.")
        self.last_completion_metrics = {
            "usage": data.get("usage"),
            "seconds": time.monotonic() - started,
        }
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            target = self.audit_dir / (str(time.time_ns()) + ".json")
            target.write_text(
                json.dumps(
                    {
                        "request": payload,
                        "response": data,
                        "timing": self.last_completion_metrics,
                    },
                    indent=2,
                )
                + "\n"
            )
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "Glimmer response has no usable choice: %s" % exc
            ) from exc
        if not isinstance(choice, dict):
            raise ValueError("Glimmer response choice must be an object.")
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "Glimmer exhausted the response budget before a complete action."
            )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Glimmer response has no message object.")
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise ValueError("Glimmer tool calls must be a list.")
        for call in tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                raise ValueError("Glimmer tool call must name a function.")
            arguments = function.get("arguments")
            try:
                parsed = (
                    json.loads(arguments)
                    if isinstance(arguments, str)
                    else arguments
                )
            except ValueError as exc:
                raise ValueError(
                    "Glimmer tool arguments must be a JSON object: %s" % exc
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    "Glimmer tool arguments must be a JSON object."
                )
            if self.decode_arguments:
                function["arguments"] = parsed
        if not message.get("content") and not message.get("tool_calls"):
            raise ValueError("Empty Glimmer completion.")
        return cast(dict[str, Any], message)
