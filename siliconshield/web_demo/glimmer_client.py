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
        self.calls += 1
        messages = copy.deepcopy(messages)
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = messages[0]["content"].replace(
                "Reasoning strength: high.", "Reasoning strength: low."
            )
            messages[0]["content"] += (
                "\nReasoning strength: low. Keep explanations concise, at most 100 words. Complete the required tool call."
            )
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
            decided = any(
                call.get("id") == last.get("tool_call_id")
                and call.get("function", {}).get("name") == "operator.decide"
                for message in messages[:-1]
                for call in message.get("tool_calls") or []
            )
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
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "Glimmer exhausted the response budget before a complete action."
            )
        message = choice["message"]
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            parsed = (
                json.loads(arguments)
                if isinstance(arguments, str)
                else arguments
            )
            if not isinstance(parsed, dict):
                raise ValueError(
                    "Glimmer tool arguments must be a JSON object."
                )
            if self.decode_arguments:
                call["function"]["arguments"] = parsed
        if not message.get("content") and not message.get("tool_calls"):
            raise ValueError("Empty Glimmer completion.")
        return cast(dict[str, Any], message)
