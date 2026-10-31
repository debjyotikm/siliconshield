"""Run isolated Muse coding-agent completions for procedure replay.

Muse emits messages describing replay tool calls. The Python replay dispatcher
executes those calls; Muse receives no native filesystem, shell, or job tools.
The existing Muse login is used without copying credentials into artifacts.
"""

import copy
import hashlib
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from siliconshield.research_loop.episodes.contracts import atomic_json
from siliconshield.research_loop.execution.campaign_supervisor import (
    parse_exec_jsonl,
)


class MuseClient:
    """Expose Muse CLI responses through the replay client's message interface."""

    endpoint = "muse-cli"

    def __init__(
        self,
        executable: Path,
        audit_dir: Path,
        model: str = "muse-spark-1.3-contributor",
        effort: str = "high",
        timeout: int = 300,
    ) -> None:
        """Bind the executable and settings; every call gets a fresh workspace."""
        self.executable = executable.resolve(strict=True)
        self.audit_dir = audit_dir.resolve()
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.model, self.effort, self.timeout = model, effort, timeout
        if timeout <= 0:
            raise ValueError("Muse completion timeout must be positive")
        self.execution_identity: dict[str, Any] = {
            "backend": "muse-exec-empty-preset-v2",
            "model": model,
            "executable_sha256": hashlib.sha256(
                self.executable.read_bytes()
            ).hexdigest(),
            "reasoning_effort": effort,
            "timeout_seconds": timeout,
            "max_model_steps": 1,
            "weight_identity": "provider-managed; not locally verifiable",
        }
        self.last_completion_metrics: dict[str, Any] | None = None

    def argv(self, workspace: Path, prompt: Path) -> list[str]:
        """Build a tool-free request without inheriting campaign permissions."""
        return [
            str(self.executable),
            "exec",
            "--model",
            self.model,
            "--preset",
            "empty",
            "--json",
            "--max-model-steps",
            "1",
            "--reasoning-effort",
            self.effort,
            "--disable-shell",
            "--disable-write",
            "--disable-web-tools",
            "--disable-reminders",
            "--disable-request-user-input",
            "--no-foreign-personal-context",
            "--no-session-log",
            "--approval-mode",
            "never",
            "--workspace",
            str(workspace),
            "--prompt-file",
            str(prompt),
            "--output-schema",
            str(prompt.parent / "response-schema.json"),
        ]

    def complete(
        self, messages: list[dict[str, Any]], tools: Any = ()
    ) -> dict[str, Any]:
        """Return one validated assistant message and save its CLI receipt."""
        self.last_completion_metrics = None
        if (
            hashlib.sha256(self.executable.read_bytes()).hexdigest()
            != (self.execution_identity["executable_sha256"])
        ):
            raise ValueError("Muse executable changed during the trial")
        attempt = Path(tempfile.mkdtemp(prefix="call-", dir=self.audit_dir))
        workspace = attempt / "workspace"
        workspace.mkdir()
        instruction = (
            "Respond to the supplied conversation as its next assistant. "
            "Return ONLY a JSON object with role=assistant, content (text, "
            "a JSON object, or null), and tool_calls (array). To use a replay "
            "tool, emit {id, type: function, function: {name, arguments: "
            "object}} in tool_calls. A separate offline dispatcher runs it; "
            "you have no native tools. Do not fabricate tool results. For a "
            "final answer, place it in content and use tool_calls=[]. "
            "For JSON answers or harness edits, put the OBJECT directly in "
            "content; do not manually escape or stringify it. Optional tool "
            "arguments use null when absent. In an operator assessment, encode "
            "comparisons as an array of objects with view, accuracy_delta_pp, "
            "nll_delta, false_safe_delta_pp. The dispatcher converts this array "
            "to the tool's mapping format without changing values.\n"
        )
        prompt = attempt / "prompt.txt"
        prompt.write_text(
            instruction
            + json.dumps(
                {"messages": messages, "tools": tools}, allow_nan=False
            )
        )
        atomic_json(attempt / "response-schema.json", response_schema(tools))
        argv = self.argv(workspace, prompt)
        with (
            (attempt / "stdout.jsonl").open("w") as stdout,
            (attempt / "stderr.txt").open("w") as stderr,
        ):
            process = subprocess.Popen(
                argv,
                cwd=workspace,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=self.timeout)
            except BaseException:
                # Reap the whole *owned* process group on timeout/cancellation.
                # Never search for or signal a campaign process by name.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                atomic_json(attempt / "receipt.json", {"completed": False})
                raise
        parsed = parse_exec_jsonl((attempt / "stdout.jsonl").read_text())
        atomic_json(
            attempt / "receipt.json",
            {"identity": self.execution_identity, "exit_code": code, **parsed},
        )
        if code != 0 or parsed["terminal"] != "completed":
            raise RuntimeError(
                "Muse procedure completion failed: " + parsed["terminal"]
            )
        self.last_completion_metrics = parsed.get("usage", {})
        return parse_message(parsed["text"], tools)


def closed_object(properties: dict[str, Any]) -> dict[str, Any]:
    """Describe a provider-compatible object with explicit required fields."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def operator_assessment_schema() -> dict[str, Any]:
    """Represent variable view names as entries, without encoding any answers."""
    comparison = closed_object(
        {
            "view": {"type": "string"},
            "accuracy_delta_pp": {"type": ["number", "null"]},
            "nll_delta": {"type": ["number", "null"]},
            "false_safe_delta_pp": {"type": ["number", "null"]},
        }
    )
    return closed_object(
        {
            "execution_failed": {"type": ["boolean", "null"]},
            "acceptance": {
                "type": "string",
                "enum": ["passed", "failed", "unknown"],
            },
            "selected_step": {"type": ["integer", "null"]},
            "plateau_reductions": {"type": ["integer", "null"]},
            "comparisons": {"type": "array", "items": comparison},
        }
    )


def argument_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Encode optional arguments as nullable required fields for the provider."""
    schema = copy.deepcopy(tool["function"]["parameters"])
    properties = schema["properties"]
    if tool["function"]["name"] == "operator.decide":
        properties["assessment"] = operator_assessment_schema()
    required = set(schema.get("required", []))
    for name, description in properties.items():
        if name not in required:
            properties[name] = {"anyOf": [description, {"type": "null"}]}
    return closed_object(properties)


def content_schema(tools: Any) -> dict[str, Any]:
    """Allow structured final answers without JSON nested inside JSON strings."""
    names = {tool["function"]["name"] for tool in tools}
    if "operator.decide" in names:
        structured = closed_object(
            {
                "action": {"type": "string"},
                "run_id": {"type": "string"},
                "executed": {"type": "boolean"},
                "scope": {"type": "string"},
            }
        )
    elif "research.record_decision" in names:
        structured = closed_object(
            {
                "action": {"type": "string"},
                "agreement_delta_pp": {"type": "number"},
                "nll_delta": {"type": "number"},
                "scope": {"type": "string"},
            }
        )
    else:
        structured = closed_object(
            {
                "files": closed_object(
                    {"LongTermMEMORY.md": {"type": "string"}}
                ),
                "notes": {"type": "string"},
            }
        )
    return {"anyOf": [{"type": "string"}, {"type": "null"}, structured]}


def response_schema(tools: Any) -> dict[str, Any]:
    """Constrain the provider's final response to the replay message shape."""
    functions = [
        {
            "type": "object",
            "properties": {
                "name": {"const": tool["function"]["name"]},
                "arguments": argument_schema(tool),
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        for tool in tools
    ]
    calls: dict[str, Any] = {"type": "array", "maxItems": 0}
    if functions:
        calls = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "type": {"const": "function"},
                    "function": {"anyOf": functions},
                },
                "required": ["id", "type", "function"],
                "additionalProperties": False,
            },
        }
    return {
        "type": "object",
        "properties": {
            "role": {"const": "assistant"},
            "content": content_schema(tools),
            "tool_calls": calls,
        },
        "required": ["role", "content", "tool_calls"],
        "additionalProperties": False,
    }


def parse_message(text: str, tools: Any) -> dict[str, Any]:
    """Reject malformed messages or attempts to call unadvertised tools."""
    message = json.loads(text)
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("Expected an assistant message object")
    content, calls = message.get("content"), message.get("tool_calls", [])
    if isinstance(content, dict):
        content = json.dumps(content, allow_nan=False)
    if content is not None and not isinstance(content, str):
        raise ValueError("Assistant content must be text or null")
    if not isinstance(calls, list) or (not content and not calls):
        raise ValueError("Empty or malformed assistant message")
    allowed = {tool["function"]["name"] for tool in tools}
    identifiers = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError("Malformed replay tool call")
        identifier, function = call.get("id"), call.get("function")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or not isinstance(function, dict)
            or function.get("name") not in allowed
            or not isinstance(function.get("arguments"), dict)
        ):
            raise ValueError("Unknown tool or malformed replay arguments")
        identifiers.add(identifier)
        if function["name"] == "operator.decide":
            decode_operator_arguments(function["arguments"])
    return {"role": "assistant", "content": content, "tool_calls": calls}


def decode_operator_arguments(arguments: dict[str, Any]) -> None:
    """Restore the tool contract without inferring actions, values, or evidence."""
    for name in ("lr", "checkpoint_dir", "assessment"):
        if arguments.get(name) is None:
            arguments.pop(name, None)
    assessment = arguments.get("assessment")
    if not isinstance(assessment, dict):
        return
    comparisons = assessment.get("comparisons")
    if not isinstance(comparisons, list):
        return
    mapped = {}
    for item in comparisons:
        if not isinstance(item, dict) or set(item) != {
            "view",
            "accuracy_delta_pp",
            "nll_delta",
            "false_safe_delta_pp",
        }:
            raise ValueError("Malformed operator comparison entry")
        name = item["view"]
        if not isinstance(name, str) or not name or name in mapped:
            raise ValueError("Duplicate or invalid comparison view")
        mapped[name] = {
            key: value for key, value in item.items() if key != "view"
        }
    assessment["comparisons"] = mapped
