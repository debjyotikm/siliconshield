"""Drive expert-router assessments from a Glimmer chat session.

Glimmer reasons about the requirement and scope. Every verdict comes
from a routed expert through the standard assessment path. The skill
at .agents/skills/glimmer-router/SKILL.md is the system prompt.
"""

import argparse
import http.client
import json
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from siliconshield.experts.cli import run_assess
from siliconshield.experts.identity import assemble_state
from siliconshield.experts.service import InferenceService
from siliconshield.experts.worker import WorkerPool
from siliconshield.research_loop.controller.client import (
    ChatClient,
    run_agent,
)

DEFAULT_SKILL = (
    Path(__file__).resolve().parents[3]
    / ".agents/skills/glimmer-router/SKILL.md"
)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "router.roster",
            "description": (
                "List registered experts with supported mechanism "
                "tags, strengths, weaknesses, and measurements."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "router.assess",
            "description": (
                "Assess RTL against one requirement under a "
                "comma-separated scope. Returns the routing "
                "decision with every expert response or error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": (
                            "Comma-separated mechanism tags from the roster."
                        ),
                    },
                    "requirement": {
                        "type": "string",
                        "description": "The requirement under test.",
                    },
                    "assumptions": {
                        "type": "string",
                        "description": "Stated analysis assumptions.",
                    },
                    "rtl": {
                        "type": "string",
                        "description": "The RTL region to assess.",
                    },
                },
                "required": [
                    "scope",
                    "requirement",
                    "assumptions",
                    "rtl",
                ],
                "additionalProperties": False,
            },
        },
    },
]


class RouterTools:
    """Model-facing router tools; the registry stays operator-bound."""

    def __init__(
        self,
        registry_path: str,
        pool_factory: Any = WorkerPool,
        state_dir: str | None = None,
    ) -> None:
        self.registry_path = registry_path
        self.pool_factory = pool_factory
        self.state_dir = Path(state_dir) if state_dir else None

    def _roster(self) -> dict[str, Any]:
        service = InferenceService(self.registry_path)
        experts = []
        for expert_id in service.expert_ids:
            profile = service.profiles_for([expert_id])[expert_id]
            experts.append(
                {
                    "id": expert_id,
                    "release": profile["release"],
                    "supported_mechanisms": (profile["supported_mechanisms"]),
                    "strengths": profile["strengths"],
                    "weaknesses": profile["weaknesses"],
                    "known_failures": profile["known_failures"],
                    "measurements": profile["measurements"],
                    "qualification_ref": (profile["qualification_ref"]),
                }
            )
        return {
            "experts": experts,
            "broken_experts": [
                {"id": expert_id, "detail": detail}
                for expert_id, detail in sorted(service.broken_experts.items())
            ],
            "unprofiled_experts": service.unprofiled_experts,
        }

    def _assess(self, args: dict[str, Any]) -> dict[str, Any]:
        fields = {}
        for key in ("scope", "requirement", "assumptions", "rtl"):
            value = args.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("router.assess needs non-empty %r" % key)
            fields[key] = value.strip()
        tags = [
            tag.strip() for tag in fields["scope"].split(",") if tag.strip()
        ]
        if not tags:
            raise ValueError("router.assess needs a scope tag")
        state = assemble_state(
            fields["requirement"], fields["assumptions"], fields["rtl"]
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            dir=str(self.state_dir) if self.state_dir else None,
            delete=False,
        ) as handle:
            handle.write(state)
            target = Path(handle.name)
        namespace = argparse.Namespace(
            state_file=str(target),
            scope=",".join(tags),
            registry=self.registry_path,
            arbitrate=False,
            requirement="",
            issue="",
            relevance="",
            rtl=[],
        )
        try:
            report = run_assess(namespace, pool_factory=self.pool_factory)
        except RuntimeError as exc:
            raise ValueError("assessment failed: %s" % exc) from exc
        finally:
            target.unlink(missing_ok=True)
        assessment: dict[str, Any] = report["assessment"]
        return assessment

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call; unknown tools and arguments raise ValueError."""
        if name == "router.roster":
            if args:
                raise ValueError(
                    "router.roster takes no arguments, got %s" % sorted(args)
                )
            return self._roster()
        if name == "router.assess":
            unknown = set(args) - {
                "scope",
                "requirement",
                "assumptions",
                "rtl",
            }
            if unknown:
                raise ValueError(
                    "router.assess takes no extra arguments %s"
                    % sorted(unknown)
                )
            return self._assess(args)
        raise ValueError("Unknown tool: %s" % name)


def _successful_assessments(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect router.assess tool results that carry expert responses.

    Each result pairs with the earliest unmatched call carrying its
    id, so duplicate ids pair positionally instead of collapsing.
    """
    pending: list[list[Any]] = []
    for message in history:
        for call in message.get("tool_calls", []) or []:
            if isinstance(call, dict):
                pending.append(
                    [
                        call.get("id"),
                        (call.get("function") or {}).get("name"),
                        False,
                    ]
                )
    found: list[dict[str, Any]] = []
    for message in history:
        if message.get("role") != "tool":
            continue
        target = message.get("tool_call_id")
        name = None
        for entry in pending:
            if not entry[2] and entry[0] == target:
                entry[2] = True
                name = entry[1]
                break
        if name != "router.assess":
            continue
        try:
            payload = json.loads(message.get("content", ""))
        except ValueError:
            continue
        responses = (
            payload.get("responses") if isinstance(payload, dict) else None
        )
        if isinstance(responses, list) and responses:
            found.append(payload)
    return found


def _names_token(final: str, token: str) -> bool:
    """Whether the verdict names one token exactly, not a longer word.

    A leading dot still disqualifies (dotted paths are not bare
    citations), but a trailing dot is sentence punctuation: verdicts
    routinely end with a period.
    """
    return (
        re.search(
            r"(?<![A-Za-z0-9_.-])%s(?![A-Za-z0-9_-])" % re.escape(token),
            final,
        )
        is not None
    )


def run_router_session(
    client: Any,
    registry: str,
    task: str,
    output: str,
    *,
    skill: str | None = None,
    max_turns: int = 12,
    pool_factory: Any = WorkerPool,
) -> list[dict[str, Any]]:
    """Run one Glimmer-driven router session; return the history.

    The registry, pool factory, and output dir are validated before
    any model call, and the final verdict is accepted only when the
    session holds at least one successful router.assess result
    naming expert responses, with every consulted expert id and
    label present in the verdict text. Anything else raises
    ValueError: no verdict is ever returned without evidence
    behind it.
    """
    skill_path = Path(skill) if skill else DEFAULT_SKILL
    try:
        system = skill_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("Cannot read skill file: %s" % exc) from exc
    if not task.strip():
        raise ValueError("The session task must be non-empty")
    if not output or not str(output).strip():
        raise ValueError("The session output dir must be non-empty")
    InferenceService(registry)
    try:
        probe = pool_factory()
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        raise ValueError("Cannot construct expert pool: %s" % exc) from exc
    close = getattr(probe, "close", None)
    if callable(close):
        close()
    out = Path(output)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            "Cannot use session output dir %s: %s" % (output, exc)
        ) from exc
    tools = RouterTools(registry, pool_factory, str(out))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task.strip()},
    ]
    trace = out / ("trace-" + uuid.uuid4().hex + ".jsonl")
    history: list[dict[str, Any]] = run_agent(
        client, messages, TOOL_SCHEMAS, tools.dispatch, str(trace), max_turns
    )
    assessments = _successful_assessments(history)
    if not assessments:
        raise ValueError(
            "Session produced no successful router.assess result; "
            "refusing an evidence-free verdict."
        )
    final = history[-1].get("content", "") if history else ""
    if not isinstance(final, str):
        raise ValueError("Session final message has no verdict text.")
    for assessment in assessments:
        responses = assessment.get("responses")
        if not isinstance(responses, list):
            raise ValueError("A router.assess result has no response list.")
        for response in responses:
            if not isinstance(response, dict):
                raise ValueError("A router.assess response is malformed.")
            expert_id = response.get("expert_id")
            label = response.get("label")
            if (
                not isinstance(expert_id, str)
                or not expert_id
                or not isinstance(label, str)
                or not label
            ):
                raise ValueError(
                    "A router.assess response names no expert or label."
                )
            if not _names_token(final, expert_id) or not _names_token(
                final, label
            ):
                raise ValueError(
                    "Final verdict does not report expert %r with "
                    "label %r." % (expert_id, label)
                )
    return history


def build_parser() -> argparse.ArgumentParser:
    """Build the session CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run one Glimmer-driven router session."
    )
    parser.add_argument("--registry", required=True, help="Registry path.")
    parser.add_argument("--endpoint", required=True, help="Chat endpoint URL.")
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument(
        "--ssh-host", default=None, help="SSH host for the endpoint."
    )
    parser.add_argument(
        "--skill", default=None, help="Skill file (default shipped)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", default=None, help="Task text.")
    group.add_argument("--task-file", default=None, help="Task file.")
    parser.add_argument("--output", required=True, help="Output dir.")
    parser.add_argument(
        "--max-turns", type=int, default=12, help="Turn budget."
    )
    parser.add_argument(
        "--max-calls", type=int, default=None, help="Chat call budget."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Completion token budget.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; return the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        if args.task_file:
            task = Path(args.task_file).read_text(encoding="utf-8")
        else:
            task = args.task or ""
        client = ChatClient(
            args.endpoint,
            args.model,
            ssh_host=args.ssh_host,
            max_calls=args.max_calls,
            max_tokens=args.max_tokens,
        )
        history = run_router_session(
            client,
            args.registry,
            task,
            args.output,
            skill=args.skill,
            max_turns=args.max_turns,
        )
    except (
        ValueError,
        RuntimeError,
        OSError,
        KeyError,
        IndexError,
        TypeError,
        subprocess.SubprocessError,
        http.client.HTTPException,
    ) as exc:
        print("glimmer-router: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(history[-1], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
