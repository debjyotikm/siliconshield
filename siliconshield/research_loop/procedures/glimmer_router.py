"""Drive expert-router assessments from a Glimmer chat session.

Glimmer reasons about the requirement and scope. Every verdict comes
from a routed expert through the standard assessment path. The skill
at .agents/skills/glimmer-router/SKILL.md is the system prompt.
"""

import argparse
import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from siliconshield.experts.cli import run_assess
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
        return {"experts": experts}

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
        state = (
            "REQUIREMENT\n"
            + fields["requirement"]
            + "\n\nASSUMPTIONS\n"
            + fields["assumptions"]
            + "\n\nRTL\n"
            + fields["rtl"]
            + "\n"
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
        assessment: dict[str, Any] = report["assessment"]
        return assessment

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call; unknown tools raise ValueError."""
        if name == "router.roster":
            return self._roster()
        if name == "router.assess":
            return self._assess(args)
        raise ValueError("Unknown tool: %s" % name)


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
    """Run one Glimmer-driven router session; return the history."""
    skill_path = Path(skill) if skill else DEFAULT_SKILL
    try:
        system = skill_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("Cannot read skill file: %s" % exc) from exc
    if not task.strip():
        raise ValueError("The session task must be non-empty")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    tools = RouterTools(registry, pool_factory, str(out))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task.strip()},
    ]
    trace = out / ("trace-" + uuid.uuid4().hex + ".jsonl")
    history: list[dict[str, Any]] = run_agent(
        client, messages, TOOL_SCHEMAS, tools.dispatch, str(trace), max_turns
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
        )
        history = run_router_session(
            client,
            args.registry,
            task,
            args.output,
            skill=args.skill,
            max_turns=args.max_turns,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print("glimmer-router: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(history[-1], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
