"""Command line for multi-expert assessment (Qt-free, no Torch).

``python -m siliconshield.experts assess`` routes one scope through
the registry, consults every qualified expert over persistent
subprocess workers, and prints the assessment as JSON. With
``--expert`` it instead consults one named model on a design file,
requirement, and assumptions given on the command line. Optional
Muse arbitration and finding assembly stay in this CLI: the GUI
demo never owns router work.
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any

from siliconshield.experts import arbitration, findings
from siliconshield.experts.identity import assemble_state
from siliconshield.experts.service import InferenceService
from siliconshield.experts.worker import WorkerPool
from siliconshield.research_loop.procedures.muse_client import MuseClient
from siliconshield.web_demo.glimmer_client import LocalGlimmerClient


def build_parser() -> argparse.ArgumentParser:
    """Build the experts CLI parser (also used for --help checks)."""
    parser = argparse.ArgumentParser(
        prog="siliconshield.experts",
        description="Route and consult RTL security experts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    assess = sub.add_parser("assess", help="Assess one state text.")
    assess.add_argument("--registry", required=True, help="Registry path.")
    assess.add_argument(
        "--scope",
        default="",
        help="Comma-separated mechanism tags for routing.",
    )
    assess.add_argument(
        "--expert",
        default="",
        help="Consult exactly this registered expert id, skipping "
        "scope routing.",
    )
    assess.add_argument("--state-file", help="File with state text.")
    assess.add_argument(
        "--design",
        action="append",
        default=[],
        help="RTL design file to assess (repeat for several files). "
        "Builds the state text with --requirement and --assumptions.",
    )
    assess.add_argument(
        "--assumptions",
        default="",
        help="Plain-text assumptions, used with --design.",
    )
    assess.add_argument(
        "--arbitrate",
        action="store_true",
        help="Have Muse interpret routed expert evidence and RTL.",
    )
    assess.add_argument(
        "--brief",
        action="store_true",
        help="Print predictions, scores, timing, and any Muse conclusion.",
    )
    assess.add_argument(
        "--model",
        choices=(
            "glimmer-base",
            "muse-spark-1.3",
            "muse-spark-1.3-contributor",
        ),
        help="Arbitration model (default: muse-spark-1.3-contributor).",
    )
    assess.add_argument("--endpoint", help="Local Glimmer chat endpoint.")
    assess.add_argument("--ssh-host", help="SSH alias for local Glimmer.")
    assess.add_argument(
        "--live",
        action="store_true",
        help="Show routing, inference, and live Muse status on stderr.",
    )
    assess.add_argument(
        "--live-logs",
        action="store_true",
        help="Show one-line native Muse events without JSON payloads; implies --live.",
    )
    assess.add_argument("--requirement", default="", help="Requirement.")
    assess.add_argument("--issue", default="", help="Suspected issue.")
    assess.add_argument("--relevance", default="", help="Issue relevance.")
    assess.add_argument(
        "--rtl", action="append", default=[], help="RTL location (repeat)."
    )
    assess.add_argument(
        "--local-workers",
        action="store_true",
        help="Consult local decision-model workers. Local models are "
        "off by default and not supported; use at own risk. The "
        "Meta-only path uses Glimmer specialists.",
    )
    return parser


DEFAULT_ASSUMPTIONS = (
    "Two-state sampled-clock RTL; only the stated requirement is assessed."
)
"""Assumptions used when --design is given without --assumptions."""


def _load_state(args: argparse.Namespace) -> str:
    """Resolve the state text from a state file or design files."""
    designs = getattr(args, "design", [])
    assumptions = getattr(args, "assumptions", "").strip()
    if args.state_file and designs:
        raise ValueError("Pass --state-file, or --design, not both.")
    if designs:
        if not args.requirement.strip():
            raise ValueError("--design needs --requirement.")
        parts = []
        for path in designs:
            try:
                with open(path, encoding="utf-8") as handle:
                    parts.append(handle.read())
            except OSError as exc:
                raise ValueError("Cannot read design file: %s" % exc)
        assumptions = assumptions or DEFAULT_ASSUMPTIONS
        rtl = "\n".join(part.strip() for part in parts if part.strip())
        return assemble_state(args.requirement.strip(), assumptions, rtl)
    if not args.state_file:
        raise ValueError("Pass --state-file, or --design with --requirement.")
    if assumptions:
        raise ValueError("--assumptions is used with --design.")
    try:
        with open(args.state_file, encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise ValueError("Cannot read state file: %s" % exc)


def run_assess(
    args: argparse.Namespace,
    pool_factory: Any = WorkerPool,
) -> dict[str, Any]:
    """Run one assessment; return the JSON-serializable report."""
    state = _load_state(args)
    expert = getattr(args, "expert", "").strip()
    scope = [tag.strip() for tag in args.scope.split(",") if tag.strip()]
    if scope and expert:
        raise ValueError(
            "Pass --scope for routed assessment, or --expert to "
            "consult one model, not both."
        )
    if not scope and not expert:
        raise ValueError("Scope must name at least one mechanism tag.")
    # Usage errors must fail before the consult, never after it.
    if args.arbitrate and not args.requirement.strip():
        raise ValueError("--arbitrate needs --requirement.")
    if bool(args.issue.strip()) != bool(args.relevance.strip()):
        raise ValueError("--issue and --relevance are required together.")
    if (args.issue.strip() or args.relevance.strip()) and not args.arbitrate:
        raise ValueError("--issue and --relevance need --arbitrate.")
    agent_model = getattr(args, "model", None)
    endpoint = getattr(args, "endpoint", None)
    ssh_host = getattr(args, "ssh_host", None)
    live_logs = getattr(args, "live_logs", False)
    live = getattr(args, "live", False) or live_logs
    if not args.arbitrate and (agent_model or endpoint or ssh_host or live):
        raise ValueError(
            "--model, --endpoint, --ssh-host, --live, --live-logs need --arbitrate."
        )
    if live_logs and agent_model == "glimmer-base":
        raise ValueError(
            "--live-logs requires a Muse Spark CLI model; use --live for Glimmer."
        )
    if args.arbitrate and expert:
        raise ValueError(
            "Use --scope with --arbitrate; --expert is direct inference."
        )
    if args.arbitrate and agent_model == "glimmer-base" and not endpoint:
        raise ValueError("--model glimmer-base needs --endpoint.")
    if agent_model != "glimmer-base" and (endpoint or ssh_host):
        raise ValueError("--endpoint and --ssh-host are only for glimmer-base.")
    _live_status(live, "ROUTER: assessing scope " + ", ".join(scope))
    service = InferenceService(args.registry)
    report: dict[str, Any] = {
        "scope": scope,
        "broken_experts": service.broken_experts,
        "unprofiled_experts": service.unprofiled_experts,
    }
    if expert:
        report["expert"] = expert
    timings = {"load_seconds": 0.0, "inference_seconds": 0.0}
    with pool_factory() as pool:
        loaded: set[str] = set()

        def transport(expert_id: str, judged_state: str) -> Any:
            if expert_id not in loaded:
                _live_status(live, "DECISION MODEL: loading " + expert_id)
                started = time.monotonic()
                pool.load(service.model_entry(expert_id))
                timings["load_seconds"] += time.monotonic() - started
                loaded.add(expert_id)
            _live_status(
                live, "DECISION MODEL: assessing RTL with " + expert_id
            )
            started = time.monotonic()
            try:
                return pool.transport()(expert_id, judged_state)
            finally:
                timings["inference_seconds"] += time.monotonic() - started

        if expert:
            assessment = service.assess_expert(expert, state, transport)
        else:
            assessment = service.assess(scope, state, transport)
    for response in assessment["responses"]:
        _live_status(
            live,
            "DECISION MODEL: %s returned %s"
            % (response["expert_id"], response["label"]),
        )
    for error in assessment["errors"]:
        _live_status(
            live,
            "DECISION MODEL: %s failed at %s"
            % (error["expert_id"], error["stage"]),
        )
    report["assessment"] = assessment
    report["load_seconds"] = timings["load_seconds"]
    report["inference_seconds"] = timings["inference_seconds"]
    if args.arbitrate:
        report["arbitration_model"] = (
            agent_model or "muse-spark-1.3-contributor"
        )
        try:
            verdict = arbitration.arbitrate_assessment(
                dict(assessment),
                state,
                args.requirement,
                service.profiles_for(assessment["routing"]["experts"]),
                client_factory=lambda: _arbitration_client(args),
            )
        except (
            ValueError,
            RuntimeError,
            TimeoutError,
            TimeoutExpired,
            OSError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            report["arbitration_error"] = str(exc)
            return report
        _live_status(live, "ARBITRATION: validated " + verdict["conclusion"])
        report["arbitration"] = verdict
        if args.issue.strip() and args.relevance.strip():
            report["finding"] = findings.build_finding(
                args.issue,
                args.relevance,
                args.requirement,
                list(args.rtl),
                dict(assessment),
                verdict,
            )
    return report


def _arbitration_client(args: argparse.Namespace) -> Any:
    model = getattr(args, "model", None) or "muse-spark-1.3-contributor"
    audit = arbitration.default_muse_audit_dir()
    live_logs = getattr(args, "live_logs", False)
    live = getattr(args, "live", False) or live_logs
    # Observe only this invocation, never another agent's audit files.
    if live:
        audit.mkdir(parents=True, exist_ok=True)
        audit = Path(tempfile.mkdtemp(prefix="live-", dir=audit))
    if model == "glimmer-base":
        audit.mkdir(parents=True, exist_ok=True)
        client: Any = LocalGlimmerClient(
            endpoint=args.endpoint,
            model=model,
            ssh_host=args.ssh_host,
            max_tokens=1536,
            timeout=180,
            max_calls=1,
            audit_dir=Path(tempfile.mkdtemp(prefix="glimmer-", dir=audit)),
            decode_arguments=True,
        )
    else:
        executable = shutil.which("muse")
        if not executable:
            raise ValueError(
                "Muse CLI not found on PATH. Install Muse and sign in."
            )
        client = MuseClient(
            executable=Path(executable),
            audit_dir=audit,
            model=model,
            effort="low",
            timeout=180,
        )
    return (
        _LiveClient(
            client,
            model,
            audit if model != "glimmer-base" else None,
            live_logs=live_logs,
        )
        if live
        else client
    )


def _live_status(enabled: bool, message: str) -> None:
    if enabled:
        print(_recording_text(message), file=sys.stderr, flush=True)


def _muse_events(
    directory: Path,
    offsets: dict[Path, int],
    seen: set[str],
    *,
    live_logs: bool = False,
) -> None:
    """Tail native events and show compact status without prompt or output bodies."""
    labels = {
        "run_model_configured": "META SESSION: model configured",
        "run_started": "META SESSION: running",
        "run_output_delta": "META SESSION: receiving response",
    }
    for path in directory.glob("call-*/stdout.jsonl"):
        try:
            with path.open("rb") as handle:
                handle.seek(offsets.get(path, 0))
                while line := handle.readline():
                    if not line.endswith(b"\n"):
                        break  # Retry a partly written event at the next poll.
                    offsets[path] = handle.tell()
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    payload = (
                        event.get("payload")
                        if isinstance(event, dict)
                        else None
                    )
                    if not isinstance(payload, dict):
                        continue
                    kind = payload.get("kind")
                    if not isinstance(kind, str) or kind in seen:
                        continue
                    if live_logs:
                        message = _muse_log_line(payload)
                    else:
                        message = labels.get(kind, "")
                    if message:
                        seen.add(kind)
                        _live_status(True, message)
        except OSError:
            continue  # Display availability never decides inference success.


def _muse_log_line(payload: dict[str, Any]) -> str:
    """Render one native lifecycle event in a bounded, recording-safe line."""
    kind = payload.get("kind")
    if not isinstance(kind, str) or kind not in {
        "run_model_configured",
        "run_started",
        "run_output_delta",
        "run_usage",
        "run_terminal",
    }:
        return ""
    line = "[muse] " + kind
    for field in ("model_id", "completed_calls", "failed_calls", "terminal"):
        value = payload.get(field)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            value = " ".join(_recording_text(str(value)).split())[:64]
            line += " %s=%s" % (field, value)
    return line[:160]


class _LiveClient:
    """Display call progress while preserving private prompts and audit paths."""

    def __init__(
        self,
        client: Any,
        model: str,
        event_dir: Path | None = None,
        *,
        live_logs: bool = False,
    ) -> None:
        self.client, self.model, self.event_dir = client, model, event_dir
        self.live_logs = live_logs

    def complete(self, messages: list[dict[str, Any]], tools: Any = ()) -> Any:
        _live_status(True, "LIVE META CALL: " + self.model)
        _live_status(
            True,
            "Sending RTL, requirement, assumptions and specialist evidence.",
        )
        started = time.monotonic()
        stopped = threading.Event()
        offsets: dict[Path, int] = {}
        seen: set[str] = set()

        def observe() -> None:
            if self.event_dir is not None:
                _muse_events(
                    self.event_dir, offsets, seen, live_logs=self.live_logs
                )

        def watch() -> None:
            next_status = 5.0
            while not stopped.wait(0.2):
                observe()
                elapsed = time.monotonic() - started
                if elapsed >= next_status:
                    _live_status(True, "META WAIT: %.0fs elapsed" % elapsed)
                    next_status += 5.0

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            response = self.client.complete(messages, tools=tools)
        except (ValueError, RuntimeError, OSError, TimeoutExpired):
            _live_status(True, "META CALL FAILED")
            raise
        finally:
            stopped.set()
            watcher.join()
            observe()
        _live_status(
            True,
            "LIVE META RESPONSE: received in %.1fs; validating evidence."
            % (time.monotonic() - started),
        )
        # This is the returned structured function response, not a native
        # Muse shell/tool execution. Display only the expected public name.
        calls = (
            response.get("tool_calls", []) if isinstance(response, dict) else []
        )
        if isinstance(calls, list):
            for call in calls:
                function = (
                    call.get("function") if isinstance(call, dict) else None
                )
                if (
                    isinstance(function, dict)
                    and function.get("name") == "experts.arbitrate"
                ):
                    _live_status(
                        True,
                        "META RESPONSE TOOL: experts.arbitrate (structured output)",
                    )
                    break
        return response


def _recording_text(text: str) -> str:
    """Hide local absolute paths in brief output; JSON retains full details."""
    return re.sub(
        r"/(?:home|Users|data|tmp|private)/[^\s\"'<>]+", "[local path]", text
    )


def format_brief(report: dict[str, Any]) -> str:
    """Render one short human-readable assessment summary.

    Shows each consulted expert's label with all three decision
    scores, any consult errors, and the elapsed time.
    """
    lines = []
    assessment = report["assessment"]
    for response in assessment["responses"]:
        lines.append("%s: %s" % (response["expert_id"], response["label"]))
        for name in (
            "satisfies_requirement",
            "violates_requirement",
            "insufficient_context",
        ):
            lines.append("  %s: %.4f" % (name, response["scores"][name]))
    for error in assessment["errors"]:
        lines.append(
            "%s: %s (%s)"
            % (
                error["expert_id"],
                error["stage"],
                error.get("detail", ""),
            )
        )
    lines.append("load_seconds: %.2f" % (report["load_seconds"],))
    lines.append("inference_seconds: %.2f" % (report["inference_seconds"],))
    routing = assessment.get("routing", {})
    for field in ("unsupported", "partial"):
        if routing.get(field):
            lines.append("%s scopes: %s" % (field, ", ".join(routing[field])))
    if "arbitration_error" in report:
        lines.append("Muse arbitration failed: " + report["arbitration_error"])
    elif "arbitration" in report:
        verdict = report["arbitration"]
        lines.append(
            "%s: %s" % (report["arbitration_model"], verdict["conclusion"])
        )
        lines.append(verdict["explanation"])
        for question in verdict["open_questions"]:
            lines.append("Open question: " + question)
        lines.append("Model assessment; not independently verified.")
    return _recording_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; return the process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "assess":
        if not args.local_workers:
            print(
                "experts: local decision models are off by default "
                "and not supported; use at own risk. Pass "
                "--local-workers to consult them, or assess "
                "through the Glimmer specialists path.",
                file=sys.stderr,
            )
            return 2
        try:
            report = run_assess(args)
        except (
            ValueError,
            RuntimeError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            detail = _recording_text(str(exc)) if args.brief else str(exc)
            print("experts: %s" % detail, file=sys.stderr)
            return 2
        if args.brief:
            print(format_brief(report))
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
