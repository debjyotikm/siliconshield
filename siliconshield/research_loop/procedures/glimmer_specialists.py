"""Route assessments to Glimmer specialists assisted by skills.

This is the same router flow as the local-worker path
(``InferenceService.assess`` with scope routing, validated expert
envelopes, and optional Spark arbitration), with one change: the
transport calls Muse Glimmer instead of spawning a local worker.
Each registry expert selects its skill from its pinned checkpoint
directory, and the skill text is digest-checked on every request.

Glimmer returns one verdict label with self-reported confidence.
The transport maps that to a unit score distribution by assigning
the verdict its confidence and splitting the remainder equally;
these are self-reported confidences, never calibrated
probabilities. A verdict whose confidence cannot top the
distribution is refused instead of reshaped. The final
``--arbitrate`` verdict comes from Muse Spark through the existing
arbitration path. No custom decision model runs in this flow.
"""

import argparse
import hashlib
import http.client
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from siliconshield.experts import arbitration
from siliconshield.experts.cli import DEFAULT_ASSUMPTIONS, format_brief
from siliconshield.experts.identity import (
    assemble_state,
    digest,
    text_digest,
)
from siliconshield.experts.service import InferenceService
from siliconshield.research_loop.controller.client import ChatClient
from siliconshield.training.laya.question_contracts import (
    GLIMMER_CONTRACT,
    digest_for,
)

LABELS = (
    "satisfies_requirement",
    "violates_requirement",
    "insufficient_context",
)

VERDICT_PATTERN = re.compile(
    r"^VERDICT:\s*(satisfies_requirement|violates_requirement"
    r"|insufficient_context)\s*$",
    re.MULTILINE,
)
CONFIDENCE_PATTERN = re.compile(
    r"^CONFIDENCE:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE
)

CHARS_PER_TOKEN = 4
"""Conservative rendering estimate for the context budget check."""

CONTEXT_LEN = 8192
"""Glimmer serving context both sides of the call must respect."""


def parse_verdict(text: str) -> dict[str, Any]:
    """Parse one specialist verdict block from Glimmer text.

    Requires exactly one VERDICT line naming a known label and
    exactly one CONFIDENCE line with a number in [0, 1]. Raises
    ValueError otherwise; callers record the failure, never a
    verdict.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cannot parse an empty verdict")
    verdicts = VERDICT_PATTERN.findall(text)
    if len(verdicts) != 1:
        raise ValueError(
            "Verdict needs exactly one VERDICT line, found %d" % len(verdicts)
        )
    confidences = CONFIDENCE_PATTERN.findall(text)
    if len(confidences) != 1:
        raise ValueError(
            "Verdict needs exactly one CONFIDENCE line, found %d"
            % len(confidences)
        )
    try:
        confidence = float(confidences[0])
    except ValueError:
        raise ValueError(
            "Verdict confidence is not a number: %r" % confidences[0]
        )
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "Verdict confidence out of [0, 1]: %r" % confidences[0]
        )
    return {"label": verdicts[0], "confidence": confidence}


def scores_for(label: str, confidence: float) -> dict[str, float]:
    """Map one verdict to a unit score distribution.

    The verdict keeps its self-reported confidence and the
    remainder splits equally. A confidence below one third
    cannot top the distribution, so it is refused: reshaping
    it would invent a verdict the model did not support.
    """
    if label not in LABELS:
        raise ValueError("Unknown verdict label: %r" % (label,))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence out of [0, 1]: %r" % (confidence,))
    if confidence <= 1.0 / 3.0:
        raise ValueError(
            "Verdict confidence %r cannot top a three-way "
            "distribution" % (confidence,)
        )
    rest = (1.0 - confidence) / 2.0
    return {name: (confidence if name == label else rest) for name in LABELS}


class GlimmerTransport:
    """Consult registry experts through Glimmer and their skills."""

    def __init__(
        self,
        client: Any,
        model: str,
        endpoint: str,
        fingerprint: str,
        entries: dict[str, Any],
        max_tokens: int = 2048,
        device_type: str = "remote",
    ) -> None:
        """Bind one chat client to the registry entries served.

        ``fingerprint`` is the serving Glimmer weights digest; it
        must match the registry and profile binding exactly.
        ``device_type`` names where the serving happens; callers
        with an SSH hop pass an ``ssh:<host>`` label.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Glimmer model name must be text")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("Glimmer endpoint must be text")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("Glimmer fingerprint must be text")
        if not isinstance(device_type, str) or not device_type.strip():
            raise ValueError("Glimmer device type must be text")
        self._client = client
        self._model = model.strip()
        self._endpoint = endpoint.strip()
        self._fingerprint = fingerprint.strip()
        self._device_type = device_type.strip()
        self._entries = dict(entries)
        self._max_tokens = max_tokens
        self._question_sha256 = digest_for(GLIMMER_CONTRACT)
        self._runtime_sha256 = digest(
            {"endpoint": self._endpoint, "model": self._model}
        )

    def _skill(self, expert_id: str) -> tuple[str, str]:
        """Read and digest-check one expert's skill on every call."""
        entry = self._entries.get(expert_id)
        if entry is None:
            raise ValueError("Unknown expert %r" % (expert_id,))
        try:
            skill_bytes = (Path(entry["checkpoint"]) / "SKILL.md").read_bytes()
        except OSError as exc:
            raise ValueError("Cannot read skill for %r: %s" % (expert_id, exc))
        skill_digest = hashlib.sha256(skill_bytes).hexdigest()
        if skill_digest != entry["expected_files"]["SKILL.md"]:
            raise ValueError(
                "Skill for %r changed since registry load" % (expert_id,)
            )
        try:
            skill = skill_bytes.decode("utf-8")
        except ValueError as exc:
            raise ValueError(
                "Skill for %r is not UTF-8 text: %s" % (expert_id, exc)
            )
        return skill, skill_digest

    def __call__(self, expert_id: str, state: str) -> dict[str, Any]:
        """Consult one expert; return the raw worker-shaped payload."""
        if not isinstance(state, str) or not state.strip():
            raise ValueError("Cannot consult on empty state text")
        skill, skill_digest = self._skill(expert_id)
        estimate = (len(skill) + len(state)) // CHARS_PER_TOKEN
        budget = CONTEXT_LEN - self._max_tokens - 512
        if estimate > budget:
            raise ValueError(
                "Input estimate %d tokens exceeds the %d-token "
                "Glimmer budget; narrow the RTL region" % (estimate, budget)
            )
        message = self._client.complete(
            [
                {"role": "system", "content": skill},
                {"role": "user", "content": state},
            ]
        )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Glimmer returned no verdict text")
        verdict = parse_verdict(content)
        scores = scores_for(verdict["label"], verdict["confidence"])
        return {
            "label": verdict["label"],
            "probabilities": scores,
            "coverage": {
                "truncated": False,
                "state_tokens_dropped": 0,
                "truncated_questions": [],
            },
            "model_id": expert_id,
            "execution_identity": {
                "checkpoint_sha256": self._fingerprint,
                "export_sha256": digest(
                    {
                        "model": self._model,
                        "skill_sha256": skill_digest,
                    }
                ),
                "question_sha256": self._question_sha256,
                "runtime_sha256": self._runtime_sha256,
                "question_contract": GLIMMER_CONTRACT,
                "inference": {
                    "backend": "glimmer.chat",
                    "precision": "server-default",
                    "torch_version": "remote",
                    "device_type": self._device_type,
                    "batch_size": 1,
                    "max_len": CONTEXT_LEN,
                    "head_max_len": self._max_tokens,
                    "sort_by_length": False,
                    "option_order": [0, 1, 2],
                },
            },
            "state_sha256": text_digest(state),
        }


def build_parser() -> argparse.ArgumentParser:
    """Build the Glimmer specialists CLI parser."""
    parser = argparse.ArgumentParser(
        description="Route assessments to Glimmer specialists.",
    )
    parser.add_argument("--registry", required=True, help="Registry path.")
    parser.add_argument(
        "--scope", default="", help="Comma-separated tags for routing."
    )
    parser.add_argument(
        "--expert",
        default="",
        help="Consult exactly this expert id, skipping routing.",
    )
    parser.add_argument("--state-file", help="File with state text.")
    parser.add_argument(
        "--design",
        action="append",
        default=[],
        help="RTL design file (repeat for several files).",
    )
    parser.add_argument("--requirement", default="", help="Requirement.")
    parser.add_argument(
        "--assumptions", default="", help="Assumptions for --design."
    )
    parser.add_argument("--endpoint", required=True, help="Chat endpoint.")
    parser.add_argument("--ssh-host", default=None, help="SSH host.")
    parser.add_argument("--model", required=True, help="Served model.")
    parser.add_argument(
        "--glimmer-fingerprint",
        required=True,
        help="Serving weights digest; must match the registry.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Completion budget."
    )
    parser.add_argument("--output", required=True, help="Receipt dir.")
    parser.add_argument(
        "--arbitrate",
        action="store_true",
        help="Final Spark assessment over the verdicts.",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Print only labels, scores, and inference time.",
    )
    return parser


def _load_state(args: argparse.Namespace) -> str:
    """Resolve the judged state from a file or design inputs."""
    if args.state_file and args.design:
        raise ValueError("Pass --state-file, or --design, not both.")
    if args.design:
        if not args.requirement.strip():
            raise ValueError("--design needs --requirement.")
        parts = []
        for path in args.design:
            try:
                with open(path, encoding="utf-8") as handle:
                    parts.append(handle.read())
            except OSError as exc:
                raise ValueError("Cannot read design file: %s" % exc)
        assumptions = args.assumptions.strip() or DEFAULT_ASSUMPTIONS
        rtl = "\n".join(part.strip() for part in parts if part.strip())
        return assemble_state(args.requirement.strip(), assumptions, rtl)
    if not args.state_file:
        raise ValueError("Pass --state-file, or --design with --requirement.")
    if args.assumptions.strip():
        raise ValueError("--assumptions is used with --design.")
    try:
        with open(args.state_file, encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise ValueError("Cannot read state file: %s" % exc)


def run_specialists(
    args: argparse.Namespace,
    client_factory: Any = None,
    arbitration_client_factory: Any = None,
) -> dict[str, Any]:
    """Run one Glimmer-specialist assessment; return the report."""
    state = _load_state(args)
    expert = args.expert.strip()
    scope = [tag.strip() for tag in args.scope.split(",") if tag.strip()]
    if scope and expert:
        raise ValueError(
            "Pass --scope for routed assessment, or --expert to "
            "consult one model, not both."
        )
    if not scope and not expert:
        raise ValueError("Scope must name at least one mechanism tag.")
    if args.arbitrate and not args.requirement.strip():
        raise ValueError("--arbitrate needs --requirement.")
    if args.arbitrate and not scope:
        raise ValueError(
            "--arbitrate needs --scope: Spark weighs routing "
            "evidence, which direct selection does not produce."
        )
    service = InferenceService(args.registry)
    entries = {
        model_id: service.model_entry(model_id)
        for model_id in service.expert_ids
    }
    consulted = [expert] if expert else service.route(scope)["experts"]
    for model_id in consulted:
        entry = entries.get(model_id)
        if entry is None:
            # Unknown or unprofiled experts fall through to the
            # service, which reports them with a usable message.
            continue
        if entry.get("kind", "local-worker") != "glimmer":
            raise ValueError(
                "Refusing to route %r through Glimmer: its registry "
                "entry is not a glimmer specialist." % (model_id,)
            )
    if client_factory is None:
        client = ChatClient(
            args.endpoint,
            args.model,
            ssh_host=args.ssh_host,
            max_tokens=args.max_tokens,
        )
    else:
        client = client_factory()
    transport = GlimmerTransport(
        client,
        args.model,
        args.endpoint,
        args.glimmer_fingerprint,
        entries,
        max_tokens=args.max_tokens,
        device_type=(
            "ssh:%s" % args.ssh_host.strip()
            if args.ssh_host and args.ssh_host.strip()
            else "remote"
        ),
    )
    timings = {"inference_seconds": 0.0}
    started = time.monotonic()
    try:
        if expert:
            assessment = service.assess_expert(expert, state, transport)
        else:
            assessment = service.assess(scope, state, transport)
    finally:
        timings["inference_seconds"] = time.monotonic() - started
    report: dict[str, Any] = {
        "scope": scope,
        "expert": expert,
        "load_seconds": 0.0,
        "inference_seconds": timings["inference_seconds"],
        "broken_experts": service.broken_experts,
        "unprofiled_experts": service.unprofiled_experts,
        "assessment": assessment,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    skills = {
        model_id: entries[model_id]["expected_files"]["SKILL.md"]
        for model_id in consulted
    }

    def _write_receipt() -> None:
        receipt = {
            "scope": scope,
            "expert": expert,
            "state_sha256": text_digest(state),
            "glimmer_model": args.model,
            "glimmer_fingerprint": args.glimmer_fingerprint,
            "skill_sha256": skills,
            "responses": [
                {
                    "expert_id": item["expert_id"],
                    "label": item["label"],
                    "scores": item["scores"],
                }
                for item in assessment["responses"]
            ],
            "errors": assessment["errors"],
            "arbitration": report.get("arbitration"),
            "arbitration_model": report.get("arbitration_model"),
            "arbitration_error": report.get("arbitration_error"),
        }
        (output / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True)
        )

    # Preserve successful specialist evidence before arbitration: a
    # Spark outage must not void the verdicts already collected.
    _write_receipt()
    if args.arbitrate:
        report["arbitration_model"] = "muse-spark-1.3-contributor"
        try:
            report["arbitration"] = arbitration.arbitrate_assessment(
                dict(assessment),
                state,
                args.requirement,
                service.profiles_for(assessment["routing"]["experts"]),
                client_factory=arbitration_client_factory,
            )
        except Exception as exc:
            report["arbitration_error"] = "%s: %s" % (
                type(exc).__name__,
                exc,
            )
            raise
        finally:
            _write_receipt()
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; return the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        report = run_specialists(args)
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
        print("glimmer-specialists: %s" % exc, file=sys.stderr)
        return 2
    if args.brief:
        print(format_brief(report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
