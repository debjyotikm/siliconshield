"""Persistent campaign supervisor driven by Muse Spark sessions.

The supervisor owns the mechanical loop; agent sessions own every
judgment. One pass reconciles each unfinished round through the
campaign runner, records newly terminal rounds, runs at most one
agent session for work needing judgment (analysis, preparation,
recovery, planning), commits staged preparations only after
re-verifying them, and refreshes the status files.

The supervisor never prepares a round without a validated agent
proposal, never edits verification outcomes, and never relaxes a
registered threshold: dataset acceptance is re-audited from live
bytes, and selection or recipe rotations refuse while a round is
active. All progress lives in durable records, so a restart
recovers by rereading state; agent sessions are fresh each time
and continue across context limits through the persisted handoff.
"""

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

from siliconshield.research_loop.adapters.laya_training import decision_study
from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    read_json,
)
from siliconshield.research_loop.execution import laya_campaign, study_contract
from siliconshield.research_loop.procedures.bundles import read_bundle
from siliconshield.research_loop.procedures.campaign_learning import (
    queue_report,
)
from siliconshield.schemas import NEW_MODEL_PROPOSAL_SCHEMA
from siliconshield.training.laya import (
    dataset_versions,
    new_study,
)

HANDOFF_SCHEMA = "siliconshield-supervisor-handoff-v1"
ANALYSIS_SCHEMA = "siliconshield-round-analysis-v1"
PREP_SCHEMA = "siliconshield-round-prep-v1"

TERMINAL_PHASES = ("done", "failed")

MAX_AGENT_FAILURES = 3
MAX_PREP_ATTEMPTS = 2
MAX_PLAN_ATTEMPTS = 3

DEFAULT_TIMEOUTS = {
    "analyze": 1800,
    "prepare": 5400,
    "recover": 3600,
    "plan": 1800,
}
DEFAULT_STEPS = {
    "analyze": 60,
    "prepare": 200,
    "recover": 120,
    "plan": 60,
}
DEFAULT_EFFORTS = {
    "analyze": "medium",
    "prepare": "high",
    "recover": "high",
    "plan": "medium",
}


def campaign_dir(home: Path) -> Path:
    """Inner campaign directory holding rounds and registry."""
    return Path(home) / "campaign"


def supervisor_dir(home: Path) -> Path:
    """Durable supervisor state beside the campaign registry."""
    path = campaign_dir(home) / "supervisor"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_handoff() -> dict[str, Any]:
    """Fresh handoff: nothing observed, nothing proposed."""
    return {
        "schema": HANDOFF_SCHEMA,
        "updated": None,
        "mode": "observe",
        "active_round": None,
        "observed_terminal": [],
        "last_analysis": None,
        "pending_proposal": None,
        "proposal_round": None,
        "prep_attempts": 0,
        "plan_attempts": 0,
        "consecutive_agent_failures": 0,
        "parked_reason": None,
        "stopped_reason": None,
        "last_error": None,
    }


def load_handoff(home: Path) -> dict[str, Any]:
    """Read the persisted handoff, or a fresh one when absent."""
    path = supervisor_dir(home) / "handoff.json"
    if not path.is_file():
        return default_handoff()
    handoff = read_json(path)
    if not isinstance(handoff, dict):
        raise ValueError("Handoff is not an object")
    if handoff.get("schema") != HANDOFF_SCHEMA:
        raise ValueError("Unknown handoff schema")
    merged = default_handoff()
    merged.update(handoff)
    for key in (
        "prep_attempts",
        "plan_attempts",
        "consecutive_agent_failures",
    ):
        value = merged.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("Handoff needs " + key)
    if not isinstance(merged.get("observed_terminal"), list):
        raise ValueError("Handoff needs observed_terminal")
    if not isinstance(merged.get("mode"), str):
        raise ValueError("Handoff needs mode")
    pending = merged.get("pending_proposal")
    if pending is not None and not isinstance(pending, dict):
        raise ValueError("Handoff pending proposal is malformed")
    analysis = merged.get("last_analysis")
    if analysis is not None and not isinstance(analysis, dict):
        raise ValueError("Handoff analysis is malformed")
    for key in ("parked_reason", "stopped_reason"):
        value = merged.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError("Handoff needs " + key)
    return merged


def save_handoff(home: Path, handoff: dict[str, Any]) -> dict[str, Any]:
    """Persist the handoff atomically with a fresh timestamp."""
    handoff["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    handoff["schema"] = HANDOFF_SCHEMA
    atomic_json(supervisor_dir(home) / "handoff.json", handoff)
    return handoff


def record_session(home: Path, entry: dict[str, Any]) -> None:
    """Append one agent-session record to the durable journal."""
    journal = supervisor_dir(home) / "sessions.jsonl"
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def load_reference(home: Path) -> Optional[dict[str, Any]]:
    """Permanent-baseline results, or None when not registered."""
    path = campaign_dir(home) / "permanent_reference.json"
    if not path.is_file():
        return None
    reference = read_json(path)
    if not isinstance(reference, dict) or "results" not in reference:
        raise ValueError("Permanent reference has no results")
    return reference["results"]  # type: ignore[no-any-return]


def study_dir(home: Path) -> Path:
    """Study manifests beside the campaign registry."""
    path = campaign_dir(Path(home)) / "studies"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_study(home: Path, study_id: str) -> dict[str, Any]:
    """Read a study manifest and re-verify every frozen byte."""
    path = study_dir(home) / study_id / "study.json"
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError("Study manifest is not an object")
    decision_study.verify_study_manifest(manifest)
    return manifest


def _study_max_rounds(home: Path, study_id: str) -> int:
    """Round budget from the manifest without byte verification.

    Used only for cheap per-pass guards; every commit and every
    recommendation re-verifies the manifest bytes first.
    """
    path = study_dir(Path(home)) / study_id / "study.json"
    try:
        manifest = read_json(path)
    except OSError as exc:
        raise ValueError("Study manifest is missing") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Study manifest is not an object")
    try:
        rounds = int(manifest["budget"]["max_rounds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Study manifest needs a round budget") from exc
    return rounds


def study_exhausted(home: Path, study_id: str) -> bool:
    """True once prepared rounds reach the frozen round budget."""
    root = campaign_dir(Path(home))
    total = sum(
        (laya_campaign.read_round(root, rid).get("study") or {}).get("study_id")
        == study_id
        for rid in laya_campaign.list_rounds(root)
    )
    return bool(total >= _study_max_rounds(home, study_id))


def study_procedure(home: Path, study_id: str) -> dict[str, Any]:
    """Procedure block from the study manifest without byte checks.

    Only supplies default bundle/queue/model paths; every use
    revalidates the bundle bytes, and every commit re-verifies
    the whole manifest.
    """
    path = study_dir(Path(home)) / study_id / "study.json"
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError("Study manifest is not an object")
    procedure = manifest.get("procedure", {"enabled": False})
    if not isinstance(procedure, dict):
        raise ValueError("Study procedure block is not an object")
    return procedure


def recommend_new_model(
    home: Path, study_id: str, proposal: dict[str, Any]
) -> dict[str, Any]:
    """Propose a separate specialist after the budget is exhausted.

    Requires the frozen study manifest to verify, the round
    budget spent, every terminal round analyzed with a recorded
    diagnosis, and at least one qualified attempt: a finished
    job with complete delivered exposure. The two load-bearing
    claims are re-derived from run records — receipt phase for
    completion, tracked draws for exposure — and any overclaim
    refuses the recommendation. Execution-only failures can
    never justify a new model. The incumbent is recorded,
    never touched; the handoff parks until the owner approves
    the new study.
    """
    home = Path(home)
    manifest = load_study(home, study_id)
    campaign = campaign_dir(home)
    rounds = [
        rid
        for rid in laya_campaign.list_rounds(campaign)
        if (laya_campaign.read_round(campaign, rid).get("study") or {}).get(
            "sha256"
        )
        == manifest["manifest_sha256"]
    ]
    if len(rounds) < manifest["budget"]["max_rounds"]:
        raise ValueError("Study budget is not exhausted yet")
    diagnoses: dict[str, dict[str, Any]] = {}
    for round_id in rounds:
        phase = laya_campaign.round_phase(campaign, round_id)
        if phase not in TERMINAL_PHASES:
            raise ValueError("Study still has an active round: " + round_id)
        files = _round_files(home, round_id)
        if not files["analysis"].is_file():
            raise ValueError("Terminal round lacks analysis: " + round_id)
        analysis = read_json(files["analysis"])
        if not isinstance(analysis, dict):
            raise ValueError("Round analysis is not an object: " + round_id)
        diagnosis = analysis.get("diagnosis")
        if not isinstance(diagnosis, dict):
            raise ValueError("Round lacks a diagnosis: " + round_id)
        if diagnosis.get("category") not in decision_study.DIAGNOSES:
            raise ValueError("Round diagnosis names no known cause")
        evidence = diagnosis.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("Round diagnosis lacks evidence: " + round_id)
        claimed_job = evidence.get("job_complete") is True
        claimed_exposure = evidence.get("exposure_complete") is True
        derived = _derived_attempt(home, round_id)
        if claimed_job and not derived["job_complete"]:
            raise ValueError(
                "Round %s claims a finished job its records deny" % round_id
            )
        if claimed_exposure and not derived["exposure_complete"]:
            raise ValueError(
                "Round %s claims full exposure its draws deny" % round_id
            )
        decision = (
            laya_campaign.read_round(campaign, round_id).get("decision") or {}
        )
        if (
            decision.get("promote_research") is True
            or analysis.get("promoted") is True
        ):
            raise ValueError(
                "A successful improvement cannot justify an exhausted-model claim"
            )
        if diagnosis["category"] in (
            "poor_generalization",
            "retention_tradeoff",
        ):
            if (
                evidence.get("sdk_parity_ok") is not True
                or evidence.get("train_fit_ok") is not True
            ):
                raise ValueError(
                    "New-model advice requires verified SDK parity and TRAIN fit"
                )
        diagnoses[round_id] = {
            "category": diagnosis["category"],
            "job_complete": derived["job_complete"],
            "exposure_complete": derived["exposure_complete"],
        }
    qualified = [
        round_id
        for round_id, item in diagnoses.items()
        if item["job_complete"]
        and item["exposure_complete"]
        and item["category"] in ("poor_generalization", "retention_tradeoff")
    ]
    if not qualified:
        raise ValueError(
            "No qualified attempt yet: failed execution cannot justify a new model"
        )
    if not isinstance(proposal, dict):
        raise ValueError("New-model proposal must be an object")
    incumbent = laya_campaign.load_incumbent(campaign)
    if not isinstance(incumbent.get("export"), dict):
        raise ValueError("Incumbent export is missing")
    record = {
        "schema": NEW_MODEL_PROPOSAL_SCHEMA,
        "study_id": study_id,
        "domain": _require_text(proposal, "domain", "A proposal"),
        "dataset_plan": _require_text(proposal, "dataset_plan", "A proposal"),
        "eval_plan": _require_text(proposal, "eval_plan", "A proposal"),
        "preserved_incumbent": incumbent["export"],
        "diagnoses": diagnoses,
        "qualified_rounds": qualified,
    }
    atomic_json(study_dir(home) / study_id / "new-model-proposal.json", record)
    handoff = load_handoff(home)
    handoff["parked_reason"] = (
        "study %s exhausted its round budget; new-model proposal "
        "awaits owner approval" % study_id
    )
    handoff["mode"] = "parked"
    save_handoff(home, handoff)
    return record


def _derived_attempt(home: Path, round_id: str) -> dict[str, bool]:
    """Re-derive completion and exposure from run records.

    A done receipt with a written trainer report counts as a
    finished job; anything else does not. Exposure comes from
    the run's own tracked draws, never from the analysis. Both
    default to false when their records are missing.
    """
    campaign = campaign_dir(Path(home))
    try:
        phase = laya_campaign.read_round(campaign, round_id)["phase"]
    except (OSError, ValueError):
        return {"job_complete": False, "exposure_complete": False}
    report = _job_report_path(home, round_id)
    if phase != "done" or report is None:
        return {"job_complete": False, "exposure_complete": False}
    events = report.parent / "events.jsonl"
    try:
        exposure = decision_study.read_terminal_exposure(events)
    except (OSError, ValueError):
        return {"job_complete": True, "exposure_complete": False}
    return {
        "job_complete": True,
        "exposure_complete": decision_study.exposure_gate_passed(exposure),
    }


def _require_first_commit_qualification(home: Path, study_id: str) -> None:
    """Require measured receipts bound to this exact study and recipe."""
    base = study_dir(Path(home)) / study_id
    try:
        manifest = load_study(home, study_id)
        probe = read_json(base / "fitting-probe.json")
        resources = read_json(base / "resource-check.json")
        if not isinstance(probe, dict) or not isinstance(resources, dict):
            raise ValueError("Qualification receipts must be objects")
        study_contract.verify_receipts(manifest, probe, resources)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PrepCommitError(
            "Study fitting probe/resource qualification failed: %s" % exc
        ) from exc


def _require_text(record: dict[str, Any], key: str, what: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(what + " needs " + key)
    return value


def check_proposal_next_action(next_action: Any) -> dict[str, Any]:
    """Validate an agent's next-action proposal."""
    if not isinstance(next_action, dict):
        raise ValueError("Next action must be an object")
    kind = next_action.get("type")
    if kind == "prepare_round":
        _require_text(next_action, "round_id", "A prepare action")
        _require_text(next_action, "hypothesis", "A prepare action")
        _require_text(next_action, "dataset_plan", "A prepare action")
        selection = next_action.get("selection")
        if selection is not None:
            if not isinstance(selection, dict):
                raise ValueError("Proposed selection must be an object")
            laya_campaign.check_selection_policy(selection)
        recipe = next_action.get("recipe_plan")
        if recipe is not None and not isinstance(recipe, dict):
            raise ValueError("Proposed recipe plan must be an object")
        overrides = next_action.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("Proposed overrides must be an object")
        return {
            "type": "prepare_round",
            "round_id": next_action["round_id"],
            "hypothesis": next_action["hypothesis"],
            "dataset_plan": next_action["dataset_plan"],
            "selection": selection,
            "recipe_plan": recipe,
            "overrides": dict(overrides),
        }
    if kind == "investigate":
        return {
            "type": "investigate",
            "question": _require_text(
                next_action, "question", "An investigate action"
            ),
        }
    if kind == "stop":
        return {
            "type": "stop",
            "reason": _require_text(next_action, "reason", "A stop action"),
        }
    if kind == "recommend_new_model":
        return {
            "type": "recommend_new_model",
            "study_id": _require_text(
                next_action, "study_id", "A recommendation"
            ),
            "domain": _require_text(next_action, "domain", "A recommendation"),
            "dataset_plan": _require_text(
                next_action, "dataset_plan", "A recommendation"
            ),
            "eval_plan": _require_text(
                next_action, "eval_plan", "A recommendation"
            ),
        }
    raise ValueError("Unknown next-action type: " + repr(kind))


def check_analysis(record: Any, round_id: str) -> dict[str, Any]:
    """Validate an agent-written round analysis."""
    if not isinstance(record, dict):
        raise ValueError("Analysis must be an object")
    if record.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError("Analysis names another schema")
    if record.get("round_id") != round_id:
        raise ValueError("Analysis names another round")
    failed = record.get("what_failed")
    if not isinstance(failed, dict):
        raise ValueError("Analysis needs a what_failed object")
    for key in ("metrics", "artifacts"):
        if key not in failed:
            raise ValueError("Analysis what_failed needs " + key)
    for key in (
        "verdict_summary",
        "explanation",
        "proposed_change",
        "predicted_improvement",
        "possible_regressions",
        "falsifier",
    ):
        _require_text(record, key, "An analysis")
    diagnosis = record.get("diagnosis")
    checked_diagnosis = None
    if diagnosis is not None:
        if not isinstance(diagnosis, dict):
            raise ValueError("Analysis diagnosis must be an object")
        if diagnosis.get("category") not in decision_study.DIAGNOSES:
            raise ValueError("Analysis diagnosis names no known cause")
        evidence = diagnosis.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("Analysis diagnosis lacks evidence")
        for key in ("job_complete", "exposure_complete"):
            if not isinstance(evidence.get(key), bool):
                raise ValueError("Analysis diagnosis evidence needs " + key)
        reasons = diagnosis.get("reasons")
        if (
            not isinstance(reasons, list)
            or not reasons
            or any(not isinstance(item, str) for item in reasons)
        ):
            raise ValueError("Analysis diagnosis needs reasons")
        checked_diagnosis = {
            "category": diagnosis["category"],
            "evidence": dict(evidence),
            "reasons": list(reasons),
            "next": _require_text(diagnosis, "next", "A diagnosis"),
        }
    return {
        "schema": ANALYSIS_SCHEMA,
        "round_id": round_id,
        "diagnosis": checked_diagnosis,
        "verdict_summary": record["verdict_summary"],
        "what_failed": {
            "metrics": failed["metrics"],
            "artifacts": failed["artifacts"],
        },
        "explanation": record["explanation"],
        "proposed_change": record["proposed_change"],
        "predicted_improvement": record["predicted_improvement"],
        "possible_regressions": record["possible_regressions"],
        "falsifier": record["falsifier"],
        "next_action": check_proposal_next_action(record.get("next_action")),
    }


def check_prep_receipt(record: Any, round_id: str) -> dict[str, Any]:
    """Validate an agent-written preparation receipt."""
    if not isinstance(record, dict):
        raise ValueError("Prep receipt must be an object")
    if record.get("schema") != PREP_SCHEMA:
        raise ValueError("Prep receipt names another schema")
    if record.get("round_id") != round_id:
        raise ValueError("Prep receipt names another round")
    _require_text(record, "dataset_version", "A prep receipt")
    _require_text(record, "manifest_path", "A prep receipt")
    _require_text(record, "study_path", "A prep receipt")
    _require_text(record, "hypothesis", "A prep receipt")
    _require_text(record, "qualification", "A prep receipt")
    recipe = record.get("recipe_path")
    if recipe is not None and not isinstance(recipe, str):
        raise ValueError("Prep receipt recipe_path must be a path")
    selection = record.get("selection")
    if selection is not None:
        if not isinstance(selection, dict):
            raise ValueError("Prep receipt selection must be an object")
        laya_campaign.check_selection_policy(selection)
    return {
        "schema": PREP_SCHEMA,
        "round_id": round_id,
        "dataset_version": record["dataset_version"],
        "manifest_path": record["manifest_path"],
        "recipe_path": recipe,
        "study_path": record["study_path"],
        "hypothesis": record["hypothesis"],
        "selection": selection,
        "qualification": record["qualification"],
        "notes": record.get("notes", ""),
    }


def _token_sum(models: list[Any], key: str) -> float:
    """Sum one token count, skipping malformed model entries."""
    total = 0.0
    for model in models:
        if not isinstance(model, dict):
            continue
        value = model.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        total += value
    return total


def parse_exec_jsonl(text: str) -> dict[str, Any]:
    """Extract the session id, terminal state, and final text.

    Raises when the transcript carries no terminal record, so a
    truncated or killed run never reads as a completed analysis.
    Malformed records are skipped the same way as unparsable
    lines: only a well-formed terminal record counts.
    """
    session: Optional[str] = None
    terminal: Optional[str] = None
    final_text = ""
    usage: dict[str, Any] = {}
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        stream = record.get("stream") or {}
        if not isinstance(stream, dict):
            continue
        if stream.get("kind") == "session" and session is None:
            candidate = stream.get("id")
            if isinstance(candidate, str):
                session = candidate
        payload = record.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        kind = record.get("payload_type", "")
        if not isinstance(kind, str):
            continue
        if kind.startswith("run.terminal."):
            terminal_value = payload.get("terminal")
            if not isinstance(terminal_value, str):
                continue
            terminal = terminal_value
            text_value = payload.get("text") or ""
            final_text = text_value if isinstance(text_value, str) else ""
        elif kind == "run.usage":
            models = payload.get("models", [])
            if not isinstance(models, list):
                continue
            usage = {
                "calls": payload.get("calls"),
                "input_tokens": _token_sum(models, "input_tokens"),
                "output_tokens": _token_sum(models, "output_tokens"),
            }
    if terminal is None:
        raise ValueError("Agent transcript has no terminal record")
    return {
        "session": session,
        "terminal": terminal,
        "text": final_text,
        "usage": usage,
    }


def agent_argv(config: dict[str, Any], kind: str, prompt: str) -> list[str]:
    """Build the headless agent command for one session.

    The agent runs sandboxed by default. ``--disable-sandbox`` is added
    only when the operator-owned config explicitly sets
    ``allow_unsandboxed_agent``: unattended sessions must not silently
    gain full workspace access.
    """
    if not isinstance(config, dict):
        raise ValueError("Agent config must be an object")
    if kind not in DEFAULT_STEPS:
        raise ValueError("Unknown agent session kind: " + str(kind))
    for key in ("muse_bin", "model", "repo"):
        if not config.get(key) or not isinstance(config[key], str):
            raise ValueError("Agent config needs " + key)
    steps = config.get("max_model_steps", {})
    efforts = config.get("reasoning_effort", {})
    if not isinstance(steps, dict) or not isinstance(efforts, dict):
        raise ValueError("Agent step/effort maps must be objects")
    extra = config.get("extra_args", [])
    if not isinstance(extra, list) or any(
        not isinstance(arg, str) for arg in extra
    ):
        raise ValueError("Agent extra_args must be a string list")
    if not isinstance(prompt, str):
        raise ValueError("Agent prompt must be a string")
    argv = [
        config["muse_bin"],
        "exec",
        "--model",
        config["model"],
    ]
    if config.get("allow_unsandboxed_agent"):
        argv.append("--disable-sandbox")
    argv.extend(
        [
            "--json",
            "--max-model-steps",
            str(steps.get(kind, DEFAULT_STEPS[kind])),
            "--reasoning-effort",
            str(efforts.get(kind, DEFAULT_EFFORTS[kind])),
            "--disable-web-tools",
            "--workspace",
            config["repo"],
        ]
    )
    argv.extend(config.get("extra_args", []))
    argv.append(prompt)
    return argv


def run_muse_session(
    config: dict[str, Any],
    kind: str,
    prompt: str,
    workdir: Path,
) -> dict[str, Any]:
    """Run one headless agent session to completion or timeout.

    Returns the parsed transcript with the process exit code. A
    timeout kills the session and reports it as failed, never as
    an analysis.
    """
    from siliconshield.research_loop.execution.sandbox import (
        limited_preexec,
        restricted_env,
    )

    if not isinstance(config, dict):
        raise ValueError("Agent config must be an object")
    if kind not in DEFAULT_TIMEOUTS:
        raise ValueError("Unknown agent session kind: " + str(kind))
    timeouts = config.get("session_timeout_s", {})
    if not isinstance(timeouts, dict):
        raise ValueError("Agent timeout map must be an object")
    raw = timeouts.get(kind, DEFAULT_TIMEOUTS[kind])
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("Agent session timeout must be a number")
    timeout = int(raw)
    if timeout <= 0:
        raise ValueError("Agent session timeout must be positive")
    try:
        run = subprocess.run(
            agent_argv(config, kind, prompt),
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=restricted_env(config.get("agent_env_extra")),
            preexec_fn=limited_preexec(timeout),
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        return {
            "session": None,
            "exit_code": None,
            "terminal": "timeout",
            "text": partial[-2000:],
            "usage": {},
            "timed_out": True,
        }
    try:
        parsed = parse_exec_jsonl(run.stdout)
    except ValueError:
        parsed = {
            "session": None,
            "terminal": "unparsed",
            "text": run.stdout[-2000:] + run.stderr[-2000:],
            "usage": {},
        }
    parsed["exit_code"] = run.returncode
    parsed["timed_out"] = False
    return parsed


_PROHIBITIONS = """You must not rewrite verification outcomes, move
evaluation examples (or their near-duplicates) into training, or
relax any acceptance threshold to manufacture success. Strengthening
a threshold with a recorded reason is allowed. Labels come from
compilation and the fixed solver verification procedures only."""

_OUTPUT_CONTRACT = """Write your result as JSON to exactly this path:
{path}
Your final reply must be exactly one line: DONE:{path}"""


def build_prompt(kind: str, context: dict[str, Any]) -> str:
    """Render one agent-session prompt from durable context."""
    home = context["home"]
    repo = context["repo"]
    header = (
        "You own one step of the SiliconShield 24x7 training campaign.\n"
        "Repository: %s\nCampaign: %s\n" % (repo, home)
    )
    if kind == "analyze":
        body = (
            "Round %(round_id)s reached %(phase)s. Diagnose it.\n"
            "Round receipt: %(receipt)s\n"
            "Incumbent: %(incumbent)s\n"
            "Trainer report: %(report)s\n"
            "Leaderboard: %(leaderboard)s\n"
            "Record: what failed with exact metrics and artifacts; your "
            "leading explanation; the proposed change; the predicted "
            "improvement and possible regressions; what result would "
            "contradict the explanation. Classify the failure with one "
            "machine-checked diagnosis category: execution_defect, "
            "poor_labels, insufficient_exposure, failure_to_fit, "
            "poor_generalization, or retention_tradeoff. Derive "
            "job_complete and exposure_complete from the trainer "
            "report and its tracked draws "
            "(decision_study.assess_round), never from guesses; run "
            "a fitting probe for TRAIN fit. Then propose the next "
            "action: prepare_round (with hypothesis, dataset_plan, "
            "and optional selection/recipe_plan/overrides), "
            "investigate (with a question), recommend_new_model "
            "(with study_id, domain, dataset_plan, eval_plan), or "
            "stop (with a reason).\n" % context
        )
        schema = (
            '{"schema": "%s", "round_id": "%s", '
            '"verdict_summary": str, "what_failed": '
            '{"metrics": object, "artifacts": [str]}, "explanation": str, '
            '"proposed_change": str, "predicted_improvement": str, '
            '"possible_regressions": str, "falsifier": str, '
            '"diagnosis": {"category": str, "evidence": '
            '{"job_complete": bool, "exposure_complete": bool}, '
            '"reasons": [str], "next": str}, '
            '"next_action": {"type": "prepare_round", "round_id": str, '
            '"hypothesis": str, "dataset_plan": str, '
            '"selection": object|null, "recipe_plan": object|null, '
            '"overrides": object} | {"type": "investigate", '
            '"question": str} | {"type": "recommend_new_model", '
            '"study_id": str, "domain": str, "dataset_plan": str, '
            '"eval_plan": str} | {"type": "stop", "reason": str}}'
            % (ANALYSIS_SCHEMA, context["round_id"])
        )
    elif kind == "prepare":
        body = (
            "Execute the accepted proposal for round %(round_id)s.\n"
            "Proposal: %(proposal)s\n"
            "Generate the planned TRAIN-only data with solver-verified "
            "labels, keep development examples and near-duplicates out "
            "of training, qualify and accept the dataset version, stage "
            "the study manifest and any new recipe file, and record a "
            "prep receipt. Do not launch training; the supervisor "
            "commits and launches. Do not train on calibration or "
            "final-test partitions.\n" % context
        )
        schema = (
            '{"schema": "%s", "round_id": "%s", '
            '"dataset_version": str, "manifest_path": str, '
            '"recipe_path": str|null, "study_path": str, '
            '"hypothesis": str, "selection": object|null, '
            '"qualification": str, "notes": str}. The selection, when '
            "present, must equal the proposal's selection exactly."
            % (PREP_SCHEMA, context["round_id"])
        )
    elif kind == "recover":
        body = (
            "An execution failure needs recovery, not science.\n"
            "Failure: %(failure)s\n"
            "Round receipt: %(receipt)s\n"
            "Inspect the evidence and perform the appropriate recovery: "
            "reattach, reconnect, retry finalization, or resume a valid "
            "checkpoint. If the failure is an implementation bug, "
            "reproduce it, make a focused correction, run the relevant "
            "regression tests, freeze a new source snapshot for "
            "subsequent jobs (preserving existing run identities), "
            "and restart the supervisor service to load it. "
            "Describe what you did in your final reply; the supervisor "
            "re-runs reconciliation itself.\n" % context
        )
        schema = ""
    elif kind == "plan":
        body = (
            "No round is active and no proposal is pending. Review the "
            "campaign state and either propose the next round, "
            "recommend a separate new model once the study budget "
            "is spent, or stop with a reason.\n"
            "Handoff: %(handoff)s\n"
            "Incumbent: %(incumbent)s\n"
            "Leaderboard: %(leaderboard)s\n" % context
        )
        schema = (
            '{"type": "prepare_round", "round_id": str, '
            '"hypothesis": str, "dataset_plan": str, '
            '"selection": object|null, "recipe_plan": object|null, '
            '"overrides": object} | {"type": "recommend_new_model", '
            '"study_id": str, "domain": str, "dataset_plan": str, '
            '"eval_plan": str} | {"type": "stop", "reason": str}.'
        )
    else:
        raise ValueError("Unknown agent session kind: " + kind)
    parts = [header, body, _PROHIBITIONS]
    if schema:
        parts.append("Result schema:\n" + schema)
        parts.append(_OUTPUT_CONTRACT.format(path=context["output_path"]))
    else:
        parts.append(
            "Your final reply must summarize the recovery actions taken."
        )
    return "\n\n".join(parts)


def _round_files(home: Path, round_id: str) -> dict[str, Path]:
    base = campaign_dir(home) / "rounds" / round_id
    return {
        "receipt": base / "round.json",
        "analysis": base / "analysis.json",
        "prep": base / "prep-receipt.json",
    }


def commit_prep(
    home: Path,
    handoff: dict[str, Any],
    prep: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Verify staged artifacts and prepare the proposed round.

    Re-audits dataset acceptance from live bytes, rotates the
    selection policy and recipe binding when the proposal changes
    them, and prepares the round. Refuses while any round is
    active. Raises PrepCommitError with the defect; the round, when
    created but misbound, is failed with its reason preserved.
    """
    home = Path(home)
    campaign = campaign_dir(home)
    if not isinstance(config, dict):
        raise PrepCommitError("Supervisor config must be an object")
    for key in ("audit_extra_splits", "retired_paths"):
        value = config.get(key, [])
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise PrepCommitError("Supervisor config needs " + key)
    proposal = handoff.get("pending_proposal") or {}
    if proposal.get("round_id") != prep["round_id"]:
        raise PrepCommitError("Prep receipt targets another proposal")
    if proposal.get("hypothesis") != prep["hypothesis"]:
        raise PrepCommitError("Prep hypothesis differs from the proposal")
    if (proposal.get("selection") is None) != (prep["selection"] is None):
        raise PrepCommitError("Prep selection differs from the proposal")
    if proposal.get("selection") is not None and json_digest(
        proposal["selection"]
    ) != json_digest(prep["selection"]):
        raise PrepCommitError("Prep selection differs from the proposal")
    for round_id in laya_campaign.list_rounds(campaign):
        try:
            phase = laya_campaign.round_phase(campaign, round_id)
        except ValueError as exc:
            raise PrepCommitError(
                "Round receipt is unreadable: " + round_id
            ) from exc
        if phase not in TERMINAL_PHASES:
            raise PrepCommitError("A round is still active: " + round_id)
    study_manifest = None
    if config.get("study_id"):
        try:
            manifest = load_study(home, config["study_id"])
        except (OSError, ValueError) as exc:
            raise PrepCommitError(
                "Study manifest failed verification: %s" % exc
            ) from exc
        study_manifest = manifest
        spent = sum(
            (laya_campaign.read_round(campaign, rid).get("study") or {}).get(
                "study_id"
            )
            == config["study_id"]
            for rid in laya_campaign.list_rounds(campaign)
        )
        if spent >= manifest["budget"]["max_rounds"]:
            raise PrepCommitError(
                "Study round budget exhausted: %d of %d spent"
                % (spent, manifest["budget"]["max_rounds"])
            )
        live_policy = laya_campaign.load_campaign(campaign).get("selection")
        if not isinstance(live_policy, dict):
            raise PrepCommitError("Campaign has no registered selection")
        if (
            decision_study.policy_digest(live_policy)
            != manifest["evaluation"]["selection_policy_sha256"]
        ):
            raise PrepCommitError(
                "Campaign selection drifted from the frozen study; "
                "amend the study before committing"
            )
        _require_first_commit_qualification(home, config["study_id"])
        if (
            prep.get("selection") is not None
            and decision_study.policy_digest(prep["selection"])
            != manifest["evaluation"]["selection_policy_sha256"]
        ):
            raise PrepCommitError(
                "Proposed selection differs from the frozen study"
            )
    if study_manifest is not None:
        operator_profile = laya_campaign.load_campaign(campaign).get(
            "operator_profile"
        )
        if not isinstance(operator_profile, str):
            raise PrepCommitError("Campaign has no operator profile")
        profile_path = Path(operator_profile)
        raw_profile = read_json(profile_path)
        if not isinstance(raw_profile, dict):
            raise PrepCommitError("Operator profile must be an object")
        proposed = dict(raw_profile)
        if prep.get("recipe_path"):
            proposed["recipe"] = prep["recipe_path"]
        proposed["decision_study_manifest"] = str(
            study_dir(home) / config["study_id"] / "study.json"
        )
        proposed["decision_study_sha256"] = study_manifest["manifest_sha256"]
        expected_dataset = study_manifest["dataset"]
        if (
            prep["dataset_version"] != expected_dataset["version"]
            or Path(prep["manifest_path"]).resolve()
            != Path(expected_dataset["root"])
            / "versions"
            / expected_dataset["version"]
            / "manifest.json"
        ):
            raise PrepCommitError(
                "Prepared dataset differs from the frozen study"
            )
        campaign_id = laya_campaign.load_campaign(campaign).get("campaign_id")
        if not isinstance(campaign_id, str):
            raise PrepCommitError("Campaign has no campaign id")
        experiment = campaign_id + "-" + prep["round_id"]
        try:
            study_contract.verify_job(
                proposed, experiment, proposal.get("overrides") or {}
            )
        except (OSError, ValueError, KeyError) as exc:
            raise PrepCommitError(
                "Study execution binding failed: %s" % exc
            ) from exc
        atomic_json(profile_path, proposed)
    store = Path(prep["manifest_path"]).parents[2]
    version = prep["dataset_version"]
    manifest = dataset_versions.verify_version(store, version)
    splits = {
        role: Path(entry["path"]) for role, entry in manifest["splits"].items()
    }
    for extra in config.get("audit_extra_splits", []):
        splits[Path(extra).stem] = Path(extra)
    retired = [Path(path) for path in config.get("retired_paths", [])]
    try:
        findings = dataset_versions.audit_partition_overlap(splits, retired)
    except (OSError, ValueError) as exc:
        raise PrepCommitError("Dataset re-audit failed: %s" % exc) from exc
    if (
        findings["cross_split_exact_rtl_overlap"]
        or findings["retired_rtl_overlap"]
    ):
        raise PrepCommitError(
            "Dataset re-audit found overlap: %r" % (findings,)
        )
    if not dataset_versions.is_accepted(store, version):
        raise PrepCommitError("Dataset version is not accepted: " + version)
    manifest_path = Path(prep["manifest_path"])
    laya_campaign.register_dataset(
        campaign,
        version,
        str(manifest_path),
        _sha_file(manifest_path),
    )
    registered = laya_campaign.load_campaign(campaign)
    operator_profile = registered.get("operator_profile")
    if not isinstance(operator_profile, str):
        raise PrepCommitError("Campaign has no operator profile")
    profile_path = Path(operator_profile)
    profile = read_json(profile_path)
    if not isinstance(profile, dict):
        raise PrepCommitError("Operator profile is not an object")
    incumbent = laya_campaign.load_incumbent(campaign)
    if not isinstance(incumbent.get("export"), dict):
        raise PrepCommitError("Incumbent export is missing")
    incumbent_export = incumbent["export"]
    campaign_id = registered.get("campaign_id")
    if not isinstance(campaign_id, str):
        raise PrepCommitError("Campaign has no campaign id")
    train_entry = manifest["splits"].get("train")
    if not isinstance(train_entry, dict):
        raise PrepCommitError("Dataset version needs a train split")
    if not isinstance(profile.get("runs"), str):
        raise PrepCommitError("Operator profile needs runs")
    try:
        new_study.verify_new_study(
            prep["study_path"],
            incumbent_export,
            str(Path(train_entry["path"]).parent),
            str(Path(profile["runs"]) / (campaign_id + "-" + prep["round_id"])),
            campaign_id + "-" + prep["round_id"],
        )
    except (OSError, ValueError) as exc:
        raise PrepCommitError("Study verification failed: %s" % exc) from exc
    if prep["selection"] is not None:
        laya_campaign.update_campaign_selection(
            campaign,
            prep["selection"],
            "proposal from %s analysis" % prep["round_id"],
        )
    if prep["recipe_path"] is not None:
        recipe_path = Path(prep["recipe_path"])
        laya_campaign.update_campaign_recipe(
            campaign,
            str(recipe_path),
            _sha_file(recipe_path),
            "proposal from %s analysis" % prep["round_id"],
        )
        profile = read_json(profile_path)
        if not isinstance(profile, dict):
            raise PrepCommitError("Operator profile is not an object")
        profile["recipe"] = str(recipe_path.resolve())
        atomic_json(profile_path, profile)
    receipt = laya_campaign.prepare_round(
        campaign,
        prep["round_id"],
        version,
        prep["hypothesis"],
        proposal.get("overrides") or None,
    )
    if study_manifest is not None:
        receipt["study"] = {
            "study_id": study_manifest["study_id"],
            "sha256": study_manifest["manifest_sha256"],
        }
        atomic_json(_round_files(home, prep["round_id"])["receipt"], receipt)
    round_recipe = (receipt.get("bindings") or {}).get("round_recipe") or {}
    if not round_recipe.get("path"):
        raise PrepCommitError("Prepared round lacks a round recipe")
    profile = read_json(profile_path)
    if not isinstance(profile, dict):
        raise PrepCommitError("Operator profile is not an object")
    profile["recipe"] = round_recipe["path"]
    atomic_json(profile_path, profile)
    try:
        laya_campaign.verify_round_bindings(campaign, prep["round_id"], profile)
    except (OSError, ValueError) as exc:
        laya_campaign.advance(
            campaign,
            prep["round_id"],
            "failed",
            reason="binding verification at commit: %s" % exc,
        )
        raise PrepCommitError(
            "Prepared round failed binding verification: %s" % exc
        ) from exc
    return receipt


class PrepCommitError(ValueError):
    """A staged preparation failed supervisor verification."""


def json_digest(payload: Any) -> str:
    """Canonical digest for comparing agent-staged objects."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _note_agent_failure(handoff: dict[str, Any], what: str) -> None:
    """Count a failed agent session; park after the budget."""
    handoff["consecutive_agent_failures"] += 1
    if handoff["consecutive_agent_failures"] >= MAX_AGENT_FAILURES:
        handoff["parked_reason"] = "%s failed %d times consecutively" % (
            what,
            handoff["consecutive_agent_failures"],
        )
        handoff["mode"] = "parked"


def _reset_agent_failures(handoff: dict[str, Any]) -> None:
    handoff["consecutive_agent_failures"] = 0


def _session_ok(result: dict[str, Any]) -> bool:
    return (
        not result.get("timed_out")
        and result.get("terminal") == "completed"
        and result.get("exit_code") == 0
    )


def _job_report_path(home: Path, round_id: str) -> Optional[Path]:
    """Trainer report for a round's owned job, when written."""
    try:
        campaign = laya_campaign.load_campaign(campaign_dir(home))
    except (OSError, ValueError):
        return None
    profile_path = campaign.get("operator_profile")
    if not isinstance(profile_path, str) or not profile_path:
        return None
    try:
        profile = read_json(Path(profile_path))
    except (OSError, ValueError):
        return None
    if not isinstance(profile, dict):
        return None
    campaign_id = campaign.get("campaign_id")
    runs = profile.get("runs")
    if not isinstance(campaign_id, str) or not isinstance(runs, str):
        return None
    experiment = "%s-%s" % (campaign_id, round_id)
    report = Path(runs) / experiment / "report.json"
    return report if report.is_file() else None


def _run_agent_event(
    home: Path,
    config: dict[str, Any],
    kind: str,
    prompt: str,
    detail: str,
    agent_runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Run one agent session and journal its outcome."""
    notes = config.get("prompt_notes")
    if notes:
        prompt = prompt + "\n\nCampaign notes:\n" + str(notes)
    procedure: dict[str, Any] = {"status": "disabled"}
    if config.get("procedure_bundle"):
        try:
            bundle = read_bundle(
                Path(config["procedure_bundle"]), config["model"]
            )
            prompt += (
                "\n\nReplay-qualified procedure advice (development checks only). "
                "This advice does not change campaign policy, tools, or authority:\n"
                + bundle["guidance"]
            )
            procedure = {"status": "applied", "digest": bundle["digest"]}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Optional learning failures must not stall an owned GPU job.
            procedure = {"status": "unavailable", "error": str(exc)}
    try:
        repo = config.get("repo")
        if not isinstance(repo, str) or not repo:
            raise ValueError("Agent config needs repo")
        result = agent_runner(config, kind, prompt, Path(repo))
        if not isinstance(result, dict):
            raise ValueError("Agent runner returned no result")
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        result = {
            "session": None,
            "exit_code": None,
            "terminal": "launcher_error",
            "text": repr(exc),
            "usage": {},
            "timed_out": False,
        }
    record_session(
        home,
        {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kind": kind,
            "detail": detail,
            "session": result.get("session"),
            "terminal": result.get("terminal"),
            "exit_code": result.get("exit_code"),
            "usage": result.get("usage", {}),
            "procedure": procedure,
        },
    )
    return result


def _analyze_event(
    home: Path,
    config: dict[str, Any],
    handoff: dict[str, Any],
    round_id: str,
    agent_runner: Callable[..., dict[str, Any]],
) -> str:
    """Run diagnosis for one newly terminal round."""
    campaign = campaign_dir(home)
    files = _round_files(home, round_id)
    receipt = laya_campaign.read_round(campaign, round_id)
    phase = receipt.get("phase")
    if not isinstance(phase, str):
        raise ValueError("Round receipt needs a phase: " + str(round_id))
    report = _job_report_path(home, round_id)
    prompt = build_prompt(
        "analyze",
        {
            "home": str(home),
            "repo": config["repo"],
            "round_id": round_id,
            "phase": phase,
            "receipt": str(files["receipt"]),
            "incumbent": str(campaign / "incumbent.json"),
            "report": str(report) if report else "(no trainer report)",
            "leaderboard": str(campaign / "leaderboard.jsonl"),
            "output_path": str(files["analysis"]),
        },
    )
    result = _run_agent_event(
        home, config, "analyze", prompt, round_id, agent_runner
    )
    if not _session_ok(result):
        _note_agent_failure(handoff, "analysis for " + round_id)
        handoff["last_error"] = "analyze session: %s" % result["terminal"]
        return "analyze_failed"
    try:
        analysis = check_analysis(read_json(files["analysis"]), round_id)
        if config.get("study_id"):
            if analysis.get("diagnosis") is None:
                raise ValueError(
                    "Study analysis requires an evidence-based diagnosis"
                )
            derived = _derived_attempt(home, round_id)
            evidence = analysis["diagnosis"]["evidence"]
            for key in ("job_complete", "exposure_complete"):
                if evidence[key] and not derived[key]:
                    raise ValueError("Diagnosis overclaims " + key)
    except (OSError, ValueError) as exc:
        _note_agent_failure(handoff, "analysis for " + round_id)
        handoff["last_error"] = "analysis invalid: %s" % exc
        return "analyze_invalid"
    _reset_agent_failures(handoff)
    handoff["last_error"] = None
    if config.get("procedure_queue") and report is not None:
        try:
            queued = queue_report(Path(config["procedure_queue"]), report)
            handoff["procedure_learning"] = {
                "status": "queued",
                "request": queued.name,
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            handoff["procedure_learning"] = {
                "status": "unavailable",
                "error": str(exc),
            }
    handoff["last_analysis"] = {
        "round_id": round_id,
        "session": result.get("session"),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": analysis["verdict_summary"],
        "next_action": analysis["next_action"]["type"],
    }
    handoff["observed_terminal"] = sorted(
        set(handoff["observed_terminal"]) | {round_id}
    )
    action = analysis["next_action"]
    if action["type"] == "prepare_round":
        handoff["pending_proposal"] = action
        handoff["proposal_round"] = action["round_id"]
        handoff["proposal_source"] = str(files["analysis"])
        handoff["prep_attempts"] = 0
        handoff["prep_defect"] = None
        handoff["mode"] = "prepare"
    elif action["type"] == "investigate":
        handoff["investigation"] = action["question"]
        handoff["mode"] = "observe"
    elif action["type"] == "recommend_new_model":
        try:
            recommend_new_model(home, action["study_id"], action)
            parked = load_handoff(home)
            handoff["parked_reason"] = parked["parked_reason"]
            handoff["mode"] = "parked"
        except (OSError, ValueError) as exc:
            handoff["last_error"] = "recommendation refused: %s" % exc
            handoff["mode"] = "observe"
            return "recommendation_invalid"
    else:
        handoff["stopped_reason"] = action["reason"]
        handoff["mode"] = "stopped"
    return "analyzed"


def _prepare_event(
    home: Path,
    config: dict[str, Any],
    handoff: dict[str, Any],
    agent_runner: Callable[..., dict[str, Any]],
) -> str:
    """Run data/code preparation for the pending proposal."""
    round_id = handoff["proposal_round"]
    files = _round_files(home, round_id)
    context: dict[str, Any] = {
        "home": str(home),
        "repo": config["repo"],
        "round_id": round_id,
        "proposal": handoff.get("proposal_source", "(handoff)"),
        "output_path": str(files["prep"]),
    }
    prompt = build_prompt("prepare", context)
    if handoff.get("prep_defect"):
        prompt += (
            "\n\nPrior staging was rejected: %s\nFix the defect."
            % (handoff["prep_defect"])
        )
    result = _run_agent_event(
        home, config, "prepare", prompt, round_id, agent_runner
    )
    if not _session_ok(result):
        _note_agent_failure(handoff, "prepare for " + round_id)
        handoff["last_error"] = "prepare session: %s" % result["terminal"]
        return "prepare_failed"
    if not files["prep"].is_file():
        _note_agent_failure(handoff, "prepare for " + round_id)
        handoff["last_error"] = "prepare session wrote no receipt"
        return "prepare_missing"
    try:
        check_prep_receipt(read_json(files["prep"]), round_id)
    except (OSError, ValueError) as exc:
        _note_agent_failure(handoff, "prepare for " + round_id)
        handoff["last_error"] = "prep receipt invalid: %s" % exc
        return "prepare_invalid"
    _reset_agent_failures(handoff)
    handoff["prep_defect"] = None
    return _commit_attempt(home, handoff, config)


def _commit_attempt(
    home: Path, handoff: dict[str, Any], config: dict[str, Any]
) -> str:
    """Verify staged prep and prepare the round when idle."""
    round_id = handoff["proposal_round"]
    files = _round_files(home, round_id)
    try:
        prep = check_prep_receipt(read_json(files["prep"]), round_id)
    except (OSError, ValueError):
        return "prepare_invalid"
    try:
        commit_prep(home, handoff, prep, config)
    except PrepCommitError as exc:
        defect = str(exc)
        if defect.startswith("A round is still active"):
            return "commit_deferred"
        handoff["prep_attempts"] += 1
        handoff["prep_defect"] = defect
        if handoff["prep_attempts"] >= MAX_PREP_ATTEMPTS:
            handoff["parked_reason"] = "prep for %s rejected %d times: %s" % (
                round_id,
                handoff["prep_attempts"],
                defect,
            )
            handoff["mode"] = "parked"
        return "commit_rejected"
    handoff["pending_proposal"] = None
    handoff["proposal_round"] = None
    handoff["proposal_source"] = None
    handoff["prep_attempts"] = 0
    handoff["prep_defect"] = None
    handoff["active_round"] = round_id
    handoff["mode"] = "observe"
    handoff["last_error"] = None
    return "committed"


def _recover_event(
    home: Path,
    config: dict[str, Any],
    handoff: dict[str, Any],
    errors: dict[str, str],
    agent_runner: Callable[..., dict[str, Any]],
) -> str:
    """Run one recovery session for reconciliation failures."""
    round_id = sorted(errors)[0]
    files = _round_files(home, round_id)
    prompt = build_prompt(
        "recover",
        {
            "home": str(home),
            "repo": config["repo"],
            "failure": errors[round_id],
            "receipt": str(files["receipt"]),
        },
    )
    result = _run_agent_event(
        home, config, "recover", prompt, round_id, agent_runner
    )
    if not _session_ok(result):
        _note_agent_failure(handoff, "recovery for " + round_id)
    handoff["last_error"] = errors[round_id]
    return "recovered" if _session_ok(result) else "recover_failed"


def _plan_event(
    home: Path,
    config: dict[str, Any],
    handoff: dict[str, Any],
    agent_runner: Callable[..., dict[str, Any]],
) -> str:
    """Run one planning session when nothing is queued."""
    campaign = campaign_dir(home)
    output = supervisor_dir(home) / "plan-proposal.json"
    if output.is_file():
        output.unlink()
    prompt = build_prompt(
        "plan",
        {
            "home": str(home),
            "repo": config["repo"],
            "handoff": str(supervisor_dir(home) / "handoff.json"),
            "incumbent": str(campaign / "incumbent.json"),
            "leaderboard": str(campaign / "leaderboard.jsonl"),
            "output_path": str(output),
        },
    )
    if handoff.get("investigation"):
        prompt += "\n\nOpen investigation: %s" % handoff["investigation"]
    result = _run_agent_event(
        home, config, "plan", prompt, "next", agent_runner
    )
    if not _session_ok(result):
        _note_agent_failure(handoff, "planning")
        return "plan_failed"
    try:
        action = check_proposal_next_action(read_json(output))
    except (OSError, ValueError) as exc:
        _note_agent_failure(handoff, "planning")
        handoff["last_error"] = "plan proposal invalid: %s" % exc
        return "plan_invalid"
    if action["type"] == "stop":
        handoff["stopped_reason"] = action["reason"]
        handoff["mode"] = "stopped"
        handoff["plan_attempts"] = 0
        return "planned_stop"
    if action["type"] == "investigate":
        handoff["investigation"] = action["question"]
        handoff["plan_attempts"] += 1
        if handoff["plan_attempts"] >= MAX_PLAN_ATTEMPTS:
            handoff["parked_reason"] = (
                "planning investigated %d times without a proposal"
                % handoff["plan_attempts"]
            )
            handoff["mode"] = "parked"
        return "planned_investigate"
    if action["type"] == "recommend_new_model":
        try:
            recommend_new_model(home, action["study_id"], action)
        except (OSError, ValueError) as exc:
            _note_agent_failure(handoff, "planning")
            handoff["last_error"] = "recommendation refused: %s" % exc
            return "plan_invalid"
        parked = load_handoff(home)
        handoff["parked_reason"] = parked["parked_reason"]
        handoff["mode"] = "parked"
        _reset_agent_failures(handoff)
        return "planned_recommendation"
    if config.get("study_id"):
        try:
            exhausted = study_exhausted(home, config["study_id"])
        except (OSError, ValueError) as exc:
            handoff["last_error"] = "study guard unreadable: %s" % exc
            return "plan_invalid"
        if exhausted:
            handoff["parked_reason"] = (
                "study %s exhausted its round budget; propose "
                "recommend_new_model or stop" % config["study_id"]
            )
            handoff["mode"] = "parked"
            return "study_exhausted"
    _reset_agent_failures(handoff)
    handoff["pending_proposal"] = action
    handoff["proposal_round"] = action["round_id"]
    handoff["proposal_source"] = str(output)
    handoff["prep_attempts"] = 0
    handoff["prep_defect"] = None
    handoff["plan_attempts"] = 0
    handoff["investigation"] = None
    handoff["mode"] = "prepare"
    return "planned"


def _summarize_counts(summary: dict[str, Any]) -> str:
    """One-line development counts for the status record."""
    if not isinstance(summary, dict):
        return "(no candidate)"
    parts = []
    for role in ("new_coverage", "retention"):
        view = summary.get(role) or {}
        if not isinstance(view, dict):
            continue
        if view.get("n") is None:
            continue
        parts.append(
            "%s %s/%s fs %s/%s"
            % (
                "transfer" if role == "new_coverage" else "retention",
                view.get("correct"),
                view.get("n"),
                view.get("false_safe"),
                view.get("violations"),
            )
        )
    return "; ".join(parts) if parts else "(no candidate)"


def _leaderboard_tail(campaign: Path) -> Optional[dict[str, Any]]:
    board = campaign / "leaderboard.jsonl"
    if not board.is_file():
        return None
    for line in reversed(board.read_text(encoding="utf-8").splitlines()):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            return entry
    return None


def _gpu_note(
    phases: dict[str, str], handoff: dict[str, Any], action: str
) -> str:
    """Explain GPU state, including why idle when applicable."""
    active = {
        rid: phase
        for rid, phase in phases.items()
        if phase not in TERMINAL_PHASES
    }
    if active:
        rid, phase = sorted(active.items())[0]
        if phase == "preflight":
            return "round %s waiting for free GPUs" % rid
        return "round %s in %s" % (rid, phase)
    if handoff.get("parked_reason"):
        return "idle: parked (%s)" % handoff["parked_reason"]
    if handoff.get("stopped_reason"):
        return "idle: stopped (%s)" % handoff["stopped_reason"]
    if handoff.get("pending_proposal"):
        if action in ("prepared", "committed", "commit_deferred"):
            return "idle: committing round %s" % handoff["proposal_round"]
        return "idle: preparing round %s on CPU" % handoff["proposal_round"]
    if action.startswith("analyze") or action == "analyzed":
        return "idle: analyzing the finished round"
    if action.startswith("plan"):
        return "idle: planning the next experiment"
    if action.startswith("recover"):
        return "idle: recovering an execution failure"
    return "idle: reconciling"


def build_status(
    home: Path,
    handoff: dict[str, Any],
    phases: dict[str, str],
    action: str,
) -> dict[str, Any]:
    """Assemble the concise machine-readable status record."""
    campaign = campaign_dir(home)
    try:
        incumbent = laya_campaign.load_incumbent(campaign)
    except (OSError, ValueError):
        incumbent = {}
    tail = _leaderboard_tail(campaign)
    candidate = None
    active = handoff.get("active_round")
    if active:
        try:
            record = laya_campaign.read_round(campaign, active)
            held = record.get("candidate") or {}
            candidate = held.get("export") if isinstance(held, dict) else None
        except (OSError, ValueError):
            candidate = None
    analysis = handoff.get("last_analysis") or {}
    proposal = handoff.get("pending_proposal") or {}
    if handoff.get("parked_reason"):
        next_action = "operator review: " + handoff["parked_reason"]
    elif handoff.get("stopped_reason"):
        next_action = "stopped: " + handoff["stopped_reason"]
    elif proposal:
        next_action = "prepare round %s: %s" % (
            handoff.get("proposal_round"),
            proposal.get("hypothesis", ""),
        )
    elif active:
        next_action = "drive round %s (%s)" % (
            active,
            phases.get(active, "?"),
        )
    elif analysis:
        next_action = "follow analysis of %s (%s)" % (
            analysis.get("round_id"),
            analysis.get("next_action"),
        )
    else:
        next_action = "plan the next experiment"
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": handoff.get("mode"),
        "action": action,
        "rounds": phases,
        "active_round": active,
        "incumbent": {
            "export": incumbent.get("export"),
            "round_id": incumbent.get("round_id"),
        },
        "candidate": candidate,
        "latest_results": _summarize_counts((tail or {}).get("summary", {})),
        "latest_round": (tail or {}).get("round_id"),
        "last_analysis": analysis or None,
        "next_action": next_action,
        "recovery": handoff.get("last_error"),
        "parked_reason": handoff.get("parked_reason"),
        "stopped_reason": handoff.get("stopped_reason"),
        "gpu_note": _gpu_note(phases, handoff, action),
        "agent_failures": handoff.get("consecutive_agent_failures", 0),
    }


def render_status_markdown(status: dict[str, Any]) -> str:
    """Render the concise human-readable status record."""
    analysis = status.get("last_analysis") or {}
    lines = [
        "# Campaign supervisor status",
        "",
        "Updated: %s | Mode: %s | Last action: %s"
        % (status.get("time"), status.get("mode"), status.get("action")),
        "",
        "## Rounds",
        "",
    ]
    rounds = status.get("rounds") or {}
    if rounds:
        for round_id in sorted(rounds):
            lines.append("- %s: %s" % (round_id, rounds[round_id]))
    else:
        lines.append("- (none yet)")
    incumbent = status.get("incumbent") or {}
    lines.extend(
        [
            "",
            "## Position",
            "",
            "- Incumbent: %s (round %s)"
            % (incumbent.get("export"), incumbent.get("round_id")),
            "- Candidate: %s" % status.get("candidate"),
            "- Latest results: %s" % status.get("latest_results"),
            "",
            "## Agent",
            "",
            "- Last analysis: %s" % (analysis.get("summary") or "(none)"),
            "- Next action: %s" % status.get("next_action"),
            "- Recovery: %s" % (status.get("recovery") or "none"),
            "- Agent failures: %s" % status.get("agent_failures"),
            "",
            "## GPUs",
            "",
            "- %s" % status.get("gpu_note"),
            "",
        ]
    )
    return "\n".join(lines)


def write_status(home: Path, status: dict[str, Any]) -> dict[str, Any]:
    """Persist the machine and human status records."""
    home = Path(home)
    (home / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (home / "STATUS.md").write_text(
        render_status_markdown(status), encoding="utf-8"
    )
    return status


def supervise_once(
    home: Path,
    config: dict[str, Any],
    *,
    operator_factory: Callable[[str], Any],
    evaluator: Callable[[Path, str, dict[str, Any]], dict[str, Any]],
    agent_runner: Optional[Callable[..., dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run one supervisor pass and return the status record.

    Reconciles every unfinished round first, records newly
    terminal rounds, runs at most one agent session for work
    needing judgment, and refreshes the status files. Never
    prepares a round without a validated agent proposal.
    """
    home = Path(home)
    campaign = campaign_dir(home)
    runner = agent_runner or run_muse_session
    if not isinstance(config, dict):
        raise ValueError("Supervisor config must be an object")
    if config.get("study_id") and not config.get("procedure_bundle"):
        try:
            frozen = study_procedure(home, config["study_id"])
            if frozen.get("enabled"):
                bundle = frozen.get("bundle")
                if not isinstance(bundle, dict) or not isinstance(
                    bundle.get("path"), str
                ):
                    raise ValueError("Study procedure bundle is malformed")
                if not isinstance(frozen.get("queue"), str):
                    raise ValueError("Study procedure queue is malformed")
                if not isinstance(frozen.get("model"), str):
                    raise ValueError("Study procedure model is malformed")
                config = dict(config)
                config.setdefault("procedure_bundle", bundle["path"])
                config.setdefault("procedure_queue", frozen["queue"])
                config.setdefault("model", frozen["model"])
        except (OSError, ValueError):
            frozen = {"enabled": False}
    handoff = load_handoff(home)
    try:
        reference = load_reference(home)
    except (OSError, ValueError) as exc:
        reference = None
        handoff["last_error"] = "permanent reference unreadable: %s" % exc
    phases: dict[str, str] = {}
    errors: dict[str, str] = {}
    for round_id in laya_campaign.list_rounds(campaign):
        try:
            receipt = laya_campaign.read_round(campaign, round_id)
        except (OSError, ValueError) as exc:
            errors[round_id] = "unreadable round receipt: %s" % exc
            continue
        if not isinstance(receipt.get("phase"), str):
            errors[round_id] = "round receipt needs a phase"
            continue
        if receipt["phase"] in TERMINAL_PHASES:
            phases[round_id] = receipt["phase"]
            continue
        try:
            settled = laya_campaign.run_round(
                campaign,
                round_id,
                operator_factory=operator_factory,
                evaluator=evaluator,
                reference_results=reference,
            )
            phases[round_id] = settled["phase"]
        except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            errors[round_id] = repr(exc)
            try:
                phases[round_id] = laya_campaign.round_phase(campaign, round_id)
            except (OSError, ValueError):
                phases[round_id] = "unknown"
    active = sorted(
        rid for rid, phase in phases.items() if phase not in TERMINAL_PHASES
    )
    handoff["active_round"] = active[0] if active else None
    for round_id, phase in sorted(phases.items()):
        if phase in TERMINAL_PHASES:
            try:
                laya_campaign.record_leaderboard_entry(campaign, round_id)
            except (OSError, ValueError) as exc:
                errors[round_id] = "leaderboard refused: %s" % exc
    unanalyzed = sorted(
        rid
        for rid, phase in phases.items()
        if phase in TERMINAL_PHASES and rid not in handoff["observed_terminal"]
    )
    halted = handoff.get("parked_reason") or handoff.get("stopped_reason")
    action = "reconciled"
    if errors:
        handoff["last_error"] = errors[sorted(errors)[0]]
        if not halted:
            action = _recover_event(home, config, handoff, errors, runner)
    elif unanalyzed and not halted:
        action = _analyze_event(home, config, handoff, unanalyzed[0], runner)
    elif handoff.get("pending_proposal") and not halted:
        target = handoff.get("proposal_round")
        if not isinstance(target, str) or not target:
            handoff["pending_proposal"] = None
            handoff["last_error"] = "pending proposal round is malformed"
            action = "proposal_invalid"
        else:
            files = _round_files(home, target)
            if files["prep"].is_file():
                action = _commit_attempt(home, handoff, config)
            else:
                action = _prepare_event(home, config, handoff, runner)
    elif not active and not handoff.get("pending_proposal") and not halted:
        action = _plan_event(home, config, handoff, runner)
    committed = handoff.get("active_round")
    if committed and committed not in phases:
        try:
            phases[committed] = laya_campaign.read_round(campaign, committed)[
                "phase"
            ]
        except (OSError, ValueError):
            phases[committed] = "unknown"
    save_handoff(home, handoff)
    return write_status(home, build_status(home, handoff, phases, action))
