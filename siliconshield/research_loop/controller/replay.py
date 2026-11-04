"""Run checkpoint-review tasks and grade the recorded decisions and evidence.

Only deliverable.json is byte-stable across replays: trajectory.jsonl
carries wall-clock timing, so repeated runs differ there by design.
"""

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from siliconshield.research_loop.adapters.glimmer_training import load_rows
from siliconshield.research_loop.controller.client import ChatClient, run_agent
from siliconshield.research_loop.episodes.contracts import (
    OPERATOR_TOOLS,
    TOOLS,
    JsonValue,
    atomic_json,
    compare_metrics,
    decision,
    digest,
    final_payload,
    finite_number,
)
from siliconshield.schemas import REPLAY_SCHEMA


class ResearchTools:
    def __init__(self, summary):
        self.summary = summary
        self.read = False
        self.compared = False
        self.recorded = None

    def dispatch(self, name, args):
        if args.get("run_id") != self.summary["run_id"]:
            raise ValueError("Unknown registered experiment")
        if name == "research.read_run":
            self.read = True
            return self.summary
        if name == "research.compare_metrics":
            if not self.read:
                raise ValueError("Read the experiment before comparing metrics")
            self.compared = True
            return compare_metrics(self.summary)
        if name == "research.record_decision":
            if not self.compared:
                raise ValueError(
                    "Calculate the comparison before recording a decision"
                )
            if (
                args.get("action")
                not in TOOLS[2]["function"]["parameters"]["properties"][
                    "action"
                ]["enum"]
            ):
                raise ValueError("Unknown checkpoint-review action")
            expected = compare_metrics(self.summary)
            for field, key in (
                ("agreement_delta_pp", "agreement_delta_pp"),
                ("nll_delta", "nll_delta"),
            ):
                value = args.get(field)
                if not finite_number(value):
                    raise ValueError(
                        "A decision requires a finite structured "
                        + field
                        + " from compare_metrics"
                    )
                if abs(value - expected[key]) > 1e-6:
                    raise ValueError(
                        "Structured "
                        + field
                        + " does not match the calculated comparison"
                    )
            if (
                not isinstance(args.get("reason"), str)
                or not args["reason"].strip()
            ):
                raise ValueError("A decision requires an explanation")
            self.recorded = dict(args)
            return {"recorded": True, "job_submitted": False}
        raise ValueError("Unknown tool")


class OperatorTools:
    """Replay observations only; these tools cannot launch a job."""

    def __init__(self, row):
        self.summary, self.resources = row["summary"], row["resources"]
        self.read = self.compared = False
        self.recorded = None

    def dispatch(self, name, args):
        if args.get("run_id") != self.summary["run_id"]:
            raise ValueError("Unknown registered experiment")
        if name == "operator.inspect_run":
            self.read = True
            return self.summary
        if name == "operator.inspect_resources":
            self.compared = True
            return self.resources
        if name == "operator.decide":
            if not self.read or not self.compared:
                raise ValueError("Inspect run and resources before deciding")
            allowed = OPERATOR_TOOLS[-1]["function"]["parameters"][
                "properties"
            ]["action"]["enum"]
            if (
                args.get("action") not in allowed
                or not isinstance(args.get("reason"), str)
                or not args["reason"].strip()
            ):
                raise ValueError(
                    "An allowlisted action and explanation are required"
                )
            self.recorded = dict(args)
            return {"recorded": True, "executed": False}
        raise ValueError("Unknown tool")


def safe_grade(row: Any, delivered: Any) -> dict[str, Any]:
    """Grade one replay, recording a malformed row instead of aborting.

    Grading isolation matches execution isolation: one malformed row
    must not void the batch or its report.
    """
    try:
        graded: dict[str, Any] = grade(row, delivered)
        return graded
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        return {
            "execution_complete": 0,
            "analysis_correct": 0,
            "verified": False,
            "grade_error": "%s: %s" % (type(exc).__name__, exc),
        }


def safe_episode_dir(output: Path, episode_id: Any) -> Path:
    """Join one episode directory, refusing path escapes.

    Episode ids come from reviewed episodes files, but a hostile or
    mistyped id must never write outside the output directory.
    """
    if not isinstance(output, (str, Path)):
        raise ValueError("Episode output must be a path")
    if (
        not isinstance(episode_id, str)
        or not episode_id
        or episode_id.startswith(".")
        or "/" in episode_id
        or "\\" in episode_id
        or Path(episode_id).name != episode_id
    ):
        raise ValueError("Episode id is not a safe name: %r" % (episode_id,))
    return Path(output) / episode_id


def execute(row, client, output, system=None, max_turns=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "trajectory.jsonl").exists() or (
        output / "deliverable.json"
    ).exists():
        raise ValueError("Use a fresh replay attempt directory")
    tools = None
    error = None
    final = ""
    system_sha = ""
    episode_id = row.get("id", "") if isinstance(row, dict) else ""
    try:
        messages = row["messages"][:2]
        if system is not None:
            messages = [{"role": "system", "content": system}, messages[1]]
        operator = any(
            t["function"]["name"] == "operator.decide" for t in row["tools"]
        )
        tools = (
            OperatorTools(row) if operator else ResearchTools(row["summary"])
        )
        history = run_agent(
            client,
            messages,
            row["tools"],
            tools.dispatch,
            output / "trajectory.jsonl",
            max_turns,
        )
        final = history[-1].get("content", "")
        if not isinstance(final, str) or not final.strip():
            raise ValueError("Replay produced no final text")
        system_sha = digest(messages[0])
    except Exception as exc:
        # Endpoint isolation: a model/transport failure is recorded in the
        # deliverable so one bad episode never aborts the replay batch.
        error, final = f"{type(exc).__name__}: {exc}", ""
    delivered = {
        "schema": REPLAY_SCHEMA,
        "episode_id": episode_id,
        "model": getattr(client, "model", None),
        "checkpoint": {
            "adapter_path": getattr(client, "adapter_path", None),
            "checkpoint_fingerprint": getattr(
                client, "checkpoint_fingerprint", None
            ),
        },
        "system_sha256": system_sha,
        "read_evidence": tools.read if tools else False,
        "compared_metrics": tools.compared if tools else False,
        "decision": tools.recorded if tools else None,
        "final": final,
        "error": error,
    }
    atomic_json(output / "deliverable.json", delivered)
    return delivered


_DELTA_AGREEMENT = re.compile(
    r"agreement change:\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE
)
_DELTA_NLL = re.compile(r"nll change:\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_CLAIM = re.compile(
    r"(sealed|final|held[ -]?out|independent)[^.]*"
    r"(test|validation)[^.]*"
    r"\b(proves?|passed|pass|accuracy|results?|shows?|confirms?|correct|gain|improv\w*|reached)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(does not|do not|did not|not established|no evidence|never|cannot|"
    r"not prove[ns]?|not show[ns]?|not confirm[ns]?)\b",
    re.IGNORECASE,
)
_EXEC_FAILURE_CLAIM = re.compile(
    r"\b(execution|job|worker|process|unit|training run|training job)\b[^.]{0,80}"
    r"\b(fail\w*|error|crash\w*|died|abort\w*)\b"
    r"|\bexecution failure\b|\bjob failure\b|\bworker error\b",
    re.IGNORECASE,
)
_EXEC_DENIAL = re.compile(
    r"\bno\b[^.]{0,40}\b(observed|occurred|present|detected|found|failure)\b",
    re.IGNORECASE,
)
_PCT_CLAIM = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")
_NLL_CLAIM = re.compile(
    r"\bnll\b(?![^\d.]{0,40}?\bstep\b)[^\d.]{0,40}?([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_STEP_CITE = re.compile(r"\bstep\s+(\d+)", re.IGNORECASE)
_LR_CITE = re.compile(
    r"\b(?:\blr\b|learning rate)\b(?![^\d.]{0,40}?\bstep\b)[^\d.]{0,40}?([+-]?\d+(?:\.\d+)?(?:[eE]-?\d+)?)",
    re.IGNORECASE,
)
_ACCEPT_PASS_CLAIM = re.compile(
    r"\b(all acceptance|acceptance checks? (passed|pass)|all checks passed|no failed|"
    r"meets? all|met all|fully accepted|acceptance criteria[^.]{0,40}?\bmet\b|"
    r"passed (the )?acceptance|acceptance[^.]{0,40}?\bsatisfied)\b",
    re.IGNORECASE,
)
_ACCEPT_FAIL_CLAIM = re.compile(
    r"\b(acceptance (failed|not met|rejected)|failed acceptance|did not pass|"
    r"(?:has|have|did)\s+not\s+passed|never\s+passed)\b",
    re.IGNORECASE,
)
_TRAIN_LOSS_PROOF = re.compile(
    r"\btrain(ing)?\s+loss\b[^.]{0,80}?\b(prov\w+|confirm\w+|demonstrat\w+|establish\w+|guarantee\w+)\b",
    re.IGNORECASE,
)
_QUALITY_OBJECT = re.compile(
    r"\b(generaliz\w+|quality|successful model|model is good|model has improved|proves? the run)\b",
    re.IGNORECASE,
)
_LR_COUNT_CLAIM = re.compile(
    r"\b(twice|thrice|once|one time|\d+\s*(?:times?|\bx\b))\b", re.IGNORECASE
)
_PLATEAU_REDUCTION_CLAIM = re.compile(
    r"\b(plateau[^.]{0,40}?\b(reduc\w+|cut|lower\w+|decreas\w+)"
    r"|\b(reduc\w+|cut|lower\w+|decreas\w+)[^.]{0,40}?\b(learning rate|\blr\b)"
    r"|\b(learning rate|\blr\b)[^.]{0,40}?\b(reduc\w+|cut|lower\w+|decreas\w+))\b",
    re.IGNORECASE,
)
_CHECKPOINT_STEP_CITE = re.compile(
    r"\b(selected|best)\b[^.]{0,40}?\bstep\s+(\d+)"
    r"|\bstep\s+(\d+)[^.]{0,40}?\b(selected|best)\b"
    r"|\bcheckpoint\s+(\d+)\b",
    re.IGNORECASE,
)
_DENIAL_WORDS = re.compile(
    r"\b(no|not|never|without|zero|none|hasn\'t|haven\'t|isn\'t|aren\'t|cannot|can\'t)\b",
    re.IGNORECASE,
)
_DENY_VERB = re.compile(
    r"\b(not|never|without|neither|nor|hardly|barely|cannot|can\'t|didn\'t|did not|"
    r"doesn\'t|does not|hasn\'t|has not|haven\'t|have not|isn\'t|is not|aren\'t|are not|"
    r"wasn\'t|was not|won\'t|will not|wouldn\'t|would not|fails?\s+to|failed\s+to|"
    r"lacks?|lacked|absence\s+of)\b",
    re.IGNORECASE,
)
# Plateau denial needs no/none/zero in addition to _DENY_VERB ("No plateau
# reduction occurred"), but must not treat hyphenated values ("Near-zero
# training loss") as denials. Whole-sentence _DENIAL_WORDS suppressed a real
# claim when a trailing denial followed it ("reduced twice; no further
# action"), so use proximity instead.
_PLATEAU_DENY = re.compile(
    r"(?<!-)\b(no|none|zero|not|never|without|neither|nor|hardly|barely|cannot|can\'t|didn\'t|did not|"
    r"doesn\'t|does not|hasn\'t|has not|haven\'t|have not|isn\'t|is not|aren\'t|are not|"
    r"wasn\'t|was not|won\'t|will not|wouldn\'t|would not|fails?\s+to|failed\s+to|"
    r"lacks?|lacked|absence\s+of)\b",
    re.IGNORECASE,
)


def _denied_before(text, match_start, window=30):
    """Whether a negation word directly precedes the match (not a distant sentence)."""
    window_text = text[max(0, match_start - window) : match_start]
    return _DENY_VERB.search(window_text) is not None


def _plateau_denied_before(text, match_start, window=30):
    """Plateau denial with no/none/zero, excluding hyphenated values like Near-zero."""
    window_text = text[max(0, match_start - window) : match_start]
    return _PLATEAU_DENY.search(window_text) is not None


def _unsupported_claims(text):
    """Prose asserting sealed/final/independent-test outcomes the tools never provided.

    Explanation quality only: reported separately, never part of deterministic
    evidence correctness. 'Not only did the sealed test pass' is an assertion,
    not a negation.
    """
    claims = []
    for sentence in re.split(r"(?<=[.!])\s+", text or ""):
        if _CLAIM.search(sentence) and not _NEGATION.search(sentence):
            claims.append(sentence.strip())
    return claims


def _failure_claims(text):
    """Sentences asserting an execution/job failure, excluding explicit denials.

    'No execution failure was observed' is a denial, not a claim.
    """
    claims = []
    for sentence in re.split(r"(?<=[.!?;])\s+", text or ""):
        # Scope negation to the failure predicate, never an unrelated claim
        # later in the sentence ("This execution failure does not prove ...").
        for match in _EXEC_FAILURE_CLAIM.finditer(sentence):
            prefix = sentence[max(0, match.start() - 40) : match.start()]
            phrase = match.group()
            denied = re.search(
                r"\b(?:no|without|neither|not(?:\s+(?:an?|the))?)\s*$",
                prefix,
                re.I,
            )
            denied = denied or re.search(
                r"\b(?:did not|does not|has not|have not|never|not|no)\s+(?:\w+\s+){0,2}"
                r"(?:fail\w*|crash\w*|error|abort\w*)\b",
                phrase,
                re.I,
            )
            denied = denied or re.match(
                r"\s+(?:was|is|has been)\s+not\s+(?:observed|detected|present)",
                sentence[match.end() :],
                re.I,
            )
            if not denied:
                claims.append(sentence.strip())
                break
    return claims


def _observed_metrics(summary):
    """Metric values and steps visible in the operator observation.

    Includes per-view (transfer/retention) values so citing a view's metric
    is checkable, and acceptance thresholds so citing the failed criterion
    (for example its 25% budget) is checkable.
    """
    values, steps = [], []
    history = summary.get("development_history")
    entries = history.get("entries", []) if isinstance(history, dict) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if type(entry.get("step")) is int:
            steps.append(entry["step"])
        metrics = entry.get("metrics", {})
        if isinstance(metrics, dict):
            values.extend(v for v in metrics.values() if finite_number(v))
        views = entry.get("views", {})
        if isinstance(views, dict):
            for view in views.values():
                if isinstance(view, dict):
                    values.extend(v for v in view.values() if finite_number(v))
    for key in (
        "accuracy",
        "nll",
        "false_safe_rate",
        "min_class_recall",
        "both_pair_members_correct",
        "gradient_norm",
    ):
        if finite_number(summary.get(key)):
            values.append(summary[key])
    acceptance = summary.get("acceptance")
    if isinstance(acceptance, dict):
        thresholds = acceptance.get("thresholds", {})
        if isinstance(thresholds, dict):
            values.extend(v for v in thresholds.values() if finite_number(v))
    if type(summary.get("step")) is int:
        steps.append(summary["step"])
    return values, steps


def _count_words_to_number(text):
    """English count words to integers for intervention-count claims."""
    counts = []
    for raw in _LR_COUNT_CLAIM.findall(text):
        word = raw.lower().strip()
        if word in ("twice",):
            counts.append(2)
        elif word in ("thrice",):
            counts.append(3)
        elif word in ("once", "one time"):
            counts.append(1)
        else:
            match = re.match(r"(\d+)", word)
            if match:
                counts.append(int(match.group(1)))
    return counts


def _claimed_pass_acceptance(text):
    claims = []
    for match in _ACCEPT_PASS_CLAIM.finditer(text or ""):
        if _denied_before(text, match.start()):
            continue
        claims.append(match.group(0).strip())
    return claims


def _claimed_fail_acceptance(text):
    claims = []
    for match in _ACCEPT_FAIL_CLAIM.finditer(text or ""):
        if _denied_before(text, match.start()):
            continue
        claims.append(match.group(0).strip())
    return claims


def _claims_train_loss_proves(text):
    """Sentences using a training loss to prove generalization or quality."""
    claims = []
    for sentence in re.split(r"(?<=[.!])\s+", text or ""):
        match = _TRAIN_LOSS_PROOF.search(sentence)
        if not match:
            continue
        if _denied_before(sentence, match.start()):
            continue
        if _QUALITY_OBJECT.search(sentence[match.end() : match.end() + 80]):
            claims.append(sentence.strip())
    return claims


def _claimed_plateau_reductions(text):
    """Sentences asserting plateau LR reductions happened and any cited count."""
    claims = []
    for sentence in re.split(r"(?<=[.!])\s+", text or ""):
        matches = list(_PLATEAU_REDUCTION_CLAIM.finditer(sentence))
        if not matches:
            continue
        # A denial only negates a claim it precedes ("No plateau reduction
        # occurred"), not a trailing clause ("reduced twice; no further action").
        if not any(
            not _plateau_denied_before(sentence, m.start()) for m in matches
        ):
            continue
        counts = _count_words_to_number(sentence)
        claims.append((sentence.strip(), counts[0] if counts else None))
    return claims


def _claimed_checkpoint_steps(text):
    """Step numbers cited as the selected or best checkpoint."""
    cited = []
    for match in _CHECKPOINT_STEP_CITE.finditer(text or ""):
        groups = [g for g in match.groups() if g is not None]
        digits = [g for g in groups if g.isdigit()]
        if digits:
            cited.append(int(digits[0]))
    return cited


def _operator_explanation_contradictions(
    summary: Mapping[str, JsonValue],
    decision: Mapping[str, JsonValue],
    texts: list[tuple[str, str]],
) -> list[str]:
    """Check the explanation against the recorded observations.

    Action correctness is decided elsewhere; these issues cover only whether
    the cited evidence matches what the tools showed: execution status,
    metric values, referenced steps, action parameters, acceptance status,
    train-loss inferences, plateau interventions, and checkpoint selection.
    """
    issues = []
    for source, text in texts:
        if _failure_claims(text):
            no_failure = summary.get("execution_failed") is False or (
                summary.get("execution_failed") is not True
                and summary.get("systemd") in ("running", "completed")
                and not summary.get("error")
            )
            if no_failure:
                issues.append(
                    "explanation_contradicts_execution_evidence_in_" + source
                )
    acceptance = summary.get("acceptance")
    if isinstance(acceptance, dict) and "accepted" in acceptance:
        accepted = bool(acceptance["accepted"])
        reasons = acceptance.get("reasons", [])
        for source, text in texts:
            if not accepted or reasons:
                if _claimed_pass_acceptance(text):
                    issues.append(
                        "explanation_claims_acceptance_despite_failed_criteria_in_"
                        + source
                    )
                    break
            if accepted and not reasons:
                if _claimed_fail_acceptance(text):
                    issues.append(
                        "explanation_claims_failure_despite_acceptance_in_"
                        + source
                    )
                    break
    for source, text in texts:
        if _claims_train_loss_proves(text):
            issues.append("explanation_train_loss_proves_quality_in_" + source)
            break
    actual_reductions = summary.get("lr_reductions")
    if type(actual_reductions) is not int:
        schedule = summary.get("schedule")
        if isinstance(schedule, dict) and isinstance(
            schedule.get("plateau_lr_reductions"), list
        ):
            actual_reductions = len(schedule["plateau_lr_reductions"])
    if type(actual_reductions) is int:
        for source, text in texts:
            for _, claimed in _claimed_plateau_reductions(text):
                if claimed is None or claimed != actual_reductions:
                    issues.append(
                        "explanation_plateau_reduction_differs_from_record_in_"
                        + source
                    )
                    break
    selected = summary.get("selected_step")
    if selected is None:
        selection = summary.get("selection")
        if isinstance(selection, dict):
            selected = selection.get("selected_step")
    if type(selected) is int:
        for source, text in texts:
            cited = _claimed_checkpoint_steps(text)
            # Flag only when none of the cited steps is the selected one, so
            # contrastive prose ("not 200 but 300") is not punished.
            if cited and all(step != selected for step in cited):
                issues.append(
                    "explanation_checkpoint_differs_from_selected_in_" + source
                )
                break
    values, steps = _observed_metrics(summary)
    if values or steps:
        for source, text in texts:
            for raw in _PCT_CLAIM.findall(text):
                cited = float(raw) / 100
                if not any(abs(cited - v) <= 0.001 for v in values):
                    issues.append(
                        "explanation_metric_not_observed_in_" + source
                    )
                    break
            for raw in _NLL_CLAIM.findall(text):
                if not any(abs(float(raw) - v) <= 0.011 for v in values):
                    issues.append("explanation_nll_not_observed_in_" + source)
                    break
            for raw in _STEP_CITE.findall(text):
                if int(raw) not in steps:
                    issues.append("explanation_step_not_observed_in_" + source)
                    break
    lr = (decision or {}).get("lr")
    if (
        (decision or {}).get("action") == "adjust_lr"
        and isinstance(lr, (int, float))
        and not isinstance(lr, bool)
        and finite_number(lr)
    ):
        for source, text in texts:
            for raw in _LR_CITE.findall(text):
                if abs(float(raw) - lr) > abs(lr) * 1e-3 + 1e-9:
                    issues.append(
                        "explanation_lr_differs_from_decision_in_" + source
                    )
                    break
    checkpoint = (decision or {}).get("checkpoint_dir")
    if (decision or {}).get("action") == "resume" and isinstance(
        checkpoint, str
    ):
        for source, text in texts:
            for token in re.findall(r"[\w./-]*runs/[\w./-]+", text):
                if token != checkpoint and checkpoint not in token:
                    issues.append(
                        "explanation_checkpoint_differs_from_decision_in_"
                        + source
                    )
                    break
    return issues


def evidence_issues(row, delivered):
    """Check evidence access, the run reference, and the reported metric deltas."""
    issues = []
    if not delivered.get("read_evidence"):
        issues.append("evidence_not_read")
    if not delivered.get("compared_metrics"):
        issues.append("metrics_not_compared")
    chosen = delivered.get("decision") or {}
    if chosen and chosen.get("run_id") != row["summary"]["run_id"]:
        issues.append("evidence_reference_wrong")
    expected = compare_metrics(row["summary"])
    for field in ("agreement_delta_pp", "nll_delta"):
        value = chosen.get(field)
        if value is None:
            issues.append(field + "_missing")
        elif not finite_number(value):
            issues.append(field + "_not_finite")
        elif abs(value - expected[field]) > 1e-6:
            issues.append(field + "_wrong")
    return issues


def grade(row, delivered):
    if not isinstance(delivered, dict) or (
        delivered.get("schema") != REPLAY_SCHEMA
    ):
        # Zero counters keep the batch report sums total: an
        # ungradable deliverable scores nothing instead of breaking
        # the report with a missing key.
        return {
            "execution_complete": 0,
            "analysis_correct": 0,
            "verified": False,
            "schema_issue": ("deliverable is not a %s record" % REPLAY_SCHEMA),
        }
    if any(t["function"]["name"] == "operator.decide" for t in row["tools"]):
        chosen = delivered.get("decision") or {}
        from siliconshield.research_loop.episodes.operator_assessment import (
            assessment_issues,
            parameter_issues,
        )

        acceptable = row["target"].get(
            "acceptable_actions", [row["target"]["action"]]
        )
        exact_action = chosen.get("action") == row["target"][
            "action"
        ] and not delivered.get("error")
        action = chosen.get("action") in acceptable and not delivered.get(
            "error"
        )
        parameter_errors = parameter_issues(row, chosen)
        fact_errors = (
            assessment_issues(row["summary"], chosen.get("assessment"))
            if row["target"].get("require_assessment")
            else []
        )
        evidence = bool(
            delivered.get("read_evidence")
            and delivered.get("compared_metrics")
            and chosen.get("run_id") == row["summary"]["run_id"]
            and not fact_errors
        )
        final = final_matches(
            delivered.get("final", ""),
            final_payload(
                chosen.get("action"), run_id=row["summary"]["run_id"]
            ),
        )
        texts = [
            ("reason", str(chosen.get("reason", ""))),
            ("final", str(delivered.get("final", ""))),
        ]
        explanation_issues = (
            []
            if delivered.get("error")
            else _operator_explanation_contradictions(
                row["summary"], chosen, texts
            )
        )
        explanation_correct = not explanation_issues
        return {
            "execution_complete": bool(chosen) and not delivered.get("error"),
            "action_correct": bool(action),
            "exact_action_correct": bool(exact_action),
            "evidence_correct": evidence,
            "assessment_issues": fact_errors,
            "parameters_correct": not parameter_errors,
            "parameter_issues": parameter_errors,
            "analysis_correct": bool(action and evidence),
            "final_consistent": final,
            "explanation_correct": explanation_correct,
            "explanation_issues": explanation_issues,
            "verified": bool(
                action
                and evidence
                and final
                and explanation_correct
                and not parameter_errors
            ),
            "expected_action": row["target"]["action"],
            "actual_action": chosen.get("action"),
        }
    chosen = delivered.get("decision") or {}
    expected = decision(row["summary"])
    action_correct = (
        not delivered.get("error")
        and chosen.get("action") == expected["action"]
    )
    issues = (
        evidence_issues(row, delivered)
        if not delivered.get("error")
        else ["execution_error"]
    )
    evidence_correct = not issues
    explanation_issues = []
    for source, text in (
        ("reason", str(chosen.get("reason", ""))),
        ("final", str(delivered.get("final", ""))),
    ):
        if _unsupported_claims(text):
            explanation_issues.append("unsupported_claim_in_" + source)
    final_text = str(delivered.get("final", ""))
    expected_deltas = compare_metrics(row["summary"])
    contradictions = []
    if not final_matches(
        final_text, final_payload(chosen.get("action"), expected_deltas)
    ):
        contradictions.append("final_report_does_not_match_recorded_evidence")
    for label, pattern, key in (
        ("agreement", _DELTA_AGREEMENT, "agreement_delta_pp"),
        ("nll", _DELTA_NLL, "nll_delta"),
    ):
        match = pattern.search(final_text)
        if match and abs(float(match.group(1)) - expected_deltas[key]) > 0.011:
            contradictions.append("final_" + label + "_delta_contradicts")
    if _unsupported_claims(final_text):
        contradictions.append("final_unsupported_claim")
    return {
        "execution_complete": not delivered.get("error") and bool(chosen),
        "final_consistent": not contradictions,
        "final_contradictions": contradictions,
        "action_correct": action_correct,
        "evidence_correct": evidence_correct,
        "evidence_issues": issues,
        "explanation_correct": not explanation_issues,
        "explanation_issues": explanation_issues,
        # Analysis requires the right action and metric deltas.
        # Prose quality has its own result above.
        "analysis_correct": action_correct and evidence_correct,
        "verified": action_correct
        and evidence_correct
        and not contradictions
        and not explanation_issues,
        "read_evidence": delivered.get("read_evidence", False),
        "compared_metrics": delivered.get("compared_metrics", False),
        "expected_action": expected["action"],
        "actual_action": chosen.get("action"),
    }


def final_matches(text, expected):
    """Reject free-form claims that have not been checked against the tool evidence."""
    try:
        actual = json.loads(text)
    except (ValueError, TypeError):
        return False
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    for key, value in expected.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if (
                not finite_number(actual[key])
                or abs(actual[key] - value) > 1e-6
            ):
                return False
        elif type(actual[key]) is not type(value) or actual[key] != value:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        required=True,
        help="JSONL file containing review episodes",
    )
    parser.add_argument(
        "--endpoint", required=True, help="Chat-completions endpoint URL"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name expected by the serving endpoint",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for conversations, decisions, and grades",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Maximum episodes to run (0: all)"
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("Replay limit cannot be negative")
    rows = load_rows(args.episodes)
    if args.limit:
        rows = rows[: args.limit]
    client = ChatClient(args.endpoint, args.model)
    results = []
    for row in rows:
        dest = safe_episode_dir(Path(args.output), row["id"])
        delivered = execute(row, client, dest)
        result = safe_grade(row, delivered)
        atomic_json(dest / "grade.json", result)
        results.append(result)
    report = {
        "n": len(results),
        "model": args.model,
        "execution_complete": sum(r["execution_complete"] for r in results),
        "analysis_correct": sum(r["analysis_correct"] for r in results),
        "results": results,
    }
    atomic_json(Path(args.output) / "report.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "results"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
