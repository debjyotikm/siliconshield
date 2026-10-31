"""Preference pairs for factual grounding of the Glimmer operator.

Each pair holds the observation fixed and contrasts two responses with the
same action: a chosen response whose explanation is supported by the
observation, and a rejected response whose explanation contradicts it or
cites unobserved evidence. This isolates factual grounding instead of
teaching a simplistic preference for stopping or continuing.

Provenance is explicit per pair. Authored corrections are marked as such;
they are not model samples. The sealed Glimmer test, the consumed LAYA
test, and unfamiliar-RTL results are excluded from pair construction.
"""

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Union

from siliconshield.research_loop.controller.replay import (
    _operator_explanation_contradictions,
)
from siliconshield.research_loop.episodes.contracts import (
    OPERATOR_SYSTEM,
    JsonValue,
    final_payload,
)

EXCLUDED_SOURCES = (
    "reserved controller test",
    "task-model final test",
    "reserved transfer challenge",
)

# Overlap guard: synthetic example only. Real campaign histories live outside
# Git in private manifests; committed tests must use authored synthetic values.
# This example uses steps 201/301 and must stay out of training when marked
# regression-only.
REGRESSION_ONLY_IDS = frozenset(
    {"dpo-stop-grounding-v1", "dpo-baseline-stop-supported"}
)
OVERLAP_NOTE = (
    "Synthetic example shares its history with the regression-check "
    "baseline illustration; regression use only."
)

# Authored synthetic correction: action right, as-stated reason not. Wording is
# fixed and reviewed; do not paraphrase silently. Values are illustrative,
# not campaign measurements.
CORRECTED_STOP_REASON = (
    "Development accuracy fell from 82.31% at step 201 to 80.12% at step 301, "
    "while NLL increased from 0.0412 to 0.0523. Stop under the configured regression "
    "rule and preserve the selected checkpoint. No execution failure was observed."
)

# Authored unsupported explanation for the synthetic rejected response.
AS_STATED_STOP_REASON = (
    "The worker crashed. Stop the run and preserve the execution error."
)


def _observation(
    run_id: str,
    history_entries: list[dict[str, JsonValue]],
    systemd: str = "running",
    execution_failed: bool = False,
    **extra: JsonValue,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Build one inspectable run state plus its resource snapshot."""
    state: dict[str, JsonValue] = {
        "run_id": run_id,
        "phase": "training",
        "systemd": systemd,
        "execution_failed": execution_failed,
        "development_history": {
            "entries": history_entries,
            "complete": True,
            "total": len(history_entries),
        },
        **extra,
    }
    resources: dict[str, JsonValue] = {
        "run_id": run_id,
        "gpus": [0, 1, 2],
        "reservation_owner": None,
        "budget": {"max_lr_adjustments": 1, "lr_bounds": [1e-6, 4e-5]},
    }
    return state, resources


def _messages(
    run_id: str,
    state: Mapping[str, JsonValue],
    resources: Mapping[str, JsonValue],
) -> list[dict[str, JsonValue]]:
    """Build the inspect/state prompt; the observation stays fixed per pair."""
    return [
        {"role": "system", "content": OPERATOR_SYSTEM},
        {
            "role": "user",
            "content": "Inspect run %s and record the appropriate bounded action."
            % run_id,
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "inspect",
                    "type": "function",
                    "function": {
                        "name": "operator.inspect_run",
                        "arguments": {"run_id": run_id},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "inspect",
            "content": json.dumps(state, sort_keys=True),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "resources",
                    "type": "function",
                    "function": {
                        "name": "operator.inspect_resources",
                        "arguments": {"run_id": run_id},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "resources",
            "content": json.dumps(resources, sort_keys=True),
        },
    ]


def _response(
    decide_args: Mapping[str, JsonValue], final: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    """Bundle one recorded decision with its final receipt."""
    return {"decide": decide_args, "final": final}


def corrected_stop_pair() -> dict[str, JsonValue]:
    """Synthetic stop with authored supported reason vs unsupported reason."""
    run_id = "run-synth-regression-stop"
    history: list[dict[str, JsonValue]] = [
        {
            "step": 201,
            "time": 1000.0,
            "event": "development",
            "metrics": {
                "accuracy": 0.82311,
                "nll": 0.0412,
                "false_safe_rate": 0.011,
            },
        },
        {
            "step": 301,
            "time": 2000.0,
            "event": "development",
            "metrics": {
                "accuracy": 0.80123,
                "nll": 0.0523,
                "false_safe_rate": 0.022,
            },
        },
    ]
    state, resources = _observation(run_id, history, step=301)
    prompt = _messages(run_id, state, resources)
    final = final_payload("stop", run_id=run_id)
    chosen = _response(
        {"run_id": run_id, "action": "stop", "reason": CORRECTED_STOP_REASON},
        final,
    )
    rejected = _response(
        {"run_id": run_id, "action": "stop", "reason": AS_STATED_STOP_REASON},
        final,
    )
    return {
        "id": "dpo-stop-grounding-v1",
        "action": "stop",
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "provenance": {
            "kind": "authored_correction",
            "note": (
                "Authored illustrative observations and responses; not model samples. "
                + OVERLAP_NOTE
            ),
            "overlap": "baseline regression_stop_supported",
            "regression_only": True,
            "excluded": list(EXCLUDED_SOURCES),
        },
    }


def is_regression_only(pair: Mapping[str, JsonValue]) -> bool:
    """Whether a pair must stay out of training (held-out overlap)."""
    if pair.get("id") in REGRESSION_ONLY_IDS:
        return True
    provenance = pair.get("provenance", {})
    if not isinstance(provenance, dict):
        return False
    return bool(provenance.get("regression_only"))


def training_pairs(
    pairs: list[dict[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    """Pairs eligible for DPO training (excludes regression-only overlap)."""
    return [p for p in pairs if not is_regression_only(p)]


def pair_content_hash(pair: Mapping[str, JsonValue]) -> str:
    """Stable hash of the scored bytes: prompt + both responses.

    Changing any scored token while keeping the pair ID must invalidate
    cached reference scores. Uses canonical JSON so file formatting does
    not affect the binding.
    """
    canonical = {
        "prompt": pair.get("prompt"),
        "chosen": pair.get("chosen"),
        "rejected": pair.get("rejected"),
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def same_action_pair(
    run_id: str,
    state: Mapping[str, JsonValue],
    resources: Mapping[str, JsonValue],
    action: str,
    supported_reason: str,
    unsupported_reason: str,
    pair_id: str,
    provenance_note: str,
    extra_args: Union[dict[str, JsonValue], None] = None,
) -> dict[str, JsonValue]:
    """One observation, one action, two explanations: only one is supported."""
    decide_base = {"run_id": run_id, "action": action, **(extra_args or {})}
    prompt = _messages(run_id, state, resources)
    final = final_payload(action, run_id=run_id)
    chosen = _response({**decide_base, "reason": supported_reason}, final)
    rejected = _response({**decide_base, "reason": unsupported_reason}, final)
    provenance: dict[str, JsonValue] = {
        "kind": "same_action_explanation",
        "note": provenance_note,
        "excluded": list(EXCLUDED_SOURCES),
    }
    if pair_id in REGRESSION_ONLY_IDS:
        provenance["overlap"] = "baseline regression_stop_supported"
        provenance["regression_only"] = True
        provenance["note"] = provenance_note + " " + OVERLAP_NOTE
    return {
        "id": pair_id,
        "action": action,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "provenance": provenance,
    }


def validate_pair(pair: Mapping[str, JsonValue]) -> list[str]:
    """Check one pair: chosen clean, rejected contradictory, action fixed.

    Returns a list of issues; an empty list means the pair is valid.
    Malformed inputs yield issues, never exceptions.
    """
    issues: list[str] = []
    if set(pair) < {
        "id",
        "action",
        "prompt",
        "chosen",
        "rejected",
        "provenance",
    }:
        return ["pair schema incomplete"]
    prompt = pair.get("prompt")
    if not isinstance(prompt, list):
        return ["pair schema incomplete"]
    try:
        observation = next(
            m
            for m in prompt
            if isinstance(m, dict)
            and m.get("role") == "tool"
            and m.get("tool_call_id") == "inspect"
        )
        content = observation.get("content")
        if not isinstance(content, str):
            raise ValueError("inspect observation has no text content")
        state = json.loads(content)
    except (StopIteration, ValueError) as exc:
        return ["prompt lacks a parseable inspect observation: %s" % exc]
    for side in ("chosen", "rejected"):
        response = pair.get(side)
        if not isinstance(response, dict):
            issues.append("%s decision is incomplete" % side)
            continue
        decide = response.get("decide")
        if not isinstance(decide, dict) or not {
            "run_id",
            "action",
            "reason",
        } <= set(decide):
            issues.append("%s decision is incomplete" % side)
    if issues:
        return issues
    for side, expect_clean in (("chosen", True), ("rejected", False)):
        response = pair.get(side)
        decide = response.get("decide") if isinstance(response, dict) else None
        reason = decide.get("reason") if isinstance(decide, dict) else None
        final = response.get("final") if isinstance(response, dict) else None
        if (
            not isinstance(decide, dict)
            or not isinstance(reason, str)
            or not isinstance(final, dict)
        ):
            issues.append("%s decision is incomplete" % side)
            continue
        contradictions = _operator_explanation_contradictions(
            state,
            decide,
            [
                ("reason", reason),
                ("final", json.dumps(final)),
            ],
        )
        if expect_clean and contradictions:
            issues.append(
                "chosen response contradicts the observation: %s"
                % contradictions
            )
        if not expect_clean and not contradictions:
            issues.append(
                "rejected response shows no explanation contradiction"
            )
    chosen = pair.get("chosen")
    rejected = pair.get("rejected")
    if (
        not isinstance(chosen, dict)
        or not isinstance(rejected, dict)
        or not isinstance(chosen.get("decide"), dict)
        or not isinstance(rejected.get("decide"), dict)
        or chosen["decide"].get("action") != rejected["decide"].get("action")
    ):
        issues.append(
            "pair must hold the action fixed to isolate explanation grounding"
        )
    provenance = pair.get("provenance", {})
    if not isinstance(provenance, dict) or provenance.get("kind") not in (
        "authored_correction",
        "same_action_explanation",
        "trace_derived_reviewed",
    ):
        issues.append("pair provenance kind is not a reviewed category")
    return issues


def write_pairs(
    pairs: list[dict[str, JsonValue]], path: Union[str, os.PathLike[str]]
) -> dict[str, Union[str, int]]:
    """Validate every pair, then write the JSONL file with exclusive creation.

    Refusing to overwrite an existing file keeps a reviewed pair file from
    being silently replaced. Returns the pair count, SHA-256, and path.
    Raises ValueError naming the first invalid pair.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for pair in pairs:
            issues = validate_pair(pair)
            if issues:
                raise ValueError(
                    "Pair %s invalid: %s" % (pair.get("id"), issues)
                )
            stream.write(json.dumps(pair, allow_nan=False) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"pairs": len(pairs), "sha256": digest, "path": str(path)}


def main() -> None:
    """Write reviewed pairs outside Git from explicit CLI flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="JSONL file for reviewed pairs (outside Git)",
    )
    parser.add_argument("--include-corrected-stop", action="store_true")
    args = parser.parse_args()
    pairs = [corrected_stop_pair()] if args.include_corrected_stop else []
    result = write_pairs(pairs, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
