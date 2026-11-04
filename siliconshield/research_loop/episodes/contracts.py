"""Validate review examples, calculate metric changes, and assign training targets."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, TypeGuard, Union, cast

JsonValue = Union[
    None,
    bool,
    int,
    float,
    str,
    Sequence["JsonValue"],
    Mapping[str, "JsonValue"],
]
"""Any JSON-serializable tree.

Sequence/Mapping (not list/dict) keep the alias covariant, so ``list[str]``
and ``dict[str, str]`` values are accepted where ``JsonValue`` is expected.
Use explicit ``isinstance`` narrowing -- never ``Any`` -- before touching
unknown JSON inputs."""


class DevelopmentMetrics(TypedDict):
    """One development checkpoint: population size and quality metrics."""

    decisions: int
    agreement: float
    nll: float


class DevelopmentSummary(TypedDict, total=False):
    """Measured development evidence admitted for checkpoint review."""

    run_id: str
    split_role: str
    phase: str
    completed_updates: int
    initial: DevelopmentMetrics
    selected: DevelopmentMetrics
    reviewed_checkpoint: str
    security: Mapping[str, JsonValue]
    development_acceptance: JsonValue
    selected_checkpoint: JsonValue
    selection_key: JsonValue
    stop_reason: JsonValue


class MetricDeltas(TypedDict):
    """Like-for-like checkpoint comparison over one development population."""

    agreement_delta_pp: float
    nll_delta: float
    decisions: int
    quality_claim: str


class ReviewDecision(MetricDeltas):
    """Checkpoint-triage action with its evidence and explanation."""

    action: str
    reason: str


SYSTEM = (
    "You operate a local LAYA research workflow. Read evidence before deciding. "
    "Use the supplied tools, report measured deltas accurately, and distinguish functional smoke "
    "tests, reused development evidence, and independent validation. Never infer model quality "
    "from successful execution alone. A recommendation is not a guarantee that an experiment will help. "
    "Explain the decision in the tool call reason. Your final response must be only a JSON object with "
    'action, agreement_delta_pp, nll_delta and scope="open_development_only", matching the recorded decision. '
    "Reasoning strength: high."
)


def tool(
    name: str,
    description: str,
    properties: Mapping[str, JsonValue],
    required: list[str],
) -> dict[str, JsonValue]:
    """Build one chat-completions function-tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    tool(
        "research.read_run",
        "Read a registered experiment development summary.",
        {"run_id": {"type": "string"}},
        ["run_id"],
    ),
    tool(
        "research.compare_metrics",
        "Validate aligned development metrics and calculate changes.",
        {"run_id": {"type": "string"}},
        ["run_id"],
    ),
    tool(
        "research.record_decision",
        "Record an evidence-supported recommendation; does not start a GPU job. "
        "Copy agreement_delta_pp and nll_delta exactly from research.compare_metrics.",
        {
            "run_id": {"type": "string"},
            "action": {
                "type": "string",
                "enum": [
                    "inspect_regression",
                    "inspect_tradeoff",
                    "retain_reviewed",
                    "retain_initial",
                    "smoke_only",
                ],
            },
            "agreement_delta_pp": {"type": "number"},
            "nll_delta": {"type": "number"},
            "reason": {"type": "string"},
        },
        ["run_id", "action", "agreement_delta_pp", "nll_delta", "reason"],
    ),
]


def digest(value: object) -> str:
    """Return the SHA-256 hex digest of a JSON-canonicalized value."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def read_json(path: Union[str, Path]) -> JsonValue:
    """Read and parse a JSON file. Raises on missing files or bad JSON."""
    return cast(JsonValue, json.loads(Path(path).read_text()))


def atomic_json(path: Union[str, Path], value: JsonValue) -> None:
    """Write JSON atomically: fsync a temp file, then replace the target.

    Creates parent directories. Never leaves a half-written receipt behind;
    a crash leaves either the old file or the complete new one.
    """
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def compare_metrics(summary: DevelopmentSummary) -> MetricDeltas:
    """Validate denominators, then compare like-for-like checkpoints.

    Requires positive integer populations, finite in-range metrics, equal
    populations, and the ``open_development`` split role. Raises
    ValueError describing the first failed check.
    """
    initial = summary["initial"]
    selected = summary["selected"]
    for metrics in (initial, selected):
        n = metrics["decisions"]
        if type(n) is not int or n <= 0:
            raise ValueError(
                "Development population must be a positive integer"
            )
        for field in ("agreement", "nll"):
            v: object = metrics.get(field)
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
            ):
                raise ValueError("Metric must be a finite number: " + field)
        if not 0 <= metrics["agreement"] <= 1 or metrics["nll"] < 0:
            raise ValueError("Metric outside its valid range")
    if initial["decisions"] != selected["decisions"]:
        raise ValueError("Cannot compare unequal development populations")
    if summary.get("split_role") != "open_development":
        raise ValueError(
            "Only registered open development evidence is admitted"
        )
    agreement = 100 * (selected["agreement"] - initial["agreement"])
    nll = selected["nll"] - initial["nll"]
    return {
        "agreement_delta_pp": round(agreement, 6),
        "nll_delta": round(nll, 8),
        "decisions": initial["decisions"],
        "quality_claim": "open_development_only",
    }


def decision(summary: DevelopmentSummary) -> ReviewDecision:
    """Choose a review action using the fixed development-metric policy.

    Returns the action, its explanation, and the supporting deltas.
    """
    delta = compare_metrics(summary)
    a, n = delta["agreement_delta_pp"], delta["nll_delta"]
    if (
        summary.get("phase") == "smoke"
        or summary.get("completed_updates", 0) <= 2
    ):
        action = "smoke_only"
        reason = "This is a functional smoke run. Confirm finite learning and loading; do not claim a quality gain."
    elif a < -1e-5 and n > 1e-7:
        action = "inspect_regression"
        reason = "Both development agreement and NLL worsened. Retain the initial checkpoint and inspect failures."
    elif (a < -1e-5 and n < -1e-7) or (a > 1e-5 and n > 1e-7):
        action = "inspect_tradeoff"
        reason = "Agreement and NLL disagree. Inspect error populations and probability quality before promotion."
    elif (a > 1e-5 and n <= 1e-7) or (n < -1e-7 and a >= -1e-5):
        action = "retain_reviewed"
        reason = "Retain the reviewed checkpoint as a development candidate; confirm on independent cases later."
    else:
        action = "retain_initial"
        reason = "No measured development benefit justifies replacement. Keep the initial checkpoint and investigate."
    reason += (
        f" Agreement change: {a:+.4f} percentage points; NLL change: {n:+.6f}."
    )
    reason += " This reused development population does not establish unseen-family generalization."
    return {"action": action, "reason": reason, **delta}


REBUILD_HINT = (
    "rebuild episodes with python -m siliconshield.research_loop.episodes.dataset "
    "from the original development receipts"
)
STRUCTURED_FIELDS = ("agreement_delta_pp", "nll_delta")


def finite_number(value: object) -> TypeGuard[float | int]:
    """Return true only for finite int/float values (never bools).

    Doubles as a narrowing guard: a true result means ``value`` is usable
    in metric arithmetic.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


# Actions the operator execution path actually honors (BOUNDED_INTERVENTIONS
# minus finalize/report, which run unconditionally after cycles complete).
OPERATOR_ACTIONS = [
    "continue",
    "leave_unchanged",
    "stop",
    "adjust_lr",
    "resume",
    "finalize_retry",
]
OPERATOR_TOOLS = [
    tool(
        "operator.inspect_run",
        "Read the current run state, phase, and receipts.",
        {"run_id": {"type": "string"}},
        ["run_id"],
    ),
    tool(
        "operator.inspect_resources",
        "Read GPU allocation and reservation state.",
        {"run_id": {"type": "string"}},
        ["run_id"],
    ),
    tool(
        "operator.decide",
        "Record one bounded operator action; does not itself execute it.",
        {
            "run_id": {"type": "string"},
            "action": {"type": "string", "enum": OPERATOR_ACTIONS},
            "reason": {"type": "string"},
            "lr": {"type": "number"},
            "checkpoint_dir": {"type": "string"},
            "assessment": {
                "type": "object",
                "description": (
                    "Observed execution_failed (boolean or null), acceptance (passed/failed/unknown), "
                    "selected_step (integer or null), plateau_reductions (integer or null), and comparisons. "
                    "comparisons maps each comparable development view to accuracy_delta_pp, nll_delta, "
                    "and false_safe_delta_pp (after minus before; null when unavailable). "
                    "When checkpoint_candidates are provided, also include checkpoint_review: "
                    "feasible_ids (sorted IDs satisfying every checkpoint_constraint), "
                    "selected_feasible (boolean, or null if the selected ID is absent), "
                    "failed_constraints and unknown_constraints (each ID maps to a list of "
                    "view.metric names, in constraint order; unknown means missing evidence). "
                    "Without checkpoint_candidates, checkpoint_review is null."
                ),
            },
        },
        ["run_id", "action", "reason"],
    ),
]

OPERATOR_SYSTEM = (
    "You operate bounded LAYA training runs. Inspect state and resources before deciding. "
    "Use only the allowlisted actions, respect the intervention budget, and never submit a "
    "duplicate job, resume invalid or test-touched state, or treat a successful exit as proof "
    "of a good intervention. Distinguish execution failure from model-quality evidence. "
    "Explain the decision in the tool call reason. Report observed facts and comparable development "
    "deltas in assessment; use null or unknown for unavailable facts and an empty comparisons object "
    "when populations differ. A failed acceptance gate is distinct from an execution failure. "
    "Do not invent earlier interventions or treat low training loss as evidence of generalization. "
    "End with only JSON containing action, run_id, "
    'executed=false and scope="operator_replay", matching the recorded decision. Reasoning strength: high.'
)


def _tool_definitions_malformed() -> ValueError:
    return ValueError("Episode tool definitions are malformed; " + REBUILD_HINT)


def _validate_tool_contract(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Episodes must carry the current tool contract, including structured deltas.

    Pre-contract episodes are rejected at ingestion (not executed, not trained
    on) with a deterministic rebuild path; their stored summaries are not
    re-parsed from prose.
    """
    raw_tools = row.get("tools")
    if not isinstance(raw_tools, list):
        raise _tool_definitions_malformed()
    definitions: dict[str, JsonValue] = {}
    for entry in raw_tools:
        if not isinstance(entry, dict):
            raise _tool_definitions_malformed()
        function = entry.get("function")
        if not isinstance(function, dict):
            raise _tool_definitions_malformed()
        name = function.get("name")
        if not isinstance(name, str):
            raise _tool_definitions_malformed()
        definitions[name] = function
    record = definitions.get("research.record_decision")
    if record is not None and not isinstance(record, dict):
        raise _tool_definitions_malformed()
    operator_tool = definitions.get("operator.decide")
    if record is None and isinstance(operator_tool, dict):
        parameters = operator_tool.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(
                "Operator episode tool contract is incomplete; " + REBUILD_HINT
            )
        operator_required = parameters.get("required", [])
        if not isinstance(operator_required, list) or not all(
            key in operator_required for key in ("run_id", "action", "reason")
        ):
            raise ValueError(
                "Operator episode tool contract is incomplete; " + REBUILD_HINT
            )
        return definitions
    if not isinstance(record, dict):
        raise ValueError(
            "Episode uses an obsolete tool contract without structured "
            "agreement_delta_pp/nll_delta; " + REBUILD_HINT
        )
    parameters = record.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(
            "Episode uses an obsolete tool contract without structured "
            "agreement_delta_pp/nll_delta; " + REBUILD_HINT
        )
    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError(
            "Episode uses an obsolete tool contract without structured "
            "agreement_delta_pp/nll_delta; " + REBUILD_HINT
        )
    required_raw = parameters.get("required", [])
    required: set[JsonValue] = (
        set(required_raw) if isinstance(required_raw, list) else set()
    )
    structured_ok = True
    for field in STRUCTURED_FIELDS:
        prop = properties.get(field, {})
        if not isinstance(prop, dict) or prop.get("type") != "number":
            structured_ok = False
    if (
        record is None
        or not all(
            key in required
            for key in ("run_id", "action", "reason", *STRUCTURED_FIELDS)
        )
        or not structured_ok
    ):
        raise ValueError(
            "Episode uses an obsolete tool contract without structured "
            "agreement_delta_pp/nll_delta; " + REBUILD_HINT
        )
    return definitions


def validate_episode(row: dict[str, JsonValue]) -> None:
    """Reject malformed episodes deterministically with a rebuild path.

    Raises ValueError describing the first structural problem found.
    Accepts only the current tool contract; see _validate_tool_contract.
    """
    if not row.get("id") or not row.get("group_id"):
        raise ValueError("Episode and lineage group IDs are required")
    verification = row.get("verification", {})
    if not isinstance(verification, dict) or verification.get("kind") not in (
        "checked_calculation",
        "authored_fixture",
    ):
        raise ValueError("Episode has no supported verification provenance")
    definitions = _validate_tool_contract(row)
    names = set(definitions)
    pending: dict[str, str] = {}
    seen: set[str] = set()
    completed: set[str] = set()
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            "A complete episode must end with an assistant response"
        )
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        raise ValueError(
            "A complete episode must end with an assistant response"
        )
    for raw in messages:
        if not isinstance(raw, dict):
            raise ValueError("Unknown message role")
        _validate_message(raw, definitions, names, pending, seen, completed)
    if set(pending) != completed:
        raise ValueError("Incomplete tool trace")
    json.dumps(row, allow_nan=False)


def _validate_message(
    message: dict[str, JsonValue],
    definitions: dict[str, JsonValue],
    names: set[str],
    pending: dict[str, str],
    seen: set[str],
    completed: set[str],
) -> None:
    """Validate one trace message against the tool contract.

    Tool responses must answer a pending call exactly once; assistant
    turns may only proceed when every pending call has a result. Mutates
    the pending/seen/completed sets. Raises ValueError on the first defect.
    """
    role = message.get("role")
    if role not in ("system", "user", "assistant", "tool"):
        raise ValueError("Unknown message role")
    if role == "tool":
        cid = message.get("tool_call_id")
        if not isinstance(cid, str) or cid not in pending or cid in completed:
            raise ValueError("Unmatched or duplicate tool response")
        completed.add(cid)
    elif role == "assistant":
        if set(pending) != completed:
            raise ValueError("Assistant continued before pending tool results")
        raw_calls = message.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise ValueError("Invalid or duplicate tool call")
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                raise ValueError("Invalid or duplicate tool call")
            _validate_tool_call(raw_call, definitions, names, pending, seen)
    text = message.get("content") or ""
    if not isinstance(text, str) or any(
        token in text
        for token in ("<|start|>", "<|eot|>", "<|eom|>", "<|message|>")
    ):
        raise ValueError("Invalid text or embedded role delimiter")


def _validate_tool_call(
    call: dict[str, JsonValue],
    definitions: dict[str, JsonValue],
    names: set[str],
    pending: dict[str, str],
    seen: set[str],
) -> None:
    """Validate one assistant tool call and record it as pending.

    Checks the call identity, the allowlisted tool name, the declared
    argument schema, finite numeric deltas, and the adjust_lr/resume
    parameter rules. Mutates pending/seen. Raises ValueError on defect.
    """
    cid = call.get("id")
    fn = call.get("function")
    if not isinstance(fn, dict):
        raise ValueError("Invalid or duplicate tool call")
    name = fn.get("name")
    raw_arguments = fn.get("arguments")
    if (
        not isinstance(cid, str)
        or not isinstance(name, str)
        or cid in seen
        or name not in names
        or not isinstance(raw_arguments, dict)
    ):
        raise ValueError("Invalid or duplicate tool call")
    arguments: dict[str, JsonValue] = raw_arguments
    parameters = definitions[name]
    if not isinstance(parameters, dict):
        raise ValueError("Invalid or duplicate tool call")
    schema = parameters.get("parameters")
    if not isinstance(schema, dict):
        raise ValueError("Invalid or duplicate tool call")
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    if (
        not isinstance(required, list)
        or not isinstance(properties, dict)
        or not set(required) <= set(arguments)
        or set(arguments) - set(properties)
    ):
        raise ValueError("Tool arguments do not match the declared schema")
    for key, value in arguments.items():
        prop = properties.get(key, {})
        if not isinstance(prop, dict):
            raise ValueError("Tool arguments do not match the declared schema")
        if prop.get("type") == "string" and not isinstance(value, str):
            raise ValueError("Tool argument must be text: " + key)
        if prop.get("type") == "number" and not finite_number(value):
            raise ValueError("Tool argument must be finite: " + key)
        if "enum" in prop:
            enum = prop["enum"]
            if not isinstance(enum, list):
                raise ValueError(
                    "Tool contract has a malformed allowlist; " + REBUILD_HINT
                )
            if value not in enum:
                raise ValueError("Unknown allowlisted action")
    if fn["name"] == "research.record_decision" and not all(
        finite_number(arguments.get(f)) for f in STRUCTURED_FIELDS
    ):
        raise ValueError(
            "Episode record_decision call lacks finite structured deltas; "
            + REBUILD_HINT
        )
    if (
        fn["name"] == "operator.decide"
        and arguments.get("action") == "adjust_lr"
        and not finite_number(arguments.get("lr"))
    ):
        raise ValueError(
            "Episode adjust_lr decision lacks a finite lr; " + REBUILD_HINT
        )
    if (
        fn["name"] == "operator.decide"
        and arguments.get("action") == "resume"
        and not isinstance(arguments.get("checkpoint_dir"), str)
    ):
        raise ValueError(
            "Episode resume decision lacks a checkpoint_dir; " + REBUILD_HINT
        )
    pending[cid] = name
    seen.add(cid)


def call_message(
    name: str, args: Mapping[str, JsonValue], cid: str
) -> dict[str, JsonValue]:
    """Build one assistant tool-call message with no visible content."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    }


def final_payload(
    action: str,
    deltas: Union[MetricDeltas, None] = None,
    run_id: Union[str, None] = None,
) -> dict[str, JsonValue]:
    """Build the receipt the final assistant message must match exactly.

    With deltas, a checkpoint-review receipt; otherwise an operator-replay
    receipt for ``run_id``. Callers must serialize with sorted keys.
    """
    if deltas is not None:
        return {
            "action": action,
            "agreement_delta_pp": deltas["agreement_delta_pp"],
            "nll_delta": deltas["nll_delta"],
            "scope": "open_development_only",
        }
    return {
        "action": action,
        "run_id": run_id,
        "executed": False,
        "scope": "operator_replay",
    }


def episode_from_summary(
    summary: DevelopmentSummary,
    group_id: str,
    provenance: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Build, validate, and return one review episode for the summary.

    Raises ValueError when the summary fails validation; see validate_episode.
    """
    target = decision(summary)
    run_id = summary["run_id"]
    messages: list[dict[str, JsonValue]] = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Review experiment {run_id}. Read its evidence, calculate checkpoint changes, "
            "and record the appropriate next action under the checkpoint-triage policy.",
        },
        call_message("research.read_run", {"run_id": run_id}, "read"),
        {
            "role": "tool",
            "tool_call_id": "read",
            "content": json.dumps(summary, sort_keys=True),
        },
        call_message("research.compare_metrics", {"run_id": run_id}, "compare"),
        {
            "role": "tool",
            "tool_call_id": "compare",
            "content": json.dumps(compare_metrics(summary)),
        },
        call_message(
            "research.record_decision",
            {
                "run_id": run_id,
                "action": target["action"],
                "agreement_delta_pp": target["agreement_delta_pp"],
                "nll_delta": target["nll_delta"],
                "reason": target["reason"],
            },
            "record",
        ),
        {
            "role": "tool",
            "tool_call_id": "record",
            "content": '{"recorded": true, "job_submitted": false}',
        },
        {
            "role": "assistant",
            "content": json.dumps(
                final_payload(target["action"], target), sort_keys=True
            ),
        },
    ]
    row: dict[str, JsonValue] = {
        "id": digest(summary)[:20],
        "group_id": group_id,
        "tools": TOOLS,
        "messages": messages,
        # TypedDicts are plain JSON dicts at runtime; the cast records the
        # structural promise mypy cannot prove for heterogeneous mappings.
        "summary": cast(JsonValue, summary),
        "target": cast(JsonValue, target),
        "verification": provenance,
    }
    validate_episode(row)
    return row
