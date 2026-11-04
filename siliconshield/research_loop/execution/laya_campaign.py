"""Persistent multi-round campaign state around the bounded operator.

One campaign directory (outside Git) holds::

    campaign.json              registered campaign: parent export, recipe,
                               reporting contract, selection policy
    incumbent.json             frozen incumbent export and its results
    datasets.json              registered immutable dataset versions
    rounds/<round-id>/round.json   per-round receipt with phase history
    rounds/<round-id>/recipe.json  round recipe: registered settings plus
                               the frozen trainer selection reference

Round phases persist every transition::

    prepare -> validate_data -> preflight -> acquire -> submit -> observe
    -> evaluate -> decide -> report -> done

Any active phase may transition to ``failed`` instead; the incumbent is
preserved and the failure recorded. The single-job lifecycle underneath
(systemd ownership, resume validation, finalization) stays in
:mod:`siliconshield.research_loop.execution.operator`; this module
schedules rounds and applies the registered selection policy.

Identity binds every round to the parent export, the incumbent it was
compared against, the dataset version, the recipe, the policy, and the
reporting contract. Selection inputs are normalized development views
with count-derived metrics, matched population and inference bindings,
and no float inputs, so incomparable or out-of-range results are
refused rather than compared.
"""

# Non-meta local decision models are not supported. Use at own risk.

import dataclasses
import fcntl
import hashlib
import json
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeGuard, cast

from siliconshield.research_loop.adapters import laya_training
from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    digest,
    read_json,
)
from siliconshield.research_loop.execution.security_jobs import (
    BOOL_FLAGS,
    CHOICE_FLAGS,
    FLAGS,
)
from siliconshield.schemas import (
    CAMPAIGN_ROUND_SCHEMA,
    LAYA_CAMPAIGN_SCHEMA,
)

PHASES = (
    "prepare",
    "validate_data",
    "preflight",
    "acquire",
    "submit",
    "observe",
    "evaluate",
    "decide",
    "report",
    "done",
)
TERMINAL = ("done", "failed")

ROUND_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
HEX64 = re.compile(r"[0-9a-f]{64}")
TOLERANCE = 1e-9


def _sha(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _hex(value: Any, name: str) -> str:
    """Require a lowercase SHA-256 hex digest string."""
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError("Expected a SHA-256 hex digest: " + name)
    return value


def _is_count(value: Any) -> TypeGuard[int]:
    """True for a genuine JSON integer count (never a boolean)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _campaign_file(root: Path) -> Path:
    return Path(root) / "campaign.json"


def _incumbent_file(root: Path) -> Path:
    return Path(root) / "incumbent.json"


def _datasets_file(root: Path) -> Path:
    return Path(root) / "datasets.json"


def _round_file(root: Path, round_id: str) -> Path:
    return Path(root) / "rounds" / round_id / "round.json"


def _round_recipe_file(root: Path, round_id: str) -> Path:
    return Path(root) / "rounds" / round_id / "recipe.json"


def _intent_file(root: Path, round_id: str) -> Path:
    return Path(root) / "rounds" / round_id / "decision-intent.json"


@contextmanager
def _campaign_lock(root: Path) -> Iterator[None]:
    """Serialize multi-write campaign transactions on one host."""
    lock_path = Path(root) / ".campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _record(path: Path) -> dict[str, Any]:
    """Read a JSON object file; refuse anything else."""
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return cast(dict[str, Any], value)


def load_campaign(root: Path) -> dict[str, Any]:
    """Read and schema-check the registered campaign."""
    campaign = _record(_campaign_file(root))
    if campaign.get("schema") != LAYA_CAMPAIGN_SCHEMA:
        raise ValueError("Unknown campaign schema")
    return campaign


def load_incumbent(root: Path) -> dict[str, Any]:
    """Read the frozen incumbent export and its results."""
    return _record(_incumbent_file(root))


def list_rounds(root: Path) -> list[str]:
    """Sorted round ids with a persisted receipt."""
    rounds = Path(root) / "rounds"
    if not rounds.is_dir():
        return []
    return sorted(
        child.name
        for child in rounds.iterdir()
        if (child / "round.json").is_file()
    )


def read_round(root: Path, round_id: str) -> dict[str, Any]:
    """Read and schema-check one round receipt."""
    if not isinstance(round_id, str) or not round_id:
        raise ValueError("Round id must be a non-empty string")
    receipt = _record(_round_file(root, round_id))
    if receipt.get("schema") != CAMPAIGN_ROUND_SCHEMA:
        raise ValueError("Unknown round schema: " + round_id)
    return receipt


def round_phase(root: Path, round_id: str) -> str:
    """Return one round's phase, refusing receipts that lack it."""
    phase = read_round(root, round_id).get("phase")
    if not isinstance(phase, str):
        raise ValueError("Round receipt needs a phase: " + str(round_id))
    return phase


def update_campaign_selection(
    root: Path,
    policy: dict[str, Any],
    reason: str,
    *,
    allow_relaxation: bool = False,
) -> dict[str, Any]:
    """Register a new selection policy for future rounds.

    Decided rounds keep the policy content frozen in their own
    receipts; this changes only what the next prepared round binds.
    Refuses while any round is still active, so a policy change can
    never land mid-round, refuses any relaxation of the registered
    thresholds unless explicitly allowed, and appends the previous
    policy digest to a history log with the recorded reason.
    """
    root = Path(root)
    _check_policy(policy)
    if not reason or not str(reason).strip():
        raise ValueError("A selection update needs a recorded reason")
    with _campaign_lock(root):
        campaign = load_campaign(root)
        stored = campaign.get("selection")
        try:
            _check_policy(stored)
        except ValueError as exc:
            raise ValueError(
                "Registered selection policy is malformed"
            ) from exc
        if not allow_relaxation and selection_relaxes(stored, policy):
            raise ValueError("Selection update relaxes a threshold")
        for round_id in list_rounds(root):
            if round_phase(root, round_id) not in TERMINAL:
                raise ValueError(
                    "Selection is frozen while a round is active: " + round_id
                )
        history = Path(root) / "selection_history.jsonl"
        with history.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "reason": str(reason),
                        "previous_digest": digest(stored),
                        "policy_digest": digest(policy),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        campaign["selection"] = policy
        atomic_json(_campaign_file(root), campaign)
    return campaign


def check_selection_policy(policy: dict[str, Any]) -> None:
    """Publicly validate a selection policy's thresholds."""
    _check_policy(policy)


def selection_relaxes(old: Any, new: Any) -> bool:
    """True when any known threshold moves toward leniency.

    Unknown research or deployment keys fail closed: a policy the
    checker cannot compare is treated as a relaxation until the
    comparison learns it.
    """
    if not isinstance(old, dict) or not isinstance(new, dict):
        return True
    try:
        return _relaxes_known(old, new)
    except TypeError:
        return True


def _relaxes_known(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Compare thresholds the checker understands."""
    for section in ("research", "deployment"):
        if not isinstance(new.get(section), dict):
            return True
    old_research = old.get("research", {})
    new_research = new.get("research", {})
    known_research = {
        "min_retention_accuracy",
        "max_retention_false_safe_increase",
        "require_no_missing_context_regression",
        "require_no_pair_regression",
        "min_new_coverage_gain_pp",
        "max_new_coverage_false_safe_increase",
        "family_false_safe_caps",
        "selection_ranking",
        "expansion_views",
    }
    if (
        set(new_research) - known_research
        or new_research.get("min_retention_accuracy", 1)
        < old_research.get("min_retention_accuracy", 0)
        or new_research.get("max_retention_false_safe_increase", 0)
        > old_research.get("max_retention_false_safe_increase", 0)
        or old_research.get("require_no_missing_context_regression")
        and not new_research.get("require_no_missing_context_regression")
        or old_research.get("require_no_pair_regression")
        and not new_research.get("require_no_pair_regression")
        or new_research.get("min_new_coverage_gain_pp", 100)
        < old_research.get("min_new_coverage_gain_pp", 0)
    ):
        return True
    old_budget = old_research.get("max_new_coverage_false_safe_increase")
    new_budget = new_research.get("max_new_coverage_false_safe_increase")
    if old_budget is not None and (
        new_budget is None or new_budget > old_budget
    ):
        return True
    if old_research.get(
        "selection_ranking", laya_training.RANKING_V1_TRANSFER
    ) != new_research.get(
        "selection_ranking", laya_training.RANKING_V1_TRANSFER
    ):
        return True
    try:
        old_caps = laya_training.normalize_family_caps(
            old_research.get("family_false_safe_caps", {})
        )
        new_caps = laya_training.normalize_family_caps(
            new_research.get("family_false_safe_caps", {})
        )
    except ValueError:
        return True
    for role in ("new_coverage", "retention"):
        for name, cap in old_caps[role].items():
            if name not in new_caps[role] or new_caps[role][name] > cap:
                return True
    old_views = old_research.get("expansion_views", {})
    new_views = new_research.get("expansion_views", {})
    if not isinstance(old_views, dict) or not isinstance(new_views, dict):
        return True
    for name, gate in old_views.items():
        new_gate = new_views.get(name)
        if not isinstance(new_gate, dict):
            return True
        if new_gate.get("population_sha256") != gate.get("population_sha256"):
            return True
        if new_gate.get("min_correct", 0) < gate.get("min_correct", 0):
            return True
        if new_gate.get("max_false_safe", 0) > gate.get("max_false_safe", 0):
            return True
        old_groups = gate.get("max_group_false_safe", {}) or {}
        new_groups = new_gate.get("max_group_false_safe", {}) or {}
        if not isinstance(old_groups, dict) or not isinstance(new_groups, dict):
            return True
        for group, cap in old_groups.items():
            if group not in new_groups or new_groups[group] > cap:
                return True
    old_deploy = old.get("deployment", {})
    new_deploy = new.get("deployment", {})
    known_deploy = {
        "max_worst_family_false_safe_rate",
        "min_class_recall",
        "min_pair_accuracy",
    }
    if (
        set(new_deploy) - known_deploy
        or new_deploy.get("max_worst_family_false_safe_rate", 0)
        > old_deploy.get("max_worst_family_false_safe_rate", 1)
        or new_deploy.get("min_class_recall", 1)
        < old_deploy.get("min_class_recall", 0)
        or new_deploy.get("min_pair_accuracy", 1)
        < old_deploy.get("min_pair_accuracy", 0)
    ):
        return True
    return False


def update_campaign_recipe(
    root: Path, path: str, sha256: str, reason: str
) -> dict[str, Any]:
    """Bind a new recipe file for future rounds.

    Prepared rounds keep the recipe bytes frozen in their own
    receipts; this changes only what the next prepared round binds.
    Refuses while any round is still active and when the file bytes
    disagree with the recorded hash, and logs the rotation.
    """
    root = Path(root)
    if not reason or not str(reason).strip():
        raise ValueError("A recipe update needs a recorded reason")
    if not isinstance(path, str) or not isinstance(sha256, str):
        raise ValueError("A recipe update needs a path and hash")
    try:
        actual = _sha(Path(path))
    except OSError as exc:
        raise ValueError("Recipe file is missing: " + path) from exc
    if actual != sha256:
        raise ValueError("Recipe bytes mismatch: " + path)
    with _campaign_lock(root):
        campaign = load_campaign(root)
        for round_id in list_rounds(root):
            if round_phase(root, round_id) not in TERMINAL:
                raise ValueError(
                    "Recipe is frozen while a round is active: " + round_id
                )
        recipe = campaign.get("recipe")
        if not isinstance(recipe, dict) or not isinstance(
            recipe.get("sha256"), str
        ):
            raise ValueError("Campaign recipe block is malformed")
        history = Path(root) / "recipe_history.jsonl"
        with history.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "reason": str(reason),
                        "previous_sha256": recipe["sha256"],
                        "sha256": sha256,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        campaign["recipe"] = {
            "path": str(Path(path).resolve()),
            "sha256": sha256,
        }
        atomic_json(_campaign_file(root), campaign)
    return campaign


def verify_round_bindings(
    root: Path, round_id: str, profile: dict[str, Any]
) -> None:
    """Check a prepared round's job profile against its bindings.

    Runs the same comparison the runner performs before launch, so
    a supervisor can reject a mis-specified round at commit time
    instead of after GPU acquisition.
    """
    receipt = read_round(Path(root), round_id)
    if round_phase(Path(root), round_id) not in ("prepare", "validate_data"):
        raise ValueError("Bindings verify before launch: " + round_id)
    _require_profile_bindings(receipt, profile)


def record_leaderboard_entry(root: Path, round_id: str) -> dict[str, Any]:
    """Append one terminal round's outcome to the leaderboard.

    Keeps every evaluated candidate's key metrics beside the
    verdict, so useful trade-offs survive even when they do not
    promote. Repeating a recorded round returns its entry.
    """
    root = Path(root)
    receipt = read_round(root, round_id)
    if round_phase(root, round_id) not in TERMINAL:
        raise ValueError("Leaderboard records terminal rounds: " + round_id)
    board = Path(root) / "leaderboard.jsonl"
    if board.is_file():
        for line in board.read_text().splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and entry.get("round_id") == round_id:
                return entry
    candidate = receipt.get("candidate") or {}
    if not isinstance(candidate, dict):
        raise ValueError("Round candidate is malformed: " + str(round_id))
    summary = {}
    results = candidate.get("results") or {}
    if not isinstance(results, dict):
        raise ValueError("Round candidate is malformed: " + str(round_id))
    for role, view in results.items():
        if view is None:
            view = {}
        if not isinstance(view, dict):
            raise ValueError("Round candidate is malformed: " + str(round_id))
        counts = view.get("counts") or {}
        if not isinstance(counts, dict):
            raise ValueError("Round candidate is malformed: " + str(round_id))
        summary[role] = {
            key: counts.get(key)
            for key in (
                "n",
                "correct",
                "false_safe",
                "violations",
                "mc_correct",
                "mc_total",
                "pairs_correct",
                "pairs_total",
            )
        }
    entry = {
        "round_id": round_id,
        "phase": receipt["phase"],
        "export": candidate.get("export"),
        "decision": receipt.get("decision"),
        "summary": summary,
        "time": time.time(),
    }
    with board.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def _check_policy(policy: Any) -> None:
    """Require explicit research and deployment thresholds."""
    if not isinstance(policy, dict):
        raise ValueError("Selection policy must be an object")
    for key in ("research", "deployment"):
        if key not in policy:
            raise ValueError("Selection policy is missing: " + key)
    research = policy["research"]
    if not isinstance(research, dict):
        raise ValueError("Research policy must be an object")
    for key in (
        "min_retention_accuracy",
        "max_retention_false_safe_increase",
        "require_no_missing_context_regression",
        "require_no_pair_regression",
        "min_new_coverage_gain_pp",
    ):
        if key not in research:
            raise ValueError("Research policy is missing: " + key)
    deployment = policy["deployment"]
    if not isinstance(deployment, dict):
        raise ValueError("Deployment policy must be an object")
    for key in (
        "max_worst_family_false_safe_rate",
        "min_class_recall",
        "min_pair_accuracy",
    ):
        if key not in deployment:
            raise ValueError("Deployment policy is missing: " + key)
    increase = research.get("max_new_coverage_false_safe_increase")
    if increase is not None and (
        not isinstance(increase, int)
        or isinstance(increase, bool)
        or increase < 0
    ):
        raise ValueError("Research policy needs an integer coverage budget")
    gain = research.get("min_new_coverage_gain_pp")
    if (
        not isinstance(gain, (int, float))
        or isinstance(gain, bool)
        or not math.isfinite(gain)
    ):
        raise ValueError("Research policy needs a gain threshold")
    budget = research.get("max_retention_false_safe_increase")
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise ValueError("Research policy needs a retention budget")
    floor = research.get("min_retention_accuracy")
    if (
        not isinstance(floor, (int, float))
        or isinstance(floor, bool)
        or not math.isfinite(floor)
        or not 0 <= floor <= 1
    ):
        raise ValueError("Research policy needs a retention floor")
    for key in (
        "require_no_missing_context_regression",
        "require_no_pair_regression",
    ):
        if not isinstance(research.get(key), bool):
            raise ValueError("Research policy needs " + key)
    for key in (
        "max_worst_family_false_safe_rate",
        "min_class_recall",
        "min_pair_accuracy",
    ):
        value = deployment.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Deployment policy needs " + key)
    ranking = research.get(
        "selection_ranking", laya_training.RANKING_V1_TRANSFER
    )
    if ranking not in (
        laya_training.RANKING_V1_TRANSFER,
        laya_training.RANKING_V2_RETENTION,
    ):
        raise ValueError("Research policy names an unknown ranking")
    try:
        laya_training.normalize_family_caps(
            research.get("family_false_safe_caps", {})
        )
    except ValueError as exc:
        raise ValueError("Research policy family caps are invalid") from exc
    views = research.get("expansion_views", {})
    if not isinstance(views, dict):
        raise ValueError("Research expansion views must be an object")
    for name, gate in views.items():
        if not isinstance(name, str) or not isinstance(gate, dict):
            raise ValueError("Research expansion view is invalid")
        if not HEX64.fullmatch(gate.get("population_sha256") or ""):
            raise ValueError(
                "Expansion view needs a population sha256: " + name
            )
        for key in ("min_correct", "max_false_safe"):
            value = gate.get(key)
            if not _is_count(value) or value < 0:
                raise ValueError("Expansion view needs %s: %s" % (key, name))
        groups = gate.get("max_group_false_safe", {})
        if not isinstance(groups, dict):
            raise ValueError("Expansion view group caps are invalid: " + name)
        for group, cap in groups.items():
            if not isinstance(group, str) or not _is_count(cap) or cap < 0:
                raise ValueError("Expansion view group cap is invalid: " + name)


def _checked_counts(counts: dict[str, Any]) -> dict[str, Any]:
    """Validate count blocks and derive their rates from integers."""
    for key in (
        "n",
        "correct",
        "false_safe",
        "violations",
        "mc_correct",
        "mc_total",
        "pairs_correct",
        "pairs_total",
    ):
        value = counts.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("Count must be an integer: " + key)
    n = counts["n"]
    if n <= 0:
        raise ValueError("Population needs a positive total")
    if not 0 <= counts["correct"] <= n:
        raise ValueError("Correct count outside its population")
    violations = counts["violations"]
    if not 0 <= violations <= n:
        raise ValueError("Violation count outside its population")
    if not 0 <= counts["false_safe"] <= violations:
        raise ValueError("False-safe count outside its violations")
    for stem in ("mc", "pairs"):
        total = counts[stem + "_total"]
        if total <= 0:
            raise ValueError("Metric needs a positive total: " + stem)
        if not 0 <= counts[stem + "_correct"] <= total:
            raise ValueError("Metric count outside its total: " + stem)
    classes = counts.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("Per-class counts are required")
    recalls = []
    for entry in classes:
        correct, total = entry.get("correct"), entry.get("total")
        if (
            not _is_count(correct)
            or not _is_count(total)
            or total <= 0
            or not 0 <= correct <= total
        ):
            raise ValueError("Per-class counts are invalid")
        recalls.append(correct / total)
    return {
        "n": n,
        "accuracy": counts["correct"] / n,
        "correct": counts["correct"],
        "false_safe": counts["false_safe"],
        "violations": violations,
        "mc_recall": counts["mc_correct"] / counts["mc_total"],
        "pair_accuracy": counts["pairs_correct"] / counts["pairs_total"],
        "pairs_correct": counts["pairs_correct"],
        "class_recall": min(recalls),
    }


def _checked_family(entry: dict[str, Any]) -> dict[str, Any]:
    """Validate a worst-family block with a positive violation base."""
    if not entry.get("id") or not isinstance(entry["id"], str):
        raise ValueError("Worst family needs an identifier")
    false_safe = entry.get("false_safe")
    violations = entry.get("violations")
    if (
        not _is_count(false_safe)
        or not _is_count(violations)
        or violations <= 0
        or not 0 <= false_safe <= violations
    ):
        raise ValueError("Worst-family counts are invalid")
    return {
        "id": entry["id"],
        "false_safe": false_safe,
        "violations": violations,
        "rate": false_safe / violations,
    }


def _checked_families(raw: Any) -> dict[str, dict[str, Any]]:
    """Validate an optional per-family table; absent means unknown."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Family table is invalid")
    families: dict[str, dict[str, Any]] = {}
    for name, entry in raw.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("Family entry is invalid")
        false_safe = entry.get("false_safe")
        violations = entry.get("violations")
        if (
            not _is_count(false_safe)
            or not _is_count(violations)
            or violations <= 0
            or not 0 <= false_safe <= violations
        ):
            raise ValueError("Family counts are invalid: " + name)
        families[name] = {
            "false_safe": false_safe,
            "violations": violations,
            "rate": false_safe / violations,
        }
    return families


def _checked_view(view: dict[str, Any], role: str) -> dict[str, Any]:
    """Validate one development view and derive its metrics."""
    population = view.get("population", {})
    for key in ("dataset_version", "split"):
        if not population.get(key) or not isinstance(population[key], str):
            raise ValueError("Population needs " + key)
    population_sha = _hex(population.get("sha256"), "population sha256")
    if population.get("n") != view.get("counts", {}).get("n"):
        raise ValueError("Population size disagrees with counts")
    inference = view.get("inference", {})
    inference_digest = _hex(inference.get("digest"), "inference digest")
    nll = view.get("nll")
    if nll is not None and (
        not isinstance(nll, (int, float))
        or isinstance(nll, bool)
        or not math.isfinite(nll)
        or nll < 0
    ):
        raise ValueError("View NLL is invalid")
    derived = _checked_counts(view.get("counts", {}))
    derived["nll"] = None if nll is None else float(nll)
    family = _checked_family(view.get("worst_family", {}))
    return {
        "role": role,
        "population": {
            "dataset_version": population["dataset_version"],
            "split": population["split"],
            "sha256": population_sha,
            "n": derived["n"],
        },
        "inference": {"digest": inference_digest},
        "derived": derived,
        "worst_family": family,
        "families": _checked_families(view.get("families")),
    }


def _checked_results(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate both development views of one evaluation."""
    checked = {}
    for role in ("retention", "new_coverage"):
        view = results.get(role)
        if not isinstance(view, dict):
            raise ValueError("Results are missing view: " + role)
        checked[role] = _checked_view(view, role)
    return checked


def _match_views(old: dict[str, Any], new: dict[str, Any], role: str) -> None:
    """Require identical populations and inference before comparing."""
    if old["population"] != new["population"]:
        raise ValueError(
            "Incomparable %s populations: %s vs %s"
            % (
                role,
                old["population"].get("sha256"),
                new["population"].get("sha256"),
            )
        )
    if old["inference"] != new["inference"]:
        raise ValueError("Incomparable %s inference settings" % role)


def _rank_record(checked: dict[str, dict[str, Any]]) -> list[Any]:
    """JSON-safe retention rank components for one checked evaluation."""
    retention = checked["retention"]["derived"]
    coverage = checked["new_coverage"]["derived"]
    return [
        retention["false_safe"],
        -retention["correct"],
        -retention["pairs_correct"],
        retention["nll"],
        coverage["nll"],
    ]


def _rank_tuple(record: list[Any]) -> tuple[Any, ...]:
    """Orderable rank key; unknown NLLs sort after measured ones."""
    return tuple(float("inf") if value is None else value for value in record)


def candidate_rank_key(results: dict[str, Any]) -> list[Any]:
    """Retention rank components for checked campaign results.

    Shared ordering with the trainer's v2-retention ranking:
    fewer retention false-safes, more correct retention decisions,
    more correct retention pairs, lower retention NLL, then
    transfer NLL. Compare with unknown NLLs sorting last.
    """
    return _rank_record(_checked_results(results))


def apply_selection_policy(
    policy: Any,
    incumbent: Any,
    candidate: Any,
    reference: Optional[Any] = None,
) -> dict[str, Any]:
    """Apply research and deployment gates to one candidate.

    Both evaluations must share populations and inference settings
    per view; mismatches raise instead of comparing. Returns research
    promotion, deployment acceptance, and the reasons for each. A
    promising research candidate may still fail the deployment gate.
    ``reference`` adds a record-only comparison against a permanent
    baseline; it never gates promotion by itself.
    """
    _check_policy(policy)
    old = _checked_results(incumbent)
    new = _checked_results(candidate)
    for role in ("retention", "new_coverage"):
        _match_views(old[role], new[role], role)
    research = policy["research"]
    deployment = policy["deployment"]
    reasons = []
    research_ok = True

    retention = new["retention"]["derived"]
    if retention["accuracy"] < research["min_retention_accuracy"]:
        research_ok = False
        reasons.append(
            "retention accuracy %.4f below %.4f"
            % (retention["accuracy"], research["min_retention_accuracy"])
        )
    allowed = (
        old["retention"]["derived"]["false_safe"]
        + research["max_retention_false_safe_increase"]
    )
    if retention["false_safe"] > allowed:
        research_ok = False
        reasons.append(
            "retention false-safe errors %d above budget %d"
            % (retention["false_safe"], allowed)
        )
    coverage_budget = research.get("max_new_coverage_false_safe_increase")
    if coverage_budget is not None:
        budget = old["new_coverage"]["derived"]["false_safe"] + coverage_budget
        if new["new_coverage"]["derived"]["false_safe"] > budget:
            research_ok = False
            reasons.append(
                "new-coverage false-safe errors %d above budget %d"
                % (new["new_coverage"]["derived"]["false_safe"], budget)
            )
    caps = laya_training.normalize_family_caps(
        research.get("family_false_safe_caps", {})
    )
    for role in ("new_coverage", "retention"):
        for name, cap in caps[role].items():
            entry = new[role]["families"].get(name)
            scope = "" if role == "new_coverage" else "retention "
            if entry is None:
                research_ok = False
                reasons.append(
                    "family table lacks capped %sfamily: %s" % (scope, name)
                )
            elif entry["false_safe"] > cap:
                research_ok = False
                reasons.append(
                    "%sfamily %s false-safe errors %d above cap %d"
                    % (scope, name, entry["false_safe"], cap)
                )
    for role in ("retention", "new_coverage"):
        if research["require_no_missing_context_regression"] and (
            new[role]["derived"]["mc_recall"]
            < old[role]["derived"]["mc_recall"]
        ):
            research_ok = False
            reasons.append("missing-context regression on " + role)
        if research["require_no_pair_regression"] and (
            new[role]["derived"]["pair_accuracy"]
            < old[role]["derived"]["pair_accuracy"]
        ):
            research_ok = False
            reasons.append("paired-decision regression on " + role)
    gain_pp = (
        new["new_coverage"]["derived"]["accuracy"]
        - old["new_coverage"]["derived"]["accuracy"]
    ) * 100
    if not laya_training.new_coverage_gain_ok(
        new["new_coverage"]["derived"]["accuracy"],
        old["new_coverage"]["derived"]["accuracy"],
        research["min_new_coverage_gain_pp"],
    ):
        research_ok = False
        reasons.append(
            "new-coverage gain %.2fpp below %.2fpp"
            % (gain_pp, research["min_new_coverage_gain_pp"])
        )
    for name, gate in research.get("expansion_views", {}).items():
        raw = candidate.get(name)
        if not isinstance(raw, dict):
            research_ok = False
            reasons.append("candidate lacks required expansion view: " + name)
            continue
        try:
            expanded = _checked_view(raw, name)
        except ValueError as exc:
            raise ValueError(
                "Expansion view %s is invalid: %s" % (name, exc)
            ) from exc
        if expanded["population"]["sha256"] != gate["population_sha256"]:
            raise ValueError(
                "Expansion view %s scored another population" % name
            )
        derived = expanded["derived"]
        if derived["correct"] < gate["min_correct"]:
            research_ok = False
            reasons.append(
                "expansion view %s correct %d below %d"
                % (name, derived["correct"], gate["min_correct"])
            )
        if derived["false_safe"] > gate["max_false_safe"]:
            research_ok = False
            reasons.append(
                "expansion view %s false-safe errors %d above cap %d"
                % (name, derived["false_safe"], gate["max_false_safe"])
            )
        for group, cap in gate.get("max_group_false_safe", {}).items():
            entry = expanded["families"].get(group)
            if entry is None:
                research_ok = False
                reasons.append(
                    "expansion view %s lacks gated group: %s" % (name, group)
                )
            elif entry["false_safe"] > cap:
                research_ok = False
                reasons.append(
                    "expansion view %s group %s false-safe errors "
                    "%d above cap %d" % (name, group, entry["false_safe"], cap)
                )
    if (
        new["new_coverage"]["derived"]["false_safe"]
        < old["new_coverage"]["derived"]["false_safe"]
    ):
        reasons.append("fewer new-coverage false-safe errors")

    gate_reasons = []
    accepted = True
    coverage = new["new_coverage"]
    worst = coverage["worst_family"]["rate"]
    if worst > deployment["max_worst_family_false_safe_rate"]:
        accepted = False
        gate_reasons.append(
            "worst-family false-safe rate %.4f above %.4f"
            % (worst, deployment["max_worst_family_false_safe_rate"])
        )
    derived = coverage["derived"]
    for key, metric in (
        ("min_class_recall", "class_recall"),
        ("min_pair_accuracy", "pair_accuracy"),
    ):
        if derived[metric] < deployment[key]:
            accepted = False
            gate_reasons.append(
                "%s %.4f below %.4f"
                % (metric, derived[metric], deployment[key])
            )
    verdict: dict[str, Any] = {
        "promote_research": research_ok,
        "deployment_accepted": accepted,
        "research_reasons": reasons,
        "deployment_reasons": gate_reasons,
    }
    if (
        laya_training.ranking_version(research)
        == laya_training.RANKING_V2_RETENTION
    ):
        incumbent_record = _rank_record(old)
        candidate_record = _rank_record(new)
        verdict["ranking"] = {
            "version": laya_training.RANKING_V2_RETENTION,
            "incumbent": incumbent_record,
            "candidate": candidate_record,
            "candidate_ranks_first": _rank_tuple(candidate_record)
            < _rank_tuple(incumbent_record),
        }
    if reference is not None:
        base = _checked_results(reference)
        comparison = {}
        for role in ("retention", "new_coverage"):
            _match_views(base[role], new[role], role)
            first = base[role]["derived"]
            second = new[role]["derived"]
            comparison[role] = {
                "accuracy_pp": (second["accuracy"] - first["accuracy"]) * 100,
                "false_safe_delta": (
                    second["false_safe"] - first["false_safe"]
                ),
                "mc_recall_delta": (second["mc_recall"] - first["mc_recall"]),
                "pair_accuracy_delta": (
                    second["pair_accuracy"] - first["pair_accuracy"]
                ),
            }
        verdict["reference_comparison"] = comparison
    return verdict


def _checked_matrix(block: dict[str, Any]) -> list[list[int]]:
    """Require a 3x3 confusion matrix of nonnegative integers."""
    matrix = block.get("confusion")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(
            not isinstance(row, list)
            or len(row) != 3
            or any(
                not isinstance(cell, int) or isinstance(cell, bool) or cell < 0
                for cell in row
            )
            for row in matrix
        )
    ):
        raise ValueError("Development block needs a 3x3 confusion matrix")
    return matrix


def _checked_rate(block: dict[str, Any], key: str) -> float:
    """Require a finite rate in range from a trainer report block."""
    value = block.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("Trainer rate out of range: " + key)
    return float(value)


def _checked_nll(block: dict[str, Any]) -> float:
    """Require a finite nonnegative NLL from a trainer report block."""
    value = block.get("nll")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Trainer NLL is invalid")
    return float(value)


def _adapt_view(
    block: dict[str, Any],
    population: dict[str, Any],
    inference: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    """Normalize one trainer development block into a selection view.

    Metrics derive from confusion counts and pair totals; reported
    rates must agree within tolerance or the block is refused as
    inconsistent. Population and inference bindings come from the
    caller, which knows the registered dataset and job settings.
    """
    matrix = _checked_matrix(block)
    total = sum(sum(row) for row in matrix)
    if block.get("n") != total or total <= 0:
        raise ValueError("Development block total disagrees")
    supports = [sum(row) for row in matrix]
    if any(support <= 0 for support in supports):
        raise ValueError("Development block lacks a supported class")
    correct = sum(matrix[i][i] for i in range(3))
    false_safe, violations = matrix[1][0], supports[1]
    if block.get("false_safe_count") != false_safe:
        raise ValueError("False-safe count disagrees with confusion")
    if block.get("violation_count") != violations:
        raise ValueError("Violation count disagrees with confusion")
    recalls = [matrix[i][i] / supports[i] for i in range(3)]
    reported_recall = _checked_rate(block, "min_class_recall")
    if abs(min(recalls) - reported_recall) > TOLERANCE:
        raise ValueError("Class recall disagrees with confusion")
    reported_accuracy = _checked_rate(block, "accuracy")
    if abs(correct / total - reported_accuracy) > TOLERANCE:
        raise ValueError("Accuracy disagrees with confusion")
    pairs_total = block.get("complete_pairs")
    if not isinstance(pairs_total, int) or pairs_total <= 0:
        raise ValueError("Pair total is required")
    pairs_rate = _checked_rate(block, "both_pair_members_correct")
    pairs_correct = int(round(pairs_rate * pairs_total))
    if (
        not 0 <= pairs_correct <= pairs_total
        or abs(pairs_rate - pairs_correct / pairs_total) > TOLERANCE
    ):
        raise ValueError("Pair accuracy has no integer count")
    families = block.get("per_family")
    if not isinstance(families, dict) or not families:
        raise ValueError("Per-family breakdown is required")
    ranked: list[tuple[float, str, int, int]] = []
    for name in sorted(families):
        entry = families[name]
        family_fs = entry.get("false_safe_count")
        family_violations = entry.get("violation_count")
        if (
            not _is_count(family_fs)
            or not _is_count(family_violations)
            or family_violations < 0
            or not 0 <= family_fs <= family_violations
        ):
            raise ValueError("Per-family counts are invalid: " + name)
        if family_violations == 0:
            continue
        ranked.append(
            (family_fs / family_violations, name, family_fs, family_violations)
        )
    if not ranked:
        raise ValueError("No family with violations to gate on")
    worst_rate, worst_id, worst_fs, worst_violations = max(ranked)
    reported_worst = _checked_rate(block, "worst_family_false_safe_rate")
    if abs(worst_rate - reported_worst) > TOLERANCE:
        raise ValueError("Worst-family rate disagrees with breakdown")
    for key in ("dataset_version", "split"):
        if not population.get(key) or not isinstance(population[key], str):
            raise ValueError("Population needs " + key)
    families_table = {
        name: {"false_safe": fs, "violations": violations}
        for _, name, fs, violations in ranked
    }
    return {
        "population": {
            "dataset_version": population["dataset_version"],
            "split": population["split"],
            "sha256": _hex(population.get("sha256"), "population sha256"),
            "n": total,
        },
        "inference": {"digest": _hex(inference.get("digest"), "digest")},
        "nll": _checked_nll(block),
        "families": families_table,
        "counts": {
            "n": total,
            "correct": correct,
            "false_safe": false_safe,
            "violations": violations,
            "mc_correct": matrix[2][2],
            "mc_total": supports[2],
            "pairs_correct": pairs_correct,
            "pairs_total": pairs_total,
            "classes": [
                {"correct": matrix[i][i], "total": supports[i]}
                for i in range(3)
            ],
        },
        "worst_family": {
            "id": worst_id,
            "false_safe": worst_fs,
            "violations": worst_violations,
        },
    }


def selection_inputs_from_report(
    transfer: dict[str, Any],
    retention: dict[str, Any],
    *,
    populations: dict[str, dict[str, Any]],
    inference: dict[str, Any],
) -> dict[str, Any]:
    """Normalize transfer and retention blocks into selection inputs.

    ``populations`` carries the registered ``{"transfer": ...,
    "retention": ...}`` population bindings; ``inference`` carries the
    job's settings digest. The transfer block becomes the
    ``new_coverage`` view.
    """
    return {
        "retention": _adapt_view(
            retention, populations["retention"], inference, "retention"
        ),
        "new_coverage": _adapt_view(
            transfer, populations["transfer"], inference, "new_coverage"
        ),
    }


def export_manifest(export: Path) -> dict[str, str]:
    """Hash every file under a candidate export directory."""
    root = Path(export)
    manifest = {}
    for item in sorted(root.rglob("*")):
        if item.is_file():
            manifest[str(item.relative_to(root))] = _sha(item)
    if not manifest:
        raise ValueError("Export holds no files: " + str(root))
    return manifest


def init_campaign(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Register a campaign; refuse to overwrite an existing one.

    Verifies the parent export weights, recipe bytes, evaluation
    receipt, and incumbent results before writing anything.
    ``config`` needs ``campaign_id``, ``parent`` (``export``,
    ``weights_sha256``), ``recipe`` (``path``, ``sha256``),
    ``selection`` policy, and ``incumbent`` (``export``, ``results``,
    ``receipt``). ``reporting_contract`` (``path``, ``sha256``),
    ``operator_profile``, and ``notes`` are optional.
    """
    root = Path(root)
    if _campaign_file(root).exists():
        raise ValueError("Campaign already registered: " + str(root))
    for key in ("campaign_id", "parent", "recipe", "selection", "incumbent"):
        if key not in config:
            raise ValueError("Campaign config is missing: " + key)
    parent = config["parent"]
    weights = Path(parent["export"]) / "model.safetensors"
    if _sha(weights) != parent["weights_sha256"]:
        raise ValueError("Parent export weights mismatch")
    recipe = config["recipe"]
    if _sha(recipe["path"]) != recipe["sha256"]:
        raise ValueError("Campaign recipe mismatch")
    contract = config.get("reporting_contract")
    if contract and _sha(contract["path"]) != contract["sha256"]:
        raise ValueError("Reporting-contract mismatch")
    _check_policy(config["selection"])
    incumbent = config["incumbent"]
    _checked_results(incumbent["results"])
    receipt = _checked_receipt(incumbent.get("receipt"))
    _checked_evaluation(
        receipt["path"],
        incumbent["results"],
        str(Path(incumbent["export"]).resolve()),
        None,
        None,
    )
    campaign: dict[str, Any] = {
        "schema": LAYA_CAMPAIGN_SCHEMA,
        "campaign_id": config["campaign_id"],
        "created": time.time(),
        "parent": {
            "export": str(Path(parent["export"]).resolve()),
            "weights_sha256": parent["weights_sha256"],
        },
        "recipe": {
            "path": str(Path(recipe["path"]).resolve()),
            "sha256": recipe["sha256"],
        },
        "selection": config["selection"],
        "notes": config.get("notes", ""),
    }
    if contract:
        campaign["reporting_contract"] = {
            "path": str(Path(contract["path"]).resolve()),
            "sha256": contract["sha256"],
        }
    if config.get("operator_profile"):
        campaign["operator_profile"] = str(
            Path(config["operator_profile"]).resolve()
        )
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(_campaign_file(root), campaign)
    atomic_json(
        _datasets_file(root),
        {"schema": LAYA_CAMPAIGN_SCHEMA, "versions": {}},
    )
    atomic_json(
        _incumbent_file(root),
        {
            "export": str(Path(incumbent["export"]).resolve()),
            "export_manifest": export_manifest(incumbent["export"]),
            "results": incumbent["results"],
            "receipt": receipt,
            "round_id": incumbent.get("round_id"),
            "promoted": time.time(),
        },
    )
    return campaign


def _checked_receipt(receipt: Any) -> dict[str, str]:
    """Require an evaluation receipt with verified file bytes."""
    if not isinstance(receipt, dict):
        raise ValueError("Evaluation receipt is required")
    path = receipt.get("path")
    if not path or not Path(path).is_file():
        raise ValueError("Evaluation receipt file is missing")
    if _sha(Path(path)) != receipt.get("sha256"):
        raise ValueError("Evaluation receipt bytes mismatch")
    return {
        "path": str(Path(path).resolve()),
        "sha256": receipt["sha256"],
    }


def _checked_evaluation(
    receipt_path: str,
    results: dict[str, Any],
    export: str,
    experiment_id: Optional[str],
    round_id: Optional[str],
) -> None:
    """Require receipt contents to bind model, results, and run.

    The receipt file names the evaluated export's full file-hash
    manifest, the digest of the evaluated results (which carry the
    populations and inference digests), and the run identity. Any
    binding that disagrees with the live files refuses, so a bare
    ``{}`` receipt or one written for another run cannot accompany
    unrelated results.
    """
    content = read_json(Path(receipt_path))
    if not isinstance(content, dict):
        raise ValueError("Evaluation receipt has no bindings")
    if content.get("results_sha256") != digest(results):
        raise ValueError("Evaluation receipt does not bind these results")
    if content.get("export_manifest") != export_manifest(Path(export)):
        raise ValueError("Evaluation receipt does not bind this export")
    run = content.get("run")
    if not isinstance(run, dict):
        raise ValueError("Evaluation receipt names no run")
    if round_id is None:
        if run.get("round_id") is not None:
            raise ValueError("Founding receipt must not name a round")
        if not run.get("experiment_id"):
            raise ValueError("Founding receipt must name its eval run")
        return
    if (
        run.get("experiment_id") != experiment_id
        or run.get("round_id") != round_id
    ):
        raise ValueError("Evaluation receipt names a different run")


def _revalidate_candidate(
    root: Path, round_id: str, candidate: dict[str, Any]
) -> None:
    """Re-hash a recorded candidate and its receipt before promotion.

    Registration-time checks do not survive the wait for selection:
    weights edited after evaluation, or a receipt file swapped under
    its recorded hash, must refuse here rather than promote bytes
    the results never described.
    """
    if not isinstance(candidate, dict):
        raise ValueError("Round candidate is malformed")
    for key in ("export", "export_manifest", "results", "receipt"):
        if key not in candidate:
            raise ValueError("Round candidate is malformed")
    export = candidate["export"]
    if export_manifest(Path(export)) != candidate["export_manifest"]:
        raise ValueError("Candidate export changed after evaluation")
    receipt = candidate["receipt"]
    try:
        receipt_sha = _sha(Path(receipt["path"]))
    except OSError as exc:
        raise ValueError("Candidate receipt changed after evaluation") from exc
    if receipt_sha != receipt["sha256"]:
        raise ValueError("Candidate receipt changed after evaluation")
    campaign = load_campaign(root)
    _checked_evaluation(
        receipt["path"],
        candidate["results"],
        export,
        "%s-%s" % (campaign["campaign_id"], round_id),
        round_id,
    )


def register_dataset(
    root: Path, version: str, manifest_path: str, sha256: str
) -> dict[str, Any]:
    """Record one immutable dataset version manifest for the campaign.

    Refuses to re-register a version with a different hash or to
    register a manifest whose bytes disagree. The manifest itself is
    a dataset-version manifest; its splits and bound pools are
    re-verified before every training use.
    """
    if not isinstance(version, str) or not isinstance(sha256, str):
        raise ValueError("Dataset registration needs a version and hash")
    if not isinstance(manifest_path, str):
        raise ValueError("Dataset registration needs a manifest path")
    record = _record(_datasets_file(root))
    versions = record.get("versions")
    if not isinstance(versions, dict):
        raise ValueError("Campaign dataset registry is malformed")
    if version in versions:
        stored = versions[version]
        if not isinstance(stored, dict) or stored.get("sha256") != sha256:
            raise ValueError(
                "Dataset version re-registered with a new hash: " + version
            )
        return cast(dict[str, Any], stored)
    try:
        actual = _sha(Path(manifest_path))
    except OSError as exc:
        raise ValueError(
            "Dataset manifest is missing: " + manifest_path
        ) from exc
    if actual != sha256:
        raise ValueError("Dataset manifest mismatch: " + version)
    entry = {
        "manifest": str(Path(manifest_path).resolve()),
        "sha256": sha256,
        "registered": time.time(),
    }
    versions[version] = entry
    atomic_json(_datasets_file(root), record)
    return entry


def _revalidate_inputs(
    root: Path, campaign: dict[str, Any], dataset_version: str
) -> None:
    """Re-hash parent, recipe, contract, and dataset before a round."""
    parent = campaign.get("parent")
    if not isinstance(parent, dict):
        raise ValueError("Campaign parent block is malformed")
    weights = Path(str(parent.get("export", ""))) / "model.safetensors"
    try:
        parent_sha = _sha(weights)
    except OSError as exc:
        raise ValueError("Parent export is missing") from exc
    if parent_sha != parent.get("weights_sha256"):
        raise ValueError("Parent export changed since registration")
    recipe = campaign.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("Campaign recipe block is malformed")
    try:
        recipe_sha = _sha(Path(str(recipe.get("path", ""))))
    except OSError as exc:
        raise ValueError("Campaign recipe is missing") from exc
    if recipe_sha != recipe.get("sha256"):
        raise ValueError("Campaign recipe changed since registration")
    contract = campaign.get("reporting_contract")
    if contract:
        if not isinstance(contract, dict):
            raise ValueError("Reporting contract block is malformed")
        try:
            contract_sha = _sha(Path(str(contract.get("path", ""))))
        except OSError as exc:
            raise ValueError("Reporting contract is missing") from exc
        if contract_sha != contract.get("sha256"):
            raise ValueError("Reporting contract changed since registration")
    versions = _record(_datasets_file(root)).get("versions")
    if not isinstance(versions, dict):
        raise ValueError("Campaign dataset registry is malformed")
    entry = versions.get(dataset_version)
    if not isinstance(entry, dict):
        raise ValueError("Unknown dataset version: " + str(dataset_version))
    try:
        entry_sha = _sha(Path(str(entry.get("manifest", ""))))
    except OSError as exc:
        raise ValueError("Dataset manifest is missing") from exc
    if entry_sha != entry.get("sha256"):
        raise ValueError("Dataset manifest changed since registration")


def _freeze_round_recipe(
    root: Path,
    round_id: str,
    campaign: dict[str, Any],
    incumbent: dict[str, Any],
) -> dict[str, str]:
    """Write the round recipe: registered settings plus a reference.

    The selection reference freezes the incumbent results and the
    research gates, so the trainer's checkpoint ranking enforces
    the same joint constraints promotion applies. Deterministic
    from frozen inputs, so a retry after a crash rewrites the same
    bytes before the round receipt pins their digest.
    """
    recipe_ref = campaign.get("recipe") or {}
    if not isinstance(recipe_ref, dict):
        raise ValueError("Campaign recipe block is malformed")
    try:
        spec = read_json(Path(recipe_ref.get("path", "")))
    except (OSError, ValueError) as exc:
        raise ValueError("Bound recipe is unreadable") from exc
    if not isinstance(spec, dict):
        raise ValueError("Bound recipe must be a JSON object")
    selection = campaign.get("selection")
    try:
        _check_policy(selection)
    except ValueError as exc:
        raise ValueError("Campaign selection policy is malformed") from exc
    if not isinstance(selection, dict):
        raise ValueError("Campaign selection policy is malformed")
    results = incumbent.get("results")
    if not isinstance(results, dict):
        raise ValueError("Incumbent results are missing")
    reference = laya_training.build_selection_reference_from_results(
        results, selection.get("research"), digest(results)
    )
    path = _round_recipe_file(Path(root), round_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, {**spec, "selection_reference": reference})
    return {"path": str(path), "sha256": _sha(path)}


def prepare_round(
    root: Path,
    round_id: str,
    dataset_version: str,
    hypothesis: str,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Open a round against a registered dataset version.

    Revalidates the parent export, recipe, reporting contract, and
    dataset manifest bytes, then binds the round to all of them plus
    the incumbent under comparison and the selection policy. The
    frozen values persist in the round receipt, so later validation
    and selection compare against this spec even if the campaign
    moves on. Also freezes a round recipe — the registered settings
    plus a selection reference built from the frozen incumbent
    results and research gates — so the trainer ranks checkpoints
    under the same joint constraints promotion enforces. The job
    profile must point at that round recipe, not the registered
    recipe. Requires a non-empty recorded hypothesis. Repeating an
    identical spec returns the existing round; a differing spec under
    the same id is refused.
    """
    if not isinstance(round_id, str) or not ROUND_ID.fullmatch(round_id):
        raise ValueError("Round ID must be a simple stable identifier")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("A round needs a recorded hypothesis")
    campaign = load_campaign(root)
    experiment_id = "%s-%s" % (campaign["campaign_id"], round_id)
    if len(experiment_id) > 64 or not ROUND_ID.fullmatch(experiment_id):
        raise ValueError("Campaign and round IDs exceed operator limits")
    versions = _record(_datasets_file(root)).get("versions")
    if not isinstance(versions, dict):
        raise ValueError("Campaign dataset registry is malformed")
    datasets = versions
    if not isinstance(dataset_version, str) or dataset_version not in datasets:
        raise ValueError("Unknown dataset version: " + str(dataset_version))
    _revalidate_inputs(root, campaign, dataset_version)
    incumbent = load_incumbent(root)
    if not isinstance(overrides, dict) and overrides is not None:
        raise ValueError("Round overrides must be an object")
    overrides = overrides or {}
    for key in ("parent", "recipe", "selection", "campaign_id"):
        if key not in campaign:
            raise ValueError("Campaign is missing " + key)
    for key in ("export", "export_manifest", "results"):
        if key not in incumbent:
            raise ValueError("Incumbent is missing " + key)
    bindings = {
        "dataset": datasets[dataset_version],
        "parent": campaign["parent"],
        "incumbent": {
            "export": incumbent["export"],
            "export_manifest": incumbent["export_manifest"],
            "results": incumbent["results"],
        },
        "recipe": campaign["recipe"],
        "policy": digest(campaign["selection"]),
        "policy_content": campaign["selection"],
        "reporting_contract": campaign.get("reporting_contract"),
    }
    bindings["round_recipe"] = _freeze_round_recipe(
        root, round_id, campaign, incumbent
    )
    key = digest(
        {
            "campaign": campaign["campaign_id"],
            "round": round_id,
            **bindings,
            "hypothesis": hypothesis,
            "overrides": overrides,
        }
    )
    path = _round_file(root, round_id)
    if path.exists():
        existing = _record(path)
        if existing.get("idempotency_key") != key:
            raise ValueError(
                "Round ID already belongs to a different spec: " + round_id
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema": CAMPAIGN_ROUND_SCHEMA,
        "round_id": round_id,
        "idempotency_key": key,
        "phase": "prepare",
        "dataset_version": dataset_version,
        "hypothesis": hypothesis,
        "overrides": overrides,
        "bindings": bindings,
        "candidate": None,
        "decision": None,
        "history": [{"phase": "prepare", "time": time.time()}],
    }
    atomic_json(path, receipt)
    return receipt


def advance(
    root: Path, round_id: str, phase: str, **details: Any
) -> dict[str, Any]:
    """Persist one round transition along the ordered phase chain.

    ``failed`` is reachable from any active phase and records the
    failure; ``done`` is reachable only from ``report``.
    """
    if not isinstance(round_id, str) or not isinstance(phase, str):
        raise ValueError("Round transition needs string ids")
    path = _round_file(root, round_id)
    receipt = _record(path)
    current = receipt.get("phase")
    if current not in PHASES and current not in TERMINAL:
        raise ValueError("Round has an unknown phase: " + str(round_id))
    if current in TERMINAL:
        raise ValueError("Round is already terminal: " + round_id)
    if phase == "failed":
        receipt["phase"] = "failed"
    elif phase == "done":
        if current != "report":
            raise ValueError("Round reaches done only from report: " + round_id)
        receipt["phase"] = "done"
    else:
        expected = PHASES[PHASES.index(current) + 1]
        if phase != expected:
            raise ValueError(
                "Invalid round transition %s -> %s" % (current, phase)
            )
        receipt["phase"] = phase
    history = receipt.get("history")
    if not isinstance(history, list):
        raise ValueError("Round history is malformed: " + str(round_id))
    history.append({"phase": receipt["phase"], "time": time.time(), **details})
    atomic_json(path, receipt)
    return receipt


def record_candidate(
    root: Path,
    round_id: str,
    export: str,
    results: dict[str, Any],
    receipt: dict[str, str],
) -> dict[str, Any]:
    """Attach a validated candidate export and its evaluation.

    The export must exist with weights, the evaluation receipt file
    must verify, its contents must bind this export, these results,
    and this run, and the results must pass shape validation. Runs
    during ``observe`` and advances the round to ``evaluate``.
    """
    if not isinstance(export, str):
        raise ValueError("Candidate export must be a path")
    path = _round_file(root, round_id)
    round_receipt = _record(path)
    if round_phase(Path(root), round_id) != "observe":
        raise ValueError("Candidate attaches during observe: " + round_id)
    export_path = Path(export)
    if not export_path.is_dir():
        raise ValueError("Candidate export is missing: " + export)
    if not (export_path / "model.safetensors").is_file():
        raise ValueError("Candidate export lacks weights: " + export)
    _checked_results(results)
    bound_receipt = _checked_receipt(receipt)
    campaign = load_campaign(root)
    _checked_evaluation(
        bound_receipt["path"],
        results,
        str(export_path.resolve()),
        "%s-%s" % (campaign["campaign_id"], round_id),
        round_id,
    )
    round_receipt["candidate"] = {
        "export": str(export_path.resolve()),
        "export_manifest": export_manifest(export_path),
        "results": results,
        "receipt": bound_receipt,
    }
    atomic_json(path, round_receipt)
    return advance(root, round_id, "evaluate")


def _finish_decision(
    root: Path,
    round_id: str,
    intent: dict[str, Any],
    incumbent: dict[str, Any],
) -> dict[str, Any]:
    """Complete one intended decision; safe to retry after a crash.

    The intent outlives every write it governs: incumbent swap, then
    receipt, then the phase advance, with intent removal last. Any
    interruption retries the same intended verdict against the same
    previous incumbent. The candidate's bytes revalidate on every
    attempt, fresh or retried, so promotion never selects weights
    that changed after evaluation.
    """
    candidate = intent.get("candidate")
    verdict = intent.get("verdict")
    if not isinstance(candidate, dict) or not isinstance(verdict, dict):
        raise ValueError("Decision intent is unreadable")
    if not isinstance(incumbent, dict):
        raise ValueError("Incumbent record is malformed")
    _revalidate_candidate(root, round_id, candidate)
    if verdict.get("promote_research"):
        if incumbent.get("export") != candidate.get("export"):
            for key in (
                "export",
                "export_manifest",
                "results",
                "receipt",
            ):
                if key not in candidate:
                    raise ValueError("Decision intent is unreadable")
            atomic_json(
                _incumbent_file(root),
                {
                    "export": candidate["export"],
                    "export_manifest": candidate["export_manifest"],
                    "results": candidate["results"],
                    "receipt": candidate["receipt"],
                    "round_id": round_id,
                    "promoted": time.time(),
                },
            )
    path = _round_file(root, round_id)
    receipt = _record(path)
    receipt["decision"] = verdict
    if "previous_incumbent" not in intent:
        raise ValueError("Decision intent is unreadable")
    receipt["previous_incumbent"] = intent["previous_incumbent"]
    atomic_json(path, receipt)
    if receipt.get("phase") != "decide":
        advance(root, round_id, "decide")
    _intent_file(root, round_id).unlink(missing_ok=True)
    return _record(path)


def _frozen_incumbent(receipt: dict[str, Any]) -> dict[str, Any]:
    """Return the incumbent the round was specified against."""
    bindings = receipt.get("bindings") or {}
    frozen = bindings.get("incumbent")
    if not isinstance(frozen, dict) or "results" not in frozen:
        raise ValueError("Round lacks frozen incumbent bindings")
    return frozen


def _frozen_policy(root: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    """Return the selection policy the round was specified against.

    The stored content must match its own digest, and the live
    campaign policy must still match it: selection always runs
    under the frozen policy, and a policy change requires a newly
    specified round.
    """
    if not isinstance(receipt, dict):
        raise ValueError("Round receipt must be an object")
    bindings = receipt.get("bindings") or {}
    if not isinstance(bindings, dict):
        raise ValueError("Round policy binding is damaged")
    frozen = bindings.get("policy_content")
    if not isinstance(frozen, dict) or digest(frozen) != bindings.get("policy"):
        raise ValueError("Round policy binding is damaged")
    live = load_campaign(root).get("selection")
    if not isinstance(live, dict) or digest(live) != bindings.get("policy"):
        raise ValueError(
            "Campaign policy changed; re-specify round "
            + str(receipt.get("round_id"))
        )
    return frozen


def _require_unmoved_incumbent(
    current: dict[str, Any], frozen: dict[str, Any], round_id: str
) -> None:
    """Refuse a decision when another round promoted meanwhile."""
    for record in (current, frozen):
        if not isinstance(record, dict) or not isinstance(
            record.get("export"), str
        ):
            raise ValueError("Incumbent record is malformed")
    try:
        moved = current["export"] != frozen["export"] or digest(
            current["results"]
        ) != digest(frozen["results"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Incumbent record is malformed") from exc
    if moved:
        raise ValueError(
            "Incumbent moved during round; re-specify: " + str(round_id)
        )


def _require_retry_consistency(
    current: dict[str, Any], intent: dict[str, Any], round_id: str
) -> None:
    """Refuse a retried decision the intent no longer describes.

    A crash may land before or after the incumbent swap, so the
    live incumbent must be either the previous holder or the
    intended candidate; anything else means another round decided
    during the crash window.
    """
    for record in (
        current,
        intent.get("previous_incumbent"),
        intent.get("candidate"),
    ):
        if not isinstance(record, dict) or not isinstance(
            record.get("export"), str
        ):
            raise ValueError("Decision intent is unreadable")
    try:
        live = (current["export"], digest(current["results"]))
        allowed = {
            (
                intent["previous_incumbent"]["export"],
                digest(intent["previous_incumbent"]["results"]),
            ),
            (
                intent["candidate"]["export"],
                digest(intent["candidate"]["results"]),
            ),
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("Decision intent is unreadable") from exc
    if live not in allowed:
        raise ValueError(
            "Incumbent moved during round; re-specify: " + str(round_id)
        )


def decide(
    root: Path,
    round_id: str,
    reference_results: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Apply the frozen policy; promote or retain the incumbent.

    The candidate compares against the round's frozen incumbent
    under the round's frozen policy, not whoever holds the title
    or whatever the campaign file says at decision time; a round
    whose incumbent or policy moved meanwhile must be
    re-specified. Promotion is a recoverable transaction under the
    campaign lock: the intended verdict and previous incumbent
    persist before the incumbent changes, so an interruption
    retries the same decision instead of comparing the candidate
    against itself. ``reference_results`` adds a record-only
    comparison against a permanent baseline to the verdict.
    """
    root = Path(root)
    with _campaign_lock(root):
        intent_path = _intent_file(root, round_id)
        incumbent = load_incumbent(root)
        receipt = _record(_round_file(root, round_id))
        policy = _frozen_policy(root, receipt)
        if intent_path.exists():
            intent = _record(intent_path)
            _require_retry_consistency(incumbent, intent, round_id)
            return _finish_decision(root, round_id, intent, incumbent)
        if receipt.get("phase") != "evaluate":
            raise ValueError("Decision follows evaluation: " + str(round_id))
        candidate = receipt.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError("No candidate recorded: " + str(round_id))
        frozen = _frozen_incumbent(receipt)
        _require_unmoved_incumbent(incumbent, frozen, round_id)
        verdict = apply_selection_policy(
            policy,
            frozen["results"],
            candidate.get("results"),
            reference_results,
        )
        intent = {
            "verdict": verdict,
            "candidate": receipt["candidate"],
            "previous_incumbent": incumbent,
            "time": time.time(),
        }
        atomic_json(intent_path, intent)
        return _finish_decision(root, round_id, intent, incumbent)


def close_round(root: Path, round_id: str, summary: str) -> dict[str, Any]:
    """File the round report and mark the round done.

    Resumable: an interruption between the report write and the final
    advance retries into ``done`` instead of refusing.
    """
    path = _round_file(root, round_id)
    receipt = _record(path)
    if receipt.get("phase") == "report" and "report" in receipt:
        return advance(root, round_id, "done")
    if receipt.get("phase") != "decide":
        raise ValueError("Report follows decision: " + str(round_id))
    receipt["report"] = {"summary": summary, "time": time.time()}
    atomic_json(path, receipt)
    advance(root, round_id, "report")
    return advance(root, round_id, "done")


def _systemd_active(entries: list[dict[str, Any]]) -> Optional[bool]:
    """Read the latest owned-job state in the real Operator shape.

    Operator history entries carry ``systemd: {"state": ...}`` with the
    raw ``systemctl show`` fields nested under ``Detail``; "running" is
    the only non-terminal owned state.
    """
    if not isinstance(entries, list):
        return None
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        state = entry.get("systemd")
        if isinstance(state, dict) and "state" in state:
            return bool(state["state"] == "running")
    return None


def run_round(
    root: Path,
    round_id: str,
    *,
    operator_factory: Callable[[str], Any],
    evaluator: Callable[[Path, str, dict[str, Any]], dict[str, Any]],
    max_steps: int = 64,
    reference_results: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Drive one prepared round toward a terminal phase and return it.

    ``operator_factory`` builds the bounded operator from a profile
    path; ``evaluator`` consumes the finished owned job and returns
    ``{"export", "results", "receipt"}`` for the candidate. The runner
    is re-entrant: a round interrupted at any phase resumes from its
    persisted receipt, and an already-terminal round returns at once.
    ``reference_results`` adds a record-only permanent-baseline
    comparison to the decision verdict.
    """
    root = Path(root)
    campaign = load_campaign(root)
    profile = campaign.get("operator_profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("Campaign has no operator profile")
    campaign_id = campaign.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("Campaign has no campaign id")
    if (
        not isinstance(max_steps, int)
        or isinstance(max_steps, bool)
        or max_steps < 1
    ):
        raise ValueError("Round step budget must be a positive integer")
    experiment_id = "%s-%s" % (campaign_id, round_id)

    drivers = _RoundDrivers(
        root,
        round_id,
        experiment_id,
        operator_factory,
        evaluator,
        reference_results,
    )
    handlers = {
        "prepare": _prepare_step,
        "validate_data": _validate_step,
        "preflight": _preflight_step,
        "acquire": _acquire_step,
        "submit": _submit_step,
        "observe": _observe_step,
        "evaluate": _evaluate_step,
        "decide": _decide_step,
        "report": _report_step,
    }
    for _ in range(max_steps):
        receipt = _record(_round_file(root, round_id))
        phase = receipt.get("phase")
        if not isinstance(phase, str):
            raise ValueError("Round has an unknown phase: " + str(round_id))
        if phase in TERMINAL:
            return receipt
        handler = handlers.get(phase)
        if handler is None:
            raise ValueError("Unknown round phase: " + phase)
        settled = handler(drivers)
        if settled is not None:
            return settled
    raise ValueError("Round did not settle within its step budget")


@dataclasses.dataclass(frozen=True)
class _RoundDrivers:
    """Operator access and evaluation for one driven round."""

    root: Path
    round_id: str
    experiment_id: str
    operator_factory: Callable[[str], Any]
    evaluator: Callable[[Path, str, dict[str, Any]], dict[str, Any]]
    reference_results: Optional[dict[str, Any]] = None


def _round_receipt(drivers: _RoundDrivers) -> dict[str, Any]:
    """Read the current round receipt."""
    return _record(_round_file(drivers.root, drivers.round_id))


def _driver_operator(drivers: _RoundDrivers) -> Any:
    """Build the bounded operator from the campaign profile."""
    campaign = load_campaign(drivers.root)
    return drivers.operator_factory(campaign.get("operator_profile", ""))


def _prepare_step(drivers: _RoundDrivers) -> None:
    """Verify the registered dataset version and its acceptance."""
    receipt = _round_receipt(drivers)
    versions = _record(_datasets_file(drivers.root)).get("versions")
    if not isinstance(versions, dict):
        raise ValueError("Campaign dataset registry is malformed")
    version = receipt.get("dataset_version")
    if not isinstance(version, str):
        raise ValueError("Round dataset version is malformed")
    entry = versions.get(version)
    if not isinstance(entry, dict) or not isinstance(
        entry.get("manifest"), str
    ):
        raise ValueError("Round dataset version is malformed")
    manifest = Path(entry["manifest"])
    laya_training.verify_dataset_version(manifest.parents[2], version)
    if not laya_training.dataset_version_accepted(manifest.parents[2], version):
        raise ValueError("Dataset version not accepted for use")
    advance(drivers.root, drivers.round_id, "validate_data")


def _bound_file(profile: dict[str, Any], key: str, what: str) -> Path:
    """Resolve a profile path setting, refusing a missing one."""
    if not isinstance(profile, dict):
        raise ValueError("Binding check needs a receipt and profile")
    raw = profile.get(key)
    if not isinstance(raw, str) or not raw or not Path(raw).is_file():
        raise ValueError("Job profile lacks a bound " + what)
    return Path(raw)


def _require_profile_bindings(
    receipt: dict[str, Any], profile: dict[str, Any]
) -> None:
    """Compare the job profile against the frozen round spec.

    The operator profile is the campaign's launch configuration,
    but it is edited by hand; a stale model path, swapped dataset,
    missing recipe, unreviewed pool file, or retuned training
    setting must fail the round before submission, not train
    silently on the wrong inputs. The executed training settings —
    profile defaults plus round overrides — must equal the bound
    recipe's training section exactly; any retune needs a new
    recipe, not an edit. The bound recipe is the round recipe,
    which carries the frozen selection reference the trainer
    ranks under; pointing at the registered recipe instead fails.
    """
    if not isinstance(receipt, dict) or not isinstance(profile, dict):
        raise ValueError("Binding check needs a receipt and profile")
    bindings = receipt.get("bindings") or {}
    if not isinstance(bindings, dict):
        raise ValueError("Round lacks frozen incumbent bindings")
    frozen = bindings.get("incumbent")
    if not isinstance(frozen, dict):
        raise ValueError("Round lacks frozen incumbent bindings")
    if not isinstance(profile.get("model"), str):
        raise ValueError("Job profile needs model")
    model = Path(profile["model"]).resolve()
    if str(model) != frozen.get("export"):
        raise ValueError(
            "Job model is not the round's bound incumbent: "
            + str(profile.get("model"))
        )
    if export_manifest(model) != frozen.get("export_manifest"):
        raise ValueError("Bound incumbent export changed under the job")
    dataset = bindings.get("dataset") or {}
    if not isinstance(dataset, dict):
        raise ValueError("Round lacks frozen dataset bindings")
    if not isinstance(dataset.get("manifest"), str):
        raise ValueError("Round lacks frozen dataset bindings")
    manifest_path = Path(dataset["manifest"])
    try:
        current = _sha(manifest_path)
    except OSError as exc:
        raise ValueError("Bound dataset manifest is unreadable") from exc
    if current != dataset.get("sha256"):
        raise ValueError("Dataset manifest changed since registration")
    manifest_raw = read_json(manifest_path)
    if not isinstance(manifest_raw, dict):
        raise ValueError("Bound dataset manifest is unreadable")
    manifest = cast(dict[str, Any], manifest_raw)
    defaults = profile.get("defaults", {})
    overrides = receipt.get("overrides") or {}
    if not isinstance(defaults, dict) or not isinstance(overrides, dict):
        raise ValueError("Binding check needs mapping settings")
    values: dict[str, Any] = {**defaults, **overrides}
    roles = ("train",) if values.get("limit_train_rows") else ("train", "dev")
    splits = manifest.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError("Bound dataset manifest is unreadable")
    if not isinstance(profile.get("data"), str):
        raise ValueError("Job profile needs data")
    for role in roles:
        entry = splits.get(role)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ValueError("Bound dataset manifest is unreadable")
        actual = Path(profile["data"]) / (role + ".jsonl")
        if not actual.is_file() or _sha(actual) != entry.get("sha256"):
            raise ValueError(
                "Job data does not match the bound %s split" % role
            )
    recipe = bindings.get("round_recipe") or bindings.get("recipe") or {}
    if not isinstance(recipe, dict):
        raise ValueError("Round lacks frozen recipe bindings")
    if _sha(_bound_file(profile, "recipe", "recipe")) != recipe.get("sha256"):
        raise ValueError("Job recipe is not the round's bound recipe")
    contract = bindings.get("reporting_contract")
    if contract:
        if not isinstance(contract, dict):
            raise ValueError("Round lacks frozen contract bindings")
        path = _bound_file(profile, "reporting_contract", "contract")
        if _sha(path) != contract.get("sha256"):
            raise ValueError("Job contract is not the bound contract")
    pools = manifest.get("pools")
    job_pools = values.get("pools")
    if pools:
        if not isinstance(pools, dict):
            raise ValueError("Bound dataset manifest is unreadable")
        if (
            not isinstance(job_pools, str)
            or not job_pools
            or not Path(job_pools).is_file()
        ):
            raise ValueError("Job lacks the bound sampling pools")
        if _sha(Path(job_pools)) != pools.get("sha256"):
            raise ValueError("Job pools are not the bound pools")
    elif job_pools:
        raise ValueError("Job sets pools the round did not bind")
    recipe_path = Path(recipe.get("path", ""))
    try:
        spec = read_json(recipe_path)
    except (OSError, ValueError) as exc:
        raise ValueError("Bound recipe is unreadable") from exc
    expected = spec.get("training") if isinstance(spec, dict) else None
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Bound recipe names no training settings")
    for key, value in expected.items():
        if values.get(key) != value:
            raise ValueError(
                "Job setting differs from the bound recipe: " + key
            )
    for key in values:
        if (
            key in FLAGS or key in BOOL_FLAGS or key in CHOICE_FLAGS
        ) and key not in expected:
            raise ValueError(
                "Job setting is missing from the bound recipe: " + key
            )


def _validate_step(drivers: _RoundDrivers) -> None:
    """Prepare, preflight, and bind-check the owned training job."""
    operator = _driver_operator(drivers)
    receipt = _round_receipt(drivers)
    job = operator.prepare(drivers.experiment_id, receipt.get("overrides"))
    operator.preflight(job)
    campaign = load_campaign(drivers.root)
    version = receipt.get("dataset_version")
    if not isinstance(version, str):
        raise ValueError("Round dataset version is malformed")
    _revalidate_inputs(drivers.root, campaign, version)
    _frozen_policy(drivers.root, receipt)
    _require_profile_bindings(receipt, operator.profile)
    advance(drivers.root, drivers.round_id, "preflight")


def _preflight_step(drivers: _RoundDrivers) -> Optional[dict[str, Any]]:
    """Wait for free GPUs, then take the acquisition step."""
    operator = _driver_operator(drivers)
    if not operator.jobs.gpu_available():
        return _round_receipt(drivers)
    advance(drivers.root, drivers.round_id, "acquire")
    return None


def _acquire_step(drivers: _RoundDrivers) -> None:
    """Submit the owned job under the operator's reservation."""
    operator = _driver_operator(drivers)
    receipt = _round_receipt(drivers)
    job = operator.prepare(drivers.experiment_id, receipt.get("overrides"))
    _frozen_policy(drivers.root, receipt)
    operator.submit(job)
    advance(
        drivers.root,
        drivers.round_id,
        "submit",
        experiment_id=drivers.experiment_id,
    )


def _submit_step(drivers: _RoundDrivers) -> None:
    """Move a submitted round into observation."""
    advance(drivers.root, drivers.round_id, "observe")


def _observe_step(drivers: _RoundDrivers) -> Optional[dict[str, Any]]:
    """Poll one owned job; record its candidate once it completes."""
    operator = _driver_operator(drivers)
    manifest = operator.prepare(
        drivers.experiment_id, _round_receipt(drivers).get("overrides")
    )
    if manifest.get("phase") == "report":
        # A prior attempt reported this job, then crashed before the
        # campaign recorded its candidate. The operator revalidates
        # the completion evidence from live files on every report,
        # so resume there instead of re-observing a reported job.
        reported = operator.report(manifest)
    else:
        observed = operator.observe(manifest)
        if _systemd_active(observed.get("history", [])):
            return _round_receipt(drivers)
        finalized = operator.finalize(observed)
        entries = finalized.get("history", [])
        done = bool(entries) and bool(entries[-1].get("finalized", False))
        if not done:
            reason = (
                "job still running" if _systemd_active(entries) else "no report"
            )
            return advance(
                drivers.root, drivers.round_id, "failed", reason=reason
            )
        reported = operator.report(finalized)
    if reported.get("terminal") != "complete":
        return advance(
            drivers.root, drivers.round_id, "failed", reason="job incomplete"
        )
    outcome = drivers.evaluator(drivers.root, drivers.round_id, reported)
    if not isinstance(outcome, dict):
        raise ValueError("Evaluator returned no candidate")
    for key in ("export", "results", "receipt"):
        if key not in outcome:
            raise ValueError("Evaluator omitted candidate " + key)
    record_candidate(
        drivers.root,
        drivers.round_id,
        outcome["export"],
        outcome["results"],
        outcome["receipt"],
    )
    return None


def _evaluate_step(drivers: _RoundDrivers) -> Optional[dict[str, Any]]:
    """Decide, or fail the round on incomparable results."""
    try:
        decide(drivers.root, drivers.round_id, drivers.reference_results)
    except ValueError as exc:
        return advance(
            drivers.root, drivers.round_id, "failed", reason=str(exc)
        )
    return None


def _decide_step(drivers: _RoundDrivers) -> dict[str, Any]:
    """File the round report from its recorded decision."""
    verdict = _round_receipt(drivers)["decision"]
    summary = (
        "promoted" if verdict["promote_research"] else "retained incumbent"
    )
    return close_round(drivers.root, drivers.round_id, summary)


def _report_step(drivers: _RoundDrivers) -> dict[str, Any]:
    """Finish a round interrupted after its report write."""
    summary = _round_receipt(drivers)["report"]["summary"]
    return close_round(drivers.root, drivers.round_id, summary)
