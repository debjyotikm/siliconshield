"""Select on joint campaign constraints with a transfer tiebreak.

Without a selection reference the ranking keeps its historical shape:
a retention-accuracy regression flag, then the transfer key (whose
final element is transfer NLL). With a reference frozen from the
incumbent's development report plus the registered research gates,
retention floors, false-safe budgets, no-regression gates, and
population-scoped family caps all precede the transfer tiebreak, so
a lower transfer loss can never outrank a joint violation the
campaign would refuse.
"""

import math

from siliconshield.training.laya.checkpoint_selection import (
    objective_selection,
    population_indices,
)

NEW_COVERAGE_ROLE = "new_coverage"
RETENTION_ROLE = "retention"
REFERENCE_SCHEMA = "siliconshield-selection-reference-v1"
RANKING_V1_TRANSFER = "v1-transfer"
RANKING_V2_RETENTION = "v2-retention"
RANKINGS = (RANKING_V1_TRANSFER, RANKING_V2_RETENTION)


def attach_views(result, rows, logits, config, summarize):
    if not isinstance(config, dict):
        raise ValueError("Development views need a config object")
    definitions = config.get("views")
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError("Development views need named populations")
    memberships = {
        name: population_indices(rows, definition)
        for name, definition in definitions.items()
    }
    flat = [i for indices in memberships.values() for i in indices]
    if sorted(flat) != list(range(len(rows))):
        raise ValueError(
            "Development views must partition the full development population exactly"
        )
    views = {}
    for name, indices in memberships.items():
        if not indices:
            raise ValueError("Empty development view: " + str(name))
        views[name] = summarize(
            [rows[i] for i in indices], [logits[i] for i in indices]
        )
    selected = config.get("selection_view")
    if not isinstance(selected, str) or selected not in views:
        raise ValueError("Missing selection view")
    if not config.get("single_view_study"):
        if config.get("retention_view") not in views:
            raise ValueError("Missing transfer or retention view")
    result["development_views"] = views
    result["selection_view"] = selected
    return result


def config_from_views_manifest(manifest: dict) -> dict:
    """Build a trainer development-views config from a review manifest.

    The manifest (schema siliconshield-dev-views-v3) lists reviewed
    templates with triplet members; quarantined rows are already
    excluded by the emitter. The config selects exact row ids for
    one selection view and marks the study single-view, since a
    fresh model has no retention population to preserve.
    """
    if manifest.get("schema") != "siliconshield-dev-views-v3":
        raise ValueError("Unsupported development views manifest")
    row_ids = []
    templates = manifest.get("views", [])
    if not isinstance(templates, list):
        raise ValueError("Selection views manifest needs a views list")
    for template in templates:
        if not isinstance(template, dict):
            raise ValueError("Selection views template is not an object")
        if template.get("review") != "agree":
            raise ValueError(
                "Unreviewed template in selection views: "
                + str(template.get("record_id"))
            )
        triplets = template.get("triplets", [])
        if not isinstance(triplets, list):
            raise ValueError(
                "Selection views triplets need a list: "
                + str(template.get("record_id"))
            )
        for triplet in triplets:
            if not isinstance(triplet, dict):
                raise ValueError("Selection views triplet is not an object")
            try:
                row_ids.extend(
                    [
                        triplet["satisfies"],
                        triplet["violates"],
                        triplet["insufficient"],
                    ]
                )
            except KeyError as exc:
                raise ValueError(
                    "Selection views triplet lacks %s" % exc
                ) from exc
    if not all(isinstance(name, str) and name for name in row_ids):
        raise ValueError("Selection views row ids must be strings")
    if not row_ids:
        raise ValueError("Selection views manifest names no rows")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("Selection views manifest repeats a row")
    if len(row_ids) != manifest.get("n_rows"):
        raise ValueError("Selection views row count disagrees")
    return {
        "views": {"fresh_dev": {"row_ids": sorted(row_ids)}},
        "selection_view": "fresh_dev",
        "single_view_study": True,
        "score_triplets": True,
    }


def normalize_family_caps(caps):
    """Normalize flat or nested family caps to population sections.

    Flat {family: cap} constrains the new-coverage population (the
    historical shape). Nested {population: {family: cap}} names each
    population explicitly; unknown populations or mixed shapes
    refuse. Limits are nonnegative integers.
    """
    if not isinstance(caps, dict):
        raise ValueError("family caps must be an object")
    sections = {NEW_COVERAGE_ROLE: {}, RETENTION_ROLE: {}}
    if any(isinstance(value, dict) for value in caps.values()):
        if not set(caps) <= set(sections) or not all(
            isinstance(value, dict) for value in caps.values()
        ):
            raise ValueError("family caps mix populations and limits")
        items = caps.items()
    else:
        items = [(NEW_COVERAGE_ROLE, caps)]
    for role, table in items:
        for name, cap in table.items():
            if (
                not isinstance(name, str)
                or not isinstance(cap, int)
                or isinstance(cap, bool)
                or cap < 0
            ):
                raise ValueError("family caps entry is invalid: " + repr(name))
            sections[role][name] = cap
    return sections


def new_coverage_gain_ok(new_accuracy, old_accuracy, min_gain_pp):
    """True when the new-coverage gain meets the required improvement.

    Shared by the trainer's checkpoint feasibility and the
    campaign's promotion gate so both enforce the same threshold
    with the same arithmetic.
    """
    return (new_accuracy - old_accuracy) * 100 >= min_gain_pp


def _confusion_counts(matrix, what):
    """Return a validated 3x3 count matrix or raise ValueError."""
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
    ):
        raise ValueError(what + " needs a 3x3 confusion matrix")
    for row in matrix:
        for cell in row:
            if not isinstance(cell, int) or isinstance(cell, bool) or cell < 0:
                raise ValueError(what + " holds invalid confusion counts")
    return matrix


def mc_recall_from_confusion(matrix):
    """Missing-context recall from a 3x3 confusion matrix.

    Class 2 is insufficient_context; recall is its diagonal over
    its support. An unsupported class recalls vacuously.
    """
    matrix = _confusion_counts(matrix, "MC recall")
    support = sum(matrix[2])
    if support == 0:
        return 1.0
    return matrix[2][2] / support


def _finite_rate(block, key):
    value = block.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("Reference rate out of range: " + key)
    return float(value)


def _count(block, key):
    value = block.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Reference count is invalid: " + key)
    return value


def _reference_view(block):
    """Freeze one raw trainer view block for selection comparison."""
    matrix = _confusion_counts(block.get("confusion"), "Reference view")
    families = block.get("per_family")
    if not isinstance(families, dict) or not families:
        raise ValueError("Reference view needs a per-family breakdown")
    table = {}
    for name, entry in families.items():
        if not isinstance(entry, dict):
            raise ValueError("Reference family entry is invalid: " + str(name))
        table[name] = {
            "false_safe_count": _count(entry, "false_safe_count"),
            "violation_count": _count(entry, "violation_count"),
        }
    supports = [sum(row) for row in matrix]
    return {
        "accuracy": _finite_rate(block, "accuracy"),
        "false_safe_count": _count(block, "false_safe_count"),
        "violation_count": _count(block, "violation_count"),
        "min_class_recall": _finite_rate(block, "min_class_recall"),
        "both_pair_members_correct": _finite_rate(
            block, "both_pair_members_correct"
        ),
        "mc_correct": matrix[2][2],
        "mc_total": supports[2],
        "per_family": table,
    }


def _reference_policy(policy):
    """Structural check of the research gates selection consumes."""
    if not isinstance(policy, dict):
        raise ValueError("Reference policy must be an object")
    floor = policy.get("min_retention_accuracy")
    if (
        not isinstance(floor, (int, float))
        or isinstance(floor, bool)
        or not math.isfinite(floor)
        or not 0 <= floor <= 1
    ):
        raise ValueError("Reference policy needs a retention floor")
    budget = policy.get("max_retention_false_safe_increase")
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise ValueError("Reference policy needs a retention budget")
    ranking = policy.get("selection_ranking", RANKING_V1_TRANSFER)
    if ranking not in RANKINGS:
        raise ValueError("Reference policy names an unknown ranking")
    gain = policy.get("min_new_coverage_gain_pp")
    if (
        not isinstance(gain, (int, float))
        or isinstance(gain, bool)
        or not math.isfinite(gain)
    ):
        raise ValueError("Reference policy needs a gain threshold")
    for key in (
        "require_no_missing_context_regression",
        "require_no_pair_regression",
    ):
        if not isinstance(policy.get(key), bool):
            raise ValueError("Reference policy needs " + key)
    coverage = policy.get("max_new_coverage_false_safe_increase")
    if coverage is not None and (
        not isinstance(coverage, int)
        or isinstance(coverage, bool)
        or coverage < 0
    ):
        raise ValueError("Reference policy coverage budget is invalid")
    normalize_family_caps(policy.get("family_false_safe_caps", {}))


def check_selection_reference(reference):
    """Validate a frozen selection reference structurally."""
    if not isinstance(reference, dict):
        raise ValueError("Selection reference must be an object")
    if reference.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("Selection reference names another schema")
    if not isinstance(reference.get("report_digest"), str):
        raise ValueError("Selection reference needs a report digest")
    for role in ("transfer", "retention"):
        view = reference.get(role)
        if not isinstance(view, dict):
            raise ValueError("Selection reference needs " + role)
        for key in (
            "accuracy",
            "min_class_recall",
            "both_pair_members_correct",
        ):
            _finite_rate(view, key)
        for key in (
            "false_safe_count",
            "violation_count",
            "mc_correct",
            "mc_total",
        ):
            _count(view, key)
        families = view.get("per_family")
        if not isinstance(families, dict):
            raise ValueError("Selection reference needs families")
        for name, entry in families.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    "Selection reference family is invalid: " + str(name)
                )
            _count(entry, "false_safe_count")
            _count(entry, "violation_count")
    _reference_policy(reference.get("policy"))


def build_selection_reference(
    transfer_block, retention_block, policy, report_digest
):
    """Freeze incumbent raw views plus research gates for selection."""
    reference = {
        "schema": REFERENCE_SCHEMA,
        "report_digest": report_digest,
        "transfer": _reference_view(transfer_block),
        "retention": _reference_view(retention_block),
        "policy": policy,
    }
    check_selection_reference(reference)
    return reference


def _is_count(value):
    """Nonnegative integer that is not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _reference_count(counts, key):
    value = counts.get(key)
    if not _is_count(value):
        raise ValueError("Reference results count is invalid: " + key)
    return value


def _reference_view_from_counts(view, role):
    """Freeze one adapted selection view for selection comparison.

    Reads the same ``counts``/``families`` shape the campaign
    stores, recomputing every rate from integers. The family table
    documents the incumbent breakdown when known; an empty table
    records an unknown breakdown, and no gate reads it — family
    caps check live candidate views against the policy instead.
    """
    if not isinstance(view, dict):
        raise ValueError("Reference results need view: " + role)
    counts = view.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Reference results need counts: " + role)
    total = _reference_count(counts, "n")
    if total <= 0:
        raise ValueError("Reference results need a total: " + role)
    correct = _reference_count(counts, "correct")
    violations = _reference_count(counts, "violations")
    false_safe = _reference_count(counts, "false_safe")
    if not 0 <= correct <= total:
        raise ValueError("Reference results correct outside total")
    if not 0 <= violations <= total:
        raise ValueError("Reference results violations outside total")
    if not 0 <= false_safe <= violations:
        raise ValueError("Reference results false-safe outside violations")
    pairs_total = _reference_count(counts, "pairs_total")
    pairs_correct = _reference_count(counts, "pairs_correct")
    if pairs_total <= 0 or not 0 <= pairs_correct <= pairs_total:
        raise ValueError("Reference results pair counts are invalid")
    mc_total = _reference_count(counts, "mc_total")
    mc_correct = _reference_count(counts, "mc_correct")
    if mc_total <= 0 or not 0 <= mc_correct <= mc_total:
        raise ValueError("Reference results MC counts are invalid")
    classes = counts.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("Reference results need classes: " + role)
    recalls = []
    for entry in classes:
        if not isinstance(entry, dict):
            raise ValueError("Reference results class is invalid")
        class_correct = entry.get("correct")
        class_total = entry.get("total")
        if (
            not _is_count(class_correct)
            or not _is_count(class_total)
            or class_total <= 0
            or not 0 <= class_correct <= class_total
        ):
            raise ValueError("Reference results class is invalid")
        recalls.append(class_correct / class_total)
    families = view.get("families")
    if families is None:
        families = {}
    if not isinstance(families, dict):
        raise ValueError("Reference results families are invalid")
    table = {}
    for name, entry in families.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("Reference results family is invalid")
        family_fs = entry.get("false_safe")
        family_violations = entry.get("violations")
        if (
            not _is_count(family_fs)
            or not _is_count(family_violations)
            or family_violations <= 0
            or not 0 <= family_fs <= family_violations
        ):
            raise ValueError(
                "Reference results family is invalid: " + str(name)
            )
        table[name] = {
            "false_safe_count": family_fs,
            "violation_count": family_violations,
        }
    return {
        "accuracy": correct / total,
        "false_safe_count": false_safe,
        "violation_count": violations,
        "min_class_recall": min(recalls),
        "both_pair_members_correct": pairs_correct / pairs_total,
        "mc_correct": mc_correct,
        "mc_total": mc_total,
        "per_family": table,
    }


def build_selection_reference_from_results(results, policy, report_digest):
    """Freeze adapted incumbent results plus research gates.

    ``results`` is the stored campaign shape with ``new_coverage``
    and ``retention`` views; ``policy`` is the frozen research
    section. The transfer slot takes the new-coverage view.
    """
    if not isinstance(results, dict):
        raise ValueError("Reference results must be an object")
    reference = {
        "schema": REFERENCE_SCHEMA,
        "report_digest": report_digest,
        "transfer": _reference_view_from_counts(
            results.get(NEW_COVERAGE_ROLE), NEW_COVERAGE_ROLE
        ),
        "retention": _reference_view_from_counts(
            results.get(RETENTION_ROLE), RETENTION_ROLE
        ),
        "policy": policy,
    }
    check_selection_reference(reference)
    return reference


def _view_rate(view, key):
    """Finite rate from a live view, defaulting missing pairs to zero."""
    value = view.get(key)
    if value is None and key == "both_pair_members_correct":
        return 0.0
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError("Selection view rate is invalid: " + key)
    return float(value)


def _view_count(view, key):
    value = view.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Selection view count is invalid: " + key)
    return value


def _view_nll(view):
    """Finite nonnegative NLL from a live view."""
    value = view.get("nll")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Selection view NLL is invalid")
    return float(value)


def ranking_version(policy):
    """Resolve the ranking version; absent means v1-transfer."""
    version = policy.get("selection_ranking", RANKING_V1_TRANSFER)
    if version not in RANKINGS:
        raise ValueError("Reference policy names an unknown ranking")
    return version


def retention_rank_key(
    false_safe, correct, pairs_correct, retention_nll, transfer_nll
):
    """Post-feasibility ordering shared by trainer and campaign.

    Fewer retention false-safes first, then more correct retention
    decisions, more correct retention pairs, lower retention NLL,
    and transfer NLL last. Counts are integers, NLLs finite; the
    campaign maps unknown NLLs before calling.
    """
    return (
        false_safe,
        -correct,
        -pairs_correct,
        retention_nll,
        transfer_nll,
    )


def _view_scaled_count(view, rate_key, total_key):
    """Integer count behind a live rate on a validated total."""
    total = _view_count(view, total_key)
    count = int(round(_view_rate(view, rate_key) * total))
    if not 0 <= count <= total:
        raise ValueError("Selection view count is invalid: " + rate_key)
    return count


def _caps_violated(view, caps):
    """True when a capped family exceeds its limit or is missing."""
    families = view.get("per_family")
    if not isinstance(families, dict):
        return bool(caps)
    for name, cap in caps.items():
        entry = families.get(name)
        if not isinstance(entry, dict):
            return True
        false_safe = entry.get("false_safe_count")
        if (
            not isinstance(false_safe, int)
            or isinstance(false_safe, bool)
            or false_safe > cap
        ):
            return True
    return False


def joint_flags(result, config):
    """Violation flags against the frozen selection reference.

    Empty without a reference (legacy transfer ranking). With one,
    retention floors, false-safe budgets, no-regression gates, and
    population-scoped family caps all precede the transfer key, so
    a lower transfer loss can never outrank a joint violation.
    """
    reference = config.get("selection_reference")
    if reference is None:
        return ()
    check_selection_reference(reference)
    views = result["development_views"]
    selected = views[config["selection_view"]]
    retained = views[config["retention_view"]]
    policy = reference["policy"]
    transfer_ref = reference["transfer"]
    retention_ref = reference["retention"]
    flags = [
        int(
            _view_rate(retained, "accuracy") < policy["min_retention_accuracy"]
        ),
        int(
            _view_count(retained, "false_safe_count")
            > retention_ref["false_safe_count"]
            + policy["max_retention_false_safe_increase"]
        ),
    ]
    if policy["require_no_missing_context_regression"]:
        flags.append(
            int(
                mc_recall_from_confusion(retained.get("confusion"))
                < _ref_mc(retention_ref)
            )
        )
        flags.append(
            int(
                mc_recall_from_confusion(selected.get("confusion"))
                < _ref_mc(transfer_ref)
            )
        )
    if policy["require_no_pair_regression"]:
        flags.append(
            int(
                _view_rate(retained, "both_pair_members_correct")
                < retention_ref["both_pair_members_correct"]
            )
        )
        flags.append(
            int(
                _view_rate(selected, "both_pair_members_correct")
                < transfer_ref["both_pair_members_correct"]
            )
        )
    budget = policy.get("max_new_coverage_false_safe_increase")
    if budget is not None:
        flags.append(
            int(
                _view_count(selected, "false_safe_count")
                > transfer_ref["false_safe_count"] + budget
            )
        )
    caps = normalize_family_caps(policy.get("family_false_safe_caps", {}))
    flags.append(
        int(
            _caps_violated(selected, caps[NEW_COVERAGE_ROLE])
            or _caps_violated(retained, caps[RETENTION_ROLE])
        )
    )
    flags.append(
        int(
            not new_coverage_gain_ok(
                _view_rate(selected, "accuracy"),
                transfer_ref["accuracy"],
                policy["min_new_coverage_gain_pp"],
            )
        )
    )
    return tuple(flags)


def _ref_mc(view_ref):
    """Reference MC recall from frozen counts."""
    total = view_ref["mc_total"]
    if total <= 0:
        return 1.0
    return view_ref["mc_correct"] / total


def selection(result, config, reference, base_key):
    """Rank a checkpoint: feasibility first, ranking key last.

    v1-transfer keeps the historical transfer key after the joint
    flags. v2-retention orders checkpoints by retention quality
    first (false-safes, correct, pairs, NLL) with transfer NLL as
    the final tiebreak, matching the campaign's comparison.
    """
    if "checkpoint_objective" in config:
        return objective_selection(result, config["checkpoint_objective"])
    views = result["development_views"]
    retained = views[config["retention_view"]]["accuracy"]
    limit = config["max_retention_accuracy_drop"]
    if (
        not all(
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
            for v in (reference, retained, limit)
        )
        or not 0 <= limit <= 1
    ):
        raise ValueError("Invalid retention reference or tolerance")
    regression = int(retained < reference - limit)
    flags = joint_flags(result, config)
    stored = views[config["retention_view"]]
    selected = views[config["selection_view"]]
    frozen = config.get("selection_reference")
    if frozen is not None and (
        ranking_version(frozen["policy"]) == RANKING_V2_RETENTION
    ):
        return (
            regression,
            *flags,
            *retention_rank_key(
                _view_count(stored, "false_safe_count"),
                _view_scaled_count(stored, "accuracy", "n"),
                _view_scaled_count(
                    stored, "both_pair_members_correct", "complete_pairs"
                ),
                _view_nll(stored),
                _view_nll(selected),
            ),
        )
    return (regression, *flags, *base_key(selected))
