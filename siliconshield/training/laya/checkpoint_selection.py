"""Count-based protection and objective ranking across development populations."""

import math

from siliconshield.schemas import (
    CHECKPOINT_BALANCED_SCHEMA,
    CHECKPOINT_OBJECTIVE_SCHEMA,
    CHECKPOINT_SOURCED_SCHEMA,
    CHECKPOINT_TRIPLETS_SCHEMA,
)


def population_indices(rows: list[dict], definition: list | dict) -> list[int]:
    """Resolve source/group/row selectors without changing row provenance.

    List definitions retain the legacy source-name behavior. Object selectors
    combine their filters; exclusions let separate development views share a
    source without sharing rows. A row_ids selector names exact members so a
    reviewed manifest (with quarantine applied) maps onto the population
    without re-deriving provenance. The caller checks the complete partition.
    """
    if isinstance(definition, list):
        definition = {"source_names": definition}
    allowed = {
        "source_names",
        "split_groups",
        "exclude_split_groups",
        "row_ids",
    }
    if not isinstance(definition, dict) or not definition:
        raise ValueError("Development population needs a selector")
    if set(definition) - allowed:
        raise ValueError("Unknown development population selector")
    for values in definition.values():
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ValueError("Development selector values must be strings")
    wanted = set(definition["row_ids"]) if "row_ids" in definition else None
    indices = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Development row %d is not an object" % index)
        if "source_names" in definition:
            source = row.get("source")
            name = source.get("name") if isinstance(source, dict) else None
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Development row %d has no source name" % index
                )
            if name not in definition["source_names"]:
                continue
        if "split_groups" in definition and (
            row.get("split_group_id") not in definition["split_groups"]
        ):
            continue
        if row.get("split_group_id") in definition.get(
            "exclude_split_groups", []
        ):
            continue
        if wanted is not None and row.get("id") not in wanted:
            continue
        indices.append(index)
    return indices


def _count(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Expected nonnegative integer: " + str(name))
    return value


def triplet_counts(
    rows: list[dict], predictions: list[int], labels: list[str]
) -> dict[str, int]:
    """Score whole triplets grouped by original pair_id.

    Every row must carry a pair_id; every group must hold exactly
    three rows with one expected label each. A triplet is correct
    only when all three members are predicted correctly. Grouping
    is order-invariant: shuffled rows give identical triplets.
    """
    if len(rows) != len(predictions):
        raise ValueError("Triplet scoring needs aligned rows")
    if len(labels) != 3 or len(set(labels)) != 3:
        raise ValueError("Triplet scoring needs three distinct labels")
    groups: dict[str, list[int]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                "Triplet scoring needs objects, not row %d" % position
            )
        pair = row.get("pair_id")
        if not isinstance(pair, str) or not pair:
            raise ValueError("Triplet scoring needs pair_id on every row")
        groups.setdefault(pair, []).append(position)
    correct = 0
    for pair, members in groups.items():
        if len(members) != 3:
            raise ValueError("Triplet %s does not hold three rows" % pair)
        expected = []
        for i in members:
            block = rows[i].get("expected")
            label = block.get("security") if isinstance(block, dict) else None
            if not isinstance(label, str):
                raise ValueError(
                    "Triplet %s member %d has no expected label" % (pair, i)
                )
            expected.append(label)
        if set(expected) != set(labels):
            raise ValueError("Triplet %s lacks one row per label" % pair)
        if all(
            predictions[i] == labels.index(expected[k])
            for k, i in enumerate(members)
        ):
            correct += 1
    return {"triplets": len(groups), "triplets_correct": correct}


def view_counts(view: dict) -> dict:
    """Derive counts from confusion and validate accompanying summary metrics."""
    matrix = view.get("confusion")
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise ValueError(
            "Development view needs a three-class confusion matrix"
        )
    for row in matrix:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("Development confusion matrix must be 3 by 3")
        for cell in row:
            _count(cell, "confusion")
    support = [sum(row) for row in matrix]
    if not all(support):
        raise ValueError("Every development class must be represented")
    n = sum(support)
    correct = [matrix[i][i] for i in range(3)]
    if view.get("n") != n or view.get("false_safe_count") != matrix[1][0]:
        raise ValueError("Development counts disagree with confusion matrix")
    accuracy = view.get("accuracy")
    if (
        not isinstance(accuracy, (int, float))
        or isinstance(accuracy, bool)
        or not math.isclose(
            accuracy, sum(correct) / n, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError("Development accuracy disagrees with counts")
    total_pairs = _count(view.get("complete_pairs"), "complete_pairs")
    if total_pairs == 0:
        raise ValueError("Development view needs complete pairs")
    fraction = view.get("both_pair_members_correct")
    if (
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("Development view needs finite pair accuracy")
    pairs = round(fraction * total_pairs)
    if not math.isclose(fraction * total_pairs, pairs, abs_tol=1e-9):
        raise ValueError("Pair accuracy cannot represent an integer count")
    nll = view.get("nll")
    if (
        not isinstance(nll, (int, float))
        or isinstance(nll, bool)
        or not math.isfinite(nll)
        or nll < 0
    ):
        raise ValueError("Development NLL must be finite and nonnegative")
    triplets = view.get("triplets")
    triplets_correct = view.get("triplets_correct")
    if (triplets is None) != (triplets_correct is None):
        raise ValueError("Triplet scores need both triplets and correct")
    if triplets is not None:
        triplets = _count(triplets, "triplets")
        triplets_correct = _count(triplets_correct, "triplets_correct")
        if not triplets or triplets_correct > triplets:
            raise ValueError("Triplet scores are inconsistent")
    return {
        "rows": n,
        "correct": sum(correct),
        "class_correct": correct,
        "class_support": support,
        "false_safe": matrix[1][0],
        "false_violation": matrix[0][1],
        "unnecessary_abstention": matrix[0][2] + matrix[1][2],
        "pairs_correct": pairs,
        "triplets": triplets,
        "triplets_correct": triplets_correct,
        "min_recall": min(a / b for a, b in zip(correct, support)),
        "nll": float(nll),
    }


def constraint_violations(counts: dict, limits: dict) -> list[str]:
    """Check explicit population sizes and count thresholds, rejecting typos."""
    allowed = {
        "rows",
        "min_correct",
        "max_false_safe",
        "max_false_violation",
        "max_unnecessary_abstention",
        "min_pairs_correct",
        "whole_triplets_min",
        "min_class_correct",
    }
    if not isinstance(limits, dict) or set(limits) - allowed:
        raise ValueError("Unknown development constraint")
    failures = []
    for name, value in limits.items():
        if name == "min_class_correct":
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError("Class constraints need three integer counts")
            for index, minimum in enumerate(value):
                _count(minimum, name)
                if minimum > counts["class_support"][index]:
                    raise ValueError("Class minimum exceeds its population")
                if counts["class_correct"][index] < minimum:
                    failures.append("class_%d" % index)
            continue
        _count(value, name)
        if name == "rows":
            if counts["rows"] != value:
                raise ValueError("Development population size changed")
        elif name == "max_false_safe":
            if counts["false_safe"] > value:
                failures.append(name)
        elif name == "max_false_violation":
            if counts["false_violation"] > value:
                failures.append(name)
        elif name == "max_unnecessary_abstention":
            if counts["unnecessary_abstention"] > value:
                failures.append(name)
        elif name == "whole_triplets_min":
            if counts.get("triplets_correct") is None:
                raise ValueError(
                    "Triplet gate needs triplet scores in the view"
                )
            if counts["triplets_correct"] < value:
                failures.append(name)
        elif counts[name.removeprefix("min_")] < value:
            failures.append(name)
    return failures


def source_counts(entry: dict, name: str) -> dict:
    """Validate one per-source breakdown entry into checkable counts.

    Floors use exact counts, never recall floats: populations are
    fixed, so count thresholds are unambiguous. Entries come from
    the trainer's by_source summary, not from caller arithmetic.
    """
    if not isinstance(entry, dict):
        raise ValueError("Per-source entry must be an object: " + str(name))
    support = entry.get("class_support")
    correct = entry.get("class_correct")
    if (
        not isinstance(support, list)
        or len(support) != 3
        or not isinstance(correct, list)
        or len(correct) != 3
    ):
        raise ValueError(
            "Per-source entry needs three-class vectors: " + str(name)
        )
    support = [_count(value, name) for value in support]
    correct = [_count(value, name) for value in correct]
    if any(done > total for done, total in zip(correct, support)):
        raise ValueError("Per-source correct exceeds support: " + str(name))
    size = _count(entry.get("n"), name)
    if size != sum(support):
        raise ValueError("Per-source size disagrees with support: " + str(name))
    if _count(entry.get("correct"), name) != sum(correct):
        raise ValueError("Per-source correct disagrees by class: " + str(name))
    triplets = _count(entry.get("triplets"), name)
    triplets_correct = _count(entry.get("triplets_correct"), name)
    if triplets_correct > triplets:
        raise ValueError("Per-source triplets are inconsistent: " + str(name))
    return {
        "rows": size,
        "correct": sum(correct),
        "class_correct": correct,
        "class_support": support,
        "false_safe": _count(entry.get("false_safe"), name),
        "triplets_correct": triplets_correct,
    }


def objective_selection(result: dict, objective: dict) -> tuple:
    """Rank only after protected views pass; record distinct quality targets.

    The v1 objective's weakest class comes before pairs, overall accuracy
    and NLL. The v2 triplets objective ranks safety first, then whole
    triplets, correct decisions, false violations, unnecessary abstention
    and NLL; earlier checkpoints win ties through the caller's strict
    improvement comparison. The v4 sourced objective keeps the v3
    ranking key but additionally requires every registered source to
    pass its own floors, so one population cannot silently erase
    another. The v3 balanced objective ranks whole
    triplets, correct decisions, false-safes, weakest-class recall,
    false violations, unnecessary abstention and NLL: capability first
    with false-safe safeguards in feasibility and ranking. Quality
    targets describe readiness separately from preserving the incumbent.
    All names, thresholds and row counts are frozen in the run signature.
    """
    if objective.get("version") not in (
        CHECKPOINT_OBJECTIVE_SCHEMA,
        CHECKPOINT_TRIPLETS_SCHEMA,
        CHECKPOINT_BALANCED_SCHEMA,
        CHECKPOINT_SOURCED_SCHEMA,
    ):
        raise ValueError("Unsupported development objective version")
    sourced = objective["version"] == CHECKPOINT_SOURCED_SCHEMA
    expected_fields = {"version", "protected", "view", "targets"}
    if sourced:
        expected_fields.add("per_source")
    if set(objective) != expected_fields:
        raise ValueError(
            "Development objective fields are incomplete or unknown"
        )
    if not isinstance(result, dict):
        raise ValueError("Selection needs a result object")
    views = result.get("development_views")
    if not isinstance(views, dict) or not views:
        raise ValueError("Selection needs named development views")
    for name, view in views.items():
        if not isinstance(view, dict):
            raise ValueError("Development view is not an object: " + str(name))
    protected = objective["protected"]
    if not isinstance(protected, dict) or set(protected) != set(views):
        raise ValueError("Every development view needs explicit protection")
    per_source = objective.get("per_source", {})
    if sourced:
        if not isinstance(per_source, dict) or not per_source:
            raise ValueError("Sourced objective needs per-source floors")
        for name, view in views.items():
            breakdown = view.get("by_source")
            if not isinstance(breakdown, dict) or set(breakdown) != set(
                per_source
            ):
                raise ValueError(
                    "Per-source floors must match the view breakdown: " + name
                )
    counts = {name: view_counts(view) for name, view in views.items()}
    violations = [
        name + ":" + issue
        for name, limits in protected.items()
        for issue in constraint_violations(counts[name], limits)
    ]
    if sourced:
        for name, view in views.items():
            breakdown = view["by_source"]
            for source, limits in sorted(per_source.items()):
                entry = source_counts(breakdown[source], name + ":" + source)
                violations.extend(
                    "%s:%s:%s" % (name, source, issue)
                    for issue in constraint_violations(entry, limits)
                )
    if objective["view"] not in counts:
        raise ValueError("Objective names an absent development view")
    target = counts[objective["view"]]
    target_failures = constraint_violations(target, objective["targets"])
    result["checkpoint_acceptance"] = {
        "version": objective["version"],
        "protected_passed": not violations,
        "protected_violations": violations,
        "quality_targets_passed": not target_failures,
        "quality_target_failures": target_failures,
        "counts": counts,
    }
    if objective["version"] == CHECKPOINT_TRIPLETS_SCHEMA:
        if target["triplets_correct"] is None:
            raise ValueError("Triplets objective needs triplet scores")
        return (
            int(bool(violations)),
            len(violations),
            target["false_safe"],
            -target["triplets_correct"],
            -target["correct"],
            target["false_violation"],
            target["unnecessary_abstention"],
            target["nll"],
        )
    if objective["version"] in (
        CHECKPOINT_BALANCED_SCHEMA,
        CHECKPOINT_SOURCED_SCHEMA,
    ):
        if target["triplets_correct"] is None:
            raise ValueError("Capability-first objective needs triplet scores")
        return (
            int(bool(violations)),
            len(violations),
            -target["triplets_correct"],
            -target["correct"],
            target["false_safe"],
            -target["min_recall"],
            target["false_violation"],
            target["unnecessary_abstention"],
            target["nll"],
        )
    return (
        int(bool(violations)),
        len(violations),
        -target["min_recall"],
        -target["pairs_correct"],
        -target["correct"],
        target["nll"],
    )
