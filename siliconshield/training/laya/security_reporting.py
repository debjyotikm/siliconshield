"""Report decision quality with explicit weights and source-stratified lineage uncertainty."""

import hashlib
from collections import defaultdict

import numpy as np

LABELS = (
    "satisfies_requirement",
    "violates_requirement",
    "insufficient_context",
)
METRICS = (
    "accuracy",
    "macro_f1",
    "nll",
    "false_safe_rate",
    "missing_context_recall",
    "pair_accuracy",
)
VERSION = "security-reporting-v1"


def _metrics(statistics, supported):
    matrix = statistics[:9].reshape(3, 3)
    actual, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    total = actual.sum()
    return {
        "accuracy": float(diagonal.sum() / total),
        "macro_f1": float(
            (2 * diagonal / np.maximum(actual + predicted, 1e-300))[
                supported
            ].mean()
        ),
        "nll": float(statistics[9] / total),
        "false_safe_rate": float(matrix[1, 0] / actual[1])
        if actual[1]
        else None,
        "missing_context_recall": float(matrix[2, 2] / actual[2])
        if actual[2]
        else None,
        "pair_accuracy": float(statistics[10] / statistics[11])
        if statistics[11]
        else None,
    }


def _cluster_statistics(rows, targets, predictions, losses, weights):
    # One statistic vector per lineage: confusion, loss sum, correct-pair mass, pair mass.
    values, owners, pairs = {}, {}, defaultdict(list)
    for i, row in enumerate(rows):
        group = row["split_group_id"]
        source = row["source"]["name"]
        if owners.setdefault(group, source) != source:
            raise ValueError(
                "a split group spans sources; source-stratified bootstrap is undefined"
            )
        item = values.setdefault(group, np.zeros(12, dtype=np.float64))
        item[3 * targets[i] + predictions[i]] += weights[i]
        item[9] += weights[i] * losses[i]
        pairs[row["pair_id"]].append(i)
    complete_pairs = 0
    for indices in pairs.values():
        groups = {rows[i]["split_group_id"] for i in indices}
        if len(groups) != 1:
            raise ValueError("a decision pair crosses split groups")
        members = [i for i in indices if targets[i] in (0, 1)]
        if {targets[i] for i in members} != {0, 1}:
            continue
        mass = float(weights[members].sum())
        stat = values[rows[members[0]]["split_group_id"]]
        stat[10] += mass * all(predictions[i] == targets[i] for i in members)
        stat[11] += mass
        complete_pairs += 1
    return values, owners, complete_pairs


def _intervals(values, owners, supported, samples, seed):
    strata = defaultdict(list)
    for group, stats in sorted(values.items()):
        strata[owners[group]].append(stats)
    counts = {
        source: len(vectors) for source, vectors in sorted(strata.items())
    }
    result = {
        "method": "source-stratified whole-cluster percentile bootstrap",
        "cluster_key": "split_group_id",
        "source_cluster_counts": counts,
        "source_mass_policy": "hold each observed source stratum mass fixed",
        "confidence": 0.95,
        "samples": samples,
        "seed": seed,
    }
    singletons = [name for name, count in counts.items() if count < 2]
    if singletons or samples == 0:
        return dict(
            result,
            status="unavailable",
            reason="fewer_than_two_clusters_in_source"
            if singletons
            else "bootstrap_disabled",
            affected_sources=singletons,
            intervals={key: None for key in METRICS},
        )
    rng = np.random.default_rng(seed)
    draws = np.zeros((samples, 12), dtype=np.float64)
    for vectors in strata.values():
        vectors = np.asarray(vectors)
        chosen = rng.integers(len(vectors), size=(samples, len(vectors)))
        totals = vectors[chosen].sum(axis=1)
        mass = vectors[:, :9].sum()
        draws += totals * (mass / totals[:, :9].sum(axis=1))[:, None]
    measured = [_metrics(draw, supported) for draw in draws]
    intervals = {}
    for name in METRICS:
        valid = [m[name] for m in measured if m[name] is not None]
        intervals[name] = (
            {
                "lower": float(np.quantile(valid, 0.025)),
                "upper": float(np.quantile(valid, 0.975)),
                "valid_samples": len(valid),
            }
            if valid
            else None
        )
    return dict(result, status="available", intervals=intervals)


def evaluation_report(
    rows,
    logits,
    weights,
    source_mass,
    bootstrap_samples=1000,
    seed=0,
    temperature=1.0,
    from_probabilities=False,
):
    """Keep reporting separate from checkpoint selection and train/dev/test access."""
    if (
        not rows
        or isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 0
    ):
        raise ValueError(
            "nonempty rows and a nonnegative bootstrap count are required"
        )
    z, w = (
        np.asarray(logits, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )
    if z.shape != (len(rows), 3) or w.shape != (len(rows),):
        raise ValueError("misaligned rows, logits, or weights")
    if not np.isfinite(z).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("logits and positive weights must be finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate row IDs in evaluation")
    if not source_mass or any(
        isinstance(v, bool) or not np.isfinite(v) or v <= 0
        for v in source_mass.values()
    ):
        raise ValueError("configured source masses must be finite and positive")
    if from_probabilities:
        if np.any(z < 0) or np.any(z.sum(axis=1) <= 0):
            raise ValueError("invalid probability rows")
        z = np.clip(z, 0.00005, 1.0)
        z = np.log(z / z.sum(axis=1, keepdims=True))
    z = z / temperature
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    targets = np.array(
        [LABELS.index(row["expected"]["security"]) for row in rows]
    )
    predictions = logp.argmax(axis=1)
    losses = -logp[np.arange(len(rows)), targets]
    source_indices, mechanism_indices = defaultdict(list), defaultdict(list)
    group_owner = {}
    for i, row in enumerate(rows):
        source = row["source"]["name"]
        if source not in source_mass:
            raise ValueError(
                "evaluation source absent from configured source masses"
            )
        if group_owner.setdefault(row["split_group_id"], source) != source:
            raise ValueError("a split group spans sources")
        source_indices[source].append(i)
        mechanism_indices[(source, row["mechanism_id"])].append(i)

    def view(indices, name):
        selected = [rows[i] for i in indices]
        y, p, loss, weight = (
            targets[indices],
            predictions[indices],
            losses[indices],
            w[indices],
        )
        support = np.bincount(y, minlength=3)
        raw, owners, pairs = _cluster_statistics(
            selected, y, p, loss, np.ones(len(indices))
        )
        weighted, _, _ = _cluster_statistics(selected, y, p, loss, weight)
        source_support = {}
        for source in sorted({r["source"]["name"] for r in selected}):
            subset = [r for r in selected if r["source"]["name"] == source]
            source_support[source] = {
                "decision_rows": len(subset),
                "split_groups": len({r["split_group_id"] for r in subset}),
            }
        local_seed = (
            seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        ) % (2**32)
        result = {
            "support": {
                "decision_rows": len(indices),
                "split_groups": len(raw),
                "class_support": support.tolist(),
                "complete_pairs": pairs,
                "sources": source_support,
            }
        }
        for key, statistics in (("unweighted", raw), ("weighted", weighted)):
            aggregate = np.asarray(list(statistics.values())).sum(axis=0)
            result[key] = {
                "metrics": _metrics(aggregate, support > 0),
                "confusion_mass": aggregate[:9].reshape(3, 3).tolist(),
                "intervals": _intervals(
                    statistics,
                    owners,
                    support > 0,
                    bootstrap_samples,
                    local_seed,
                ),
            }
        result["weighting"] = {
            "definition": "Fixed supplied source/mechanism/design/label row weights, normalized within this reported population.",
            "pair_definition": "Sum of member row weights for labels 0 and 1; a pair is correct only when both verdicts are correct.",
            "kish_row_ess": float(weight.sum() ** 2 / np.square(weight).sum()),
            "ess_scope": "Weight dispersion only, not a count of independent designs.",
            "observed_source_mass": {
                source: float(
                    sum(
                        weight[j]
                        for j, r in enumerate(selected)
                        if r["source"]["name"] == source
                    )
                    / weight.sum()
                )
                for source in source_support
            },
        }
        return result

    configured = {
        k: float(v / sum(source_mass.values())) for k, v in source_mass.items()
    }
    missing = sorted(set(configured) - set(source_indices))
    result = {
        "schema": VERSION,
        "labels": list(LABELS),
        "configured_source_mass": configured,
        "missing_sources": missing,
        "full_configured_mixture_estimable": not missing,
        "estimand": "Observed evaluation population; missing sources are not imputed.",
        "overall": view(list(range(len(rows))), "overall"),
        "per_source": {
            source: view(indices, "source:" + source)
            for source, indices in sorted(source_indices.items())
        },
        "per_source_mechanism": {},
    }
    for (source, mechanism), indices in sorted(mechanism_indices.items()):
        result["per_source_mechanism"].setdefault(source, {})[mechanism] = view(
            indices, source + ":" + mechanism
        )
    return result
