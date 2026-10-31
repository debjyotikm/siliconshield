"""Fresh-study selection flow through the actual trainer consumers.

Covers the executable contract end to end on synthetic rows: views
manifest to trainer config, view attachment with strict triplet
scoring, v2 objective ranking and gates, job flag forwarding, and
exposure-gated plateau refusal. No live services, GPUs, or data.
"""

import pytest

from siliconshield.research_loop.execution.security_jobs import command_for
from siliconshield.schemas import (
    CHECKPOINT_BALANCED_SCHEMA,
    CHECKPOINT_SOURCED_SCHEMA,
    CHECKPOINT_TRIPLETS_SCHEMA,
)
from siliconshield.training.laya.checkpoint_selection import (
    constraint_violations,
    objective_selection,
    population_indices,
    triplet_counts,
    view_counts,
)
from siliconshield.training.laya.coverage_audit import (
    delivered_exposure,
    exposure_gate_passed,
    reconstruct_seen,
)
from siliconshield.training.laya.development_views import (
    attach_views,
    config_from_views_manifest,
    selection,
)
from siliconshield.training.laya.security_train import (
    grouped_epoch_indices,
    plateau_action,
    retention_check_for,
    source_breakdown,
    summarize_development_view,
)

LABELS = [
    "satisfies_requirement",
    "violates_requirement",
    "insufficient_context",
]


def _row(row_id, pair, label):
    return {
        "id": row_id,
        "pair_id": pair,
        "expected": {"security": label},
        "source": {"name": "OpenTitan"},
        "split_group_id": "g1",
        "mechanism_id": "lock",
        "base_design_id": "d0",
        "topic": "t",
        "family_id": "f",
        "state": "REQUIREMENT\nr\nASSUMPTIONS\na\nRTL\nassign q = 1;\n",
    }


def _rows():
    rows = []
    for pair in ("pair-a", "pair-b"):
        for index, label in enumerate(LABELS):
            rows.append(_row("%s-%d" % (pair, index), pair, label))
    return rows


def _logits(correct=True, wrong=()):
    table = []
    for position in range(6):
        target = position % 3
        pred = target if correct and position not in wrong else (target + 1) % 3
        row = [0.0, 0.0, 0.0]
        row[pred] = 5.0
        table.append(row)
    return table


def _manifest():
    return {
        "schema": "siliconshield-dev-views-v3",
        "n_rows": 6,
        "views": [
            {
                "record_id": "D0000",
                "review": "agree",
                "triplets": [
                    {
                        "triplet_id": "pair-a",
                        "satisfies": "pair-a-0",
                        "violates": "pair-a-1",
                        "insufficient": "pair-a-2",
                    },
                    {
                        "triplet_id": "pair-b",
                        "satisfies": "pair-b-0",
                        "violates": "pair-b-1",
                        "insufficient": "pair-b-2",
                    },
                ],
            }
        ],
    }


def _objective():
    return {
        "version": CHECKPOINT_TRIPLETS_SCHEMA,
        "protected": {
            "fresh_dev": {
                "rows": 6,
                "max_false_safe": 0,
                "whole_triplets_min": 1,
            }
        },
        "view": "fresh_dev",
        "targets": {"whole_triplets_min": 2},
    }


def _attached(rows, logits):
    config = config_from_views_manifest(_manifest())
    result = {}
    attach_views(
        result,
        rows,
        logits,
        config,
        lambda selected_rows, selected_logits: (
            summarize_development_view(
                selected_rows,
                selected_logits,
                score_triplets=True,
            )
        ),
    )
    return result


def test_manifest_builds_single_view_row_config():
    config = config_from_views_manifest(_manifest())
    assert config["selection_view"] == "fresh_dev"
    assert config["single_view_study"] is True
    assert config["score_triplets"] is True
    assert population_indices(_rows(), config["views"]["fresh_dev"]) == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]


def test_manifest_refuses_unreviewed_or_repeated_rows():
    bad = _manifest()
    bad["views"][0]["review"] = "pending"
    with pytest.raises(ValueError, match="Unreviewed template"):
        config_from_views_manifest(bad)
    bad = _manifest()
    bad["views"][0]["triplets"][1]["satisfies"] = "pair-a-0"
    with pytest.raises(ValueError, match="repeats a row"):
        config_from_views_manifest(bad)


def test_strict_triplets_follow_pair_id_not_row_order():
    rows = _rows()
    predictions = [0, 1, 2, 0, 1, 2]
    assert triplet_counts(rows, predictions, LABELS) == {
        "triplets": 2,
        "triplets_correct": 2,
    }
    shuffled = [rows[i] for i in (3, 0, 4, 1, 5, 2)]
    shuffled_preds = [predictions[i] for i in (3, 0, 4, 1, 5, 2)]
    assert triplet_counts(shuffled, shuffled_preds, LABELS) == {
        "triplets": 2,
        "triplets_correct": 2,
    }
    # All rows right except one original triplet's member: 2 -> 1.
    assert triplet_counts(rows, [0, 1, 2, 0, 1, 0], LABELS) == {
        "triplets": 2,
        "triplets_correct": 1,
    }
    broken = [dict(row) for row in rows]
    del broken[0]["pair_id"]
    with pytest.raises(ValueError, match="pair_id"):
        triplet_counts(broken, predictions, LABELS)


def test_attached_view_carries_strict_triplet_counts():
    result = _attached(_rows(), _logits())
    view = result["development_views"]["fresh_dev"]
    assert view["triplets"] == 2
    assert view["triplets_correct"] == 2
    counts = view_counts(view)
    assert counts["triplets_correct"] == 2
    assert counts["false_violation"] == 0
    assert counts["unnecessary_abstention"] == 0


def test_triplet_gate_fails_without_triplet_scores():
    counts = {
        "rows": 6,
        "correct": 6,
        "class_correct": [2, 2, 2],
        "class_support": [2, 2, 2],
        "false_safe": 0,
        "false_violation": 0,
        "unnecessary_abstention": 0,
        "pairs_correct": 2,
        "triplets": None,
        "triplets_correct": None,
        "min_recall": 1.0,
        "nll": 0.1,
    }
    with pytest.raises(ValueError, match="triplet scores"):
        constraint_violations(counts, {"whole_triplets_min": 1})
    counts["triplets"] = 2
    counts["triplets_correct"] = 1
    assert constraint_violations(counts, {"whole_triplets_min": 2}) == [
        "whole_triplets_min"
    ]
    assert constraint_violations(counts, {"whole_triplets_min": 1}) == []


def _vdict(confusion, triplets_correct, nll=0.5):
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(3))
    return {
        "confusion": confusion,
        "n": total,
        "false_safe_count": confusion[1][0],
        "accuracy": correct / total,
        "complete_pairs": 3,
        "both_pair_members_correct": 1 / 3,
        "nll": nll,
        "triplets": 3,
        "triplets_correct": triplets_correct,
    }


def _balanced_objective():
    return {
        "version": CHECKPOINT_BALANCED_SCHEMA,
        "protected": {
            "fresh_dev": {
                "rows": 9,
                "min_correct": 4,
                "max_false_safe": 2,
                "whole_triplets_min": 0,
            }
        },
        "view": "fresh_dev",
        "targets": {"whole_triplets_min": 3},
    }


def test_v3_balanced_ranks_capability_before_safety():
    weak = {
        "development_views": {
            "fresh_dev": _vdict([[2, 1, 0], [1, 2, 0], [1, 2, 0]], 0)
        }
    }
    capable = {
        "development_views": {
            "fresh_dev": _vdict([[2, 1, 0], [2, 1, 0], [0, 1, 2]], 2)
        }
    }
    weak_key = objective_selection(weak, _balanced_objective())
    capable_key = objective_selection(capable, _balanced_objective())
    assert weak["checkpoint_acceptance"]["protected_passed"]
    assert capable["checkpoint_acceptance"]["protected_passed"]
    assert capable_key < weak_key
    assert capable_key[2:5] == (-2, -5, 2)
    assert weak_key[2:5] == (0, -4, 1)


def test_v3_recall_breaks_capability_ties():
    sharp = {
        "development_views": {
            "fresh_dev": _vdict([[2, 1, 0], [1, 2, 0], [0, 1, 2]], 1)
        }
    }
    dull = {
        "development_views": {
            "fresh_dev": _vdict([[3, 0, 0], [1, 2, 0], [0, 2, 1]], 1)
        }
    }
    assert objective_selection(sharp, _balanced_objective()) < (
        objective_selection(dull, _balanced_objective())
    )


def test_v3_feasibility_rejects_one_class_predictors():
    always_violates = {
        "development_views": {
            "fresh_dev": _vdict([[0, 3, 0], [0, 3, 0], [0, 3, 0]], 0)
        }
    }
    objective_selection(always_violates, _balanced_objective())
    assert not always_violates["checkpoint_acceptance"]["protected_passed"]
    always_satisfies = {
        "development_views": {
            "fresh_dev": _vdict([[3, 0, 0], [3, 0, 0], [3, 0, 0]], 0)
        }
    }
    objective_selection(always_satisfies, _balanced_objective())
    assert not always_satisfies["checkpoint_acceptance"]["protected_passed"]


def test_v2_objective_ranks_safety_before_triplets():
    safe = _attached(_rows(), _logits())
    # One false-safe plus otherwise perfect: violates row 1 -> satisfies.
    risky_logits = _logits()
    risky_logits[1] = [5.0, 0.0, 0.0]
    risky = _attached(_rows(), risky_logits)
    assert objective_selection(safe, _objective()) < objective_selection(
        risky, _objective()
    )
    assert safe["checkpoint_acceptance"]["protected_passed"]
    assert not risky["checkpoint_acceptance"]["protected_passed"]


def test_objective_selection_needs_no_retention_view():
    result = _attached(_rows(), _logits())
    config = config_from_views_manifest(_manifest())
    config["checkpoint_objective"] = _objective()
    assert "retention_view" not in config
    key = selection(result, config, None, None)
    assert key[:5] == (0, 0, 0, -2, -6)
    assert retention_check_for(config, result, None) is None
    legacy = {
        "retention_view": "fresh_dev",
        "max_retention_accuracy_drop": 0.01,
    }
    check = retention_check_for(legacy, result, 1.0)
    assert check == {
        "passed": True,
        "reference_accuracy": 1.0,
        "accuracy": 1.0,
        "max_accuracy_drop": 0.01,
    }
    risky_logits = _logits()
    risky_logits[1] = [5.0, 0.0, 0.0]
    risky = _attached(_rows(), risky_logits)
    regressed = retention_check_for(legacy, risky, 1.0)
    assert regressed["passed"] is False
    assert regressed["accuracy"] == pytest.approx(5 / 6)


def test_job_forwards_review_packets_flag(tmp_path):
    packets = tmp_path / "packets.json"
    packets.write_text("{}")
    profile = {
        "source": str(tmp_path),
        "runs": str(tmp_path / "runs"),
        "python": "python",
        "model": str(tmp_path / "model"),
        "data": str(tmp_path / "data"),
        "defaults": {"max_updates": 8, "micro_batch": 2, "accumulation": 4},
    }
    command, _ = command_for(profile, "fresh", {"review_packets": str(packets)})
    assert command[command.index("--review-packets") + 1] == str(packets)


def _pool_plan():
    return ({index: "pool" for index in range(6)}, {"pool": 1.0}, {})


def test_replay_matches_sampler_draws_exactly():
    rows = _rows()
    plan = _pool_plan()
    direct = grouped_epoch_indices(rows, 1, 0, 102, 0, 6, plan, _queue_state={})
    replayed = reconstruct_seen(
        rows,
        plan,
        102,
        0,
        6,
        world=1,
        global_batch=6,
        grouped_epoch_indices=grouped_epoch_indices,
    )
    expanded = sorted(
        index for index, draws in replayed.items() for _ in range(draws)
    )
    assert expanded == sorted(direct)
    partial = reconstruct_seen(
        rows,
        plan,
        102,
        0,
        3,
        world=1,
        global_batch=6,
        grouped_epoch_indices=grouped_epoch_indices,
    )
    expanded = sorted(
        index for index, draws in partial.items() for _ in range(draws)
    )
    assert expanded == sorted(direct[:3])


def test_plateau_refuses_before_full_exposure():
    rows = _rows()
    plan = _pool_plan()
    progress = {"step": 100, "stale": 6, "reduced": False}
    partial = reconstruct_seen(
        rows,
        plan,
        102,
        0,
        0,
        world=1,
        global_batch=6,
        grouped_epoch_indices=grouped_epoch_indices,
    )
    exposure = delivered_exposure(partial, rows, plan[0])
    assert not exposure_gate_passed(exposure)
    assert (
        plateau_action(progress, 0, 10, 2, exposure_gate_passed(exposure))
        is None
    )
    full = reconstruct_seen(
        rows,
        plan,
        102,
        0,
        6,
        world=1,
        global_batch=6,
        grouped_epoch_indices=grouped_epoch_indices,
    )
    exposure = delivered_exposure(full, rows, plan[0])
    assert exposure_gate_passed(exposure)
    assert (
        plateau_action(progress, 0, 10, 2, exposure_gate_passed(exposure))
        == "reduce_lr"
    )


def _two_source_rows():
    rows = []
    specs = (
        ("pair-a", "OpenTitan", "g1", "d0"),
        ("pair-b", "HardSecBench", "g2", "d1"),
    )
    for pair, source, group, design in specs:
        for index, label in enumerate(LABELS):
            row = _row("%s-%d" % (pair, index), pair, label)
            row["source"] = {"name": source}
            row["split_group_id"] = group
            row["base_design_id"] = design
            rows.append(row)
    return rows


def test_summarize_scores_per_source_breakdown():
    rows = _two_source_rows()
    logits = _logits(correct=True, wrong=(4,))
    summary = summarize_development_view(
        rows, logits, score_triplets=True, score_sources=True
    )
    assert summary["triplets_correct"] == 1
    by_source = summary["by_source"]
    assert by_source["OpenTitan"]["correct"] == 3
    assert by_source["OpenTitan"]["triplets_correct"] == 1
    assert by_source["HardSecBench"]["correct"] == 2
    assert by_source["HardSecBench"]["triplets_correct"] == 0
    assert by_source["HardSecBench"]["false_safe"] == 0
    assert by_source["HardSecBench"]["class_correct"] == [1, 0, 1]


def test_breakdown_ignores_split_triplets_per_source():
    rows = _two_source_rows()
    rows[5]["source"] = {"name": "OpenTitan"}
    logits = _logits()
    by_source = source_breakdown(rows, logits)
    assert by_source["OpenTitan"]["triplets"] == 1
    assert by_source["HardSecBench"]["triplets"] == 0
    assert by_source["HardSecBench"]["correct"] == 2


def _sourced_view(by_source, triplets_correct=1):
    support = [0, 0, 0]
    class_correct = [0, 0, 0]
    for entry in by_source.values():
        for index in range(3):
            support[index] += entry["class_support"][index]
            class_correct[index] += entry["class_correct"][index]
    total = sum(support)
    correct = sum(class_correct)
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for index in range(3):
        matrix[index][index] = class_correct[index]
    matrix[2][1] = total - correct
    return {
        "confusion": matrix,
        "n": total,
        "false_safe_count": 0,
        "accuracy": correct / total,
        "complete_pairs": 2,
        "both_pair_members_correct": 0.5,
        "nll": 0.5,
        "triplets": 2,
        "triplets_correct": triplets_correct,
        "by_source": by_source,
    }


def _sourced_objective():
    return {
        "version": CHECKPOINT_SOURCED_SCHEMA,
        "protected": {
            "fresh_dev": {
                "rows": 6,
                "min_correct": 4,
                "max_false_safe": 1,
                "whole_triplets_min": 1,
            }
        },
        "per_source": {
            "OpenTitan": {
                "rows": 3,
                "min_correct": 2,
                "max_false_safe": 0,
                "whole_triplets_min": 1,
            },
            "HardSecBench": {
                "rows": 3,
                "min_correct": 1,
                "max_false_safe": 1,
                "whole_triplets_min": 0,
            },
        },
        "view": "fresh_dev",
        "targets": {"whole_triplets_min": 2},
    }


def _source_entry(class_correct, support, fs=0, triplets=1, trip_ok=1):
    return {
        "n": sum(support),
        "correct": sum(class_correct),
        "class_correct": class_correct,
        "class_support": support,
        "false_safe": fs,
        "triplets": triplets,
        "triplets_correct": trip_ok,
        "min_recall": 0.5,
    }


def test_v4_passes_when_every_source_passes():
    view = _sourced_view(
        {
            "OpenTitan": _source_entry([1, 1, 1], [1, 1, 1]),
            "HardSecBench": _source_entry([1, 1, 0], [1, 1, 1]),
        }
    )
    result = {"development_views": {"fresh_dev": view}}
    key = objective_selection(result, _sourced_objective())
    assert result["checkpoint_acceptance"]["protected_passed"]
    assert key[2:5] == (-1, -5, 0)


def test_v4_blocks_when_one_source_regresses():
    view = _sourced_view(
        {
            "OpenTitan": _source_entry([1, 0, 0], [1, 1, 1], trip_ok=0),
            "HardSecBench": _source_entry([1, 1, 1], [1, 1, 1]),
        }
    )
    result = {"development_views": {"fresh_dev": view}}
    objective_selection(result, _sourced_objective())
    acceptance = result["checkpoint_acceptance"]
    assert not acceptance["protected_passed"]
    assert any(
        violation.startswith("fresh_dev:OpenTitan:")
        for violation in acceptance["protected_violations"]
    )


def test_v4_rejects_missing_or_unknown_sources():
    view = _sourced_view({"OpenTitan": _source_entry([1, 1, 1], [1, 1, 1])})
    result = {"development_views": {"fresh_dev": view}}
    with pytest.raises(ValueError, match="Per-source"):
        objective_selection(result, _sourced_objective())
    view = _sourced_view(
        {
            "OpenTitan": _source_entry([1, 1, 1], [1, 1, 1]),
            "HardSecBench": _source_entry([1, 1, 0], [1, 1, 1]),
            "authored-semantic": _source_entry([1, 0, 0], [1, 1, 1], trip_ok=0),
        }
    )
    result = {"development_views": {"fresh_dev": view}}
    with pytest.raises(ValueError, match="Per-source"):
        objective_selection(result, _sourced_objective())
