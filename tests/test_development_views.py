"""Joint-constraint checkpoint selection and family-cap scope."""

import pytest

from siliconshield.training.laya import development_views as views


def block(
    accuracy=0.99,
    false_safe=1,
    violations=10,
    recall=0.9,
    pairs=0.95,
    mc=(10, 10),
    families=None,
    nll=0.002,
    n=30,
    complete_pairs=10,
):
    mc_correct, mc_total = mc
    return {
        "accuracy": accuracy,
        "false_safe_count": false_safe,
        "violation_count": violations,
        "min_class_recall": recall,
        "both_pair_members_correct": pairs,
        "complete_pairs": complete_pairs,
        "confusion": [
            [8, 1, 1],
            [1, 8, 1],
            [mc_total - mc_correct, 0, mc_correct],
        ],
        "per_family": families
        if families is not None
        else {"famA": {"false_safe_count": 1, "violation_count": 6}},
        "n": n,
        "nll": nll,
    }


def policy(**overrides):
    base = {
        "min_retention_accuracy": 0.95,
        "max_retention_false_safe_increase": 0,
        "require_no_missing_context_regression": True,
        "require_no_pair_regression": True,
        "min_new_coverage_gain_pp": 0.0,
    }
    base.update(overrides)
    return base


def reference(**overrides):
    transfer = block()
    retention = block()
    return views.build_selection_reference(
        transfer, retention, policy(**overrides), "digest"
    )


def result(transfer, retention):
    return {"development_views": {"transfer": transfer, "retention": retention}}


def config(**overrides):
    base = {
        "selection_view": "transfer",
        "retention_view": "retention",
        "max_retention_accuracy_drop": 0.01,
    }
    base.update(overrides)
    return base


def base_key(view):
    return (view["nll"],)


def test_legacy_selection_ranks_transfer_nll_only():
    good = result(block(nll=0.002), block(pairs=0.95))
    bad = result(block(nll=0.001), block(pairs=0.90))
    assert views.selection(good, config(), 0.99, base_key) > (
        views.selection(bad, config(), 0.99, base_key)
    )


def test_joint_selection_preserves_pairs_over_lower_nll():
    ref = reference(
        max_new_coverage_false_safe_increase=0,
        family_false_safe_caps={"famA": 1},
    )
    good = result(block(nll=0.002), block(pairs=0.95))
    bad = result(block(nll=0.001), block(pairs=0.90))
    assert views.selection(
        good, config(selection_reference=ref), 0.99, base_key
    ) < views.selection(bad, config(selection_reference=ref), 0.99, base_key)


def test_joint_selection_enforces_coverage_gain():
    incumbent = block(accuracy=84 / 90)
    ref = views.build_selection_reference(
        incumbent, block(), policy(min_new_coverage_gain_pp=4.0), "digest"
    )
    cfg = config(selection_reference=ref)
    feasible = result(block(accuracy=88 / 90, nll=0.002), block())
    short = result(block(accuracy=87 / 90, nll=0.001), block())
    assert views.joint_flags(feasible, cfg)[-1] == 0
    assert views.joint_flags(short, cfg)[-1] == 1
    assert views.selection(feasible, cfg, 0.99, base_key) < views.selection(
        short, cfg, 0.99, base_key
    )
    assert views.selection(
        feasible, config(), 0.99, base_key
    ) > views.selection(short, config(), 0.99, base_key)


def test_v2_ranking_prefers_retention_quality_over_transfer_nll():
    v2 = reference(
        selection_ranking="v2-retention",
        max_retention_false_safe_increase=1,
    )
    v1 = reference(max_retention_false_safe_increase=1)
    safer = result(block(nll=0.002), block(false_safe=1))
    cheaper = result(block(nll=0.001), block(false_safe=2))
    v2_config = config(selection_reference=v2)
    assert sum(views.joint_flags(safer, v2_config)) == 0
    assert sum(views.joint_flags(cheaper, v2_config)) == 0
    safer_key = views.selection(safer, v2_config, 0.99, base_key)
    cheaper_key = views.selection(cheaper, v2_config, 0.99, base_key)
    assert safer_key < cheaper_key
    assert tuple(safer_key[-5:]) == views.retention_rank_key(
        1, 30, 10, 0.002, 0.002
    )
    assert tuple(cheaper_key[-5:]) == views.retention_rank_key(
        2, 30, 10, 0.002, 0.001
    )
    v1_config = config(selection_reference=v1)
    assert views.selection(safer, v1_config, 0.99, base_key) > views.selection(
        cheaper, v1_config, 0.99, base_key
    )


def test_v2_ranking_orders_correct_pairs_then_nll():
    ref = views.build_selection_reference(
        block(),
        block(pairs=0.85),
        policy(selection_ranking="v2-retention"),
        "digest",
    )
    cfg = config(selection_reference=ref)

    def rank(transfer, retention):
        return views.selection(result(transfer, retention), cfg, 0.99, base_key)

    more_correct = rank(block(), block(accuracy=0.99))
    less_correct = rank(block(), block(accuracy=29 / 30))
    assert more_correct < less_correct
    more_pairs = rank(block(), block(pairs=1.0))
    fewer_pairs = rank(block(), block(pairs=0.9))
    assert more_pairs < fewer_pairs
    lower_retention_nll = rank(block(), block(nll=0.001))
    higher_retention_nll = rank(block(), block(nll=0.002))
    assert lower_retention_nll < higher_retention_nll
    lower_transfer_nll = rank(block(nll=0.001), block())
    higher_transfer_nll = rank(block(nll=0.002), block())
    assert lower_transfer_nll < higher_transfer_nll


def test_ranking_defaults_to_v1_transfer():
    assert views.ranking_version(policy()) == "v1-transfer"
    assert views.ranking_version(policy(selection_ranking="v2-retention")) == (
        "v2-retention"
    )
    with pytest.raises(ValueError, match="unknown ranking"):
        views.ranking_version(policy(selection_ranking="v9"))
    with pytest.raises(ValueError, match="unknown ranking"):
        views.build_selection_reference(
            block(), block(), policy(selection_ranking="v9"), "d"
        )


def test_negative_budget_requires_a_decrease():
    ref = reference(max_retention_false_safe_increase=-1)
    cfg = config(selection_reference=ref)
    held = result(block(), block(false_safe=1))
    improved = result(block(), block(false_safe=0))
    assert views.joint_flags(held, cfg)[1] == 1
    assert views.joint_flags(improved, cfg)[1] == 0


def test_shared_gain_predicate_matches_campaign_math():
    assert views.new_coverage_gain_ok(88 / 90, 84 / 90, 4.0)
    assert not views.new_coverage_gain_ok(87 / 90, 84 / 90, 4.0)
    assert views.new_coverage_gain_ok(0.9, 0.9, 0.0)
    assert not views.new_coverage_gain_ok(0.9, 0.9, 0.01)


def test_joint_flags_cover_budgets_and_caps():
    ref = reference(
        max_new_coverage_false_safe_increase=0,
        family_false_safe_caps={
            "new_coverage": {"famA": 1},
            "retention": {"famR": 0},
        },
    )
    held = result(
        block(families={"famA": {"false_safe_count": 0, "violation_count": 6}}),
        block(
            false_safe=1,
            families={"famR": {"false_safe_count": 0, "violation_count": 4}},
        ),
    )
    assert views.joint_flags(held, config(selection_reference=ref)) == (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    over_budget = result(
        block(false_safe=3),
        block(false_safe=1),
    )
    assert (
        views.joint_flags(over_budget, config(selection_reference=ref))[6] == 1
    )
    capped = result(
        block(families={"famA": {"false_safe_count": 2, "violation_count": 6}}),
        block(false_safe=1),
    )
    assert views.joint_flags(capped, config(selection_reference=ref))[7] == 1
    missing = result(
        block(
            families={"other": {"false_safe_count": 0, "violation_count": 6}}
        ),
        block(false_safe=1),
    )
    assert views.joint_flags(missing, config(selection_reference=ref))[7] == 1
    retention_capped = result(
        block(),
        block(
            false_safe=1,
            families={"famR": {"false_safe_count": 1, "violation_count": 4}},
        ),
    )
    assert (
        views.joint_flags(retention_capped, config(selection_reference=ref))[7]
        == 1
    )


def test_joint_flags_detect_mc_regression():
    ref = reference()
    regressed = result(block(), block(mc=(9, 10)))
    flags = views.joint_flags(regressed, config(selection_reference=ref))
    assert flags[2] == 1


def test_normalize_family_caps_shapes():
    assert views.normalize_family_caps({"a": 1}) == {
        "new_coverage": {"a": 1},
        "retention": {},
    }
    nested = {"new_coverage": {"a": 1}, "retention": {"b": 0}}
    assert views.normalize_family_caps(nested) == nested
    assert views.normalize_family_caps({}) == {
        "new_coverage": {},
        "retention": {},
    }
    with pytest.raises(ValueError, match="family caps"):
        views.normalize_family_caps({"a": -1})
    with pytest.raises(ValueError, match="family caps"):
        views.normalize_family_caps({"a": 1, "retention": {}})
    with pytest.raises(ValueError, match="family caps"):
        views.normalize_family_caps({"unknown": {}})


def test_reference_builder_validates_blocks():
    with pytest.raises(ValueError, match="confusion"):
        views.build_selection_reference({}, block(), policy(), "d")
    with pytest.raises(ValueError, match="retention floor"):
        views.build_selection_reference(
            block(), block(), {"min_retention_accuracy": 2}, "d"
        )
    incomplete = policy()
    del incomplete["min_new_coverage_gain_pp"]
    with pytest.raises(ValueError, match="gain threshold"):
        views.build_selection_reference(block(), block(), incomplete, "d")
    with pytest.raises(ValueError, match="schema"):
        views.check_selection_reference({"schema": "other"})


def adapted_view(false_safe=1, pairs=(9, 10), families="default"):
    pairs_correct, pairs_total = pairs
    return {
        "counts": {
            "n": 30,
            "correct": 28,
            "false_safe": false_safe,
            "violations": 10,
            "mc_correct": 10,
            "mc_total": 10,
            "pairs_correct": pairs_correct,
            "pairs_total": pairs_total,
            "classes": [
                {"correct": 10, "total": 10},
                {"correct": 8, "total": 10},
                {"correct": 10, "total": 10},
            ],
        },
        "families": (
            {"famA": {"false_safe": 1, "violations": 6}}
            if families == "default"
            else families
        ),
    }


def stored_results(**overrides):
    stored = {
        "new_coverage": adapted_view(false_safe=2),
        "retention": adapted_view(false_safe=1),
    }
    stored.update(overrides)
    return stored


def test_reference_builder_freezes_adapted_results():
    ref = views.build_selection_reference_from_results(
        stored_results(),
        policy(max_new_coverage_false_safe_increase=0),
        "report-digest",
    )
    assert ref["schema"] == views.REFERENCE_SCHEMA
    assert ref["report_digest"] == "report-digest"
    assert ref["transfer"]["false_safe_count"] == 2
    assert ref["retention"]["false_safe_count"] == 1
    assert ref["transfer"]["accuracy"] == pytest.approx(28 / 30)
    assert ref["retention"]["min_class_recall"] == pytest.approx(0.8)
    assert ref["retention"]["both_pair_members_correct"] == pytest.approx(0.9)
    assert ref["retention"]["per_family"] == {
        "famA": {"false_safe_count": 1, "violation_count": 6}
    }


def test_reference_builder_from_results_allows_unknown_families():
    stored = stored_results(
        retention=adapted_view(families=None),
        new_coverage=adapted_view(families={}),
    )
    ref = views.build_selection_reference_from_results(
        stored, policy(), "report-digest"
    )
    assert ref["transfer"]["per_family"] == {}
    assert ref["retention"]["per_family"] == {}


def test_reference_builder_from_results_validates():
    with pytest.raises(ValueError, match="new_coverage"):
        views.build_selection_reference_from_results(
            {"retention": adapted_view()}, policy(), "d"
        )
    bad_counts = stored_results()
    bad_counts["retention"]["counts"]["false_safe"] = 99
    with pytest.raises(ValueError, match="false-safe"):
        views.build_selection_reference_from_results(bad_counts, policy(), "d")
    bad_family = stored_results(
        retention=adapted_view(
            families={"famA": {"false_safe": 9, "violations": 6}}
        ),
    )
    with pytest.raises(ValueError, match="famA"):
        views.build_selection_reference_from_results(bad_family, policy(), "d")
    with pytest.raises(ValueError, match="retention floor"):
        views.build_selection_reference_from_results(
            stored_results(), {"min_retention_accuracy": 2}, "d"
        )


def test_mc_recall_uses_insufficient_class():
    assert views.mc_recall_from_confusion(
        [[1, 0, 0], [0, 1, 0], [1, 0, 9]]
    ) == (pytest.approx(0.9))
    assert views.mc_recall_from_confusion(
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    ) == (1.0)
    with pytest.raises(ValueError, match="3x3"):
        views.mc_recall_from_confusion([[1]])
