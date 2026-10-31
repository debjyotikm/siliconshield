"""Known answers for weighted decisions and lineage-clustered uncertainty."""

import copy

import numpy as np
import pytest

from siliconshield.training.laya.security_reporting import (
    LABELS,
    evaluation_report,
)


def fixture():
    rows = []
    for group in range(2):
        for label in range(3):
            rows.append(
                {
                    "id": str(len(rows)),
                    "split_group_id": "g" + str(group),
                    "pair_id": "pair" + str(group),
                    "source": {"name": "source-a"},
                    "mechanism_id": "access",
                    "expected": {"security": LABELS[label]},
                }
            )
    p = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    return rows, np.log(p), np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0])


def test_weighted_and_unweighted_metrics_have_known_answers():
    rows, logits, weights = fixture()
    report = evaluation_report(
        rows, logits, weights, {"source-a": 0.8, "source-b": 0.2}, 128, 7
    )
    raw, weighted = (
        report["overall"][name]["metrics"]
        for name in ("unweighted", "weighted")
    )
    assert raw["accuracy"] == pytest.approx(5 / 6)
    assert weighted["accuracy"] == pytest.approx(11 / 12)
    assert weighted["macro_f1"] == pytest.approx((8 / 9 + 6 / 7 + 1) / 3)
    assert raw["false_safe_rate"] == 0.5 and weighted["false_safe_rate"] == 0.25
    assert raw["pair_accuracy"] == 0.5 and weighted["pair_accuracy"] == 0.75
    assert (
        raw["missing_context_recall"] == weighted["missing_context_recall"] == 1
    )
    assert raw["nll"] == pytest.approx(-(5 * np.log(0.8) + np.log(0.1)) / 6)
    assert weighted["nll"] == pytest.approx(
        -(11 * np.log(0.8) + np.log(0.1)) / 12
    )
    assert report["missing_sources"] == ["source-b"]
    assert not report["full_configured_mixture_estimable"]
    assert report["overall"]["weighting"]["observed_source_mass"] == {
        "source-a": 1.0
    }
    interval = report["overall"]["unweighted"]["intervals"]["intervals"][
        "accuracy"
    ]
    assert interval["lower"] == pytest.approx(2 / 3)
    assert interval["upper"] == 1


def test_related_rewrites_do_not_create_more_bootstrap_clusters():
    rows, logits, weights = fixture()
    first = evaluation_report(rows, logits, weights, {"source-a": 1}, 128, 3)
    copies = []
    for rewrite in range(4):
        for row in rows:
            copy_row = copy.deepcopy(row)
            copy_row["id"] += "-rewrite-" + str(rewrite)
            copy_row["pair_id"] += "-rewrite-" + str(rewrite)
            copies.append(copy_row)
    second = evaluation_report(
        copies,
        np.tile(logits, (4, 1)),
        np.tile(weights, 4),
        {"source-a": 1},
        128,
        3,
    )
    assert second["overall"]["support"]["split_groups"] == 2
    for view in ("weighted", "unweighted"):
        a, b = (
            report["overall"][view]["intervals"] for report in (first, second)
        )
        assert a["source_cluster_counts"] == b["source_cluster_counts"]
        for metric in a["intervals"]:
            assert a["intervals"][metric] == pytest.approx(
                b["intervals"][metric]
            )


def test_singleton_source_does_not_get_a_false_precision_interval():
    rows, logits, weights = fixture()
    for row in rows[3:]:
        row["source"]["name"] = "source-b"
        row["mechanism_id"] = "another-access"
    report = evaluation_report(
        rows, logits, weights, {"source-a": 0.5, "source-b": 0.5}, 64
    )
    assert report["full_configured_mixture_estimable"]
    assert (
        report["overall"]["unweighted"]["intervals"]["status"] == "unavailable"
    )
    assert (
        report["per_source"]["source-b"]["weighted"]["intervals"]["reason"]
        == "fewer_than_two_clusters_in_source"
    )
    assert (
        report["per_source_mechanism"]["source-b"]["another-access"][
            "unweighted"
        ]["metrics"]["accuracy"]
        == 1
    )
    assert (
        report["per_source"]["source-a"]["unweighted"]["metrics"][
            "false_safe_rate"
        ]
        == 1
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_weights_rejected(bad):
    rows, logits, weights = fixture()
    weights[0] = bad
    with pytest.raises(ValueError):
        evaluation_report(rows, logits, weights, {"source-a": 1}, 0)


def test_misaligned_arrays_and_mixed_ownership_rejected():
    rows, logits, weights = fixture()
    with pytest.raises(ValueError, match="misaligned"):
        evaluation_report(rows, logits[:-1], weights, {"source-a": 1}, 0)
    rows[1]["source"]["name"] = "source-b"
    with pytest.raises(ValueError, match="spans sources"):
        evaluation_report(
            rows, logits, weights, {"source-a": 0.5, "source-b": 0.5}, 0
        )
    rows, logits, weights = fixture()
    rows[3]["pair_id"] = rows[0]["pair_id"]
    with pytest.raises(ValueError, match="pair crosses"):
        evaluation_report(rows, logits, weights, {"source-a": 1}, 0)


def test_probability_and_logit_paths_agree_and_missing_class_is_explicit():
    rows, logits, weights = fixture()
    left = evaluation_report(rows, logits, weights, {"source-a": 1}, 0)
    right = evaluation_report(
        rows,
        np.exp(logits),
        weights,
        {"source-a": 1},
        0,
        from_probabilities=True,
    )
    for mode in ("weighted", "unweighted"):
        assert left["overall"][mode]["metrics"] == pytest.approx(
            right["overall"][mode]["metrics"]
        )
    chosen = [0, 2, 3, 5]
    partial = evaluation_report(
        [rows[i] for i in chosen],
        logits[chosen],
        weights[chosen],
        {"source-a": 1},
        0,
    )
    assert (
        partial["overall"]["unweighted"]["metrics"]["false_safe_rate"] is None
    )
    assert partial["overall"]["unweighted"]["metrics"]["pair_accuracy"] is None


def test_trainer_attaches_reporting_without_changing_selection(monkeypatch):
    pytest.importorskip(
        "torch"
    )  # trainer import needs torch; rest is NumPy-only
    from argparse import Namespace

    from siliconshield.training.laya import security_train as train

    rows, logits, _ = fixture()
    for row in rows:
        row["base_design_id"] = row["split_group_id"]
    monkeypatch.setattr(train, "SOURCE_MASS", {"source-a": 1.0})
    result = train.metrics(logits, [0, 1, 2, 0, 1, 2])
    before = train.selection_key(result)
    train.attach_reporting(
        result, rows, logits, Namespace(report_bootstrap=0, seed=9), rank=0
    )
    assert result["reporting"]["overall"]["support"]["decision_rows"] == 6
    assert train.selection_key(result) == before
