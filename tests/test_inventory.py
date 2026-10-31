"""Inventory feasibility: broad release versus narrow pilot."""

from siliconshield.training.laya.inventory import feasibility, summarize


def _row(origin, group, mechanism="locking"):
    return {
        "id": "%s-%s" % (origin, group),
        "origin": origin,
        "split_group_id": group,
        "mechanism_id": mechanism,
    }


def test_narrow_pilot_when_inventory_short():
    rows = [_row("auth", "g%d" % i) for i in range(10)]
    report = summarize(rows)
    assert report["train_groups"] == 10
    verdict = feasibility(
        {**report, "dev_groups": 2},
        {"train_groups": 80, "dev_groups": 24},
    )
    assert verdict["broad_release"] is False
    assert verdict["release_class"] == "narrow-pilot"


def test_broad_release_when_targets_met():
    rows = [_row("o%d" % (i % 5), "g%d" % i) for i in range(90)]
    report = summarize(rows)
    verdict = feasibility(
        {
            **report,
            "dev_groups": 30,
            "calibration_groups": 12,
            "test_groups": 24,
        },
        {"train_groups": 80, "dev_groups": 24},
    )
    assert verdict["broad_release"] is True


def test_partial_report_is_narrow_pilot():
    rows = [_row("o%d" % (i % 5), "g%d" % i) for i in range(90)]
    report = summarize(rows)
    verdict = feasibility({**report, "dev_groups": 30})
    assert verdict["broad_release"] is False
    assert verdict["release_class"] == "narrow-pilot"
