"""Fitting-plan and assessment input-shape tests: synthetic fixtures."""

from siliconshield.research_loop.execution import strategy


def test_fitting_plan_rejects_malformed_arms():
    import pytest

    good = {"name": "a", "overrides": {"limit_train_rows": 8}}
    assert strategy.fitting_plan("e", arms=[good])[0]["arm"] == "a"
    for arms in (
        ["x"],
        [{"overrides": {"limit_train_rows": 8}}],
        [{"name": "a", "overrides": {"limit_train_rows": -5}}],
        [{"name": "a", "overrides": {"limit_train_rows": True}}],
        [{"name": "a", "overrides": []}],
        [],
    ):
        with pytest.raises(ValueError):
            strategy.fitting_plan("e", arms=arms)
    with pytest.raises(ValueError):
        strategy.fitting_plan("", arms=[good])
    with pytest.raises(ValueError):
        strategy.fitting_plan("e", "common", arms=[good])


def test_assess_fitting_rejects_malformed_results():
    import pytest

    good = {
        "arm": "a",
        "initial_train_accuracy": 0.3,
        "train_accuracy": 0.9,
        "nll": 0.4,
    }
    assert strategy.assess_fitting([good])["promotable"] == ["a"]
    with pytest.raises(ValueError, match="gain threshold"):
        strategy.assess_fitting([good], min_gain="x")
    with pytest.raises(ValueError, match="result list"):
        strategy.assess_fitting({"arm": "a"})
    for bad in (
        ["x"],
        [dict(good, nll="bad")],
        [dict(good, train_accuracy=2.0)],
        [{k: v for k, v in good.items() if k != "nll"}],
    ):
        with pytest.raises(ValueError):
            strategy.assess_fitting(bad)


def test_inspect_record_rejects_malformed_stdout():
    import json

    import pytest

    from siliconshield.research_loop.execution import campaign

    good = {
        "Config": {"Labels": {"laya.glimmer.run": "r1"}},
        "State": {"Running": False, "ExitCode": 0},
    }
    assert campaign._inspect_record(json.dumps([good])) == good
    for stdout in (
        "[]",
        "[1]",
        "{}",
        "not json",
        json.dumps([{"Config": [], "State": {}}]),
        json.dumps([{"Config": {}, "State": []}]),
        json.dumps([{"Config": {"Labels": ["x"]}, "State": {"Running": True}}]),
    ):
        with pytest.raises(RuntimeError, match="container record"):
            campaign._inspect_record(stdout)
