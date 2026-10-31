"""Run a local, bounded training plan using the Glimmer operator.

Recipes and promotion thresholds belong in the local campaign configuration.
A TRAIN-only fitting result can qualify a recipe for development evaluation;
it is never evidence of generalization.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

from siliconshield.research_loop.episodes.contracts import atomic_json


def fitting_plan(experiment_prefix, base_overrides=None, *, arms):
    plan = []
    for index, arm in enumerate(arms):
        overrides = {
            **(base_overrides or {}),
            **copy.deepcopy(arm["overrides"]),
        }
        if not overrides.get("limit_train_rows"):
            raise ValueError(
                "Fitting plans require a positive TRAIN-only row limit"
            )
        plan.append(
            {
                "experiment_id": "%s-fit-%02d-%s"
                % (experiment_prefix, index, arm["name"]),
                "arm": arm["name"],
                "overrides": overrides,
                "split_access": ["train"],
            }
        )
    return plan


def assess_fitting(results, min_gain=0.05):
    assessed = []
    for result in results:
        initial, final = (
            result["initial_train_accuracy"],
            result["train_accuracy"],
        )
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1
            for v in (initial, final)
        ):
            raise ValueError("Fitting requires finite measured accuracies")
        gain = final - initial
        assessed.append(
            {**result, "gain": gain, "promotable": gain >= min_gain}
        )
    return {
        "arms": assessed,
        "promotable": [a["arm"] for a in assessed if a["promotable"]],
        "scope": "TRAIN-only fitting; development evaluation remains necessary",
    }


def run_campaign(config, run=None):
    from siliconshield.research_loop.execution.glimmer_operator import (
        run_operator,
    )

    run = run or run_operator
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    plan = fitting_plan(
        config["experiment_prefix"],
        config["fitting_common"],
        arms=config["arms"],
    )
    results = []
    started = time.time()

    def execute(spec):
        operator_config = {
            **config["operator"],
            "profile": config["profile"],
            "experiment_id": spec["experiment_id"],
            "overrides": spec["overrides"],
            "output": str(root / spec["experiment_id"]),
        }
        while time.time() - started < config["timeout_seconds"]:
            outcome = run(operator_config)
            if outcome["manifest"].get("terminal") == "complete":
                return json.loads(
                    (
                        Path(outcome["manifest"]["output"]) / "report.json"
                    ).read_text()
                )
            status = json.loads(Path(config["profile"]).read_text())
            execution_path = (
                Path(status["runs"]) / spec["experiment_id"] / "execution.json"
            )
            if (
                execution_path.exists()
                and json.loads(execution_path.read_text()).get("phase")
                == "failed"
            ):
                raise RuntimeError(
                    "Owned job failed; inspect its execution receipt before continuing"
                )
        raise TimeoutError(
            "Campaign observation budget exhausted; owned jobs retain their own timeout"
        )

    for spec in plan:
        report = execute(spec)
        if report.get("evaluation_role") != "train_fitting":
            raise ValueError("Expected a TRAIN-only fitting report")
        results.append(
            {
                "arm": spec["arm"],
                "overrides": spec["overrides"],
                "train_accuracy": report["train_accuracy"],
                "initial_train_accuracy": report["initial_train_accuracy"],
                "nll": report["train"]["nll"],
                "selected_checkpoint": report["selected_checkpoint"],
            }
        )
        atomic_json(
            root / "campaign-status.json",
            {"phase": "fitting", "results": results},
        )
    assessment = assess_fitting(results, config.get("min_fit_gain", 0.05))
    eligible = [
        r
        for r in assessment["arms"]
        if r["promotable"] and not r["overrides"].get("disable_adapters")
    ]
    if not eligible:
        atomic_json(
            root / "campaign-status.json",
            {"phase": "no_qualified_recipe", "assessment": assessment},
        )
        return assessment
    chosen = min(eligible, key=lambda r: (-r["train_accuracy"], r["nll"]))
    overrides = {
        k: v
        for k, v in chosen["overrides"].items()
        if k not in ("limit_train_rows", "no_development_in_fitting")
    }
    overrides.update(config["development_overrides"])
    development_spec = {
        "experiment_id": config["experiment_prefix"] + "-development",
        "overrides": overrides,
    }
    atomic_json(
        root / "campaign-status.json",
        {"phase": "development", "chosen": chosen, "assessment": assessment},
    )
    report = execute(development_spec)
    atomic_json(
        root / "campaign-status.json",
        {
            "phase": "complete",
            "chosen": chosen,
            "assessment": assessment,
            "development": report["development"],
            "acceptance": report["development_acceptance"],
            "test_evaluated": False,
        },
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Local campaign JSON")
    args = parser.parse_args()
    run_campaign(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
