"""Build transfer-review episodes from an external development catalog.

Experiment observations and authored targets belong in a local catalog outside
Git. The default catalog is a synthetic illustration of failed acceptance,
regression, completion, and an absent learning-rate intervention.
"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    validate_episode,
)
from siliconshield.research_loop.episodes.operator_episodes import episode_for
from siliconshield.schemas import (
    TRANSFER_CATALOG_SCHEMA,
    TRANSFER_EPISODES_SCHEMA,
)

CAMPAIGN_FAMILY = "authored-transfer-example"


def _examples():
    def observation(step, accuracy, nll):
        return {
            "step": step,
            "metrics": {"accuracy": accuracy, "nll": nll},
            "views": {
                "transfer": {"n": 20, "accuracy": accuracy, "nll": nll},
                "retention": {"n": 40, "accuracy": 0.9, "nll": 0.2},
            },
        }

    state = {
        "phase": "training",
        "systemd": "running",
        "step": 2,
        "execution_failed": False,
        "development_history": {
            "entries": [observation(1, 0.6, 0.7), observation(2, 0.8, 0.4)],
            "complete": True,
            "total": 2,
        },
        "acceptance": {
            "accepted": False,
            "reasons": ["false_safe_budget_exceeded"],
            "thresholds": {"max_worst_family_false_safe_rate": 0.1},
        },
        "selected_step": 2,
        "schedule": {"plateau_lr_reductions": [], "operator_interventions": []},
    }
    resources = {"gpus_free": 0, "reservation": "self"}
    scenarios = [
        (
            "transfer_best_so_far_failed_acceptance",
            copy.deepcopy(state),
            resources,
            "continue",
            "Development accuracy improved while NLL fell. Continue within budget, but acceptance has not passed.",
        )
    ]
    state["step"] = 3
    state["development_history"]["entries"].append(observation(3, 0.5, 0.9))
    state["development_history"]["total"] = 3
    scenarios.append(
        (
            "transfer_regression_stop",
            copy.deepcopy(state),
            resources,
            "stop",
            "Development accuracy fell while NLL rose. Stop and preserve the selected checkpoint. "
            "No execution failure was observed.",
        )
    )
    state.update(phase="training_complete", systemd="completed")
    scenarios.append(
        (
            "transfer_completion_unaccepted",
            copy.deepcopy(state),
            resources,
            "leave_unchanged",
            "Execution completed, but acceptance has not passed. Preserve the report without claiming promotion.",
        )
    )
    state["schedule"].update(lr_scale=1.0, stale=2)
    scenarios.append(
        (
            "transfer_plateau_no_intervention",
            copy.deepcopy(state),
            resources,
            "leave_unchanged",
            "No plateau reduction occurred. The schedule records no interventions. Leave the completed run unchanged.",
        )
    )
    return scenarios


SCENARIOS = _examples()


def load_catalog(path=None):
    """Read external observations or return clearly marked authored examples."""
    if path is None:
        return {
            "schema": TRANSFER_CATALOG_SCHEMA,
            "campaign_family": CAMPAIGN_FAMILY,
            "provenance": "authored illustration; not measured experiment results",
            "scenarios": [
                dict(
                    zip(
                        ("scenario", "state", "resources", "action", "reason"),
                        item,
                    )
                )
                for item in copy.deepcopy(SCENARIOS)
            ],
        }
    if not isinstance(path, (str, Path)):
        raise ValueError("Transfer catalog path must be a path")
    catalog = json.loads(Path(path).read_text())
    if not isinstance(catalog, dict):
        raise ValueError("Transfer catalog must be an object")
    if catalog.get("schema") != TRANSFER_CATALOG_SCHEMA:
        raise ValueError("Unknown transfer catalog schema")
    if (
        not isinstance(catalog.get("campaign_family"), str)
        or not catalog["campaign_family"].strip()
    ):
        raise ValueError("Transfer catalog needs a campaign family")
    scenarios = catalog.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Transfer catalog needs observations")
    for item in scenarios:
        if not isinstance(item, dict) or not {
            "scenario",
            "state",
            "resources",
            "action",
            "reason",
        } <= set(item):
            raise ValueError("Incomplete transfer observation")
    return catalog


def build_transfer(output, catalog=None):
    """Write development-only episodes, keeping a whole campaign in one lineage."""
    if not isinstance(output, (str, Path)):
        raise ValueError("Transfer output must be a path")
    source = load_catalog(catalog)
    family = source["campaign_family"]
    rows = []
    for index, item in enumerate(source["scenarios"]):
        row = episode_for(
            item["scenario"],
            item["state"],
            item["resources"],
            item["action"],
            item["reason"],
            family,
            index,
        )
        row["verification"].update(
            campaign_family=family,
            provenance=source.get("provenance", "external catalog"),
        )
        rows.append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "transfer.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    manifest = {
        "schema": TRANSFER_EPISODES_SCHEMA,
        "rows": len(rows),
        "campaign_family": family,
        "purpose": "Development review examples; not independent final-test evidence",
        "development_only": True,
        "actions": dict(Counter(r["target"]["action"] for r in rows)),
        "lineages": sorted({r["group_id"] for r in rows}),
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def load_transfer(path, allowed_prefix=None):
    """Validate the tool contract and common declared campaign lineage."""
    if not isinstance(path, (str, Path)):
        raise ValueError("Transfer path must be a path")
    if allowed_prefix is not None and not isinstance(allowed_prefix, str):
        raise ValueError("Transfer prefix must be a string")
    rows = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("No transfer observations")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Transfer observations must be objects")
    families = {
        row.get("verification", {}).get("campaign_family") for row in rows
    }
    if len(families) != 1 or not next(iter(families)):
        raise ValueError(
            "Transfer rows must declare one family and use campaign lineages"
        )
    family = next(iter(families))
    for row in rows:
        validate_episode(row)
        group = str(row.get("group_id", ""))
        if (group != family and not group.startswith(family + "-")) or (
            allowed_prefix and not group.startswith(allowed_prefix)
        ):
            raise ValueError(
                "Transfer rows must use campaign lineages: "
                + str(row.get("id"))
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build",
        required=True,
        help="Directory for development review episodes",
    )
    parser.add_argument(
        "--catalog",
        help="External catalog; omitted means authored illustrative examples",
    )
    args = parser.parse_args()
    print(json.dumps(build_transfer(args.build, args.catalog), indent=2))


if __name__ == "__main__":
    main()
