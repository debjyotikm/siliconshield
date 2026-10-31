"""Build authenticated operator episodes from an external scenario catalog.

Catalogs contain authored exercises or reviewed development observations. They
are research data and must remain outside the source repository. This module
does not infer successful interventions from a hypothetical target response.
"""

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from siliconshield.research_loop.episodes.contracts import (
    OPERATOR_SYSTEM,
    atomic_json,
    final_payload,
    validate_episode,
)
from siliconshield.research_loop.episodes.dataset import write_manifest
from siliconshield.research_loop.episodes.operator_assessment import assessment
from siliconshield.research_loop.episodes.operator_episodes import (
    episode_for,
    operator_coverage,
)


def episode(spec):
    """Render a catalog observation and its reviewed target into the tool contract."""
    state = copy.deepcopy(spec["state"])
    resources = copy.deepcopy(spec["resources"])
    row = episode_for(
        spec["scenario"],
        state,
        resources,
        spec["action"],
        spec["reason"],
        spec["family"],
        spec["index"],
    )
    run_id = row["summary"]["run_id"]
    state["run_id"] = run_id
    if (
        spec["action"] in ("resume", "finalize_retry")
        or "checkpoint_dir" in state
    ):
        state["checkpoint_dir"] = "/private/runs/" + run_id
    row["summary"] = state
    row["messages"][0]["content"] = OPERATOR_SYSTEM
    row["messages"][3]["content"] = json.dumps(state, sort_keys=True)
    args = {
        "run_id": run_id,
        "action": spec["action"],
        "reason": spec["reason"],
        "assessment": assessment(state),
    }
    if spec["action"] == "adjust_lr":
        args["lr"] = spec["lr"]
    if spec["action"] in ("resume", "finalize_retry"):
        args["checkpoint_dir"] = state["checkpoint_dir"]
    row["messages"][6]["tool_calls"][0]["function"]["arguments"] = args
    row["messages"][-1]["content"] = json.dumps(
        final_payload(spec["action"], run_id=run_id), sort_keys=True
    )
    row["target"] = {k: v for k, v in args.items() if k != "run_id"}
    row["target"]["require_assessment"] = True
    if "acceptable_actions" in spec:
        row["target"]["acceptable_actions"] = spec["acceptable_actions"]
    row["verification"].update(
        kind="authored_fixture",
        origin=spec.get("origin", "authored_policy_exercise"),
        review="deterministic_contract_checks; not a live intervention outcome",
    )
    for name in ("source_sha256", "decision_step", "counterfactual_of"):
        if name in spec:
            row["verification"][name] = spec[name]
    row["split"] = spec["split"]
    validate_episode(row)
    return row


def fixture_verify(row):
    """Exercise the actual replay dispatcher and grader with the authored trace."""
    from siliconshield.research_loop.controller.replay import execute, grade

    class TraceClient:
        model = "authored-curriculum-check"
        endpoint = "http://fixture.invalid"

        def complete(self, messages, tools=()):
            return copy.deepcopy(row["messages"][len(messages)])

    with tempfile.TemporaryDirectory() as output:
        result = grade(row, execute(row, TraceClient(), output))
    if not result["verified"]:
        raise ValueError(
            "Authored episode fails replay: %s %s" % (row["id"], result)
        )


def build(catalog, output, replay_train=None):
    """Validate complete situations, isolate families, and sign the dataset."""
    if not os.environ.get("SILICONSHIELD_DATASET_KEY"):
        raise ValueError(
            "Set SILICONSHIELD_DATASET_KEY before building a training dataset"
        )
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh dataset directory")
    specs = [
        json.loads(line)
        for line in Path(catalog).read_text().splitlines()
        if line.strip()
    ]
    splits = {s: [] for s in ("train", "dev", "test")}
    groups, seen, ids = {}, {}, set()
    for spec in specs:
        row = episode(spec)
        split, group = row["split"], row["group_id"]
        if split not in splits or (group in groups and groups[group] != split):
            raise ValueError("Scenario family crosses dataset splits")
        if row["id"] in ids:
            raise ValueError("Duplicate episode ID")
        ids.add(row["id"])
        normalized = copy.deepcopy(row["summary"])
        normalized.pop("run_id")
        if "checkpoint_dir" in normalized:
            normalized["checkpoint_dir"] = "<owned-checkpoint>"
        key = json.dumps(
            {"state": normalized, "resources": row["resources"]}, sort_keys=True
        )
        if key in seen:
            raise ValueError(
                "Repeated observation; numeric paraphrases do not add situations"
            )
        seen[key] = split
        groups[group] = split
        fixture_verify(row)
        splits[split].append(row)
    if replay_train:
        # Only an authenticated training partition is eligible for retention.
        from siliconshield.training.glimmer.train import load_rows

        for row in load_rows(
            replay_train, allowed_splits=("train",), require_signed=True
        ):
            if row["group_id"] in groups and groups[row["group_id"]] != "train":
                raise ValueError("Retention data crosses development lineage")
            if row["id"] in ids:
                raise ValueError("Duplicate retention episode")
            ids.add(row["id"])
            groups[row["group_id"]] = "train"
            splits["train"].append(row)
    from siliconshield.training.glimmer.train import dev_subset

    coverage = operator_coverage(
        splits["train"], splits["dev"], dev_subset(splits["dev"], 36)
    )
    if not coverage["coverage_passed"]:
        raise ValueError(
            "Operator coverage incomplete: " + "; ".join(coverage["issues"])
        )
    output.mkdir(parents=True)
    for split, rows in splits.items():
        (output / (split + ".jsonl")).write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
        )
    manifest = {
        "schema": "siliconshield-operator-curriculum-v1",
        "groups": groups,
        "rows": {s: len(rows) for s, rows in splits.items()},
        "catalog_sha256": hashlib.sha256(
            Path(catalog).read_bytes()
        ).hexdigest(),
        "actions": {
            s: dict(Counter(r["target"]["action"] for r in rows))
            for s, rows in splits.items()
        },
        "origins": {
            s: dict(
                Counter(
                    r["verification"].get("origin", r["verification"]["kind"])
                    for r in rows
                )
            )
            for s, rows in splits.items()
        },
        "independence_unit": "scenario family; parameter variants are not independent situations",
        "scope": "Authored operator exercises; no demonstrated intervention outcomes or final-test results",
    }
    write_manifest(output, manifest)
    atomic_json(output / "coverage.json", coverage)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--replay-train",
        help="Optional authenticated training partition for retention",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.catalog, args.output, args.replay_train), indent=2
        )
    )


if __name__ == "__main__":
    main()
