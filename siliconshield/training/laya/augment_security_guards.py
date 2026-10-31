"""Add conditional hold requirements to existing training examples."""

import argparse
import copy
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from siliconshield.training.laya import build_security_data as data
from siliconshield.training.laya import expand_security_data as semantic


def guard_group(group, yosys, verification):
    by_label = {row["expected"]["security"]: row for row in group}
    if len(group) != 3 or set(by_label) != set(data.LABELS):
        raise ValueError("guard augmentation requires one complete triplet")
    sample = by_label[data.LABELS[0]]
    if sample["variant"] != 0:
        raise ValueError("guard builder operates on canonical variants")
    case = semantic.scenario(
        sample["source"]["seed"], sample["source"]["case_index"]
    )
    if case["style"] == "pipeline":
        raise ValueError(
            "pipeline authorization needs a separate delayed property"
        )
    condition = "!(%s)" % case["policy"]
    requirement = (
        "If (%s) held before a rising edge, value must retain its previous value after that edge. "
        "This requirement imposes no constraint on value when that condition is false."
        % condition
    )
    assertion = data.hold_check("value", condition)
    proofs = {}
    for label in data.LABELS[:2]:
        code = by_label[label]["state"].split("RTL\n", 1)[1]
        proof = data.verify(code, assertion, yosys, verification, "rst_n")
        if not proof.get("accepted") or proof.get("verdict") != label:
            return [], {
                "pair_id": sample["pair_id"],
                "expected": label,
                "evidence": proof,
            }
        proofs[label] = proof
    unknown = by_label[data.LABELS[2]]
    opaque = unknown["state"].split("RTL\n", 1)[1]
    for label, fill in (
        (data.LABELS[0], case["good_fill"]),
        (data.LABELS[1], case["bad_fill"]),
    ):
        semantic.completion_check(
            opaque,
            by_label[label]["state"].split("RTL\n", 1)[1],
            unknown["validation"]["hole_name"],
            fill,
        )
    generated = []
    for label in data.LABELS:
        original = by_label[label]
        row = copy.deepcopy(original)
        prefix, code = row["state"].split("RTL\n", 1)
        row["state"] = (
            "REQUIREMENT\n"
            + requirement
            + "\nASSUMPTIONS\n"
            + prefix.split("ASSUMPTIONS\n", 1)[1]
            + "RTL\n"
            + code
        )
        row["id"] = data.digest(row["state"])[:24]
        row["pair_id"] += ":hold-policy"
        row["rewrite_group_id"] += ":hold-policy"
        # Same mechanism and design ownership: this is another contract, not a
        # newly independent implementation or a reason to increase source mass.
        row["source"]["contract_form"] = "conditional_hold"
        row["contract_parent_id"] = original["id"]
        if label != data.LABELS[2]:
            row["validation"] = dict(
                proofs[label],
                rendered_code_sha256=data.digest(code),
                rendered_assertion_sha256=data.digest(assertion),
            )
        else:
            row["validation"] = dict(
                unknown["validation"],
                satisfying_completion=proofs[data.LABELS[0]]["evidence_id"],
                violating_completion=proofs[data.LABELS[1]]["evidence_id"],
                rendered_assertion_sha256=data.digest(assertion),
            )
        generated.append(row)
    return generated, None


def shuffled_order(keys, seed):
    """Deterministic group ordering controlled entirely by the supplied seed."""
    chosen = sorted(keys)
    random.Random(seed).shuffle(chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-data",
        type=Path,
        required=True,
        help="Directory containing the existing dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file for the added training examples",
    )
    parser.add_argument(
        "--verification",
        type=Path,
        required=True,
        help="Directory for solver inputs and results",
    )
    parser.add_argument(
        "--yosys", required=True, help="Path to the Yosys executable"
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=0,
        help="Cases to generate (required for real runs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed (illustrative default; real runs pass their recipe seed)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel data-preparation workers",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("guard output already exists")
    rows = [
        json.loads(line)
        for line in (args.base_data / "train.jsonl").read_text().splitlines()
    ]
    groups = defaultdict(list)
    for row in rows:
        if (
            row["source"]["name"] == "authored-semantic"
            and row["variant"] == 0
            and row["source"]["template"] != "pipeline"
        ):
            if row.get("split") != "train":
                raise ValueError("guard augmentation is TRAIN only")
            groups[row["pair_id"]].append(row)
    chosen = shuffled_order(groups, args.seed)
    generated, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:

        def build(key):
            return guard_group(groups[key], args.yosys, args.verification)

        for result, failure in pool.map(build, chosen[: args.cases]):
            generated.extend(result)
            if failure:
                failures.append(failure)
    if not generated:
        raise ValueError("no verified conditional-hold groups")
    data.atomic_bytes(
        args.output,
        "".join(json.dumps(row) + "\n" for row in generated).encode(),
    )
    data.atomic_json(
        args.output.with_suffix(".manifest.json"),
        {
            "train_only": True,
            "rows": len(generated),
            "accepted_groups": len(generated) // 3,
            "failures": failures,
            "base_sha256": data.file_hash(args.base_data / "train.jsonl"),
            "builder_sha256": data.file_hash(Path(__file__)),
            "output_sha256": data.file_hash(args.output),
        },
    )
    print(
        json.dumps(
            {
                "rows": len(generated),
                "accepted_groups": len(generated) // 3,
                "rejected_groups": len(failures),
            }
        )
    )


if __name__ == "__main__":
    main()
