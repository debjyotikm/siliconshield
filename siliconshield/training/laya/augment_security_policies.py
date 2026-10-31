"""Add solver-checked training examples with changed policies and the same RTL."""

import argparse
import copy
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

from siliconshield.training.laya import build_security_data as data
from siliconshield.training.laya import expand_security_data as semantic

# Version of the augmented training manifest written by main(). Distinct from
# the base/expanded dataset manifests; the held-out split bytes are preserved.
AUGMENTED_TRAIN_MANIFEST_SCHEMA = 3


def augment_group(group, yosys, verification):
    by_label = {row["expected"]["security"]: row for row in group}
    if len(group) != 3 or set(by_label) != set(data.LABELS):
        raise ValueError(
            "policy augmentation requires exactly one complete triplet"
        )
    sample = by_label[data.LABELS[0]]
    case = semantic.scenario(
        sample["source"]["seed"], sample["source"]["case_index"]
    )
    variant = sample["variant"]
    alternate = case["requirement"].replace(case["policy"], case["bad_fill"])
    assertion = (
        semantic.pipeline_check(case["bad_fill"])
        if case["style"] == "pipeline"
        else semantic.exact_check(case["bad_fill"])
    )
    _, requirement, check = semantic.rename_rendered(
        case["code"], alternate, assertion, variant
    )
    _, original_requirement, _ = semantic.rename_rendered(
        case["code"], case["requirement"], case["assertion"], variant
    )
    new_rows, proofs = [], {}
    for old_label, new_label in (
        (data.LABELS[0], data.LABELS[1]),
        (data.LABELS[1], data.LABELS[0]),
    ):
        row = by_label[old_label]
        prefix, code = row["state"].split("RTL\n", 1)
        if (
            prefix.split("ASSUMPTIONS\n", 1)[0]
            != "REQUIREMENT\n" + original_requirement + "\n"
        ):
            raise ValueError(
                "requirement provenance differs from generated scenario"
            )
        proof = data.verify(code, check, yosys, verification, "rst_n")
        if not proof.get("accepted") or proof.get("verdict") != new_label:
            return [], {
                "pair_id": sample["pair_id"],
                "expected": new_label,
                "evidence": proof,
            }
        proofs[new_label] = proof
    hole = by_label[data.LABELS[2]]["validation"]["hole_name"]
    opaque = by_label[data.LABELS[2]]["state"].split("RTL\n", 1)[1]
    for label, fill in (
        (data.LABELS[0], case["good_fill"]),
        (data.LABELS[1], case["bad_fill"]),
    ):
        code = by_label[label]["state"].split("RTL\n", 1)[1]
        semantic.completion_check(
            opaque,
            code,
            hole,
            semantic.rename_fill(case["code"], fill, variant),
        )
    for old_label, new_label in zip(
        data.LABELS, (data.LABELS[1], data.LABELS[0], data.LABELS[2])
    ):
        original = by_label[old_label]
        row = copy.deepcopy(original)
        prefix, code = original["state"].split("RTL\n", 1)
        row["state"] = (
            "REQUIREMENT\n"
            + requirement
            + "\nASSUMPTIONS\n"
            + prefix.split("ASSUMPTIONS\n", 1)[1]
            + "RTL\n"
            + code
        )
        row["id"] = data.digest(row["state"])[:24]
        row["pair_id"] += ":alternate-policy"
        row["rewrite_group_id"] += ":alternate-policy"
        row["expected"]["security"] = new_label
        row["source"]["policy_augmentation"] = "same_rtl_alternate_requirement"
        row["policy_counterpart_id"] = original["id"]
        if new_label != data.LABELS[2]:
            row["validation"] = dict(
                proofs[new_label],
                rendered_code_sha256=data.digest(code),
                rendered_assertion_sha256=data.digest(check),
            )
        else:
            row["validation"] = dict(
                original["validation"],
                satisfying_completion=proofs[data.LABELS[0]]["evidence_id"],
                violating_completion=proofs[data.LABELS[1]]["evidence_id"],
                good_fill_sha256=original["validation"]["bad_fill_sha256"],
                bad_fill_sha256=original["validation"]["good_fill_sha256"],
                rendered_assertion_sha256=data.digest(check),
            )
        new_rows.append(row)
    return new_rows, None


def copy_held_out_splits(
    base_data: Union[str, os.PathLike[str]],
    output: Union[str, os.PathLike[str]],
) -> dict[str, dict[str, str]]:
    """Copy held-out splits byte-for-byte; reject any divergence.

    Held-out populations must reach the augmented output unchanged. A hash
    mismatch means the copy is corrupt or the base moved mid-run: the output
    directory must be discarded, never trained on. Returns the per-split
    base/output hashes. Raises ValueError naming the offending split.
    """
    held_out: dict[str, dict[str, str]] = {}
    for split in ("dev", "calibration", "test", "external"):
        source = Path(base_data) / (split + ".jsonl")
        if source.exists():
            target = Path(output) / source.name
            shutil.copyfile(source, target)
            held_out[split] = {
                "base_sha256": data.file_hash(source),
                "output_sha256": data.file_hash(target),
            }
            if (
                held_out[split]["base_sha256"]
                != held_out[split]["output_sha256"]
            ):
                raise ValueError(
                    "held-out split %s changed during copy; discard %s and rerun"
                    % (split, output)
                )
    return held_out


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
        help="New directory for the augmented dataset",
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
        "--workers",
        type=int,
        default=6,
        help="Number of parallel data-preparation workers",
    )
    args = parser.parse_args()
    if (
        args.output.resolve() == args.base_data.resolve()
        or (args.output / "manifest.json").exists()
    ):
        raise ValueError("output must be new and separate from base data")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in (args.base_data / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    groups = defaultdict(list)
    for row in rows:
        if row["source"]["name"] == "authored-semantic":
            if row.get("split") != "train" or row["source"].get(
                "policy_augmentation"
            ):
                raise ValueError("requires original TRAIN scenarios")
            groups[row["pair_id"]].append(row)
    added, rejected = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:

        def build(group):
            return augment_group(group, args.yosys, args.verification)

        for index, (new_rows, failure) in enumerate(
            pool.map(build, groups.values()), 1
        ):
            if failure:
                rejected.append(failure)
            else:
                added.extend(new_rows)
            if index % 100 == 0:
                print(
                    json.dumps(
                        {
                            "processed_triplets": index,
                            "added_rows": len(added),
                            "rejected_triplets": len(rejected),
                        }
                    ),
                    flush=True,
                )
    states, ids = semantic.index_rows(rows)
    unique, duplicates = [], []
    for start in range(0, len(added), 3):
        group = added[start : start + 3]
        if semantic.duplicate_group(group, states, ids):
            duplicates.append(group[0]["pair_id"])
            continue
        semantic.remember_group(group, states, ids)
        unique.extend(group)
    added = unique
    if not added:
        raise ValueError("no distinct verified policy reversals")
    data.atomic_bytes(
        args.output / "train.jsonl",
        "".join(json.dumps(row) + "\n" for row in rows + added).encode(),
    )
    held_out = copy_held_out_splits(args.base_data, args.output)
    manifest = {
        "schema": AUGMENTED_TRAIN_MANIFEST_SCHEMA,
        "train_only": True,
        "base_train_sha256": data.file_hash(args.base_data / "train.jsonl"),
        "train_sha256": data.file_hash(args.output / "train.jsonl"),
        "train_rows": len(rows) + len(added),
        "added_rows": len(added),
        "accepted_triplets": len(added) // 3,
        "rejected_triplets": rejected,
        "duplicate_triplets": duplicates,
        "base_designs": len({row["base_design_id"] for row in rows}),
        "note": "Adds semantic supervision, not independent reference designs; held-out bytes unchanged.",
        "labels": dict(
            Counter(row["expected"]["security"] for row in rows + added)
        ),
        "held_out": held_out,
        "builder_sha256": data.file_hash(Path(__file__)),
    }
    data.atomic_json(args.output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                key: value
                for key, value in manifest.items()
                if key != "rejected_triplets"
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
