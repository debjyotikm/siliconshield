"""Validate a local source-contract catalog and materialize RTL decisions.

Catalogs contain RTL and experiment definitions and belong outside Git. Every
complete implementation is compiled and checked by the solver. Missing-context
labels require two byte-exact completions with opposite verified verdicts.
"""

import argparse
import fcntl
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from siliconshield.training.laya import build_security_data as data

COUNT_DEFINITIONS = {
    "decision_rows": "Number of labeled requirement/RTL decisions.",
    "source_files": "Distinct (source.name, source.path) pairs.",
    "source_records": "Distinct (source.name, source.case) catalog records; a file may supply several.",
    "base_design_ids": "Distinct (source.name, base_design_id) extracted design IDs; may include cone or parameter specializations.",
    "split_groups": "Distinct split_group_id values; related records remain in one partition.",
    "unique_rtl_bodies": "Distinct exact RTL strings, including rewrites and mutations; not independent source designs.",
}


def population_counts(rows):
    """Name each counting unit explicitly instead of overloading 'design'."""
    return {
        "decision_rows": len(rows),
        "source_files": len(
            {(r["source"]["name"], r["source"]["path"]) for r in rows}
        ),
        "source_records": len(
            {(r["source"]["name"], r["source"]["case"]) for r in rows}
        ),
        "base_design_ids": len(
            {(r["source"]["name"], r["base_design_id"]) for r in rows}
        ),
        "split_groups": len({r["split_group_id"] for r in rows}),
        "unique_rtl_bodies": len(
            {r["state"].split("RTL\n", 1)[1] for r in rows}
        ),
    }


def exact_antecedent(assertion):
    """Keep the sampled-time guard intact when asking whether it is reachable."""
    if assertion.count("assert(") != 1:
        raise ValueError("expected one assertion")
    start = assertion.index("assert(")
    depth = 1
    for index in range(start + len("assert("), len(assertion)):
        depth += (assertion[index] == "(") - (assertion[index] == ")")
        if depth == 0:
            return assertion[:start] + "assert(1'b0)" + assertion[index + 1 :]
    raise ValueError("unbalanced assertion expression")


def checked_rows(case, yosys, iverilog, directory):
    # Placeholder names must not change when only the requirement changes.
    # Keep record identity separate from the identity used for rendering.
    rendered_case = dict(case, uid=case.get("base_uid", case["uid"]))
    rows = data.process_case(rendered_case, yosys, directory)
    for variant in data.VARIANTS:
        rendered = data.render_variant(rendered_case, variant)
        reachable = data.verify(
            rendered["good"],
            exact_antecedent(rendered["assertion"]),
            yosys,
            directory,
            case["reset"],
        )
        if (
            not reachable.get("accepted")
            or reachable.get("verdict") != data.LABELS[1]
        ):
            raise ValueError(
                "assertion antecedent is unreachable or unproved: "
                + case["uid"]
            )
        for row in rows:
            if row["variant"] == variant:
                row["validation"]["exact_antecedent_evidence"] = reachable[
                    "evidence_id"
                ]
    for row in rows:
        row["pair_id"] = case["uid"] + ":" + str(row["variant"])
        row["rewrite_group_id"] = case["uid"]
        row["base_design_id"] = case.get("base_uid", case["uid"])
        row["split_group_id"] = case.get("split_group", case["family"])
        row["mechanism_id"] = case.get(
            "mechanism", case["topic"] + ":" + case["kind"]
        )
        row["property_origin"] = case.get(
            "property_origin", "explicit_local_contract"
        )
        row["policy"] = case.get("policy", "original")
        row["validation"]["scope"] = (
            "Only the supplied contract under the supplied execution assumptions."
        )
        # Compile opaque snippets too: the omission is an undriven declared wire,
        # not a syntax error that could serve as the class label.
        compile_id = hashlib.sha256(
            row["state"].split("RTL\n", 1)[1].encode()
        ).hexdigest()
        lock_dir = Path(directory) / "compile-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / compile_id).open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            compiled = data.compile_rtl(
                row["state"].split("RTL\n", 1)[1],
                compile_id,
                iverilog,
                directory,
            )
        row["validation"]["compile_evidence"] = compiled["evidence_id"]
    return rows


def materialize(cases, output, verification, yosys, iverilog, workers=8):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("refusing to overwrite a materialized dataset")
    splits = defaultdict(list)
    admitted, rejected, seen = [], [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(checked_rows, case, yosys, iverilog, verification)
            for case in cases
        ]
        for index, (case, future) in enumerate(zip(cases, futures)):
            try:
                rows = future.result()
            except Exception as exc:
                # Isolate per-case solver failures: one bad case is recorded
                # as rejected instead of killing the whole batch.
                rejected.append({"uid": case["uid"], "reason": str(exc)})
                continue
            # Deduplicate complete triplets. Never leave a partial label group.
            for variant in data.VARIANTS:
                triplet = [r for r in rows if r["variant"] == variant]
                keys = [r["id"] for r in triplet]
                if any(key in seen for key in keys):
                    if any(
                        seen.get(key, case["split"]) != case["split"]
                        for key in keys
                    ):
                        raise ValueError(
                            "duplicate input crosses dataset partitions"
                        )
                    rejected.append(
                        {
                            "uid": case["uid"],
                            "variant": variant,
                            "reason": "duplicate_input_triplet",
                        }
                    )
                    continue
                seen.update({key: case["split"] for key in keys})
                splits[case["split"]].extend(triplet)
            admitted.append(case["uid"])
            if index % 50 == 0:
                print(
                    json.dumps(
                        {
                            "checked": index + 1,
                            "total": len(cases),
                            "rows": sum(map(len, splits.values())),
                            "rejected": len(rejected),
                        }
                    ),
                    flush=True,
                )
    data.validate_dataset(splits)
    hashes = {}
    for split, rows in splits.items():
        rows.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
        data.atomic_bytes(output / (split + ".jsonl"), payload)
        hashes[split] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema": 5,
        "split_sha256": hashes,
        "count_schema": "source-population-v1",
        "count_definitions": COUNT_DEFINITIONS,
        "counts": {s: population_counts(rs) for s, rs in splits.items()},
        "row_counts": {s: len(rs) for s, rs in splits.items()},
        "split_group_counts": {
            s: len({r["split_group_id"] for r in rs})
            for s, rs in splits.items()
        },
        "label_counts": {
            s: dict(Counter(r["expected"]["security"] for r in rs))
            for s, rs in splits.items()
        },
        "source_counts": {
            s: dict(Counter(r["source"]["name"] for r in rs))
            for s, rs in splits.items()
        },
        "verification": str(Path(verification).resolve()),
        "solver_version": data.solver_version(yosys),
        "admitted_contracts": admitted,
        "rejections": rejected,
        "builder_sha256": data.file_hash(__file__),
        "checker_sha256": data.file_hash(data.__file__),
        "claims": "Local contracts only. No whole-project certification or model-quality result.",
    }
    data.atomic_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--yosys", required=True)
    parser.add_argument("--iverilog", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cases = json.loads(args.catalog.read_text())
    manifest = materialize(
        cases,
        args.output,
        args.verification,
        args.yosys,
        args.iverilog,
        args.workers,
    )
    print(
        json.dumps(
            {
                "row_counts": manifest["row_counts"],
                "split_sha256": manifest["split_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
