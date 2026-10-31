"""Add solver-checked satisfying, violating, and missing-context examples to training."""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from siliconshield.training.laya.build_security_data import (
    LABELS,
    QUESTION,
    digest,
    rename_example,
    verify,
)

FAMILY_ID = "authored:semantic_register_templates"
# Version of the expanded dataset manifest written by main(). Distinct from
# the base dataset manifest (build_security_data.DATASET_MANIFEST_SCHEMA).
EXPANDED_DATASET_MANIFEST_SCHEMA = 2
POLICY_TOKEN = "POLICY"
ASSERTION_TOKEN = "__SEMANTIC_ASSERTION_BOUNDARY__"
VARIANTS = (0, 1, 2)
ASSUMPTIONS = (
    "Two-state sampled-clock RTL with defined inputs and arbitrary-but-defined initial state. rst_n is zero "
    "for the initial cycle and one thereafter. Values are observed after rising clock edges. Reset takes "
    "precedence; sequential properties apply only at edges with rst_n high immediately before and after the "
    "edge. Only this requirement is assessed. Any referenced expression without a supplied definition is an "
    "unspecified, side-effect-free combinational expression over the shown signals, with no other assignments "
    "to the shown state."
)


HOLE_NAMES = (
    "condition_expr",
    "guard_expr",
    "control_expr",
    "access_expr",
    "request_expr",
    "permission_expr",
)
# Load scenario(seed, index) from SILICONSHIELD_SEMANTIC_RECIPE_MODULE.
# The fallback below supplies one synthetic example for tests.
import importlib.util
import os


def exact_check(trigger, data="wdata", extra_guard="1'b1"):
    return """reg __past_valid=0;
always @(posedge clk) begin
 __past_valid<=1;
 if (__past_valid && rst_n && $past(rst_n) && (%s))
 assert(value == ($past(%s) ? $past(%s) : $past(value)));
end""" % (extra_guard, trigger, data)


def pipeline_check(trigger):
    return (
        """reg __past_valid=0;
always @(posedge clk) begin
 __past_valid<=1;
 if (__past_valid && $past(__past_valid) && rst_n && $past(rst_n) && $past(rst_n, 2))
 assert(value == ($past(%s, 2) ? $past(wdata, 2) : $past(value)));
end"""
        % trigger
    )


def _load_recipe_module():
    selected = os.environ.get("SILICONSHIELD_SEMANTIC_RECIPE_MODULE")
    if not selected:
        return None
    spec = importlib.util.spec_from_file_location(
        "siliconshield_semantic_recipe", selected
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RECIPE = None
_RECIPE_LOADED = False


def _recipe():
    global _RECIPE, _RECIPE_LOADED
    if not _RECIPE_LOADED:
        _RECIPE = _load_recipe_module()
        _RECIPE_LOADED = True
    return _RECIPE


MECHANISMS = ("privilege_lock",)
STYLES = ("direct",)


def scenario(seed, index):
    """Authored illustrative scenario; real recipes load from the local module."""
    recipe = _recipe()
    if recipe is not None:
        return recipe.scenario(seed, index)
    policy = "(wr) && (addr == 4'd0) && (!locked && (priv == 2'd1))"
    pair = "authored-semantic:%d:%d" % (seed, index)
    code = (
        "module topmodule(input clk, input rst_n, input wr, input service_wr,\n"
        "input locked, input debug, input aux_value,\n"
        "input [1:0] priv, input [1:0] role, input [1:0] owner, input [3:0] addr,\n"
        "input [7:0] wdata, output reg [7:0] value, output wire status_out);\n"
        "assign status_out = aux_value;\n"
        "wire request = POLICY;\n"
        "always @(posedge clk) begin\nif (!rst_n) value <= 0;\n"
        "else if (request) value <= wdata;\nend\nendmodule\n"
    )
    return {
        "style": "direct",
        "code": code,
        "policy": policy,
        "alternative": "!(%s)" % policy,
        "good_fill": policy,
        "bad_fill": "!(%s)" % policy,
        "assertion": exact_check(policy),
        "requirement": (
            "After each rising edge, value must take wdata exactly when (%s) held "
            "before the edge; otherwise retain its previous value." % policy
        ),
        "index": index,
        "seed": seed,
        "family_id": FAMILY_ID,
        "split_group_id": FAMILY_ID,
        "base_design_id": pair,
        "mechanism_id": "privilege_lock",
        "rewrite_group_id": pair,
        "pair_id": pair,
        "lexical_stratum": "authored_example",
        "good_rewrite": "policy",
        "bad_kind": "complement",
        "source": {
            "name": "authored-semantic",
            "template": "direct",
            "seed": seed,
            "case_index": index,
            "license": "project-authored",
            "independent_reference_design": False,
            "mechanism": "privilege_lock",
            "near_miss": "complement",
            "good_rewrite": "policy",
            "lexical_stratum": "authored_example",
        },
    }


def render_row(code, requirement, label, evidence, case, variant):
    state = "REQUIREMENT\n%s\nASSUMPTIONS\n%s\nRTL\n%s" % (
        requirement,
        ASSUMPTIONS,
        code,
    )
    return {
        "id": digest(state)[:24],
        "family_id": FAMILY_ID,
        "split_group_id": case["split_group_id"],
        "split": "train",
        "base_design_id": case["base_design_id"],
        "mechanism_id": case["mechanism_id"],
        "rewrite_group_id": case["rewrite_group_id"],
        "pair_id": case["pair_id"] + ":" + str(variant),
        "variant": variant,
        "lexical_stratum": case["lexical_stratum"],
        "topic": case["mechanism_id"],
        "state": state,
        "questions": QUESTION,
        "expected": {"security": label},
        "validation": evidence,
        "source": dict(case["source"]),
    }


def rename_rendered(code, requirement, assertion, variant):
    combined = requirement + "\n" + ASSERTION_TOKEN + "\n" + assertion
    rendered, renamed = rename_example(code, combined, variant)
    if renamed.count(ASSERTION_TOKEN) != 1:
        raise ValueError("assertion boundary changed during renaming")
    rule, check = renamed.split("\n" + ASSERTION_TOKEN + "\n", 1)
    return rendered, rule, check


def rename_fill(code, fill, variant):
    return rename_example(code, fill, variant)[1]


def add_irrelevant_hole(code, hole_name):
    if (
        code.count("input aux_value,\n") != 1
        or code.count("assign status_out = aux_value;") != 1
    ):
        raise ValueError("irrelevant expression template is not exact")
    code = code.replace("input aux_value,\n", "")
    return code.replace(
        "assign status_out = aux_value;",
        "wire %s;\nassign status_out = %s;" % (hole_name, hole_name),
    )


def completion_check(opaque, rendered, hole_name, fill):
    declaration = "wire %s;" % hole_name
    if opaque.count(declaration) != 1:
        raise ValueError("shared-hole declaration must occur exactly once")
    marker = "__PRESERVE_SEMANTIC_HOLE_DECLARATION__"
    template = opaque.replace(declaration, marker, 1)
    if template.count(hole_name) < 1:
        raise ValueError("shared hole must be used")
    completed = template.replace(hole_name, fill).replace(
        marker, declaration, 1
    )
    if completed != rendered:
        raise ValueError("shared-hole completion is not byte-exact")


def expand_one(case, yosys, verification):
    rows = []
    for variant in VARIANTS:
        hole_index = (case["index"] + variant) % len(HOLE_NAMES)
        relevant_hole = HOLE_NAMES[hole_index]
        irrelevant_hole = None
        common = case["code"]
        if (case["index"] + variant) % 2 == 0:
            irrelevant_hole = HOLE_NAMES[(hole_index + 1) % len(HOLE_NAMES)]
            if irrelevant_hole == relevant_hole:
                raise ValueError("co-occurring holes must have distinct names")
            common = add_irrelevant_hole(common, irrelevant_hole)
        if (
            common.count("wire request = POLICY;") != 1
            or common.count(POLICY_TOKEN) != 1
        ):
            raise ValueError("semantic template must contain one policy site")
        common = common.replace(
            "wire request = POLICY;",
            "wire %s;\nwire request = POLICY;" % relevant_hole,
            1,
        )
        good_code = common.replace(POLICY_TOKEN, case["good_fill"])
        bad_code = common.replace(POLICY_TOKEN, case["bad_fill"])
        opaque_code = common.replace(POLICY_TOKEN, relevant_hole)
        good_rendered, requirement, check = rename_rendered(
            good_code, case["requirement"], case["assertion"], variant
        )
        bad_rendered, bad_requirement, bad_check = rename_rendered(
            bad_code, case["requirement"], case["assertion"], variant
        )
        opaque_rendered, opaque_requirement, opaque_check = rename_rendered(
            opaque_code, case["requirement"], case["assertion"], variant
        )
        if (
            requirement != bad_requirement
            or requirement != opaque_requirement
            or check != bad_check
            or check != opaque_check
        ):
            raise ValueError(
                "renaming changed the shared requirement or assertion"
            )
        good_fill = rename_fill(good_code, case["good_fill"], variant)
        bad_fill = rename_fill(bad_code, case["bad_fill"], variant)
        rendered_hole = (
            relevant_hole
            if variant == 0
            else relevant_hole + "_" + str(variant)
        )
        completion_check(
            opaque_rendered, good_rendered, rendered_hole, good_fill
        )
        completion_check(opaque_rendered, bad_rendered, rendered_hole, bad_fill)
        good = verify(good_rendered, check, yosys, verification, "rst_n")
        bad = verify(bad_rendered, bad_check, yosys, verification, "rst_n")
        if (
            not good.get("accepted")
            or good.get("verdict") != LABELS[0]
            or not bad.get("accepted")
            or bad.get("verdict") != LABELS[1]
        ):
            return [], {
                "index": case["index"],
                "seed": case["seed"],
                "style": case["style"],
                "mechanism": case["mechanism_id"],
                "variant": variant,
                "good": good,
                "bad": bad,
            }
        shared = {
            "alpha_renaming": variant,
            "rendered_assertion_sha256": digest(check),
            "irrelevant_hole": irrelevant_hole,
        }
        good_evidence = dict(good, **shared)
        good_evidence["rendered_code_sha256"] = digest(good_rendered)
        bad_evidence = dict(bad, **shared)
        bad_evidence["rendered_code_sha256"] = digest(bad_rendered)
        unknown = {
            "accepted": True,
            "basis": "exact_shared_hole_completions",
            "satisfying_completion": good["evidence_id"],
            "violating_completion": bad["evidence_id"],
            "hole_name": rendered_hole,
            "good_fill_sha256": digest(good_fill),
            "bad_fill_sha256": digest(bad_fill),
            "good_completion": "byte_exact",
            "bad_completion": "byte_exact",
            "alpha_renaming": variant,
            "rendered_code_sha256": digest(opaque_rendered),
            "irrelevant_hole": irrelevant_hole,
        }
        rows.append(
            render_row(
                good_rendered,
                requirement,
                LABELS[0],
                good_evidence,
                case,
                variant,
            )
        )
        rows.append(
            render_row(
                bad_rendered,
                requirement,
                LABELS[1],
                bad_evidence,
                case,
                variant,
            )
        )
        rows.append(
            render_row(
                opaque_rendered, requirement, LABELS[2], unknown, case, variant
            )
        )
    return rows, None


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_name(row):
    source = row.get("source", {})
    return (
        source.get("name", "unknown")
        if isinstance(source, dict)
        else str(source)
    )


def index_rows(rows):
    states = {}
    ids = {}
    for row in rows:
        if "state" in row:
            states[row["state"]] = row
        if "id" in row:
            ids[row["id"]] = row
    return states, ids


def duplicate_group(group, states, ids):
    duplicate = False
    local_states, local_ids = {}, {}
    for row in group:
        for existing in (
            states.get(row["state"]),
            local_states.get(row["state"]),
        ):
            if existing is not None:
                if existing.get("expected") != row.get("expected"):
                    raise ValueError(
                        "conflicting labels for identical input: " + row["id"]
                    )
                duplicate = True
        existing = ids.get(row["id"], local_ids.get(row["id"]))
        if existing is not None:
            if existing.get("state") != row.get("state"):
                raise ValueError(
                    "identifier collision for distinct inputs: " + row["id"]
                )
            duplicate = True
        local_states[row["state"]] = row
        local_ids[row["id"]] = row
    return duplicate


def remember_group(group, states, ids):
    for row in group:
        states[row["state"]] = row
        ids[row["id"]] = row


def canonical_input(state):
    import re

    text = re.sub(r"\b([A-Za-z_]\w*)_[12]\b", r"\1", state)
    return re.sub(r"\s+", "", text)


def validate_combined(split_rows):
    owners, ids, states, canonical, pairs = {}, set(), set(), {}, {}
    for split, rows in split_rows.items():
        for row in rows:
            group = row.get("split_group_id", row["family_id"])
            if owners.setdefault(group, split) != split:
                raise ValueError("split group crosses splits: " + group)
            if row.get("split", split) != split:
                raise ValueError("row split ownership mismatch")
            if row["id"] in ids or row["state"] in states:
                raise ValueError("duplicate input within or across splits")
            ids.add(row["id"])
            states.add(row["state"])
            normalized = canonical_input(row["state"])
            if canonical.setdefault(normalized, split) != split:
                raise ValueError("alpha/format-equivalent input crosses splits")
            pairs.setdefault((split, row["pair_id"]), set()).add(
                row["expected"]["security"]
            )
    if any(labels != set(LABELS) for labels in pairs.values()):
        raise ValueError("combined output contains an incomplete triplet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-data",
        type=Path,
        required=True,
        help="Directory containing the existing dataset",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        required=True,
        help="Directory containing the referenced RTL sources",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for the expanded dataset",
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel data-preparation workers",
    )
    args = parser.parse_args()
    if args.output.resolve() == args.base_data.resolve():
        raise ValueError("output must differ from base data")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise ValueError("output already materialized")
    split_names = ("train", "dev", "calibration", "test", "external")
    present = [
        name
        for name in split_names
        if (args.base_data / (name + ".jsonl")).is_file()
    ]
    if "train" not in present:
        raise ValueError("base train split is required")
    base_train = (args.base_data / "train.jsonl").read_bytes()
    base_splits = {
        name: [
            json.loads(line)
            for line in (args.base_data / (name + ".jsonl"))
            .read_text()
            .splitlines()
            if line.strip()
        ]
        for name in present
    }
    base_rows = base_splits["train"]
    states, ids = index_rows(base_rows)
    accepted_groups = []
    dropped = []
    verified_groups = 0
    duplicate_groups = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:

        def build(index):
            return expand_one(
                scenario(args.seed, index), args.yosys, args.verification
            )

        for processed, (produced, failure) in enumerate(
            pool.map(build, range(args.cases)), 1
        ):
            if failure:
                dropped.append(dict(failure, reason="verification"))
            else:
                verified_groups += 1
                if duplicate_group(produced, states, ids):
                    duplicate_groups += 1
                    dropped.append(
                        {
                            "index": produced[0]["source"]["case_index"],
                            "seed": args.seed,
                            "style": produced[0]["source"]["template"],
                            "reason": "duplicate_group",
                        }
                    )
                else:
                    remember_group(produced, states, ids)
                    accepted_groups.append(produced)
            if processed % 128 == 0:
                print(
                    json.dumps(
                        {
                            "verified_groups": verified_groups,
                            "accepted_groups": len(accepted_groups),
                            "dropped_groups": len(dropped),
                        }
                    ),
                    flush=True,
                )
    if len(accepted_groups) < args.cases * 0.95:
        raise ValueError(
            "too many dropped semantic groups: " + json.dumps(dropped[:3])
        )
    random.Random(args.seed).shuffle(accepted_groups)
    generated_rows = [row for group in accepted_groups for row in group]
    generated_payload = "".join(
        json.dumps(row) + "\n" for row in generated_rows
    ).encode("utf-8")
    separator = b"\n" if base_train and not base_train.endswith(b"\n") else b""
    combined = dict(base_splits)
    combined["train"] = base_rows + generated_rows
    validate_combined(combined)
    (args.output / "train.jsonl").write_bytes(
        base_train + separator + generated_payload
    )
    for name in present:
        if name != "train":
            shutil.copyfile(
                args.base_data / (name + ".jsonl"),
                args.output / (name + ".jsonl"),
            )
    split_hashes = {
        name: file_hash(args.output / (name + ".jsonl")) for name in present
    }
    held_out = [name for name in present if name != "train"]
    held_out_hashes = {}
    for name in held_out:
        base_hash = file_hash(args.base_data / (name + ".jsonl"))
        output_hash = split_hashes[name]
        held_out_hashes[name] = {
            "base_sha256": base_hash,
            "output_sha256": output_hash,
            "preserved": base_hash == output_hash,
        }
    all_rows = base_rows + generated_rows
    generated_labels = Counter(
        row["expected"]["security"] for row in generated_rows
    )
    generated_sources = Counter(source_name(row) for row in generated_rows)
    styles = Counter(row["source"]["template"] for row in generated_rows)
    lexical = Counter(row["lexical_stratum"] for row in generated_rows)
    mechanisms = Counter(row["mechanism_id"] for row in generated_rows)
    base_manifest = args.base_data / "manifest.json"
    base_manifest_hash = (
        file_hash(base_manifest) if base_manifest.is_file() else None
    )
    manifest = {
        "schema": EXPANDED_DATASET_MANIFEST_SCHEMA,
        "seed": args.seed,
        "family_id": FAMILY_ID,
        "train_only": True,
        "variants": list(VARIANTS),
        "requested_groups": args.cases,
        "verified_groups": verified_groups,
        "accepted_groups": len(accepted_groups),
        "dropped_groups": len(dropped),
        "dropped_group_details": dropped,
        "duplicate_groups_dropped": duplicate_groups,
        "authored_parameter_cases_accepted": len(accepted_groups),
        "authored_parameter_cases_rejected": dropped,
        "train_rows": len(all_rows),
        "generated_rows": len(generated_rows),
        "train_labels": dict(
            Counter(row["expected"]["security"] for row in all_rows)
        ),
        "train_sources": dict(Counter(source_name(row) for row in all_rows)),
        "generated_labels": dict(generated_labels),
        "generated_sources": dict(generated_sources),
        "train_styles": dict(styles),
        "lexical_strata": dict(lexical),
        "mechanisms": dict(mechanisms),
        "authored_templates": list(STYLES),
        "hole_names": list(HOLE_NAMES),
        "templates_are_not_independent_designs": True,
        "base_data_manifest_sha256": base_manifest_hash,
        "builder_sha256": file_hash(Path(__file__)),
        "build_security_data_sha256": file_hash(
            Path(__file__).with_name("build_security_data.py")
        ),
        "split_sha256": split_hashes,
        "held_out_hashes": held_out_hashes,
        "held_out_hashes_preserved": all(
            item["preserved"] for item in held_out_hashes.values()
        ),
        "verification_directory": str(args.verification),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
