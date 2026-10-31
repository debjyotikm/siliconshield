"""Build an external security-evaluation set from local case definitions."""

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from siliconshield.training.laya.build_security_data import (
    LABELS,
    QUESTION,
    digest,
    verify,
)

# Load case definitions from SILICONSHIELD_EXTERNAL_CASES_MODULE.
# The module provides build_cases(sources) and an optional SPLIT_GROUP_ID.
# Without it, the builder uses a small synthetic example.
SPLIT_GROUP_ID = "external:authored-example-v1"
# Version of the external dataset manifest written by main(). Distinct from
# the base/expanded manifests; the split is final-evaluation-only.
EXTERNAL_DATASET_MANIFEST_SCHEMA = 2


def require_source(raw, fragments, name):
    for fragment in fragments:
        if fragment not in raw:
            raise ValueError(
                "source extraction anchor missing for %s: %s" % (name, fragment)
            )


def read_raw_source(path):
    data = path.read_bytes()
    return data.decode("utf-8"), hashlib.sha256(data).hexdigest()


def hold_check(signal, condition):
    return """reg __past_valid = 0;
always @(posedge clk) begin
  __past_valid <= 1;
  if (__past_valid && rst_n && $past(rst_n) && $past(%s)) assert(%s == $past(%s));
end""" % (condition, signal, signal)


def value_after_check(signal, condition, value):
    return """reg __past_valid = 0;
always @(posedge clk) begin
  __past_valid <= 1;
  if (__past_valid && rst_n && $past(rst_n) && $past(%s)) assert(%s == %s);
end""" % (condition, signal, value)


def comb_check(body):
    return "always @* begin %s end" % body


def make_case(
    name,
    family_id,
    base_design_id,
    mechanism_id,
    good,
    bad,
    good_hole,
    bad_hole,
    opaque_hole,
    requirement,
    assertion,
    reset,
    source,
    scope,
    reset_assumptions="",
):
    if "condition_expr" in good or "condition_expr" in bad:
        raise ValueError(
            "external instrumentation name collides with source logic"
        )
    monitor = "wire condition_expr;\nwire status_monitor = condition_expr;"
    good = good.replace(");\n", ");\n" + monitor + "\n", 1)
    bad = bad.replace(");\n", ");\n" + monitor + "\n", 1)
    if good == bad:
        raise ValueError("mutation did not apply: " + name)
    good_lines = good.splitlines()
    bad_lines = bad.splitlines()
    if len(good_lines) != len(bad_lines):
        raise ValueError("mutant changed line count: " + name)
    changed = [
        (index, before, after)
        for index, (before, after) in enumerate(zip(good_lines, bad_lines))
        if before != after
    ]
    if len(changed) != 1:
        raise ValueError(
            "bad mutant must change exactly one targeted line: " + name
        )
    index, before, after = changed[0]
    if before != good_hole or after != bad_hole:
        raise ValueError(
            "shared hole must cover the targeted mutant line: " + name
        )
    if good.count(good_hole) != 1 or bad.count(bad_hole) != 1:
        raise ValueError("completion hole is not unique: " + name)
    opaque = good.replace(good_hole, opaque_hole, 1)
    if opaque.count(opaque_hole) != 1:
        raise ValueError("opaque hole is not unique: " + name)
    good_reconstructed = opaque.replace(opaque_hole, good_hole, 1)
    bad_reconstructed = opaque.replace(opaque_hole, bad_hole, 1)
    if good_reconstructed != good:
        raise ValueError(
            "good completion does not reconstruct exactly: " + name
        )
    if bad_reconstructed != bad:
        raise ValueError("bad completion does not reconstruct exactly: " + name)
    checks = {
        "shared_hole": True,
        "good_exact": True,
        "bad_exact": True,
        "bad_changed_lines": 1,
        "bad_changed_line_number": index + 1,
        "good_code_sha256": digest(good),
        "bad_code_sha256": digest(bad),
        "opaque_code_sha256": digest(opaque),
        "good_hole_sha256": digest(good_hole),
        "bad_hole_sha256": digest(bad_hole),
        "opaque_hole_sha256": digest(opaque_hole),
    }
    return {
        "name": name,
        "family_id": family_id,
        "base_design_id": base_design_id,
        "mechanism_id": mechanism_id,
        "good": good,
        "bad": bad,
        "opaque": opaque,
        "requirement": requirement,
        "assertion": assertion,
        "reset": reset,
        "source": dict(source, scope=scope),
        "reset_assumptions": reset_assumptions,
        "completion_checks": checks,
    }


def _load_cases_module():
    import importlib.util
    import os

    selected = os.environ.get("SILICONSHIELD_EXTERNAL_CASES_MODULE")
    if not selected:
        return None
    spec = importlib.util.spec_from_file_location(
        "siliconshield_external_cases", selected
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    global SPLIT_GROUP_ID
    SPLIT_GROUP_ID = getattr(module, "SPLIT_GROUP_ID", SPLIT_GROUP_ID)
    return module


def build_cases(sources):
    """Authored synthetic fallback; real evaluation cases load from the local module."""
    module = _load_cases_module()
    if module is not None:
        return module.build_cases(sources)
    good = """module topmodule(input clk, rst_n, input wr, input locked,
input [7:0] wdata, output reg [7:0] value);
always @(posedge clk) begin
  if (!rst_n) value <= 0;
  else if (wr && !locked) value <= wdata;
end
endmodule
"""
    bad = good.replace(
        "else if (wr && !locked) value <= wdata;",
        "else if (wr) value <= wdata;",
        1,
    )
    source = {
        "name": "authored-synthetic",
        "module": "authored_lock_register",
        "license": "project-authored",
        "path": "authored",
        "raw_sha256": digest(good),
        "original_sha256": digest(good),
    }
    scope = "Authored one-register lock cone; illustrative only, not an evaluation case."
    return [
        make_case(
            "authored_lock_hold",
            "authored:lock",
            "authored:lock:cone",
            "register_write_lock",
            good,
            bad,
            "  else if (wr && !locked) value <= wdata;",
            "  else if (wr) value <= wdata;",
            "  else if (condition_expr) value <= wdata;",
            "If locked is 1 immediately before a qualifying rising edge, value must retain its "
            "previous value after that edge regardless of wr or wdata.",
            hold_check("value", "locked"),
            "rst_n",
            source,
            scope,
        )
    ]


def assumptions(case):
    text = "Two-state digital RTL; all inputs are defined. Only the stated requirement is assessed. "
    if case["reset"]:
        text += (
            "rst_n is low for the initial clock cycle and high thereafter. Reset takes precedence over writes and "
            "other updates. The property begins after initialization and applies only at rising edges with rst_n "
            "high immediately before and after the edge; values are observed just after those edges. "
        )
        text += case["reset_assumptions"]
    else:
        text += "Combinational behavior is evaluated after settling. "
    text += (
        "Any undriven wire is an omitted combinational expression over the shown signals, not an independent "
        "unconstrained input; its relevance to the stated requirement must be assessed."
    )
    return text


def render_state(case, code):
    return "REQUIREMENT\n%s\nASSUMPTIONS\n%s\nRTL\n%s" % (
        case["requirement"],
        assumptions(case),
        code,
    )


def build_row(case, label, code, evidence):
    state = render_state(case, code)
    lowered = state.lower()
    for forbidden in (
        "verified completion",
        "satisfying_completion",
        "violating_completion",
    ):
        if forbidden in lowered:
            raise ValueError(
                "completion verdict leaked into input: " + case["name"]
            )
    return {
        "id": digest(state)[:24],
        "family_id": case["family_id"],
        "split_group_id": SPLIT_GROUP_ID,
        "split": "external",
        "topic": case["name"],
        "pair_id": "external:" + case["name"],
        "base_design_id": case["base_design_id"],
        "mechanism_id": case["mechanism_id"],
        "rewrite_group_id": "external:" + case["name"] + ":canonical",
        "variant": 0,
        "state": state,
        "questions": QUESTION,
        "expected": {"security": label},
        "validation": evidence,
        "source": case["source"],
    }


def validate_rows(rows, output):
    if not rows or len(rows) % len(LABELS):
        raise ValueError("external rows must form complete label triplets")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate external record ID")
    if len({row["state"] for row in rows}) != len(rows):
        raise ValueError("duplicate external input")
    if any(row["expected"]["security"] not in LABELS for row in rows):
        raise ValueError("unknown external label")
    if any(not row.get("validation", {}).get("accepted") for row in rows):
        raise ValueError("unaccepted external row")
    if any(row["variant"] != 0 for row in rows):
        raise ValueError("alpha or rewrite variants are not permitted")
    counts = Counter(row["expected"]["security"] for row in rows)
    if counts != Counter({label: len(rows) // len(LABELS) for label in LABELS}):
        raise ValueError("external labels are not balanced: " + repr(counts))
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row["expected"]["security"])
        if (
            row["split"] != "external"
            or row["split_group_id"] != SPLIT_GROUP_ID
        ):
            raise ValueError("external split ownership changed")
    if len(pairs) * len(LABELS) != len(rows):
        raise ValueError("expected complete property pairs")
    for pair_id, labels in pairs.items():
        if len(labels) != 3 or set(labels) != set(LABELS):
            raise ValueError(
                "property does not contain each label exactly once: " + pair_id
            )
    families = {row["family_id"] for row in rows}
    ids = {row["id"] for row in rows}
    for split in ("train", "dev", "calibration", "test"):
        path = output / (split + ".jsonl")
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            other = json.loads(line)
            if (
                other.get("family_id") in families
                or other.get("split_group_id") == SPLIT_GROUP_ID
            ):
                raise ValueError(
                    "external source lineage crosses split: " + split
                )
            if other.get("id") in ids:
                raise ValueError("external record ID crosses split: " + split)


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def system_verilog_yosys(yosys, verification):
    executable = Path(yosys).resolve()
    if not executable.is_file():
        raise ValueError("Yosys executable not found: " + str(executable))
    verification.mkdir(parents=True, exist_ok=True)
    launcher = verification / "yosys_sv_launcher.py"
    program = """#!/usr/bin/env python3
import os
import sys
arguments = sys.argv[1:]
for index, argument in enumerate(arguments[:-1]):
    if argument == "-p":
        arguments[index + 1] = arguments[index + 1].replace("read_verilog -formal checked.v", "read_verilog -formal -sv checked.v")
os.execv(%r, [%r] + arguments)
""" % (str(executable), str(executable))
    atomic_write(launcher, program)
    launcher.chmod(0o700)
    return str(launcher)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Directory for the external evaluation set and validation reports",
    )
    parser.add_argument(
        "--verification",
        type=Path,
        help="Directory for solver inputs and results",
    )
    parser.add_argument(
        "--yosys", required=True, help="Path to the Yosys executable"
    )
    args = parser.parse_args()
    if LABELS != [
        "satisfies_requirement",
        "violates_requirement",
        "insufficient_context",
    ]:
        raise ValueError("canonical label order changed")
    if QUESTION.get("security", {}).get("criteria") != LABELS:
        raise ValueError("question criteria do not match labels")
    args.output.mkdir(parents=True, exist_ok=True)
    verification = (
        args.verification
        if args.verification is not None
        else args.output / "verification"
    )
    cases = build_cases(args.sources)
    yosys_command = system_verilog_yosys(args.yosys, verification)

    def process(case):
        good = verify(
            case["good"],
            case["assertion"],
            yosys_command,
            verification,
            case["reset"],
        )
        bad = verify(
            case["bad"],
            case["assertion"],
            yosys_command,
            verification,
            case["reset"],
        )
        return case, good, bad

    rows = []
    property_records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for case, good, bad in pool.map(process, cases):
            if not (
                good.get("accepted")
                and bad.get("accepted")
                and good.get("verdict") == LABELS[0]
                and bad.get("verdict") == LABELS[1]
            ):
                raise ValueError(
                    (
                        "screened external property failed",
                        case["name"],
                        good,
                        bad,
                    )
                )
            good_row = build_row(
                case,
                LABELS[0],
                case["good"],
                dict(good, code_sha256=digest(case["good"])),
            )
            bad_row = build_row(
                case,
                LABELS[1],
                case["bad"],
                dict(bad, code_sha256=digest(case["bad"])),
            )
            opaque_evidence = {
                "accepted": True,
                "basis": "two_exact_verified_completions",
                "satisfying_completion": good["evidence_id"],
                "violating_completion": bad["evidence_id"],
                "satisfying_completion_row": good_row["id"],
                "violating_completion_row": bad_row["id"],
                "satisfying_completion_sha256": digest(case["good"]),
                "violating_completion_sha256": digest(case["bad"]),
                "opaque_code_sha256": digest(case["opaque"]),
            }
            opaque_row = build_row(
                case, LABELS[2], case["opaque"], opaque_evidence
            )
            rows.extend((good_row, bad_row, opaque_row))
            property_records.append(
                {
                    "pair_id": good_row["pair_id"],
                    "family_id": case["family_id"],
                    "base_design_id": case["base_design_id"],
                    "mechanism_id": case["mechanism_id"],
                    "satisfying_evidence": good["evidence_id"],
                    "violating_evidence": bad["evidence_id"],
                    "satisfying_row": good_row["id"],
                    "violating_row": bad_row["id"],
                    "opaque_row": opaque_row["id"],
                    "completion_checks": case["completion_checks"],
                }
            )
    validate_rows(rows, args.output)
    payload = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
    family_properties = Counter(
        record["family_id"] for record in property_records
    )
    family_rows = Counter(row["family_id"] for row in rows)
    label_rows = Counter(row["expected"]["security"] for row in rows)
    source_hashes = {
        row["source"]["path"]: row["source"]["raw_sha256"] for row in rows
    }
    source_details = {
        row["source"]["path"]: {
            key: value for key, value in row["source"].items() if key != "scope"
        }
        for row in rows
    }
    manifest = {
        "schema": EXTERNAL_DATASET_MANIFEST_SCHEMA,
        "target": "three_way_requirement_verdict",
        "label_order": LABELS,
        "properties": len(property_records),
        "modules": len(family_rows),
        "rows": len(rows),
        "property_counts_by_module": dict(sorted(family_properties.items())),
        "row_counts_by_module": dict(sorted(family_rows.items())),
        "row_counts_by_label": {label: label_rows[label] for label in LABELS},
        "split": "external",
        "split_role": "final_evaluation_only",
        "split_group_id": SPLIT_GROUP_ID,
        "split_groups": {
            SPLIT_GROUP_ID: {
                "rows": len(rows),
                "properties": len(property_records),
                "families": sorted(family_rows),
            }
        },
        "no_split_crossing": True,
        "alpha_variants": 0,
        "builder_sha256": digest(Path(__file__).read_text()),
        "source_sha256": dict(sorted(source_hashes.items())),
        "source_raw_sha256": dict(sorted(source_hashes.items())),
        "sources": dict(sorted(source_details.items())),
        "external_jsonl_sha256": digest(payload),
        "verification": {
            "interface": "build_security_data.verify",
            "directory": str(verification),
            "frontend": "read_verilog -formal -sv through local launcher",
            "yosys": str(Path(args.yosys).resolve()),
            "accepted_property_pairs": len(property_records),
        },
        "properties_detail": property_records,
        "exact_completion_checks": {
            record["pair_id"]: record["completion_checks"]
            for record in property_records
        },
        "scope": "Precise properties in small source-derived combinational and sequential cones; not whole-module or whole-SoC certification.",
        "scope_limitations": [
            "Two-state defined-input evaluation only; X/Z behavior, analog timing, and reset glitches are excluded.",
            "Case-specific scope and source-license statements are recorded per case in the local case definitions.",
            "All external rows share one split group and must remain disjoint from train, dev, calibration, and test when integrated.",
        ],
        "write_mode": "atomic_replace",
    }
    atomic_write(args.output / "external.jsonl", payload)
    atomic_write(
        args.output / "external_manifest.json",
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
    )
    print(
        json.dumps(
            {
                "properties": len(property_records),
                "modules": len(family_rows),
                "rows": len(rows),
                "labels": manifest["row_counts_by_label"],
                "builder_sha256": manifest["builder_sha256"],
                "output": str(args.output / "external.jsonl"),
            }
        )
    )


if __name__ == "__main__":
    main()
