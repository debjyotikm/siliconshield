"""Build register-security datasets and check labels with a solver."""

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from siliconshield.training.laya.question_contracts import (
    LABELS as _CONTRACT_LABELS,
)
from siliconshield.training.laya.question_contracts import (
    LEGACY_CONTRACT as _LEGACY_CONTRACT,
)
from siliconshield.training.laya.question_contracts import (
    question_for as _question_for,
)

LABELS = list(_CONTRACT_LABELS)
QUESTION = _question_for(_LEGACY_CONTRACT)


def questions_for_version(version):
    """Return model-visible questions for a contract version."""
    return _question_for(version)


HOLE_NAMES = (
    "condition_expr",
    "guard_expr",
    "control_expr",
    "access_expr",
    "request_expr",
    "permission_expr",
)
EVIDENCE_SCHEMA = "security-sat-v3"
VARIANTS = (0, 1, 2)
# Version of the built dataset manifest written by main().
DATASET_MANIFEST_SCHEMA = 3

# Load case IDs, splits, and mutations from --cases-manifest or
# SILICONSHIELD_SECURITY_CASES. This catalog shows the format with example IDs.
EXAMPLE_CATALOG = {
    "train_dev_cases": [
        (
            "authored-example-001",
            "config_reg",
            "lock_reg",
            "!lock_reg",
            "lock_reg",
        ),
        (
            "authored-example-002",
            "status_reg",
            "!auth_ok",
            "auth_ok",
            "!auth_ok",
            "comb_zero",
        ),
    ],
    "outer_cases": [
        (
            "authored-example-101",
            "data_out",
            "locked && !debug_en",
            "!locked || debug_en",
            "!locked || debug_mode",
        ),
    ],
    "dev_ids": ["authored-example-001"],
    "calibration_ids": ["authored-example-101"],
    "test_ids": [],
    "mutation_occurrences": {},
    "expected_counts": None,
    "manifest_seed": 0,
    "opentitan_cases": None,
}


def _as_tuples(rows):
    return [tuple(row) for row in rows]


def load_case_catalog(path=None):
    """Load a local dataset catalog (JSON); defaults to the authored example."""
    import os

    selected = path or os.environ.get("SILICONSHIELD_SECURITY_CASES")
    if not selected:
        return EXAMPLE_CATALOG
    catalog = json.loads(Path(selected).read_text())
    return {
        "train_dev_cases": _as_tuples(catalog.get("train_dev_cases", [])),
        "outer_cases": _as_tuples(catalog.get("outer_cases", [])),
        "dev_ids": list(catalog.get("dev_ids", [])),
        "calibration_ids": list(catalog.get("calibration_ids", [])),
        "test_ids": list(catalog.get("test_ids", [])),
        "mutation_occurrences": {
            k: tuple(v)
            for k, v in catalog.get("mutation_occurrences", {}).items()
        },
        "expected_counts": catalog.get("expected_counts"),
        "manifest_seed": catalog.get("manifest_seed", 0),
        "opentitan_cases": catalog.get("opentitan_cases"),
    }


_CATALOG = {}


def apply_catalog(catalog):
    global \
        TRAIN_DEV_CASES, \
        OUTER_CASES, \
        DEV_IDS, \
        CALIBRATION_IDS, \
        TEST_IDS, \
        MUTATION_OCCURRENCES, \
        _CATALOG
    _CATALOG = catalog
    TRAIN_DEV_CASES = catalog["train_dev_cases"]
    OUTER_CASES = catalog["outer_cases"]
    DEV_IDS = set(catalog["dev_ids"])
    CALIBRATION_IDS = set(catalog["calibration_ids"])
    TEST_IDS = set(catalog["test_ids"])
    MUTATION_OCCURRENCES = dict(catalog["mutation_occurrences"])
    return catalog


apply_catalog(load_case_catalog())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_bytes(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_json(path, value):
    atomic_bytes(
        path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    )


def clean(code):
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", code, flags=re.S)
    return (
        "\n".join(line.rstrip() for line in code.splitlines() if line.strip())
        + "\n"
    )


def add_check(code, assertion):
    if code.count("endmodule") != 1:
        raise ValueError("expected one complete module")
    return code.replace("endmodule", assertion + "\nendmodule")


def hold_check(signal, condition, reset="rst_n", kind="hold"):
    if kind == "comb_zero":
        return "always @* if (%s) assert(%s == 0);" % (condition, signal)
    value = "0" if kind == "next_zero" else "$past(%s)" % signal
    return """reg __past_valid = 0;
always @(posedge clk) begin
  __past_valid <= 1;
  if (__past_valid && (%s) && $past(%s) && $past(%s)) assert(%s == %s);
end""" % (reset, reset, condition, signal, value)


@lru_cache(maxsize=8)
def solver_version(yosys):
    run = subprocess.run(
        [yosys, "-V"], capture_output=True, text=True, timeout=10, check=True
    )
    return run.stdout.strip()


def reset_options(reset):
    if not reset:
        return ""
    name = reset.lstrip("!")
    active, inactive = (1, 0) if reset.startswith("!") else (0, 1)
    return "-set %s %d -set-at 1 %s %d" % (name, inactive, name, active)


def solver_commands(
    assertion,
    reset,
    filename="checked.v",
    bounded_cycles=6,
    induction_steps=12,
):
    """Return bounded and induction Yosys commands for a bound.

    ``bounded_cycles`` is the declared counterexample horizon for
    this property and ``induction_steps`` the temporal-induction
    depth. Both are recorded in evidence so a short default
    cannot silently stand in for a deeper obligation.
    """
    for name, value in (
        ("bounded_cycles", bounded_cycles),
        ("induction_steps", induction_steps),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            or value > 64
        ):
            raise ValueError("%s must be an integer in 1..64" % name)
    pre = (
        "read_verilog -formal -sv %s; prep -top topmodule; "
        "async2sync; dffunmap; memory_map; opt; " % filename
    )
    options = reset_options(reset)
    bounded = (
        pre + "sat -seq %d -set-init-def -set-def-inputs %s -prove-asserts "
        "-show-ports -dump_json counterexample.json" % (bounded_cycles, options)
    )
    induction = (
        pre + "sat -seq 4 -tempinduct -tempinduct-def -maxsteps %d "
        "-set-init-def -set-def-inputs %s -prove-asserts"
        % (induction_steps, options)
    )
    return bounded, induction


def conditional_control(assertion):
    if assertion.count("assert(") != 1 or " if (" not in assertion:
        return None
    control = assertion.replace("$past(", "(")
    control = re.sub(r",\s*2\)", ")", control)
    return re.sub(r"assert\(.*\);", "assert(1'b0);", control)


def verify(
    code: str,
    assertion: str,
    yosys: str,
    directory: Any,
    reset: str = "rst_n",
    bounded_cycles: int = 6,
    induction_steps: int = 12,
) -> dict[str, Any]:
    """Verify a property with a declared bound and induction depth."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    version = solver_version(yosys)
    checker = file_hash(__file__)
    key = digest(
        EVIDENCE_SCHEMA
        + version
        + checker
        + code
        + assertion
        + reset
        + str(bounded_cycles)
        + str(induction_steps)
    )
    folder = directory / key
    folder.mkdir(parents=True, exist_ok=True)
    lock_path = folder / "lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        record = folder / "result.json"
        if record.exists():
            cached: dict[str, Any] = json.loads(record.read_text())
            if (
                cached.get("schema") == EVIDENCE_SCHEMA
                and cached.get("checker_sha256") == checker
            ):
                return cached
        bounded, induction = solver_commands(
            assertion,
            reset,
            bounded_cycles=bounded_cycles,
            induction_steps=induction_steps,
        )
        result: dict[str, Any] = {
            "accepted": False,
            "schema": EVIDENCE_SCHEMA,
            "evidence_id": key,
            "verification": "yosys_sat",
            "solver_version": version,
            "checker_sha256": checker,
            "reset": reset,
            "bounded_cycles": bounded_cycles,
            "induction_steps": induction_steps,
            "design_sha256": digest(code),
            "assertion_sha256": digest(assertion),
            "bounded_command": bounded,
            "induction_command": induction,
        }
        atomic_bytes(folder / "design.v", code.encode())
        atomic_bytes(folder / "checked.v", add_check(code, assertion).encode())
        try:
            run = subprocess.run(
                [yosys, "-Q", "-T", "-p", bounded],
                cwd=folder,
                capture_output=True,
                text=True,
                timeout=45,
            )
            atomic_bytes(
                folder / "bounded.log", (run.stdout + run.stderr).encode()
            )
            result["bounded_returncode"] = run.returncode
            if (
                run.returncode == 0
                and "SAT proof finished - model found: FAIL!" in run.stdout
            ):
                result.update(
                    accepted=True,
                    verdict=LABELS[1],
                    basis="reachable_counterexample_%d_cycles" % bounded_cycles,
                )
            elif (
                run.returncode == 0
                and "SAT proof finished - no model found: SUCCESS!"
                in run.stdout
            ):
                proof = subprocess.run(
                    [yosys, "-Q", "-T", "-p", induction],
                    cwd=folder,
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                atomic_bytes(
                    folder / "induction.log",
                    (proof.stdout + proof.stderr).encode(),
                )
                result["induction_returncode"] = proof.returncode
                if (
                    proof.returncode == 0
                    and "Induction step proven: SUCCESS!" in proof.stdout
                ):
                    control = conditional_control(assertion)
                    if control is None:
                        result.update(
                            accepted=True,
                            verdict=LABELS[0],
                            basis="temporal_induction",
                            antecedent="unconditional",
                        )
                    else:
                        atomic_bytes(
                            folder / "antecedent.v",
                            add_check(code, control).encode(),
                        )
                        control_command, _ = solver_commands(
                            control,
                            reset,
                            "antecedent.v",
                            bounded_cycles=bounded_cycles,
                            induction_steps=induction_steps,
                        )
                        result["antecedent_command"] = control_command
                        reach = subprocess.run(
                            [yosys, "-Q", "-T", "-p", control_command],
                            cwd=folder,
                            capture_output=True,
                            text=True,
                            timeout=45,
                        )
                        atomic_bytes(
                            folder / "antecedent.log",
                            (reach.stdout + reach.stderr).encode(),
                        )
                        result["antecedent_returncode"] = reach.returncode
                        if (
                            reach.returncode == 0
                            and "SAT proof finished - model found: FAIL!"
                            in reach.stdout
                        ):
                            result.update(
                                accepted=True,
                                verdict=LABELS[0],
                                basis="temporal_induction",
                                antecedent="reachable_bounded_%d_cycles"
                                % bounded_cycles,
                            )
                        else:
                            result["reason"] = (
                                "unreachable_or_unproved_assertion_antecedent"
                            )
                else:
                    result["reason"] = "bounded_pass_without_inductive_proof"
            else:
                result["reason"] = "solver_or_elaboration_failure"
        except subprocess.TimeoutExpired as exc:
            result["reason"] = "solver_timeout"
            result["timeout_stage"] = str(exc.cmd)[:120]
        atomic_json(record, result)
        return result


def compile_rtl(
    code: str, evidence_id: str, iverilog: str, directory: Any
) -> dict[str, Any]:
    folder = Path(directory) / "compiled" / evidence_id
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "design.v"
    atomic_bytes(source, code.encode())
    command = [iverilog]
    helper = Path(iverilog).parent.parent / "lib/x86_64-linux-gnu/ivl"
    if helper.is_dir():
        command.extend(["-B", str(helper)])
    command.extend(
        ["-g2012", "-s", "topmodule", "-o", str(folder / "simv"), str(source)]
    )
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    atomic_bytes(folder / "compile.log", (run.stdout + run.stderr).encode())
    result = {
        "evidence_id": evidence_id,
        "returncode": run.returncode,
        "command": command,
    }
    atomic_json(folder / "result.json", result)
    if run.returncode:
        raise ValueError("Icarus compilation failed for " + evidence_id)
    return result


def mutation_count(code, before):
    if re.fullmatch(r"[A-Za-z_]\w*", before):
        pattern = r"(?<!\w)" + re.escape(before) + r"(?!\w)(?=[^;\n]*\))"
        return len(re.findall(pattern, code))
    return code.count(before)


def replace_reviewed(code, before, after, uid, occurrences):
    if re.fullmatch(r"[A-Za-z_]\w*", before):
        pattern = r"(?<!\w)" + re.escape(before) + r"(?!\w)(?=[^;\n]*\))"
        matches = list(re.finditer(pattern, code))
    else:
        matches = list(re.finditer(re.escape(before), code))
    if len(matches) < len(occurrences) or sorted(set(occurrences)) != list(
        occurrences
    ):
        raise ValueError("reviewed mutation occurrences are invalid for " + uid)
    if max(occurrences) >= len(matches) or (
        occurrences == (0,) and len(matches) != 1
    ):
        raise ValueError(
            "reviewed mutation occurrence count changed for " + uid
        )
    selected = set(occurrences)
    pieces = []
    end = 0
    for index, match in enumerate(matches):
        pieces.append(code[end : match.start()])
        pieces.append(after if index in selected else match.group())
        end = match.end()
    pieces.append(code[end:])
    return "".join(pieces)


def hole_declaration(width: int, name: str) -> str:
    return (
        "wire %s;" % name
        if width == 1
        else "wire [%d:0] %s;" % (width - 1, name)
    )


def complete_rendered(opaque: str, hole: str, fill: str, width: int) -> str:
    declaration = hole_declaration(width, hole)
    marker = "__PRESERVE_HOLE_DECLARATION__"
    if opaque.count(declaration) != 1:
        raise ValueError("shared-hole declaration is not unique")
    completed = opaque.replace(declaration, marker, 1).replace(hole, fill)
    return completed.replace(marker, declaration, 1)


def insert_declarations(code, declarations):
    marker = ");\n"
    if code.count(marker) < 1:
        raise ValueError("module header not found")
    return code.replace(marker, ");\n" + declarations + "\n", 1)


def requirement_for(signal, condition, kind):
    if kind == "comb_zero":
        return (
            "Whenever (%s) is true after combinational settling, %s must equal zero."
            % (condition, signal)
        )
    if kind == "next_zero":
        return (
            "Except during reset, if (%s) holds immediately before a qualifying rising clock edge, "
            "%s must be zero immediately after that edge." % (condition, signal)
        )
    return (
        "Except during reset, if (%s) holds immediately before a qualifying rising clock edge, "
        "%s must retain its immediately preceding value after that edge."
        % (condition, signal)
    )


def split_for(uid):
    if uid in DEV_IDS:
        return "dev"
    if uid in CALIBRATION_IDS:
        return "calibration"
    if uid in TEST_IDS:
        return "test"
    return "train"


def hardsec_cases(source, entries=None):
    root = source / "HardSecBench_Data"
    metadata = json.loads((root / "result_filtered.json").read_text())
    index = {
        s["seed_uuid"]: (cid, s)
        for cid, c in metadata["cwes"].items()
        for s in c["seeds"]
    }
    specs = list(TRAIN_DEV_CASES) + list(OUTER_CASES)
    for entry in specs if entries is None else entries:
        uid, signal, condition, guard, bad_guard = entry[:5]
        kind = entry[5] if len(entry) > 5 else "hold"
        reset = entry[6] if len(entry) > 6 else "rst_n"
        cid, seed = index[uid]
        path = root / cid.lower() / uid / seed["golden_file"]
        original = clean(path.read_text())
        occurrences = MUTATION_OCCURRENCES.get(uid, (0,))
        if mutation_count(original, guard) < len(occurrences) or (
            occurrences == (0,) and mutation_count(original, guard) != 1
        ):
            raise ValueError("reviewed guard occurrences changed for " + uid)
        yield {
            "uid": uid,
            "family": "hardsec:" + uid,
            "topic": cid,
            "code": original,
            "guard": guard,
            "bad_guard": bad_guard,
            "occurrences": occurrences,
            "requirement": requirement_for(signal, condition, kind),
            "assertion": hold_check(signal, condition, reset, kind),
            "reset": reset,
            "kind": kind,
            "split": split_for(uid),
            "source": {
                "name": "HardSecBench",
                "case": uid,
                "cwe": cid,
                "license": "MIT",
                "path": str(path.relative_to(source)),
                "original_sha256": digest(original),
                "raw_sha256": file_hash(path),
            },
            "hole_width": 1,
            "mutation": "targeted_near_miss",
        }


def opentitan_cases(source):
    """Write-arbitration cases from the local case provider, or an authored example.

    The original source-derived case definitions (requirements, assertions,
    mutations, completion answers, split assignment) live in the local catalog
    ("opentitan_cases"); the public fallback is one authored synthetic case.
    """
    specs = (_CATALOG or {}).get("opentitan_cases")
    if not specs:
        # Synthetic data-priority mux for tests and examples.
        code = (
            "module topmodule(input hw_valid, sw_valid, input [7:0] hw_data, sw_data, "
            "output wire [7:0] out_data);\nassign out_data = hw_valid ? hw_data : sw_data;\nendmodule\n"
        )
        yield {
            "uid": "authored:hw_data_priority",
            "family": "authored:data_priority",
            "topic": "output_arbitration",
            "code": code,
            "guard": "assign out_data = hw_valid ? hw_data : sw_data;",
            "bad_guard": "assign out_data = sw_valid ? sw_data : hw_data;",
            "occurrences": (0,),
            "requirement": "Whenever hw_valid is 1, out_data must equal hw_data; otherwise out_data must equal sw_data.",
            "assertion": "always @* begin assert(out_data == (hw_valid ? hw_data : sw_data)); end",
            "reset": "",
            "kind": "combinational_exact",
            "split": "train",
            "source": {
                "name": "authored-synthetic",
                "revision": "authored",
                "license": "project-authored",
                "path": "authored",
                "raw_sha256": digest(code),
                "specialization": "authored illustrative data-priority case",
            },
            "hole_width": 8,
            "mutation": "priority_swap",
            "good_fill": "hw_valid ? hw_data : sw_data",
            "bad_fill": "sw_valid ? sw_data : hw_data",
            "opaque_guard": "assign out_data = %s;",
        }
        return
    root = source / "opentitan"
    revision = json.loads((root / "revision.json").read_text())["revision"]
    original = (root / "hw/ip/prim/rtl/prim_subreg_arb.sv").read_text()
    for spec in specs:
        en, data = spec["en"], spec["data"]
        if en not in original or data not in original:
            raise ValueError("OpenTitan source changed")
        code = (
            "module topmodule(input we, de, input [7:0] wd, d, q, output wire wr_en, output wire [7:0] wr_data);\n%s\n%s\nendmodule\n"
            % (en, data)
        )
        yield {
            "uid": "opentitan:" + spec["kind"],
            "family": "opentitan:prim_subreg_arb",
            "topic": "software_hardware_permissions",
            "code": code,
            "guard": spec["guard"],
            "bad_guard": spec["bad_guard"],
            "occurrences": (0,),
            "requirement": spec["requirement"],
            "assertion": spec["assertion"],
            "reset": "",
            "kind": "combinational_exact",
            "split": "train",
            "source": {
                "name": "OpenTitan",
                "revision": revision,
                "license": "Apache-2.0",
                "path": "opentitan/hw/ip/prim/rtl/prim_subreg_arb.sv",
                "raw_sha256": file_hash(
                    root / "hw/ip/prim/rtl/prim_subreg_arb.sv"
                ),
                "specialization": "DW=8; explicit RO/RW branch; authored port wrapper",
            },
            "hole_width": spec["width"],
            "mutation": "single_assessed_assignment",
            "good_fill": spec["good_fill"],
            "bad_fill": spec["bad_fill"],
            "opaque_guard": spec["opaque_guard"],
        }


def rename_example(code, requirement, variant):
    identifiers = set(
        re.findall(
            r"\b(?:input|output|reg|wire|logic)\s+(?:(?:wire|reg|logic)\s+)?(?:\[[^\]]+\]\s*)?([A-Za-z_]\w*)",
            code,
        )
    )
    identifiers -= {"clk", "rst_n", "rst", "sys_clk", "sys_rst_n"}
    if variant:
        names = {name: name + "_" + str(variant) for name in identifiers}
        code = re.sub(
            r"\b[A-Za-z_]\w*\b",
            lambda match: names.get(match.group(), match.group()),
            code,
        )
        requirement = re.sub(
            r"\b[A-Za-z_]\w*\b",
            lambda match: names.get(match.group(), match.group()),
            requirement,
        )
    if variant % 3 == 1:
        code = "\n".join(line.strip() for line in code.splitlines())
    elif variant % 3 == 2:
        code = code.replace(" begin", "\n begin").replace(";", ";\n")
    return code, requirement


def render_variant(case, variant):
    relevant = HOLE_NAMES[
        (sum(ord(c) for c in case["uid"]) + variant) % len(HOLE_NAMES)
    ]
    declarations = hole_declaration(case["hole_width"], relevant)
    irrelevant = None
    if variant == 1:
        irrelevant = HOLE_NAMES[
            (HOLE_NAMES.index(relevant) + 1) % len(HOLE_NAMES)
        ]
        declarations += "\nwire %s;\nwire status_monitor = %s;" % (
            irrelevant,
            irrelevant,
        )
    common = insert_declarations(case["code"], declarations)
    good = common
    bad = replace_reviewed(
        common,
        case["guard"],
        case["bad_guard"],
        case["uid"],
        case["occurrences"],
    )
    opaque_fill = case.get("opaque_guard", "%s") % relevant
    opaque = replace_reviewed(
        common, case["guard"], opaque_fill, case["uid"], case["occurrences"]
    )
    marker = "__ASSERTION_BOUNDARY__"
    combined = case["requirement"] + "\n" + marker + "\n" + case["assertion"]
    good, renamed = rename_example(good, combined, variant)
    bad, bad_renamed = rename_example(bad, combined, variant)
    opaque, opaque_renamed = rename_example(opaque, combined, variant)
    if renamed != bad_renamed or renamed != opaque_renamed:
        raise ValueError("variant changed shared requirement/assertion")
    requirement, assertion = renamed.split("\n" + marker + "\n", 1)
    renamed_relevant = (
        relevant if variant == 0 else relevant + "_" + str(variant)
    )
    good_fill = rename_example(
        common, case.get("good_fill", case["guard"]), variant
    )[1]
    bad_fill = rename_example(
        common, case.get("bad_fill", case["bad_guard"]), variant
    )[1]
    if (
        complete_rendered(
            opaque, renamed_relevant, good_fill, case["hole_width"]
        )
        != good
    ):
        raise ValueError(
            "satisfying completion is not byte-exact for " + case["uid"]
        )
    if (
        complete_rendered(
            opaque, renamed_relevant, bad_fill, case["hole_width"]
        )
        != bad
    ):
        raise ValueError(
            "violating completion is not byte-exact for " + case["uid"]
        )
    return {
        "good": good,
        "bad": bad,
        "opaque": opaque,
        "requirement": requirement,
        "assertion": assertion,
        "relevant_hole": renamed_relevant,
        "irrelevant_hole": irrelevant,
        "good_fill": good_fill,
        "bad_fill": bad_fill,
    }


def assumptions(case: dict[str, Any]) -> str:
    if case["reset"] and case["kind"] not in (
        "comb_zero",
        "combinational_exact",
    ):
        text = (
            "Two-state sampled-clock RTL with defined inputs and arbitrary-but-defined initial state. Reset is asserted "
            "for the initial clock cycle and inactive thereafter. Reset takes precedence over the requirement; each "
            "sequential property applies only at rising edges with reset inactive immediately before and after the edge. "
            "Values are observed just after those edges. Only the stated property is assessed. "
        )
    else:
        text = "Two-state digital RTL with defined inputs. Combinational behavior is evaluated after settling. Only the stated property is assessed. "
    return text + (
        "An undriven wire denotes an omitted, side-effect-free combinational expression over the shown signals, "
        "not an independent unconstrained input. Its relevance to the requirement must be assessed; no other code is omitted."
    )


def render_state(case, code, requirement):
    state = "REQUIREMENT\n%s\nASSUMPTIONS\n%s\nRTL\n%s" % (
        requirement,
        assumptions(case),
        code,
    )
    lowered = state.lower()
    if (
        "verified completion" in lowered
        or "satisfying_completion" in lowered
        or "violating_completion" in lowered
    ):
        raise ValueError("completion verdict leaked into input")
    return state


def build_row(case, rendered, variant, label, code, evidence):
    state = render_state(case, code, rendered["requirement"])
    pair = case["uid"] + ":" + str(variant)
    return {
        "id": digest(state)[:24],
        "family_id": case["family"],
        "split_group_id": case["family"],
        "base_design_id": case["uid"],
        "mechanism_id": case["kind"],
        "rewrite_group_id": case["uid"],
        "pair_id": pair,
        "variant": variant,
        "topic": case["topic"],
        "split": case["split"],
        "lexical_stratum": case["mutation"],
        "state": state,
        "questions": QUESTION,
        "expected": {"security": label},
        "validation": evidence,
        "source": case["source"],
    }


def process_case(case, yosys, verification):
    rows = []
    for variant in VARIANTS:
        rendered = render_variant(case, variant)
        good = verify(
            rendered["good"],
            rendered["assertion"],
            yosys,
            verification,
            case["reset"],
        )
        bad = verify(
            rendered["bad"],
            rendered["assertion"],
            yosys,
            verification,
            case["reset"],
        )
        if not good.get("accepted") or good.get("verdict") != LABELS[0]:
            raise ValueError(
                ("satisfying reference failed", case["uid"], variant, good)
            )
        if not bad.get("accepted") or bad.get("verdict") != LABELS[1]:
            raise ValueError(
                ("violating reference failed", case["uid"], variant, bad)
            )
        common = {
            "alpha_renaming": variant,
            "irrelevant_hole": rendered["irrelevant_hole"],
        }
        unknown = {
            "accepted": True,
            "basis": "exact_shared_hole_completions",
            "satisfying_completion": good["evidence_id"],
            "violating_completion": bad["evidence_id"],
            "hole_name": rendered["relevant_hole"],
            "good_completion": "byte_exact",
            "bad_completion": "byte_exact",
            "good_fill_sha256": digest(rendered["good_fill"]),
            "bad_fill_sha256": digest(rendered["bad_fill"]),
            "opaque_code_sha256": digest(rendered["opaque"]),
            **common,
        }
        rows.append(
            build_row(
                case,
                rendered,
                variant,
                LABELS[0],
                rendered["good"],
                dict(good, **common),
            )
        )
        rows.append(
            build_row(
                case,
                rendered,
                variant,
                LABELS[1],
                rendered["bad"],
                dict(bad, **common),
            )
        )
        rows.append(
            build_row(
                case, rendered, variant, LABELS[2], rendered["opaque"], unknown
            )
        )
    return rows


def validate_dataset(splits):
    owners, ids, states = {}, set(), set()
    for split, rows in splits.items():
        labels = Counter()
        pairs = defaultdict(set)
        for row in rows:
            if row["split"] != split:
                raise ValueError("row split ownership mismatch")
            group = row["split_group_id"]
            if owners.setdefault(group, split) != split:
                raise ValueError("split group crosses splits: " + group)
            if row["id"] in ids or row["state"] in states:
                raise ValueError("duplicate row or input across dataset")
            ids.add(row["id"])
            states.add(row["state"])
            label = row["expected"]["security"]
            labels[label] += 1
            pairs[row["pair_id"]].add(label)
        if set(labels) != set(LABELS) or len(set(labels.values())) != 1:
            raise ValueError("split labels are not balanced: " + split)
        if any(set(labels) != set(LABELS) for labels in pairs.values()):
            raise ValueError("incomplete triplet in " + split)
        if (
            split in ("dev", "calibration", "test")
            and len({r["family_id"] for r in rows}) < 8
        ):
            raise ValueError(
                "held-out split has fewer than eight source families: " + split
            )


def canonical_tokens(code):
    return re.findall(r"[A-Za-z_]\w*|\d+|[^\s]", code)


def ngrams(code, size=5):
    tokens = canonical_tokens(code)
    return {
        tuple(tokens[i : i + size])
        for i in range(max(0, len(tokens) - size + 1))
    }


def structural_audit(cases):
    grams = [ngrams(case["code"]) for case in cases]
    nearest = []
    for i, left in enumerate(cases):
        for j in range(i + 1, len(cases)):
            right = cases[j]
            union = grams[i] | grams[j]
            similarity = len(grams[i] & grams[j]) / max(1, len(union))
            if similarity >= 0.55:
                nearest.append(
                    {
                        "left": left["uid"],
                        "right": right["uid"],
                        "left_split": left["split"],
                        "right_split": right["split"],
                        "token_5gram_jaccard": similarity,
                    }
                )
    return sorted(
        nearest, key=lambda item: item["token_5gram_jaccard"], reverse=True
    )


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
        help="New directory for dataset splits and their manifest",
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
        "--iverilog",
        help="Optional Icarus Verilog executable for compilation checks",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel data-preparation workers",
    )
    parser.add_argument(
        "--cases-manifest",
        type=Path,
        help="Local dataset catalog JSON (default: authored example catalog)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Manifest seed (default: catalog seed)",
    )
    args = parser.parse_args()
    catalog = apply_catalog(load_case_catalog(args.cases_manifest))
    if args.output.exists() and (args.output / "manifest.json").exists():
        raise ValueError("output already materialized")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = list(hardsec_cases(args.sources)) + list(
        opentitan_cases(args.sources)
    )
    expected = catalog.get("expected_counts")
    counts = Counter(case["split"] for case in cases)
    if expected and counts != Counter(expected):
        raise ValueError(("frozen split ownership mismatch", counts, expected))
    audit = structural_audit(cases)
    crossing = [
        item for item in audit if item["left_split"] != item["right_split"]
    ]
    if crossing:
        raise ValueError(("structural near-duplicates cross splits", crossing))
    rows = defaultdict(list)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process_case, case, args.yosys, args.verification)
            for case in cases
        ]
        for case, future in zip(cases, futures):
            produced = future.result()
            rows[case["split"]].extend(produced)
    validate_dataset(rows)
    if args.iverilog:
        complete = [
            (
                row["state"].split("RTL\n", 1)[1],
                row["validation"]["evidence_id"],
            )
            for split in rows.values()
            for row in split
            if row["expected"]["security"] != LABELS[2]
        ]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    compile_rtl,
                    code,
                    evidence,
                    args.iverilog,
                    args.verification,
                )
                for code, evidence in complete
            ]
            for future in futures:
                future.result()
    split_hashes = {}
    for split in ("train", "dev", "calibration", "test"):
        random.Random(catalog["manifest_seed"] + len(split)).shuffle(
            rows[split]
        )
        payload = "".join(
            json.dumps(row) + "\n" for row in rows[split]
        ).encode()
        atomic_bytes(args.output / (split + ".jsonl"), payload)
        split_hashes[split] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "seed": catalog["manifest_seed"] if args.seed is None else args.seed,
        "variants": list(VARIANTS),
        "split_ownership_preregistered": bool(expected),
        "rejected_case_backfill": False,
        "expected_case_counts": expected,
        "family_counts": {
            s: len({r["family_id"] for r in rs}) for s, rs in rows.items()
        },
        "row_counts": {s: len(rs) for s, rs in rows.items()},
        "label_counts": {
            s: dict(Counter(r["expected"]["security"] for r in rs))
            for s, rs in rows.items()
        },
        "source_counts": {
            s: dict(Counter(r["source"]["name"] for r in rs))
            for s, rs in rows.items()
        },
        "split_sha256": split_hashes,
        "builder_sha256": file_hash(__file__),
        "solver_version": solver_version(args.yosys),
        "evidence_schema": EVIDENCE_SCHEMA,
        "nonvacuity": "bounded antecedent counterexample required for every conditional satisfying variant",
        "completion_rule": "byte-exact shared-hole reconstruction after every rendered variant",
        "mutation_rule": "one reviewed targeted occurrence with diverse near-miss mutations",
        "structural_similarity_audit": audit,
        "source_archive_sha256": file_hash(
            args.sources / "HardSecBench_Data.zip"
        ),
        "verification_directory": str(args.verification),
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "row_counts",
                    "family_counts",
                    "label_counts",
                    "split_sha256",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
