"""Authored fixtures for source-contract extraction and temporal validation."""

import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from siliconshield.training.laya import build_security_data as data
from siliconshield.training.laya.build_source_security import (
    checked_rows,
    exact_antecedent,
    population_counts,
)
from siliconshield.training.laya.source_contracts import candidates

RTL = """module topmodule(input clk, rst_n, hw_en, sw_en, authorized,
input [3:0] hw_data, sw_data, output reg [3:0] value);
always @(posedge clk) begin
if (!rst_n) value <= 0;
else if (hw_en) value <= hw_data;
else if (sw_en && authorized) value <= sw_data;
end
endmodule"""
SOURCE = {
    "name": "authored",
    "case": "example",
    "family": "example",
    "topic": "priority",
}


def test_population_counts_distinguish_files_records_specializations_and_lineage():
    def row(source, record, design, group, rtl):
        return {
            "source": {"name": source, "path": "register.v", "case": record},
            "base_design_id": design,
            "split_group_id": group,
            "state": "RTL\n" + rtl,
        }

    rows = [
        row("authored-a", "record-a", "width-small", "group-a", "small"),
        row("authored-a", "record-a", "width-large", "group-a", "large"),
        row("authored-a", "record-b", "width-large", "group-a", "large"),
        row("authored-b", "record-a", "width-small", "group-b", "other"),
    ]
    assert population_counts(rows) == {
        "decision_rows": 4,
        "source_files": 2,
        "source_records": 3,
        "base_design_ids": 3,
        "split_groups": 2,
        "unique_rtl_bodies": 3,
    }


def extract():
    pytest.importorskip("pyverilog")
    result, error = candidates(RTL, SOURCE)
    assert error is None
    return result


def test_sampled_antecedent_preserves_past_and_reset():
    assertion = data.hold_check("value", "!authorized", "rst_n")
    control = exact_antecedent(assertion)
    assert "$past(!authorized)" in control
    assert "$past(rst_n)" in control
    assert "assert(value ==" not in control
    assert "assert(1'b0);" in control
    assert (
        exact_antecedent("if (a) assert(q == $past(d)); done = 1;")
        == "if (a) assert(1'b0); done = 1;"
    )


def test_hardware_override_not_discarded_from_hold_contract():
    result = extract()
    holds = [
        c for c in result if c["kind"] == "hold" and c["policy"] == "original"
    ]
    assert holds
    for case in holds:
        assert "hw_en" in case["assertion"] and "sw_en" in case["assertion"]
        assert "authorized" in case["assertion"]
    updates = [c for c in result if c["kind"] == "update"]
    assert any("hw_data" in c["assertion"] for c in updates)
    assert any("sw_data" in c["assertion"] for c in updates)


def test_counterfactuals_and_missing_expressions_are_exact():
    result = extract()
    assert {c["policy"] for c in result} == {"original", "counterfactual"}
    for case in result:
        for variant in data.VARIANTS:
            rendered = data.render_variant(case, variant)
            assert rendered["good"] != rendered["bad"]
            assert rendered["good"] != rendered["opaque"]
            # render_variant checks both completions after identifier rewriting.
            assert rendered["relevant_hole"] in rendered["opaque"]


def test_unsupported_reset_is_not_silently_guessed():
    pytest.importorskip("pyverilog")
    result, error = candidates(RTL.replace("rst_n", "reset_custom"), SOURCE)
    assert not result and error == "requires_supported_clock_and_reset"


def test_additional_clock_or_async_reset_requires_another_model():
    pytest.importorskip("pyverilog")
    extra_clock = RTL.replace(
        "endmodule", "always @(posedge other_clk) value <= 0; endmodule"
    )
    result, error = candidates(extra_clock, SOURCE)
    assert not result and error == "unsupported_additional_clock_or_async_reset"
    extra_reset = RTL.replace(
        "posedge clk)", "posedge clk or negedge warm_reset)"
    )
    result, error = candidates(extra_reset, SOURCE)
    assert not result and error == "unsupported_additional_clock_or_async_reset"


@pytest.mark.skipif(
    not (shutil.which("yosys") and shutil.which("iverilog")),
    reason="requires Yosys and Icarus",
)
def test_solver_proves_both_policies_and_compiles_opaque_rows(tmp_path):
    result = extract()
    pair = [
        c
        for c in result
        if c["kind"] == "update" and "hw_data" in c["assertion"]
    ][:2]
    assert len(pair) == 2
    for case in pair:
        case["split"] = "train"
    # Repeated callers share content-addressed proof and compilation directories.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda c: checked_rows(
                    c, shutil.which("yosys"), shutil.which("iverilog"), tmp_path
                ),
                pair + pair,
            )
        )
    for rows in results:
        assert len(rows) == 9
        assert {r["expected"]["security"] for r in rows} == set(data.LABELS)
        assert all(r["validation"]["compile_evidence"] for r in rows)
        assert all(r["validation"]["exact_antecedent_evidence"] for r in rows)
    for variant in data.VARIANTS:
        left = next(
            r
            for r in results[0]
            if r["variant"] == variant
            and r["expected"]["security"] == data.LABELS[0]
        )
        right = next(
            r
            for r in results[1]
            if r["variant"] == variant
            and r["expected"]["security"] == data.LABELS[1]
        )
        assert (
            left["state"].split("RTL\n")[1] == right["state"].split("RTL\n")[1]
        )
        assert (
            left["state"].split("RTL\n")[0] != right["state"].split("RTL\n")[0]
        )
        assert left["pair_id"] != right["pair_id"]
