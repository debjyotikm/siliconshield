"""Authored GUI demo presets for the model's validated showcase scope.

The main dropdown uses three synthetic counterfactual pairs. Each pair keeps
the plain-English requirement fixed and changes one security-relevant RTL
guard. The large-file dropdown uses one label-free peripheral design under a
single privilege requirement. These examples are teaching fixtures, never
training data. Older lock and mixed cases remain regression fixtures on disk.
"""

from typing import Any

_SAMPLED_CLOCK_ASSUMPTIONS = (
    "Two-state sampled-clock RTL. rst_n is low initially and high "
    "thereafter. Values are observed just after rising clock edges. "
    "Reset behavior is excluded after reset completes."
)

_CLEAR_ASSUMPTIONS = (
    "Two-state sampled-clock RTL. rst_n is low initially and high "
    "thereafter. hw_set may set status_reg. Reset behavior is excluded "
    "after reset completes."
)

_STRENGTH_SHOWCASE: tuple[dict[str, str], ...] = (
    {
        "id": "privilege-check-missing",
        "title": "Should violate - missing privilege guard",
        "expected": "violates_requirement",
        "requirement": (
            "After reset, cfg_reg may change only on a write performed while "
            "privileged is 1. A write while privileged is 0 must leave "
            "cfg_reg unchanged."
        ),
        "assumptions": _SAMPLED_CLOCK_ASSUMPTIONS,
        "rtl": (
            "module topmodule(input logic clk, rst_n, wr_en, privileged,\n"
            "               input logic [7:0] wdata, output logic [7:0] cfg_reg);\n"
            "  always_ff @(posedge clk) begin\n"
            "    if (!rst_n) cfg_reg <= 8'h00;\n"
            "    else if (wr_en) cfg_reg <= wdata;\n"
            "  end\n"
            "endmodule\n"
        ),
    },
    {
        "id": "privileged-write-guarded",
        "title": "Should satisfy - privilege guard enforced",
        "expected": "satisfies_requirement",
        "requirement": (
            "After reset, cfg_reg may change only on a write performed while "
            "privileged is 1. A write while privileged is 0 must leave "
            "cfg_reg unchanged."
        ),
        "assumptions": _SAMPLED_CLOCK_ASSUMPTIONS,
        "rtl": (
            "module topmodule(input logic clk, rst_n, wr_en, privileged,\n"
            "               input logic [7:0] wdata, output logic [7:0] cfg_reg);\n"
            "  always_ff @(posedge clk) begin\n"
            "    if (!rst_n) cfg_reg <= 8'h00;\n"
            "    else if (wr_en && privileged) cfg_reg <= wdata;\n"
            "  end\n"
            "endmodule\n"
        ),
    },
    {
        "id": "clear-enable-omitted",
        "title": "Should violate - clear authorization omitted",
        "expected": "violates_requirement",
        "requirement": (
            "After reset, status_reg may be cleared only when clear_req and "
            "clear_enable are both 1. Otherwise a clear request must leave "
            "status_reg unchanged."
        ),
        "assumptions": _CLEAR_ASSUMPTIONS,
        "rtl": (
            "module topmodule(input logic clk, rst_n, hw_set, clear_req, clear_enable,\n"
            "               output logic status_reg);\n"
            "  always_ff @(posedge clk) begin\n"
            "    if (!rst_n) status_reg <= 1'b0;\n"
            "    else if (hw_set) status_reg <= 1'b1;\n"
            "    else if (clear_req) status_reg <= 1'b0;\n"
            "  end\n"
            "endmodule\n"
        ),
    },
    {
        "id": "clear-enable-enforced",
        "title": "Should satisfy - clear authorization enforced",
        "expected": "satisfies_requirement",
        "requirement": (
            "After reset, status_reg may be cleared only when clear_req and "
            "clear_enable are both 1. Otherwise a clear request must leave "
            "status_reg unchanged."
        ),
        "assumptions": _CLEAR_ASSUMPTIONS,
        "rtl": (
            "module topmodule(input logic clk, rst_n, hw_set, clear_req, clear_enable,\n"
            "               output logic status_reg);\n"
            "  always_ff @(posedge clk) begin\n"
            "    if (!rst_n) status_reg <= 1'b0;\n"
            "    else if (hw_set) status_reg <= 1'b1;\n"
            "    else if (clear_req && clear_enable) status_reg <= 1'b0;\n"
            "  end\n"
            "endmodule\n"
        ),
    },
    {
        "id": "address-check-too-broad",
        "title": "Should violate - address window too broad",
        "expected": "violates_requirement",
        "requirement": (
            "After reset, secure_reg may change on a software write only when "
            "addr equals 8'hA0. Writes at every other address must leave "
            "secure_reg unchanged."
        ),
        "assumptions": _SAMPLED_CLOCK_ASSUMPTIONS,
        "rtl": (
            "module topmodule(input logic clk, rst_n, wr_en,\n"
            "               input logic [7:0] addr, wdata, output logic [7:0] secure_reg);\n"
            "  always_ff @(posedge clk) begin\n"
            "    if (!rst_n) secure_reg <= 8'h00;\n"
            "    else if (wr_en && addr[7:4] == 4'hA) secure_reg <= wdata;\n"
            "  end\n"
            "endmodule\n"
        ),
    },
    {
        "id": "exact-address-guard",
        "title": "Should satisfy - exact address guard",
        "expected": "satisfies_requirement",
        "requirement": (
            "After reset, secure_reg may change on a software write only when "
            "addr equals 8'hA0. Writes at every other address must leave "
            "secure_reg unchanged."
        ),
        "assumptions": _SAMPLED_CLOCK_ASSUMPTIONS,
        "rtl": (
            "module topmodule(input logic clk, rst_n, wr_en,\n"
            "               input logic [7:0] addr, wdata, output logic [7:0] secure_reg);\n"
            "  always_ff @(posedge clk) begin\n"
            "    if (!rst_n) secure_reg <= 8'h00;\n"
            "    else if (wr_en && addr == 8'hA0) secure_reg <= wdata;\n"
            "  end\n"
            "endmodule\n"
        ),
    },
)


def list_demos() -> list[dict[str, Any]]:
    """Return validated-scope showcase presets in dropdown order."""
    return [dict(preset) for preset in _STRENGTH_SHOWCASE]


def _legacy_regression_demos() -> list[dict[str, Any]]:
    """Return the original six synthetic cases for regression fixtures."""
    from siliconshield.app.demo_cases import CASES

    order = (
        "unauthorized-write",
        "lock-bypass-second-path",
        "legitimate-access",
        "missing-context",
    )
    demos: list[dict[str, Any]] = []
    for demo_id in order:
        case = CASES.get(demo_id)
        if not isinstance(case, dict):
            continue
        demos.append(
            {
                "id": demo_id,
                "title": str(case.get("title", demo_id)),
                "requirement": str(case.get("requirement", "")),
                "assumptions": str(case.get("assumptions", "")),
                "rtl": str(case.get("rtl", "")),
            }
        )
    legacy_satisfying = (
        {
            "id": "debug-lock-respected",
            "title": "Should satisfy - debug lock respected",
            "requirement": _LOCK_REQUIREMENT,
            "assumptions": _LOCK_ASSUMPTIONS,
            "rtl": (
                "module topmodule(input logic clk, rst_n, wr_en, dbg_wr, locked,\n"
                "               input logic [7:0] wdata, dbg_data, output logic [7:0] cfg_reg);\n"
                "  always_ff @(posedge clk) begin\n"
                "    if (!rst_n) cfg_reg <= 8'h00;\n"
                "    else if (dbg_wr && !locked) cfg_reg <= dbg_data;\n"
                "    else if (wr_en && !locked) cfg_reg <= wdata;\n"
                "  end\n"
                "endmodule\n"
            ),
        },
        {
            "id": "write-once-then-locked",
            "title": "Should satisfy - write once then locked",
            "requirement": (
                "Once locked is 1, cfg_reg must never change. "
                "While locked is 0, writes are allowed."
            ),
            "assumptions": _LOCK_ASSUMPTIONS,
            "rtl": (
                "module topmodule(input logic clk, rst_n, wr_en, locked,\n"
                "               input logic [7:0] wdata, output logic [7:0] cfg_reg);\n"
                "  always_ff @(posedge clk) begin\n"
                "    if (!rst_n) cfg_reg <= 8'h00;\n"
                "    else if (wr_en && !locked) cfg_reg <= wdata;\n"
                "  end\n"
                "endmodule\n"
            ),
        },
    )
    demos.extend([dict(preset) for preset in legacy_satisfying])
    return demos


def get_demo(demo_id: str) -> dict[str, Any]:
    """Return one demo preset by id, or raise ValueError."""
    for demo in list_demos():
        if demo["id"] == demo_id:
            return demo
    raise ValueError("Unknown demo: %r." % (demo_id,))


_LOCK_REQUIREMENT = (
    "cfg_reg may change only when locked is 0. No write path, "
    "including the debug path, may modify cfg_reg while locked is 1."
)

_LOCK_ASSUMPTIONS = (
    "Two-state sampled-clock RTL; rst_n is low initially and high "
    "thereafter; values are observed just after rising clock edges. "
    "Only this requirement is assessed."
)

_PERIPHERAL_REQUIREMENT = (
    "After reset, cfg_reg may change on a software write only when "
    "privileged is 1. A software write while privileged is 0 must leave "
    "cfg_reg unchanged."
)

_PERIPHERAL_ASSUMPTIONS = (
    "Two-state sampled-clock RTL. rst_n is low initially and high "
    "thereafter. Values are observed just after rising clock edges. "
    "Reset assignments are exempt. Each assessed module contains every "
    "writer of its cfg_reg."
)

_PERIPHERAL_MODULES: tuple[tuple[str, str], ...] = (
    ("clock_config", "wr_en"),
    ("pad_config", "wr_en && privileged"),
    ("alert_config", "wr_en"),
    ("timer_config", "wr_en && privileged"),
    ("irq_config", "wr_en"),
    ("dma_config", "wr_en && privileged"),
    ("bus_config", "wr_en"),
    ("power_config", "wr_en && privileged"),
    ("reset_config", "wr_en"),
    ("memory_config", "wr_en && privileged"),
    ("trace_config", "wr_en"),
    ("watchdog_config", "wr_en && privileged"),
)


def _peripheral_config_rtl() -> str:
    """Build a neutral multi-module configuration design."""
    header = (
        "// Authored synthetic peripheral configuration design.\n"
        "// Each block contains one local configuration register.\n\n"
    )
    blocks = []
    for name, write_condition in _PERIPHERAL_MODULES:
        blocks.append(
            "module %s(\n"
            "    input logic clk,\n"
            "    input logic rst_n,\n"
            "    input logic wr_en,\n"
            "    input logic privileged,\n"
            "    input logic [7:0] wdata,\n"
            "    output logic [7:0] cfg_reg\n"
            ");\n"
            "  always_ff @(posedge clk) begin\n"
            "    if (!rst_n)\n"
            "      cfg_reg <= 8'h00;\n"
            "    else if (%s)\n"
            "      cfg_reg <= wdata;\n"
            "  end\n"
            "endmodule\n" % (name, write_condition)
        )
    return header + "\n".join(blocks)


_BIG_FAULTY_MODULES: tuple[tuple[str, str], ...] = (
    ("VIOLATION", "faulty_dbg_0"),
    ("VIOLATION", "faulty_dbg_1"),
    ("VIOLATION", "faulty_dbg_2"),
    ("VIOLATION", "faulty_main_3"),
    ("VIOLATION", "faulty_main_4"),
    ("VIOLATION", "faulty_main_5"),
    ("CLEAN", "clean_both_6"),
    ("CLEAN", "clean_both_7"),
    ("CLEAN", "clean_both_8"),
    ("CLEAN", "clean_once_9"),
    ("CLEAN", "clean_once_10"),
    ("CLEAN", "clean_once_11"),
)


def _faulty_module(kind: str, name: str, index: int) -> str:
    """Render one synthetic module: violating or clean by construction."""
    if kind == "VIOLATION" and "dbg" in name:
        body = (
            "    else if (dbg_wr) cfg_reg <= dbg_data;\n"
            "    else if (wr_en && !locked) cfg_reg <= wdata;\n"
        )
        ports = "wr_en, dbg_wr, locked"
        data = "wdata, dbg_data"
    elif kind == "VIOLATION":
        body = "    else if (wr_en) cfg_reg <= wdata;\n"
        ports = "wr_en, locked"
        data = "wdata"
    elif "once" in name:
        body = "    else if (wr_en && !locked) cfg_reg <= wdata;\n"
        ports = "wr_en, locked"
        data = "wdata"
    else:
        body = (
            "    else if (dbg_wr && !locked) cfg_reg <= dbg_data;\n"
            "    else if (wr_en && !locked) cfg_reg <= wdata;\n"
        )
        ports = "wr_en, dbg_wr, locked"
        data = "wdata, dbg_data"
    return (
        "// %s %d: %s\n"
        "module %s(input logic clk, rst_n, %s,\n"
        "               input logic [7:0] %s, output logic [7:0] cfg_reg);\n"
        "  always_ff @(posedge clk) begin\n"
        "    if (!rst_n) cfg_reg <= 8'h00;\n"
        "%s"
        "  end\n"
        "endmodule\n"
        % (
            kind,
            index,
            "debug path ignores lock"
            if kind == "VIOLATION" and "dbg" in name
            else "main path ignores lock"
            if kind == "VIOLATION"
            else "respects lock",
            name,
            ports,
            data,
            body,
        )
    )


def _big_faulty_rtl() -> str:
    """Build the 12-module faulty design (6 violate, 6 clean)."""
    header = (
        "// Large synthetic faulty design for decision-model scan testing.\n"
        "// NEVER training data: authored fixtures only. 12 self-contained\n"
        "// modules: 6 VIOLATE the locked-requirement by construction, 6\n"
        "// SATISFY it. Pair with the locked requirement.\n"
    )
    return header + "\n".join(
        _faulty_module(kind, name, index)
        for index, (kind, name) in enumerate(_BIG_FAULTY_MODULES)
    )


def _big_mixed_rtl() -> str:
    """Concatenate the six regression demos as uniquely-named modules."""
    import re

    parts = [
        "// Large synthetic mixed design for decision-model scan testing.\n"
        "// NEVER training data: concatenation of the six authored single-file\n"
        "// demos with renamed modules. Pair with any of their requirements.\n"
    ]
    for demo in _legacy_regression_demos():
        name = "m_" + re.sub(r"[^a-z0-9]+", "_", demo["id"]).strip("_")
        renamed = demo["rtl"].replace("module topmodule", "module " + name, 1)
        parts.append(
            "// ===== demo:%s | %s =====\n%s"
            % (demo["id"], demo["title"], renamed)
        )
    return "\n".join(parts)


def list_scan_demos() -> list[dict[str, Any]]:
    """Return large-file scan presets (synthetic, never training data)."""
    return [
        {
            "id": "peripheral-config-bank",
            "title": "Peripheral configuration bank",
            "requirement": _PERIPHERAL_REQUIREMENT,
            "assumptions": _PERIPHERAL_ASSUMPTIONS,
            "rtl": _peripheral_config_rtl(),
        },
    ]


def get_scan_demo(demo_id: str) -> dict[str, Any]:
    """Return one large-file scan preset by id, or raise ValueError."""
    for demo in list_scan_demos():
        if demo["id"] == demo_id:
            return demo
    raise ValueError("Unknown scan demo: %r." % (demo_id,))
