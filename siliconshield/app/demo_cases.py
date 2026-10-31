"""Authored demonstration cases for the RTL security workflow.

Small, self-contained SystemVerilog designs. Each case carries the explicit
requirement, assumptions, protected/control signals, an executable check
(Verilator testbench printing CHECK_PASS/CHECK_FAIL), a regression check, and
an authored candidate patch as exact text replacements. These are synthetic
teaching cases, not excerpts from any experiment dataset.
"""

CASES = {}


def _case(name, **fields):
    CASES[name] = {"name": name, **fields}


_TB_CLK = """module tb;
  logic clk=0, rst_n=0;
  always #5 clk = ~clk;
"""

_case(
    "unauthorized-write",
    title="Unauthorized register write (missing privilege check)",
    requirement="cfg_reg may change only when privileged is 1. An unprivileged write must leave cfg_reg unchanged.",
    assumptions="Two-state sampled-clock RTL; rst_n is low initially and high thereafter; "
    "values are observed just after rising clock edges. Only this requirement is assessed.",
    protected_signals=["cfg_reg"],
    control_signals=["wr_en", "privileged"],
    rtl="""module topmodule(input logic clk, rst_n, wr_en, privileged,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en) cfg_reg <= wdata;
  end
endmodule
""",
    checks=[
        {
            "name": "unprivileged_write_blocked",
            "kind": "requirement",
            "testbench": _TB_CLK
            + """  logic wr_en=0, privileged=0; logic [7:0] wdata=0, cfg_reg;
  topmodule dut(.clk(clk), .rst_n(rst_n), .wr_en(wr_en), .privileged(privileged), .wdata(wdata), .cfg_reg(cfg_reg));
  initial begin
    #12 rst_n=1; wr_en=1; privileged=0; wdata=8'hA5;
    @(posedge clk); #1;
    if (cfg_reg === 8'h00) $display("CHECK_PASS cfg_reg held at %h", cfg_reg);
    else $display("CHECK_FAIL cfg_reg changed to %h by unprivileged write", cfg_reg);
    $finish;
  end
endmodule
""",
        },
        {
            "name": "privileged_write_accepted",
            "kind": "regression",
            "testbench": _TB_CLK
            + """  logic wr_en=0, privileged=0; logic [7:0] wdata=0, cfg_reg;
  topmodule dut(.clk(clk), .rst_n(rst_n), .wr_en(wr_en), .privileged(privileged), .wdata(wdata), .cfg_reg(cfg_reg));
  initial begin
    #12 rst_n=1; wr_en=1; privileged=1; wdata=8'h3C;
    @(posedge clk); #1;
    if (cfg_reg === 8'h3C) $display("CHECK_PASS privileged write applied, cfg_reg=%h", cfg_reg);
    else $display("CHECK_FAIL privileged write lost, cfg_reg=%h", cfg_reg);
    $finish;
  end
endmodule
""",
        },
    ],
    patch=[
        (
            "else if (wr_en) cfg_reg <= wdata;",
            "else if (wr_en && privileged) cfg_reg <= wdata;",
        )
    ],
    expected_original="violation",
    expected_patched="pass",
)

_case(
    "lock-bypass-second-path",
    title="Lock bypass through a second (debug) write path",
    requirement="cfg_reg may change only when locked is 0. No write path, including the debug path, "
    "may modify cfg_reg while locked is 1.",
    assumptions="Two-state sampled-clock RTL; rst_n is low initially and high thereafter; "
    "values are observed just after rising clock edges. Only this requirement is assessed.",
    protected_signals=["cfg_reg"],
    control_signals=["wr_en", "dbg_wr", "locked"],
    rtl="""module topmodule(input logic clk, rst_n, wr_en, dbg_wr, locked,
               input logic [7:0] wdata, dbg_data, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (dbg_wr) cfg_reg <= dbg_data;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule
""",
    checks=[
        {
            "name": "locked_debug_write_blocked",
            "kind": "requirement",
            "testbench": _TB_CLK
            + """  logic wr_en=0, dbg_wr=0, locked=0; logic [7:0] wdata=0, dbg_data=0, cfg_reg;
  topmodule dut(.clk(clk), .rst_n(rst_n), .wr_en(wr_en), .dbg_wr(dbg_wr), .locked(locked),
                .wdata(wdata), .dbg_data(dbg_data), .cfg_reg(cfg_reg));
  initial begin
    #12 rst_n=1; locked=1; dbg_wr=1; dbg_data=8'hE7;
    @(posedge clk); #1;
    if (cfg_reg === 8'h00) $display("CHECK_PASS cfg_reg held at %h while locked", cfg_reg);
    else $display("CHECK_FAIL debug path wrote %h while locked", cfg_reg);
    $finish;
  end
endmodule
""",
        },
        {
            "name": "unlocked_write_accepted",
            "kind": "regression",
            "testbench": _TB_CLK
            + """  logic wr_en=0, dbg_wr=0, locked=0; logic [7:0] wdata=0, dbg_data=0, cfg_reg;
  topmodule dut(.clk(clk), .rst_n(rst_n), .wr_en(wr_en), .dbg_wr(dbg_wr), .locked(locked),
                .wdata(wdata), .dbg_data(dbg_data), .cfg_reg(cfg_reg));
  initial begin
    #12 rst_n=1; locked=0; wr_en=1; wdata=8'h5A;
    @(posedge clk); #1;
    if (cfg_reg === 8'h5A) $display("CHECK_PASS unlocked write applied, cfg_reg=%h", cfg_reg);
    else $display("CHECK_FAIL unlocked write lost, cfg_reg=%h", cfg_reg);
    $finish;
  end
endmodule
""",
        },
    ],
    patch=[
        (
            "else if (dbg_wr) cfg_reg <= dbg_data;",
            "else if (dbg_wr && !locked) cfg_reg <= dbg_data;",
        )
    ],
    expected_original="violation",
    expected_patched="pass",
)

_case(
    "legitimate-access",
    title="Legitimate access that should be accepted",
    requirement="cfg_reg may change only when privileged is 1. An unprivileged write must leave cfg_reg unchanged.",
    assumptions="Two-state sampled-clock RTL; rst_n is low initially and high thereafter; "
    "values are observed just after rising clock edges. Only this requirement is assessed.",
    protected_signals=["cfg_reg"],
    control_signals=["wr_en", "privileged"],
    rtl="""module topmodule(input logic clk, rst_n, wr_en, privileged,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en && privileged) cfg_reg <= wdata;
  end
endmodule
""",
    checks=None,  # reuses unauthorized-write checks below
    patch=[],
    expected_original="pass",
    expected_patched="pass",
)
CASES["legitimate-access"]["checks"] = CASES["unauthorized-write"]["checks"]

_case(
    "missing-context",
    title="Missing context that should remain unresolved",
    requirement="cfg_reg may change only when the access is authorized. An unauthorized write must leave cfg_reg unchanged.",
    assumptions="Two-state sampled-clock RTL; rst_n is low initially and high thereafter. "
    "The authorization policy itself is not supplied with this case.",
    protected_signals=["cfg_reg"],
    control_signals=["wr_en"],
    rtl="""module topmodule(input logic clk, rst_n, wr_en,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en && auth_ok) cfg_reg <= wdata;
  end
endmodule
""",
    checks=[CASES["unauthorized-write"]["checks"][0]],
    patch=[],
    expected_original="inconclusive",
    expected_patched="inconclusive",
)
