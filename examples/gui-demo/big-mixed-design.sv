// Authored synthetic GUI scan demo (never training data).
// Title: Big mixed design — 6 renamed demos
// Large synthetic mixed design for decision-model scan testing.
// NEVER training data: concatenation of the six authored single-file
// demos with renamed modules. Pair with any of their requirements.

// ===== demo:unauthorized-write | Unauthorized register write (missing privilege check) =====
module m_unauthorized_write(input logic clk, rst_n, wr_en, privileged,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en) cfg_reg <= wdata;
  end
endmodule

// ===== demo:lock-bypass-second-path | Lock bypass through a second (debug) write path =====
module m_lock_bypass_second_path(input logic clk, rst_n, wr_en, dbg_wr, locked,
               input logic [7:0] wdata, dbg_data, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (dbg_wr) cfg_reg <= dbg_data;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule

// ===== demo:legitimate-access | Legitimate access that should be accepted =====
module m_legitimate_access(input logic clk, rst_n, wr_en, privileged,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en && privileged) cfg_reg <= wdata;
  end
endmodule

// ===== demo:missing-context | Missing context that should remain unresolved =====
module m_missing_context(input logic clk, rst_n, wr_en,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en && auth_ok) cfg_reg <= wdata;
  end
endmodule

// ===== demo:debug-lock-respected | Should satisfy - debug lock respected =====
module m_debug_lock_respected(input logic clk, rst_n, wr_en, dbg_wr, locked,
               input logic [7:0] wdata, dbg_data, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (dbg_wr && !locked) cfg_reg <= dbg_data;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule

// ===== demo:write-once-then-locked | Should satisfy - write once then locked =====
module m_write_once_then_locked(input logic clk, rst_n, wr_en, locked,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule
