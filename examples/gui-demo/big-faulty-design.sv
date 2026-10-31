// Authored synthetic GUI scan demo (never training data).
// Title: Big faulty design — 12 modules, 6 violations
// Large synthetic faulty design for decision-model scan testing.
// NEVER training data: authored fixtures only. 12 self-contained
// modules: 6 VIOLATE the locked-requirement by construction, 6
// SATISFY it. Pair with the locked requirement.
// VIOLATION 0: debug path ignores lock
module faulty_dbg_0(input logic clk, rst_n, wr_en, dbg_wr, locked,
               input logic [7:0] wdata, dbg_data, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (dbg_wr) cfg_reg <= dbg_data;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule

// VIOLATION 1: debug path ignores lock
module faulty_dbg_1(input logic clk, rst_n, wr_en, dbg_wr, locked,
               input logic [7:0] wdata, dbg_data, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (dbg_wr) cfg_reg <= dbg_data;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule

// VIOLATION 2: debug path ignores lock
module faulty_dbg_2(input logic clk, rst_n, wr_en, dbg_wr, locked,
               input logic [7:0] wdata, dbg_data, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (dbg_wr) cfg_reg <= dbg_data;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule

// VIOLATION 3: main path ignores lock
module faulty_main_3(input logic clk, rst_n, wr_en, locked,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en) cfg_reg <= wdata;
  end
endmodule

// VIOLATION 4: main path ignores lock
module faulty_main_4(input logic clk, rst_n, wr_en, locked,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en) cfg_reg <= wdata;
  end
endmodule

// VIOLATION 5: main path ignores lock
module faulty_main_5(input logic clk, rst_n, wr_en, locked,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en) cfg_reg <= wdata;
  end
endmodule

// CLEAN 6: respects lock
module clean_both_6(input logic clk, rst_n, wr_en, dbg_wr, locked,
               input logic [7:0] wdata, dbg_data, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (dbg_wr && !locked) cfg_reg <= dbg_data;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule

// CLEAN 7: respects lock
module clean_both_7(input logic clk, rst_n, wr_en, dbg_wr, locked,
               input logic [7:0] wdata, dbg_data, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (dbg_wr && !locked) cfg_reg <= dbg_data;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule

// CLEAN 8: respects lock
module clean_both_8(input logic clk, rst_n, wr_en, dbg_wr, locked,
               input logic [7:0] wdata, dbg_data, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (dbg_wr && !locked) cfg_reg <= dbg_data;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule

// CLEAN 9: respects lock
module clean_once_9(input logic clk, rst_n, wr_en, locked,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule

// CLEAN 10: respects lock
module clean_once_10(input logic clk, rst_n, wr_en, locked,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule

// CLEAN 11: respects lock
module clean_once_11(input logic clk, rst_n, wr_en, locked,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule
