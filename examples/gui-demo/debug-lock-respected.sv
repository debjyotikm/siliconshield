// Authored synthetic GUI demo (never training data).
// Title: Should satisfy - debug lock respected
module topmodule(input logic clk, rst_n, wr_en, dbg_wr, locked,
               input logic [7:0] wdata, dbg_data, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (dbg_wr && !locked) cfg_reg <= dbg_data;
    else if (wr_en && !locked) cfg_reg <= wdata;
  end
endmodule
