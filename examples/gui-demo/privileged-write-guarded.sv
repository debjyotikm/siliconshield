// Authored synthetic GUI demo (never training data).
// Expected decision: satisfies_requirement
module topmodule(input logic clk, rst_n, wr_en, privileged,
               input logic [7:0] wdata, output logic [7:0] cfg_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) cfg_reg <= 8'h00;
    else if (wr_en && privileged) cfg_reg <= wdata;
  end
endmodule
