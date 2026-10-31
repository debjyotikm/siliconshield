// Authored synthetic GUI demo (never training data).
// Expected decision: violates_requirement
module topmodule(input logic clk, rst_n, wr_en,
               input logic [7:0] addr, wdata, output logic [7:0] secure_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) secure_reg <= 8'h00;
    else if (wr_en && addr[7:4] == 4'hA) secure_reg <= wdata;
  end
endmodule
