// Authored synthetic GUI demo (never training data).
// Expected decision: violates_requirement
module topmodule(input logic clk, rst_n, hw_set, clear_req, clear_enable,
               output logic status_reg);
  always_ff @(posedge clk) begin
    if (!rst_n) status_reg <= 1'b0;
    else if (hw_set) status_reg <= 1'b1;
    else if (clear_req) status_reg <= 1'b0;
  end
endmodule
