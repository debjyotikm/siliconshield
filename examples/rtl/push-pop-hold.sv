module topmodule(
  input logic clk, rst_n, push, pop,
  output logic [3:0] count
);
  always_ff @(posedge clk) begin
    if (!rst_n) count <= 4'd0;
    else case ({push, pop})
      2'b10: count <= count + 1'b1;
      2'b01: count <= count - 1'b1;
      default: count <= count;
    endcase
  end
endmodule
