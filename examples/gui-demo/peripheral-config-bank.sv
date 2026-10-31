// Authored synthetic peripheral configuration design.
// Each block contains one local configuration register.

module clock_config(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic privileged,
    input logic [7:0] wdata,
    output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en)
      cfg_reg <= wdata;
  end
endmodule

module pad_config(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic privileged,
    input logic [7:0] wdata,
    output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en && privileged)
      cfg_reg <= wdata;
  end
endmodule

module alert_config(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic privileged,
    input logic [7:0] wdata,
    output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en)
      cfg_reg <= wdata;
  end
endmodule

module timer_config(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic privileged,
    input logic [7:0] wdata,
    output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en && privileged)
      cfg_reg <= wdata;
  end
endmodule

module irq_config(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic privileged,
    input logic [7:0] wdata,
    output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en)
      cfg_reg <= wdata;
  end
endmodule

module dma_config(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic privileged,
    input logic [7:0] wdata,
    output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en && privileged)
      cfg_reg <= wdata;
  end
endmodule

module bus_config(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic privileged,
    input logic [7:0] wdata,
    output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en)
      cfg_reg <= wdata;
  end
endmodule

module power_config(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic privileged,
    input logic [7:0] wdata,
    output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en && privileged)
      cfg_reg <= wdata;
  end
endmodule

module reset_config(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic privileged,
    input logic [7:0] wdata,
    output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en)
      cfg_reg <= wdata;
  end
endmodule

module memory_config(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic privileged,
    input logic [7:0] wdata,
    output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en && privileged)
      cfg_reg <= wdata;
  end
endmodule

module trace_config(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic privileged,
    input logic [7:0] wdata,
    output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en)
      cfg_reg <= wdata;
  end
endmodule

module watchdog_config(
    input logic clk,
    input logic rst_n,
    input logic wr_en,
    input logic privileged,
    input logic [7:0] wdata,
    output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en && privileged)
      cfg_reg <= wdata;
  end
endmodule
