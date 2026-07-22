/* -----------------------------------------------------------------------------
MIT License

Copyright (c) 2023 Drexel Distributed, Intelligent, and Scalable COmputing (DISCO) Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
// Author   	: Anup Das
// Email    	: anup.das@drexel.edu
// Date     	: Aug 28, 2023
// File     	: xmem.v
// Desc     	: This is an implementation of a 1D array of memory.
			  This array is built using YDIM memory elements.
			  This parameter can be configured from the top level.
// Changes	: 1.1. Renamed the out_data port as rd_data.
// 		  1.2. Removed the original rd_data port.
// 		  2.1. Removed the read enable and read address ports.
// 		  3.1. Changed the address width to not use one-hot encoding.
// 		       Using block memory.
// 		  3.2. Input spikes from pre-synaptic neurons now reaches this block.
// 		       These spikes are utilized to compute the activation, which is the sum of synaptic weights.
// 		       activation = sum(x_i * w_i).
// 		  3.3. Added serial adder with saturation upon overflow or underflow to perform signed addition of synaptic weights.
// 		  3.4. Use a quantized adder (qadd) to perform addition.
// 		  3.5. Removed rd_data port which is a wide bus.
// 		  3.6. Added a new parameter called WT_PRECISION. This is used as the precision for the weights.
// 		       The original PRECISION parameter is used for all state variables.
// 		  4.1. Changed input port wr_data to be of precision WT_PRECISION.
// -----------------------------------------------------------------------------*/
`timescale 1ns / 1ps

module xmem #(
	parameter YDIM 		= 2,		//Y-dimension
	parameter YWIDTH 	= $clog2(YDIM),	//address width for the memory addresses of fanin of each neuron. 
	parameter WT_PRECISION	= 2,		//weight precision
	parameter PRECISION 	= 8		//precision of state variables	
)(
	input rst,				//reset
	input clk,				//clock
	input wr_en,				//write enable for writing to the synaptic weight memory
	input [YWIDTH-1:0] wr_addr,		//memory address
	input [WT_PRECISION-1:0] wr_data,	//synaptic weight of precision = WT_PRECISION
	input [YDIM-1:0] inspk,			//input spikes from pre-synaptic connections
	output outspk,				//output spike to the LIF module
	output [PRECISION-1:0] activation	//output activation to the LIF module
);

	localparam INT_PRECISION 	= PRECISION - WT_PRECISION;	//integer precision

	reg  [WT_PRECISION-1:0] mem [YDIM-1:0];	//memory as a 2D array. 
	//Because all memory contents are used in parallel (in one clock period), this memory cannot be implemented as block memory.
		
	//writing into the memory. Observe we are not reseting the memory. Synthsis tool cannot optimize memory with reset signals.
	always @(posedge clk) begin
		if (wr_en) begin
			mem[wr_addr] <= wr_data;
		end
	end

	//accumulation using serialized addition
	wire [PRECISION-1:0] psum [YDIM-1:0];	//partial sum
	qalu #(
		.N(PRECISION)
	) qalu_inst (
		.a({PRECISION{1'b0}}),
		.b({ {INT_PRECISION{mem[0][WT_PRECISION-1]}},mem[0] }),
		.en(inspk[0]),
		.sel(1'b0),
		.q(psum[0])
	);

	genvar i;
	generate
		for (i=1; i<YDIM; i=i+1) begin	: alu
			qalu #(
				.N(PRECISION)
			) qalu_inst (
				.a(psum[i-1]),
				.b({ {INT_PRECISION{mem[i][WT_PRECISION-1]}},mem[i] }),
				.en(inspk[i]),
				.sel(1'b0),
				.q(psum[i])
			);
		end
	endgenerate


	//generating output from the module
	assign outspk 		= |inspk;		//OR of all input
	assign activation 	= psum[YDIM-1];		//sum from the last adder
endmodule
