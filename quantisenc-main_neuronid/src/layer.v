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
// Date     	: Feb 09, 2023
// File     	: layer.v
// Desc     	: This is a layer, which instantiates programmable number of neurons.
// Changes	: 1.1. Cleaned up this file to rename out_data port as rd_data. 
// 		  1.2. The original rd_data port is removed.
// 		  1.3. Facilitate monitoring the internal membrane potential of only the first neuron of the layer.
// 		  2.1. A new implementation of a layer, which instantiates cuba_lif module.
// 		       This module has the definition of a lif neuron and memory to perform cuba integration.
// 		       Essentially, we have made a distributed design.
// 		  3.1. Added parameter RESET_MECHANISM.
// 		       This parameter controls how the membrane potential changes upon firing a spike.
// 		  4.1. Changed the width of X & Y address bus. No use of one-hot encoding at IO level.
// 		       Once the address is received, X-address is converted into one-hot encoding internally.
// 		       The Y-address is forwarded to the cuba_lif module.
// 		  4.2. Added parameter WT_PRECISION for the synaptic weights.
// 		       Added parameter XWIDTH for the address of neurons in the layer.
// 		       Added parameter YWIDTH for the address of pre-synaptic connections of each neuron in the layer.
// 		  5.1. Added memclk, a seperate fast clock for reading synaptic weights from memory.
// 		       If MEM_CLK primitive is defined in the header, then the main clock clk is used for processing neurons and memclk for synaptic weights.
// 		       Otherwise, clk is used for both.
// 		  6.1. Added LIF parameters as input ports. These parameters are obtained from configuration registers.
// 		  6.2. Removed the following parameters from the code.
// 		       VTH, DECAY_RATE, GROW_RATE, VREST, RESET_MECHANISM, REFRACTORY_PERIOD from cuba_lif.
// 		  6.3. Added port neuron_mon to monitor the membrane potential of a neuron in the layer.
// 		  7.1. Rename cmslk to spkclk.
// 		  7.2. Rename parameter LAYER_NEURONS to FANOUT.
// 		  7.3. The write_addr port is made ADDR_WIDTH bits wide.
// 		  7.4. Implemented a parameterized decoder for address decoding.
// -----------------------------------------------------------------------------*/
`timescale 1ns / 1ps

//`include "defines.vh"

module layer #(
	//configuration parameters
	parameter FANIN 		= 256,	//fanin of each neuron of the layer
	parameter FANOUT 		= 256,	//number of neurons in the layer
	parameter INTEGER_PRECISION	= 3,	//integer precision
	parameter DECIMAL_PRECISION 	= 4,	//decimal precision
	parameter ADDR_WIDTH_NEURON 	= 10,	//addr width for neuron
	parameter ADDR_WIDTH_FANIN 	= 10,	//addr width for fanin	
	//local parameters
	localparam ADDR_WIDTH 		= (ADDR_WIDTH_NEURON+ADDR_WIDTH_FANIN),		//address width
	localparam FANIN_WIDTH		= $clog2(FANIN),				//log2 of FANIN
	localparam FANOUT_WIDTH		= $clog2(FANOUT),				//log2 of FANOUT
	localparam WT_PRECISION		= (1+DECIMAL_PRECISION),			//bit precision for synaptic weights
	localparam PRECISION 		= (1+INTEGER_PRECISION+DECIMAL_PRECISION)	//bit precision for state variables
)(
	input rst,				//reset
	input memclk,				//memory clock
	input spkclk,				//spike clock
	//neuron parameters from configuration registers
	input [PRECISION-1:0] vth,		//neuron threshold voltage
	input [PRECISION-1:0] decay_rate,	//membrane decay rate
	input [PRECISION-1:0] grow_rate,	//membrane grow rate
	input [PRECISION-1:0] vrest,		//neuron resting potential
	input [PRECISION-1:0] reset_mechanism,	//neuron reset mechanism
	input [PRECISION-1:0] refractory_period,//neuron refractory period
	input wr_en,				//write enable to synaptic memory
	input [ADDR_WIDTH-1:0] wr_addr,		//write address to synaptic memory
	input [WT_PRECISION-1:0] wr_data,	//write data (weights) to synaptic memory of precision = WT_PRECISION.
	input [FANIN-1:0] inspk,		//spike input from pre-synaptic connections
	input [ADDR_WIDTH_NEURON-1:0] monitor_id,	//id of the neuron to be monitored
	output [FANOUT-1:0] outspk,		//spike output from lifs
	output [PRECISION-1:0] vmem		//membrane potential of a lif
);


	//instantiate the write address decoder
	wire [FANOUT-1:0] wr_addr_decode;	//decoded write address
	parameterized_decoder #(
		.N(FANOUT)
	) wr_addr_decoder(
		.en(wr_en),
		.in(wr_addr[(FANOUT_WIDTH+ADDR_WIDTH_FANIN-1):ADDR_WIDTH_FANIN]),
		.out(wr_addr_decode)
	);

	//instantiate the read address decoder
	wire spk_int,rst_acc,rd_en;
	wire [FANIN_WIDTH-1:0] rd_addr;
	syn_access #(
		.FANIN(FANIN)
	) rd_addr_decoder(
		.rst(rst),
		.memclk(memclk),
		.inspk(inspk),
		.outspk(spk_int),
		.rst_acc(rst_acc),
		.rd_en(rd_en),
		.rd_addr(rd_addr)
	);

	wire [PRECISION-1:0] vmem_int [FANOUT-1:0];	//internal signal to capture vmem output of all LIF neurons

	//convert the XWIDTH into one-hot encoding
	//wire [FANOUT-1:0] addrXOneHot;			//one-hot encoding wires
	//assign addrXOneHot = 1 << wr_addrX;			//convert to one-hot encoding
	
	//parametrically instantiate cuba_lif modules 
	genvar i;
	generate
	for (i=0;i<FANOUT;i=i+1) begin : neuron
		cuba_lif #(
			//configuration parameters
			.FANIN(FANIN),					//number of fanin of each neuron in the layer
			.INTEGER_PRECISION(INTEGER_PRECISION),		//integer precision
			.DECIMAL_PRECISION(DECIMAL_PRECISION)		//decimal precision
		) cuba_lif_inst(
			.rst(rst),					//reset
			.memclk(memclk),				//memory clock
			.spkclk(spkclk),				//spike clock
			//neuron parameters from configuration registers
			.vth(vth),					//neuron threshold voltage
			.decay_rate(decay_rate),			//membrane decay rate
			.grow_rate(grow_rate),				//membrane grow rate
			.vrest(vrest),					//neuron resting potential
			.reset_mechanism(reset_mechanism),		//neuron reset mechanism
			.refractory_period(refractory_period),		//neuron refractory period
			.wr_en(wr_addr_decode[i]),			//write enable for writing into the synaptic weights
			.wr_addr(wr_addr[FANIN_WIDTH-1:0]),		//memory address
			.wr_data(wr_data),				//write data
			.rd_en(rd_en),					//read enable
			.rd_addr(rd_addr),				//read address
			.rst_acc(rst_acc),				//reset accumulator
			.inspk(spk_int),				//input spike
			.outspk(outspk[i]),				//output spike
			.vmem(vmem_int[i])				//output membrane potential for monitoring
		);
	end
	endgenerate
	
	assign vmem = vmem_int[monitor_id]; 	//monitor a neuron of the layer
endmodule
