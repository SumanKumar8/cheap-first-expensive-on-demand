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
// Date     	: Feb 05, 2023
// File     	: cuba_lif.v
// Desc     	: This is an implementation of a single lif neuron with its synaptic weights.
		  While programming the synaptic weights, we use one-hot encoding.
		  The module instantiates xmem, which is  a memory array of FANIN * PRECISION bits.
		  Essentially, these are synaptic weights of each pre-synaptic connections to the LIF neuron.
		  The top module also instantiates the lif module, which is the main code for a leaky integrate-and-fire neuron.
		  The weights from the memory (xmem) are input to the LIF.
// Changes	: 1.1. Added parameter RESET_MECHANISM.
// 		       This parameter controls how the membrane potential changes upon firing a spike.
//		  2.1. Changed the IO to consider the y-addresss. No need for one-hot encodding.
//		  2.2. xmem now calculates the activation (sum-of-synaptic-weights) and OR of all spikes.
//		       These are now forwarded to the LIF module.
//		       Changes in I/O due to these improvements.
//		  2.3. Port change in xmem (no wide buses across modules).
//		       Added port inspk. Basically, the input spikes from previous layers, all go to the xmem module.
//		       Removed port rd_data from xmem.
//		       Added port outspk (OR of all inspks and latched).
//		       Added port activation. This is the accumulation of all synaptic weights for the input muxed using the inspk signal.
//		       Bsically, accumulation = sum(x_i * w_i).
//		       Here, x_i = spike input from ith pre-synaptic neuron and w_i = weight of the connection from the pre-synaptic neuron.
//		  2.4. Port changes to LIF module.
//		       Added a single bit input inspk (output from xmem).
//		       Added activation and connect it to the output of xmem.  
//		  2.5. Removed FANIN parameter from LIF module.
//		       lif is an implementation of a single neuron, which receives a single spike and activation.
//		       The basic philosophy for this change is to make the lif module independent of its presynaptic connections.
//		       This makes the design modular and can be used for any other architecture.
//		       The fundamental units are the lif modules, which can then be wrapped inside the cuba_lif wrapper to implement any type of connections.
//		       For a different architecture (with say, irregular connections), the lif modules can be reused.
//		  2.6. Added the following new parameters.
//		       WT_PRECISION: This is the bit precision for synaptic weights.
//		       YWIDTH: This is the address width for programming synaptic cunnection of this neuron.
//		  3.1. Added memclk, a separate clock to read and process synaptic weights from the memory.
//		       If MEM_CLK primitive is defined in the header file, then the main clock clk is used for neurons and memclk for synaptic weights.
//		       Otherwise, main clk clk is used for both neurons and synaptic weeights.
//		  4.1. Added neuron parameters as input.
//		       These parameters are obtained from configuration registers.
//		  4.2. Removed the following parameters from the code.
//		       VTH, DECAY_RATE, GROW_RATE, VREST, RESET_MECHANISM, REFRACTORY_PERIOD from lif.
//		  5.1. Cleaned the code to remane the ports and parameters.
//		       FRACTION_PRECISION ==> DECIMAL_PRECISION
//		       YWIDTH ==> ADDR_WIDTH (also converted it as a local parameters).
//		       clk ==> spkclk
// -----------------------------------------------------------------------------*/
`timescale 1ns / 1ps
module cuba_lif #(
	//configurable parameters
	parameter FANIN 		= 256,			//fanin
	parameter INTEGER_PRECISION	= 3,			//integer precision
	parameter DECIMAL_PRECISION 	= 4,			//fraction precision
	//local parameters
	localparam WT_PRECISION 	= (1+DECIMAL_PRECISION),			//precision for synaptic weights
	localparam PRECISION 		= (1+INTEGER_PRECISION+DECIMAL_PRECISION),	//precision for state variables
	localparam ADDR_WIDTH		= $clog2(FANIN)		//address width for the memory addresses of fanin of each neuron. 
)(
	input rst,				//common reset
	input memclk,				//memory clock
	input spkclk,				//spike clock
	//neuron parameters from congiguration registers
	input [PRECISION-1:0] vth,		//neuron threshold voltage
	input [PRECISION-1:0] decay_rate,	//membrane decay rate
	input [PRECISION-1:0] grow_rate,	//membrane grow rate
	input [PRECISION-1:0] vrest,		//neuron resting potential
	input [PRECISION-1:0] reset_mechanism,	//neuron reset mechanism
	input [PRECISION-1:0] refractory_period,//neuron refractory period
	//memory write
	input wr_en,				//write enable to synaptic memory
	input [ADDR_WIDTH-1:0] wr_addr,		//write address to synaptic memory
	input [WT_PRECISION-1:0] wr_data,	//write data (weights) to synaptic memory
	//memory read
	input rd_en,				//read enable for synaptic memory
	input [ADDR_WIDTH-1:0] rd_addr,		//read address for synaptic memory
	//accumulator clear
	input rst_acc,				//clear signal for the accumulator
	//input spike
	input inspk,				//spike input from pre-synaptic connections
	//output
	output outspk,				//spike output from lif
	output [PRECISION-1:0] vmem		//membrane potential of the lif 		
);
	//signals
	wire int_spk;
	wire [PRECISION-1:0] int_activation;
	//instantiate the bmem
	bmem #(
		.FANIN(FANIN),
		.INTEGER_PRECISION(INTEGER_PRECISION),
		.DECIMAL_PRECISION(DECIMAL_PRECISION)
	) bmem_dut(
		.rst(rst),
		.memclk(memclk),
		.spkclk(spkclk),
		.wr_en(wr_en),
		.wr_addr(wr_addr),
		.wr_data(wr_data),
		.rd_en(rd_en),
		.rd_addr(rd_addr),
		.rst_acc(rst_acc),
		.inspk(inspk),
		.outspk(int_spk),
		.activation(int_activation)
	);
	//instantiate the neuron
	lif #(
		.INTEGER_PRECISION(INTEGER_PRECISION),
		.DECIMAL_PRECISION(DECIMAL_PRECISION)
	) lif_dut(
		.rst(rst),
		.clk(spkclk),
		.vth(vth),
		.decay_rate(decay_rate),
		.grow_rate(grow_rate),
		.vrest(vrest),
		.reset_mechanism(reset_mechanism),
		.refractory_period(refractory_period),
		.inspk(int_spk),
		.activation(int_activation),
		.outspk(outspk),
		.vmem(vmem)
	);
endmodule
