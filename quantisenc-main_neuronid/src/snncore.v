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
// Date     	: Feb 10, 2024
// File     	: snncore.v
// Desc     	: This is an implementation of an snn core. 
			  It is a MLP of programmable number of LAYERS, with programmable number of neurons per layer. 
			  All operations are integer-based. 
			  Neuron parameters are programmable and can be set in this file.
			  In addition to the neuron functionalities, the core support the following monitoring capabilities.
			  (1) Monitoring two membrane potential of any neuron.
			  (2) All weights that are programmed in the memory.
			  (3) Any other signals.
			  These monitoring capabilities are to be hand programmed for debug.
// Changes	: 1.1. All parameters of the design is moved to a separate parammeters.vh file and is included in this file at line 44.
// 		  1.2. Added capabilities to monitor two membrane potentials from two different layers. 
// 		       Currently the design allows monitoring neuron 1 of Layer 1 and neuron 1 of Layer N.
// 		  2.1. Distributed implementation of synaptic memory
// 		       A layer now instantiates cuba_lif modules, which is a LIF module with all of its pre-synaptic memory.
// 		  2.2. Added a GROW_RATE that allows the membrane potential to increase exponentially with activation (biology-inspired).
// 		  2.3. Removed support to monitor synaptic weights.
// 		  3.1. Added parameter RESET_MECHANISM.
// 		       This parameter controls how the membrane potential changes upon firing a spike.
// 		  4.1. Added a new parameter FANOUT to control the number of neurons of the output layer.
// 		       The snncore now implements one input layer (spike generator layer), multiple hidden layers, and one output layer.
// 		       The spike generator layer is implemented in the software (tb).
// 		       The spike output of this layer is input to the first hidden layer, which is implemented in the snncore.
// 		  5.1. Changed the IO to accomodate variable widths for layers, neurons per layer, and synaptic weights per neuron.
// 		       ADDR_WIDTH_L: Address bus width for selecting a layer.
// 		       ADDR_WIDTH_X: Address bus width for selecting a neuron within a layer.
// 		       ADDR_WIDTH_Y: Address bus width for selecting a pre-synaptic neuron of a neuron within a layer.
// 		  5.2. No one-hot encoding for addresses X and Y.
// 		  5.3. A new parameter called WT_PRECISION is introduced.
// 		       This parameter is used to specify the precision for the synaptic weights.
// 		       In this new version of the design we use the following precisions.
// 		       WT_PRECISION: for (signed) synaptic weights.
// 		       PRECISION: for (signed) state variables, including membrane potential.
// 		  5.4. The design now supports signed number system (with quantization).
// 		  6.1. Added memclk, a separate clock for fast reading of synaptic weights from the memory.
// 		       This clock is controlled using the MEM_CLK premitive which is defined in defines.vh file.
// 		  7.1. Removed vmem ports and added output ports gpout.
// 		       Membrane potential is being monnitored on gpout ports.
// 		       It can be used to monitor several other internal signals.
// 		  8.1. Change in IO ports for programming synaptic weights and configuration registers.
// 		       wr_en renamed as mem_write.
// 		       wr_addr{L,X,Y} are removed and replaced with wr_addr;
// 		  9.1. Use register configuration for neuron parameters VTH, DECAY_RATE, and GROWTH_RATE.
// 		  9.2. Remove the use of the following parameters in the code.
// 		       VTH, DECAY_RATE, GROW_RATE, VREST, RESET_MECHANISM, REFRACTORY_PERIOD from layer.
// 		  9.3. Added neuron_mon to the layer module to monitor a neuron's membrane potential.
// 		  10.1. Incorporate changes in layer architecture.
// 		        Renamed several ports and cleaned implementation.
// -----------------------------------------------------------------------------*/
`timescale 1ns / 1ps

module snncore #(
	`include "parameters.vh"
	)(
	//IOs for loading layer-by-layer synaptic weights into the memory and programming the configuration registers.
	//mem_write = 0, reg_write = 1 ==> write to configuration registers
	//mem_write = 1, reg_write = 0 ==> write to synaptic memory
	input mem_write,				//write enable for synaptic memory
	input cfg_write,				//write enable for configuration registers
	input neuronid_faulttype,
	input [ADDR_WIDTH-1:0] wr_addr,			//memory/configuration address
	input [DATA_WIDTH-1:0] wr_data,			//memory/configuration data 
	//IOs for data processing
	input rst,					//reset signal 
	input memclk,					//memory clock
	input spkclk,					//spike clock
	input [INPUT_NEURONS-1:0] spk_in,		//spike input to the input layer
	output [OUTPUT_NEURONS-1:0] spk_out,		//spike output from the output layer
	//GPIOs for monitoring internal signals
	output [GPOUT_WIDTH-1:0] gpout			//general purpose output to monitor internal signals
	);

	//instantiate the address decoder
	wire [HARDWARE_LAYERS-1:0] wr_addr_en;
	parameterized_decoder #(
		.N(HARDWARE_LAYERS)
	) wr_addr_decoder(
		.en(mem_write),
		.in(wr_addr[LAYER_WIDTH+LAYER_ADDR_START-1:LAYER_ADDR_START]),
		.out(wr_addr_en)
	);

	//instantiate the cfg decoder
	wire [PRECISION-1:0] vth;
	wire [PRECISION-1:0] decay_rate;
	wire [PRECISION-1:0] grow_rate;
	wire [PRECISION-1:0] vrest;
	wire [PRECISION-1:0] reset_mechanism;
	wire [PRECISION-1:0] refractory_period;
	wire [DATA_WIDTH-1:0] layer_to_monitor;
	wire [DATA_WIDTH-1:0] neuron_to_monitor;
	decoder #(
		.ADDR_WIDTH(ADDR_WIDTH),
		.DATA_WIDTH(DATA_WIDTH),
		.CFG_REG(CONFIG_REG),
		.PRECISION(PRECISION)
	) cfg_decoder(
		.rst(rst),
		.clk(memclk),
		.wr_en(cfg_write),
		.wr_addr(wr_addr),
		.wr_data(wr_data),
		.vth(vth),
		.decay_rate(decay_rate),
		.grow_rate(grow_rate),
		.vrest(vrest),
		.reset_mechanism(reset_mechanism),
		.refractory_period(refractory_period),
		.layer_to_monitor(layer_to_monitor),
		.faulty(faulty),
		.neuron_to_monitor(neuron_to_monitor)
	);

	//instantiate the first layer
	wire [HIDDEN_LAYER_NEURONS-1:0] spikes_int [0:HARDWARE_LAYERS-1];
	wire [PRECISION-1:0] vmem_int[0:HARDWARE_LAYERS-1];
	layer_f #(
		.FANIN(INPUT_NEURONS),
		.FANOUT(HIDDEN_LAYER_NEURONS),
		.INTEGER_PRECISION(INTEGER_PRECISION),
		.DECIMAL_PRECISION(DECIMAL_PRECISION),
		.ADDR_WIDTH_NEURON(NEURON_ENC_BITS),
		.ADDR_WIDTH_FANIN(FANIN_ENC_BITS)
	) layer_0(
		.rst(rst),
		.memclk(memclk),
		.spkclk(spkclk),
		.neuronid_faulttype(neuronid_faulttype),
		.vth(vth),
		.decay_rate(decay_rate),
		.grow_rate(grow_rate),
		.vrest(vrest),
		.reset_mechanism(reset_mechanism),
		.refractory_period(refractory_period),
		.wr_en(wr_addr_en[0]),
		.wr_addr(wr_addr),
		.wr_data(wr_data),
		.inspk(spk_in),
		.monitor_id(neuron_to_monitor),
		.outspk(spikes_int[0]),
		.vmem(vmem_int[0])
	);

	//instantiate the next few layers
	genvar i;
	generate
	for (i=1;i<HARDWARE_LAYERS-1;i=i+1) begin	: layer
		layer_f #(
			.FANIN(HIDDEN_LAYER_NEURONS),
			.FANOUT(HIDDEN_LAYER_NEURONS),
			.INTEGER_PRECISION(INTEGER_PRECISION),
			.DECIMAL_PRECISION(DECIMAL_PRECISION),
			.ADDR_WIDTH_NEURON(NEURON_ENC_BITS),
			.ADDR_WIDTH_FANIN(FANIN_ENC_BITS)
		) layer(
			.rst(rst),
			.memclk(memclk),
			.spkclk(spkclk),
			.neuronid_faulttype(neuronid_faulttype),
			.vth(vth),
			.decay_rate(decay_rate),
			.grow_rate(grow_rate),
			.vrest(vrest),
			.reset_mechanism(reset_mechanism),
			.refractory_period(refractory_period),
			.wr_en(wr_addr_en[i]),
			.wr_addr(wr_addr),
			.wr_data(wr_data),
			.inspk(spikes_int[i-1]),
			.monitor_id(neuron_to_monitor),
			.outspk(spikes_int[i]),
			.vmem(vmem_int[i])
		);
	end
	endgenerate
	
//	//instantiate the last
	layer_o #(
		.FANIN(HIDDEN_LAYER_NEURONS),
		.FANOUT(OUTPUT_NEURONS),
		.INTEGER_PRECISION(INTEGER_PRECISION),
		.DECIMAL_PRECISION(DECIMAL_PRECISION),
		.ADDR_WIDTH_NEURON(NEURON_ENC_BITS),
		.ADDR_WIDTH_FANIN(FANIN_ENC_BITS)
	) layer_out(
		.rst(rst),
		.memclk(memclk),
		.spkclk(spkclk),
		.neuronid_faulttype(neuronid_faulttype),
		.vth(vth),
		.decay_rate(decay_rate),
		.grow_rate(grow_rate),
		.vrest(vrest),
		.reset_mechanism(reset_mechanism),
		.refractory_period(refractory_period),
		.wr_en(wr_addr_en[HARDWARE_LAYERS-1]),
		.wr_addr(wr_addr),
		.wr_data(wr_data),
		.inspk(spikes_int[HARDWARE_LAYERS-2]),
		.monitor_id(neuron_to_monitor),
		.outspk(spk_out),
		.vmem(vmem_int[HARDWARE_LAYERS-1])
	);

	assign gpout = vmem_int[layer_to_monitor];
	
endmodule
