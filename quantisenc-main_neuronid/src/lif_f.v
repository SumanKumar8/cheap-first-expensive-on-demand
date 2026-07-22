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
// File     	: lif.v
// Desc     	: This is an implementation of a leaky integrate-and-fire (LIF) neuron with current-based (CUBA) synapses.
//            	  All operations are fixed-point arithmatic.
// Changes	: 1.1. All operations are fixed point arithmetic, i.e., implemented Qn.m addition/multiplication.
// 		       Here, n = PRECISION (=8) and m = FRACTION_PRECISION (=4).
// 		       These parameters can be configured via the top-level parameters.vh file. 
// 		  1.2. Exponential increase of membrane potential with input activation.
// 		  2.1. Added support for different reset mechanisms following a spike.
// 		       0 = no reset of vmem.
// 		       1 = reset-by-subtraction (basically, subtracting Vth from vmem after a spike)
// 		       2 = reset-to-zero (basically, vmem is reset to 0 after a spike) 
// 		  3.1. Redesign of LIF module to account for vmem overflow.
// 		       The vmem output is PRECISION bit wide.
// 		       This includes 1-bit sign and (PRECISION-1) bits magnitude.
// 		       State variables use 2's complement representation.
// 		  3.2. The LIF module gets a single bit input spike (which is ORed version of all input spikes).
// 		  3.3. The activation (which is the sum of weights) is now calculated in the xmem module.
// 		       The LIF module now receives a single activation signal from the xmem.
// 		       The LIF module can now be reused to implement any neural architecture.
// 		  3.4. All the wide buses are eliminated from the module.
// 		  3.5. Removed FANIN parameter from the module.
// 		       The LIF module now receives a single spike input.
// 		  3.6. Rename vmem as vmem_out.
// 		  3.7. Design of hierarchies for quantized addition and multiplication.
// 		       All additions are now replaced as quantized and signed addition.
// 		       Adders and multipliers are designed to saturate at the max or min value upon overflow or underflow, respectively.
// 		       Comparators are replaced with a quantized alu, which internally uses subtraction to identify if vmem > vth.
// 		       A new module 2's complement is defined and used in the design.
// 		  4.1. Spike and activation input signals are synchronized to the spike clk before using.
// 		       This is only when these signals are generated from a seperate clock domain (memclk).
// 		       The primitive MEM_CLK controls this double synchronization behavior of clock domain crossing.
// 		  5.1. Added neuron parameters as input.
// 		       These parameters are obtained from configuration registers.
// 		  5.2. Removed the following parameters from code.
// 		       VTH, DECAY_RATE, GROW_RATE, VREST, RESET_MECHANISM, REFRACTORY_PERIOD
// 		  6.1. Bug fixes for reset_mechanism = 0.
// 		  7.1. Renamed FRACTION_PRECISION ==> DECIMAL_PRECISION.
// 		  7.2. Renamed output port vmem_out ==> vmem.
// 		  7.3. Removed the double flip-flop synchronization. This is now performed inside the bmem module.
// -----------------------------------------------------------------------------*/
`timescale 1ns / 1ps

module lif_f #(
	//configurable parameters
	parameter INTEGER_PRECISION = 3,	//integer precision
	parameter DECIMAL_PRECISION = 4,	//decimal precision
	//local parameters
	localparam PRECISION = (1+INTEGER_PRECISION+DECIMAL_PRECISION)	//bit precision
	)(
	//IOs for data processing
	input rst,				//reset
	input clk,				//clock
	//neuron parameters from configuration registers
	input [PRECISION-1:0] vth,		//neuron threshold voltage
	input [PRECISION-1:0] decay_rate,	//membrane decay rate
	input [PRECISION-1:0] grow_rate,	//membrane grow rate
	input [PRECISION-1:0] vrest,		//neuron resting potential
	input [PRECISION-1:0] reset_mechanism,	//neuron reset mechanism
	input [PRECISION-1:0] refractory_period,//neuron refractory period
	input inspk,				//input spike
	input [PRECISION-1:0] activation,	//activation
	output reg outspk,			//output spike
	//IOs for monitoring
	output [PRECISION-1:0] vmem		//output membrane voltage
);

	reg [PRECISION-1:0] refr_cnt;		//refractory Counter
	reg [PRECISION-1:0] int_vmem;		//vmem internal signal
	
	wire [PRECISION-1:0] grow;		//growth of activation
	wire [PRECISION-1:0] decay;		//decay of vmem		

	//wires
	localparam MSB = PRECISION-1;
	wire [PRECISION-1:0] decay_n;		//-decay
	wire [PRECISION-1:0] grow_decay_n;	//grow-decay
	wire [PRECISION-1:0] vmem_decay_n;	//vmem - decay
	wire [PRECISION-1:0] vmem_grow_decay_n;	//vmem - decay + grow
	wire [PRECISION-1:0] vrest_grow_decay_n;//vrest - decay + grow
	wire [PRECISION-1:0] vmem_grow_decay_n_vth_n;//vmem - decay + grow -vth
	wire [PRECISION-1:0] vth_n;
	wire [PRECISION-1:0] vmem_sub;
	wire vmem_greater_vth;
	reg [1:0] fd_vmem=2'b00;
	reg [2:0] bit_selection_vmem=3'b111;
	reg [PRECISION-1:0] vmem_t;
	genvar i;
	wire [PRECISION-1:0] mask;

	//vmem_grow_decay_n = vmem - decay + grow. A three step design.
	//Step 1: 2's complement of decay to generate decay_n.
	twos_complement #(
		.N(PRECISION)
	) decay_n_inst(
		.in(decay),
		.out(decay_n)
	);
	//Step 2: Add vmem to decay_n. So, vmem_decay_n = vmem - decay.
	qadd #(
		.N(PRECISION)
	) decay_vmem_inst(
		.a(int_vmem),
		.b(decay_n),
		.q_result(vmem_decay_n)
	);
	//Step 3: Add grow to vmem_decay_n. So, vmem_grow_decay_n = vmem - decay + grow.
	qadd #(
		.N(PRECISION)
	) decay_grow_vmem_inst(
		.a(vmem_decay_n),
		.b(grow),
		.q_result(vmem_grow_decay_n)
	);
	//Step 3: calculate grow_decay_n = grow - decay
	qadd #(
		.N(PRECISION)
	) grow_decay_n_inst(
		.a(grow),
		.b(decay_n),
		.q_result(grow_decay_n)
	);
	//Step 4: calculate vrest_grow_decay_n = vrest + grow - decay	
	qadd #(
		.N(PRECISION)
	) vrest_grow_decay_n_inst(
		.a(vrest),
		.b(grow_decay_n),
		.q_result(vrest_grow_decay_n)
	);
	//Step 6: calculate vmem_grow_decay_n_vth_n = vmem + grow - decay - vth	
	qadd #(
		.N(PRECISION)
	) vmem_grow_decay_n_vth_n_inst(
		.a(vmem_grow_decay_n),
		.b(vth_n),
		.q_result(vmem_grow_decay_n_vth_n)
	);


	qmult_f #(
		.Q(DECIMAL_PRECISION),
		.N(PRECISION)
	) decay_inst(
		.a(int_vmem),
		.b(decay_rate),
		.q_result(decay)
	);
	
	qmult_f #(
		.Q(DECIMAL_PRECISION),
		.N(PRECISION)
	) grow_inst(
		.a(activation),
		.b(grow_rate),
		.q_result(grow)
	);

	//wire [PRECISION-1:0] vth;
	//threshold comparison.
	//This is a two step process.
	//Step 1: Calculate negative of VTH. So, VTH_n = 0 - VTH (2's complement).
	//assign vth = VTH;
	twos_complement #(
		.N(PRECISION)
	) vth_n_inst(
		.in(vth),
		.out(vth_n)
	);
	//Step 2: Calculate vmem_sub = vmem - vth. Additionally, we need a bit to check if vmem > vth.
	//So, we use the qalu instead of qadd.
	qalu #(
		.N(PRECISION)
	) vmem_sub_inst(
		.a(vmem_t),
		.b(vth_n),
		.en(1'b1),
		.sel(1'b0),
		.q(vmem_sub),
		.pos(vmem_greater_vth)
	);

	//update of state variables	
	generate
	for(i=0; i<PRECISION; i=i+1) begin
	assign mask[i] = (i == bit_selection_vmem) ? 1'b1 : 1'b0;    //bit flip based on bit selection
	end
	endgenerate
	always @(posedge clk or posedge rst) begin
		if (rst) begin
			int_vmem     <= 0;
			refr_cnt <= 0;
			outspk   <= 0;
		end
		else begin
			if (refr_cnt > 0) begin
				//LIF neuron is in refractory period
				outspk		<= 0;				//no output spike
				refr_cnt 	<= refr_cnt - 1;		//decrement the refractory counter 
				//membrane dynamics in refractory period
				case (reset_mechanism)
					0:	int_vmem <= vmem_decay_n;		//decay vmem exponentially
					default:int_vmem <= int_vmem;			//no change in vmem
				endcase
			end
			else begin
				//LIF neuron is not in refractory period
				//membrane voltage dynamics
				//increase vmem if there is spike
				if (inspk) begin
					//spike output dynamics
					if (vmem_greater_vth) begin
						outspk   <= 1;
						refr_cnt <= refractory_period;
						//implement different reset mechanisms
						case (reset_mechanism)
							0:	int_vmem <= vmem_grow_decay_n;
							1: 	int_vmem <= vmem_grow_decay_n_vth_n;
							2: 	int_vmem <= 0;
							//2: 	int_vmem <= grow_decay_n;
							default:int_vmem <= vrest_grow_decay_n;
						endcase
					end
					else begin
						outspk <= 0;
						int_vmem <= vmem_grow_decay_n;
					end
				end
				//leakage if there is no spike
				else begin
					if (vmem_greater_vth) begin
						outspk   <= 1;
						refr_cnt <= refractory_period;
						//implement different reset mechanisms
						case (reset_mechanism)
							0:	int_vmem <= vmem_decay_n;
							1: 	int_vmem <= vmem_sub;
							2: 	int_vmem <= 0;
							default:int_vmem <= vmem_decay_n;
						endcase
					end
					else begin
						outspk <= 0;
						int_vmem <= vmem_decay_n;			//exponential decay of vmem
					end
				end
			end
			case(fd_vmem)
                2'b00 : vmem_t = int_vmem ^ 8'b0;// fault free case
                2'b01 : vmem_t = int_vmem ^ mask; //bit flip of vmem to impact output spikes
                2'b10 : vmem_t = int_vmem & (~mask); //and gate used for stuck at 0 
                2'b11 : vmem_t = int_vmem | mask; //or gate used for stuck at 1 
            endcase
		end
	end

	//output assignment
	assign vmem = vmem_t;
	//assign vmem_out = activation_q2;

endmodule
