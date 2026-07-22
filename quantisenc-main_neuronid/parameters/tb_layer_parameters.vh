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
// Date     	: Feb 09, 2024
// File     	: tb_layer_parameters.vh
// Desc     	: This is the parameter file for the layer module that is to be generated using a Python script.
// 		  This file contains the install directory.
// 		  A new directive swctrl is defined. 
// 		  This directive is used by a python interpreter to change the parameters.
// -----------------------------------------------------------------------------*/
parameter string INSTALL_DIR 	= "C:/LIF/5.1_2/SNN5.1_5/quantisenc-main_neuronid",//base installation directory
//neuron configuration
parameter VTH = 17,	//swctrl
parameter DECAY_RATE = 3,	//swctrl
parameter GROW_RATE = 16,	//swctrl
parameter VREST = 0,	//swctrl
parameter RESET_MECHANISM = 2,	//swctrl
parameter REFRACTORY_PERIOD = 0,	//swctrl
parameter NEURON_TO_MONITOR = 0,	//swctrl
//design parameters
parameter CLK_BFR		= 7,
parameter FANIN = 256,	//swctrl
parameter FANOUT = 16,	//swctrl
parameter INTEGER_PRECISION = 3,	//swctrl
parameter DECIMAL_PRECISION = 4,	//swctrl
parameter PRECISION 		= (1+INTEGER_PRECISION+DECIMAL_PRECISION), //total precision
parameter ADDR_WIDTH_NEURON = 12,	//swctrl
parameter ADDR_WIDTH_FANIN = 12,	//swctrl
//testbench parameters
parameter WTS_CNT = 4096,	//swctrl
parameter SIM_CNT = 2100,	//swctrl
parameter MEM_CLK_PERIOD	= 1,					//memory clock period
parameter SPK_CLK_PERIOD	= (MEM_CLK_PERIOD*FANIN)+CLK_BFR,	//spkclk is atleast n-times slower than memclk. We need to add a few clock cycles of buffer.
parameter PRG_CLK_PERIOD	= (2*MEM_CLK_PERIOD),			//clock to program synaptic memory
parameter DELAY			= 100,					//testbench delay
parameter EXTRA_CYCLES		= 20,					//20 extra cycles for the output 
//dummy parameter to mark the end of file
parameter DUMMY_PARAMETER	= 1					//dummy parameter, indicating the end of all parameters
