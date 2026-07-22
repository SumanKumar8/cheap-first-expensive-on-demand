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
// File     	: parameters.vh
// Desc     	: This is the top-level parameter file that is to be generated using a Python script.
// Changes	: 1.1. Added a new parameter called GROW_RATE.
// 		  2.1. Added a new parameter called RESET_MECHANISM.
// 		       This parameter controls how the membrane potential changes upon firing a spike.
// 		  3.1. Added a new parameter called FANOUT to control the number of neurons of the output layer.
// 		       The snncore now implements the first input hidden layer, multiple other hidden layers, and one output layer.
// 		       The first hidden layer receives input from spike generator layer.
// 		       So, this layers requires different configuration.
// 		       The spike generator layer is implemented in the software (tb).
// 		       The spike output of this layer is input to the next set of hidden layers, which are all implemented in the snncore.
// 		       Except the fist hidden layer, all other hidden layers are symmetrical.
// 		       The output layer on the other hand uses a different configuration because the number of output needed can be different.
// 		  4.1. Following new parameters were added.
// 		       (a) Address widths for programming different layers, neurons within the layers, and synaptic weights of each neuron.
// 		       (b) WT_PRECISION is a new parameter that controls the bit precision of the synaptic weights.
// 		  4.2. The parameter ADDR_WIDTH is removed.
// 		  5.1. Addere parameter GPIO_WIDTH to specify the width of GPIO pins.
// 		  5.2. Added parameter REG_WIDTH to specify the width of the configuration register.
// 		  5.3. Added parameter RESET_WIDTH to specify the type of neuron reset mechanism post firing.
// 		  6.1. Added general configuration parameters.
// 		       ADDR_WIDTH = address bits for synaptic memory. Total number of synaptic weights = 2^(ADDR_WIDTH).
// 		       DATA_WIDTH = number of bits needed to represent each synaptic weight.
// 		  7.1. Correct set of parameters for tcas-I release.
// 		  8.1. Removed extra parameters.
// 		       Renamed parameters.
// -----------------------------------------------------------------------------*/
//hardware parameters
parameter INTEGER_PRECISION = 3,	//swctrl
parameter DECIMAL_PRECISION = 4,	//swctrl
parameter DATA_WIDTH = 32, 				//swctrl
parameter LAYER_ENC_BITS = 8,	//swctrl
parameter NEURON_ENC_BITS = 12,	//swctrl
parameter FANIN_ENC_BITS = 12,	//swctrl
localparam ADDR_WIDTH = (LAYER_ENC_BITS+NEURON_ENC_BITS+FANIN_ENC_BITS),
localparam LAYER_ADDR_START = (NEURON_ENC_BITS+FANIN_ENC_BITS),
localparam PRECISION = (1+INTEGER_PRECISION+DECIMAL_PRECISION),
//model parameters
//this configuration implements
//software: <input-layer><hidden-layer-1><hidden-layer-2><hidden-layer-3>...<output-layer>
//hardware: <hidden-layer-1><hidden-layer-2><hidden-layer-3>...<output-layer>
//important note: the input layer neurons are implemented in software/testbench
parameter INPUT_NEURONS = 400,	//swctrl
parameter HIDDEN_LAYERS = 2,	//swctrl
parameter HIDDEN_LAYER_NEURONS = 300,	//swctrl
parameter OUTPUT_NEURONS = 11,	//swctrl
localparam HARDWARE_LAYERS = (HIDDEN_LAYERS + 1),
localparam LAYER_WIDTH = $clog2(1+HARDWARE_LAYERS),
//gpout/configuration parameters
parameter CONFIG_REG = 8,
parameter GPOUT_WIDTH = 32,
//dummy parameter
localparam DUMMY_HW_PARAMETER = 0
