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
// Date     	: Nov 06, 2023
// File     	: tb_lif_parameters.vh
// Desc     	: This is the parameter file for the lif module that is to be generated using a Python script.
// -----------------------------------------------------------------------------*/
parameter PERIOD 		= 5,						//Half-Clock Period
parameter SIM_CNT 		= 100,						//Simulation Counter
parameter ACT 			= 128,						//Neuron Activation.  This is 8 in Q9.4 fixed-point representation
parameter INTEGER_PRECISION 	= 4,						//integer precision
parameter FRACTION_PRECISION 	= 4,						//fraction precision
parameter PRECISION 		= INTEGER_PRECISION + FRACTION_PRECISION,	//bit precision for state variables
parameter DECAY_RATE 		= 3,						//Membrane Decay Rate. This is 0.8333333333333334 in Q9.4 fixed-point representation
parameter GROW_RATE 		= 10,						//Membrane Grow Rate. This is 0.8333333333333334 in Q9.4 fixed-point representation
parameter VTH 			= 160,						//Membrane Threshold Voltage. This is 10.0 in Q9.4 fixed-point representation
parameter VREST 		= 160,						//Membrane Resting Potential.  This is 10 in Q9.4 fixed-point representation
parameter RESET_MECHANISM 	= 2,						//Membrane Reset Mechanism.  This is 2 in Q9.4 fixed-point representation
parameter REFRACTORY_PERIOD 	= 0,						//Membrane Refractory Period.  This is 0 in Q9.4 fixed-point representation
parameter DUMMY_PARAMETER	= 1
