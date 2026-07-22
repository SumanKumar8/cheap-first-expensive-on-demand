//10 image_30 june

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
// Date     	: Feb 10, 2023
// File     	: tb_snncore.sv
// Desc     	: This is an implementation of the testbench for snncore with all layers.
//                Each layer has neurons and their synaptic memory.
// -----------------------------------------------------------------------------*/

`timescale 1ns / 1ps


module tb_snncore #(
	`include "parameters.vh",
	`include "tb_snncore_parameters.vh"
) ();
	localparam WTS_CFG_CNT = WTS_CNT + CONFIG_REG;
	localparam File_naming = 100;	//swctrl
	//dut IOs	
	reg spkclk;			//dut clock for spike input
	reg memclk;			//dut clock for memory accesses
	reg rst;			//dut reset
	reg wr_en;			//dut write enable
	wire mem_write;			//memory write
	wire cfg_write;			//configuration write
	int neuronid_faulttype; // inject fault qmul s@1 at 8th bit for neuron ID
	int wr_addr;			//dut write address
	int wr_data;			//dut write data
	reg [INPUT_NEURONS-1:0] inspk;	//spike input to dut
	localparam Number_input_files = 1;
	localparam Number_output_files = 1;
	//file name
	string wtFname 	= {INSTALL_DIR,"/pysenc/weight/FMNIST/snncore.synaptic_weight.txt"};
	string addrFname= {INSTALL_DIR,"/pysenc/weight/FMNIST/snncore.synaptic_address.txt"};
//	string ispkFname= {INSTALL_DIR,"C:/LIF/5.1_2/SNN5.1_3/quantisenc-main_neuronid/pysenc/input/SpikeGenData/snncore.spikes_input.txt"};
//	string vmemFname= {INSTALL_DIR,"C:/LIF/5.1_2/SNN5.1_3/quantisenc-main_neuronid/pysenc/output/SpikeGenData/vmem/snncore.vmem_output.txt"};
//	string ospkFname= {INSTALL_DIR,"C:/LIF/5.1_2/SNN5.1_3/quantisenc-main_neuronid/pysenc/output/SpikeGenData/spike/snncore.spikes_output.txt"};
	string vmemFname;
	string ospkFname;
	string ispkFname;
	//file io
	int wrcnt;				//weight write counter
	int file;
	//other variables
	reg prgclk;				//programming clock
	int i;					//integer variable for loop iterations

	int mem_data [WTS_CFG_CNT-1:0];				//weight mem
	int mem_addr [WTS_CFG_CNT-1:0];				//weight addr
	reg [INPUT_NEURONS-1:0] inspk_bfr [SIM_CNT-1:0][Number_input_files-1:0];	//input spk buffer
	reg [PRECISION-1:0] vmem_bfr [SIM_CNT+EXTRA_CYCLES-1:0][Number_output_files-1:0];//output vmem buffer
	reg [OUTPUT_NEURONS-1:0] ospk_bfr [SIM_CNT+EXTRA_CYCLES-1:0][Number_output_files-1:0];	//output spike buffer
	reg [12:0] sim_cycles_i;
	reg [12:0] sim_cycles_o;	
	reg inp_en;	//input enable
	reg out_en;	//output enable

	int inp_cnt;	//input counter
	int out_cnt;	//output counter
	int inp_file;   //number of input files
	int out_file;   //number of output files

	//define all clocks
	always #SPK_CLK_PERIOD 	spkclk 	= ~spkclk;
	always #MEM_CLK_PERIOD 	memclk 	= ~memclk;
	always #PRG_CLK_PERIOD 	prgclk 	= ~prgclk;

	//instantiate the DUTs here.
	wire [OUTPUT_NEURONS-1:0] outspk;
	wire [PRECISION-1:0] vmem;
	wire [GPOUT_WIDTH-1:0] gpout;

	//instantiate the DUT
	snncore snncore_dut(
		.mem_write(mem_write),
		.cfg_write(cfg_write),
		.neuronid_faulttype(neuronid_faulttype),
		.wr_addr(wr_addr),
		.wr_data(wr_data),
		.rst(rst),
		.memclk(memclk),
		.spkclk(spkclk),
		.spk_in(inspk),
		.spk_out(outspk),
		.gpout(gpout)
	);
	assign vmem = gpout[PRECISION-1:0];


	//generate and control wr_en, wr_addr, wr_data
	always @(posedge prgclk or posedge rst) begin
		if (rst) begin
			wrcnt = 0;
		end
		else begin
			if (wr_en) begin
				if (wrcnt == WTS_CFG_CNT-1) begin
					wr_en 	= 0;
					wrcnt = 0;
				end
				else begin
					wrcnt = wrcnt + 1;
				end

			end
		end
	end
	assign wr_addr = mem_addr[wrcnt];
	assign wr_data = mem_data[wrcnt];
	assign mem_write = (wrcnt < WTS_CNT)? wr_en : 0;
	assign cfg_write = (wrcnt >= WTS_CNT)? wr_en : 0;
	//done
	
	//generate and control spike input to the dut
	always @(posedge spkclk or posedge rst) begin
		if (rst) begin
			inp_cnt <= 0;
		end
		else begin
			if (inp_en) begin
				if ((inp_cnt == SIM_CNT-1) && (inp_file == Number_input_files-1)) begin
					inp_en 	<= 0;
					inp_cnt <= 0;
					inp_file <= 0;
				end
				else if ((inp_cnt == SIM_CNT-1) && (inp_file != Number_input_files-1) ) begin
					inp_file <= inp_file + 1;
					inp_cnt <= 0;
				end
                else begin
					inp_cnt <= inp_cnt + 1;
				end

				inspk <= inspk_bfr[inp_cnt][inp_file];
			end
			else begin
				inspk <= 0;
			end
		end
	end
	//assign inspk = inp_en ? inspk_bfr[inp_cnt] : 0;
        
	//delay the inp_en to capture output
	always @(posedge spkclk or posedge rst) begin
		if (rst) begin
			out_cnt = 0;
		end
		else begin
			if (out_en) begin
				if ((out_cnt == SIM_CNT+EXTRA_CYCLES-1) && (out_file == Number_output_files-1)) begin
					out_cnt <= 0;
					out_en	<= 0;
					out_file <= 0;
				end
				else if ((out_cnt == SIM_CNT+EXTRA_CYCLES-1) && (out_file != Number_output_files-1)) begin
					out_cnt <= 0;
					out_file <= out_file + 1;
				end
                else begin
					out_cnt <= out_cnt + 1;
				end
				vmem_bfr[out_cnt][out_file] <= vmem;
				ospk_bfr[out_cnt][out_file] <= outspk;
			end
		end
	end


	initial
	begin
		//design input
		spkclk 	= 0;
		memclk 	= 0;
		rst 	= 0;
		wr_en 	= 0;
		wr_data = 0;
		//testbench signals
		inp_en  = 0;
		out_en  = 0;
		inp_cnt = 0;
		out_cnt = 0;
		prgclk 	= 0;
		wrcnt	= 0;
        neuronid_faulttype = 6;

		//simulation start
		//control reset
		#DELAY;
		rst 	= 1;
		#DELAY;
		rst 	= 0;

		#DELAY;
		//read the input weights
		file=$fopen(wtFname,"r");	// wtFname
		if (file)
			$display("%s was opened successfully", wtFname);
		else
			$display("%s NOT opened", wtFname);
		for (i=0; i<WTS_CNT; i=i+1) begin
			$fscanf(file,"%h",mem_data[i]);
			//wts.push_back(wt);
		end
		$fclose(file);

		//read the input weight addresses
		file=$fopen(addrFname,"r");	// addrFname
		if (file)
			$display("%s was opened successfully", addrFname);
		else
			$display("%s NOT opened", addrFname);
		for (i=0; i<WTS_CNT; i=i+1) begin
			$fscanf(file,"%h",mem_addr[i]);
			//wts.push_back(wt);
		end
		$fclose(file);

		//config registers
		i = 0;
		mem_addr[i+WTS_CNT] = i;
		mem_data[i+WTS_CNT] = VTH;
		i = i + 1;

		mem_addr[i+WTS_CNT] = i;
		mem_data[i+WTS_CNT] = DECAY_RATE;
		i = i + 1;

		mem_addr[i+WTS_CNT] = i;
		mem_data[i+WTS_CNT] = GROW_RATE;
		i = i + 1;

		mem_addr[i+WTS_CNT] = i;
		mem_data[i+WTS_CNT] = VREST;
		i = i + 1;

		mem_addr[i+WTS_CNT] = i;
		mem_data[i+WTS_CNT] = RESET_MECHANISM;
		i = i + 1;

		mem_addr[i+WTS_CNT] = i;
		mem_data[i+WTS_CNT] = REFRACTORY_PERIOD;
		i = i + 1;

		mem_addr[i+WTS_CNT] = i;
		mem_data[i+WTS_CNT] = LAYER_TO_MONITOR;
		i = i + 1;

		mem_addr[i+WTS_CNT] = i;
		mem_data[i+WTS_CNT] = NEURON_TO_MONITOR;
		i = i + 1;

		//read the input spikes
		for  (sim_cycles_i=0; sim_cycles_i<Number_input_files; sim_cycles_i=sim_cycles_i+1) begin
		$sformat(ispkFname,"C:/LIF/5.1_2/SNN5.1_5/quantisenc-main_neuronid/pysenc/input/test_file_%0d.txt",sim_cycles_i+File_naming);
		file=$fopen(ispkFname,"r");	//input 
		if (file)
			$display("%s was opened successfully", ispkFname);
		else
			$display("%s NOT opened", ispkFname);
		for (i=0; i<SIM_CNT; i=i+1) begin
			$fscanf(file,"%b",inspk_bfr[i][sim_cycles_i]);
		end;
		$fclose(file);
		end;

		#DELAY;
		#DELAY;
		#DELAY;
		#DELAY;
		
		//drive the input weights to the dut
		wr_en 	= 1;

		@(negedge wr_en);
		#DELAY;
		@(posedge spkclk);
		inp_en 	= 1;
		out_en 	= 1;
		
		//detect event to write the output file
		@(negedge out_en);
		@(posedge spkclk);
		#DELAY;
		@(posedge spkclk);
		//write to vmem output file
		for  (sim_cycles_o=0; sim_cycles_o<Number_output_files; sim_cycles_o=sim_cycles_o+1) begin
		$sformat(vmemFname,"C:/LIF/5.1_2/SNN5.1_5/quantisenc-main_neuronid/pysenc/output/vmem/test_vmem_output_file_fd_%0d.txt",sim_cycles_o+File_naming);
		file=$fopen(vmemFname,"w");	//vmem outfile
		if (file)
			$display("%s was opened successfully", vmemFname);
		else
			$display("%s NOT opened", vmemFname);
		for (i=0; i<SIM_CNT+EXTRA_CYCLES; i=i+1) begin
			$fwrite(file,"%b\n",vmem_bfr[i][sim_cycles_o]);
		end;
		$fclose(file);
		end;
		//write to ospk file	
		for  (sim_cycles_o=0; sim_cycles_o<Number_output_files; sim_cycles_o=sim_cycles_o+1) begin
		$sformat(ospkFname,"C:/LIF/5.1_2/SNN5.1_5/quantisenc-main_neuronid/pysenc/output/spike/test_spike_output_11_file_fd_%0d.txt",sim_cycles_o+File_naming);
		file=$fopen(ospkFname,"w");	//vmem outfile
		if (file)
			$display("%s was opened successfully", ospkFname);
		else
			$display("%s NOT opened", ospkFname);
		for (i=0; i<SIM_CNT+EXTRA_CYCLES; i=i+1) begin
			$fwrite(file,"%b\n",ospk_bfr[i][sim_cycles_o]);
		end;
		$fclose(file);
		end
		$display("end testbench");
	end
endmodule