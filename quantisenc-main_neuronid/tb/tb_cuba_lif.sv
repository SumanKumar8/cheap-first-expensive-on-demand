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
// File     	: tb_cuba_lif.sv
// Desc     	: This is an implementation of the testbench for a neuron with all its synaptic memory.
// Changes	: Imported from tb_cuba_lif.v. 
// 		  Added SystemVerilog constructs for flexibility.
// 		  Cleaned the file for file IO operations.
// 		  All file paths are relative.
// -----------------------------------------------------------------------------*/

`timescale 1ns / 1ps


module tb_cuba_lif #(
	`include "tb_cuba_lif_parameters.vh"
) ();
	//dut IOs	
	reg spkclk;			//dut clock for spike input
	reg memclk;			//dut clock for memory accesses
	reg rst;			//dut reset
	reg wr_en;			//dut write enable
	int wr_addr;			//dut write address
	int wr_data;			//dut write data
	reg [FANIN-1:0] inspk;		//spike input to dut

	//file name
	string wtFname 	= {INSTALL_DIR,"/pysenc/weight/cuba_lif.synaptic_weight.txt"};
	string addrFname= {INSTALL_DIR,"/pysenc/weight/cuba_lif.synaptic_address.txt"};
	string ispkFname= {INSTALL_DIR,"/pysenc/input/cuba_lif.spikes_input.txt"};
	string vmemFname= {INSTALL_DIR,"/pysenc/output/cuba_lif.vmem_output.txt"};
	string ospkFname= {INSTALL_DIR,"/pysenc/output/cuba_lif.spikes_output.txt"};
	//file io
	int wrcnt;				//weight write counter
	int file;
	//other variables
	reg prgclk;				//programming clock
	int i;					//integer variable for loop iterations

	int mem_data [WTS_CNT-1:0];				//weight mem
	int mem_addr [WTS_CNT-1:0];				//weight addr
	reg [FANIN-1:0] inspk_bfr [SIM_CNT-1:0];		//input spk buffer
	reg [PRECISION-1:0] vmem_bfr [SIM_CNT+EXTRA_CYCLES-1:0];//output vmem buffer
	reg ospk_bfr [SIM_CNT+EXTRA_CYCLES-1:0];		//output spike buffer
	
	reg inp_en;	//input enable
	reg out_en;	//output enable

	int inp_cnt;	//input counter
	int out_cnt;	//output counter

	//define all clocks
	always #SPK_CLK_PERIOD 	spkclk 	= ~spkclk;
	always #MEM_CLK_PERIOD 	memclk 	= ~memclk;
	always #PRG_CLK_PERIOD 	prgclk 	= ~prgclk;

	//instantiate the DUTs here.
	wire intspk;
	wire rst_acc;
	wire rd_en;
	int rd_addr;	
	//instantiate the DUT
	syn_access #(
		.FANIN(FANIN)
	) syn_access_dut(
		.rst(rst),
		.memclk(memclk),
		.inspk(inspk),
		.outspk(intspk),
		.rst_acc(rst_acc),
		.rd_en(rd_en),
		.rd_addr(rd_addr)
	);

	wire outspk;
	wire [PRECISION-1:0] vmem;
	cuba_lif #(
		.FANIN(FANIN),
		.INTEGER_PRECISION(INTEGER_PRECISION),
		.DECIMAL_PRECISION(DECIMAL_PRECISION)
	) cuba_lif_dut(
		.rst(rst),
		.memclk(memclk),
		.spkclk(spkclk),
		.vth(VTH),
		.decay_rate(DECAY_RATE),
		.grow_rate(GROW_RATE),
		.vrest(VREST),
		.reset_mechanism(RESET_MECHANISM),
		.refractory_period(REFRACTORY_PERIOD),
		.wr_en(wr_en),
		.wr_addr(wr_addr),
		.wr_data(wr_data),
		.rd_en(rd_en),
		.rd_addr(rd_addr),
		.rst_acc(rst_acc),
		.inspk(intspk),
		.outspk(outspk),
		.vmem(vmem)
	);

	//generate and control wr_en, wr_addr, wr_data
	always @(posedge prgclk or posedge rst) begin
		if (rst) begin
			wrcnt = 0;
		end
		else begin
			if (wr_en) begin
				if (wrcnt == WTS_CNT-1) begin
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
	//done
	
	//generate and control spike input to the dut
	always @(posedge spkclk or posedge rst) begin
		if (rst) begin
			inp_cnt <= 0;
		end
		else begin
			if (inp_en) begin
				if (inp_cnt == SIM_CNT-1) begin
					inp_en 	<= 0;
					inp_cnt <= 0;
				end
				else begin
					inp_cnt <= inp_cnt + 1;
				end

				inspk <= inspk_bfr[inp_cnt];
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
				if (out_cnt == SIM_CNT+EXTRA_CYCLES-1) begin
					out_cnt <= 0;
					out_en	<= 0;
				end
				else begin
					out_cnt <= out_cnt + 1;
				end

				vmem_bfr[out_cnt] <= vmem;
				ospk_bfr[out_cnt] <= outspk;
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

		//read the input spikes
		file=$fopen(ispkFname,"r");
		if (file)
			$display("%s was opened successfully", ispkFname);
		else
			$display("%s NOT opened", ispkFname);
		for (i=0; i<SIM_CNT; i=i+1) begin
			$fscanf(file,"%b",inspk_bfr[i]);
		end
		$fclose(file);

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
		file=$fopen(vmemFname,"w");	//vmem outfile
		for (i=0; i<SIM_CNT+EXTRA_CYCLES; i=i+1) begin
			$fwrite(file,"%b \n",vmem_bfr[i]);
		end
		$fclose(file);
		//write to ospk file	
		file=$fopen(ospkFname,"w");	//ospk outfile
		for (i=0; i<SIM_CNT+EXTRA_CYCLES; i=i+1) begin
			$fwrite(file,"%b \n",ospk_bfr[i]);
		end
		$fclose(file);

		$display("end testbench");
	end

	
endmodule
