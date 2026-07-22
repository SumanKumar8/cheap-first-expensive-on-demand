#variables
#set spike frequency in MHz
set SPK_FREQ 		0.1
#spike clock period is in ns
set CLK_PERIOD 		[expr 1000 / $SPK_FREQ];

#clock uncertainty
set CLK_UNCERTAINTY 0.1

#input/output delay
set INPUT_DELAY_CLK	[expr $CLK_PERIOD * 0.2]
set OUTPUT_DELAY_CLK	[expr $CLK_PERIOD * 0.2]

#clock constraints
create_clock 		-period $CLK_PERIOD 		-name clk 	[get_ports clk]
set_clock_uncertainty 	-setup $CLK_UNCERTAINTY 			[get_clocks clk]
set_clock_uncertainty 	-hold $CLK_UNCERTAINTY 				[get_clocks clk]

#reset constraints
set_false_path -from [get_ports rst]

#set input delays
set_input_delay -clock clk -rise -max $INPUT_DELAY_CLK [get_ports vth*]
set_input_delay -clock clk -fall -max $INPUT_DELAY_CLK [get_ports vth*]

set_input_delay -clock clk -rise -max $INPUT_DELAY_CLK [get_ports decay*]
set_input_delay -clock clk -fall -max $INPUT_DELAY_CLK [get_ports decay*]

set_input_delay -clock clk -rise -max $INPUT_DELAY_CLK [get_ports grow*]
set_input_delay -clock clk -fall -max $INPUT_DELAY_CLK [get_ports grow*]

set_input_delay -clock clk -rise -max $INPUT_DELAY_CLK [get_ports vrest*]
set_input_delay -clock clk -fall -max $INPUT_DELAY_CLK [get_ports vrest*]

set_input_delay -clock clk -rise -max $INPUT_DELAY_CLK [get_ports reset*]
set_input_delay -clock clk -fall -max $INPUT_DELAY_CLK [get_ports reset*]

set_input_delay -clock clk -rise -max $INPUT_DELAY_CLK [get_ports refractory*]
set_input_delay -clock clk -fall -max $INPUT_DELAY_CLK [get_ports refractory*]

set_input_delay -clock clk -rise -max $INPUT_DELAY_CLK [get_ports inspk*]
set_input_delay -clock clk -fall -max $INPUT_DELAY_CLK [get_ports inspk*]

set_input_delay -clock clk -rise -max $INPUT_DELAY_CLK [get_ports activation*]
set_input_delay -clock clk -fall -max $INPUT_DELAY_CLK [get_ports activation*]

set_output_delay -clock clk -rise -max $OUTPUT_DELAY_CLK [all_outputs]
set_output_delay -clock clk -fall -max $OUTPUT_DELAY_CLK [all_outputs]

#set IOSTANDARDS
set_property IOSTANDARD LVCMOS12 [get_ports clk]
set_property IOSTANDARD LVCMOS12 [get_ports rst]
set_property IOSTANDARD LVCMOS12 [all_outputs]

#static & dynamic power constraints
set_operating_conditions -process maximum
set_operating_conditions -design_power_budget 2.0
