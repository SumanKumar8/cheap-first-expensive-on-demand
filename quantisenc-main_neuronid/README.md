# quantisenc  
A (quanti)zed (s)pike-(e)nabled (n)eural (c)ore design  
  
## Design Description  
The design structure is as follows    



    snncore.v

        |---layer.v
        
        |           |---lif.v
        
        |                   |---acc.v
        
        |
        
        |---lutmem.v
        
                    |---lmem.v
        
                            |---xmem.v
        
                                    |---ymem.v

The top-level module is snncore, which can be configured to implement a multilayer perceptron.   
The core provides an interface for loading synaptic weights into its memory.  
These weights are then used to perfform synaptic computations.  
The design uses a single clock (clk) to botth load synaptic weights in the memory and also perform neural computations.  
There are two resets in the design.  
(1) mem\_rst is used to reset the memory (if needed).  
(2) spk\_rst is used to reset the lif design.  
Users can use the same reset to drive both these resets.  
  
The design implements a multi-layer percceptron model.  
It implements a N x N x N .... unit.  
In other words, the design can be configured to implement a M-Layer fully-connected architecture with N neurons per layer.  
These configurations (number of layers, number of neurons per layer, etc) can be set from the top-leveel design file snncore.v.  
The first layer of the design can interface with external input.  
Therefore, it can receive spikes from one source or multiple sources.  
This is configured using the FANIN parameter.   
  
The neuron in this design is a leaky-integrate-and-fire unit.  
The membrane potential increases linearly due to its activation.   
When there is no activation, the membrane potential decays over time.   
This decay is controlled using the R and C parameters, which can be configured from the top-level file.  
If the membrane potential exceeds a threshold (VTH), the neuron generates a spike and subsequently, the membrane voltage resets to the resting potential (VREST).  
Currently, we use the same R and C parameters to decay the membrane potential following a spike.  
There is also a refractory period, which can be controlled using the top-level REFRACTORY\_PERIOD parameter.  
   
The activation of a neuron is computed as the sum of (spike * weights) of all its pre-synaptic connections.  
This is performed in the file acc.v  
Essentially, this implements a current-based (CUBA) synaptic model.  
  
All LIF parameters can be configured via the top-level file (snncore.v).  
 
All operations use fixed point arithmetic with Qm.n notation.
Here n is the number of fraction bits and m is the number for integer bits.
We use 1 bit for sign and (m-1) bits for integer value. 
  
## Directory Structure  
src     : All verilog files of the design.  
tb      : A sample testbench to load synaptic weights and perform computations for a step input.  
xdc     : Constraints file for synthesizing the design.  
synth   : Helpful tcl commands for synthesis and sample power reports. 
  
## Bug Reporting  
If you find any bug in the design, please email anup.das@drexel.edu.  
