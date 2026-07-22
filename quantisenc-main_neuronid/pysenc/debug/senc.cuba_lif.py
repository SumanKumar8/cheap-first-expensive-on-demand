import numpy as np
import pickle
import argparse

from senclib import parameters
from senclib import fp2q
from senclib import flush_neuron_weights
from senclib import SearchReplaceStr
'''
###############################################################
# instantiate parameters
###############################################################
'''
param                   = parameters()
input_neurons           = param.model.input_neurons
hidden_layers           = param.model.hidden_layers
hidden_layer_neurons    = param.model.hidden_layer_neurons
output_neurons          = param.model.output_neurons
integer_precision       = param.hardware.integer_precision
decimal_precision       = param.hardware.decimal_precision
layer_enc_bits          = param.hardware.layer_enc_bits
neuron_enc_bits         = param.hardware.neuron_enc_bits
fanin_enc_bits          = param.hardware.fanin_enc_bits
vth                     = param.lif.vth
decay_rate              = param.lif.neuron_decay_rate
grow_rate               = param.lif.neuron_grow_rate
vrest                   = param.lif.vrest
reset_mechanism         = param.lif.reset_mechanism
refractory_period       = param.lif.refractory_period
'''
###############################################################
'''

'''
###############################################################
# process arguments
###############################################################
'''
#add arguments
parser = argparse.ArgumentParser()
parser.add_argument('-layer','--layer',default=0)
parser.add_argument('-neuron','--neuron',default=0)
#parse arguments
args    = vars(parser.parse_args())
layer   = int(args['layer'])
neuron  = int(args['neuron'])
'''
###############################################################
'''

'''
###############################################################
# save model weight
###############################################################
'''
print('[info] writing weights')
mfname      = '../output/torch_out.pkl'
afname      = '../weight/cuba_lif.synaptic_address.txt'
wfname      = '../weight/cuba_lif.synaptic_weight.txt'

mnist_data  = pickle.load(open(mfname,'rb'))
wts         = mnist_data['weights']
sim_cnt     = mnist_data['n_timesteps']
vmems       = mnist_data['vmems']
spikes      = mnist_data['spikes']
#syn_mem     = neuron_address_gen(wts,afname,wfname,layer=layer,neuron=neuron)
wts_cnt     = flush_neuron_weights(wts,afname,wfname,layer=layer,neuron=neuron)

vmem_ref    = vmems[layer][:,neuron]
spike_ref   = spikes[layer].reshape(sim_cnt,-1)[:,neuron]
'''
###############################################################
'''

'''
###############################################################
# write testbench parameters
###############################################################
'''
print('[info] writing testbench parameters')
tbfname = '../../parameters/tb_cuba_lif_parameters.vh'     #testbench parameter name

search_str =  'parameter VTH'
replace_str = 'parameter VTH = '+str(fp2q(vth))+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter DECAY_RATE'
replace_str = 'parameter DECAY_RATE = '+str(fp2q(decay_rate))+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter GROW_RATE'
replace_str = 'parameter GROW_RATE = '+str(fp2q(grow_rate))+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter VREST'
replace_str = 'parameter VREST = '+str(vrest)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter RESET_MECHANISM'
replace_str = 'parameter RESET_MECHANISM = '+str(reset_mechanism)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter REFRACTORY_PERIOD'
replace_str = 'parameter REFRACTORY_PERIOD = '+str(refractory_period)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter FANIN'
replace_str = 'parameter FANIN = '+str(input_neurons)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter INTEGER_PRECISION'
replace_str = 'parameter INTEGER_PRECISION = '+str(integer_precision)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter DECIMAL_PRECISION'
replace_str = 'parameter DECIMAL_PRECISION = '+str(decimal_precision)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter SIM_CNT'
replace_str = 'parameter SIM_CNT = '+str(sim_cnt)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)

search_str =  'parameter WTS_CNT'
replace_str = 'parameter WTS_CNT = '+str(wts_cnt)+',\t//swctrl'
SearchReplaceStr(search_str,replace_str,file=tbfname)
'''
###############################################################
'''

'''
###############################################################
# write testbench parameters
###############################################################
'''
print('[info] creating references')
vfname = 'ref/cuba_lif.vmem.ref.txt'
np.savetxt(vfname,vmem_ref,fmt='%f')

sfname = 'ref/cuba_lif.spikes.ref.txt'
np.savetxt(sfname,spike_ref,fmt='%d')
