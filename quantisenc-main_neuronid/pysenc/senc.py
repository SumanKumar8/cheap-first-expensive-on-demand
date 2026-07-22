import snntorch as snn
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import pickle
import argparse
import click

from senclib import parameters
from senclib import str2bool
from senclib import data_loader
from senclib import accuracy
from senclib import fsnn
from senclib import flush_aer
from senclib import flush_weights
from senclib import write_hw_parameters
from senclib import train_snn

'''
###############################################################
# process arguments
###############################################################
'''
#add arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dataset','--dataset',default='mnist')
parser.add_argument('-model_train','--model_train',type=str2bool, nargs='?',const=True, default=False, help="enable/disable model training")
parser.add_argument('-model_dir','--model_dir',default='./trained_models/')
parser.add_argument('-n_images','--n_images',default=10)
#parse arguments
args            = vars(parser.parse_args())
dataset         = args['dataset']
model_train     = args['model_train']
model_dir       = args['model_dir']
n_images        = int(args['n_images'])
'''
###############################################################
'''

'''
###############################################################
# instantiate parameters
###############################################################
'''
param                   = parameters()
seed                    = param.dataset.seed
input_shape             = (param.dataset.xdim,param.dataset.ydim)
time_embedding          = param.dataset.time_embedding
time_steps              = param.dataset.time_steps
pad_samples             = param.dataset.pad_samples
batch_size              = param.train.batch_size
num_epochs              = param.train.training_epochs
vth                     = param.lif.vth
beta                    = param.lif.beta
reset_mechanism         = param.lif.reset_mechanism
torch_reset             = param.lif.torch_reset
grow_rate               = param.lif.neuron_grow_rate
decay_rate              = param.lif.neuron_decay_rate
vrest                   = param.lif.vrest
refractory_period       = param.lif.refractory_period
input_neurons           = param.model.input_neurons
hidden_layers           = param.model.hidden_layers
hidden_layer_neurons    = param.model.hidden_layer_neurons
output_neurons          = param.model.output_neurons
layer_enc_bits          = param.hardware.layer_enc_bits
neuron_enc_bits         = param.hardware.neuron_enc_bits
fanin_enc_bits          = param.hardware.fanin_enc_bits
integer_precision       = param.hardware.integer_precision
decimal_precision       = param.hardware.decimal_precision
'''
###############################################################
'''

'''
###############################################################
# seed for random number generation
###############################################################
'''
np.random.seed(seed)
torch.manual_seed(seed)
'''
###############################################################
'''

'''
###############################################################
# model instatiation
###############################################################
'''
#define the architecture
arch = [input_neurons]
for hl in range(hidden_layers):
    arch.append(hidden_layer_neurons)
arch.append(output_neurons)
arch_str = 'x'
arch_str = arch_str.join([str(a) for a in arch])
#design parameters
design_parameters = {
    'dataset': dataset,
    'time_embedding':time_embedding,
    'model_dir':model_dir,
    'arch_str':arch_str,
    'in_shape': input_shape,
    'model_arch': arch,
    'vth': vth,
    'beta': beta,
    'torch_reset': torch_reset,
    'num_steps' : time_steps,
    'num_epochs': num_epochs,
    'infer_size': n_images,
    'batch_size': batch_size,
    'integer_precision': integer_precision,
    'decimal_precision': decimal_precision
}
print('[info] define snn ',arch)
'''
###############################################################
'''

'''
###############################################################
# load dataset
###############################################################
'''
train_loader,test_loader = data_loader(design_parameters)
print('[info] loading dataset ',dataset)
'''
###############################################################
'''

'''
###############################################################
# train/test model
###############################################################
'''
if model_train:
    print('[info] train the model')
    inference_data = train_snn(design_parameters,train_loader,test_loader)
    #exit()
else:
    print('[info] load the model')
    ofname = model_dir + dataset + '.' + arch_str + '.inference.pkl'
    if os.path.exists(ofname):
        inference_data = pickle.load(open(ofname,'rb'))
    else:
        print('[info] trained model doesnot exist in path')
        if click.confirm('do you want to train the model?', default=True):
            inference_data = train_snn(design_parameters,train_loader,test_loader)
        else:
            exit()


#load all inference data
wts             = inference_data['weights']
aer_input       = inference_data['input']
aer_output      = inference_data['output']
actual_targets  = inference_data['actual_targets']
torch_targets   = inference_data['torch_targets']
print('[info] torch accuracy = ',accuracy(actual_targets,torch_targets))
'''
###############################################################
'''

'''
###############################################################
# define a reference model
###############################################################
'''
hwnet = fsnn(design_parameters)
hwnet.set_weights(wts)
#verification samples
#sample_ids  = np.random.randint(low=0, high=batch_size, size = n_images)
sample_ids  = np.arange(n_images)
sample_aer  = aer_input[:,sample_ids,:]
#obtain all predictions from the hardware-aware model
hwnet_targets = list()
for b in range(batch_size):
    data = aer_input[:,b,:].view(time_steps,1,input_neurons)
    hwnet.simulate(data,n_steps=time_steps)
    _, idx = hwnet.torch_spikes[-1].sum(dim=0).max(1)
    hwnet_targets.append(idx)
hwnet_targets = torch.stack(hwnet_targets).view(-1)

print('[info] hwnet accuracy (wrt golden) = ',accuracy(actual_targets,hwnet_targets))
print('[info] hwnet accuracy (wrt torch) = ',accuracy(torch_targets,hwnet_targets))
'''
###############################################################
'''

'''
###############################################################
# save the sample labels
###############################################################
'''
actual_labels = actual_targets[sample_ids]
torch_labels  = torch_targets[sample_ids]
soft_labels   = hwnet_targets[sample_ids]
'''
###############################################################
'''


'''
###############################################################
# flush aer, weights, and prepare hw configuration
###############################################################
'''
#aer
print('[info] flushing aer to hardware ...')
aer = flush_aer(sample_aer,pad_samples=pad_samples)
sim_cnt = aer.shape[0]
#weights
print('[info] flushing weights to hardware ...')
afname = 'weight/snncore.synaptic_address.txt'
wfname = 'weight/snncore.synaptic_weight.txt'
wts_cnt = flush_weights(wts,afname,wfname)
#parameters
print('[info] writing parameters ...')
tbfname = '../parameters/tb_snncore_parameters.vh'  #testbench parameters
dfname  = '../parameters/parameters.vh'             #design parameters
control_parameters = {
    'vth':vth,
    'decay_rate':decay_rate,
    'grow_rate':grow_rate,
    'vrest':vrest,
    'reset_mechanism':reset_mechanism,
    'refractory_period':refractory_period,
    'layer_to_monitor':0,
    'neuron_to_monitor':0,
    'input_neurons':input_neurons,
    'output_neurons':output_neurons,
    'hidden_layers':hidden_layers,
    'hidden_layer_neurons':hidden_layer_neurons,
    'integer_precision':integer_precision,
    'decimal_precision':decimal_precision,
    'layer_enc_bits':layer_enc_bits,
    'neuron_enc_bits':neuron_enc_bits,
    'fanin_enc_bits':fanin_enc_bits,
    'sim_cnt':sim_cnt,
    'wts_cnt':wts_cnt
}
write_hw_parameters(tbfname,dfname,control_parameters)

#save data
ofname = 'output/torch_out.pkl'
out_dict = {
    'weights':wts,
    'sim_cnt':sim_cnt,
    'actual_labels':actual_labels,
    'torch_labels':torch_labels,
    'soft_labels':soft_labels
}
pickle.dump(out_dict,open(ofname,'wb'))
print('#####################################')
print('[info] hardware programming complete')
print('[info] run hardware simulation (vivado)')
print('[info] after hardware simulation completes, run hardware_decoder.py to compare hardware vs. software performance')
