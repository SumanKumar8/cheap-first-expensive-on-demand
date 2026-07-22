import numpy as np

import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as func
import snntorch.utils as utils

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from senclib import parameters

class fsnn():
    def __init__(self,design_parameters):
        self.layers  = list() #list of layers
        self.lifs    = list() #list of lif neuron types
        self.mems    = list() #list of membrane potentials

        torch_reset = design_parameters['torch_reset']
        beta        = design_parameters['beta']
        vth         = design_parameters['vth']
        arch        = design_parameters['model_arch']

        #param = parameters()
        #reset_mechanism = param.lif.reset_mechanism
        #beta            = param.lif.beta
        #vth             = param.lif.vth

        #instantiate the layers
        #reset encoding
        #if reset_mechanism == 0:
        #    torch_reset = 'none'
        #elif reset_mechanism == 1:
        #    torch_reset = 'subtract'
        #elif reset_mechanism == 2:
        #    torch_reset = 'zero'
        #else:
        #    torch_reset = 'subtract'

        self.n_layers = len(arch)   #number of layers
        for i in range(1,self.n_layers):    #for each layer
            in_sz = arch[i-1]    #input size
            out_sz = arch[i]     #output size
            layer = nn.Linear(in_sz,out_sz,bias=False) #layer shape
            lif = snn.Leaky(beta=beta,
                            threshold=vth,
                            reset_mechanism=torch_reset)    #lif neuron
            self.layers.append(layer)    #save the layer
            self.lifs.append(lif)        #save the neuron type
        #initialize the state of lif neurons
        for lif in self.lifs:
            mem = lif.init_leaky()
            self.mems.append(mem)
        print('[info] Instantiate a fully-connected snn ',arch)

    def get_weights(self):
        wts = list()
        for i,layer in enumerate(self.layers):
            wt = layer.weight.detach().numpy().transpose() * 10
            wt = np.clip(wt,-0.9375,0.9375)
            wts.append(wt)
        return wts

    def get_scale(self):
        wts = list()
        for layer in self.layers:
            wt = layer.weight.detach().numpy().transpose().flatten()
            wts += list(wt)
        return (min(wts),max(wts))

    def set_weights(self,wts):
        for i in range(self.n_layers-1):
            #get the new layer weights
            wt = wts[i].transpose()
            #extract the current set of parameters
            sd = self.layers[i].state_dict()
            #update the layer weights
            sd['weight'] = torch.Tensor(wt)
            #load the new layer weights to the model
            self.layers[i].load_state_dict(sd)
            print('[info] Loading weight(',i,'-',(i+1),'):',sd['weight'].shape)

    def simulate(self,aer,n_steps=10):
        mem_recs = [list() for mem in self.mems]   #list to record membrane potential
        spk_recs = [list() for lif in self.lifs]   #list to record spikes
        #simulate the network
        for t in range(n_steps):
            spk = aer[t]    #input
            for i,layer in enumerate(self.layers):
                cur                 = layer(spk)    #current = spike * activation
                spk,self.mems[i]    = self.lifs[i](cur,self.mems[i])#mem[t+1] = post_synaptic current + decayed vmem
                spk_recs[i].append(spk)             #record the output spikes
                mem_recs[i].append(self.mems[i])    #record the membrane potential

        #process the output for final recording
        self.torch_vmems = [torch.stack(mr) for mr in mem_recs]
        self.torch_spikes= [torch.stack(sr) for sr in spk_recs]
        #convert to numpy
        self.np_vmems   = [m.detach().numpy()[:,0,:] for m in self.torch_vmems]
        self.np_spikes  = [s.detach().numpy() for s in self.torch_spikes]

        return torch.stack(spk_recs[-1], dim=0), torch.stack(mem_recs[-1], dim=0)
