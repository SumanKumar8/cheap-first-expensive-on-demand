import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import snntorch.utils as utils

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

import pickle

from senclib import fpmax

# Define Network
class Net(nn.Module):
    def __init__(self,
                 num_inputs=1,
                 num_hidden=1,
                 num_outputs=1,
                 num_steps=10,
                 beta=0.8,
                 vth=1.0,
                 rst='zero',
                 bias=False):
        super().__init__()
        
        #num steps
        self.num_steps = num_steps

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden, bias=bias)
        self.lif1 = snn.Leaky(beta=beta,threshold=vth,reset_mechanism=rst)
        self.fc2 = nn.Linear(num_hidden, num_outputs, bias=bias)
        self.lif2 = snn.Leaky(beta=beta,threshold=vth,reset_mechanism=rst)

    def forward(self, x):
        # Convert to rate
        #x_rate = snn.rate_code(x)
        #x_rate = spikegen.rate(x,num_steps=self.num_steps)
        x_rate = x

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(x_rate[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

def train_snn(design_parameters,train_loader,test_loader):
    dataset     = design_parameters['dataset']
    time_embedding  = design_parameters['time_embedding']
    model_dir   = design_parameters['model_dir']
    arch_str    = design_parameters['arch_str']
    batch_size  = design_parameters['batch_size']
    num_epochs  = design_parameters['num_epochs']
    model_arch  = design_parameters['model_arch']
    num_steps   = design_parameters['num_steps']
    beta        = design_parameters['beta']
    vth         = design_parameters['vth']
    rst         = design_parameters['torch_reset']

    n = design_parameters['integer_precision']
    q = design_parameters['decimal_precision']
    wt_max = fpmax(n=0,q=q)
    wt_min = - wt_max

    num_inputs = model_arch[0]
    num_hidden = model_arch[1]
    num_outputs = model_arch[-1]
    num_hidden_layers = len(model_arch)-2

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(train_loader)

    # Load the network onto CUDA if available
    net = Net(num_inputs=num_inputs,
              num_hidden=num_hidden,
              num_outputs=num_outputs,
              num_steps=num_steps,
              beta=beta,
              vth=vth,
              rst=rst).to(device)
    # Loss and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    # Output file names
    oname = model_dir + dataset + '.' + arch_str
    dict_name = oname + '.dict'
    model_name = oname + '.mdl'
    #wts_name = oname + '.pkl'
    inference_name = oname + '.inference.pkl'

    # Main training framework starts here
    utils.reset(net)
    loss_hist = []
    test_loss_hist = []
    counter = 0
    weight_scaling = True
    #if not time_embedding:
    #    print('bla bla')
    #exit()

    # Outer training loop
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)
        # Minibatch training loop
        for data, targets in train_batch:
            #print(data.type(),targets.type())
            #exit()
            if not time_embedding:
                data = spikegen.rate(data.view(batch_size,-1),num_steps=num_steps).to(device)
            else:
                data = torch.swapaxes(data,0,1).view(num_steps,batch_size,-1).to(device)
            #data = data.to(device)
            targets = targets.to(device)
            # forward pass
            net.train()
            spk_rec, mem_rec = net(data)
            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)
            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            # Update weights
            for model_params in net.parameters():
                if weight_scaling:
                    model_params.data = torch.clamp(model_params.data,min=wt_min,max=wt_max)
            # Store loss history for future plotting
            loss_hist.append(loss_val.item())
            # For inference
            # batch_infer = list()
            # Test set
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                if not time_embedding:
                    test_data = spikegen.rate(test_data.view(batch_size,-1),num_steps=num_steps).to(device)
                else:
                    test_data = torch.swapaxes(test_data,0,1).view(num_steps,batch_size,-1).to(device)
                #test_data = test_data.to(device)
                test_targets = test_targets.to(device)
                # Test set forward pass
                test_spk, test_mem = net(test_data)
                # Test set loss
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())
                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    train_printer(
                        net, batch_size,
                        data, targets, epoch,
                        counter, iter_counter,
                        loss_hist, test_loss_hist,
                        test_data, test_targets)
                counter += 1
                iter_counter +=1
    # Save output
    torch.save(net,model_name)
    torch.save(net.state_dict(),dict_name)

    # Prepare for hardware inference
    wts = list()
    for model_params in net.parameters():
        wts.append(model_params.data.cpu().detach().numpy().transpose())
    test_data, test_targets = next(iter(test_loader))
    if not time_embedding:
        test_data = spikegen.rate(test_data.view(batch_size,-1),num_steps=num_steps).to(device)
    else:
        test_data = torch.swapaxes(test_data,0,1).view(num_steps,batch_size,-1).to(device)
    #test_data = test_data.to(device)
    test_targets = test_targets.to(device)
    output, _ = net(test_data)
    _, idx    = output.sum(dim=0).max(1)
    acc = np.mean((test_targets == idx).detach().cpu().numpy())
    print(f"Test batch accuracy: {acc*100:.2f}%")
    inference = {
        'input': test_data.cpu(),
        'output': output.cpu(),
        'actual_targets': test_targets.cpu(),
        'torch_targets': idx.cpu(),
        'weights': wts
    }
    pickle.dump(inference,open(inference_name,'wb'))
    return inference


# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target
def print_batch_accuracy(net, batch_size, data, targets, train=False):
    output, _ = net(data)
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(
    net, batch_size,
    data, targets, epoch,
    counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(net, batch_size, data, targets, train=True)
    print_batch_accuracy(net, batch_size, test_data, test_targets, train=False)
    print("\n")
