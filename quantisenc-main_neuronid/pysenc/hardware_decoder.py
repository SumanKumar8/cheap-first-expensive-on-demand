import numpy as np
import pickle
import torch
from senclib import parameters
from senclib import accuracy

def process_spikes(ofname,reverse=False):
    file    = open(ofname,'r')
    lines   = file.readlines()
    hwout   = list()
    for line  in lines:
        out = [int(x) for x in list(line.strip())]
        if reverse:
            hwout.append(out[::-1])
        else:
            hwout.append(out)
    hardware_skips_start = 15
    hardware_skip_end    = 15
    #
    hwout_processed = hwout[hardware_skips_start:-hardware_skip_end]
    total_samples  = len(hwout_processed)
    n_output = len(hwout_processed[0])
    #
    n_images = total_samples / (time_steps + pad_samples)
    hwout_processed_np = np.zeros((total_samples,n_output))
    for i,oevent in enumerate(hwout_processed):
        hwout_processed_np[i] = np.array(oevent)
    #
    hwout_labels = list()
    start_index = 0
    for i in range(int(n_images)):
        end_index = start_index + time_steps + pad_samples
        output = torch.tensor(hwout_processed_np[start_index:end_index].reshape(-1,1,n_classes))
        _, idx    = output.sum(dim=0).max(1)
        hwout_labels.append(idx)
        start_index = end_index
    
    return torch.stack(hwout_labels).view(-1)

if __name__ == "__main__":
    #process the hardware
    #ofile name
    param = parameters()
    pad_samples = param.dataset.pad_samples
    time_steps  = param.dataset.time_steps
    n_classes   = param.dataset.n_classes
    ospk_fname = 'output/snncore.spikes_output.txt'
    hardware_labels = process_spikes(ospk_fname,reverse=True)

    #read the software results
    ofname = 'output/torch_out.pkl'
    out_dict = pickle.load(open(ofname,'rb'))
    actual_labels = out_dict['actual_labels']
    torch_labels = out_dict['torch_labels']
    soft_labels = out_dict['soft_labels']

    print('Actual Labels = ',actual_labels)
    print('Hardware Labels = ',hardware_labels)

    #accuracy comparison
    print('Hardware vs. Software (Ground Truth): ',accuracy(actual_labels,hardware_labels))
    print('Hardware vs. Software (Torch): ',accuracy(torch_labels,hardware_labels))
    print('Hardware vs. Software (Hardware-Aware Torch Model): ',accuracy(soft_labels,hardware_labels))
