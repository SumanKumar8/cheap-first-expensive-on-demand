import argparse
import numpy as np

from senclib import parameters

def process_spikes(ofname,reverse=False,skip_samples=False):
    file    = open(ofname,'r')
    lines   = file.readlines()
    hwout   = list()
    for line  in lines:
        out = [int(x) for x in list(line.strip())]
        if reverse:
            hwout.append(out[::-1])
        else:
            hwout.append(out)
    #print(len(hwout),time_per_image)
    file.close()
    if skip_samples:
        processed_hwout     = hwout[pad_samples:-pad_samples]
    else:
        processed_hwout     = hwout

    n_processed_images  = int(len(processed_hwout) / time_per_image)
    labels              = list()
    outputs             = list()
    start_index = 0
    for i in range(n_processed_images):
        end_index   = start_index + time_per_image
        hw_image_spikes = np.zeros(n_output)
        for a in processed_hwout[start_index:end_index]:
            hw_image_spikes += np.array(a)
        #print(i,hw_image_spikes,np.argmax(hw_image_spikes))
        labels.append(np.argmax(hw_image_spikes))
        outputs.append(hw_image_spikes)
        start_index = end_index
    return labels,outputs

def print_performance(pred,act,pval,aval):
    debug = True
    correct = 0
    almost_correct = 0
    incorrect = 0

    for i in range(len(pval)):
        p = pval[i]
        a = aval[i]
        
        pmax_idx = np.argmax(p)
        amax_idx = np.argmax(a)

        pidx = np.where(p == np.max(p))[0]
        aidx = np.where(a == np.max(a))[0]

        common_idx = np.intersect1d(pidx,aidx)
        
        if pmax_idx == amax_idx:
            correct += 1
            if debug:
                print('Prediction: ',pmax_idx, ' and Actual = ',amax_idx)
                print('HW: ',p)
                print('SW: ',a)
        elif len(common_idx) > 0:
            almost_correct += 1
        else:
            incorrect += 1
            if debug:
                print('Prediction: ',pmax_idx, ' and Actual = ',amax_idx)
                #print(common_idx)
                #print('Predictions = ',pidx)
                #print('Actual = ',aidx)
                print('HW: ',p)
                print('SW: ',a)

    cp = 100.0 * (correct + almost_correct) / len(pred)
    ip = 100.0 * incorrect / len(pred)

    print('[info] total samples = ',len(pred), ' correct samples = ',(correct+almost_correct),' (',cp,')', ' and incorrect samples = ',incorrect,'(',ip,')')

'''
###############################################################
# instantiate parameters
###############################################################
'''
param = parameters()
time_per_image  = param.mnist.time_samples_per_pixel + param.mnist.pad_samples
pad_samples     = param.mnist.pad_samples
n_output        = param.model.output_neurons
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
parser.add_argument('-scope','--scope',default='snncore')
parser.add_argument('-n_images','--n_images',default=10)
#parse arguments
args    = vars(parser.parse_args())
scope   = args['scope']
n_images  = int(args['n_images'])
'''
###############################################################
'''

'''
###############################################################
# load actual output
###############################################################
'''
hwfname = '../output/'+scope+'.spikes_output.txt'
swfname = 'ref/'+scope+'.spikes.ref.txt'

hw_labels,hw_output = process_spikes(hwfname,reverse=True,skip_samples=True)
sw_labels,sw_output = process_spikes(swfname)
#print(hw_labels)
#print(sw_labels)

print_performance(hw_labels,sw_labels,hw_output,sw_output)
