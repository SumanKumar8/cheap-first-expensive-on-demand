import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt

from senclib import parameters
from senclib import q2fp
from senclib import rmse
'''
###############################################################
# instantiate parameters
###############################################################
'''
param                   = parameters()
integer_precision       = param.hardware.integer_precision
decimal_precision       = param.hardware.decimal_precision
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
parser.add_argument('-scope','--scope',default='cuba_lif')
parser.add_argument('-hw_start','--hw_start',default=0)
parser.add_argument('-sw_start','--sw_start',default=0)
parser.add_argument('-samples','--samples',default=0)
#parse arguments
args    = vars(parser.parse_args())
scope   = args['scope']
hw_start= int(args['hw_start'])
sw_start= int(args['sw_start'])
c_samples = int(args['samples'])

'''
###############################################################
'''

'''
###############################################################
# read hardware output
###############################################################
'''
print('[info] reading hardware output')
hfname      = '../output/'+scope+'.vmem_output.txt'
file        = open(hfname,'r')
lines       = file.readlines()
vmem_hw     = list()
for line  in lines:
    fp = q2fp(line.strip(),integer_precision=integer_precision,decimal_precision=decimal_precision)
    vmem_hw.append(fp)
file.close()
#for v in vmem_hw:
#    print(v)
#exit()
'''
###############################################################
'''

'''
###############################################################
# read software output
###############################################################
'''
print('[info] reading software output')
sfname  = 'ref/'+scope+'.vmem.ref.txt'
vmem_sw = np.loadtxt(sfname)
'''
###############################################################
'''

'''
###############################################################
# compare vmem and spikes
###############################################################
'''
print('[info] compare vmem and spikes')
hfname = '../output/'+scope+'.spikes_output.txt'
sfname = 'ref/'+scope+'.spikes.ref.txt'

if scope == 'cuba_lif':
    hspikes = np.loadtxt(hfname)
    sspikes = np.loadtxt(sfname)
else:
    hspikes = list()
    hspikes_ind = list()
    hfile   = open(hfname,'r')
    hlines  = hfile.readlines()
    for line  in hlines:
        line_array = list(line.strip())
        line_list = [int(a) for a in line_array]
        hspikes = hspikes + line_list
        hspikes_ind.append(line_list)
    hfile.close()
    
    sspikes = list()
    sspikes_ind = list()
    sfile   = open(sfname,'r')
    slines  = sfile.readlines()
    for line  in slines:
        line_array = list(line.strip())
        line_list = [int(a) for a in line_array]
        sspikes = sspikes + line_list
        sspikes_ind.append(line_list)
    sfile.close()
    n = len(hspikes_ind[0])
    for k in range(n):
        chw = [l[n-k-1] for l in hspikes_ind]
        csw = [l[k] for l in sspikes_ind]
        print('[info] software spikes = ',int(np.sum(csw)),' and hardware spikes = ',int(np.sum(chw)))

#find the first non-zero element of vmem_hw and vmem_sw
a = next((i for i, x in enumerate(vmem_hw) if x), None) - 1
if (hw_start > 0):
    a = hw_start

b = next((i for i, x in enumerate(vmem_sw) if x), None)
if (sw_start > 0):
    b = sw_start

comparable_vmem_hw = vmem_hw[a:]
comparable_vmem_sw = vmem_sw[b:]
n_samples_hw = len(comparable_vmem_hw)
n_samples_sw = len(comparable_vmem_sw)
n_samples = min(n_samples_hw,n_samples_sw)
if c_samples > 0:
    n_samples = c_samples
data_hw = comparable_vmem_hw[0:n_samples]
data_sw = comparable_vmem_sw[0:n_samples]
r = rmse(data_hw,data_sw)
print('[info] software spikes = ',int(np.sum(sspikes)),' and hardware spikes = ',int(np.sum(hspikes)), ' vmem difference (rmse) = ',r)
'''
###############################################################
'''


'''
###############################################################
# plot vmems
###############################################################
'''
print('[info] plotting vmems')
font_size = 14
fig, ax = plt.subplots(figsize=(12, 4.0))
p1 = ax.plot(data_sw,'-k',label='software')
p2 = ax.plot(data_hw,'-r',label='hardware')
#p1 = ax.plot(sspikes[b:b+n_samples],'-k',label='software')
#p2 = ax.plot(hspikes[a:a+n_samples],'-r',label='hardware')
legend =  ax.legend((p1[0], p2[0]), ('software','hardware'),fontsize=font_size,loc=4,ncol=1,shadow=False,columnspacing=0.5,handletextpad=0.3)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(font_size)

ax.set_ylabel('Membrane Potential ($V$)', multialignment='center',fontsize=font_size)
ax.set_xlabel('Time (ms)', multialignment='center',fontsize=font_size)
ax.autoscale_view()
plt.grid(linestyle='dotted')
plt.tight_layout()
plt.show()
'''
###############################################################
'''
