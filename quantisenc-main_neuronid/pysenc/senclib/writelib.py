import numpy as np
#from senclib import fp2q
import math
import numpy as np
from binary_fractions import Binary

from senclib import parameters
from senclib import fpmax
from senclib import SearchReplaceStr

'''
# parameters
'''
param           = parameters()
layer_enc_bits  = param.hardware.layer_enc_bits
neuron_enc_bits = param.hardware.neuron_enc_bits
fanin_enc_bits  = param.hardware.fanin_enc_bits

def process_aer(aer,pad_samples=10):
    aer_data = aer.detach().cpu().numpy()
    (time_steps,n_images,n_input) = aer_data.shape

    aer_pad = np.zeros((pad_samples,n_input))
    aer_out = np.zeros((pad_samples,n_input))

    for i in range(n_images):
        aer_out = np.concatenate((aer_out,aer_data[:,i,:],aer_pad),axis=0)

    return aer_out
    
def flush_aer(aer,pad_samples=10,all_dump=True):
    aer_in = process_aer(aer,pad_samples)
    aer_hw = np.flip(aer_in,axis=1)     #flip input. verilog uses [(n-1):0] while numpy uses [0:(n-1)]
    #exit()
    #this function writes aer to hardware
    #aer_in = aer.reshape(len(aer),-1)   #convert to a 2D array
    #aer_hw = np.flip(aer_in,axis=1)[0:n,:]     #flip input. verilog uses [(n-1):0] while numpy uses [0:(n-1)]
    ofname = 'input/snncore.spikes_input.txt'
    np.savetxt(ofname,aer_hw,delimiter='',fmt='%d')
    if (all_dump):
        ofname = 'input/cuba_lif.spikes_input.txt'
        np.savetxt(ofname,aer_hw,delimiter='',fmt='%d')
        ofname = 'input/layer.spikes_input.txt'
        np.savetxt(ofname,aer_hw,delimiter='',fmt='%d')
    
    return aer_hw

def flush_weights_org(weights):
    #this function writes weights to hardware
    wts   = list()   #weights
    addrs = list()   #addresses
    n_layers = len(weights) #number of hardware layers
    for l in range(n_layers):
        #for each layer
        wt_mat = weights[l] #weight matrix
        (in_sz,out_sz) = wt_mat.shape
        for j in range(out_sz):
            for i in range(in_sz):
                wt = wt_mat[i,j]
                addr = address_encoding(layer=l,neuron=j,fanin=i)
                wts.append(process_hex(fp2q(wt),nbits=32))
                addrs.append(addr)
    afname = 'weight/synaptic_address.txt'
    dfname = 'weight/synaptic_weight.txt'

    #write addresses
    with open(afname,'w') as f:
        for addr in addrs:
            f.write(addr + '\n')
    #write data
    with open(dfname,'w') as f:
        for wt in wts:
            f.write(str(wt) + '\n')

def address_encoding(layer=1,neuron=16,fanin=256):
    #address = <--8 bits--><--12 bits--><--12 bits-->
    lhex = process_hex(layer,nbits=layer_enc_bits)
    nhex = process_hex(neuron,nbits=neuron_enc_bits)
    fhex = process_hex(fanin,nbits=fanin_enc_bits)
    return lhex + nhex + fhex

def process_hex(x,nbits=4):
    nhex = nbits / 4    #number of hex bits
    xhex = hex(x)       #hex of x
    xhex_array = xhex.split('x')    #xhex converted into array
    xhex_char = xhex_array[1]       #xhex character
    nhex_char = len(list(xhex_char)) #number of hext characters in the converted value
    extra_hex_characters = int(nhex) - nhex_char
    for i in range(extra_hex_characters):
        xhex_char = '0' + xhex_char
    return xhex_char

def flush_neuron_weights(wts,afname,wfname,layer=0,neuron=0):
    #generate address for a single neuron
    wt_layer  = wts[layer]          #layer's weights
    wt_neuron = wt_layer[:,neuron]  #neurons's fanin weights
    addr = list()   #memory address
    data = list()   #memory data
    for i,wt in enumerate(wt_neuron):
        data.append(process_hex(fp2q(wt),nbits=32))
        addr.append(address_encoding(layer=layer,neuron=neuron,fanin=i))
    #write addresses
    with open(afname,'w') as f:
        for a in addr:
            f.write(a + '\n')
    #write data
    with open(wfname,'w') as f:
        for d in data:
            f.write(str(d) + '\n')
    
    programmed_weights = len(addr)
    return programmed_weights

def flush_layer_weights(wts,afname,wfname,layer=0,neurons=2):
    #generate address for a single neuron
    wt_layer  = wts[layer]          #layer's weights
    addr = list()   #memory address
    data = list()   #memory data

    for neuron in range(neurons):
        wt_neuron = wt_layer[:,neuron]  #neurons's fanin weights
        for i,wt in enumerate(wt_neuron):
            data.append(process_hex(fp2q(wt),nbits=32))
            addr.append(address_encoding(layer=layer,neuron=neuron,fanin=i))
        #write addresses
        with open(afname,'w') as f:
            for a in addr:
                f.write(a + '\n')
        #write data
        with open(wfname,'w') as f:
            for d in data:
                f.write(str(d) + '\n')
    
    programmed_weights = len(addr)
    return programmed_weights

def flush_weights(wts,afname,wfname):
    #generate address for a single neuron
    addr = list()   #memory address
    data = list()   #memory data

    n_layers = len(wts) #number of layers
    for layer in range(n_layers):
        wt_layer    = wts[layer]          #layer's weights
        neurons     = wt_layer.shape[1]
        for neuron in range(neurons):
            wt_neuron = wt_layer[:,neuron]  #neurons's fanin weights
            for i,wt in enumerate(wt_neuron):
                processed_wt = fp2q(wt,weights=True)
                #print(wt,processed_wt)
                data.append(process_hex(processed_wt,nbits=32))
                addr.append(address_encoding(layer=layer,neuron=neuron,fanin=i))
            #write addresses
            with open(afname,'w') as f:
                for a in addr:
                    f.write(a + '\n')
            #write data
            with open(wfname,'w') as f:
                for d in data:
                    f.write(str(d) + '\n')
    
    programmed_weights = len(addr)
    return programmed_weights



def TwosComplement(str):
    n = len(str)

    # Traverse the string to get first
    # '1' from the last of string
    i = n - 1
    while(i >= 0):
        if (str[i] == '1'):
            break

        i -= 1

    # If there exists no '1' concatenate 1
    # at the starting of string
    if (i == -1):
        return '1'+str

    # Continue traversal after the
    # position of first '1'
    k = i - 1
    while(k >= 0):

        # Just flip the values
        if (str[k] == '1'):
            str = list(str)
            str[k] = '0'
            str = ''.join(str)
        else:
            str = list(str)
            str[k] = '1'
            str = ''.join(str)

        k -= 1

    # return the modified string
    return str

def bin2fp(bin_integer,bin_fraction):
    v = 0
    for i,b in enumerate(bin_integer):
        v += b * (2 ** i)

    for i,b in enumerate(bin_fraction):
        v += b * 1 / (2 ** (i+1))

    return v

def fp2q(fp,weights=False):
    #A quantized representation is of the form n.q, where n is the number of precision bits and
    #q is the number of quantized bits. 
    #keep record if the number is negative and process it as a positive number
    param = parameters()
    n_fraction  = param.hardware.decimal_precision
    n_integer   = param.hardware.integer_precision
    if weights:
        n_integer   = 0
        fp_pos_max  = fpmax(n=0,q=n_fraction)
        fp_pos_min  = - fp_pos_max
        #print(fp_pos_max,fp_pos_min)
        fp_clip = np.clip(abs(fp),fp_pos_min,fp_pos_max)
        if fp < 0:  #
            fp = 0 - fp_clip
        else:
            fp = fp_clip
    #print('FP = ',fp)
    debug       = False
    negative    = False
    if fp < 0:
        fp = -fp
        negative = True

    #extract fraction and integer part
    (frac,dec)  = math.modf(fp)
    dec         = int(dec)
    if debug:
        print('Is the number negative: ',negative)
        print('Decimal component: ',dec, ' of type ',type(dec))
        print('Fraction component: ',frac, ' of type ',type(frac))
    
    #convert to binary
    #process the fraction part here
    ref_frac_str    = ['0' for i in range(n_fraction)]
    if frac == 0:       #if frac == 0, then replace it with a very small number
        frac = 0.0001
    frac        = f"Binary({frac}) = {Binary(frac)}".split('=')[1].split('b')[1].split('.')[1]
    frac_list   = list(frac)

    if len(frac_list) <= n_fraction:
        frac_list = frac_list + ref_frac_str
    frac_part   = frac_list[0:n_fraction]
    if debug:
        print('Converted fraction bits: ',frac)
        print('Fraction bits = ',frac_list)
        print('Truncated fraction bits = ',frac_part)

    #process the integer part here
    ref_dec_str = ['0' for i in range(n_integer)]
    if (dec == 0) or (n_integer == 0):
        dec_list = ref_dec_str
    else:
        dec         = f"Binary({dec}) = {Binary(dec)}".split('=')[1].split('b')[1]
        dec_list    = ref_dec_str + list(dec)

    dec_part    = dec_list[-n_integer:]
    if debug:
        print('Converted decimal bits: ',dec)
        print('Decimal bits = ',dec_list)
        print('Truncated decimal bits = ',dec_part)

    bin_str         = dec_part + frac_part
    if debug:
        print('Binary string with given config: ',bin_str)
    bin_str.insert(0,'0')
    if debug:
        print('Binary string adter sign addition: ',bin_str)
    
    bin_str = ''.join(bin_str)

    if debug:
        print('Absolute binary string: ',bin_str)
    #2's complement if negative
    if negative:
        bin_str = TwosComplement(bin_str)

    decimal_bin_str = int(bin_str,2)
    if debug:
        print('Signed binary string: ',bin_str)
        print('Decimal value: ',decimal_bin_str)

    return decimal_bin_str

def write_hw_parameters(tbparam,hwparam,control_parameters):
    vth = control_parameters['vth']
    grow_rate = control_parameters['grow_rate']
    decay_rate = control_parameters['decay_rate']
    vrest = control_parameters['vrest']
    reset_mechanism = control_parameters['reset_mechanism']
    refractory_period = control_parameters['refractory_period']
    layer_to_monitor = control_parameters['layer_to_monitor']
    neuron_to_monitor = control_parameters['neuron_to_monitor']
    input_neurons = control_parameters['input_neurons']
    output_neurons = control_parameters['output_neurons']
    hidden_layers = control_parameters['hidden_layers']
    hidden_layer_neurons = control_parameters['hidden_layer_neurons']
    integer_precision = control_parameters['integer_precision']
    decimal_precision = control_parameters['decimal_precision']
    layer_enc_bits = control_parameters['layer_enc_bits']
    neuron_enc_bits = control_parameters['neuron_enc_bits']
    fanin_enc_bits = control_parameters['fanin_enc_bits']
    sim_cnt = control_parameters['sim_cnt']
    wts_cnt = control_parameters['wts_cnt']


    '''
    ###############################################################
    # write testbench and design parameters
    ###############################################################
    '''
    #tbparam = '../parameters/tb_snncore_parameters.vh'  #testbench parameter name
    #hwparam  = '../parameters/parameters.vh'             #design parameters

    search_str =  'parameter VTH'
    replace_str = 'parameter VTH = '+str(fp2q(vth))+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=tbparam)

    search_str =  'parameter DECAY_RATE'
    replace_str = 'parameter DECAY_RATE = '+str(fp2q(decay_rate))+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=tbparam)

    search_str =  'parameter GROW_RATE'
    replace_str = 'parameter GROW_RATE = '+str(fp2q(grow_rate))+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=tbparam)

    search_str =  'parameter VREST'
    replace_str = 'parameter VREST = '+str(vrest)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=tbparam)

    search_str =  'parameter RESET_MECHANISM'
    replace_str = 'parameter RESET_MECHANISM = '+str(reset_mechanism)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=tbparam)

    search_str =  'parameter REFRACTORY_PERIOD'
    replace_str = 'parameter REFRACTORY_PERIOD = '+str(refractory_period)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=tbparam)

    search_str =  'parameter LAYER_TO_MONITOR'
    replace_str = 'parameter LAYER_TO_MONITOR = '+str(layer_to_monitor)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=tbparam)

    search_str =  'parameter NEURON_TO_MONITOR'
    replace_str = 'parameter NEURON_TO_MONITOR = '+str(neuron_to_monitor)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=tbparam)

    search_str =  'parameter INPUT_NEURONS'
    replace_str = 'parameter INPUT_NEURONS = '+str(input_neurons)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=hwparam)

    search_str =  'parameter OUTPUT_NEURONS'
    replace_str = 'parameter OUTPUT_NEURONS = '+str(output_neurons)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=hwparam)

    search_str =  'parameter HIDDEN_LAYERS'
    replace_str = 'parameter HIDDEN_LAYERS = '+str(hidden_layers)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=hwparam)

    search_str =  'parameter HIDDEN_LAYER_NEURONS'
    replace_str = 'parameter HIDDEN_LAYER_NEURONS = '+str(hidden_layer_neurons)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=hwparam)

    search_str =  'parameter INTEGER_PRECISION'
    replace_str = 'parameter INTEGER_PRECISION = '+str(integer_precision)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=hwparam)

    search_str =  'parameter DECIMAL_PRECISION'
    replace_str = 'parameter DECIMAL_PRECISION = '+str(decimal_precision)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=hwparam)

    search_str =  'parameter LAYER_ENC_BITS'
    replace_str = 'parameter LAYER_ENC_BITS = '+str(layer_enc_bits)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=hwparam)

    search_str =  'parameter NEURON_ENC_BITS'
    replace_str = 'parameter NEURON_ENC_BITS = '+str(neuron_enc_bits)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=hwparam)

    search_str =  'parameter FANIN_ENC_BITS'
    replace_str = 'parameter FANIN_ENC_BITS = '+str(fanin_enc_bits)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=hwparam)

    search_str =  'parameter SIM_CNT'
    replace_str = 'parameter SIM_CNT = '+str(sim_cnt)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=tbparam)

    search_str =  'parameter WTS_CNT'
    replace_str = 'parameter WTS_CNT = '+str(wts_cnt)+',\t//swctrl'
    SearchReplaceStr(search_str,replace_str,file=tbparam)
    '''
    ###############################################################
    '''
