#param.py

class parameters():
    def __init__(self):
        self.lif        = lif_params()
        self.dataset    = dataset_params()
        self.hardware   = hardware_params()
        self.model      = model_params()
        self.train      = train_params()

        #set the model parameters
        self.model.input_neurons  = self.dataset.xdim * self.dataset.ydim
        self.model.output_neurons = self.dataset.n_classes

class lif_params():
    #lif parameters
    def __init__(self):
        self.vth = 1.0
        self.neuron_c = 1
        self.neuron_r = 5
        self.neuron_decay_rate = 1 / (self.neuron_r * self.neuron_c)
        self.neuron_grow_rate = 1 / self.neuron_c
        self.vrest = 0
        self.reset_mechanism = 1
        self.refractory_period = 0
        self.beta = 0.95  #still investigating what this parameter is

        #reset for torch model
        if self.reset_mechanism == 0:
            self.torch_reset = 'none'
        elif self.reset_mechanism == 1:
            self.torch_reset = 'subtract'
        elif self.reset_mechanism == 2:
            self.torch_reset = 'zero'
        else:
            self.torch_reset = 'subtract'

class dataset_params():
    #dataset parameters
    def __init__(self):
        self.seed = 40
        self.xdim = 16 #32
        self.ydim = 16 #32
        self.n_classes = 10
        self.time_steps = 100
        self.time_embedding = False
        self.pad_samples = 10

class hardware_params():
    #hardware parameters
    def __init__(self):
        #quantisenc parameters
        self.integer_precision  = 3
        self.decimal_precision  = 4
        self.data_width         = 32
        self.layer_enc_bits     = 8
        self.neuron_enc_bits    = 12
        self.fanin_enc_bits     = 12

class model_params():
    #snn model parameters
    def __init__(self):
        #network parameters
        self.hidden_layers = 1
        self.hidden_layer_neurons = 256
        self.input_neurons = -1
        self.output_neurons = -1

class train_params():
    #snn training parameters
    def __init__(self):
        self.batch_size = 128
        self.training_epochs = 20 #40
