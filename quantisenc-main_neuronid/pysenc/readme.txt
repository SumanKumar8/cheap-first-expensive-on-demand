This is the implementation of software (compiler and run-time) for QUANTISENC.
The design is organized into the following directories.
.
├── dataset (this is where the dataset is downloaded)
│   ├── mnist (for mnist)
│   └── s2s (for speech-to-spike)
├── debug (this folder contains files for debugging the design) 
├── input (this folder contains the spike input for the hardware)
├── output (this folder contains output from the hardware)
├── senclib (this follder contains libraries for model compilation)
├── trained_models (this folder contains the trained model)
└── weight (this folder contains the synaptic weights for the hardware)

File contrent of these folders are desribed next.

1. dataset
==========
The dataset is the folder where all the datasets are downloaded.
The design has been verified with MNIST.
The speech-to-spike (s2s) is a work-in-progress.
dataset
├── mnist
│   └── MNIST
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
└── s2s
    ├── x_test.npy
    ├── x_train_balanced.npy
    ├── y_test.npy
    └── y_train_balanced.npy

2. input
========
The input folder contains the spike input to the QUANTISENC design.
The design can be simulated at three levels of hierarchy:
a) cuba_lif: An implementation of a single neuron with a programmable number of pre-synaptic connections.
To simulate cuba_lif, the testbench (in the design folder) uses cuba_lif.spikes_input.txt file.

b) layer: An implementation of a layer of multiple neurons,each with a programmable number of pre-synaptic connections.
To simulate a layer, the testbench (in the design folder) uses layer.spikes_input.txt file.

c) snncore: An implementation of the design with multiple layers, each with neurons and their pre-synaptic connections.
To simulate a core, the testbench (in the design folder) uses snncore.spikes_input.txt file.

input
├── cuba_lif.spikes_input.txt
├── layer.spikes_input.txt
└── snncore.spikes_input.txt

