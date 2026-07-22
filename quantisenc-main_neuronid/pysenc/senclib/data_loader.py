import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets, transforms

def data_loader(design_parameters):
    dataset = design_parameters['dataset']
    in_shape = design_parameters['in_shape']
    batch_size = design_parameters['batch_size']

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define a transform
    transform = transforms.Compose([
                transforms.Resize(in_shape),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    if dataset.lower() == 'mnist':
        data_path = './dataset/mnist'
        mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        # Create DataLoaders
        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)
    elif dataset.lower() == 's2s':
        data_path = './dataset/s2s'
        s2s_x_train = torch.tensor(np.load(data_path + '/x_train_balanced.npy')).to(torch.float).to(device)
        s2s_y_train = torch.tensor(np.load(data_path + '/y_train_balanced.npy')).to(device)
        s2s_x_test  = torch.tensor(np.load(data_path + '/x_test.npy')).to(torch.float).to(device)
        s2s_y_test  = torch.tensor(np.load(data_path + '/y_test.npy')).to(device)
        # Convert to tensor dataset
        s2s_train = TensorDataset(s2s_x_train, s2s_y_train)
        s2s_test  = TensorDataset(s2s_x_test,  s2s_y_test)
        # Create DataLoaders
        train_loader = DataLoader(s2s_train, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader  = DataLoader(s2s_test, batch_size=batch_size, shuffle=True, drop_last=True)
    elif dataset.lower() == 'cifar10':
        trainset = datasets.CIFAR10(root='./dataset/cifar10', train=True,download=True, transform=transform)
        testset = datasets.CIFAR10(root='./dataset/cifar10', train=False,download=True, transform=transform)
         # Create DataLoaders
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        print('[info] dataloader for dataset ',dataset, ' is not implemented')
        print('[info] edit senclib/data_loader.py to add a torch dataloader for this dataset and rerun the command')
        exit()

    return train_loader,test_loader
