import sys
from time import time

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt

from rnn import *


model = sys.argv[1]

# Hyper parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01
threshold = 0.5

# MNIST Dataset
train_dataset = dsets.MNIST(root='../data/',
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='../data/',
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


if model == 'rnn':
    rnn = RNN(input_size, hidden_size, num_layers, num_classes)
elif model == 'lstm':
    rnn = LSTM(input_size, hidden_size, num_layers, num_classes)
elif model == 'gru':
    rnn = GRU(input_size, hidden_size, num_layers, num_classes)

if torch.cuda.is_available():
	rnn.cuda()
	rnn.load_state_dict(torch.load('model/'+model+'.pkl'))
else:
	rnn.load_state_dict(torch.load('model/'+model+'.pkl'))
	rnn.cpu()










