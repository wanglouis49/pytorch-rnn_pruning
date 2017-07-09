import sys
from time import time
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# import matplotlib.pyplot as plt
from rnn import *


model = sys.argv[1]

# Hyper parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 10
batch_size = 128
num_epochs = 1
learning_rate = 0.001
pruning_percentage = sys.argv[2]  # %


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

rnn.load_state_dict(torch.load('model/'+model+'.pkl'))
if torch.cuda.is_available():
	rnn.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(rnn.parameters(), lr=learning_rate)

# prune the weights
pruned_inds_by_layer = []
w_original = []
w_pruned = []
# count = 0
for p in rnn.parameters():
    pruned_inds = 'None'
    if len(p.data.size()) == 2:
    	threshold = np.percentile(p.cpu().data.abs().numpy(), pruning_percentage)
        pruned_inds = p.data.abs() < threshold
        w_original.append(p.cpu().clone())
        p.data[pruned_inds] = 0.
        w_pruned.append(p.cpu().clone())
    pruned_inds_by_layer.append(pruned_inds)


accuracies = [compute_accuracy(rnn, sequence_length, input_size, test_loader)]
losses = []
# Re-train the network but don't update zero-weights (by setting the corresponding gradients to zero)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
    	t0 = time()
        images = to_var(images.view(-1, sequence_length, input_size))
        labels = to_var(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = rnn(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # zero-out all the gradients corresponding to the pruned connections
        for l,p in enumerate(rnn.parameters()):
            pruned_inds = pruned_inds_by_layer[l]
            if type(pruned_inds) is not str:
                p.grad.data[pruned_inds] = 0.

        optimizer.step()

        losses.append(loss.data[0])
        
        if (i+1) % 100 == 0:
            accuracy = compute_accuracy(rnn, sequence_length, input_size, test_loader)
            accuracies.append(accuracy)
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Accuracy: %.2f%%, Time: %.2fs' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0], \
                     accuracy, time()-t0))

w_retrained = []
for p in rnn.parameters():
    if len(p.data.size()) == 2:
        w_retrained.append(p.cpu().clone())

pruned_inds = []
for item in pruned_inds_by_layer:
    if type(item) != str:
        pruned_inds.append(item.cpu())

# compute_accuracy(rnn, sequence_length, input_size, test_loader, model='test')
with open('model/'+model+'_retrained_conv.pkl','w') as f:
    pkl.dump(dict(losses=losses, accuracies=accuracies, w_original=w_original,\
    	w_pruned=w_pruned, w_retrained=w_retrained, pruned_inds_by_layer=pruned_inds), f)

# Save the Model
torch.save(rnn.cpu().state_dict(), 'model/'+model+'_retrained.pkl')





