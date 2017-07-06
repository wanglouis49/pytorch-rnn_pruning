import sys
from time import time

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt

from rnn import MY_RNN, MY_LSTM, MY_GRU, to_var

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


rnn = MY_GRU(input_size, hidden_size, num_layers, num_classes)
# if sys.argv[1] == 'rnn':
#     rnn = MY_RNN(input_size, hidden_size, num_layers, num_classes)
# elif sys.argv[1] == 'lstm':
#     rnn = MY_LSTM(input_size, hidden_size, num_layers, num_classes)
# elif sys.argv[1] == 'gru':
#     rnn = MY_GRU(input_size, hidden_size, num_layers, num_classes)
if torch.cuda.is_available():
	rnn.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(rnn.parameters(), lr=learning_rate)


# Train the Model
for epoch in range(num_epochs):
    t0 = time()
    for i, (images, labels) in enumerate(train_loader):
        images = to_var(images.view(-1, sequence_length, input_size))
        labels = to_var(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = rnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Time: %.2f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0], \
                     time()-t0))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = to_var(images.view(-1, sequence_length, input_size))
    outputs = rnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100. * float(correct) / total)) 


# Find out entries with weights smaller than a certain threshold

pruned_inds_by_layer = []
weight_tensors0 = []
weight_tensors1 = []
# count = 0
for p in rnn.parameters():
    pruned_inds = 'None'
    if len(p.data.size()) == 2:
    # if count == 0:
        pruned_inds = p.data.abs() < threshold
        weight_tensors0.append(p.clone())
        p.data[pruned_inds] = 0.
        weight_tensors1.append(p.clone())
    pruned_inds_by_layer.append(pruned_inds)
    # count += 1


# Re-initialize a GRU with previous weights pruned

# Re-train the network but don't update zero-weights (by setting the corresponding gradients to zero)
for epoch in range(num_epochs):
    t0 = time()
    for i, (images, labels) in enumerate(train_loader):
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
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Time: %.2f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0], \
                     time()-t0))

# Test the model
correct = 0
total = 0
for images, labels in test_loader:
    images = to_var(images.view(-1, sequence_length, input_size))
    outputs = rnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100. * float(correct) / total)) 


weight_tensors2 = []
for p in rnn.parameters():
    pruned_inds = 'None'
    if len(p.data.size()) == 2:
        pruned_inds = p.data.abs() < threshold
        weight_tensors2.append(p.clone())
    pruned_inds_by_layer.append(pruned_inds)

param = list(rnn.parameters())

plt.figure(0)
plt.subplot(3,1,1)
plt.hist(weight_tensors0[0].data.numpy().reshape(384*28,1), bins=100)
plt.subplot(3,1,2)
plt.hist(weight_tensors0[1].data.numpy().reshape(384*128,1), bins=100)
plt.subplot(3,1,3)
plt.hist(weight_tensors0[2].data.numpy().reshape(10*128,1), bins=100)
plt.figure(1)
plt.subplot(3,1,1)
plt.hist(weight_tensors1[0].data.numpy().reshape(384*28,1), bins=100)
plt.subplot(3,1,2)
plt.hist(weight_tensors1[1].data.numpy().reshape(384*128,1), bins=100)
plt.subplot(3,1,3)
plt.hist(weight_tensors1[2].data.numpy().reshape(10*128,1), bins=100)
plt.figure(2)
plt.subplot(3,1,1)
plt.hist(weight_tensors2[0].data.numpy().reshape(384*28,1), bins=100)
plt.subplot(3,1,2)
plt.hist(weight_tensors2[1].data.numpy().reshape(384*128,1), bins=100)
plt.subplot(3,1,3)
plt.hist(weight_tensors2[2].data.numpy().reshape(10*128,1), bins=100)
plt.show()

# plt.figure(0)
# plt.subplot(5,1,1)
# plt.hist(weight_tensors0[0].numpy().reshape(384*28,1), bins=100)
# plt.subplot(5,1,2)
# plt.hist(weight_tensors0[1].numpy().reshape(384*128,1), bins=100)
# plt.subplot(5,1,3)
# plt.hist(weight_tensors0[2].numpy().reshape(384,1), bins=100)
# plt.subplot(5,1,4)
# plt.hist(weight_tensors0[3].numpy().reshape(384,1), bins=100)
# plt.subplot(5,1,5)
# plt.hist(weight_tensors0[4].numpy().reshape(10*128,1), bins=100)
# plt.figure(1)
# plt.subplot(5,1,1)
# plt.hist(weight_tensors1[0].numpy().reshape(384*28,1), bins=100)
# plt.subplot(5,1,2)
# plt.hist(weight_tensors1[1].numpy().reshape(384*128,1), bins=100)
# plt.subplot(5,1,3)
# plt.hist(weight_tensors1[2].numpy().reshape(384,1), bins=100)
# plt.subplot(5,1,4)
# plt.hist(weight_tensors1[3].numpy().reshape(384,1), bins=100)
# plt.subplot(5,1,5)
# plt.hist(weight_tensors1[4].numpy().reshape(10*128,1), bins=100)
# plt.figure(2)
# plt.subplot(5,1,1)
# plt.hist(weight_tensors2[0].numpy().reshape(384*28,1), bins=100)
# plt.subplot(5,1,2)
# plt.hist(weight_tensors2[1].numpy().reshape(384*128,1), bins=100)
# plt.subplot(5,1,3)
# plt.hist(weight_tensors2[2].numpy().reshape(384,1), bins=100)
# plt.subplot(5,1,4)
# plt.hist(weight_tensors2[3].numpy().reshape(384,1), bins=100)
# plt.subplot(5,1,5)
# plt.hist(weight_tensors2[4].numpy().reshape(10*128,1), bins=100)
# plt.show()

# Look at the weights distribution again








