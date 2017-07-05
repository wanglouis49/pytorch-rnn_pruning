import sys
from time import time

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from rnn import MY_RNN, MY_LSTM, to_var

# Hyper parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 20
learning_rate = 0.01

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


rnn = MY_RNN(input_size, hidden_size, num_layers, num_classes)
if torch.cuda.is_available():
	rnn.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)


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
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100. * float(correct) / total)) 

# Save the Model
# torch.save(rnn.state_dict(), 'rnn.pkl')