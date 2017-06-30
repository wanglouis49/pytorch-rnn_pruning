import sys

import numpy as np
# from time import time

import torch 
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from my_nn import *

num_layers = int(sys.argv[1])
kernel_size = int(sys.argv[2])
padding = int(sys.argv[3])


# Image Preprocessing 
transform = transforms.Compose([
    transforms.Scale(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])


# CIFAR-10 Dataset
train_dataset = dsets.CIFAR10(root='../data/',
                               train=True, 
                               transform=transform,
                               download=True)

test_dataset = dsets.CIFAR10(root='../data/',
                              train=False, 
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100, 
                                          shuffle=False)

num_epochs = 20
batch_size = 100
learning_rate = 0.001

#cnn = CNN1()
if num_layers == 1:
	cnn = CNN1(kernel_size, padding)
elif num_layers == 2:
	cnn = CNN2(kernel_size, padding)
elif num_layers == 3:
	cnn = CNN3(kernel_size, padding)
elif num_layers == 4:
	cnn = CNN4(kernel_size, padding)
elif num_layers == 5:
	cnn = CNN5(kernel_size, padding)
cnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

f = open('results.txt', 'w')
f.write("=== 1-layer cnn with adam ===\n")


# Train the Model
for epoch in range(num_epochs):
    # t0 = time()
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # if (i+1) % 100 == 0:
        #     print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Time: %.2fs' 
        #            %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0], time()-t0))


# Change model to 'eval' mode (BN uses moving mean/var).
cnn.eval()

# Training accuracy
correct = 0
total = 0
for images, labels in train_loader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

f.write('Train Accuracy of the model on the 50000 test images: %f %%\n' % (100. * correct / total))



# Test accuracy
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

f.write('Test Accuracy of the model on the 10000 test images: %f %%\n' % (100. * correct / total))


param = list(cnn.parameters())


for i in range(len(param)):
	vis_param = param[i]
	w = vis_param.view(-1).data.numpy()
	f.write("layer shape %s - active weights %d/%d\n" % (str(vis_param.size()), w.nonzero()[0].shape, w.shape))

f.write('\n')
f.close()




