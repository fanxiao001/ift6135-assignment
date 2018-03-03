# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 15:38:40 2018

@author: lingyu.yue
"""

import time
import numpy as np
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms

import matplotlib.pyplot as plt

#%%
mnist_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=mnist_transforms, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=mnist_transforms, download=True)

print('mnist_train:',len(mnist_train))
print('mnist_test:',len(mnist_test))


#%%
class MLP_dropout(nn.Module):
    """MLP with dropout"""
    def __init__(self):
        super(MLP_dropout, self).__init__()
        self.net = nn.Sequential(

            nn.Linear(784,800),
            nn.ReLU(),
            nn.Linear(800,800),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(800,10)
        )

    def forward(self, x):
        return self.net(x)
    
    
#%%
cuda = torch.cuda.is_available()
model = MLP_dropout()
if cuda:
    model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
criterion = nn.NLLLoss()
logsoft = nn.LogSoftmax()

acc_train = list()
acc_test = list()

trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True, num_workers=2)

for epoch in range(100):
    losses = []
    # Train
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        optimizer.zero_grad()
        inputs = Variable(inputs).view(-1,784)
        targets =  Variable(targets).view(-1)
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = logsoft(model.forward(inputs))
        loss = criterion(outputs, targets) #-x[class]
        loss.backward()
        optimizer.step()
        losses.append(loss.data[0])

    print('Epoch : %d Loss : %.3f ' % (epoch, np.mean(losses)))
    

#%%

# multiply 0.5 to the hidden layer before prediction
# Evaluate
def evaluate1 ():
    testloader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True, num_workers=2)
    model.eval()
    
    count = 0
    acc_test = 0
    for batch_idx, (inputs, targets) in enumerate(testloader): 
    
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        inputs = Variable(inputs).view(-1,784)
        targets = Variable(targets).view(-1)
        outputs = logsoft(model.forward(inputs))  
    
        _, predicted = torch.max(outputs.data, 1)
    
        count += targets.size(0)
        acc_test += predicted.eq(targets.data).cpu().sum()     
    
    acc_test = acc_test/(count*1.0)
    print('No mask, Test Acc : %.3f' % (100.*acc_test))  #Test Acc : 98.360
    print('--------------------------------------------------------------')
    return 100.*acc_test


#%%

# Sample N dropout masks , average before softmax
def evaluate2(num_mask) : 
    testloader = torch.utils.data.DataLoader(mnist_test, batch_size = len(mnist_test), shuffle=False, num_workers=2)
    N = num_mask
    outputs = torch.zeros(len(mnist_test),10)
    for mask in range(N) :
        model.train()
        for batch_idx, (inputs, targets) in enumerate(testloader): 
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
        
            inputs = Variable(inputs).view(-1,784)
            targets = Variable(targets).view(-1)
            outputs.add_(model.forward(inputs).data)
            
    #outputs = outputs/(1.0*N)
    _,predicted = torch.max(outputs/(1.0 * N),1)
    acc_test = predicted.eq(targets.data).cpu().sum()/(1.0*len(mnist_test))
    
    print('Pre-softmax, %d masks, Test Acc : %.3f' % ( N, 100.*acc_test))
    print('--------------------------------------------------------------')
    return 100.*acc_test
    

#%%
# Sample N dropout masks , make a prediction and take the average
def evaluate3(num_mask):
    testloader = torch.utils.data.DataLoader(mnist_test, batch_size = len(mnist_test), shuffle=False, num_workers=2)
    N = num_mask
    outputs = torch.zeros(len(mnist_test),10)
    model.train()
    for mask in range(N) :
        for batch_idx, (inputs, targets) in enumerate(testloader): 
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
        
            inputs = Variable(inputs).view(-1,784)
            targets = Variable(targets).view(-1)
            outputs.add_(logsoft(model.forward(inputs)).data)
            
    _,predicted = torch.max(outputs/(1.0 * N),1)
    acc_test = predicted.eq(targets.data).cpu().sum()/(1.0*len(mnist_test))

    print('Post-softmax, %d masks, Test Acc : %.3f' % (N, 100.*acc_test))
    print('--------------------------------------------------------------')
    return 100.*acc_test

evaluate3(10)
#%%
pred1 = np.ones(10)*evaluate1()
pred2 = []
pred3 = []

for N in range(10,110,10) :
    pred2.append(evaluate2(N))
    pred3.append(evaluate3(N))

#%%
plt.figure()
plt.plot(range(10,110,10),pred1,label='without mask')
plt.plot(range(10,110,10),pred2,label='pre-softmax')
plt.plot(range(10,110,10),pred3,label='post-softmax')
plt.legend()
plt.savefig('probalem1b.pdf')
plt.show()