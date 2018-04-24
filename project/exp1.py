#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:00:00 2018

@author: fanxiao
"""

import os
import time
import matplotlib.pyplot as plt
from scipy.misc import imresize
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random
#%%

class MLP(nn.Module) :
    def __init__(self):
        super(MLP, self).__init__()
        
        self.linear1 = nn.Linear(2,4) #input dimension:2
        self.linear2 = nn.Linear(4,2)
    
    def forward(self,input):
        x = F.relu(self.linear1(input))
        x = self.linear2(x)
        return x
    
    def init_weights_glorot(self):
        for m in self._modules :
            if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight)
                
def adjust_lr(optimizer, epoch, total_epochs):
    lr = LR0 * (1.0/ np.sqrt(total_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
            
def train(model,optimizer,loss_function, train_loader,num_epoch,lr_adjust=False) :
    for ep in range(num_epoch) :
        losses = []
        for x_, y_ in train_loader :
            x_, y_ = Variable(x_), Variable(y_)
            optimizer.zero_grad()
            loss = loss_function(model(x_),y_)
            losses.append(loss.data[0])
            loss.backward()
            optimizer.step()
        print ('Epoch %d, loss %f'%(ep, torch.mean(torch.FloatTensor(losses))))
        
        if lr_adjust == True:
            adjust_lr(optimizer,ep+1,num_epoch) 
            
def train_FGM(model,optimizer,loss_function, train_loader,num_epoch, epsilon, lr_adjust=False) :
    for ep in range(num_epoch) :
        losses = []
        half = torch.FloatTensor([0.5])
        for x_, y_ in train_loader :
            x_, y_ = Variable(x_,requires_grad=True), Variable(y_)
            optimizer.zero_grad()
            loss_true = loss_function(model(x_),y_)
            loss_true.backward(half)
            x_grad = x_.grad
            x_adversarial = x_.clone()
            x_adversarial.data = x_.data - epsilon * torch.sign(x_grad.data) * x_grad.data     
            
            x_.grad.data.zero_()
            loss_adversarial = loss_function(model(x_adversarial),y_)
            
#            f2 = model(x_adversarial)
#            _, predicted = torch.max(f2, 1)
#            if((predicted.data == y_.data).sum()!=batch_size) :
#                print ("attacked")
            
            loss_adversarial.backward(half)
            losses.append((loss_true.data[0]+loss_adversarial.data[0])/2.0)
#            losses.append(loss_sum.data[0])
            optimizer.step()
        print ('Epoch %d, loss %f'%(ep, torch.mean(torch.FloatTensor(losses))))
        
        if lr_adjust == True:
            adjust_lr(optimizer,ep+1,num_epoch)
            
def train_WRM(model,optimizer,loss_function, train_loader,num_epoch,gamma,lr_adjust=False) :
    for ep in range(num_epoch) :
        losses = []
        for x_, y_ in train_loader :
            x_, y_ = Variable(x_), Variable(y_)
            optimizer.zero_grad()
            loss = loss_function(model(x_),y_)
            losses.append(loss.data[0])
            loss.backward()
            optimizer.step()
        print ('Epoch %d, loss %f'%(ep, torch.mean(torch.FloatTensor(losses))))
        
        if lr_adjust == True:
            adjust_lr(optimizer,ep+1,num_epoch) 

def synthetic_data(N_example) : 
    data_x = np.zeros((N_example,2))
    data_y = np.zeros(N_example) 
    length = 0 
    while(length<N_example) :
        x = np.random.randn(100,2)
        l2 = np.linalg.norm(x, axis=1)
        x = x[np.any((l2>=1.3*np.sqrt(2),l2<=np.sqrt(2)/1.3), axis=0), :]
        y = [1 if (np.linalg.norm(i) - np.sqrt(2)) > 0 else 0 for i in x]  
        if length+len(x) <= N_example :
            data_x[length:length+len(x),:] = x
            data_y[length:length+len(x)] = y
        else :
            data_x[length:,:] = x[N_example-length,:]
            data_y[length:] = y[N_example-length]
        length += len(x)
    return data_x, data_y

def plotGraph(models,data_x, data_y) :
    
    plt.figure(figsize=(5,5))
    Colors = ['blue','orange','red','purple','green']
    labels = ['ERM','FGM','WRM']
    
    plt.scatter(data_x[data_y==0,0],data_x[data_y==0,1], c=Colors[0], marker='.')
    plt.scatter(data_x[data_y==1,0],data_x[data_y==1,1], facecolors='none', edgecolors=Colors[1])
    xmax = max(data_x[:,0])
    xmin = min(data_x[:,0])
    ymax = max(data_x[:,1])
    ymin = min(data_x[:,1])
     
    x1 = np.linspace(xmin, xmax)
    x2 = np.linspace(ymin,ymax)
    X1,X2 = np.meshgrid(x1,x2)
    features_X = np.vstack((X1.flatten(),X2.flatten())).T
    
    softmax = nn.Softmax()
    
    for i, model in enumerate(models):
        Z = softmax(model(Variable(torch.FloatTensor(features_X))))[:,0].data.numpy().reshape(len(x1),len(x2))
        CS = plt.contour(X1,X2,Z,[0.5],colors=Colors[i+2])
        CS.collections[0].set_label(labels[i])
        
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.legend()

def init_seed(seed=123):
    '''set seed of random number generators'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
#%%
data_x, data_y = synthetic_data(10000)


#%%
init_seed()
LR0 = 0.1
batch_size = 128
loss_function = nn.CrossEntropyLoss()

train_data = torch.utils.data.TensorDataset(
        torch.from_numpy(data_x).float(), torch.from_numpy(data_y).long())
train_data_loader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size, shuffle=True, num_workers=2)

net_ERM = MLP()
net_ERM.init_weights_glorot()

optimizer = torch.optim.SGD(net_ERM.parameters(), lr=LR0, momentum = 0.9)

train(net_ERM,optimizer,loss_function, train_data_loader,15)

#%%

net_FGM = MLP()
net_FGM.init_weights_glorot()

optimizer = torch.optim.SGD(net_FGM.parameters(), lr=LR0, momentum = 0.9)

train_FGM(net_FGM,optimizer,loss_function, train_data_loader,15, epsilon=0.1)

#%%

plotGraph([net_ERM,net_FGM],data_x, data_y)