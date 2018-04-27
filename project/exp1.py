#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:00:00 2018

@author: fanxiao
"""
#%%
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

USE_CUDA=torch.cuda.is_available()

#%%

class MLP(nn.Module) :
    def __init__(self, activation='relu'):
        super(MLP, self).__init__()
        
        self.linear1 = nn.Linear(2,4) #input dimension:2
        self.linear2 = nn.Linear(4,2)
        self.linear3 = nn.Linear(2,2)
        if activation == 'relu':
            self.active = nn.ReLU() 
        else :
            self.active = nn.ELU()
    
    def forward(self,input):
        x = self.active(self.linear1(input))
        x = self.active(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def init_weights_glorot(self):
        for m in self._modules :
            if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight)
                
def adjust_lr(optimizer, lr0, epoch, total_epochs):
    lr = lr0 * (0.1 ** (epoch / float(total_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#adjust learning rate when maximizing z_hat : alpha_t = 1/np.sqrt(t)
def adjust_lr_zt(optimizer, lr0, epoch):
    lr = lr0 * (1.0 / np.sqrt(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def evaluate (model, valid_data) :
    COUNTER = 0
    ACCURACY = 0
    for x_, y_ in valid_data :
        if USE_CUDA:
            x_, y_ = x_.cuda(), y_.cuda()
        x_, y_ = Variable(x_), Variable(y_)
        out = model(x_)
        _, predicted = torch.max(out, 1)
        COUNTER += y_.size(0)
        ACCURACY += float(torch.eq(predicted,y_).sum().cpu().data.numpy())
    return ACCURACY / float(COUNTER) *100.0

# evaluate on adversarial examples
def evaluate_adversarial (model, loss_function, valid_data, epsilon=0.5) :
    COUNTER = 0
    ACCURACY = 0
    for x_, y_ in valid_data :
        if USE_CUDA:
            x_, y_ = x_.cuda(), y_.cuda()
        x_, y_ = Variable(x_,requires_grad=True), Variable(y_)
        loss_true = loss_function(model(x_),y_)
        loss_true.backward()
        x_grad = x_.grad
        x_adversarial = x_.clone()
        if USE_CUDA:
            x_adversarial = x_adversarial.cuda()
        x_adversarial.data = x_.data + epsilon * torch.sign(x_grad.data) * x_grad.data     
        
        x_.grad.data.zero_()
        out = model(x_adversarial)
        _, predicted = torch.max(out, 1)
        COUNTER += y_.size(0)
        ACCURACY += float(torch.eq(predicted,y_).sum().cpu().data.numpy())
    return ACCURACY / float(COUNTER) *100.0

#basic training minimizing ERM          
def train(model,optimizer,loss_function, train_loader,valid_loader,num_epoch,min_lr0=0.001,min_lr_adjust=False) :
    for ep in range(num_epoch) :
        losses = []
        for x_, y_ in train_loader :
            if USE_CUDA:
                x_, y_ = x_.cuda(), y_.cuda()
            x_, y_ = Variable(x_), Variable(y_)
            optimizer.zero_grad()
            loss = loss_function(model(x_),y_)
            losses.append(loss.data[0])
            loss.backward()
            optimizer.step()

        # print ('%d epoch, %.3f loss, %.2f%% accuracy.'%(ep, torch.mean(torch.FloatTensor(losses))
        #     ,evaluate(model,valid_loader)))
        print ('%d epoch, %.3f loss, %.2f%% accuracy, %.2f%% accuracy adversarial.'%(ep, torch.mean(torch.FloatTensor(losses))
            ,evaluate(model,valid_loader), evaluate_adversarial(model,loss_function,valid_loader)))
        
        if min_lr_adjust == True:
            adjust_lr(optimizer,min_lr0,ep+1,num_epoch)
            
# one-step adversarial training using fast gradient L2
def train_FGM(model,optimizer,loss_function, train_loader, valid_loader, num_epoch, epsilon,min_lr0=0.001,min_lr_adjust=False) :
    for ep in range(num_epoch) :
        losses = []
        half = torch.FloatTensor([0.5])
        for x_, y_ in train_loader :
            if USE_CUDA:
                x_, y_ = x_.cuda(), y_.cuda()
            #J = 0.5J(theta,x,y) + 0.5 J(theta,x_adversarial,y)
            x_, y_ = Variable(x_,requires_grad=True), Variable(y_)
            optimizer.zero_grad()
            loss_true = loss_function(model(x_),y_)
            loss_true.backward(half)
            x_grad = x_.grad
            x_adversarial = x_.clone()
            if USE_CUDA:
                x_adversarial = x_adversarial.cuda()
            
            #L_infinity x_adv = x+epsilon*sign(grad_x)
            # x_adversarial.data = x_.data + epsilon * torch.sign(x_grad.data) 
            
            #L2 x_adv = x + epsilon * (grad_x/||grad_x||)
            x_adversarial.data = x_.data + epsilon * x_grad.data/torch.norm(x_grad.data)
            
            x_.grad.data.zero_()
            loss_adversarial = loss_function(model(x_adversarial),y_)
            
            loss_adversarial.backward(half)
            optimizer.step()

            losses.append((loss_true.data[0]+loss_adversarial.data[0])/2.0)
        print ('%d epoch, %.3f loss, %.2f%% accuracy, %.2f%% accuracy adversarial.'%(ep, torch.mean(torch.FloatTensor(losses))
            ,evaluate(model,valid_loader), evaluate_adversarial(model,loss_function,valid_loader)))
        
        if min_lr_adjust == True:
            adjust_lr(optimizer,min_lr0,ep+1,num_epoch)
            
def train_WRM(model,optimizer,loss_function, train_loader,valid_loader, num_epoch,gamma=2,max_lr0=0.0001,min_lr0=0.001,min_lr_adjust=False,savepath=None) :
    T_adv = 15
    start_time = time.time()
    train_hist={}
#    half = torch.FloatTensor([0.5])
    for ep in range(1,num_epoch+1) :
        epoch_start_time = time.time()
        losses = []
        for x_, y_ in train_loader :
            if USE_CUDA:
                x_, y_ = x_.cuda(), y_.cuda()
            x_, y_ = Variable(x_), Variable(y_)
            
#            loss_true = loss_function(model(x_),y_)
#            loss_true.backward(half)
            
            #initialize z_hat with x_
            z_hat = x_.data.clone()
            if USE_CUDA:
                z_hat = z_hat.cuda()
            z_hat = Variable(z_hat,requires_grad=True)
            
            #running the maximizer for z_hat
#            params = list(model.parameters()) + [z_hat]
            optimizer_zt = torch.optim.Adam([z_hat], lr=max_lr0)
            for n in range(T_adv) :
                optimizer_zt.zero_grad()
                # loss_zt = - ( loss_function(model(z_hat),y_)- gamma*(torch.norm(z_hat-x_)**2) )
                loss_zt = - ( loss_function(model(z_hat),y_)- torch.mean(gamma*(torch.norm(z_hat-x_,2,1)**2)) )
                loss_zt.backward()
                optimizer_zt.step()
                # adjust_lr_zt(optimizer_zt,max_lr0, n+1)
                
            # running the loss minimizer, using z_hat   
            optimizer.zero_grad()
            loss_adversarial = loss_function(model(z_hat),y_)
            
            loss_adversarial.backward()
            losses.append(loss_adversarial.data[0])
            
            optimizer.step()

        if min_lr_adjust == True:
            adjust_lr(optimizer,min_lr0,ep,num_epoch) 

        # display    
        mean_loss=torch.mean(torch.FloatTensor(losses))
        acc=evaluate(model,valid_loader)
        acc_adv=evaluate_adversarial(model,loss_function,valid_loader)
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print ('%d epoch, %.3f loss, %.2f%% accuracy, %.2f%% accuracy adversarial, %.2fs ptime.' \
            %(ep, mean_loss, acc, acc_adv, per_epoch_ptime))
        
        if savepath is not None and (ep % 3==0):
            end_time = time.time()
            total_ptime = end_time - start_time
            # train_hist['total_ptime'].append(total_ptime)
            saveCheckpoint(model,train_hist,savepath+'_ep'+str(ep))
        
         
def saveCheckpoint(model,train_hist, filename='Model') :
    # print('Saving..')
    state = {
        'model':  model.cpu().state_dict() if USE_CUDA else model.state_dict(),
        'train_hist' : train_hist
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+filename)
    if USE_CUDA:
        model.cuda()

def loadCheckpoint(model,filename='Model',path='./checkpoint/'):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(path+filename)
    model_params = checkpoint['model']
    model.load_state_dict(model_params)
    if USE_CUDA :
        model.cuda()
    train_hist = checkpoint['train_hist']

    return model,train_hist   

def synthetic_data(N_example) : 
    data_x = np.zeros((N_example,2))
    data_y = np.ones(N_example) 
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
    # print('num of class0:',len(data_y[data_y==0]))
    return data_x, data_y

#%%

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
    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)

#%%
init_seed()
train_x, train_y = synthetic_data(10000)
valid_x, valid_y = synthetic_data(4000)

#%%
if __name__=='__main__':
    
    use_cuda = torch.cuda.is_available()

    LR0 = 0.01
    batch_size = 128
    loss_function = nn.CrossEntropyLoss()

    train_data = torch.utils.data.TensorDataset(
            torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long())
    valid_data = torch.utils.data.TensorDataset(
            torch.from_numpy(valid_x).float(), torch.from_numpy(valid_y).long())
    train_data_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_data_loader = torch.utils.data.DataLoader(
                    valid_data, batch_size=batch_size, shuffle=True, num_workers=2)


#%%
if __name__=='__main__':
    #%%
    LR0 = 0.01
    net_ERM = MLP()
    net_ERM.init_weights_glorot()
    optimizer = torch.optim.Adam(net_ERM.parameters(), lr=LR0)
    # train(net_ERM,optimizer,loss_function, train_data_loader,valid_data_loader,30,min_lr0=LR0,min_lr_adjust=False)

    #%%
    LR0 = 0.01
    net_FGM = MLP()
    net_FGM.init_weights_glorot()

    optimizer = torch.optim.Adam(net_FGM.parameters(), lr=LR0)
    # train_FGM(net_FGM,optimizer,loss_function, train_data_loader,valid_data_loader, 30, epsilon=0.3,min_lr0=LR0,min_lr_adjust=False)

    #%%
    LR0 = 0.01
    MAX_LR0=0.08
    net_WRM = MLP(activation='relu')
    # net_WRM = MLP()
    net_WRM.init_weights_glorot()

    optimizer = torch.optim.Adam(net_WRM.parameters(), lr=LR0)
    train_WRM(net_WRM,optimizer,loss_function, train_data_loader,valid_data_loader, 30 , max_lr0=MAX_LR0, min_lr0=LR0, min_lr_adjust=False)

#%%
if __name__=='__main__':
    plotGraph([net_ERM,net_FGM,net_WRM],train_x, train_y)
