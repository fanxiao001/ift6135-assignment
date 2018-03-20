#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import random
import time
import numpy as np

import os
import os.path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.autograd as autograd
import torch.distributions as distributions

import matplotlib.pyplot as plt


# os.chdir("/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/assignment3/")
print(os.getcwd())

#%%

SEQS_TOTAL = 1000000
SEQ_SIZE=20
# BATCH_SIZE_TRAIN=25
# BATCH_SIZE_VALID=50
LR_0=0.1 #4e-4
MOMENTUM=0.9
WEIGHT_DECAY=5e-4


torch.manual_seed(1)

sample_binary=distributions.Bernoulli(torch.Tensor([0.5]))

cuda_available = torch.cuda.is_available()


'''
# Define the Model LSTM.
'''
class Model_LSTM(nn.Module):
    def __init__(self):
        super(Model_LSTM, self).__init__()

        self.copy_machine=nn.LSTM(9, 100)
        #hidden=hidden_t,cell_t
        self.hidden = self.init_hidden()
        self.mlp=nn.Linear(100,9)

    def init_hidden(self):
        return ( autograd.Variable(torch.randn(1, 1, 100)),\
          autograd.Variable(torch.randn((1, 1, 100))) )

    def forward(self, sequence):
        # out, (self.hidden, self.cell) = self.copy_machine(inputs, None)
        out, self.hidden = self.copy_machine(sequence, self.hidden)
        out=self.mlp(out)
        return out


def gen1seq():
    length=np.random.randint(2,SEQ_SIZE+1)
    # length=SEQ_SIZE+1
    seq=sample_binary.sample_n(9*length).view(length, 1, -1)
    seq[:,:,-1]=0
    seq[-1]=0
    seq[-1,-1,-1]=1
    return seq

def gen1seq_zero(length):
    seq=torch.zeros(9*length).view(length, 1, -1)
    seq[:,:,-1]=0
    seq[-1]=0
    seq[-1,-1,-1]=1
    return seq
    
def train_model(model, display=200):
    
    in_seqs=[ gen1seq() for i in range(SEQS_TOTAL)]

    # print(inputs[0])
    if cuda_available:
        model = model.cuda()

    criterion = nn.BCELoss() #nn.CrossEntropyLoss() nn.MSELoss() nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=LR_0)
    # optimizer = optim.SGD(model.parameters(), lr=LR_0, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=LR_0)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.1)

    #training
    print('Training Begining')
    losses=[]
    costs=[]
    # trainloss_every = 200
    for i,seq in enumerate(in_seqs):
        # start = time.time()
        if cuda_available:
            in_seq, targ_seq = seq.cuda(), seq.cuda()
        in_seq, targ_seq = Variable(seq), Variable(seq)
        
        optimizer.zero_grad()
        model.hidden = model.init_hidden()

        # get immiediately prediction 
        # out_seq = model.forward(in_seq)
        # after input all sequence, generate new sequence.
        model.forward(in_seq)
        # use the <start> of the sequence for first vector of prediction
        # 由于序列各dimension是p=0.5的概率独立产生的，所以上一个dimension不能预测下一个dimension
        in_seq_zero=Variable(gen1seq_zero(in_seq.size(0)))
        out_seq=model.forward(in_seq_zero)

        sigmoid_out=torch.sigmoid(out_seq.view(-1))
        loss = criterion(sigmoid_out, targ_seq.view(-1))
        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])
        out_binarized = sigmoid_out.clone().data
        out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)
        # The cost is the number of error bits per sequence
        cost = torch.sum(torch.abs(out_binarized - targ_seq.view(-1).data))/(in_seq.size(0)*9)
        costs.append(cost)

        # end = time.time()

        # print the process of the training.  
        if display!=0 and (i % display==0 or i==(SEQS_TOTAL-1)):
            print("Epoch:%d, Train Loss: %f" % (i,loss))
            # print(targ_seq,out_seq)
            # print("Epoch:%d, Train Loss %f, Spend %.3f minutes " % (i,loss,(end-start)/60.0))

    return losses,costs



'''
Train Model_LSTM
'''
model = Model_LSTM()
loss_list,cost_list=train_model(model)

#plot accuracy as a function of epoch
plt.figure()
plt.plot(range(0,SEQS_TOTAL),loss_list,label='LSTM')
# plt.plot(range(1,1000),valid_acc_list,label='Validation')
plt.xlabel('Sequence number')
plt.ylabel('Loss per sequence')
plt.legend()
# plt.savefig('lstm1.pdf')
plt.show()

plt.figure()
plt.plot(range(0,SEQS_TOTAL),cost_list,label='LSTM')
# plt.plot(range(1,1000),valid_acc_list,label='Validation')
plt.xlabel('Sequence number')
plt.ylabel('Cost per sequence')
plt.legend()
# plt.savefig('lstm1.pdf')
plt.show()

#%%
test1=Variable(gen1seq())
print(test1)
print(torch.sigmoid(model.forward(test1)))

