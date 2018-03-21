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


os.chdir("/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/assignment3/")
print(os.getcwd())

#%%

SEQ_MAX_LEN=20
# BATCH_SIZE_TRAIN=25
# BATCH_SIZE_VALID=50
LR_0=0.1 #4e-4
MOMENTUM=0.9
WEIGHT_DECAY=5e-4


RANDOM_SEED = 2333
REPORT_INTERVAL = 200
CHECKPOINT_INTERVAL = 100 #1000
CHECKPOINT_PATH='../checkpoint/'

controller_size = 100 #attrib(default=100, convert=int)
controller_layers = 1 #attrib(default=1,convert=int)
num_heads = 1 #attrib(default=1, convert=int)
sequence_width = 8 #attrib(default=8, convert=int)
sequence_min_len = 1 #attrib(default=1,convert=int)
sequence_max_len = 20 #attrib(default=20, convert=int)
memory_n = 128 #attrib(default=128, convert=int)
memory_m = 20 #attrib(default=20, convert=int)
#total of the batch 
total_batches = 1000 #attrib(default=50000, convert=int)
#in each batch, there are batch_size sequences together as same length of sequence.
batch_size = 200 #attrib(default=1, convert=int)
rmsprop_lr = 1e-4 #attrib(default=1e-4, convert=float)
rmsprop_momentum = 0.9 #attrib(default=0.9, convert=float)
rmsprop_alpha = 0.95 #attrib(default=0.95, convert=float)


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


sample_binary=distributions.Bernoulli(torch.Tensor([0.5]))

cuda_available = torch.cuda.is_available()


'''
# Define the Model LSTM.
'''
class Model_LSTM(nn.Module):
    def __init__(self):
        super(Model_LSTM, self).__init__()

        self.copy_machine=nn.LSTM(9, 100)
        self.mlp=nn.Linear(100,9)

    def init_hidden(self,batch_size):
        self.hidden= ( autograd.Variable(torch.randn(1, batch_size, 100)),\
          autograd.Variable(torch.randn((1, batch_size, 100))) )
        if cuda_available:
            self.hidden=self.hidden.cuda()

    def forward(self, sequence):
        # out, (self.hidden, self.cell) = self.copy_machine(inputs, None)
        out, self.hidden = self.copy_machine(sequence, self.hidden)
        out=self.mlp(out)
        return out


def gen1seq():
    length=np.random.randint(2,SEQ_MAX_LEN+1)
    # length=SEQ_SIZE+1
    seq=sample_binary.sample_n(9*length).view(length, 1, -1)
    seq[:,:,-1]=0.0
    seq[-1]=0.0
    seq[-1,-1,-1]=1.0
    return seq

def gen1seq_act(length):
    seq=torch.zeros(9*length).view(length, 1, -1)+0.5
    seq[:,:,-1]=0.0
    seq[-1]=0.0
    seq[-1,-1,-1]=1.0
    return seq

#total_batches: total of the batch 
#batch_size: in each batch, there are batch_size sequences together as same length of sequence.
def dataloader(total_batches,
               batch_size,
               seq_width,
               min_len,
               max_len):
    for batch_num in range(total_batches):

        # All batches have the same sequence length
        seq_len = random.randint(min_len, max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = Variable(torch.from_numpy(seq))

        # The input includes an additional channel used for the delimiter
        inp = Variable(torch.zeros(seq_len + 1, batch_size, seq_width + 1))
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0 # delimiter in our control channel
        outp = seq.clone()

        seq2 = Variable(torch.zeros(seq_len, batch_size, seq_width)+0.5)
        act_inp = Variable(torch.zeros(seq_len + 1, batch_size, seq_width + 1))
        act_inp[:seq_len, :, :seq_width] = seq2
        act_inp[seq_len, :, seq_width] = 1.0 

        yield batch_num+1, inp.float(), outp.float(), act_inp.float()

def train_model(model, seqs_loader, display=200):
    
    # in_seqs=[ gen1seq() for i in range(SEQS_TOTAL)]
    
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
    for i,(batch_num, x, y, act) in enumerate(seqs_loader):
        # start = time.time()
        if cuda_available:
            x, y, act = x.cuda(), y.cuda(), act.cuda()
        
        optimizer.zero_grad()
        model.hidden = model.init_hidden(batch_size)
        # print(x.size(),y.size(),act.size())
        #input data
        model.forward(x)
        # do copy in useing the act_seq
        # model.hidden = model.init_hidden(batch_size)
        out_seq=model.forward(act)

        out_seq=out_seq[:-1,:,:-1]
        sigmoid_out=torch.sigmoid(out_seq)
        loss = criterion(sigmoid_out, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])
        out_binarized = sigmoid_out.clone().data.numpy()
        out_binarized=np.where(out_binarized>0.5,1,0)
        # The cost is the number of error bits per sequence
        cost = np.sum(np.abs(out_binarized - y.data.numpy()))/(y.size(0)*y.size(1)*y.size(2))
        # cost = np.sum(np.abs(out_binarized - y.data.numpy()))

        costs.append(cost)

        # end = time.time()

        # print the process of the training.  
        if display!=0 and (i % display==0 or i==(total_batches-1)):
            print("Epoch:%d, Train Loss: %f" % (i,loss))
            # print(targ_seq,out_seq)
            # print("Epoch:%d, Train Loss %f, Spend %.3f minutes " % (i,loss,(end-start)/60.0))

    return losses,costs



'''
Train Model_LSTM
'''

train_loader=dataloader(total_batches, batch_size,
                    sequence_width,
                    2, SEQ_MAX_LEN)

model = Model_LSTM()
loss_list,cost_list=train_model(model,train_loader)

#plot accuracy as a function of epoch
plt.figure()
plt.plot(range(0,total_batches),loss_list,label='LSTM')
# plt.plot(range(1,1000),valid_acc_list,label='Validation')
plt.xlabel('Sequence number')
plt.ylabel('Loss per sequence')
plt.legend()
# plt.savefig('lstm1.pdf')
plt.show()

plt.figure()
plt.plot(range(0,total_batches),cost_list,label='LSTM')
# plt.plot(range(1,1000),valid_acc_list,label='Validation')
plt.xlabel('Sequence number')
plt.ylabel('Cost per sequence')
plt.legend()
# plt.savefig('lstm1.pdf')
plt.show()

#%%
test1=Variable(gen1seq())
print(test1)
model.init_hidden(1)
model.forward(test1)
actx=Variable(gen1seq_act(test1.size(0)))
# model.init_hidden(1)
print(torch.sigmoid(model.forward(actx)))
# actx=Variable(gen1seq_act(test1.size(0)))
# model.init_hidden(1)
print(torch.sigmoid(model.forward(actx)))

