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


RANDOM_SEED = 2333
REPORT_INTERVAL = 200
CHECKPOINT_INTERVAL = 100 #1000
CHECKPOINT_PATH='../checkpoint/'

controller_size = 100 #attrib(default=100, convert=int)
controller_layers = 1 #attrib(default=1,convert=int)
num_heads = 1 #attrib(default=1, convert=int)
BYTE_WIDTH = 8 #attrib(default=8, convert=int)
HIDDEN_NUM=100
SEQUENCE_MIN_LEN = 1 #attrib(default=1,convert=int)
SEQUENCE_MAX_LEN = 20 #attrib(default=20, convert=int)
memory_n = 128 #attrib(default=128, convert=int)
memory_m = 20 #attrib(default=20, convert=int)
#total of the batch 
TOTAL_BATCHES = 300 #attrib(default=50000, convert=int)
#in each batch, there are batch_size sequences together as same length of sequence.
BATCH_SIZE = 200 #attrib(default=1, convert=int)
rmsprop_lr = 1e-4 #attrib(default=1e-4, convert=float)
rmsprop_momentum = 0.9 #attrib(default=0.9, convert=float)
rmsprop_alpha = 0.95 #attrib(default=0.95, convert=float)
# BATCH_SIZE_TRAIN=25
# BATCH_SIZE_VALID=50
LR_0=0.1 #4e-4
MOMENTUM=0.9
WEIGHT_DECAY=5e-4

loss_function = nn.BCELoss()

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


sample_binary=distributions.Bernoulli(torch.Tensor([0.5]))

cuda_available = torch.cuda.is_available()


'''
# Define the Model LSTM.
'''
class LSTMcopy(nn.Module):
    def __init__(self):
        super(LSTMcopy, self).__init__()

        self.lstm=nn.LSTM(BYTE_WIDTH+1, HIDDEN_NUM)
        self.mlp=nn.Linear(HIDDEN_NUM,BYTE_WIDTH)

    def init_hidden(self,batch_size):
        self.hidden= ( autograd.Variable(torch.randn(1, batch_size, HIDDEN_NUM)),\
          autograd.Variable(torch.randn((1, batch_size, HIDDEN_NUM))) )
        if cuda_available:
            self.hidden=self.hidden.cuda()

    def forward(self, sequence):
        # out, (self.hidden, self.cell) = self.copy_machine(inputs, None)
        out, self.hidden = self.lstm(sequence, self.hidden)
        out=self.mlp(out)
        return out

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params


def gen1seq():
    length=np.random.randint(2,SEQUENCE_MAX_LEN+1)
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
        act_inp = Variable(torch.zeros(seq_len, batch_size, seq_width + 1))
        act_inp[:seq_len, :, :seq_width] = seq2

        yield batch_num+1, inp.float(), outp.float(), act_inp.float()

def train_model(model, seqs_loader, display=100):
    
    # in_seqs=[ gen1seq() for i in range(SEQS_TOTAL)]
    
    # print(inputs[0])
    if cuda_available:
        model = model.cuda()

    criterion = loss_function  #nn.CrossEntropyLoss() nn.MSELoss() nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=LR_0)
    # optimizer = optim.SGD(model.parameters(), lr=LR_0, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # optimizer = optim.Adam(model.parameters(), lr=LR_0)
    optimizer = optim.RMSprop(model.parameters(), lr=3e-5, momentum = MOMENTUM)
    
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.1)

    #training
    print('Training Begining')
    list_losses =[]
    list_costs =[]
    list_bits=[]
    list_batch_num = []
    for i,(batch_num, X, Y, act) in enumerate(seqs_loader):
        # start = time.time()
        if cuda_available:
            X, Y, act = X.cuda(), Y.cuda(), act.cuda()
        
        model.init_hidden(BATCH_SIZE)
        optimizer.zero_grad()
        # print(x.size(),y.size(),act.size())
        #input data
        model.forward(X)
        # do copy in useing the act_seq
        out_seq=model.forward(act)

        sigmoid_out=F.sigmoid(out_seq)
        loss = criterion(sigmoid_out, Y)
        loss.backward()
        optimizer.step()

        list_losses.append(loss.data[0])

        out_binarized = sigmoid_out.clone().data.numpy()
        out_binarized=np.where(out_binarized>0.5,1,0)
        # The cost is the number of error bits per sequence
        cost = np.sum(np.abs(out_binarized - Y.data.numpy()))
        list_costs.append(cost/BATCH_SIZE) #per sequence
        list_bits.append(Y.size(0)*Y.size(1)*Y.size(2))
        list_batch_num.append(i)
        # end = time.time()

        # print the process of the training.  
        if display!=0 and (i % display==0 or i==(TOTAL_BATCHES-1)):
            print("Batch %d, Train Loss %f, Train Cost %f" % (i,list_losses[-1],list_costs[-1]))
            # print(targ_seq,out_seq)

    return list_losses,list_costs,list_batch_num,list_bits

def evaluate(model, test_data_loader, criterion) : 
    costs = 0
    losses = 0
    lengthes = 0
    optimizer = optim.RMSprop(model.parameters(), lr=3e-5, momentum = MOMENTUM)

    for i,(batch_num, X, Y, act) in enumerate(test_data_loader):
        # start = time.time()
        if cuda_available:
            X, Y, act = X.cuda(), Y.cuda(), act.cuda()
        
        model.init_hidden(BATCH_SIZE)
        optimizer.zero_grad()
        # print(x.size(),y.size(),act.size())
        #input data
        model.forward(X)
        # do copy in useing the act_seq
        out_seq=model.forward(act)

        sigmoid_out=F.sigmoid(out_seq)
        loss = criterion(sigmoid_out, Y)
        loss.backward()
        optimizer.step()

        length = Y.size(0) * BATCH_SIZE
        lengthes += length
    
        losses += loss
        
        out_binarized = sigmoid_out.clone().data.numpy()
        out_binarized=np.where(out_binarized>0.5,1,0)
        # The cost is the number of error bits per sequence
        cost = np.sum(np.abs(out_binarized - Y.data.numpy()))

        costs += cost
        
    print ("T = %d, Average loss %f, average cost %f" % (Y.size(0), losses.data/lengthes, costs/lengthes))
    return losses.data/lengthes, costs/lengthes

def saveCheckpoint(model,list_batch_num,list_loss, list_cost, path='lstm') :
    print('Saving..')
    state = {
        'model': model,
        'list_batch_num': list_batch_num,
        'list_loss' : list_loss,
        'list_cost' : list_cost
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+path)

def loadCheckpoint(path='lstm'):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+path)
    model = checkpoint['model']
    list_batch_num = checkpoint['list_batch_num']
    list_loss = checkpoint['list_loss']
    list_cost = checkpoint['list_cost']
    return model, list_batch_num, list_loss, list_cost
#%%

'''
Train LSTMcopy
'''

train_loader=dataloader(TOTAL_BATCHES, BATCH_SIZE,
                    BYTE_WIDTH,2, SEQUENCE_MAX_LEN)

model = LSTMcopy()
list_loss,list_cost,list_batch_num,list_bits=train_model(model,train_loader)

saveCheckpoint(model,list_batch_num,list_loss, list_cost, path='lstm1') 

#%%
#plot accuracy as a function of epoch
plt.figure()
# plt.plot(range(0,total_batches),loss_list,label='LSTM')
plt.plot(range(0,TOTAL_BATCHES),list_bits,label='Bits Num')
# plt.plot(range(1,1000),valid_acc_list,label='Validation')
plt.xlabel('Sequence number')
plt.ylabel('Loss per sequence')
plt.legend()
# plt.savefig('lstm1.pdf')
plt.show()

plt.figure()
plt.plot(range(0,TOTAL_BATCHES),list_cost,label='LSTM')
# plt.plot(range(1,1000),valid_acc_list,label='Validation')
plt.xlabel('Sequence number')
plt.ylabel('Cost per sequence')
plt.legend()
# plt.savefig('lstm1.pdf')
plt.show()

#%%

model, list_seq_num, list_loss, list_cost = loadCheckpoint(path='lstm1')
#%%
list_avg_loss = []
list_avg_cost = []
for T in range(10,110,10) : 
    test_data_loader = dataloader(TOTAL_BATCHES, BATCH_SIZE,
                    BYTE_WIDTH,min_len=T,max_len=T)
    avg_loss, avg_cost = evaluate(model,test_data_loader,loss_function)
    list_avg_loss.append(avg_loss)
    list_avg_cost.append(avg_cost)

#%%
    
plt.plot(range(10,110,10),list_avg_cost)
plt.xlabel('T')
plt.ylabel('average cost')

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

