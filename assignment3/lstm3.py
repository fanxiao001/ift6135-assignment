#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 18:50:22 2018

@author: fanxiao
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import os

#%%
    
class LSTMCopy(nn.Module):
    def __init__(self, input_size = 9, hidden_size = 256, output_size = 8, num_layer = 3, batch_size=1):
        super(LSTMCopy, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.batch_size = batch_size
#        self.embedding = nn.Linear(input_size,hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layer)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
#        output = self.embedding(input)
        output, hidden = self.lstm(input, hidden)
        output = F.sigmoid(self.out(output))
        return output, hidden

    def initHidden(self):
        result = (Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_size)),
                Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_size)))
        if torch.cuda.is_available():
            return result.cuda()
        else:
            return result
    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params

#%%
model = LSTMCopy(9,256,8,1,1)
model.calculate_num_params()

#%%
def dataloader(num_batches,
               batch_size,
               seq_width,
               min_len,
               max_len):
    """Generator of random sequences for the copy task.

    Creates random batches of "bits" sequences.

    All the sequences within each batch have the same length.
    The length is [`min_len`, `max_len`]

    :param num_batches: Total number of batches to generate.
    :param seq_width: The width of each item in the sequence.
    :param batch_size: Batch size.
    :param min_len: Sequence minimum length.
    :param max_len: Sequence maximum length.

    NOTE: The input width is `seq_width + 1`, the additional input
    contain the delimiter.
    """
    for batch_num in range(num_batches):

        # All batches have the same sequence length
        seq_len = random.randint(min_len, max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = Variable(torch.from_numpy(seq))

        # The input includes an additional channel used for the delimiter
        inp = Variable(torch.zeros(seq_len + 1, batch_size, seq_width + 1))
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0 # delimiter in our control channel
        outp = seq.clone()
        yield batch_num+1, inp.float(), outp.float()
        
def init_seed(seed=123):
    '''set seed of random number generators'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
#%%

def train(model, train_data_loader, criterion, optimizer,  interval = 1000, display = True) : 
    list_seq_num = []
    list_loss = []
    list_cost = []
    lengthes = 0
    losses = 0
    costs = 0
    for batch_num, X, Y  in train_data_loader:
        hidden = model.initHidden()

        inp_seq_len, _, _ = X.size()
        outp_seq_len, batch_size, output_size = Y.size()
        optimizer.zero_grad()
        
        # Feed the sequence + delimiter
        for i in range(inp_seq_len):
            _, hidden = model(X[i].view(1,batch_size,-1),hidden)
    
        y_out = Variable(torch.zeros(Y.size()))
        
        for i in range(outp_seq_len):
            y_out[i], hidden = model(Variable(torch.zeros(1,batch_size, output_size+1)),hidden)
        
#        length = outp_seq_len * batch_size
        length = outp_seq_len * batch_size
        lengthes +=  length
    
#            y_out = tag_scores[:outp_seq_len,:] #remove end of sequence indicator
        loss = criterion(y_out, Y)      # calculate loss
        losses += loss
        
        y_out_binarized = y_out.clone().data # binary output
        y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)
    
        # The cost is the number of error bits per sequence
        cost = torch.sum(torch.abs(y_out_binarized - Y.data))
        costs += cost
        
        loss.backward()
        optimizer.step()
        
        if batch_num % interval == 0 or batch_num == 1 :
            list_loss.append(losses.data/lengthes)
            list_seq_num.append(lengthes/1000) # per thousand
            list_cost.append(costs/lengthes)
        
        if display and (batch_num % interval == 0  or batch_num == 1): 
            print ("Epoch %d, loss %f, cost %f" % (batch_num, losses/lengthes, costs/lengthes) )
    return list_seq_num, list_loss, list_cost

def evaluate(model, test_data_loader, criterion) : 
    costs = 0
    losses = 0
    lengthes = 0
    for batch_num, X, Y  in test_data_loader:
        hidden = model.initHidden()

        inp_seq_len, _, _ = X.size()
        outp_seq_len, batch_size, output_size = Y.size()
        optimizer.zero_grad()
        
        # Feed the sequence + delimiter
        for i in range(inp_seq_len):
            _, hidden = model(X[i].view(1,batch_size,-1),hidden)
    
        y_out = Variable(torch.zeros(Y.size()))
        
        for i in range(outp_seq_len):
            y_out[i], hidden = model(Variable(torch.randn(1,batch_size, output_size+1)),hidden)
                
        length = outp_seq_len * batch_size
        lengthes += length
    
#            y_out = tag_scores[:outp_seq_len,:] #remove end of sequence indicator
        loss = criterion(y_out, Y)      # calculate loss
        losses += loss
        
        y_out_binarized = y_out.clone().data # binary output
        y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)
    
        # The cost is the number of error bits per sequence
        cost = torch.sum(torch.abs(y_out_binarized - Y.data))
        costs += cost
        
    print ("T = %d, Average loss %f, average cost %f" % (outp_seq_len, losses.data/lengthes, costs/lengthes))
    return losses.data/lengthes, costs/lengthes

def saveCheckpoint(model,list_seq_num,list_loss, list_cost, path='lstm') :
    print('Saving..')
    state = {
        'model': model,
        'list_seq_num': list_seq_num,
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
    list_seq_num = checkpoint['list_seq_num']
    list_loss = checkpoint['list_loss']
    list_cost = checkpoint['list_cost']
    return model, list_seq_num, list_loss, list_cost
#%%
path = '/Users/fanxiao/Google Drive/UdeM/IFT6135 Representation Learning/homework3'    
os.chdir(path)

init_seed()
NUM_BITS = 8
LSTM_DIM = 100
MIN_LENGTH = 1
MAX_LENGTH = 20
LEARNING_RATE = 3e-5
MOMENTUM = 0.9 
MINI_BATCH = 1
EPOCH = 1000000

model = LSTMCopy(NUM_BITS+1,LSTM_DIM,NUM_BITS,1,MINI_BATCH)

loss_function = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum = MOMENTUM)

train_data_loader = dataloader(EPOCH,MINI_BATCH,NUM_BITS,MIN_LENGTH,MAX_LENGTH)

list_seq_num, list_loss, list_cost = train(model, train_data_loader, loss_function, optimizer, interval=1000)

#%%
plt.plot(list_seq_num,list_cost)
#%%
saveCheckpoint(model,list_seq_num,list_loss, list_cost, path='lstm3_l1_b20_e10000') 

#%%
model, list_seq_num, list_loss, list_cost = loadCheckpoint(path='lstm3')
#%%
list_avg_loss = []
list_avg_cost = []
for T in range(10,110,10) : 
    test_data_loader = dataloader(num_batches = 1,batch_size = MINI_BATCH,seq_width=8,min_len=T,max_len=T)
    avg_loss, avg_cost = evaluate(model,test_data_loader,loss_function)
    list_avg_loss.append(avg_loss)
    list_avg_cost.append(avg_cost)

#%%
    
plt.plot(range(10,110,10),list_avg_cost)
plt.xlabel('T')
plt.ylabel('average cost')