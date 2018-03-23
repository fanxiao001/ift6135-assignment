#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:01:49 2018

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
from aio import EncapsulatedNTM
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
    
def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)
    
def train(model, train_data_loader, criterion, optimizer,  interval = 1000, display = True) : 
    list_seq_num = []
    list_loss = []
    list_cost = []
    lengthes = 0
    losses = 0
    costs = 0
    for batch_num, X, Y  in train_data_loader:

        inp_seq_len, _, _ = X.size()
        outp_seq_len, batch_size, output_size = Y.size()
        model.init_sequence(batch_size)
        
        optimizer.zero_grad()
        
        # Feed the sequence + delimiter
        for i in range(inp_seq_len):
            model(X[i])
    
        y_out = Variable(torch.zeros(Y.size()))
        
        for i in range(outp_seq_len):
            y_out[i], _ =  model()
        
        length = batch_size
        lengthes +=  length
    
        loss = criterion(y_out, Y)      # calculate loss
        losses += loss
        
        y_out_binarized = y_out.clone().data # binary output
        y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)
    
        # The cost is the number of error bits per sequence
        cost = torch.sum(torch.abs(y_out_binarized - Y.data))
        costs += cost
        
        loss.backward()
        clip_grads(model)
        optimizer.step()
        
        if batch_num % interval == 0  :
            list_loss.append(losses.data/interval/batch_size)
            list_seq_num.append(lengthes/1000) # per thousand
            list_cost.append(costs/interval/batch_size)

        
        if display and (batch_num % interval == 0 ): 
            print ("Epoch %d, loss %f, cost %f" % (batch_num, losses/interval/batch_size, costs/interval/batch_size) )
            costs = 0
            losses = 0        
            
    return list_seq_num, list_loss, list_cost

def evaluate(model, test_data_loader, criterion) : 
    costs = 0
    losses = 0
    lengthes = 0
    for batch_num, X, Y  in test_data_loader:

        inp_seq_len, _, _ = X.size()
        outp_seq_len, batch_size, output_size = Y.size()
        optimizer.zero_grad()
        model.init_sequence(batch_size)
        
        # Feed the sequence + delimiter
        for i in range(inp_seq_len):
            model(X[i])
    
        y_out = Variable(torch.zeros(Y.size()))
        
        for i in range(outp_seq_len):
            y_out[i], _ = model()
                
        length =  batch_size
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

init_seed(10)
NUM_BITS = 8
LSTM_DIM = 256
MIN_LENGTH = 1
MAX_LENGTH = 20
LEARNING_RATE = 1e-4
MOMENTUM = 0.9 
MINI_BATCH = 1
EPOCH = 50000

model = EncapsulatedNTM(num_inputs=NUM_BITS+1, num_outputs=NUM_BITS,
                        controller_size=100, controller_layers=1, num_heads=1, N=128, M=20, controller_type ='mlp')

#model = EncapsulatedNTM(num_inputs=NUM_BITS+1, num_outputs=NUM_BITS,
#                        controller_size=100, controller_layers=1, num_heads=1, N=128, M=20, controller_type ='lstm')

loss_function = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum = MOMENTUM)

train_data_loader = dataloader(EPOCH,MINI_BATCH,NUM_BITS,MIN_LENGTH,MAX_LENGTH)

#%%
list_seq_num, list_loss, list_cost = train(model, train_data_loader, loss_function, optimizer, interval=500)

#%%
plt.plot(list_seq_num,list_cost)
plt.xlabel('sequence number (thousands)')
plt.ylabel('cost per sequence (bits)')
#%%
saveCheckpoint(model,list_seq_num,list_loss, list_cost, path='ntm_mlp_l1_b1_e50000_i500') 

#%%
#model, list_seq_num, list_loss, list_cost = loadCheckpoint(path='lstm3_l1_b100_e10000')
model2, list_seq_num2, list_loss2, list_cost2 = loadCheckpoint(path='lstm3_l1_b20_e50000')

#%%
plt.plot(list_seq_num2,list_cost2, label='LSTM')
plt.plot(list_seq_num,list_cost, label='NTM LSTM')
plt.legend()

#%%
list_avg_loss = []
list_avg_cost = []
for T in range(10,110,10) : 
    test_data_loader = dataloader(num_batches = 20,batch_size = MINI_BATCH,seq_width=8,min_len=T,max_len=T)
    avg_loss, avg_cost = evaluate(model,test_data_loader,loss_function)
    list_avg_loss.append(avg_loss)
    list_avg_cost.append(avg_cost)
