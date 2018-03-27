#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:01:49 2018

@author: fanxiao
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import gridspec
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

def evaluate_single_batch(net, criterion, X, Y):
    """Evaluate a single batch (without training)."""
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    states = []
    for i in range(inp_seq_len):
        o, state = net(X[i])
        states += [state]

    # Read the output (no input given)
    y_out = Variable(torch.zeros(Y.size()))
    for i in range(outp_seq_len):
        y_out[i], state = net()
        states += [state]

    loss = criterion(y_out, Y)

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    result = {
        'loss': loss.data[0],
        'cost': cost / batch_size,
        'y_out': y_out,
        'y_out_binarized': y_out_binarized,
        'states': states
    }

    return result


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

def visualize_read_write(X,result,N) :
    T, batch_size, num_bits = X.size()
    T = T - 1
    num_bits = num_bits - 1
    
    plt.figure(figsize=(8, 6)) 
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3]) 
    
    ax = plt.subplot(gs[0,0])
    y_in = torch.cat((X[:,0,:].data,torch.zeros(T,num_bits+1)),dim=0)
    ax.imshow(torch.t(y_in), cmap='gray',aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('inputs')
    
    ax = plt.subplot(gs[0,1])
    y_out = torch.cat((torch.zeros(T+1,num_bits),result['y_out_binarized'][:,0,:]),dim=0)
    y_out = torch.cat((y_out,torch.zeros(2*T+1,1)),dim=1)
    ax.imshow(torch.t(y_out), cmap='gray',aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('outputs')
    
    states = result['states']
    read_state = torch.zeros(len(states),N)  # read weight
    write_state = torch.zeros(len(states),N) # write weight
    for i in range(0,len(states)) :
        reads, controller_state, heads_states = states[i]
        read_state[i,:] = heads_states[0][0].data
        write_state[i,:] = heads_states[1][0].data
        
        
    ax = plt.subplot(gs[1,0])
    ax.imshow(torch.t(write_state[:,90:]), cmap='gray',aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('location ------------------->')
    ax.set_xlabel('time ------------------->')
    #ax.text(1,40,'Time', fontsize=11)
    ax.text(6,42,'Write Weightings',fontsize=12)
    
    
    ax = plt.subplot(gs[1,1])
    ax.imshow(torch.t(read_state[:,90:]), cmap='gray',aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('time ------------------->')
    ax.text(6,42,'Read Weightings',fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visualize_head.pdf',bbox_inches = "tight")
    plt.show()

