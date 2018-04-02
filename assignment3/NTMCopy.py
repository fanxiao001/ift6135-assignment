#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:36:42 2018

@author: fanxiao
"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F



def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results

def address(k, beta, g, s, gamma, memory, prev_weight, batch_size):
    """NTM Addressing (according to section 3.3).

    Returns a softmax weighting over the rows of the memory matrix.

    :param k: The key vector.
    :param β: The key strength (focus).
    :param g: Scalar interpolation gate (with previous weighting).
    :param s: Shift weighting.
    :param γ: Sharpen weighting scalar.
    :param w_prev: The weighting produced in the previous time step.
    """
    # Content focus
    wc = _similarity(k, beta, memory,batch_size)

    # Location focus
    wg = _interpolate(prev_weight, wc, g)
    w_hat = _shift(wg, s, batch_size)
    w = _sharpen(w_hat, gamma)

    return w

def _similarity(k, β, memory, batch_size):
    k = k.view(batch_size, 1, -1)
    w = F.softmax(β * F.cosine_similarity(memory + 1e-16, k + 1e-16, dim=-1), dim=1)
    return w

def _interpolate(w_prev, wc, g):
    return g * wc + (1 - g) * w_prev

def _shift(wg, s, batch_size):
    result = Variable(torch.zeros(wg.size()))
    for b in range(batch_size):
        result[b] = _convolve(wg[b], s[b])
    return result

def _sharpen(w_hat, gamma):
    w = w_hat ** gamma
    w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
    return w

def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    t = torch.cat([w[-2:], w, w[:2]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c[1:-1]

class NTMCopy(nn.Module):
    def __init__(self, num_inputs, num_outputs,
             controller_size, controller_layers, num_heads, N, M, controller_type ='lstm'):
        """Initialize an EncapsulatedNTM.
    
        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        """
        super(NTMCopy, self).__init__()
        
        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M
        self.controller_type = controller_type
        
        if self.controller_type == 'lstm' :
            self.controller = nn.LSTM(input_size=self.num_inputs+self.M*self.num_heads,hidden_size=self.controller_size,num_layers=self.controller_layers)
        else :
            self.controller = nn.Linear(self.num_inputs+self.M*self.num_heads,self.controller_size)
            
        
        # controller output => k(M), beta(1), g(1), s(3), gamma(1)
        self.read_transformer = nn.Linear(self.controller_size, self.M+6)
        # controller output => k(M), beta(1), g(1), s(3), gamma(1), e, a 
        self.write_transformer = nn.Linear(self.controller_size,self.M+6+self.M+self.M)
        
        # controller output + r => output 
        self.output_transformer = nn.Linear(self.controller_size+self.M, self.num_outputs)
        
        self.register_buffer('mem_bias', Variable(torch.Tensor(N, M)))

        init_r_bias = Variable(torch.randn(self.M) * 0.01)
        self.register_buffer("read_bias", init_r_bias)

        # Initialize memory bias
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform(self.mem_bias, -stdev, stdev)
    
#        for p in self.controller.parameters():
#            if p.dim() == 1:
#                nn.init.constant(p, 0)
#            else:
#                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
#                nn.init.uniform(p, -stdev, stdev)
        
        # Initialize the linear layers
        nn.init.xavier_uniform(self.output_transformer.weight, gain=1)
        nn.init.normal(self.output_transformer.bias, std=0.01)
        
        # Initialize the linear layers
        nn.init.xavier_uniform(self.read_transformer.weight, gain=1.4)
        nn.init.normal(self.read_transformer.bias, std=0.01)
        
        # Initialize the linear layers
        nn.init.xavier_uniform(self.write_transformer.weight, gain=1.4)
        nn.init.normal(self.write_transformer.bias, std=0.01)
        
            
    def init_sequence(self, batch_size) :
        self.batch_size = batch_size
        self.hidden_state = (Variable(torch.zeros(self.controller_layers, self.batch_size, self.controller_size)),
                Variable(torch.zeros(self.controller_layers, self.batch_size, self.controller_size))) #hidden state of lstm
        self.read = self.read_bias.clone().repeat(batch_size, 1)
        
#        self.read = Variable(torch.Tensor(self.batch_size, self.M).uniform_(0,1)) # read output at time t : length of M
        self.head_state = (Variable(torch.zeros(self.batch_size,self.N)), Variable(torch.zeros(self.batch_size,self.N))) # weight of read/write head : length of N
        self.memory = self.mem_bias.clone().repeat(batch_size,1,1)
        
    def forward(self, x=None) :
        if x is None :
            x = Variable(torch.zeros(self.batch_size, self.num_inputs))
        else :
            x = x.view(self.batch_size,-1)
        inputs = torch.cat((x,self.read), dim=1) #inputs of controller, concanate input and previous read head output
        
        inputs = inputs.unsqueeze(0)  # add 1 in first dimension
        
        if self.controller_type == 'lstm' :
            outp_controller, self.hidden_state = self.controller(inputs,self.hidden_state)
        else :
            outp_controller = self.controller(inputs)
        outp_controller = outp_controller.squeeze(0) #return 1 in first dimension, return batch_size * _
        
        read_state, write_state = self.head_state
        
        
        # write head
        write_transformer_output = self.write_transformer(outp_controller)
        k, beta, g, s, gamma, e, a =  _split_cols(write_transformer_output, [self.M, 1, 1, 3, 1, self.M, self.M])
        
        k=k.clone()
        beta = F.softplus(beta)
        g = F.sigmoid(g)
        s = F.softmax(F.softplus(s), dim=1)
        gamma = 1 + F.softplus(gamma)
        e = F.sigmoid(e)
        
        #write head weight
        write_state = address(k,beta,g,s,gamma,self.memory,write_state,self.batch_size)
        # write to memory
        prev_mem = self.memory.clone()
        self.memory = Variable(torch.Tensor(self.batch_size, self.N, self.M))
        for b in range(self.batch_size):
            erase = torch.ger(write_state[b], e[b])
            add = torch.ger(write_state[b], a[b])
            self.memory[b] = prev_mem[b] * (1 - erase) + add
            
        #read head 
        read_transformer_output = self.read_transformer(outp_controller)
        
        k, beta, g, s, gamma =  _split_cols(read_transformer_output, [self.M, 1, 1, 3, 1])
        k = k.clone()
        beta = F.softplus(beta)
        g = F.sigmoid(g)
        s = F.softmax(F.softplus(s), dim=1)
        gamma = 1 + F.softplus(gamma)
        
        # read head weight
        read_state = address(k,beta,g,s,gamma,self.memory,read_state,self.batch_size)
        # read from memory
        self.read = torch.matmul(read_state.unsqueeze(1), self.memory).squeeze(1)
        
        self.head_state = (read_state, write_state)
        
        # generate output
        inp2 = torch.cat((outp_controller ,self.read ), dim=1)
        o = F.sigmoid(self.output_transformer(inp2))
        
        return o, (self.read,self.hidden_state,self.head_state)
    
