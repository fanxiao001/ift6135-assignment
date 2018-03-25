#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 21:48:28 2018

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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.batch_size = batch_size
#        self.embedding = nn.Linear(input_size,hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layer)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input=None):
        if input is None:
            input = Variable(torch.zeros(self.batch_size, self.input_size))
#        output = self.embedding(input)
        output, self.hidden_state = self.lstm(input.unsqueeze(0), self.hidden_state)
        output = F.sigmoid(self.out(output))
        return output.unsqueeze(0), self.hidden_state
    
#    def forward(self, input, hidden):
#        output, hidden = self.lstm(input, hidden)
#        output = F.sigmoid(self.out(output))
#        return output, hidden

#    def initHidden(self):
#        result = (Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_size)),
#                Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_size)))
#        if torch.cuda.is_available():
#            return result.cuda()
#        else:
#            return result
        
    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.hidden_state = (Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_size)),
                Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_size)))
        
    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params