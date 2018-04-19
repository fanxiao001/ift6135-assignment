# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 19:45:06 2018

@author: fanxiao
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score2(generator, num_batch, batch_size=32, cuda=True, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    cuda -- whether or not to run on GPU

    splits -- number of splits
    
    """
    N_img = num_batch * batch_size
    hidden_size = generator.hidden_size
    
    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    
    # Get predictions
    preds = np.zeros((N_img, 1000))
    
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()
    
    for ep in range(num_batch) :
        z_ = torch.randn((batch_size, hidden_size)).view(-1, hidden_size, 1, 1)
        z_ = Variable(z_.type(dtype))
        G_result = generator(z_) #generate fake images (batch,3,64,64)
        
        preds[ep*batch_size:(ep+1)*batch_size] =  get_pred(G_result)
        
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N_img // splits): (k+1) * (N_img // splits)]
        py = np.mean(part, axis=0) #p(y)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i,:] #p(y|x)
            scores.append(entropy(pyx, py)) #KL
        split_scores.append(np.exp(np.mean(scores))) #exp(E[KL])

    return np.mean(split_scores), np.std(split_scores)



def inception_score(z_, generator, batch_size=32, cuda=True, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    cuda -- whether or not to run on GPU

    splits -- number of splits
    
    """
    N_img = len(z_)
    dataloader = torch.utils.data.DataLoader(z_, batch_size=batch_size)
    
    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    
    # Get predictions
    preds = np.zeros((N_img, 1000))
    
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()
    
    for ep, batch in enumerate(dataloader, 0):

        batch = batch.type(dtype)
        G_result = generator(Variable(batch))
        preds[ep*batch_size:(ep+1)*batch_size] =  get_pred(G_result)
        
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N_img // splits): (k+1) * (N_img // splits)]
        py = np.mean(part, axis=0) #p(y)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i,:] #p(y|x)
            scores.append(entropy(pyx, py)) #KL
        split_scores.append(np.exp(np.mean(scores))) #exp(E[KL])

    return np.mean(split_scores), np.std(split_scores)
