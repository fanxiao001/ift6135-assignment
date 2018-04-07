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

def inception_score2(generator, discriminator, num_batch, batch_size=128, cuda=True, resize=False, splits=1):
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

#    hidden_size = generator.hidden_size
    preds = torch.zeros((N_img,2))
    
    for ep in range(num_batch) :
        z_ = torch.randn((batch_size, hidden_size)).view(-1, hidden_size, 1, 1)
        z_ = Variable(z_.type(dtype))
        G_result = generator(z_) #generate fake images (batch,3,64,64)
        # Load predition model
        discriminator.eval();
        
        preds[ep*batch_size:(ep+1)*batch_size,0] =  discriminator(G_result).squeeze().data # (batch,1,1,1)
        preds[ep*batch_size:(ep+1)*batch_size,1] =  1.0 - discriminator(G_result).squeeze().data # (batch,1,1,1)
    preds = preds.numpy()
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


# exp(E_x[KL(p(y|x) || p(y))])
def inception_score(imgs, discriminator, batch_size=128, cuda=True, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    cuda -- whether or not to run on GPU

    splits -- number of splits
    
    """
    N_img = len(imgs)
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    
    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

#    hidden_size = generator.hidden_size
    preds = torch.zeros((N_img,2))
    
    for ep, batch in enumerate(dataloader, 0):
#    for ep in range(num_batch) :
#        z_ = torch.randn((batch_size, hidden_size)).view(-1, hidden_size, 1, 1)
#        z_ = Variable(z_.type(dtype))
#        G_result = generator(z_) #generate fake images (batch,3,64,64)
        # Load predition model
        discriminator.eval();
        print(batch.size())
        batch = batch.type(dtype)
        G_result = Variable(batch)
        
        preds[ep*batch_size:(ep+1)*batch_size,0] =  discriminator(G_result).squeeze().data # (batch,1,1,1)
        preds[ep*batch_size:(ep+1)*batch_size,1] =  1.0 - discriminator(G_result).squeeze().data # (batch,1,1,1)
    preds = preds.numpy()
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
