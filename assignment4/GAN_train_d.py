#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # # #
# GAN_train_d.py
# @author Zhibin.LU
# @created Tue Apr 17 2018 11:18:27 GMT-0400 (EDT)
# @last-modified Wed Apr 18 2018 17:09:29 GMT-0400 (EDT)
# @website: https://louis-udm.github.io
# @description 
# # # #


#%%
import os
import importlib

import time
import matplotlib.pyplot as plt
from scipy.misc import imresize
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import itertools
# path = 'C:/Users/lingyu.yue/Documents/Xiao_Fan/GAN'
path="/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/assignment4/"
if os.path.isdir(path):
    os.chdir(path)
else:
    os.chdir("./")
print(os.getcwd())

import GAN_CelebA
from inception_score import inception_score
from inception_score import inception_score2
#from GAN_train import loadCheckpoint,generator,generator_Upsampling,discriminator,show_result
importlib.reload(GAN_CelebA)

#%%

#path = '/Users/fanxiao/Google Drive/UdeM/IFT6135 Representation Learning/homework4'
# path = 'C:/Users/lingyu.yue/Documents/Xiao_Fan/GAN'
# path = '/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/assignment4/'
# os.chdir(path)
#img_root = '/Users/fanxiao/datasets/resized_celebA/'
# img_root = 'C:/Users/lingyu.yue/Documents/Xiao_Fan/GAN/img_align_celeba/resized_celebA/'
img_root = "img_align_celeba/resized_celebA/"
IMAGE_RESIZE = 64
sample_num=100
train_sampler = range(sample_num) #2000,4000, 150000

batch_size = 128
lr = 0.001 #0.001, 0.0002
train_epoch = 30
hidden_dim = 100
critic_max=10

use_cuda = torch.cuda.is_available()
torch.manual_seed(999)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(999)

data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
dataset = datasets.ImageFolder(root=img_root, transform=data_transform)

# generate some fake images to test data performance
#z = torch.randn(10000,hidden_dim,1,1)
#test_dataloader = torch.utils.data.DataLoader(z, batch_size=batch_size)
#if use_cuda:
#    dtype = torch.cuda.FloatTensor
#else:
#    dtype = torch.FloatTensor
#test_imgs = torch.randn(10000,3,64,64)
#for ep, z_ in enumerate(test_dataloader, 0): 
#     fakes = G(Variable(z_.type(dtype))).data.cpu()
#     test_imgs[ep*batch_size:(ep+1)*batch_size,:] = fakes
#%%

''' 
Train networks
'''
train_data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler, num_workers=10)

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()
#model = VAE()

'''
Deconvolution (transposed convolution) with paddings and strides
'''

# G = GAN_CelebA.generator(128,hidden_dim)
# D = GAN_CelebA.discriminator(128)
# G.weight_init(mean=0.0, std=0.02)
# D.weight_init(mean=0.0, std=0.02)
# if use_cuda : 
#     G.cuda()
#     D.cuda()
# # Adam optimizer
# G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
# D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# train_hist = GAN_CelebA.train(G,D,G_optimizer,D_optimizer,train_data_loader,BCE_loss,train_epoch,hidden_dim,critic=1)
# GAN_CelebA.saveCheckpoint(G,D,train_hist,'GANDeconvolution_t4000_h100_ep20.c1',use_cuda)

# Dynamic critic of grandiant descent between Distriminator and Generator.
G = GAN_CelebA.generator(128,hidden_dim)
D = GAN_CelebA.discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
if use_cuda : 
    G.cuda()
    D.cuda()
# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

train_hist = GAN_CelebA.train2(G,D,G_optimizer,D_optimizer,train_data_loader,\
        BCE_loss,train_epoch,hidden_dim,critic_max=critic_max,savepath='GANDeconv_t'+str(sample_num)+'_h'+str(hidden_dim)+'_train2')
GAN_CelebA.saveCheckpoint(G,D,train_hist,'GANDeconvolution_t'+str(sample_num)+'_h'+str(hidden_dim)+'_ep30.train2',use_cuda)
