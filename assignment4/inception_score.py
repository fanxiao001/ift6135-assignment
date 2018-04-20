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


def Wasserstein_distance(x_dataset, z_tensor, generator,w_discriminator, batch_size=32, cuda=True):
    """Computes the Wasserstein_distance of the generated images
    cuda -- whether or not to run on GPU

    """

    sample_num=len(z_tensor)
    train_sampler = range(sample_num)

    x_dataloader =  torch.utils.data.DataLoader(
            x_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=10)

    z_dataset  = torch.utils.data.TensorDataset(
        z_tensor,torch.zeros(len(z_tensor)))
    z_dataloader = torch.utils.data.DataLoader(
        z_dataset, batch_size=batch_size, shuffle=False)


    generator.eval()
    w_discriminator.eval()
    distances=[]
    for x, z in zip(x_dataloader, z_dataloader):
        x ,_ = x
        z ,_ = z

        if cuda :
            x,z = Variable(x.cuda()), Variable(z.cuda())
        else :
            x, z = Variable(x),Variable(z)

        D_x = w_discriminator(x) 
        D_z = w_discriminator(generator(z))
        w_distance=torch.abs(D_x - D_z)
        # w_distance=-(D_x - D_z)
        distances.append(w_distance.data[0])

    generator.train()
    w_discriminator.train()

    return np.mean(distances)


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



def inception_score(z_, generator, batch_size=32, cuda=True, resize=True, splits=1):
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

if __name__=='__main__':

    import os
    import importlib
    # path = 'C:/Users/lingyu.yue/Documents/Xiao_Fan/GAN'
    path="/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/assignment4/"
    if os.path.isdir(path):
        os.chdir(path)
    else:
        os.chdir("./")
    print(os.getcwd())

    import GAN_CelebA
    importlib.reload(GAN_CelebA)

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)

    model_files=[
        '1_GANBilinear_t9900_h100_train3_ep33   ',            
        '1_GANDeconv_t9900_h100_train3_ep45     ',    
        '1_GANDeconv_t9900_h200_train3_ep45     ',    
        '1_GANDeconv_t9999_h100_train2_ep21     ',    
        '1_GAN_W_fixlr_t10000_h100_b64_c5_ep45  ',        
        '1_GAN_W_fixlr_t10000_h200_b64_c5_ep45  ',        
        '1_GANnearest_t10000_h100_ep50_new      ',    
        '2_GANBilinear_t9900_h100_train3_ep18   ',        
        '2_GANDeconv_t9900_h100_train3_ep18     ',    
        '2_GANDeconv_t9900_h200_train3_ep18     ',    
        '2_GANDeconv_t9999_h100_train2_ep9      ',    
        '2_GAN_W_fixlr_t10000_h100_b64_c5_ep18  ',        
        '2_GAN_W_fixlr_t10000_h200_b64_c5_ep18  ',        
        '2_GANnearest_t9900_h100_train3_ep9     ',    
        '3_GANBilinear_t9900_h100_train3_ep6    ',        
        '3_GANDeconv_t9900_h100_train3_ep6      ',    
        '3_GANDeconv_t9900_h200_train3_ep6      ',    
        '3_GANDeconv_t9999_h100_train2_ep6      ',    
        '3_GAN_W_fixlr_t10000_h100_b64_c5_ep6   ',        
        '3_GAN_W_fixlr_t10000_h200_b64_c5_ep6   ',        
        '3_GANnearest_t9900_h100_train3_ep3     ',    
        '4_bad_GANnearest_t9900_h100_train3_ep27  ',        
        '4_bad_GANnearest_t9900_h100_train3_ep45  '        
        ]

    print('model num:',len(model_files))

    dir='./checkpoint/score_models/'

    _,D100_W,train_hist = GAN_CelebA.loadCheckpoint_W('1_GAN_W_fixlr_t10000_h100_b64_c5_ep45',100,use_cuda=use_cuda,dir=dir)
    _,D200_W,train_hist = GAN_CelebA.loadCheckpoint_W('1_GAN_W_fixlr_t10000_h200_b64_c5_ep45',200,use_cuda=use_cuda,dir=dir)
    
    for filename in model_files:

        filename = filename.strip()
        if filename.find('h100')>-1:
            hidden_dim=100
        if filename.find('h200')>-1:
            hidden_dim=200

        if filename.find('Bilinear')>-1:
            G,D,_=GAN_CelebA.loadCheckpoint_Upsampling(filename,hidden_dim,use_cuda=use_cuda,mode='bilinear',dir=dir)
        
        if filename.find('Deconv')>-1:
            G,D,_=GAN_CelebA.loadCheckpoint(filename,hidden_dim,use_cuda=use_cuda,dir=dir)
        
        if filename.find('_W_')>-1:
            G,D,_=GAN_CelebA.loadCheckpoint_W(filename,hidden_dim,use_cuda=use_cuda,dir=dir)
        
        if filename.find('nearest')>-1:
            if filename.find('new')>-1:
                G,D,_=GAN_CelebA.loadCheckpoint_Upsampling(filename,hidden_dim,use_cuda=use_cuda,mode='nearest',dir=dir)
            else:
                G,D,_=GAN_CelebA.loadCheckpoint_Upsampling_old(filename,hidden_dim,use_cuda=use_cuda,mode='nearest',dir=dir)

        inception_score_model1=[]
        inception_score_model2=[]
        inception_score_model3=[]
        inception_score_model4=[]

        test_z = torch.randn(10,hidden_dim,1,1)

        ince_score=inception_score(test_z, G, batch_size=32, cuda=use_cuda, resize=True, splits=10)
        
        if filename.startswith('1_'):
            inception_score_model1.append(ince_score)
        if filename.startswith('2_'):
            inception_score_model2.append(ince_score)
        if filename.startswith('3_'):
            inception_score_model3.append(ince_score)
        if filename.startswith('4_'):
            inception_score_model4.append(ince_score)


        G,D,train_hist = GAN_CelebA.loadCheckpoint(filename,hidden_dim,use_cuda=use_cuda)
        epoch_num=len(train_hist['D_losses'])

        test_z = torch.randn(1000,hidden_dim,1,1)
        w_score=Wasserstein_distance(dataset,test_z, G, D, batch_size=32, cuda=use_cuda)
        print('Wasserstein distance: ',score)
