# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 19:26:06 2018

@author: fanxiao
"""
import os
path = 'C:/Users/lingyu.yue/Documents/Xiao_Fan/GAN'
os.chdir(path)


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
from inception_score import inception_score
from inception_score import inception_score2
#%%

# root path depends on your computer
root = 'C:/Users/lingyu.yue/Documents/Xiao_Fan/GAN/img_align_celeba/img_align_celeba/'
save_root = 'C:/Users/lingyu.yue/Documents/Xiao_Fan/GAN/img_align_celeba/resized_celebA/' 

resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + 'celebA'):
    os.mkdir(save_root + 'celebA')
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = imresize(img, (resize_size, resize_size))
    plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

    if (i % 1000) == 0:
        print('%d images complete' % i)
        
#%%

def train(generator, discriminator, G_optimizer, D_optimizer,train_data_loader, BCE_loss, num_epochs, hidden_size=100, critic=1, score=True) :

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    
    print('Training start!')
    start_time = time.time()
    
    if score : 
        test_z = torch.randn(10000,generator.hidden_size,1,1)
        
    for epoch in range(num_epochs):
        D_losses = []
        G_losses = []
    
        # learning rate decay
        if (epoch+1) == 11:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")
    
        if (epoch+1) == 16:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")
    
        num_iter = 0
    
        epoch_start_time = time.time()
        for x_, _ in train_data_loader:
            
            #For stability, update discriminator several times before updating generator
            D_train_loss_sum = 0
            mini_batch = x_.size()[0]
    
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            
            if use_cuda :
                x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            else :
                x_, y_real_, y_fake_ = Variable(x_), Variable(y_real_), Variable(y_fake_)
                
            for n in range(critic) :
                
                # train discriminator D : maximize E[log(D(x))]+E[log(1-D(G(z)))], minimize -[]
                discriminator.zero_grad()
        

                D_result = discriminator(x_).squeeze() #(batch,100,1,1) => (batch,100)
                D_real_loss = BCE_loss(D_result, y_real_) #-log(D(x)) BEC_loss = -(ylogx+(1-y)log(1-x))
        
                z_ = torch.randn((mini_batch, hidden_size)).view(-1, hidden_size, 1, 1)
                if use_cuda : 
                    z_ = Variable(z_.cuda())
                else :
                    z_ = Variable(z_)
                G_result = generator(z_)
        
                D_result = discriminator(G_result).squeeze()
                D_fake_loss = BCE_loss(D_result, y_fake_) #-log(1-D(G(z)))
        
                D_train_loss = D_real_loss + D_fake_loss
        
                D_train_loss.backward()
                D_train_loss_sum += D_train_loss.data[0]
                D_optimizer.step()
    
            D_losses.append(D_train_loss_sum/critic)
    
            # train generator G : maximize E[log(D(G(z)))], minimize -[]
            generator.zero_grad()
    
            z_ = torch.randn((mini_batch, hidden_size)).view(-1, hidden_size, 1, 1)
            if use_cuda : 
                z_ = Variable(z_.cuda())
            else :
                z_ = Variable(z_)
    
            G_result = generator(z_)
            D_result = discriminator(G_result).squeeze()
            G_train_loss = BCE_loss(D_result, y_real_) #-log(1-D(G(z)))
            G_train_loss.backward()
            G_optimizer.step()
    
            G_losses.append(G_train_loss.data[0])
    
            num_iter += 1
    
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
    
    
        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), num_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                  torch.mean(torch.FloatTensor(G_losses))))
#        p = 'CelebA_DCGAN_results/Random_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
#        fixed_p = 'CelebA_DCGAN_results/Fixed_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
#        show_result((epoch+1), save=True, path=p, isFix=False)
#        show_result((epoch+1), save=True, path=fixed_p, isFix=True)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        
        if score :
            print(inception_score(test_z,G,D,128,splits=10))
    
    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)
    
    print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), num_epochs, total_ptime))
    print("Training finish!")
    return  train_hist

def saveCheckpoint(generator,discriminator,train_hist, path='GAN', use_cuda=True) :
    print('Saving..')
    state = {
        'generator':  generator.cpu().state_dict()if use_cuda else generator.state_dict(),
        'discriminator': discriminator.cpu().state_dict() if use_cuda else discriminator.state_dict(),
        'train_hist' : train_hist
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+path)

def loadCheckpoint(path='GAN', hidden_size = 100, use_cuda=True):
    dtype = torch.FloatTensor
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+path)
    generator_params = checkpoint['generator']
    discriminator_params = checkpoint['discriminator']
    G = generator(128,hidden_size)
    G.load_state_dict(generator_params)
    D = discriminator(128)
    D.load_state_dict(discriminator_params)
    if use_cuda :
        G.cuda()
        D.cuda()
    train_hist = checkpoint['train_hist']

    return G,D,train_hist

def test(epoch, model, test_loader, loss_function):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
#        if i == 0:
#            n = min(data.size(0), 8)
#            comparison = torch.cat([data[:n],
#                                  recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
#            save_image(comparison.data.cpu(),
#                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128, hidden_size=100):
        super(generator, self).__init__()
#        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0
        self.deconv1 = nn.ConvTranspose2d(hidden_size, d*8, 4, 1, 0) #1->4
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1) #4->8
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1) #8->16
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1) #16->32
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1) #32->64
        
        self.hidden_size = hidden_size

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input): #input (batch,hidden_size,1,1)
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x)) #output (batch,3,64,64)

        return x

class generator_Upsampling(nn.Module):
    # initializers
    def __init__(self, d=128, hidden_size=100, mode='nearest'):
        super(generator_Upsampling, self).__init__()
        self.upsampling1 = nn.Upsample(scale_factor=4,mode=mode) #1->4 input(batch,100,1,1)=>(batch,100,4,4)
        self.conv1 = nn.Conv2d(hidden_size, d*8, 3, 1, 1) # => (batch,d*8,4,4) (110)(4-1+0)/1+1 (4-k+2p)/s+1 (4-3+2)/1+1
        self.conv1_bn = nn.BatchNorm2d(d*8)
        self.upsampling2 = nn.Upsample(scale_factor=2,mode=mode) #=>(batch,d*8,8,8)
        self.conv2 = nn.Conv2d(d*8, d*4, 3, 1, 1) #=>(batch,d*4,8,8)
        self.conv2_bn = nn.BatchNorm2d(d*4)
        self.upsampling3 = nn.Upsample(scale_factor=2,mode=mode) #=>(batch,d*4,16,16)
        self.conv3 = nn.Conv2d(d*4, d*2, 3, 1, 1) #=>(batch,d*2,16,16)
        self.conv3_bn = nn.BatchNorm2d(d*2)
        self.upsampling4 = nn.Upsample(scale_factor=2,mode=mode) #=>(batch,d*2,32,32)
        self.conv4 = nn.Conv2d(d*2, d, 3, 1, 1) #=>(batch,d,32,32)
        self.conv4_bn = nn.BatchNorm2d(d)
        self.upsampling5 = nn.Upsample(scale_factor=2,mode=mode) #=>(batch,d,64,64)
        self.conv5 = nn.Conv2d(d, 3, 3, 1, 1)  #=>(batch,3,64,64)
        
        self.hidden_size = hidden_size

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.conv1_bn(self.conv1(self.upsampling1(input))))
        x = F.relu(self.conv2_bn(self.conv2(self.upsampling2(x))))
        x = F.relu(self.conv3_bn(self.conv3(self.upsampling3(x))))
        x = F.relu(self.conv4_bn(self.conv4(self.upsampling4(x))))
        x = F.tanh(self.conv5(self.upsampling5(x)))

        return x


class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
#        in_channels, out_channels, kernel_size, stride, padding
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1) # (64-4+2)/2+1 = 32
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1) # (32-4+2)/2+1= 16
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1) #(16-4+2)/2+1=8
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1) #(8-4+2)/2+1=4
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0) #(4-4)/1+1=1 =>(batch,1,1,1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input): #input (batch,3,64,64)
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x)) #output (batch,1,1,1)

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        

def show_result(G,D,num_epoch, hidden_size = 100, show = False, save = False, path = 'result.png'):
    z_ = torch.randn((5*5, hidden_size)).view(-1, hidden_size, 1, 1)
    if use_cuda : 
        z_ = Variable(z_.cuda())
    else : 
        z_ = Variable(z_)
#    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
    
#%%
#path = '/Users/fanxiao/Google Drive/UdeM/IFT6135 Representation Learning/homework4'
path = 'C:/Users/lingyu.yue/Documents/Xiao_Fan/GAN'
os.chdir(path)
#img_root = '/Users/fanxiao/datasets/resized_celebA/'
img_root = 'C:/Users/lingyu.yue/Documents/Xiao_Fan/GAN/img_align_celeba/resized_celebA/'
IMAGE_RESIZE = 64
train_sampler = range(4000)

batch_size = 128
lr = 0.001
train_epoch = 20
hidden_dim = 100
use_cuda = torch.cuda.is_available()

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

# network

G = generator_Upsampling(128, hidden_dim,'nearest')
#G = generator(128,hidden_dim)
D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
if use_cuda : 
    G.cuda()
    D.cuda()
    
train_data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler, num_workers=10)

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
#model = VAE()


train_hist = train(G,D,G_optimizer,D_optimizer,train_data_loader,BCE_loss,train_epoch,hidden_dim)

#%%
saveCheckpoint(G,D,train_hist,'GANnearest_t4000_h100_ep20',use_cuda)

#%%
G,D,train_hist = loadCheckpoint('GANvanilla_t2000_h100_ep15',hidden_dim,use_cuda=True)
#%%
show_result(G,D,train_epoch, hidden_dim, show=True,save=True, path='figures/result_nearest.pdf')

#%%

 
#%%    
inception_score(imgs,D,100,128,splits=10)
    

#from inception_score import inception_score
#for ep in range(100) :
#    imgs = torch.randn(100,hidden_dim,1,1)    
#imgs = G(Variable(imgs.cuda()))

#%%
x = dataset[0][0]
print ("image size : ", x.size())
plt.imshow((dataset[0][0].numpy().transpose(1, 2, 0) + 1) / 2)
x = Variable(x.view(1,3,64,64))
deconv = nn.ConvTranspose2d(3, 3, 4, 2, 1)
deconv.weight.data.normal_(mean=0.0, std=0.02)
deconv.bias.data.zero_()
deconv_x = deconv(x).squeeze(0)
print ('Deconvolution :' , deconv_x.size())
plt.imshow((deconv_x.data.numpy().transpose(1, 2, 0) + 1) / 2)

#%%

x = dataset[0][0]
print ("image size : ", x.size())
plt.imshow((dataset[0][0].numpy().transpose(1, 2, 0) + 1) / 2)
x = Variable(x.view(1,3,64,64))
#%%
plt.subplot(2,2,1)
x = dataset[0][0]
print ("image size : ", x.size())
plt.imshow((dataset[0][0].numpy().transpose(1, 2, 0) + 1) / 2)
x = Variable(x.view(1,3,64,64))

conv = nn.Conv2d(3,10,4,2,1)
conv_x = conv(x) # (batch,10,32,32)

# Deconvolution (transposed convolution) with paddings and strides.
deconv = nn.ConvTranspose2d(10,3,4,2,1)
#deconv.weight.data.normal_(mean=0.0, std=0.05)
#deconv.bias.data.zero_()

deconv_x = deconv(conv_x).squeeze(0)
plt.subplot(2,2,2)
plt.imshow((deconv_x.data.numpy().transpose(1, 2, 0) + 1) / 2)

# Nearest-Neighbor Upsampling followed by regular convolution.
plt.subplot(2,2,3)
upsampling_nearest = nn.Upsample(scale_factor=2,mode='nearest')
conv2 = nn.Conv2d(10, 3, 1, 1, 0)
upsampling_nearest_x = conv2(upsampling_nearest(conv_x)).squeeze(0)
plt.imshow((upsampling_nearest_x.data.numpy().transpose(1, 2, 0) + 1) / 2)

# Bilinear Upsampling followed by regular convolution
plt.subplot(2,2,4)
upsampling_bilinear = nn.Upsample(scale_factor=2,mode='bilinear')
upsampling_bilinear_x = conv2(upsampling_bilinear(conv_x)).squeeze(0)
plt.imshow((upsampling_bilinear_x.data.numpy().transpose(1, 2, 0) + 1) / 2)
plt.savefig('/Users/fanxiao/Google Drive/UdeM/IFT6135 Representation Learning/homework4/figures/faces.pdf')