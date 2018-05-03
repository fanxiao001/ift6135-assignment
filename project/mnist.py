#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # # #
# mnist.py
# @author Zhibin.LU
# @created Mon Apr 23 2018 17:19:42 GMT-0400 (EDT)
# @last-modified Thu May 03 2018 14:23:49 GMT-0400 (EDT)
# @website: https://louis-udm.github.io
# @description 
# # # #

#%%
import os
import importlib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms
import torch.utils.data.sampler as sampler
import matplotlib.pyplot as plt
from torch.autograd.gradcheck import zero_gradients

path="/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/projet/"
# path = "C:\\Users\\lingyu.yue\\Documents\\Xiao_Fan\\project"
if os.path.isdir(path):
    os.chdir(path)
else:
    os.chdir("./")
print(os.getcwd())

import  main
importlib.reload(main)

USE_CUDA=torch.cuda.is_available()
main.init_seed()

NO_CLASSES = 10
TRAIN_DATA_SIZE = 50000
TRAIN_EPOCH = 30 #10000
BATCH_SIZE = 128


'''
Load MNIST data
'''
mnist_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=mnist_transforms, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=mnist_transforms, download=True)
indices = list(range(len(mnist_train)))
np.random.shuffle(indices)
train_idx, valid_idx = indices[:TRAIN_DATA_SIZE], indices[TRAIN_DATA_SIZE:]
train_sampler = sampler.SubsetRandomSampler(train_idx)
valid_sampler = sampler.SubsetRandomSampler(valid_idx)
train_data_loader = torch.utils.data.DataLoader(
    mnist_train, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=10)
valid_data_loader = torch.utils.data.DataLoader(
    mnist_train, batch_size=BATCH_SIZE,  sampler=valid_sampler, num_workers=10)
test_data_loader = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
print('Loaded MNIST data, total',len(mnist_train)+len(mnist_test))

normlist=[]
for x,_ in train_data_loader:
    x=x.view(len(x),-1)
    normlist.append(torch.mean(torch.norm(x,2,1)))
print('Mean of Mnist norm (C2) =',torch.mean(torch.Tensor(normlist)))

#%%
'''
Arichitectur of estimateur for MNIST
'''
#class Mnist_Estimateur(nn.Module):
#    # initializers, d=num_filters
#    def __init__(self, d=32, activation='relu'):
#        super(Mnist_Estimateur, self).__init__()
##         in_channels, out_channels, kernel_size, stride, padding, dilation
#        self.conv1 = nn.Conv2d(1, d, 8, 1, 0) # (28-8)+1 = 21
#        self.conv2 = nn.Conv2d(d, d*2, 6, 1, 0) # (21-6)+1= 16
#        self.conv3 = nn.Conv2d(d*2, d*4, 5, 1, 0) # (16-5)+1= 12
#        self.fc1 = nn.Linear(18432,1024)
#        self.fc2 = nn.Linear(1024,NO_CLASSES)
#        if activation == 'relu':
#            self.active = nn.ReLU() 
#        else :
#            self.active = nn.ELU()
#
#    def init_weights(self, mean, std):
#        for m in self._modules:
#            if type(m) == nn.Linear:
#                nn.init.xavier_uniform(m.weight)
#            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
#                m.weight.data.normal_(mean, std)
#                m.bias.data.zero_()
#
#    def forward(self, input): 
#        x = self.active(self.conv1(input))
#        x = self.active(self.conv2(x))
#        x = self.active( self.conv3(x) )
#        x = x.view(x.size(0), -1)
#        x = self.active(self.fc1(x))
#        x = self.fc2(x)
#        return x
#        
    
class Mnist_Estimateur(nn.Module):
    # initializers, d=num_filters
    def __init__(self, d=32, activation='elu'):
        super(Mnist_Estimateur, self).__init__()
        
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=d, kernel_size=(8, 8)), #(28-8 )+1 = 21
            nn.BatchNorm2d(d),
            nn.ELU(),
    
            # Layer 2
            nn.Conv2d(in_channels=d, out_channels=2*d, kernel_size=(6, 6)), # (21-6)+1 = 16 
            nn.BatchNorm2d(2*d)  ,          
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # 8 
            
            # Layer 3
            nn.Conv2d(in_channels=2*d, out_channels=4*d, kernel_size=(5, 5)), # (8-5)+1 = 4
            nn.BatchNorm2d(4*d),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # chanel 128 feature map 2*2
            
        )
        # Logistic Regression
        self.clf = nn.Linear(512, 10)

    def init_weights(self, mean, std):
        for m in self._modules:
            if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()


    def forward(self, input): 
        
        x = self.conv(input)
        return self.clf(x.view(len(x), -1 ))

# plot figure 2 certificate vs. worst case
def plot_certificate(model,loss_train,gamma,valid_data_loader) :
    fig = plt.figure()
    certificate=[] #E_train[phi(theta,z)] + gamma*rho
    list_rho = []
    list_worst = []
    for rho in range(0,400,50) :
        rho = rho/100.0
        certificate.append(loss_train+GAMMA*rho)
        
    #test worst case 
    list_rho = []
    list_worst = []
    for g in [0.07, 0.09, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0] :
        rho, e = main.cal_worst_case(model,valid_data_loader, g, 0.04)
        list_rho.append(rho)
        list_worst.append(e + rho * g)
    
    plt.plot(list_rho,list_worst, c='red', label=r"Test worst-case: $\sup_{P:W_c(P,\hat{P}_{test}) \leq \rho } E_P [l(\theta_{WRM};Z)]$")
    plt.plot(np.array(range(0,400,50))/100.0,certificate,c='blue', label=r"Certificate: $E_{\hat{P}_n}[\phi_{\gamma}(\theta_{WRM};Z)]+\gamma \rho$")
    plt.xlabel(r"$\rho$")
    plt.xlim([0.0,3.6])
    plt.ylim([0.0,2.0])
    plt.xticks([0,0.5,1,1.5,2,2.5,3,3.5])
    plt.legend(loc="lower right")
    return fig

# L2 or infinity attack, return accuracy on test_data_loader
def attack_PGM(model,test_data_loader, p=2, epsilon = 0.01, alpha = 0.1, random=False) :
    model.eval()
    T_adv = 15
    loss_function = nn.CrossEntropyLoss()
    valid_data_x = torch.FloatTensor(len(test_data_loader.dataset),1,28,28)
    valid_data_y = torch.LongTensor(len(test_data_loader.dataset))
    count = 0
    
    for x_, y_ in test_data_loader :
        if USE_CUDA:
            x_, y_ = x_.cuda(), y_.cuda()
        input_var, target_var  = Variable(x_, requires_grad=True), Variable(y_)
        
        if random == True : 
            noise = torch.FloatTensor(x_.size()).uniform_(-epsilon, epsilon)
            if USE_CUDA : 
                noise = noise.cuda()
            input_var.data += noise

        #generate attack data
        for n in range(1, T_adv + 1) :
            step_alpha = float(alpha /np.sqrt(n))
            zero_gradients(input_var)
            output = model(input_var)
            loss = loss_function(output, target_var)
            loss.backward()
            x_grad = input_var.grad.data
            if p == 2:
#                delta_x = epsilon *  x_grad / torch.norm(x_grad.view(len(x_),1),2,1)
                grad_ = x_grad.view(len(x_),-1)
                grad_ = grad_/torch.norm(grad_,2,1).view(len(x_),1).expand_as(grad_)
                normed_grad = epsilon * grad_.view_as(x_grad)  
            else:
                # infinity-norm
                normed_grad =  epsilon * torch.sign(x_grad)
            # xi + alpha_t * delta_x
#            normed_grad = step_alpha * normed_grad 
#            normed_grad.clamp_(-epsilon, epsilon)
#            input_var.data +=  normed_grad

            normed_grad.clamp_(-epsilon, epsilon)
            step_adv = input_var.data + step_alpha * normed_grad # x^(t+1) = x^(t) + alpha * delta_x^t
            total_adv = step_adv - x_  #x^t - x
            total_adv.clamp_(-epsilon, epsilon) # ||x^t-x|| <= epsilon
            input_adv = x_ + total_adv 
            input_adv.clamp_(-1.0, 1.0) #mnist data between -1,1
            input_var.data = input_adv
            
#            print (np.all(input_var.data.cpu() == x_.data))
            
        valid_data_x[count:count+len(x_),:] = input_var.data.cpu()
        valid_data_y[count:count+len(x_)] = y_.clone().cpu()
        count += len(x_)
    dataset = torch.utils.data.TensorDataset(valid_data_x, valid_data_y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    return main.evaluate(model,data_loader)
    
# WRM attack, return accuracy on test_data_loader
def attack_WRM(model,test_data_loader, gamma=0.04, max_lr0=0.0001, epsilon = 0.01, random=False, get_err=False) :
    model.eval()
    T_adv = 15
    loss_function = nn.CrossEntropyLoss()
    
    if get_err:
        valid_data_x = torch.FloatTensor(len(test_data_loader.dataset),1,28,28)
        valid_data_y = torch.LongTensor(len(test_data_loader.dataset))
    
    count = 0
    err=0
    rhos=[]
    for x_, y_ in test_data_loader :
        if USE_CUDA:
            x_, y_ = x_.cuda(), y_.cuda()
        x_, y_  = Variable(x_), Variable(y_)

        #initialize z_hat with x_
        z_hat = x_.data.clone()
        if USE_CUDA:
            z_hat = z_hat.cuda()
        if random : 
            noise = torch.FloatTensor(x_.size()).uniform_(-epsilon, epsilon)
            if USE_CUDA : 
                noise = noise.cuda()
            z_hat += noise
            
        z_hat = Variable(z_hat,requires_grad=True)
        #running the maximizer for z_hat
        optimizer_zt = torch.optim.Adam([z_hat], lr=max_lr0)
        loss_zt = 0 # phi(theta,z0)
        rho = 0 #E[c(Z,Z0)]
        for n in range(1,T_adv+1) :
            optimizer_zt.zero_grad()
            delta = z_hat - x_
            rho = torch.mean((torch.norm(delta.view(len(x_),-1),2,1)**2)) 
            loss_zt = - ( loss_function(model(z_hat),y_)-  gamma * rho)
            loss_zt.backward()
            optimizer_zt.step()
            main.adjust_lr_zt(optimizer_zt,max_lr0, n+1)
            
        rhos.append(rho.data[0])
        
        if get_err:
            valid_data_x[count:count+len(x_),:] = z_hat.cpu()
            valid_data_y[count:count+len(x_)] = y_.clone().cpu()
            count += len(x_)
    if get_err:
        dataset = torch.utils.data.TensorDataset(valid_data_x, valid_data_y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        err=(1.0-main.evaluate(model,data_loader))/100

    return torch.mean(torch.FloatTensor(rhos)),err

# WRM attack, return accuracy on test_data_loader
def attack_WRM_sample(model,test_data_loader, gamma=0.04, max_lr0=0.0001, epsilon = 0.01, random=False, get_err=False) :
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    
    if get_err:
        valid_data_x = torch.FloatTensor(len(test_data_loader.dataset),1,28,28)
        valid_data_y = torch.LongTensor(len(test_data_loader.dataset))
    
    for x_, y_ in test_data_loader :
        # if y_

        if USE_CUDA:
            x_, y_ = x_.cuda(), y_.cuda()
        x_, y_  = Variable(x_), Variable(y_)

    # #initialize z_hat with x_
    # z_hat = x_.data.clone()
    # if USE_CUDA:
    #     z_hat = z_hat.cuda()
    # if random : 
    #     noise = torch.FloatTensor(x_.size()).uniform_(-epsilon, epsilon)
    #     if USE_CUDA : 
    #         noise = noise.cuda()
    #     z_hat += noise
        
    # z_hat = Variable(z_hat,requires_grad=True)
    # #running the maximizer for z_hat
    # optimizer_zt = torch.optim.Adam([z_hat], lr=max_lr0)
    # loss_zt = 0 # phi(theta,z0)
    # rho = 0 #E[c(Z,Z0)]
    # for n in range(1,T_adv+1) :
    #     optimizer_zt.zero_grad()
    #     delta = z_hat - x_
    #     rho = torch.mean((torch.norm(delta.view(len(x_),-1),2,1)**2)) 
    #     loss_zt = - ( loss_function(model(z_hat),y_)-  gamma * rho)
    #     loss_zt.backward()
    #     optimizer_zt.step()
    #     main.adjust_lr_zt(optimizer_zt,max_lr0, n+1)
        
    # rhos.append(rho.data[0])
        

    return 1

def rho_vs_gamma(model, test_data_loader, max_lr0, random=False, get_err=False) :
    C2 = 9.21
    gammas = C2/np.array(range(5,105,5))  #0.5-0.01 
    print (gammas)
    rhos = []
    errors = []
    for g in gammas :
        # print(float(g))
        rho,err=attack_WRM(model,test_data_loader,float(g),max_lr0, random, get_err)
        rhos.append(rho)
        errors.append(err)
    return np.array(range(5,105,5)),rhos, errors


# errors when attacked
def get_errors(model, test_data_loader, p=2, alpha =0.1, random=False) :
    C2 = 9.21
    Cinf =  1.0
    epsilons = np.array(range(0,22,2))/100.0 * Cinf
    if p==2  :
        epsilons = np.array(range(0,27,2))/100.0 * C2
    errors = []
    for e in epsilons :
        errors.append(1.0-attack_PGM(model,test_data_loader,p,float(e), alpha, random)/100.0)
    return epsilons, errors

def plot_attack_error(list_errors,labels, p=2) :
    fig = plt.figure()
    epsilons = np.array(range(0,22,2))/100.0 
    plt.xlabel(r"$\epsilon_{adv}/C_{\infty}$") 
    plt.xticks([0,0.05,0.1,0.15,0.2])
    if p==2  :
        epsilons = np.array(range(0,27,2))/100.0 
        plt.xlabel(r"$\epsilon_{adv}/C_2$")
        plt.xticks([0,0.05,0.1,0.15,0.2,0.25])
    
    for i, errors in enumerate(list_errors) :
        plt.plot(epsilons, errors, label=labels[i])
    plt.ylabel('Error')
    plt.yscale('log')
    plt.yticks([0.01,0.1,1.0])
    plt.legend()
    return fig

MIN_LR0 = 0.0001 
MAX_LR0 = 0.04 #step size for iterative method and attack method
#number of adversarial iterations
T_ADV = 15

C2 = 9.21
Cinf = 1.00
GAMMA = 0.04 * C2
EPSILON = 0.45

loss_function=nn.CrossEntropyLoss()

model = Mnist_Estimateur(activation='elu')
if USE_CUDA:
    model.cuda()

#def plot_attack_error(models,test_data_loader,labels, p=2) :
#    fig = plt.figure()
#    C2 = 9.21
#    epsilons = np.array(range(0,27,2))/100.0
#    plt.xlabel(r"$\epsilon_{adv}/C_{\infty}$") 
#    if p==2  :
#        epsilons = np.array(range(0,22,2))/100.0 * C2
#        plt.xlabel(r"$\epsilon_{adv}/C_2$")
#    
#    for i, model in enumerate(models) :
#        errors = []
#        for e in epsilons : 
#            errors.append(1.0-attack_PGM(model,test_data_loader,p,float(e), alpha=0.1)/100.0)   
#        plt.plot(epsilons, errors, label=labels[i])
#    plt.ylabel('Error')
#    plt.yscale('log')
#    plt.legend()
#    return fig

#%%
#    C2 = 0
#    Cinf = 0
#    count = 0
#    for x_, y_ in train_data_loader :
#        if USE_CUDA:
#            x_, y_ = x_.cuda(), y_.cuda()
#        x_, y_ = Variable(x_), Variable(y_)
#        C2 += torch.sum(torch.sqrt(torch.sum(torch.norm(x_,p=2,dim=3)**2, dim=2))).data[0]
#        infnorm, _ = torch.max(torch.sum(torch.abs(x_),dim=3), dim=2)
#        Cinf += torch.sum(infnorm).data[0]
#        count += len(x_)
#    C2 = C2/float(count)
#    Cinf = Cinf/float(count)

if False and __name__=='__main__':
    # optimizer = optim.Adam(model.parameters(), lr=LR0_MIN, betas=(0.5, 0.999))
    # optimizer = optim.RMSprop(model.parameters(), lr=LR0_MIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=MIN_LR0)

    model.init_weights(mean=0.0, std=0.02)
    main.train(model,optimizer,loss_function, train_data_loader,valid_data_loader, \
        TRAIN_EPOCH ,min_lr0=MIN_LR0,min_lr_adjust=False, savepath='mnist_erm')
    
    model.init_weights(mean=0.0, std=0.02)
    main.train_FGM(model,optimizer,loss_function, train_data_loader,valid_data_loader, \
        TRAIN_EPOCH ,EPSILON, min_lr0=MIN_LR0,min_lr_adjust=False, savepath='mnist_fgm')
    
    model.init_weights(mean=0.0, std=0.02)
    main.train_IFGM(model,optimizer,loss_function, train_data_loader,valid_data_loader, \
        TRAIN_EPOCH ,EPSILON, min_lr0=MIN_LR0,alpha=MAX_LR0, min_lr_adjust=False, savepath='mnist_ifgm')

    model.init_weights(mean=0.0, std=0.02)
    main.train_WRM(model,optimizer,loss_function, train_data_loader,valid_data_loader, \
        TRAIN_EPOCH , GAMMA, max_lr0=MAX_LR0, min_lr0=MIN_LR0, min_lr_adjust=True, savepath='mnist_wrm')

#%%
if False and __name__=='__main__':
    model = Mnist_Estimateur()
    model, train_hist = main.loadCheckpoint(model,'mnist_wrm_ep24')
    fig = plot_certificate(model,train_hist['loss_maxItr'][-1]+0.05,GAMMA,test_data_loader)  
      
#%%
#test
# if __name__=='__main__':
#     model = Mnist_Estimateur()
#     model,train_hist = main.loadCheckpoint(model,'mnist_wrm_ep30')
#     certificate=[] #E_train[phi(theta,z)] + gamma*rho
#     list_rho = []
#     list_worst = []
#     #Rho, loss_maxItr = main.cal_worst_case(model,valid_data_loader, GAMMA, 0.04)
#     for rho in range(0,400,50) :
#         rho = rho/100.0
#         certificate.append(train_hist['loss_maxItr'][-1]+0.05+GAMMA*rho)
#     #    certificate.append(loss_maxItr+GAMMA*rho)

#     for g in [0.07, 0.09, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.8, 1.2, 2.0, 3.0, 5.0] :
#         print (g)
#         rho, e = main.cal_worst_case(model,valid_data_loader, g, 0.04)
#         print (rho, e+rho*g)
#         list_rho.append(rho)
#         list_worst.append(e + rho * g)

#     plt.plot(list_rho,list_worst, c='red', label=r"Test worst-case: $\sup_{P:W_c(P,\hat{P}_{test}) \leq \rho } E_P [l(\theta_{WRM};Z)]$")
#     plt.plot(np.array(range(0,400,50))/100.0,certificate,c='blue', label=r"Certificate: $E_{\hat{P}_n}[\phi_{\gamma}(\theta_{WRM};Z)]+\gamma \rho$")
#     plt.xlim([0.0,3.6])
#     plt.ylim([0.0,2.0])
#     plt.xticks([0,0.5,1,1.5,2,2.5,3,3.5])
#     plt.legend()

#%%
if False and __name__=='__main__':
    model = Mnist_Estimateur()
    filename = 'mnist_wrm_ep30' #'mnist_wrm_elu_ep42'
    model,_= main.loadCheckpoint(model,filename)
    # print (main.evaluate(model,test_data_loader))
    print (attack_PGM(model,test_data_loader, p=2, epsilon = 0.0, alpha = 0.1))
    print('Accuracy on test data: ',main.evaluate(mnist_WRM,test_data_loader))
        
#%%
if False and __name__=='__main__':
    p=2
    list_errors = []      
    model = Mnist_Estimateur()

    model,_= main.loadCheckpoint(model,'mnist_erm_ep30')
    epsilons, errors = get_errors(model, test_data_loader, p, alpha = MAX_LR0, random=False)
    list_errors.append(errors)
    main.saveCheckpoint(model,list_errors,'mnist_erm_attack_error_list_p2')
    model,_= main.loadCheckpoint(model,'mnist_fgm_ep24')
    epsilons, errors = get_errors(model, test_data_loader, p, alpha = MAX_LR0, random=False)
    list_errors.append(errors)
    main.saveCheckpoint(model,list_errors,'mnist_fgm_attack_error_list_p2')
    model,_= main.loadCheckpoint(model,'mnist_ifgm_ep27') #xiao27 louis30
    epsilons, errors = get_errors(model, test_data_loader, p, alpha = MAX_LR0, random=False)
    list_errors.append(errors)
    main.saveCheckpoint(model,list_errors,'mnist_ifgm_attack_error_list_p2')
    model,_= main.loadCheckpoint(model,'mnist_wrm_ep30') #xiao30 louis27
    epsilons, errors = get_errors(model, test_data_loader, p, alpha = MAX_LR0, random=False)
    list_errors.append(errors)
    main.saveCheckpoint(model,list_errors,'mnist_wrm_attack_error_list_p2')


    labels =['ERM','FGM','IFGM','WRM']
    #labels =['IFGM','WRM']
    fig = plot_attack_error(list_errors,labels, p)

    #array([ 0.0178,  0.0325,  0.0552,  0.0901,  0.1384,  0.203 ,  0.2811,
    #        0.3679,  0.4615,  0.5626,  0.6601,  0.7513,  0.8354,  0.8975])

    #epsilon = []
    #epsilon = np.array(range(1,5))/10
    #errors = [0.01,0.02,0.1,1]
    #for e in epsilon :
    #    plt.plot(epsilon, errors)
    #    plt.yticks([0.01,0.1,1.0])
    #    plt.yscale('log')


#%%
if True and __name__=='__main__':
    # model = Mnist_Estimateur()
    # model,_= main.loadCheckpoint(model,'mnist_wrm_ep30')
    list_rhos=[]
    list_errors = []      

    model,_= main.loadCheckpoint(model,'mnist_erm_ep30')
    gammas, rhos, errors = rho_vs_gamma(model, test_data_loader, MAX_LR0, random=False, get_err=True)
    list_rhos.append(rhos)
    list_errors.append(errors)

    model,_= main.loadCheckpoint(model,'mnist_fgm_ep24')
    gammas, rhos, errors = rho_vs_gamma(model, test_data_loader, MAX_LR0, random=False, get_err=True)
    list_rhos.append(rhos)
    list_errors.append(errors)

    model,_= main.loadCheckpoint(model,'mnist_ifgm_ep30') #xiao27 louis30
    gammas, rhos, errors = rho_vs_gamma(model, test_data_loader, MAX_LR0, random=False, get_err=True)
    list_rhos.append(rhos)
    list_errors.append(errors)

    model,_= main.loadCheckpoint(model,'mnist_wrm_ep27') #xiao30 louis27
    gammas, rhos, errors = rho_vs_gamma(model, test_data_loader, MAX_LR0, random=False, get_err=True)
    list_rhos.append(rhos)
    list_errors.append(errors)

    main.saveCheckpoint(model,list_rhos,'mnist_rho_vs_gamma_list_rhos')
    main.saveCheckpoint(model,list_errors,'mnist_rho_vs_gamma_list_errs')

    # labels =['ERM','FGM','IFGM','WRM']
    # #labels =['IFGM','WRM']
    # fig = plot_attack_error(list_rho,labels, p)

    # plt.plot(gammas,rhos)
    # plt.xlim([0,100])
    # plt.ylim([0.0,1e4])
    # plt.xticks([25,50])
    # plt.xlabel(r"$C_{2}/\gamma_{adv}$")
    # plt.ylabel(r"$\hat\rho_{test}$")
    # plt.title(r"$\hat\rho_{test}$ vs. $1/\gamma_{adv}$")

    # plt.legend()
