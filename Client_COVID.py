# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:59:05 2020

@author: cdnguyen
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:59:45 2020
https://github.com/SimoneRosset/AUGMENTATION_GAN
@author: cdnguyen
"""
import argparse
import datetime
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython.display import clear_output
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler
now = datetime.datetime.now()
from model_GAN3 import netG, netD

class Client():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #data_dir = '4/3AHDbc50wauQLH_Uxb1lOAm9fH5_self.nz8o_gvOQGzGEebKG_8SK7yvan4div2k' #Just to check
        data_dir = 'CovidDataset'
        
        self.batch_size  = 16
        # Number of training epochs
        self.local_epochs = 2
        #All images will be resized to this size using a transformer.
        #image_size = 64
        self.imageSize = 64
        # Number of channels in the training images. For color images this is 3
        self.nc = 1
        # Size of z latent vector (i.e. size of self.generator input)
        self.nz = 100
        # Size of feature maps in self.generator
        self.ngf = 64
        # Size of feature maps in self.discriminator
        self.ndf = 64
        # No of labels
        self.nb_label = 2
        # Learning rate for optimizers
        lr = 0.0002
        # Beta1 hyperparam for Adam optimizers
        beta1 = 0.5
        # Beta2 hyperparam for Adam optimizers
        beta2 = 0.999
        
        self.real_label = 1.
        self.fake_label = 0.
        # Input to self.generator
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device) #batch of 64
        # Define Loss fuself.nction
        self.s_criterion = nn.BCELoss().to(self.device) #For synthesizing
        self.c_criterion = nn.NLLLoss().to(self.device) #For classification
        
        input = torch.FloatTensor(self.batch_size , 3, self.imageSize, self.imageSize).to(self.device)
        self.noise = torch.FloatTensor(self.batch_size , self.nz, 1, 1).to(self.device)
        self.fixed_noise = torch.FloatTensor(self.batch_size , self.nz, 1, 1).normal_(0, 1).to(self.device)
        self.s_label = torch.FloatTensor(self.batch_size ).to(self.device)
        self.c_label = torch.LongTensor(self.batch_size ).to(self.device)
        
        input = Variable(input)
        self.s_label = Variable(self.s_label)
        self.c_label = Variable(self.c_label)
        self.noise = Variable(self.noise)
        self.fixed_noise = Variable(self.fixed_noise)
        fixed_noise_ = np.random.normal(0, 1, (self.batch_size , self.nz))
        random_label = np.random.randint(0, self.nb_label, self.batch_size )
        #print('fixed label:{}'.format(random_label))
        random_onehot = np.zeros((self.batch_size , self.nb_label))
        random_onehot[np.arange(self.batch_size ), random_label] = 1
        fixed_noise_[np.arange(self.batch_size ), :self.nb_label] = random_onehot[np.arange(self.batch_size )]
        
        fixed_noise_ = (torch.from_numpy(fixed_noise_))
        fixed_noise_ = fixed_noise_.resize_(self.batch_size , self.nz, 1, 1)
        self.fixed_noise.data.copy_(fixed_noise_)
        
        if self.nc==1:
            mu = (0.5)
            sigma = (0.5)
            transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                            transforms.Grayscale(num_output_channels=1),
                                            transforms.Resize((64,64)),
                                            #transforms.Scale(self.imageSize),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mu, sigma)])
        elif self.nc==3:
            mu = (0.5,0.5,0.5)
            sigma = (0.5,0.5,0.5)
            #Originally authors used just scaling
            transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                            transforms.Resize((64,64)),
                                            #transforms.Scale(self.imageSize),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mu, sigma)])
        else:
            print("Tranformation not defined for this option")
        
        
        train_set = datasets.ImageFolder(data_dir, transform=transform)
        
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size =self.batch_size ,
                                                  shuffle=True)
        self.generator = netG(self.nz, self.ngf, self.nc).to(self.device)
        self.discriminator = netD(self.ndf, self.nc, self.nb_label).to(self.device)
        
        # setup optimizer
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.weights_g = []
        self.weights_d = []
        self.losses_g = []
        self.losses_d= []
        self.Loss_D = []
        self.Loss_G = []
        
    def test(predict, labels):
        correct = 0
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)
    
    def local_training(self):
        
        self.discriminator.train()
        self.generator.train()
        for epoch in range(self.local_epochs):
            for i, (img, label) in enumerate(self.train_loader):
                ###########################
                # (1) Update D network
                ###########################
                # train with real
                self.discriminator.zero_grad()
                self.batch_size  = img.size(0)
                input1 = Variable(torch.FloatTensor(self.batch_size , 3, self.imageSize, self.imageSize).to(self.device))
                with torch.no_grad():
                    input1.resize_(img.size()).copy_(img)
                    self.s_label.resize_(self.batch_size ).fill_(self.real_label)
                    self.c_label.resize_(self.batch_size ).copy_(label)
                s_output, c_output = self.discriminator(img)
                s_errD_real =  nn.BCELoss()(s_output, self.s_label)
                c_errD_real = nn.NLLLoss()(c_output, self.c_label)
                errD_real = s_errD_real + c_errD_real
                errD_real.backward()
                D_x = s_output.data.mean()     
                #correct, length = test(c_output, c_label)
        
                # train with fake
                with torch.no_grad():
                    self.noise.resize_(self.batch_size , self.nz, 1, 1)
                    self.noise.normal_(0, 1)
        
                label = np.random.randint(0, self.nb_label, self.batch_size )
                noise_ = np.random.normal(0, 1, (self.batch_size , self.nz))
                label_onehot = np.zeros((self.batch_size , self.nb_label))
                label_onehot[np.arange(self.batch_size ), label] = 1
                noise_[np.arange(self.batch_size ), :self.nb_label] = label_onehot[np.arange(self.batch_size )]
                
                noise_ = (torch.from_numpy(noise_))
                noise_ = noise_.resize_(self.batch_size , self.nz, 1, 1)
                self.noise.data.copy_(noise_)
        
                self.c_label.data.resize_(self.batch_size ).copy_(torch.from_numpy(label))
        
                fake = self.generator(self.noise)
                self.s_label.data.fill_(self.fake_label)
                s_output,c_output = self.discriminator(fake.detach())
                s_errD_fake = self.s_criterion(s_output, self.s_label)
                c_errD_fake = self.c_criterion(c_output, self.c_label)
                errD_fake = s_errD_fake + c_errD_fake
        
                errD_fake.backward()
                D_G_z1 = s_output.data.mean()
                errD = s_errD_real + s_errD_fake
                self.optimizerD.step()
        
                ###########################
                # (2) Update G network
                ###########################
                self.generator.zero_grad()
                self.s_label.data.fill_(self.real_label)  # fake labels are real for self.generator cost
                s_output,c_output = self.discriminator(fake)
                s_errG = self.s_criterion(s_output, self.s_label)
                c_errG = self.c_criterion(c_output, self.c_label)
                
                errG = s_errG + c_errG
                errG.backward()
                D_G_z2 = s_output.data.mean()
                self.optimizerG.step()
                print('Local epoch: %d Loss_D: %.4f Loss_G: %.4f'
                  % (epoch,errD, errG))
    
                # do checkpointing        
            #torch.save(self.discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (os.path.join('.', '0_saved_model'), epoch))
        self.Loss_D.append(errD.item())
        self.Loss_G.append(errG.item())
        self.losses_d.append(errD.item())
        self.losses_g.append(errG.item())
        self.weights_d.append((self.discriminator.state_dict()))
        self.weights_g.append(self.generator.state_dict())
        torch.save(self.generator.state_dict(), '%s/netG_epoch_%d.pth' % (os.path.join('.', '0_saved_model'), epoch))
        #saved_training(Loss_D, Loss_G)
        
       # return self.weights_d, self.weights_g, self.losses_d, self.losses_g
        
    def client_training(self,selected_client):
            print('Local training starts...')
            self.weights_d = []
            self.weights_g = []
            self.losses_d = []
            self.losses_g = []
            for client in selected_client:
                self.local_training()
            return self.weights_d, self.weights_g, self.losses_d, self.losses_g 
    
    def client_update(self,para_global_d1, para_global_g1):
            self.discriminator.load_state_dict(para_global_d1)
            self.generator.load_state_dict(para_global_g1)
    def test_image(self,model):
        PATH = '0_saved_model/netG_epoch_19.pth'
        
        model.load_state_dict(torch.load(PATH))
        loop = 10
            
        fake = model(self.fixed_noise)
        vutils.save_image(fake.data,'%s/fake_samples_epoch_%03d.png' % ('./1_output_images', loop), normalize=True)
            
    def saved_training(self,Loss_D1,Loss_G1):
        dict = {'Loss_D': Loss_D1, 'Loss_G': Loss_G1}  
        df = pd.DataFrame(dict) 
        # saving the dataframe 
        file = 'file1.csv'
        df.to_csv(file) 
        df = pd.read_csv(file)
        z1= df['Loss_D']
        z2= df['Loss_G']
        plt.plot(z1)
        plt.plot(z2)
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.title('Training Accuracy')
        #plt.legend(box_to_aself.nchor=(0.75, 0.95), loc='upper left')
        
        plt.savefig("0_plot/graph1_%s.png" % now.strftime("%H_%M_%S"))
        plt.show()
    
    
    
    #y  = client_training()     
    
    #test_image(self.generator)
