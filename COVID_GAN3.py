# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:59:45 2021

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#data_dir = '4/3AHDbc50wauQLH_Uxb1lOAm9fH5_nZ8o_gvOQGzGEebKG_8SK7yvan4div2k' #Just to check
data_dir =  'CovidDataset'  #'covid_alone_synthetic' #'Pneumonia' #'covid_alone'
batch_size = 16
# Number of training epochs
num_epochs = 500
#All images will be resized to this size using a transformer.
#image_size = 64
imageSize = 64
# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# No of labels
nb_label = 2
# Learning rate for optimizers
lr = 0.001
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Beta2 hyperparam for Adam optimizers
beta2 = 0.999

real_label = 1.
fake_label = 0.
# Input to generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device) #batch of 64
# Define Loss function
s_criterion = nn.BCELoss().to(device) #For synthesizing
c_criterion = nn.NLLLoss().to(device) #For classification

input = torch.FloatTensor(batch_size, 3, imageSize, imageSize).to(device)
noise = torch.FloatTensor(batch_size, nz, 1, 1).to(device)
fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).to(device)
s_label = torch.FloatTensor(batch_size).to(device)
c_label = torch.LongTensor(batch_size).to(device)

input = Variable(input)
s_label = Variable(s_label)
c_label = Variable(c_label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
fixed_noise_ = np.random.normal(0, 1, (batch_size, nz))
random_label = np.random.randint(0, nb_label, batch_size)
#print('fixed label:{}'.format(random_label))
random_onehot = np.zeros((batch_size, nb_label))
random_onehot[np.arange(batch_size), random_label] = 1
fixed_noise_[np.arange(batch_size), :nb_label] = random_onehot[np.arange(batch_size)]

fixed_noise_ = (torch.from_numpy(fixed_noise_))
fixed_noise_ = fixed_noise_.resize_(batch_size, nz, 1, 1)
fixed_noise.data.copy_(fixed_noise_)

if nc==1:
    mu = (0.5)
    sigma = (0.5)
    transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((64,64)),
                                    #transforms.Scale(imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mu, sigma)])
elif nc==3:
    mu = (0.5,0.5,0.5)
    sigma = (0.5,0.5,0.5)
    #Originally authors used just scaling
    transform = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                    transforms.Resize((64,64)),
                                    #transforms.Scale(imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mu, sigma)])
else:
    print("Tranformation not defined for this option")
train_set = datasets.ImageFolder(data_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)
generator = netG(nz, ngf, nc).to(device)
discriminator = netD(ndf, nc, nb_label).to(device)

# setup optimizer
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)

Loss_D = []
Loss_G = []
Accu_D = []

def training():
    for epoch in range(num_epochs+1):
        for i, (img, label) in enumerate(train_loader):
            ###########################
            # (1) Update D network
            ###########################
            # train with real
            discriminator.zero_grad()
            batch_size = img.size(0)
            input1 = Variable(torch.FloatTensor(batch_size, 3, imageSize, imageSize).to(device))
            with torch.no_grad():
                input1.resize_(img.size()).copy_(img)
                s_label.resize_(batch_size).fill_(real_label)
                c_label.resize_(batch_size).copy_(label)
            s_output, c_output = discriminator(img)
            s_errD_real =  nn.BCELoss()(s_output, s_label)
            c_errD_real = nn.NLLLoss()(c_output, c_label)
            errD_real = s_errD_real + c_errD_real
            errD_real.backward()
            D_x = s_output.data.mean()     
            #correct, length = test(c_output, c_label)
    
            # train with fake
            with torch.no_grad():
                noise.resize_(batch_size, nz, 1, 1)
                noise.normal_(0, 1)
    
            label = np.random.randint(0, nb_label, batch_size)
            noise_ = np.random.normal(0, 1, (batch_size, nz))
            label_onehot = np.zeros((batch_size, nb_label))
            label_onehot[np.arange(batch_size), label] = 1
            noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]
            
            noise_ = (torch.from_numpy(noise_))
            noise_ = noise_.resize_(batch_size, nz, 1, 1)
            noise.data.copy_(noise_)
    
            c_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
    
            fake = generator(noise)
            s_label.data.fill_(fake_label)
            s_output,c_output = discriminator(fake.detach())
            s_errD_fake = s_criterion(s_output, s_label)
            c_errD_fake = c_criterion(c_output, c_label)
            errD_fake = s_errD_fake + c_errD_fake
    
            errD_fake.backward()
            D_G_z1 = s_output.data.mean()
            errD = s_errD_real + s_errD_fake
            optimizerD.step()
    
            ###########################
            # (2) Update G network
            ###########################
            generator.zero_grad()
            s_label.data.fill_(real_label)  # fake labels are real for generator cost
            s_output,c_output = discriminator(fake)
            s_errG = s_criterion(s_output, s_label)
            c_errG = c_criterion(c_output, c_label)
            
            errG = s_errG + c_errG
            errG.backward()
            D_G_z2 = s_output.data.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, i, len(train_loader),
                 errD, errG, D_x, D_G_z1, D_G_z2))
            if (epoch) % 50 == 0:
                vutils.save_image(img,
                        '%s/real_samples.png' % './0_output_images', normalize=True)
                #fake = netG(fixed_cat)
                fake = generator(fixed_noise)
                vutils.save_image(fake.data,'%s/fake_samples_epoch_%03d.png' % ('./0_output_images', epoch), normalize=True)
           
            # do checkpointing        
        #torch.save(discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (os.path.join('.', '0_saved_model'), epoch))
        Loss_D.append(errD.item())
        Loss_G.append(errG.item())
        Accu_D.append(D_x.item())
            
    torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (os.path.join('.', '0_saved_model'), epoch))
    saved_training(Loss_D, Loss_G)

def test_image(model):
    PATH = '0_saved_model/netG_epoch_19.pth'
    
    model.load_state_dict(torch.load(PATH))
    loop = 10
        
    fake = model(fixed_noise)
    vutils.save_image(fake.data,'%s/fake_samples_epoch_%03d.png' % ('./1_output_images', loop), normalize=True)
        
def test2(generator, discriminator, num_epochs, loader):
    print('Testing Block.........')
    now = datetime.datetime.now()
    #g_losses = metrics['train.G_losses'][-1]
    #d_losses = metrics['train.D_losses'][-1]
    path='0_output_images'
    try:
      os.mkdir(os.path.join(path))
    except Exception as error:
      print(error)

    real_batch = next(iter(loader))
    
    test_img_list = []
    test_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    test_fake = generator(test_noise).detach().cpu()
    test_img_list.append(vutils.make_grid(test_fake, padding=2, normalize=True))

    fig = plt.figure(figsize=(15,15))
    ax1 = plt.subplot(1,2,1)
    ax1 = plt.axis("off")
    ax1 = plt.title("Real Images")
    ax1 = plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    ax2 = plt.subplot(1,2,2)
    ax2 = plt.axis("off")
    ax2 = plt.title("Fake Images")
    ax2 = plt.imshow(np.transpose(test_img_list[-1],(1,2,0)))
    #ax2 = plt.show()
    #fig.savefig('%s/image_%.3f_%.3f_%d_%s.png' %
    #                (path, g_losses, d_losses, num_epochs, now.strftime("%Y-%m-%d_%H:%M:%S")))
    
def saved_training(Loss_D1,Loss_G1):
    dict = {'Loss_D': Loss_D1, 'Loss_G': Loss_G1 }  
    df = pd.DataFrame(dict) 
    # saving the dataframe 
    file = 'file5007.csv'
    df.to_csv(file) 
    df = pd.read_csv(file)
    z1= df['Loss_D']
    z2= df['Loss_G']
    plt.plot(z1)
    plt.plot(z2)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title('Training Accuracy')
    plt.legend(bbox_to_anchor=(0.75, 0.95), loc='upper left')
    
    plt.savefig("0_plot/graph1_%s.png" % now.strftime("%H_%M_%S"))
    plt.show()
    
y  = training()     

#test_image(generator)
