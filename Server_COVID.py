# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:59:45 2020

This is the FL-GAN code for COVID-19 data augmentation, as part of the paper publication: 
    "Federated Learning for COVID-19 Detection with Generative Adversarial Networks in Edge Cloud Computing", 
    IEEE Internet of Things Journal, Nov. 2021, Accepted (https://ieeexplore.ieee.org/abstract/document/9580478)
@author: Dinh C. Nguyen 
"""
import pandas as pd 
import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
os.getcwd()
from Client_COVID import Client


class Server:
    def __init__(self):
        # print(self.model)
        self.clients = None
        self.client_index = []
        self.target_round = -1
        self.global_round = 10
        self.clients_total = 50
        self.frac = 0.1
        self.selected = 0
        self.loss_d_aver = 0
        self.loss_g_aver = 0
        self.client = Client()
    def run(self):
        print('GLobal Federated Learning start: ')
        for epoch in (range(1, self.global_round)):
            para_collector_g = []
            para_collector_d = []
            self.selected = self.clients_selection()
            print(self.selected)
            selected_user_length = len(self.selected)
            selected1 = [10, 20]
            weight_d, weight_g, loss_d, loss_g = self.client.client_training(selected1)

            self.loss_d_aver = sum(loss_d) / selected_user_length
            self.loss_g_aver = sum(loss_g) / selected_user_length
            
            print('Global epoch: {} \tloss_D: {:.3f} \tloss_G: {:.3f}'.format(
               epoch, self.loss_d_aver, self.loss_g_aver))
   
            para_global_d = self.FedAvg(weight_d)
            para_global_g = self.FedAvg(weight_g)

            self.client.client_update(para_global_d, para_global_g)
            
    def connect_clients(self):
        client_id = [i for i in range(0, self.clients_total)]
        self.client_index = client_id
        return self.client_index 
    def clients_selection(self):
        n_clients = max(1, int(self.clients_total * self.frac))
        self.client_index = self.connect_clients()
        training_clients = np.random.choice(self.client_index, n_clients, replace=False)
        return training_clients 
    
    def FedAvg(self,weight):
        w_avg = weight[0]
        for key in w_avg:
            for i in range(len(weight)): 
                w_avg[key] = w_avg[key] + weight[i][key]
            w_avg[key] = w_avg[key] / float(len(weight))
        return w_avg

    
if __name__ == '__main__':
    running = Server().run()
    
    
    
    

