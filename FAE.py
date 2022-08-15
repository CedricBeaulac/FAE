# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 14:49:58 2022

@author: Sidi Wu and Cédric Beaulac

Functional autoencoder implementation
"""

import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from numpy import *
import pandas as pd

####################################
# FAE: Probabilistic VAE 
# 1 Hidden layer
# Decoder has one hidden layer
####################################
class FAE(nn.Module):
    def __init__(self, lowerep=1):
        super(FAE, self).__init__()

        self.fc1 = nn.Linear(s,)

    def forward(self, x,Bsplines):
        s = self.Deterministic(x,Bsplines)
        xtilde = self.SIFO(s)
        return xtilde

    def Deterministic(self, x,Bsplines):
        s = innert(x,bsplne)
        return s

    def SIFO(self, basiscoef):
        x_tilde = SIFO()
        return xtild


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x,xtild):


    return (x-xtilde)^2/n


def ptrain(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        mux,logvarx, mu, logvar = model(data)
        loss = -ploss_function(mux,logvarx, data, mu, logvar,args.beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    train_loss /= len(train_loader.dataset)            
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss ))
    return(train_loss)
    


