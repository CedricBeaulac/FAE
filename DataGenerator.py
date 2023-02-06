# Import modules
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pandas as pd
import numpy as np
from numpy import *
import skfda as fda
from skfda import representation as representation

# nc: number of data points per *class*, classes: number of classes/cluster, noise:std of the normal noise
def DataGenerator(nc=250, tpts = np.linspace(0,1,21),classes=4,noise=2):
    n = nc*classes
    n_basis=20
    bss = representation.basis.BSpline(n_basis=n_basis, order=4)
    bss_eval = bss.evaluate(tpts, derivative=0)
    bss_fct = bss_eval[:, :, 0]
    C = np.random.random((classes,n_basis))*10
    CC =np.tile(C,(nc,1))
    Noise = np.reshape(np.random.normal(0,noise,n*n_basis),(n,n_basis))
    X = CC+Noise
    data = matmul(X, bss_fct)  
    bss_evalc = bss.evaluate(np.linspace(0,1,201), derivative=0)
    bss_fctc = bss_eval[:, :, 0]
    curves = matmul(C,bss_fctc)
    return(data,transpose(curves)) 

def SmoothDataGenerator(nc=250, tpts = np.linspace(0,1,21),classes=4,noise=2):
    n = nc*classes
    n_basis=20
    bss = representation.basis.BSpline(n_basis=n_basis, order=4)
    bss_eval = bss.evaluate(tpts, derivative=0)
    bss_fct = bss_eval[:, :, 0]
    Start= np.random.random(classes)
    Changes = (2*np.random.binomial(1, 0.5, size=(classes,n_basis)).astype(np.float64))-1
    Changes[:,0] += Start
    for j in range(1,n_basis):
        Changes[:,j] += Changes[:,j-1]
    CC =np.tile(Changes,(nc,1))
    Noise = np.reshape(np.random.normal(0,noise,n*n_basis),(n,n_basis))
    X = CC+Noise
    data = matmul(X, bss_fct)  
    bss_evalc = bss.evaluate(np.linspace(0,1,201), derivative=0)
    bss_fctc = bss_evalc[:, :, 0]
    curves = matmul(Changes,bss_fctc)
    return(data, transpose(curves))
