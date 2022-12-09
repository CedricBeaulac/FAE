from sklearn.datasets import make_classification
import pandas as pd
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
import seaborn as sns
import skfda as fda
from skfda.representation import basis as basis
import scipy
from scipy.interpolate import BSpline
import ignite
import os

## Generate reps & classes
n_sample = 500
n_feature = 5
n_class = 3
class_weight = [.2,.3,.5]
seed(163)
sim_reps, sim_labels = make_classification(n_samples=n_sample,
                                           n_features=n_feature,
                                           n_informative=5,
                                           n_redundant=0,
                                           n_classes=n_class,
                                           weights=class_weight)
sim_input = torch.tensor(sim_reps).float()

## Create a decoder(NN) generator
# Define classes
class FeedForward(nn.Module):
    def __init__(self, n_input=5, hidden=[10], n_rep=2, dropout=0, activation=F.relu, decoder=False):
        super().__init__()
        self.activation = activation
        self.dim = [n_input]+hidden+[n_rep]
        self.layers = nn.ModuleList([nn.Linear(self.dim[i-1], self.dim[i]) for i in range(1, len(self.dim))])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])
        self.decoder = decoder

    def forward(self, x):
        if self.decoder == True:
            for i in range(len(self.dim)-2):
                x = self.layers[i](x)
                x = self.activation(x)
                if i < (len(self.layers)-2):
                    x = self.dropout[i](x)
            x = self.layers[len(self.dim)-2](x)
            x = self.activation(x)
        else:
            for i in range(len(self.dim)-1):
                x = self.layers[i](x)
                x = self.activation(x)
                if i < (len(self.layers)-1):
                    x = self.dropout[i](x)
        return x

class NN_generator(nn.Module):
    def __init__(self, n_basis=20, basis_type = "BSpline", dnc_hidden=[10], n_rep=5,
                 time_grid=None, time_rescale=True, activation=nn.ReLU(), dropout=0,
                 weight_std = None, device=None):
        """
        n_basis: no. of basis functions selected, an integer
        enc_hidden: hidden layers used in the encoder, array of integers
        n_rep: no. of elements in the list of representation, an integer
        time_grid: observed time grid for each subject, array of floats
        activation: activation function used in Forward process
        dropout: dropout rate
        device: device for training
        """
        super(NN_generator, self).__init__()
        self.n_basis = n_basis
        self.device = device
        self.basis_type = basis_type
        self.time_rescale = time_rescale
        self.time_grid = time_grid

        self.decoder = FeedForward(n_input=n_rep, hidden=dnc_hidden, n_rep=n_basis,
                                   dropout=dropout, activation=activation, decoder=True)

        # initialize the weights to a specified, constant value
        if (weight_std is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=weight_std)

    def forward(self, x):
        # Rescale time grid
        obs_time = np.array(self.time_grid)
        if self.time_rescale == True:
            obs_time = (obs_time - min(obs_time))/np.ptp(obs_time)
        # produce basis functions accordingly
        if self.basis_type == "BSpline":
            bss = basis.BSpline(n_basis=self.n_basis, order=4)
        if self.basis_type == "Fourier":
            bss = basis.Fourier(domain_range=(float(min(obs_time)), float(max(obs_time))),
                                n_basis = self.n_basis)
        # Evalute basis functions at observed time grid
        bss_eval = bss.evaluate(obs_time, derivative=0)
        basis_fc = torch.from_numpy(bss_eval[:, :, 0]).float()

        s_hat = self.decoder(x)
        x_hat = self.Revert(s_hat, basis_fc)
        return x_hat, s_hat

    def Revert(self, x, basis_fc):
        """
        Reversion function: revert the estimated score to function data
        basis_fc: basis functions evaluated at observed time grid
        """
        f = torch.matmul(x, basis_fc)
        return f

# Set to CPU/GPU
device = torch.device("cpu")  # (?)should be CUDA when running on the big powerfull server

# Set up parameters
n_basis = 10
basis_type = "BSpline"
decoder_hidden = [10]
n_rep = n_feature
time_grid = np.arange(1, 50 ,1/49)
time_rescale = True
activation_function = nn.Sigmoid()
dropout=0

# Model Initialization
sim_decoder = NN_generator(n_basis=n_basis,
                           basis_type=basis_type,
                           dnc_hidden=decoder_hidden,
                           n_rep=n_rep,
                           time_grid=time_grid,
                           time_rescale=time_rescale,
                           activation=activation_function,
                           dropout=dropout,
                           weight_std=2,
                           device=device)

# Get simulated X(t) for observed t
sim_decoder.eval()
sim_x, coef = sim_decoder(sim_input)

# Plot simulated curves
sim_x_plt = sim_x.detach().numpy()
plt.figure(1)
for i in range(0, len(sim_x_plt)):
# for m in id_plt:
    plt.plot(time_grid, sim_x_plt[i])
plt.title("Simulated Curves")
plt.show()
