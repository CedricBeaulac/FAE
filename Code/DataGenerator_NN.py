"""
This script contains the data generator for simulation data sets used in the manuscript "Autoencoders for Discrete Functional Data Representation Learning and Smoothing".

@author: Sidi Wu
"""
# Import modules
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
import random
from random import seed

# Define the data generator function
def DataGenerateor_Dist_NN(n_sample_per_class=200, n_class=3, n_rep=5, mean=None, cov=None,
                           n_basis = 10, basis_type = "BSpline", decoder_hidden = [10],
                           time_grid = np.linspace(0,1,21),activation_function = nn.Sigmoid(), noise=0):
    """
    :param n_sample_per_class: number of samples per class
    :param n_class: number of class(es)
    :param n_rep: number of representations/features
    :param mean: mean matrix of the Gausssian mixture model
    :param cov: covariance matrix of the Gaussian mixture model
    :param n_basis: number of basis functions for generating the functional observations
    :param basis_type: type of basis functions for generating the functional observations
    :param decoder_hidden: # of hidden layers and nodes for the decoder generator
    :param time_grid: customized time grid with discrete observations
    :param activation_function: activation function for the decoder generator
    :param noise: observation error/noise
    :return: simulated discrete functional observations (sim_x), simulated discrete functional observations with observations errors (sim_x_noise),
    simulated labels (sim_labels), and simulated representations/features (sim_reps)
    """
    ## Generate reps & classes
    reps = []
    labels = []
    for i in range(n_class):
        reps.extend(np.random.multivariate_normal(mean[i], cov[i], n_sample_per_class))
        labels.extend([i+1]*n_sample_per_class)

    sim_reps = np.array(reps)
    sim_labels = np.array(labels)
    sim_input = torch.tensor(sim_reps).float()

    # Set up NN for generating data
    class FeedForward(nn.Module):
        def __init__(self, n_input=5, hidden=[10], n_rep=2, dropout=0, activation=F.relu, decoder=False):
            super().__init__()
            self.activation = activation
            self.dim = [n_input] + hidden + [n_rep]
            self.layers = nn.ModuleList([nn.Linear(self.dim[i - 1], self.dim[i]) for i in range(1, len(self.dim))])
            # self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])
            self.decoder = decoder

        def forward(self, x):
            if self.decoder == True:
                for i in range(len(self.dim) - 2):
                    x = self.layers[i](x)
                    x = self.activation(x)
                    # if i < (len(self.layers)-2):
                    #     x = self.dropout[i](x)
                x = self.layers[len(self.dim) - 2](x)
                x = self.activation(x)
            else:
                for i in range(len(self.dim) - 1):
                    x = self.layers[i](x)
                    x = self.activation(x)
                    # if i < (len(self.layers)-1):
                    #     x = self.dropout[i](x)
            return x

    class NN_generator(nn.Module):
        def __init__(self, n_basis=20, basis_type="BSpline", dnc_hidden=[10], n_rep=5,
                     time_grid=None, time_rescale=True, activation=nn.ReLU(),
                     dropout=0,
                     weight_std=None, noise=0, device=None):
            """
            n_basis: no. of basis functions selected, an integer
            enc_hidden: hidden layers used in the encoder, array of integers
            n_rep: no. of elements in the list of representation, an integer
            time_grid: observed time grid for each subject, array of floats
            activation: activation function used in Forward process
            dropout: dropout rate
            weight_std: sd for the normally distributed inital network weights
            noise: sd of the normally distributed observation error
            device: device for training
            """
            super(NN_generator, self).__init__()
            self.n_basis = n_basis
            self.device = device
            self.basis_type = basis_type
            self.time_rescale = time_rescale
            self.time_grid = time_grid
            self.noise = noise

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
                obs_time = (obs_time - min(obs_time)) / np.ptp(obs_time)
            # produce basis functions accordingly
            if self.basis_type == "BSpline":
                bss = basis.BSpline(n_basis=self.n_basis, order=4)
            if self.basis_type == "Fourier":
                bss = basis.Fourier(domain_range=(float(min(obs_time)), float(max(obs_time))),
                                    n_basis=self.n_basis)
            # Evalute basis functions at observed time grid
            bss_eval = bss.evaluate(obs_time, derivative=0)
            basis_fc = torch.from_numpy(bss_eval[:, :, 0]).float()

            c_hat = self.decoder(x)
            x_hat = self.Revert(c_hat, basis_fc)
            x_hat_noise = x_hat + 1/50*torch.tensor(np.reshape(np.random.normal(0, self.noise, torch.numel(x_hat)),
                                                          x_hat.shape)).float()

            return x_hat, x_hat_noise, c_hat

        def Revert(self, x, basis_fc):
            f = torch.matmul(x, basis_fc)
            return f

    # Set to CPU/GPU
    device = torch.device("cpu")

    # Model Initialization
    sim_decoder = NN_generator(n_basis=n_basis,
                               basis_type=basis_type,
                               dnc_hidden=decoder_hidden,
                               n_rep=n_rep,
                               time_grid=time_grid,
                               time_rescale=False,
                               activation=activation_function,
                               weight_std=2,
                               noise=noise,
                               device=device)

    # Get simulated X(t) for observed t
    sim_decoder.eval()
    sim_x, sim_x_noise, coef = sim_decoder(sim_input)

    return (sim_x, sim_x_noise, sim_labels, sim_reps)


# An example for geenrating a data set for Scenario 1.2 & 2.1
########
class_size = 2000
n_class=3
n_rep = 5
mean_matrix = [[0,1,0,1,0], [-3,-3,-3,-3,-3], [5,5,5,5,5]]
cov_matrix = [[[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]],
       [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]],
       [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]]
n_basis = 10
time_grid = np.linspace(0, 1, 51)
sim_x_dist, sim_x_noise_dist, sim_labels_dist, sim_reps_dist  = DataGenerateor_Dist_NN(n_sample_per_class=class_size, n_class=n_class,
                                                                        n_rep=n_rep, mean=mean_matrix, cov=cov_matrix,
                                                                        n_basis = n_basis , basis_type = "BSpline",
                                                                        decoder_hidden = [20],
                                                                        time_grid = time_grid,
                                                                        activation_function = nn.Sigmoid(),
                                                                        noise=2)

