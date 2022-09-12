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
import seaborn as sns
import skfda as fda
from skfda.representation import basis as basis
import scipy
from scipy.interpolate import BSpline
import ignite
import os

#################################################
# FAE: one hidden layer
# Encoder: a added layer for fd representation
# Decoder:
#################################################

#####################################
# Define the vanilla FAE architecture
# Create FAE Class
#####################################
class FeedForward(nn.Module):
    def __init__(self, n_input=5, hidden=[10], n_rep=2, dropout=0.1, activation=F.relu):
        super().__init__()
        self.activation = activation
        dim = [n_input]+hidden+[n_rep]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])

    def forward(self, x):
        for i in range(len(dim)-1):
            x = self.layers[i](x)
            x = self.activation(x)
            if i < (len(self.layers)-1):
                x = self.dropout[i](x)
        return x

class FAE(nn.Module):
    def __init__(self, n_basis=5, basis_type = "BSpline", enc_hidden=[100, 100, 50], n_rep=2,
                 time_grid=None, time_rescale=True,
                 activation=F.relu, dropout=0.1, device=None):
        """
        n_basis: no. of basis functions selected, an integer
        enc_hidden: hidden layers used in the encoder, array of integers
        n_rep: no. of elements in the list of representation, an integer
        time_grid: observed time grid for each subject, array of floats
        activation: activation function used in Forward process
        dropout: dropout rate
        device: device for training
        """
        super(FAE, self).__init__()
        self.n_basis = n_basis
        self.device = device
        self.basis_type = basis_type
        #self.time_rescale = time_rescale

        dnc_hidden = list(reversed(enc_hidden))

        self.encoder = FeedForward(n_input=n_basis, hidden=enc_hidden, n_rep=n_rep,
                                   dropout=dropout, activation=activation)
        self.decoder = FeedForward(n_input=n_rep, hidden=enc_hidden, n_rep=n_basis,
                                   dropout=dropout, activation=activation)

    def forward(self, x):
        # Rescale time grid
        obs_time = np.array(time_grid)
        if time_rescale == True:
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

        s = self.Project(x, basis_fc)
        rep = self.encoder(s)
        s_hat = self.decoder(rep)
        x_hat = self.Revert(s_hat, basis_fc)
        return x_hat, rep

    def Project(self, x, basis_fc):
        """
        Projection function: project discretely-observed functional data to the vector-valued socre layer
        basis_fc: basis functions evaluated at observed time grid
        """
        # basis_fc: n_time X nbasis
        # x: n_subject X n_time
        s = torch.matmul(x, torch.t(basis_fc))
        return s

    def Revert(self, x, basis_fc):
        """
        Reversion function: revert the estimated score to function data
        basis_fc: basis functions evaluated at observed time grid
        """
        f = torch.matmul(x, basis_fc)
        return f

    # def Time_Rescale(self, time_grid):
    #     """
    #     Rescale function: rescale the time grid to the range [0,1]
    #     basis_fc: basis functions evaluated at observed time grid
    #     """
    #     time_rescale = (time_grid - min(time_grid))/np.ptp(time_grid)
    #     #time_grid = torch.tensor(np.array(time_rescale))
    #     return time_rescale

#####################################
# Load Data sets
#####################################
# Import dataset
os.chdir('C:/Users/Sidi/Desktop/FAE_local/tecator')
x_raw = pd.read_csv('Data/tecator.csv')
tpts_raw = pd.read_csv('Data/tecator_tpts.csv')

# Prepare numpy/tensor data
x_np = np.array(x_raw).astype(float)
x = torch.tensor(x_np).float()
tpts = tpts_raw.x.tolist()

# Split training/test set
split.rate = 0.8
TrainData = x[0: round(len(x) * split.rate), :]
TestData = x[round(len(x) * split.rate):, :]

# Define data loaders; DataLoader is used to load the dataset for training
train_loader = torch.utils.data.DataLoader(TrainData, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(TestData)

#####################################
# Define the training procedure
#####################################
# training function
def train(epoch, loss_function):  # do I need to include "loss_function", how about "optimizer"
    FAE_model.train()
    train_loss = 0  # ?
    for i, data in enumerate(train_loader):
        data = data.to(device)
        input = data.type(torch.LongTensor)
        output, rep = FAE_model(input.float())
        loss = loss_function(output, input.float())

        optimizer.zero_grad()  # The gradients are set to zero
        loss.backward()  # The gradient is computed and stored.
        optimizer.step()  # .step() performs parameter update
    return loss


def pred(model, data):
    input = data.type(torch.LongTensor)
    output, rep = FAE_model(input.float())
    loss = loss_function(output, input.float())
    return output, rep, loss


#####################################
# Model Training
#####################################
# Set to CPU/GPU
device = torch.device("cpu")  # (?)should be CUDA when running on the big powerfull server

# Set up parameters
n_basis = 150
basis_type = "BSpline"
encoder_hidden = [80]
n_rep = 10
time_grid = tpts
time_rescale = True
activation_function = F.relu
dropout=0.1


# Model Initialization
FAE_model = FAE(n_basis=n_basis,
                basis_type=basis_type,
                enc_hidden=encoder_hidden,
                n_rep=n_rep,
                time_grid=time_grid,
                time_rescale=time_rescale,
                activation=activation_function,
                dropout=dropout,
                device=device)
# Validation using MSE Loss function
loss_function = nn.MSELoss()
# Using an Adam Optimizer with lr = 0.1
optimizer = optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)

epochs = 200
outputs = []
reps = []
losses = []
for epoch in range(1, epochs + 1):
    loss = train(epoch, loss_function)
    losses.append(loss)
    outputs, reps, pred_loss = pred(model, TestData)
    print(f"Epoch[{epoch}]-loss: {loss:.4f}")

