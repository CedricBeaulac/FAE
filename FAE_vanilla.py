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
from skfda import representation as representation
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
class FAE_vanilla(nn.Module):
    def __init__(self):
        super(FAE_vanilla, self).__init__()
        self.fc1 = nn.Linear(n_basis, 80)
        self.fc2 = nn.Linear(80, n_rep)
        self.fc3 = nn.Linear(n_rep, 80)
        self.fc4 = nn.Linear(80, n_basis)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        s = self.Project(x, basis_fc)
        t = F.relu(self.fc1(s))
        t = self.dropout(t)
        rep = F.relu(self.fc2(t))
        t = F.relu(self.fc3(rep))
        t = self.dropout(t)
        s_hat = F.relu(self.fc4(t))
        x_hat = self.Revert(s_hat, basis_fc)
        return x_hat, rep

    def Project(self, x, basis_fc):
        # basis_fc: n_time X nbasis
        # x: n_subject X n_time
        s = torch.matmul(x, torch.t(basis_fc))
        return s

    def Revert(self, x, basis_fc):
        f = torch.matmul(x, basis_fc)
        return f

class FAE_vanilla(nn.Module):
    def __init__(self):
        super(FAE_vanilla, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_basis, 80),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(80, n_rep),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_rep, 80),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(80, n_basis),
            nn.ReLU()
        )

    def forward(self, x):
        s = self.Project(x, basis_fc)
        rep = self.encoder(s)
        s_hat = self.decoder(rep)
        x_hat = self.Revert(s_hat, basis_fc)
        return x_hat, rep

    def Project(self, x, basis_fc):
        # basis_fc: n_time X nbasis
        # x: n_subject X n_time
        s = torch.matmul(x, torch.t(basis_fc))
        return s

    def Revert(self, x, basis_fc):
        f = torch.matmul(x, basis_fc)
        return f


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

# Rescale timestamp to [0,1]
tpts_np = np.array(tpts_raw)
#tpts = torch.tensor(np.array(tpts_np))
tpts_rescale = (tpts_np - min(tpts_np)) / np.ptp(tpts_np)
tpts = torch.tensor(np.array(tpts_rescale))
n_tpts = len(tpts)
# tpts = np.linspace(0,1,num=10)

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
    model.train()
    train_loss = 0  # ?
    for i, data in enumerate(train_loader):
        data = data.to(device)
        input = data.type(torch.LongTensor)
        output, rep = model(input.float())
        loss = loss_function(output, input.float())

        optimizer.zero_grad()  # The gradients are set to zero
        loss.backward()  # The gradient is computed and stored.
        optimizer.step()  # .step() performs parameter update
    return loss

def pred(model, data):
    input = data.type(torch.LongTensor)
    output, rep = model(input.float())
    loss = loss_function(output, input.float())
    return output, rep, loss

#####################################
# Model Training
#####################################
# Set up parameters
n_basis = 150
n_rep = 10
# Get basis functions evaluated
bss = representation.basis.BSpline(n_basis=n_basis, order=4)
bss_eval = bss.evaluate(tpts, derivative=0)
basis_fc = torch.from_numpy(bss_eval[:, :, 0]).float()

# Model Initialization
model = FAE_vanilla()
# Validation using MSE Loss function
loss_function = nn.MSELoss()
# Using an Adam Optimizer with lr = 0.1
optimizer = optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)
# Set to CPU/GPU
device = torch.device("cpu")  # (?)should be CUDA when running on the big powerfull server

epochs = 200
outputs = []
reps = []
losses = []

# Train model
for epoch in range(1, epochs + 1):
    loss = train(epoch, loss_function)
    losses.append(loss)
    outputs, reps, pred_loss = pred(model, TestData)
    print(f"Epoch[{epoch}]-loss: {loss:.4f}")
