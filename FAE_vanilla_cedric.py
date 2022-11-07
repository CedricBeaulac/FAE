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
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import skfda as fda
from skfda import representation as representation
from skfda.exploratory.visualization import FPCAPlot
# from skfda.exploratory.visualization import FPCAPlot
# from skfda.preprocessing.dim_reduction import FPCA
# from skfda.representation.basis import BSpline, Fourier, Monomial
import scipy
from scipy.interpolate import BSpline
import ignite
#import os
import os
import sklearn
from sklearn.decomposition import PCA

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
    def __init__(self, constant_weight=None):
        super(FAE_vanilla, self).__init__()
        self.fc1 = nn.Linear(n_basis, n_rep,bias=False)
        #self.fc2 = nn.Linear(80, n_rep)
        self.fc3 = nn.Linear(n_rep, n_basis,bias=False)
        #self.fc4 = nn.Linear(80, n_basis)
        #self.dropout = nn.Dropout(0.1)
        self.activation = nn.Identity()
        # initialize the weights to a specified, constant value
        if (constant_weight is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0,std=constant_weight)
                    #nn.init.constant_(m.bias, 0)
    def forward(self, x, basis_fc):
        s = self.Project(x, basis_fc)
        rep = self.activation(self.fc1(s))
        #t = self.dropout(t)
        #rep = self.activation(self.fc2(t))
        s_hat = self.activation(self.fc3(rep))
        #t = self.dropout(t)
        #s_hat = self.activation(self.fc4(t))
        x_hat = self.Revert(s_hat, basis_fc)
        return x_hat, rep, s, s_hat
    def Project(self, x, basis_fc):
        # basis_fc: n_time X nbasis
        # x: n_subject X n_time
        s = torch.matmul(x, torch.t(basis_fc))
        return s
    def Revert(self, x, basis_fc):
        f = torch.matmul(x, basis_fc)
        return f
    
    
class FAE_vanilla2(nn.Module):
    def __init__(self, basis_fc,STD=None):
        super(FAE_vanilla2, self).__init__()
        self.fc1 = nn.Linear(n_tpts, 80)
        self.fc2 = nn.Linear(80, n_rep)
        self.fc3 = nn.Linear(n_rep, n_basis)
        self.activation = nn.Tanh()
        if (STD is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0,std=STD)
    def forward(self, x, basis_fc):
        l1 = self.activation(self.fc1(x))
        rep = self.activation(self.fc2(l1))
        w = self.fc3(rep)
        x_hat = self.Revert(w, basis_fc)
        return x_hat, rep, w
    def Revert(self, x, basis_fc):
        f = torch.matmul(x, basis_fc)
        return f

class FAE_vanilla3(nn.Module):
    def __init__(self, basis_fc,STD=None):
        super(FAE_vanilla3, self).__init__()
        self.fc1 = nn.Linear(n_tpts, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, n_basis)
        self.activation = nn.Tanh()
        if (STD is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0,std=STD)
    def forward(self, x, basis_fc):
        l1 = self.activation(self.fc1(x))
        l2 = self.activation(self.fc2(l1))
        c = self.fc3(l2)
        x_hat = self.Revert(c, basis_fc)
        return x_hat, rep, c
    def Revert(self, x, basis_fc):
        f = torch.matmul(x, basis_fc)
        return f


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(n_tpts, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 100)
        self.fc4 = nn.Linear(100, n_tpts)
        self.activation = nn.ReLu()
    def forward(self, x):
        h1 = self.activation(self.fc1(x))
        h2 = self.activation(self.fc2(h1))
        h3 = self.activation(self.fc3(h2))
        x_hat = self.fc4(h3)
        
        return x_hat

    
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
os.chdir('C:/FAE')
# Dataset: tecator
x_raw = pd.read_csv('Datasets/tecator/tecator.csv')
tpts_raw = pd.read_csv('Datasets/tecator/tecator_tpts.csv')
# Dataset: pinch
# x_raw = pd.read_csv('Datasets/pinch/pinch.csv')
# tpts_raw = pd.read_csv('Datasets/pinch/pinch_tpts.csv')

# Prepare numpy/tensor data
x_np = np.array(x_raw).astype(float)
x = torch.tensor(x_np).float()
x = x - torch.mean(x,0)

# Rescale timestamp to [0,1]
tpts_np = np.array(tpts_raw)
#tpts = torch.tensor(np.array(tpts_np))
tpts_rescale = (tpts_np - min(tpts_np)) / np.ptp(tpts_np)
tpts = torch.tensor(np.array(tpts_rescale))
n_tpts = len(tpts)
# tpts = np.linspace(0,1,num=10)

# Split training/test set
split.rate = 1
TrainData = x[0: round(len(x) * split.rate), :]
TestData = x[round(len(x) * split.rate):, :]

# Define data loaders; DataLoader is used to load the dataset for training
train_loader = torch.utils.data.DataLoader(TrainData, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(TestData)

#####################################
# Define the training procedure
#####################################
# training function
def train(epoch, n_basis, n_rep, lamb=0):  # do I need to include "loss_function", how about "optimizer"
    # It depends if you define train locally or not
    model.train()
    train_loss = 0
    #score_loss = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # The gradients are set to zero
        # data = data.to(device)
        # input = data.type(torch.LongTensor)
        input = data.to(device)
        out,rep,w = model(input.float(),basis_fc) # inputs should matches the inputs in forward function?
        ## Loss on the score layers (network output layer)
        #score_loss += loss_function(s, s_hat)
        ## Loss for back-propagation
        # Penalty term
        penalty = 0
        #for j in range(2, n_basis):
            #delta_c = model.fc3.weight[j,:]-2*model.fc3.weight[j-1,:]+model.fc3.weight[j-2,:]
            #delta_c = w[:,j]-2*w[:,j-1]+w[:,j-2]
            #penalty += torch.mean(delta_c)
        bss_deriv = bss.evaluate(np.arange(0,1,0.001), derivative=2)
        bss_deriv = torch.from_numpy(bss_deriv[:, :, 0]).float()
        delta_c = torch.matmul(w, bss_deriv)
        penalty += torch.mean(delta_c)
        loss = loss_function(out, input.float()) + lamb*penalty # Maybe add score_loss as well?
        # loss = loss_function(s, s_hat)
        loss.backward()  # The gradient is computed and stored.
        optimizer.step()  # .step() performs parameter update
        train_loss += loss
    return train_loss # we need train_loss, instead of loss, for plotting


def pred(model, data):
    model.eval()
    # input = data.type(torch.LongTensor)
    input = data.to(device)
    output, rep, s, s_hat = model(input.float(), basis_fc)
    loss = loss_function(output, input.float())
    score_loss = loss_function(s, s_hat)
    return output, rep, loss, score_loss

#####################################
# Model Training
#####################################
# Set up parameters
n_basis = 200
n_rep = 25
lamb = 1e-5
# Get basis functions evaluated
bss = representation.basis.BSpline(n_basis=n_basis, order=4)
bss_eval = bss.evaluate(tpts, derivative=0)
basis_fc = torch.from_numpy(bss_eval[:, :, 0]).float()

# Model Initialization
model = FAE_vanilla3(basis_fc,STD=0.1)
# Validation using MSE Loss function
loss_function = nn.MSELoss()
# Using an Adam Optimizer with lr = 0.1
#optimizer = optim.Adam(model.parameters(),lr=1e-4)
# Using an SGD Optimizer with lr = 0.1
#optimizer = optim.SGD(model.parameters(), lr=1e-4)
# Using an ASGD Optimizer
optimizer = optim.ASGD(model.parameters())
# Set to CPU/GPU
device = torch.device("cpu")  # (?)should be CUDA when running on the big powerfull server

epochs = 10000
outputs = []
reps = []
losses = []
score_losses = []

# Train model
for epoch in range(1, epochs + 1):
    loss = train(epoch, n_basis=n_basis, n_rep=n_rep, lamb=lamb)
    losses.append(loss.detach().numpy())
    #score_losses.append(score_loss.detach().numpy())
    #outputs, reps, pred_loss, pred_score_loss = pred(model, TrainData)
    if epoch % 25 ==0:
        print(f"Epoch[{epoch}]-loss: {loss:.7f}")


# Debug by looking at loss
plt.plot(losses[1000:epoch], label = "train_loss")
#plt.plot(score_losses, label = "score_loss")
plt.show()

plt.legend()
plt.close()

# Debug by looking at the FAE, layer by layer

input = x[0:5,:]
#s = model.Project(input,basis_fc)
#rep = model.activation(model.fc1(s))
#s_hat = model.activation(model.fc3(rep))
out,rep,w = model(input,basis_fc)

input_plt = input.detach().numpy()
output_plt = out.detach().numpy()
plt.plot(tpts, input_plt[0])
plt.plot(tpts, output_plt[0])
plt.show()

input_plt = input.detach().numpy()
plt.figure(1)
for m in range(0, len(input_plt)):
    plt.plot(tpts, input_plt[m])
plt.title("Input Curves")
plt.show()
#plt.close()

output_plt = out.detach().numpy()
plt.figure(2)
for m in range(0, len(output_plt)):
    plt.plot(tpts, output_plt[m])
plt.title("Output Curves")
plt.show()

plt.plot(input[0,:].detach().numpy(), label = "Input")
#plt.plot(tpts, output[0,:].detach().numpy(), label = "Output")
#plt.legend()
plt.show()
plt.close()

plt.figure(3, figsize=(10, 20))
plt.subplot(211)
for m in range(0, len(input_plt)):
    plt.plot(tpts, input_plt[m])
plt.title("Input Curves")
plt.subplot(212)
for m in range(0, len(output_plt)):
    plt.plot(tpts, output_plt[m])
plt.title("Output Curves")
plt.show()

#####################################
# Principals components and representation
#####################################

#c's are estimated as weight of the fc1 function
c1 = model.fc1.weight[0].detach()
c2 = model.fc1.weight[1].detach()
#c1 = model.encoder[0].weight.detach()
pc1 = torch.matmul(c1,basis_fc).numpy()
pc2 = torch.matmul(c2,basis_fc).numpy()

plt.plot(tpts, pc1, label='FPC1-FAE')
#plt.plot(tpts, pc2, label='FPC2-FAE')
plt.xlabel('time grid')
plt.legend()
plt.show()

#####################################
# AE Tests
#####################################

# Model Initialization
model = AE()
# Validation using MSE Loss function
loss_function = nn.MSELoss()
# Using an Adam Optimizer with lr = 0.1
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)
# Set to CPU/GPU
device = torch.device("cpu")  # (?)should be CUDA when running on the big powerfull server

epochs = 2000
outputs = []
reps = []
losses = []

# Train model
for epoch in range(1, epochs + 1):
    loss = train(epoch)
    losses.append(loss.detach().numpy())
    #outputs, reps, pred_loss = pred(model, TestData)
    if epoch % 25 ==0:
        print(f"Epoch[{epoch}]-loss: {loss:.4f}")


# Debug by looking at loss
plt.plot(losses)
plt.show()

input = TrainData[0:5,:]
output = model(input)

plt.plot(input[1,:].detach().numpy())
plt.plot(output[1,:].detach().numpy())
plt.show()
plt.close()

#c's are estimated as weight of the fc3 function
c1_hat = model.fc3.weight[:,0].detach()
c2_hat = model.fc3.weight[:,1].detach()
#c1 = model.encoder[0].weight.detach()
pc1_hat = torch.matmul(c1_hat,basis_fc).numpy()
pc2_hat = torch.matmul(c2_hat,basis_fc).numpy()

plt.plot(tpts, pc1_hat, label="FPC1'-FAE")
#plt.plot(tpts, pc2_hat, label="FPC2'-FAE")
plt.xlabel('time grid')
plt.legend()
plt.show()
plt.close()

# Representatives, = FPC scores
reps_bss = reps.detach().numpy()
# reps_gt2 = reps_fpc[reps_fpc > 2]
# np.where(reps_fpc > 2)[0]
# [i for i, x in enumerate(reps_fpc > 2) if x]


#####################################
# Perform FPCA
#####################################
tpts_fd = tpts.numpy().flatten()
fd = representation.grid.FDataGrid(x.numpy(), tpts_fd)
basis_fd = fd.to_basis(bss)
fpca_basis = fda.preprocessing.dim_reduction.feature_extraction.FPCA(n_components=n_rep)
# Get FPCs
#fpca_basis_fd = fpca_basis.fit(fd)
fpca_basis = fpca_basis.fit(basis_fd)
fpca_basis.components_.plot()


# Get FPC scores
fpc_scores = fpca_basis.transform(basis_fd)
# Get mean function
fpca_basis.mean_.plot()
#fpca_basis.singular_values_

plt.figure(2)
fpca_basis.components_[0].plot(label = 'FPC1-FPCA')
plt.plot(tpts, pc1, label = "FPC1-FAE")
plt.title(f"Basis#={n_basis}, FPC#={n_rep}, lamb={lamb}")
plt.legend()
plt.show()
plt.close()

fpca_basis.components_[1].plot(label = 'FPC2-FPCA')
plt.plot(tpts, pc2, label = "FPC2-FAE")
plt.title(f"Basis#={n_basis}, FPC#={n_rep}")
plt.legend()
plt.show()
plt.close()

# PCA on scores s_m
s = model.Project(TrainData,basis_fc).detach().numpy()
pca = PCA(n_components=1)
pca.fit(s)

# NN weights vs. PCs of Cov(s_m, s_n)
s_pc1 = pca.components_.T
plt.plot(s_pc1, label ="PC - PCA")
plt.plot(c1, label = "PC - FAE")
plt.legend()
plt.show()


#####################################
# Observed vs. FAE-recovered curves
#####################################
# Observed curves
x_cen = x_np-mean(x_np,0)
for m in range(0, len(x_cen)):
    plt.plot(tpts, x_cen[m])
plt.title("Observed Curves")
plt.show()
plt.close()

# Smoothed curves
basis_fd.plot()

# FAE-recovered curves
x_rev = outputs.detach().numpy()
for n in range(0, len(x_np)):
    plt.plot(tpts, x_rev[n])
plt.title("FAE-recovered Curves")
plt.show()
