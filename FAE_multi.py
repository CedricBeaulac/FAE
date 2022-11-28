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
from skfda.representation import basis as basis
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
import random

#################################################
# FAE: one hidden layer
# Encoder: a added layer for fd representation
# Decoder:
#################################################

#####################################
# Define the architecture of FAE with weight functions comprised of different basis functions.
# Create FAE Class
#####################################
class weight_func(nn.Module):
    def __init__(self, n_basis = 5, basis_type = "BSpline", time_grid = None, time_rescale=True,
                 weight_std=None):
        super(weight_func, self).__init__()
        self.weight_n_basis = n_basis
        self.weight_basis_type = basis_type
        self.weight_activation = nn.Identity()

        self.weight_layer = nn.Linear(n_basis, 1, bias=False)

        if (weight_std is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=weight_std)
                    #nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Rescale time grid
        obs_time = np.array(time_grid)
        if time_rescale == True:
            obs_time = (obs_time - min(obs_time))/np.ptp(obs_time)
        # produce basis functions accordingly
        if self.weight_basis_type == "BSpline":
            bss = basis.BSpline(n_basis=self.weight_n_basis, order=4)
        if self.weight_basis_type == "Fourier":
            bss = basis.Fourier(domain_range=(float(min(obs_time)), float(max(obs_time))),
                                n_basis = self.weight_n_basis)
        # Evalute basis functions at observed time grid
        weight_bss_eval = bss.evaluate(obs_time, derivative=0)
        weight_basis_fc = torch.from_numpy(weight_bss_eval[:, :, 0]).float()

        feature = self.Project(x, weight_basis_fc)
        h = self.weight_activation(self.weight_layer(feature))
        return h

    def Project(self, x, weight_basis_fc):
        # basis_fc: n_time X nbasis
        # x: n_subject X n_time
        w = x.size(1) - 1
        W = torch.tensor([1/(2*w)] + [1/w]*(w-1) + [1/(2*w)])
        s = torch.matmul(torch.mul(x, W), torch.t(weight_basis_fc))
        return s

class FAE_multi(nn.Module):
    def __init__(self, weight_basis_type_list=["BSpline"], weight_n_basis_list=[5],
                 n_basis = 5, basis_type = "BSpline", n_rep=3, time_grid=None, time_rescale=True,
                 activation = nn.ReLU(),dropout=None, device=None, weight_std=None):
        """
        weight_basis_type_list: list of basis function type for weight function projection
        n_basis_list: list of no. of basis functions selected, an integer
        n_basis: no. basis function for recovering functional data (decoder)
        basis_type: basis function type for recovering functional data (decoder)
        n_rep: no. of elements in the list of representation, an integer
        time_grid: observed time grid for each subject, array of floats
        weight_activation: activation function used in weight function projection
        activation: activation function used in Forward process
        dropout: dropout rate
        device: device for training
        weight_std: SD of the Normal dist. for initialing weights
        """
        super(FAE_multi, self).__init__()
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.n_input = len(weight_basis_type_list)
        self.activation = activation
        self.device = device

        # initial the weight function project layer
        self.weight_project = nn.ModuleList([weight_func(n_basis = weight_n_basis_list[i],
                                                         basis_type=weight_basis_type_list[i],
                                                         time_grid=time_grid,
                                                         time_rescale=True,
                                                         weight_std=weight_std) for i in range(self.n_input)])
        # initial the following layers
        self.fc1 = nn.Linear(self.n_input, n_rep, bias=False)
        #self.fc3 = nn.Linear(100, n_basis,bias=False)
        self.fc3 = nn.Linear(n_rep, self.n_basis, bias=False)
        # self.fc4 = nn.Linear(100, n_basis, bias=False)

        # initialize the weights to a specified, constant value
        if (weight_std is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=weight_std)
                    #nn.init.constant_(m.bias, 0)

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

        input = torch.cat([weight(x) for weight in self.weight_project], dim=-1)
        rep = self.activation(self.fc1(input))
        # t1 = self.activation(self.fc1(s))
        # rep = self.activation(self.fc2(t1))
        # t2 = self.activation(self.fc3(rep))
        # s_hat = self.fc4(t2)
        basis_coef = self.fc3(rep)
        x_hat = self.Revert(basis_coef, basis_fc)
        return x_hat, rep, basis_coef

    def Revert(self, x, basis_fc):
        f = torch.matmul(x, basis_fc)
        return f

#####################################
# Define the training procedure
#####################################
# training function
def FAE_multi_train(train_loader, lamb=0):  # do I need to include "loss_function", how about "optimizer"
    # It depends if you define train locally or not
    FAE_multi_model.train()
    train_loss = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # The gradients are set to zero
        # data = data.to(device)
        # input = data.type(torch.LongTensor)
        input = data.to(device)
        output, rep, basis_coef = FAE_multi_model(input.float()) # inputs should matches the inputs in forward function?
        ## Loss for back-propagation
        # Penalty term
        penalty = 0
        if lamb != 0:
            delta_c = basis_coef[:,2:] - 2*basis_coef[:,1:-1] + basis_coef[:,:-2]
            penalty = torch.mean(torch.sum(delta_c**2, dim=1))
        loss = loss_function(output, input.float()) + lamb*penalty # Maybe add score_loss as well?
        # loss = loss_function(s, s_hat)
        loss.backward()  # The gradient is computed and stored.
        optimizer.step()  # .step() performs parameter update
        train_loss += loss
    return train_loss # we need train_loss, instead of loss, for plotting

def FAE_multi_pred(model, data):
    FAE_multi_model.eval()
    # input = data.type(torch.LongTensor)
    input = data.to(device)
    output, rep, basis_coef = FAE_multi_model(input.float())
    loss = loss_function(output, input.float())
    return output, rep, loss

#####################################
# Load Data sets
#####################################
# Import dataset
os.chdir('C:/FAE')
os.chdir('C:/Users/Sidi/Desktop/FAE/FAE')
# Dataset: tecator
# x_raw = pd.read_csv('Datasets/tecator/tecator.csv')
# tpts_raw = pd.read_csv('Datasets/tecator/tecator_tpts.csv')
# Dataset: pinch
# x_raw = pd.read_csv('Datasets/pinch/pinch.csv')
# tpts_raw = pd.read_csv('Datasets/pinch/pinch_tpts.csv')
# Dataset: ElNino
x_raw = pd.read_csv('Datasets/ElNino/ElNino_ERSST.csv')
tpts_raw = pd.read_csv('Datasets/ElNino/ElNino_ERSST_tpts.csv')

# Prepare numpy/tensor data
x_np = np.array(x_raw).astype(float)
x = torch.tensor(x_np).float()
x = x - torch.mean(x,0)

# Rescale timestamp to [0,1]
tpts_np = np.array(tpts_raw)
#tpts = torch.tensor(np.array(tpts_np))
tpts_rescale = (tpts_np - min(tpts_np)) / np.ptp(tpts_np)
tpts = torch.tensor(np.array(tpts_rescale))
#tpts = torch.tensor(np.array(tpts_np))
n_tpts = len(tpts)
# tpts = np.linspace(0,1,num=10)

# Split training/test set
split.rate = 0.8
# TrainData, TestData = torch.utils.data.random_split(x, [round(len(x) * split.rate), (len(x)-round(len(x) * split.rate))])
train_no = random.sample(range(0, len(x)), round(len(x) * split.rate))
TrainData = x[train_no]
if split.rate == 1:
    TestData=x
else:
    TestData = x[[i for i in range(len(x)) if i not in train_no]]

# Define data loaders; DataLoader is used to load the dataset for training
train_loader = torch.utils.data.DataLoader(TrainData, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(TestData)


#####################################
# Model Training
#####################################
# Set up parameters
weight_basis_type_list = ["BSpline", "BSpline", "BSpline", "BSpline", "BSpline"]
weight_n_basis_list = [50, 60, 80, 50, 50]
n_basis = 50
basis_type = "BSpline"
n_rep = 5
time_grid = tpts
time_rescale = True
activation_function = nn.Identity()
weight_std = 2

lamb = 0.5
# basis_type = "Bspline"
# # Get basis functions evaluated
# if basis_type == "Bspline":
#     bss = representation.basis.BSpline(n_basis=n_basis, order=4)
# elif basis_type == "Fourier":
#     bss = representation.basis.Fourier([min(tpts.numpy().flatten()), max(tpts.numpy().flatten())], n_basis=n_basis)
#
# bss_eval = bss.evaluate(tpts, derivative=0)
# basis_fc = torch.from_numpy(bss_eval[:, :, 0]).float()

# Model Initialization
FAE_multi_model = FAE_multi(weight_basis_type_list=weight_basis_type_list,
                            weight_n_basis_list=weight_n_basis_list,
                            n_basis=n_basis,
                            basis_type=basis_type,
                            n_rep=n_rep,
                            time_grid=time_grid,
                            time_rescale=time_rescale,
                            activation=activation_function,
                            weight_std=weight_std)
# Validation using MSE Loss function
loss_function = nn.MSELoss()
# Using an Adam Optimizer with lr = 0.1
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
# Using an ASGD Optimizer
# optimizer = optim.ASGD(model.parameters())
# Set to CPU/GPU
device = torch.device("cpu")  # (?)should be CUDA when running on the big powerfull server

epochs = 5000
FAE_multi_output = []
FAE_multi_rep = []
FAE_multi_loss = []

# Train model
for epoch in range(1, epochs + 1):
    loss = FAE_multi_train(train_loader=train_loader, lamb=lamb)
    FAE_multi_loss.append(loss.detach().numpy())
    FAE_multi_output, FAE_multi_rep, FAE_multi_pred_loss= FAE_multi_pred(FAE_multi_model, TestData)
    if epoch % 100 ==0:
        print(f"Epoch[{epoch}]-loss: {loss:.4f}")

# Debug by looking at loss
plt.plot(FAE_multi_loss, label = "train_loss")
#plt.plot(score_losses, label = "score_loss")
plt.legend()
plt.show()
# plt.close()

# Debug by looking at the FAE, layer by layer
input = TestData
FAE_pred = torch.clone(FAE_multi_output)
# s = model.Project(input,basis_fc)
# rep = model.activation(model.fc1(s))
# s_hat = model.activation(model.fc3(rep))
# output = model.Revert(s_hat,basis_fc)

# Plot of Input (Observed Curves) & Output Curves (Predicted Curves)
input_plt = input.detach().numpy()
FAE_pred_plt = FAE_pred.detach().numpy()

plt.figure(3, figsize=(10, 20))
plt.subplot(211)
for m in range(0, len(input_plt)):
# for m in id_plt:
    plt.plot(tpts, input_plt[m])
plt.title("Input Curves")
plt.subplot(212)
for m in range(0, len(FAE_pred_plt)):
# for m in id_plt:
    plt.plot(tpts, FAE_pred_plt[m])
plt.title("Output Curves")
plt.show()

# id_plt = random.sample(range(0, len(input_plt)), 15)
# plt.figure(4, figsize=(10, 20))
# plt.subplot(211)
# for m in id_plt:
#     plt.plot(tpts, input_plt[m])
# plt.title("Input Curves")
# plt.subplot(212)
# for m in id_plt:
#     plt.plot(tpts, FAE_pred_plt[m])
# plt.title("Output Curves")
# plt.show()

#####################################
# Perform FPCA
#####################################
n_basis_fpca = 10
if basis_type == "Bspline":
    bss_fpca = representation.basis.BSpline(n_basis=n_basis_fpca, order=4)
elif basis_type == "Fourier":
    bss = representation.basis.Fourier([min(tpts.numpy().flatten()), max(tpts.numpy().flatten())],
                                       n_basis=n_basis_fpca)

tpts_fd = tpts.numpy().flatten()
fd_train = representation.grid.FDataGrid(TrainData.numpy(), tpts_fd)
fd_test = representation.grid.FDataGrid(TestData.numpy(), tpts_fd)
basis_fd_train = fd_train.to_basis(bss_fpca)
basis_fd_test = fd_test.to_basis(bss_fpca)
# basis_fd = fd.to_basis(representation.basis.BSpline(n_basis=80, order=4))
fpca_basis = fda.preprocessing.dim_reduction.feature_extraction.FPCA(n_components=n_rep)

# Get FPCs
fpca_basis = fpca_basis.fit(basis_fd_train)
fpca_basis.components_.plot()

# Get FPC scores
fpc_scores_test = fpca_basis.transform(basis_fd_test)
FPCA_pred = fpca_basis.inverse_transform(fpc_scores_test)._evaluate(tpts_fd)[:,:,0]

# Get mean function
fpca_basis.mean_.plot()
fpca_mean = fpca_basis.mean_.to_grid().numpy()
#fpca_basis.singular_values_


plt.figure(4, figsize=(10, 20))
plt.subplot(211)
for m in range(0, len(input_plt)):
# for m in id_plt:
    plt.plot(tpts, input_plt[m])
plt.title("Input Curves")
plt.subplot(212)
for m in range(0, len(FPCA_pred)):
# for m in id_plt:
    plt.plot(tpts, FPCA_pred[m])
plt.title("Output Curves")
plt.show()

# fpca_basis.components_[1].plot(label = 'FPC2-FPCA')
# plt.plot(tpts, pc2, label = "FPC2-FAE")
# plt.title(f"Basis#={n_basis}, FPC#={n_rep}")
# plt.legend()
# plt.show()
# plt.close()

#####################################
# Evaluate FAE & Compare with FPCA
#####################################
def eval_MSE(obs_X, pred_X):
    if not torch.is_tensor(obs_X):
        obs_X = torch.tensor(obs_X)
    if not torch.is_tensor(pred_X):
        pred_X = torch.tensor(pred_X)
    loss_fct = nn.MSELoss()
    loss = loss_fct(obs_X, pred_X)
    return loss

eval_MSE(input, FAE_pred)
eval_MSE(input, FPCA_pred)

eval_tpts_FAE = []
eval_tpts_FPCA = []
for i in range(len(tpts)):
    eval_temp_FAE = eval_MSE(input[:,i], FAE_pred[:,i])
    eval_temp_FPCA = eval_MSE(input[:,i], FPCA_pred[:,i])
    eval_tpts_FAE.append(eval_temp_FAE.item())
    eval_tpts_FPCA.append(eval_temp_FPCA.item())

# Curves of raw, FAE-recoverd, FPCA-recoverd for some selected subjects
import matplotlib.backends.backend_pdf
plt.ioff()
pdf = matplotlib.backends.backend_pdf.PdfPages("layer1_linear.pdf")
for i in range(len(input_plt)): ## will open an empty extra figure :(\
    fig = plt.figure()
    plt.plot(tpts, input_plt[i], label="Raw")
    plt.plot(tpts, FAE_pred_plt[i], label="FAE-pred")
    plt.plot(tpts, FPCA_pred[i], label="FPCA-pred")
    plt.legend()
    plt.title(label=f"Observation #{i+1}")
    # plt.show()
    plt.close()
    pdf.savefig(fig)
pdf.close()
plt.ion()
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

#####################################
# Perform PCA on features s_m
#####################################
s_np = s.detach().numpy()
pca = PCA(n_components=1)
pca.fit(s_np)

# NN weights vs. PCs of Cov(s_m, s_n)
s_pc1 = pca.components_.T
plt.plot(s_pc1, label ="PC-PCA")
plt.plot(c1, label = "PC(NN weights)-FAE-encoder")
plt.plot(c1_hat, label = "PC(NN weights)-FAE-decoder")
plt.xlabel('basis #')
plt.legend()
plt.show()

#####################################
# Observed vs. FAE-recovered curves
#####################################
# Observed curves
plt.figure(1)
x_cen = x_np-mean(x_np,0)
for m in range(0, len(x_cen)):
    plt.plot(tpts, x_cen[m])
plt.title("Observed Curves")
plt.show()
plt.close()

# Smoothed curves
basis_fd.plot()
plt.title("Smoothed Curves")

# Feature-recovered curves
plt.figure(3)
# x_rec = model.Revert(model.Project(TrainData,basis_fc),basis_fc)
x_rec = model.Revert(s, basis_fc)
for n in range(0, len(x_np)):
    plt.plot(tpts, x_rec[n])
plt.title("Feature-recovered Curves")
plt.show()

# FAE-recovered curves
plt.figure(4)
x_rev = output.detach().numpy()
for n in range(0, len(x_np)):
    plt.plot(tpts, x_rev[n])
plt.title("FAE-recovered Curves")

plt.figure(5, figsize=(20, 20))
plt.subplot(221)
# Observed curves
x_cen = x_np-mean(x_np,0)
for m in range(0, len(x_cen)):
    plt.plot(tpts, x_cen[m])
plt.title("Observed Curves")
# plt.subplot(222)
# # Smoothed curves
basis_fd.plot()
plt.title("Smoothed Curves")
plt.subplot(223)
# feature-recovered curves
# x_rec = model.Revert(model.Project(TrainData,basis_fc),basis_fc)
x_rec = model.Revert(s,basis_fc)
for n in range(0, len(x_np)):
    plt.plot(tpts, x_rec[n])
plt.title("Feature-recovered Curves")
plt.subplot(224)
# FAE-recovered curves
x_rev = output.detach().numpy()
for n in range(0, len(x_np)):
    plt.plot(tpts, x_rev[n])
plt.title("FAE-recovered Curves")
plt.show()

