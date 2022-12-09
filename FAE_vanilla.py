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
import sys
import sklearn
from sklearn.decomposition import PCA
import random

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
    def __init__(self, weight_std=None):
        super(FAE_vanilla, self).__init__()
        self.fc1 = nn.Linear(n_basis, 40,bias=False)
        self.fc2 = nn.Linear(40, n_rep, bias=False)
        # self.fc1 = nn.Linear(n_basis, n_rep, bias=False)
        self.fc3 = nn.Linear(n_rep, 40, bias=False)
        # self.fc3 = nn.Linear(n_rep, n_basis, bias=False)
        self.fc4 = nn.Linear(40, n_basis, bias=False)
        self.activation = nn.ReLU()
        # initialize the weights to a specified, constant value
        if (weight_std is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=weight_std)
                    #nn.init.constant_(m.bias, 0)
    def forward(self, x, basis_fc):
        s = self.Project(x, basis_fc)
        # rep = self.activation(self.fc1(s))
        t1 = self.activation(self.fc1(s))
        rep = self.fc2(t1)
        t2 = self.activation(self.fc3(rep))
        s_hat = self.fc4(t2)
        # s_hat = self.fc3(rep)
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

# class FAE_vanilla(nn.Module):
#     def __init__(self):
#         super(FAE_vanilla, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(n_basis, 80),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(80, n_rep),
#             nn.ReLU()
#         )
#
#         self.decoder = nn.Sequential(
#             nn.Linear(n_rep, 80),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(80, n_basis),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         s = self.Project(x, basis_fc)
#         rep = self.encoder(s)
#         s_hat = self.decoder(rep)
#         x_hat = self.Revert(s_hat, basis_fc)
#         return x_hat, rep
#
#     def Project(self, x, basis_fc):
#         # basis_fc: n_time X nbasis
#         # x: n_subject X n_time
#         s = torch.matmul(x, torch.t(basis_fc))
#         return s
#
#     def Revert(self, x, basis_fc):
#         f = torch.matmul(x, basis_fc)
#         return f

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(n_tpts, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 100)
        self.fc4 = nn.Linear(100, n_tpts)
        self.activation = nn.ReLU()

    def forward(self, x):
        h1 = self.activation(self.fc1(x))
        h2 = self.activation(self.fc2(h1))
        h3 = self.activation(self.fc3(h2))
        x_hat = self.fc4(h3)

        return x_hat

#####################################
# Define the training procedure
#####################################
# training function
def train(train_loader, pen=None, lamb=0):  # do I need to include "loss_function", how about "optimizer"
    # It depends if you define train locally or not
    model.train()
    train_loss = 0
    score_loss = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # The gradients are set to zero
        # data = data.to(device)
        # input = data.type(torch.LongTensor)
        input = data.to(device)
        out,rep,s,s_hat = model(input.float(), basis_fc) # inputs should matches the inputs in forward function?
        ## Loss on the score layers (network output layer)
        score_loss += loss_function(s, s_hat)
        ## Loss for back-propagation
        # Penalty term
        penalty = 0
        if pen == "encoder":
            delta_c = model.fc1.weight[:,2:] - 2*model.fc1.weight[:,1:-1] + model.fc1.weight[:,:-2]
            penalty = torch.sum(delta_c**2) # torch.sum(torch.sum(delta_c**2, dim=1))
        if pen == "decoder" :
            delta_c = model.fc3.weight[:,2:] - 2*model.fc3.weight[:,1:-1] + model.fc3.weight[:,:-2]
            penalty = torch.sum(delta_c**2) # torch.sum(torch.sum(delta_c**2, dim=1))
        if pen == "feature":
            delta_c = s_hat[:,2:] - 2*s_hat[:,1:-1] + s_hat[:,:-2]
            penalty = torch.mean(torch.sum(delta_c**2, dim=1))
        # for j in range(0, n_rep):
        #     penalty_rep = 0
        #     # delta_c = model.fc1.weight[j][2:] - 2*model.fc1.weight[j][1:-1] + model.fc1.weight[j][:-2]
        #     # penalty_rep = torch.sum(delta_c**2)
        #     for k in range(2, n_basis):
        #         delta_c = model.fc1.weight[j][k]-2*model.fc1.weight[j][k-1]+model.fc1.weight[j][k-2]
        #         penalty_rep += delta_c**2
        #     penalty += penalty_rep
        loss = loss_function(out, input.float()) + lamb*penalty # Maybe add score_loss as well?
        # loss = loss_function(s, s_hat)
        loss.backward()  # The gradient is computed and stored.
        optimizer.step()  # .step() performs parameter update
        train_loss += loss
    return train_loss, score_loss # we need train_loss, instead of loss, for plotting

def pred(model, data):
    model.eval()
    # input = data.type(torch.LongTensor)
    input = data.to(device)
    output, rep, s, s_hat = model(input.float(), basis_fc)
    loss = loss_function(output, input.float())
    score_loss = loss_function(s, s_hat)
    return output, rep, loss, score_loss

#####################################
# Load Data sets
#####################################
# Import dataset
os.chdir('C:/FAE')
os.chdir('C:/Users/Sidi/Desktop/FAE/FAE')
# sys.path.append(os.getcwd())
import DataGenerator
from DataGenerator import *
# Dataset: tecator
# x_raw = pd.read_csv('Datasets/tecator/tecator.csv')
# tpts_raw = pd.read_csv('Datasets/tecator/tecator_tpts.csv')
# Dataset: pinch
# x_raw = pd.read_csv('Datasets/pinch/pinch.csv')
# tpts_raw = pd.read_csv('Datasets/pinch/pinch_tpts.csv')
# Dataset: ElNino
x_raw = pd.read_csv('Datasets/ElNino/ElNino_ERSST.csv')
tpts_raw = pd.read_csv('Datasets/ElNino/ElNino_ERSST_tpts.csv')

# nc=200
# tpts = np.linspace(0,1,21)
# x_raw,curves = SmoothDataGenerator(nc, tpts,8,0.4)

nc=500
classes = 10
tpts = np.linspace(0,1,21)
x_raw,curves, coef_raw = SmoothDataGenerator(nc, tpts, classes,0.5)

plt.plot(curves)
plt.show()
plt.plot(transpose(x_raw))
plt.show

tpts_raw = tpts
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
# train_no = random.sample(list(range(0, len(x))), round(len(x) * split.rate))
train_no = []
seed(142)
train_seeds = random.sample(range(1000), classes)
for i in range(0, classes):
    step = len(x)/classes
    seed(train_seeds[i])
    temp_no = random.sample(range(int(step*i), int(step*(i+1))), round(step*split.rate))
    train_no.extend(temp_no)

TrainData = x[train_no]
if split.rate == 1:
    TestData=x
else:
    TestData = x[[i for i in range(len(x)) if i not in train_no]]

# Define data loaders; DataLoader is used to load the dataset for training
train_loader = torch.utils.data.DataLoader(TrainData, batch_size=48, shuffle=True)
test_loader = torch.utils.data.DataLoader(TestData)


#####################################
# Model Training
#####################################
# Set up parameters
n_basis = 50
n_rep = 5
lamb = 0.001
pen = "feature"
basis_type = "Bspline"
# Get basis functions evaluated
if basis_type == "Bspline":
    bss = representation.basis.BSpline(n_basis=n_basis, order=4)
elif basis_type == "Fourier":
    bss = representation.basis.Fourier([min(tpts.numpy().flatten()), max(tpts.numpy().flatten())], n_basis=n_basis)

bss_eval = bss.evaluate(tpts, derivative=0)
basis_fc = torch.from_numpy(bss_eval[:, :, 0]).float()

# Model Initialization
model = FAE_vanilla(weight_std=1)
# Validation using MSE Loss function
loss_function = nn.MSELoss()
# Using an Adam Optimizer with lr = 0.1
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
# Using an ASGD Optimizer
# optimizer = optim.ASGD(model.parameters())
# Set to CPU/GPU
device = torch.device("cpu")  # (?)should be CUDA when running on the big powerfull server

epochs = 5000
outputs = []
reps = []
losses = []
score_losses = []

# Train model
for epoch in range(1, epochs + 1):
    loss, score_loss = train(train_loader=train_loader, pen=pen, lamb=lamb)
    losses.append(loss.detach().numpy())
    score_losses.append(score_loss.detach().numpy())
    outputs, reps, pred_loss, pred_score_loss = pred(model, TestData)
    if epoch % 100 ==0:
        print(f"Epoch[{epoch}]-loss: {loss:.4f}; feature loss: {score_loss: 4f}; pred_loss:{pred_loss:4f}")

# Debug by looking at loss
plt.plot(losses, label = "train_loss")
#plt.plot(score_losses, label = "score_loss")
plt.legend()
plt.show()
# plt.close()

# Debug by looking at the FAE, layer by layer
input = TestData
FAE_pred = torch.clone(outputs)
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
    plt.plot(tpts, input_plt[m])
plt.title("Input Curves")
plt.subplot(212)
for m in range(0, len(FAE_pred_plt)):
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
n_basis_fpca = 15
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
# fpca_basis_fd = fpca_basis.fit(fd)
# fpca_basis_fd.components_.plot()
fpca_basis = fpca_basis.fit(basis_fd_train)
fpca_basis.components_.plot()

fpca_basis.explained_variance_
# Get FPC scores
fpc_scores_train = fpca_basis.transform(basis_fd_train)
fpc_scores_test = fpca_basis.transform(basis_fd_test)
FPCA_pred = fpca_basis.inverse_transform(fpc_scores_test)._evaluate(tpts_fd)[:,:,0]

# Get mean function
fpca_basis.mean_.plot()
fpca_mean = fpca_basis.mean_.to_grid().numpy()
#fpca_basis.singular_values_

fpca_basis.components_[0].plot(label = 'FPC1-FPCA')
plt.plot(tpts, pc1, label='FPC1-FAE-encoder')
plt.plot(tpts, pc1_hat, label="FPC1-FAE-decoder")
plt.xlabel('time grid')
plt.title(f"Basis#={n_basis}, FPC#={n_rep}, lamb={lamb}")
plt.legend()
plt.show()
plt.close()

plt.figure(4, figsize=(10, 20))
plt.subplot(211)
for m in range(0, len(input_plt)):
    plt.plot(tpts, input_plt[m])
plt.title("Input Curves")
plt.subplot(212)
for m in range(0, len(FPCA_pred)):
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

# On test data set
eval_MSE(TestData, FAE_pred)
eval_MSE(TestData, FPCA_pred)

# On train data
FAE_pred_train, reps, pred_loss, pred_score_loss = pred(model, TrainData)
FPCA_pred_train = fpca_basis.inverse_transform(fpc_scores_train)._evaluate(tpts_fd)[:,:,0]
eval_MSE(TrainData, FAE_pred_train)
eval_MSE(TrainData, FPCA_pred_train)

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
pdf = matplotlib.backends.backend_pdf.PdfPages("Datasets/ElNino/ElNino_2layers_nonlinear_0.8Train.pdf")
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

############################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
############################################################################################################################

#####################################
# Principals components and representation
#####################################
## c's are estimated as weight of the fc1 function
c1 = model.fc1.weight[0].detach()
# c2 = model.fc1.weight[1].detach()
# c1 = model.encoder[0].weight.detach()
pc1 = torch.matmul(c1,basis_fc).numpy()
# pc2 = torch.matmul(c2,basis_fc).numpy()

## c's are estimated as weight of the fc3 function
c1_hat = model.fc3.weight[:,0].detach()
# c2_hat = model.fc3.weight[:,1].detach()
# c1 = model.encoder[0].weight.detach()
pc1_hat = torch.matmul(c1_hat,basis_fc).numpy()
# pc2_hat = torch.matmul(c2_hat,basis_fc).numpy()

plt.plot(tpts, pc1, label='FPC1-FAE-encoder')
# plt.plot(tpts, pc2, label='FPC2-FAE-encoder')
plt.plot(tpts, pc1_hat, label="FPC1-FAE-decoder")
# plt.plot(tpts, pc2_hat, label="FPC2-FAE-decoder")
plt.xlabel('time grid')
plt.legend()
plt.show()
# plt.close()

# Representatives, = FPC scores
reps_bss = reps.detach().numpy()
# reps_gt2 = reps_fpc[reps_fpc > 2]
# np.where(reps_fpc > 2)[0]
# [i for i, x in enumerate(reps_fpc > 2) if x]

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

