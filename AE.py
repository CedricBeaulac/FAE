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
import random

#####################################
# Define the AE architecture
# Create AE Class
#####################################
class AE(nn.Module):
    def __init__(self, weight_std=None):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(n_tpts, 10)
        # self.fc12 = nn.Linear(50, 25)
        self.fc2 = nn.Linear(10, n_rep)
        self.fc3 = nn.Linear(n_rep, 10)
        # self.fc34 = nn.Linear(25,50)
        self.fc4 = nn.Linear(10, n_tpts)
        self.activation = nn.ReLU()

        # initialize the weights to a specified, constant value
        if (weight_std is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=weight_std)
                    #nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h1 = self.activation(self.fc1(x))
        # h2 = self.activation(self.fc12(h1))
        rep = self.fc2(h1)
        h3 = self.activation(self.fc3(rep))
        # h4 = self.activation(self.fc34(h3))
        x_hat = self.fc4(h3)

        return x_hat, rep

#####################################
# Define the training procedure
#####################################
# training function
def train_AE(epoch, n_tpts, n_rep):
    AE_model.train()
    train_loss = 0
    score_loss = 0
    for i, data in enumerate(train_loader):
        AE_optimizer.zero_grad()  # The gradients are set to zero
        # data = data.to(device)
        # input = data.type(torch.LongTensor)
        input = data.to(device)
        out, rep = AE_model(input.float())  # inputs should matches the inputs in forward function?
        ## Loss for back-propagation
        loss = loss_function(out, input.float())  # Maybe add score_loss as well?
        loss.backward()  # The gradient is computed and stored.
        AE_optimizer.step()  # .step() performs parameter update
        train_loss += loss
    return train_loss  # we need train_loss, instead of loss, for plotting

def pred_AE(data):
    AE_model.eval()
    # input = data.type(torch.LongTensor)
    input = data.to(device)
    output, rep = AE_model(input.float())
    loss = loss_function(output, input.float())
    return output, rep, loss

def eval_MSE(obs_X, pred_X):
    if not torch.is_tensor(obs_X):
        obs_X = torch.tensor(obs_X)
    if not torch.is_tensor(pred_X):
        pred_X = torch.tensor(pred_X)
    loss_fct = nn.MSELoss()
    loss = loss_fct(obs_X, pred_X)
    return loss

#####################################
# Load Data sets
#####################################
# Import dataset
os.chdir('C:/FAE')
os.chdir('C:/Users/Sidi/Desktop/FAE/FAE')
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
#tpts = torch.tensor(np.array(tpts_np))
n_tpts = len(tpts)
# tpts = np.linspace(0,1,num=10)

# Split training/test set
split.rate = 1
TrainData = x[0: round(len(x) * split.rate), :]
TestData = x[round(len(x) * split.rate):, :]

# Define data loaders; DataLoader is used to load the dataset for training
train_loader = torch.utils.data.DataLoader(TrainData, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(TestData)

#####################################
# Model Training
#####################################
# Set up parameters
n_rep = 5

# Model Initialization
AE_model = AE(weight_std=2)
# Validation using MSE Loss function
loss_function = nn.MSELoss()
# Using an Adam Optimizer with lr = 0.1
AE_optimizer = optim.Adam(AE_model.parameters(), lr=1e-2, weight_decay=1e-6)
# Using an ASGD Optimizer
# optimizer = optim.ASGD(model.parameters())
# Set to CPU/GPU
device = torch.device("cpu")  # (?)should be CUDA when running on the big powerfull server

epochs = 5000
AE_outputs = []
AE_reps = []
AE_losses = []
AE_pred_loss = []
# Train model
for epoch in range(1, epochs + 1):
    loss = train_AE(epoch, n_tpts=n_tpts, n_rep=n_rep)
    AE_losses.append(loss.detach().numpy())
    AE_outputs, AE_reps, AE_pred_loss = pred_AE(TestData)
    if epoch % 100 == 0:
        print(f"Epoch[{epoch}]-loss: {loss:.4f}, pred_loss: {AE_pred_loss:.4f}")

# Debug by looking at loss
plt.plot(AE_losses, label = "train_loss")
#plt.plot(score_losses, label = "score_loss")
plt.legend()
plt.show()
# plt.close()


input = TestData
AE_pred = torch.clone(AE_outputs)
AE_pred_plt = AE_pred.detach().numpy()
eval_mse_sdse(input, AE_pred)
eval_mse_sdse(TrainData, pred_AE(TrainData)[0])

# s = model.Project(input,basis_fc)
# rep = model.activation(model.fc1(s))
# s_hat = model.activation(model.fc3(rep))
# output = model.Revert(s_hat,basis_fc)

# Plot of Input (Observed Curves) & Output Curves (Predicted Curves)
AE_pred_plt = AE_pred.detach().numpy()
# Plot of Input (Observed Curves) & Output Curves (Predicted Curves)
AE_output_plt = AE_outputs.detach().numpy()

plt.figure(3, figsize=(20, 20))
plt.subplot(221)
for m in range(0, len(input_plt)):
# for m in id_plt:
    plt.plot(tpts, input_plt[m], "b")
plt.title("Input Curves")
plt.subplot(222)
for m in range(0, len(AE_output_plt)):
# for m in id_plt:
    plt.plot(tpts, AE_output_plt[m])
plt.title("Output Curves (AE)")
plt.subplot(223)
for m in range(0, len(FPCA_pred)):
# for m in id_plt:
    plt.plot(tpts, FPCA_pred[m])
plt.title("Output Curves (FPCA)")
plt.subplot(224)
for m in range(0, len(FAE_pred_plt)):
# for m in id_plt:
    plt.plot(tpts, FAE_pred_plt[m])
plt.title("Output Curves (FAE)")
plt.show()

# Curves of raw, FAE-recoverd, FPCA-recoverd, AE-recovered for some selected subjects
import matplotlib.backends.backend_pdf
plt.ioff()
pdf = matplotlib.backends.backend_pdf.PdfPages("Datasets/ElNino/ElNino_2layers(20-10-5)_nonlinear(Softplus)_+AE2.pdf")
# pdf = matplotlib.backends.backend_pdf.PdfPages("Datasets/tecator/tecator_2layer(50-40-5)_nbasis80_nfpcabasis10_linear_0.2Test.pdf")
for i in range(len(input_plt)): ## will open an empty extra figure :(\
    fig = plt.figure()
    plt.plot(tpts, input_plt[i], label="Raw")
    plt.plot(tpts, FAE_pred_plt[i], label="FAE-pred")
    plt.plot(tpts, FPCA_pred[i], label="FPCA-pred")
    plt.plot(tpts, AE_pred_plt[i], label="FAE-pred")
    plt.legend()
    plt.title(label=f"Observation #{i+1}")
    # plt.show()
    plt.close()
    pdf.savefig(fig)
pdf.close()
plt.ion()




input = x[0:5]
input.size()
h1 = AE_model.activation(AE_model.fc1(input))
h12 = AE_model.activa

h1 = AE_model.activation(AE_model.fc1(input))
h12 = AE_model.activation(AE_model.fc12(h1))
h2 = AE_model.activation(AE_model.fc2(h12))
h3 = AE_model.activation(AE_model.fc3(h2))
h4 = AE_model.activation(AE_model.fc34(h3))
x_hat = AE_model.fc4(h4)