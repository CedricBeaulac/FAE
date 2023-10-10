"""
This script contains the code for implementing the proposed functional autoencoders (FAE) with irregularly spaced functional data
in the manuscript "Autoencoders for Discrete Functional Data Representation Learning and Smoothing".

@author: Sidi Wu
"""

# Import modules
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pandas as pd
import numpy as np
from numpy import *
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import skfda as fda
from skfda import representation as representation
from skfda.exploratory.visualization import FPCAPlot
import scipy
from scipy import stats
from scipy.interpolate import BSpline
import ignite
import os
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import random
from random import seed
import statistics
from statistics import stdev
from time import process_time
from datetime import datetime

os.chdir('~/Code')
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import Functions
from Functions import *

#####################################################################
# Define the vanilla FAE architecture for irregularly functional data
# Create FAE_irr Class & Data Class
#####################################################################
# Below are the settings for the creating the nonlinear FAE (we do not apply linear FAE for irregularly functional data in the application)
# FAE_irr class
class FAE_irr(nn.Module):
    def __init__(self, weight_std=None):
        super(FAE_irr, self).__init__()
        self.fc1 = nn.Linear(n_basis_input, 50, bias=False)
        self.fc2 = nn.Linear(50, n_rep, bias=False)
        self.fc3 = nn.Linear(n_rep, 50, bias=False)
        self.fc4 = nn.Linear(50, n_basis_output, bias=False)
        self.activation = nn.Softplus()

        # initialize the weights to a specified, constant value
        if (weight_std is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=weight_std)

    def forward(self, x, W, nan_ind, basis_fc_input, basis_fc_output):
        feature = self.Project(x, W, basis_fc_input)
        t1 = self.activation(self.fc1(feature))
        rep = self.fc2(t1)
        t2 = self.activation(self.fc3(rep))
        coef = self.fc4(t2)
        x_hat = self.Revert(coef, nan_ind, basis_fc_output)
        return x_hat, rep, feature, coef

    def Project(self, x, W, basis_fc_input):
        # x, time, W: n_subject X n_time
        f = torch.matmul(torch.mul(x, W), torch.t(basis_fc_input))
        return f

    def Revert(self, x, nan_ind, basis_fc_output):
        g = torch.matmul(x, basis_fc_output)
        g = torch.mul(g, nan_ind)
        return g

# Data class: convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, Omega, missing_ind):
        self.X = X
        self.Omega = Omega
        self.missing_ind = missing_ind
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Omega[index], self.missing_ind[index],

    def __len__(self):
        return self.len

#####################################
# Define the training procedure
#####################################
# The function for training FAE
def train_irr(train_loader, basis_eval_input, basis_eval_output, pen=None, lamb=0):
    model.train()
    train_loss = 0
    score_loss = 0
    for i, (data, W, missing_ind) in enumerate(train_loader):
        optimizer.zero_grad()  # The gradients are set to zero
        input = data.to(device)
        out,rep,feature,coef = model(input, W, missing_ind, basis_eval_input, basis_eval_output)
        ## Loss on the score layers (network output layer)
        if shape(feature)==shape(coef):
            score_loss += loss_fct(feature, coef) # meaningful when basis functions are orthonormal
        # Penalty term
        penalty = 0
        if pen == "diff":
            delta_c = coef[:,2:] - 2*coef[:,1:-1] + coef[:,:-2]
            penalty = torch.mean(torch.sum(delta_c**2, dim=1))
        # Loss for back-propagation
        loss = loss_fct(out, input.float()) + lamb*penalty
        loss.backward()  # The gradient is computed and stored.
        optimizer.step()  # Performs parameter update
        train_loss += loss
    return train_loss, score_loss

# The function for predicting observations with trained FAE
def pred_irr(model, data, Omega, missing_ind, basis_eval_input, basis_eval_output):
    model.eval()
    input = data.to(device)
    output, rep, feature, coef = model(input, Omega, missing_ind, basis_eval_input, basis_eval_output)
    loss = loss_fct(output, input.float())
    return output, rep, loss, coef

#####################################
# Perform FAE_irr (Model Training)
#####################################
niter = 10
seed(742)
niter_seed = random.sample(range(1000), niter)

# Set up basis functions for input functional weights
n_basis_input = 50
basis_type_input =  "Bspline"
# Get basis functions evaluated
if basis_type_input == "Bspline":
    bss_input = representation.basis.BSpline(n_basis=n_basis_input, order=4)
elif basis_type_input == "Fourier":
    bss_input = representation.basis.Fourier([min(tpts.numpy().flatten()), max(tpts.numpy().flatten())], n_basis=n_basis_input)
bss_eval_input = bss_input.evaluate(true_tpts_rescale, derivative=0)
basis_fc_input = torch.from_numpy(bss_eval_input[:, :, 0]).float()

# Set up basis functions for output functional weights
n_basis_output = 50
basis_type_output = "Bspline"
# Get basis functions evaluated
if basis_type_output == "Bspline":
    bss_output = representation.basis.BSpline(n_basis=n_basis_output, order=4)
elif basis_type_output == "Fourier":
    bss_output = representation.basis.Fourier([min(tpts.numpy().flatten()), max(tpts.numpy().flatten())], n_basis=n_basis_output)
bss_eval_output = bss_output.evaluate(true_tpts_rescale, derivative=0)
basis_fc_output = torch.from_numpy(bss_eval_output[:, :, 0]).float()

# Set up lists to save training info
FAE_train_no_niter = []
FAE_reps_train_niter = []
FAE_reps_test_niter = []
FAE_reps_all_niter = []
FAE_pred_test_niter = []
FAE_pred_all_niter = []
FAE_coef_train_niter = []
FAE_coef_test_niter = []
FAE_pred_train_acc_mean_niter = []
FAE_pred_test_acc_mean_niter = []
FAE_pred_train_acc_sd_niter = []
FAE_pred_test_acc_sd_niter = []
classification_FAE_train_niter = []
classification_FAE_test_niter = []

# Set up FAE's hyperparameters
n_rep = 5 # number of representation
lamb = 0.001 # penalty parameter
pen = "diff" # penalty type
epochs = 2000 # epochs
batch_size = 128 # batch size
init_weight_sd = 1 # SD of normal dist. for initializing NN weight
split_rate = 0.8 # percentage of training set (or 0.2)

# Set up lists for training history
FAE_reg_test_acc_epoch = [[] for x in range(int(epochs/100))]
classification_FAE_reg_test_epoch = [[] for x in range(int(epochs/100))]

# Start iterations
for i in range(niter):
    # Split training/test set
    TrainData, TestData, TrainLabel, TestLabel, TrainNan, TestNan, TrainOmega, TestOmega, train_no = train_test_split(x, label, missing_ind = nan_ind, omega = Omega, split_rate =split_rate, seed_no=niter_seed[i])
    FAE_train_no_niter.append(train_no)

    # Instantiate training and test data - Define data loaders; DataLoader is used to load the dataset for training
    train_data = Data(TrainData, TrainOmega, TrainNan)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = Data(TestData, TestOmega, TestNan)
    test_loader = DataLoader(dataset=test_data)

    # Model Initialization
    model = FAE_irr(weight_std=init_weight_sd)
    loss_fct = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-6)
    device = torch.device("cpu")

    epochs = epochs
    # Train model
    for epoch in range(1, epochs + 1):
        loss, score_loss = train_irr(train_loader=train_loader, basis_eval_input=basis_fc_input, basis_eval_output=basis_fc_output, pen=pen, lamb=lamb)
        FAE_pred_test, FAE_reps_test, FAE_pred_loss_test, FAE_coef_test = pred_irr(model, test_data.X, test_data.Omega, test_data.missing_ind,
                                                                    basis_fc_input, basis_fc_output)
        if epoch % 100 == 0:
            print(f"Epoch[{epoch}]-loss: {loss:.4f}; feature loss: {score_loss: 4f}; pred_loss:{FAE_pred_loss_test:4f}")
            FAE_reg_test_acc_epoch[int(epoch / 100) - 1].append(FAE_pred_loss_test.tolist())
            FAE_reps_train_temp = pred_irr(model, train_data.X, train_data.Omega, train_data.missing_ind, basis_fc_input, basis_fc_output)[1]
            FAE_classifier_temp = LogisticRegression(solver='liblinear', random_state=0, multi_class='auto').fit(
                FAE_reps_train_temp.detach().numpy(), TrainLabel)
            classification_FAE_reg_test_epoch[int(epoch / 100) - 1].append(FAE_classifier_temp.score(FAE_reps_test.detach().numpy(), TestLabel))

    FAE_reps_test_niter.append(FAE_reps_test)
    FAE_pred_test_niter.append(FAE_pred_test)
    FAE_coef_test_niter.append(FAE_coef_test)
    FAE_pred_all, FAE_reps_all = pred_irr(model, x, Omega, nan_ind ,basis_fc_input, basis_fc_output)[0:2]
    FAE_reps_all_niter.append(FAE_reps_all)
    FAE_pred_all_niter.append(FAE_pred_all)

    FAE_pred_test_acc_mean_niter.append(FAE_pred_loss_test.tolist())
    FAE_pred_test_acc_sd_niter.append(eval_mse_sdse(TestData, FAE_pred_test)[1].tolist())

    FAE_pred_train, FAE_reps_train, FAE_pred_loss_train, FAE_coef_train = pred_irr(model, train_data.X, train_data.Omega, train_data.missing_ind,
                                                                   basis_fc_input, basis_fc_output)
    FAE_reps_train_niter.append(FAE_reps_train)
    FAE_coef_train_niter.append(FAE_coef_train)
    FAE_pred_train_acc_mean_niter.append(FAE_pred_loss_train.tolist())
    FAE_pred_train_acc_sd_niter.append(eval_mse_sdse(TrainData, FAE_pred_train)[1].tolist())

    ## Classification
    # Create classifiers (logistic regression) & train the model with the training set
    FAE_classifier = LogisticRegression(solver='liblinear', random_state=0, multi_class='auto').fit(FAE_reps_train.detach().numpy(), TrainLabel)
    # Classification accuracy on the test set
    classification_FAE_test_niter.append(FAE_classifier.score(FAE_reps_test.detach().numpy(), TestLabel))
    # Classification accuracy on the training set
    classification_FAE_train_niter.append(FAE_classifier.score(FAE_reps_train.detach().numpy(), TrainLabel))

    print(f"Replicate {i + 1} is complete.")

# Print for result tables
print("--- FAE-Nonlinear Results --- \n"
      f"Train Pred Acc Mean: {mean(FAE_pred_train_acc_mean_niter[0:20]):.4f}; "
      f"Train Pred Acc SD: {std(FAE_pred_train_acc_mean_niter[0:20]):.4f}; \n"
      f"Test Pred Acc Mean: {mean(FAE_pred_test_acc_mean_niter[0:20]):.4f}; "
      f"Test Pred Acc SD: {std(FAE_pred_test_acc_mean_niter[0:20]):.4f}; \n"
      f"Train Classification Acc Mean: {mean(classification_FAE_train_niter[0:20]):.4f}; "
      f"Train Classification Acc SD: {std(classification_FAE_train_niter[0:20]):.4f}; \n"
      f"Test Classification Acc Mean: {mean(classification_FAE_test_niter[0:20]):.4f}; "
      f"Test Classification Acc SD: {std(classification_FAE_test_niter[0:20]):.4f}; \n")