"""
This script contains the code for implementing the conventional autoencoders (AE) with regularly spaced functional datq
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
from datetime import datetime
import matplotlib.ticker as mtick

os.chdir('~/Code')
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import Functions
from Functions import *

##########################################################
# Define the AE architecture for regularly functional data
# Create AE Class
##########################################################
# Below are the settings for the creating the linear FAE (adjust the architecure for nonlinear FAE)
class AE(nn.Module):
    def __init__(self, weight_std=None):
        super(AE, self).__init__()
        ## Select one of the following options (comment out the other one)
        # Opt 1: Linear AE with 1 hidden layer
        self.fc1 = nn.Linear(n_tpts, n_rep, bias=False)
        self.fc2 = nn.Linear(n_rep, n_tpts, bias=False)
        self.activation = nn.Identity()

        # Opt 2: Nonlinear AE with 3 hidden layers
        # self.fc1 = nn.Linear(n_tpts, 10, bias=False)
        # self.fc2 = nn.Linear(10, n_rep, bias=False)
        # self.fc3 = nn.Linear(n_rep, 10, bias=False)
        # self.fc4 = nn.Linear(10, n_tpts, bias=False)
        # self.activation = nn.Sigmoid()

        # initialize the weights to a specified, constant value
        if (weight_std is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=weight_std)
                    #nn.init.constant_(m.bias, 0)

    def forward(self, x):
        ## Select one of the following options accordingly (comment out the other one)
        # Opt 1: Linear FAE with 1 hidden layer
        rep = self.activation(self.fc1(x))
        x_hat = self.activation(self.fc2(rep))

        # Opt 2: Nonlinear inear FAE with 3 hidden layers
        # h1 = self.activation(self.fc1(x))
        # rep = self.fc2(h1)
        # h3 = self.activation(self.fc3(rep))
        # x_hat = self.fc4(h3)

        return x_hat, rep

#####################################
# Define the training procedure
#####################################
# The function for training AE
def train_AE(train_loader):
    AE_model.train()
    train_loss = 0
    for i, data in enumerate(train_loader):
        AE_optimizer.zero_grad()  # The gradients are set to zero
        input = data.to(device)
        out, rep = AE_model(input.float())
        # Loss for back-propagation
        loss = loss_function(out, input.float())
        loss.backward()  # The gradient is computed and stored.
        AE_optimizer.step()  # Performs parameter update
        train_loss += loss
    return train_loss

# The function for predicting observations with trained AE
def pred_AE(data):
    AE_model.eval()
    input = data.to(device)
    output, rep = AE_model(input.float())
    loss = loss_function(output, input.float())
    return output, rep, loss

#####################################
# Perform AE (Model Training)
#####################################
# Below are the settings for the implementation with ElNino data described in the real application section
niter = 20
seed(743)
niter_seed = random.sample(range(5000), niter)
# niter = 10
# seed(743)
# niter_seed = random.sample(range(1000), niter)

# Set up lists to save training info
AE_train_no_niter = []
AE_reps_train_niter = []
AE_reps_test_niter = []
AE_reps_all_niter = []
AE_pred_test_niter = []
AE_pred_all_niter = []
AE_pred_train_acc_mean_niter = []
AE_pred_test_acc_mean_niter = []
AE_pred_train_acc_sd_niter = []
AE_pred_test_acc_sd_niter = []
classification_AE_train_niter = []
classification_AE_test_niter = []

# Set up AE's hyperparameters
n_rep = 5 # number of representation
epochs = 5000 # epochs
batch_size = 28 # batch size
init_weight_sd = 0.5 # SD of normal dist. for initializing NN weight
split_rate = 0.8 # percentage of training set

# Set up lists for training history
AE_reg_test_acc_epoch = [[] for x in range(int(epochs/100))]
classification_AE_reg_test_epoch = [[] for x in range(int(epochs/100))]

# Start iterations
for i in range(niter):
    # Split training/test set
    TrainData, TestData, TrainLabel, TestLabel, train_no = train_test_split(x, label, split_rate = split_rate, seed_no=niter_seed[i])
    AE_train_no_niter.append(train_no)
    # Define data loaders; DataLoader is used to load the dataset for training
    train_loader = torch.utils.data.DataLoader(TrainData, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(TestData)

    # Model Initialization
    AE_model = AE(weight_std=init_weight_sd)
    loss_function = nn.MSELoss()
    AE_optimizer = optim.Adam(AE_model.parameters(), lr=1e-2, weight_decay=1e-6)
    device = torch.device("cpu")

    epochs = epochs
    # Train model
    for epoch in range(1, epochs + 1):
        loss = train_AE(train_loader)
        AE_pred_test, AE_reps_test, AE_pred_loss_test = pred_AE(TestData)
        if epoch % 100 == 0:
            print(f"Epoch[{epoch}]-loss: {loss:.4f}, pred_loss: {AE_pred_loss_test:.4f}")
            AE_reg_test_acc_epoch[int(epoch / 100) - 1].append(AE_pred_loss_test.tolist())
            AE_reps_train_temp = pred_AE(TrainData)[1]
            AE_classifier_temp = LogisticRegression(solver='liblinear', random_state=0, multi_class='auto').fit(
                AE_reps_train_temp.detach().numpy(), TrainLabel)
            classification_AE_reg_test_epoch[int(epoch / 100) - 1].append(
                AE_classifier_temp.score(AE_reps_test.detach().numpy(), TestLabel))

    AE_reps_test_niter.append(AE_reps_test)
    AE_pred_test_niter.append(AE_pred_test)
    AE_pred_all, AE_reps_all = pred_AE(x)[0:2]
    AE_reps_all_niter.append(AE_reps_all)
    AE_pred_all_niter.append(AE_pred_all)

    AE_pred_test_acc_mean_niter.append(AE_pred_loss_test.tolist())
    AE_pred_test_acc_sd_niter.append(eval_mse_sdse(TestData, AE_pred_test)[1].tolist())

    AE_pred_train, AE_reps_train, AE_pred_loss_train = pred_AE(TrainData)
    AE_reps_train_niter.append(AE_reps_train)
    AE_pred_train_acc_mean_niter.append(AE_pred_loss_train.tolist())
    AE_pred_train_acc_sd_niter.append(eval_mse_sdse(TrainData, AE_pred_train)[1].tolist())

    ## Classification
    # Create classifiers (logistic regression) & train the model with the training set
    AE_classifier = LogisticRegression(solver='liblinear', random_state=0, multi_class='auto').fit(AE_reps_train.detach().numpy(), TrainLabel)
    # Classification accuracy on the test set
    classification_AE_test_niter.append(AE_classifier.score(AE_reps_test.detach().numpy(), TestLabel))
    # Classification accuracy on the training set
    classification_AE_train_niter.append(AE_classifier.score(AE_reps_train.detach().numpy(), TrainLabel))

    print(f"Replicate {i + 1} is complete.")


# Print for result tables
print("--- AE-Nonlinear Results --- \n"
      f"Train Pred Acc Mean: {mean(AE_pred_train_acc_mean_niter):.4f}; "
      f"Train Pred Acc SD: {std(AE_pred_train_acc_mean_niter):.4f}; \n"
      f"Test Pred Acc Mean: {mean(AE_pred_test_acc_mean_niter):.4f}; "
      f"Test Pred Acc SD: {std(AE_pred_test_acc_mean_niter):.4f}; \n"
      f"Train Classification Acc Mean: {mean(classification_AE_train_niter):.4f}; "
      f"Train Classification Acc SD: {std(classification_AE_train_niter):.4f}; \n"
      f"Test Classification Acc Mean: {mean(classification_AE_test_niter):.4f}; "
      f"Test Classification Acc SD: {std(classification_AE_test_niter):.4f}; \n")
