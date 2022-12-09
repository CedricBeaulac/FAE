# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 14:49:58 2022

@author: Sidi Wu and Cédric Beaulac

Functional autoencoder implementation
"""

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
# import skfda as fda
# from skfda import representation as representation
# from skfda.exploratory.visualization import FPCAPlot
# # from skfda.exploratory.visualization import FPCAPlot
# # from skfda.preprocessing.dim_reduction import FPCA
# # from skfda.representation.basis import BSpline, Fourier, Monomial
import scipy
from scipy.interpolate import BSpline
import ignite
import os
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import random
from random import seed

os.chdir('C:/FAE')
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import DataGenerator
from DataGenerator import *
import Functions
from Functions import *

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
        self.fc1 = nn.Linear(n_basis, 10,bias=False)
        self.fc2 = nn.Linear(10, n_rep, bias=False)
        # self.fc1 = nn.Linear(n_basis, n_rep, bias=False)
        self.fc3 = nn.Linear(n_rep, 10, bias=False)
        # self.fc3 = nn.Linear(n_rep, n_basis, bias=False)
        self.fc4 = nn.Linear(10, n_basis, bias=False)
        self.activation = nn.Softplus()

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
        w = x.size(1)-1
        W = torch.tensor([1/(2*w)]+[1/w]*(w-1)+[1/(2*w)])
        s = torch.matmul(torch.mul(x, W), torch.t(basis_fc))
        return s
    def Revert(self, x, basis_fc):
        f = torch.matmul(x, basis_fc)
        return f

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
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import DataGenerator
from DataGenerator import *
import Functions
from Functions import *
# Dataset: tecator
# x_raw = pd.read_csv('Datasets/tecator/tecator.csv')
# tpts_raw = pd.read_csv('Datasets/tecator/tecator_tpts.csv')
# Dataset: pinch
# x_raw = pd.read_csv('Datasets/pinch/pinch.csv')
# tpts_raw = pd.read_csv('Datasets/pinch/pinch_tpts.csv')
# Dataset: ElNino
x_raw = pd.read_csv('Datasets/ElNino/ElNino_ERSST.csv')
tpts_raw = pd.read_csv('Datasets/ElNino/ElNino_ERSST_tpts.csv')
label_table = pd.read_csv('Datasets/ElNino/ElNino_ERSST_label.csv')
label = label_table.x.to_numpy()

# Simulated Data
nc=500
classes = 10
tpts = np.linspace(0,1,21)
x_raw,curves = SmoothDataGenerator(nc, tpts, classes,0.5)
label = np.repeat(0,nc*classes)
for j in range(1,classes):
    label[(j*nc):(j+1)*nc] = np.repeat(j,nc)

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

#####################################
# Perform FAE (Model Training)
#####################################
niter = 20
seed(1432)
niter_seed = random.sample(range(5000), niter)

# Set up parameters
n_basis = 20
n_rep = 5
lamb = 0.00
pen = "feature"
basis_type = "Bspline"
# Get basis functions evaluated
if basis_type == "Bspline":
    bss = representation.basis.BSpline(n_basis=n_basis, order=4)
elif basis_type == "Fourier":
    bss = representation.basis.Fourier([min(tpts.numpy().flatten()), max(tpts.numpy().flatten())], n_basis=n_basis)

bss_eval = bss.evaluate(tpts, derivative=0)
basis_fc = torch.from_numpy(bss_eval[:, :, 0]).float()

# Set up empty lists
FAE_train_no_niter = []
FAE_reps_train_niter = []
FAE_reps_test_niter = []
FAE_reps_all_niter = []
FAE_pred_test_niter = []
FAE_pred_all_niter = []
FAE_pred_train_acc_mean_niter = []
FAE_pred_test_acc_mean_niter = []
FAE_pred_train_acc_sd_niter = []
FAE_pred_test_acc_sd_niter = []
classification_FAE_train_niter = []
classification_FAE_test_niter = []
clustering_FAE_acc_niter = []
clustering_FAE_acc_mean_niter = []
clustering_FAE_acc_sd_niter = []

# Start iterations
for i in range(niter):
    # Split training/test set
    TrainData, TestData, TrainLabel, TestLabel, train_no = train_test_split(x, label, split_rate =0.8, seed_no=niter_seed[i])
    FAE_train_no_niter.append(train_no)
    # Define data loaders; DataLoader is used to load the dataset for training
    train_loader = torch.utils.data.DataLoader(TrainData, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(TestData)

    # Model Initialization
    model = FAE_vanilla(weight_std=1)
    # Validation using MSE Loss function
    loss_function = nn.MSELoss()
    # Using an Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
    # Set to CPU/GPU
    device = torch.device("cpu")  # (?)should be CUDA when running on the big powerfull server

    epochs = 10000
    # losses = []
    # score_losses = []

    # Train model
    for epoch in range(1, epochs + 1):
        loss, score_loss = train(train_loader=train_loader, pen=pen, lamb=lamb)
        # losses.append(loss.detach().numpy())
        # score_losses.append(score_loss.detach().numpy())
        #if epoch == epochs:
        FAE_pred_test, FAE_reps_test, FAE_pred_loss_test, FAE_pred_score_loss_test = pred(model, TestData)
        if epoch % 1000 == 0:
            print(f"Epoch[{epoch}]-loss: {loss:.4f}; feature loss: {score_loss: 4f}; pred_loss:{FAE_pred_loss_test:4f}")

    # Debug by looking at loss
    # plt.plot(losses, label = "train_loss")
    # plt.legend()
    # plt.show()
    # plt.close()

    # Debug by looking at the FAE, layer by layer
    # s = model.Project(input,basis_fc)
    # rep = model.activation(model.fc1(s))
    # s_hat = model.activation(model.fc3(rep))
    # output = model.Revert(s_hat,basis_fc)

    FAE_reps_test_niter.append(FAE_reps_test)
    FAE_pred_test_niter.append(FAE_pred_test)
    FAE_pred_all, FAE_reps_all = pred(model, x)[0:2]
    FAE_reps_all_niter.append(FAE_reps_all)
    FAE_pred_all_niter.append(FAE_pred_all)

    # FAE_pred_test_acc_mean_niter.append(eval_mse_sdse(TestData, FAE_pred_test)[0].tolist())
    FAE_pred_test_acc_mean_niter.append(FAE_pred_loss_test.tolist())
    FAE_pred_test_acc_sd_niter.append(eval_mse_sdse(TestData, FAE_pred_test)[1].tolist())

    FAE_pred_train, FAE_reps_train, FAE_pred_loss_train, FAE_pred_score_loss_train = pred(model, TrainData)
    FAE_reps_train_niter.append(FAE_reps_train)
    FAE_pred_train_acc_mean_niter.append(FAE_pred_loss_train.tolist())
    FAE_pred_train_acc_sd_niter.append(eval_mse_sdse(TrainData, FAE_pred_train)[1].tolist())

    ## Classification
    # Create classifiers (logistic regression) & train the model with the training set
    FAE_classifier = LogisticRegression(solver='liblinear', random_state=0, multi_class='auto').fit(FAE_reps_train.detach().numpy(), TrainLabel)
    # Evaluate the classifier with the test set
    # FAE_classifier.predict(FAE_reps_test)
    # Classification accuracy on the test set
    classification_FAE_test_niter.append(FAE_classifier.score(FAE_reps_test.detach().numpy(), TestLabel))
    # Classification accuracy on the training set
    classification_FAE_train_niter.append(FAE_classifier.score(FAE_reps_train.detach().numpy(), TrainLabel))

    ## Clustering
    optimal_n_cluster = len(np.unique(label)) #len(set(label))
    kmeans_par = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 0}
    kmeans_labels_FAE = KMeans(n_clusters=optimal_n_cluster, **kmeans_par).fit_predict(FAE_reps_all.detach().numpy())
    # for i in range(optimal_n_cluster):
    #     no_FAE = np.count_nonzero(kmeans_labels_FAE == i)
    acc_list_FAE = []
    label_list_FAE = []
    for j in range(optimal_n_cluster):
        ind = indices(label, lambda x: x==np.unique(label)[j])
        acc_list_FAE.append(acc(kmeans_labels_FAE[ind], optimal_n_cluster))
        label_list_FAE.append(most_frequent(kmeans_labels_FAE[ind], optimal_n_cluster))
    clustering_FAE_acc_niter.append(acc_list_FAE)
    clustering_FAE_acc_mean_niter.append(mean(acc_list_FAE))
    clustering_FAE_acc_sd_niter.append(std(acc_list_FAE))


# If activation function is nn.Identity()
FAE_identity_train_no_niter = FAE_train_no_niter.copy()
FAE_identity_reps_train_niter = FAE_reps_train_niter.copy()
FAE_identity_reps_test_niter = FAE_reps_test_niter.copy()
FAE_identity_reps_all_niter = FAE_reps_all_niter.copy()
FAE_identity_pred_test_niter = FAE_pred_test_niter.copy()
FAE_identity_pred_all_niter = FAE_pred_all_niter.copy()
FAE_identity_pred_train_acc_mean_niter = FAE_pred_train_acc_mean_niter.copy()
FAE_identity_pred_test_acc_mean_niter = FAE_pred_test_acc_mean_niter.copy()
FAE_identity_pred_train_acc_sd_niter = FAE_pred_train_acc_sd_niter.copy()
FAE_identity_pred_test_acc_sd_niter = FAE_pred_test_acc_sd_niter.copy()
classification_FAE_identity_train_niter = classification_FAE_train_niter.copy()
classification_FAE_identity_test_niter = classification_FAE_test_niter.copy()
clustering_FAE_identity_acc_niter = clustering_FAE_acc_niter.copy()
clustering_FAE_identity_acc_mean_niter = clustering_FAE_acc_mean_niter.copy()
clustering_FAE_identity_acc_sd_niter = clustering_FAE_acc_sd_niter.copy()

# Plot of Input (Observed Curves) & Output Curves (Predicted Curves)
i=1
TestData = x[[i for i in range(len(x)) if i not in train_no[i]]]
input_plt = TestData.detach().numpy()
FAE_pred_plt = FAE_pred_test_niter[i].detach().numpy()

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

