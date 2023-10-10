"""
This script contains the code for implementing the functional principal component analysis (FPCA) in the manuscript "Autoencoders for Discrete Functional Data Representation Learning and Smoothing".

@author: Sidi Wu
"""

# Import modeuls
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
from scipy.interpolate import BSpline
import ignite
import os
import sklearn
from sklearn.decomposition import PCA
import random
from random import seed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import time
from time import process_time
from datetime import datetime

os.chdir("~/Code")
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import Functions
from Functions import *

#####################################
# Perform FPCA
#####################################
# Below are the settings for the implementation with ElNino data described in the real application section
niter = 20
seed(743)
niter_seed = random.sample(range(5000), niter)
# niter = 10
# seed(743)
# niter_seed = random.sample(range(1000), niter)

# Set up parameters
n_basis = 10
n_rep = 5
basis_type = "Bspline"
if  len(tpts)*0.9 < n_basis:
    n_basis_fpca = 10
else:
    n_basis_fpca = n_basis
if basis_type == "Bspline":
    bss_fpca = representation.basis.BSpline(n_basis=n_basis_fpca, order=4)
elif basis_type == "Fourier":
    bss_fpca = representation.basis.Fourier([min(tpts.numpy().flatten()), max(tpts.numpy().flatten())],
                                       n_basis=n_basis_fpca)
tpts_FAE_plot = torch.tensor(np.arange(0, 1 + 1 / 100, 1 / 100))

# Set up lists to save training info
FPCA_train_no_niter = []
fpc_scores_train_niter = []
fpc_scores_test_niter = []
fpc_scores_all_niter = []
FPCA_pred_test_niter = []
FPCA_pred_test_plot_niter = []
FPCA_pred_all_niter = []
FPCA_pred_train_acc_mean_niter = []
FPCA_pred_test_acc_mean_niter = []
FPCA_pred_train_acc_sd_niter = []
FPCA_pred_test_acc_sd_niter = []
classification_FPCA_train_niter = []
classification_FPCA_test_niter = []

# Start iterations
for i in range(niter):
    # Split training/test set
    TrainData, TestData, TrainLabel, TestLabel, train_no = train_test_split(x, label, split_rate=0.8,seed_no=niter_seed[i])
    FPCA_train_no_niter.append(train_no)

    tpts_fd = tpts.numpy().flatten()
    fd_train = representation.grid.FDataGrid(TrainData.numpy(), tpts_fd)
    fd_test = representation.grid.FDataGrid(TestData.numpy(), tpts_fd)
    basis_fd_train = fd_train.to_basis(bss_fpca)
    basis_fd_test = fd_test.to_basis(bss_fpca)
    fpca_basis = fda.preprocessing.dim_reduction.feature_extraction.FPCA(n_components=n_rep)
    # Get FPCs
    fpca_basis = fpca_basis.fit(basis_fd_train)

    # Get FPC scores
    fpc_scores_test = fpca_basis.transform(basis_fd_test)
    fpc_scores_test_niter.append(fpc_scores_test)
    FPCA_pred = fpca_basis.inverse_transform(fpc_scores_test)._evaluate(tpts_fd)[:,:,0]
    FPCA_pred_test_niter.append(FPCA_pred)
    FPCA_pred_plot = fpca_basis.inverse_transform(fpc_scores_test)._evaluate(tpts_FAE_plot)[:,:,0]
    FPCA_pred_test_plot_niter.append(FPCA_pred_plot)

    # FPCA representation for all subjects and training subjects
    fd_all = representation.grid.FDataGrid(x.numpy(), tpts_fd)
    basis_fd_all = fd_all.to_basis(bss_fpca)
    fpc_scores_all = fpca_basis.transform(basis_fd_all)
    fpc_scores_all_niter.append(fpc_scores_all)
    fpc_scores_train = fpc_scores_all[train_no]
    fpc_scores_train_niter.append(fpc_scores_train)
    FPCA_pred_train = fpca_basis.inverse_transform(fpc_scores_train)._evaluate(tpts_fd)[:, :, 0]
    FPCA_pred_all_niter.append(fpca_basis.inverse_transform(fpc_scores_all)._evaluate(tpts_fd)[:, :, 0])

    # Calculate prediction accuracy
    FPCA_pred_test_acc_mean_niter.append(eval_mse_sdse(TestData, FPCA_pred)[0].tolist())
    FPCA_pred_test_acc_sd_niter.append(eval_mse_sdse(TestData, FPCA_pred)[1].tolist())
    FPCA_pred_train_acc_mean_niter.append(eval_mse_sdse(TrainData, FPCA_pred_train)[0].tolist())
    FPCA_pred_train_acc_sd_niter.append(eval_mse_sdse(TrainData, FPCA_pred_train)[1].tolist())

    ## Classification
    # Create classifiers (logistic regression) & train the model with the training set
    FPCA_classifier = LogisticRegression(solver='liblinear', random_state=0, multi_class='auto').fit(fpc_scores_train,TrainLabel)
    # Classification accuracy on the test set
    classification_FPCA_test_niter.append(FPCA_classifier.score(fpc_scores_test, TestLabel))
    # Classification accuracy on the training set
    classification_FPCA_train_niter.append(FPCA_classifier.score(fpc_scores_train, TrainLabel))

# Print for result tables
print("--- FPCA Results --- \n"
      f"Train Pred Acc Mean: {mean(FPCA_pred_train_acc_mean_niter):.4f}; "
      f"Train Pred Acc SD: {std(FPCA_pred_train_acc_mean_niter):.4f}; \n"
      f"Test Pred Acc Mean: {mean(FPCA_pred_test_acc_mean_niter):.4f}; "
      f"Test Pred Acc SD: {std(FPCA_pred_test_acc_mean_niter):.4f}; \n"
      f"Train Classification Acc Mean: {mean(classification_FPCA_train_niter):.4f}; "
      f"Train Classification Acc SD: {std(classification_FPCA_train_niter):.4f}; \n"
      f"Test Classification Acc Mean: {mean(classification_FPCA_test_niter):.4f}; "
      f"Test Classification Acc SD: {std(classification_FPCA_test_niter):.4f}; \n")
