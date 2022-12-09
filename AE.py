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
        self.activation = nn.Identity()

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

#####################################
# Load Data sets
#####################################
# Import dataset
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
# Perform AE (Model Training)
#####################################
niter = 20
seed(1432)
niter_seed = random.sample(range(5000), niter)

# Set up parameters
n_rep = 5

# Set up empty lists
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
clustering_AE_acc_niter = []
clustering_AE_acc_mean_niter = []
clustering_AE_acc_sd_niter = []

# Start iterations
for i in range(niter):
    # Split training/test set
    TrainData, TestData, TrainLabel, TestLabel, train_no = train_test_split(x, label, split_rate =0.8, seed_no=niter_seed[i])
    # Define data loaders; DataLoader is used to load the dataset for training
    train_loader = torch.utils.data.DataLoader(TrainData, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(TestData)

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

    epochs = 10000
    # AE_losses = []

    # Train model
    for epoch in range(1, epochs + 1):
        loss = train_AE(epoch, n_tpts=n_tpts, n_rep=n_rep)
        # AE_losses.append(loss.detach().numpy())
        AE_pred_test, AE_reps_test, AE_pred_loss_test = pred_AE(TestData)
        if epoch % 100 == 0:
            print(f"Epoch[{epoch}]-loss: {loss:.4f}, pred_loss: {AE_pred_loss_test:.4f}")

    # Debug by looking at loss
    # plt.plot(AE_losses, label = "train_loss")
    # plt.legend()
    # plt.show()
    # plt.close()

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
    # Evaluate the classifier with the test set
    # AE_classifier.predict(AE_reps_test)
    # Classification accuracy on the test set
    classification_AE_test_niter.append(AE_classifier.score(AE_reps_test.detach().numpy(), TestLabel))
    # Classification accuracy on the training set
    classification_FAE_train_niter.append(AE_classifier.score(AE_reps_train.detach().numpy(), TrainLabel))

    ## Clustering
    optimal_n_cluster = len(np.unique(label))  # len(set(label))
    kmeans_par = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 0}
    kmeans_labels_AE = KMeans(n_clusters=optimal_n_cluster, **kmeans_par).fit_predict(AE_reps_all.detach().numpy())
    # for i in range(optimal_n_cluster):
    #     no_AE = np.count_nonzero(kmeans_labels_AE == i)
    acc_list_AE = []
    label_list_AE = []
    for j in range(optimal_n_cluster):
        ind = indices(label, lambda x: x == np.unique(label)[j])
        acc_list_AE.append(acc(kmeans_labels_AE[ind], optimal_n_cluster))
        label_list_AE.append(most_frequent(kmeans_labels_AE[ind], optimal_n_cluster))
    clustering_AE_acc_niter.append(acc_list_AE)
    clustering_AE_acc_mean_niter.append(mean(acc_list_AE))
    clustering_AE_acc_sd_niter.append(std(acc_list_AE))

# If activation function is nn.Identity()
AE_identity_train_no_niter = AE_train_no_niter.copy()
AE_identity_reps_train_niter = AE_reps_train_niter.copy()
AE_identity_reps_test_niter = AE_reps_test_niter.copy()
AE_identity_reps_all_niter = AE_reps_all_niter.copy()
AE_identity_pred_test_niter = AE_pred_test_niter.copy()
AE_identity_pred_all_niter = AE_pred_all_niter.copy()
AE_identity_pred_train_acc_mean_niter = AE_pred_train_acc_mean_niter.copy()
AE_identity_pred_test_acc_mean_niter = AE_pred_test_acc_mean_niter.copy()
AE_identity_pred_train_acc_sd_niter = AE_pred_train_acc_sd_niter.copy()
AE_identity_pred_test_acc_sd_niter = AE_pred_test_acc_sd_niter.copy()
classification_AE_identity_train_niter = classification_AE_train_niter.copy()
classification_AE_identity_test_niter = classification_AE_test_niter.copy()
clustering_AE_identity_acc_niter = clustering_AE_acc_niter.copy()
clustering_AE_identity_acc_mean_niter = clustering_AE_acc_mean_niter.copy()
clustering_AE_identity_acc_sd_niter = clustering_AE_acc_sd_niter.copy()

# Plot of Input (Observed Curves) & Output Curves (Predicted Curves)
i=1
TestData = x[[i for i in range(len(x)) if i not in train_no[i]]]
input_plt = TestData.detach().numpy()
AE_pred_plt = AE_pred_test_niter[i].detach().numpy()

plt.figure(3, figsize=(10, 20))
plt.subplot(211)
for m in range(0, len(input_plt)):
# for m in id_plt:
    plt.plot(tpts, input_plt[m], "b")
plt.title("Input Curves")
plt.subplot(212)
for m in range(0, len(AE_pred_plt)):
# for m in id_plt:
    plt.plot(tpts, AE_pred_plt[m])
plt.title("Output Curves (AE)")
plt.show()

