import pandas as pd
import numpy as np
from numpy import *
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
# matplotlib.use('TkAgg')
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
import random
from random import seed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans

# os.chdir('C:/FAE')
os.chdir('C:/Users/Sidi/Desktop/FAE/FAE')
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import DataGenerator
from DataGenerator import *
import DataGenerator_NN
from DataGenerator_NN import *
import Functions
from Functions import *

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

# Simulated Data
nc=500
classes = 10
tpts = np.linspace(0,1,21)
x_raw,curves = SmoothDataGenerator(nc, tpts, classes,0.5)
label = np.repeat(0,nc*classes)
for j in range(1,classes):
    label[(j*nc):(j+1)*nc] = np.repeat(j,nc)

# Simulate Data using NN DataGenerator
n_sample = 1000
n_class = 3
tpts_raw = np.linspace(0,1,21)
sim_x, sim_x_noise, sim_labels = DataGenerateor_NN(n_sample=n_sample, n_class=n_class, n_rep=5, class_weight=[.3,.3,.4],
                                                   n_basis = 10, basis_type = "BSpline", decoder_hidden = [10],
                                                   time_grid = tpts_raw,activation_function = nn.Sigmoid(), noise=1)
x_raw = sim_x_noise.detach().numpy()
label =sim_labels.copy().astype(int64)
# Simulate Data using NN DataGenerator for diff dist
mean = [[0,0,0,0,0], [0,0,0,0,0], [2,1,0,1,2]]
cov = [[[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]],
       [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]],
       [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]]
tpts_raw = np.linspace(0,1,21)
sim_x_dist, sim_x_noise_dist, sim_labels_dist  = DataGenerateor_Dist_NN(n_sample_per_class=400, n_class=3, n_rep=5,
                                                                        mean=mean, cov=cov,
                                                                        n_basis = 10, basis_type = "BSpline",
                                                                        decoder_hidden = [10],
                                                                        time_grid = np.linspace(0,1,21),
                                                                        activation_function = nn.Sigmoid(),
                                                                        noise=1)
x_raw = sim_x_noise_dist.detach().numpy()
label =sim_labels_dist.copy().astype(int64)

#####################################
# Pre-process Data sets
#####################################
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
# Perform FPCA
#####################################
# niter = 20
# seed(1432)
# niter_seed = random.sample(range(5000), niter)
niter = 10
seed(743)
niter_seed = random.sample(range(1000), niter)

# Set up parameters
n_basis = 15
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

# Set up empty lists
FPCA_train_no_niter = []
fpc_scores_train_niter = []
fpc_scores_test_niter = []
fpc_scores_all_niter = []
FPCA_pred_test_niter = []
FPCA_pred_all_niter = []
FPCA_pred_train_acc_mean_niter = []
FPCA_pred_test_acc_mean_niter = []
FPCA_pred_train_acc_sd_niter = []
FPCA_pred_test_acc_sd_niter = []
classification_FPCA_train_niter = []
classification_FPCA_test_niter = []
clustering_FPCA_acc_niter = []
clustering_FPCA_acc_mean_niter = []
clustering_FPCA_acc_sd_niter = []

# Start iterations
for i in range(niter):
    # Split training/test set
    TrainData, TestData, TrainLabel, TestLabel, train_no = train_test_split(x, label, split_rate =0.8, seed_no=niter_seed[i])
    FPCA_train_no_niter.append(train_no)

    tpts_fd = tpts.numpy().flatten()
    fd_train = representation.grid.FDataGrid(TrainData.numpy(), tpts_fd)
    fd_test = representation.grid.FDataGrid(TestData.numpy(), tpts_fd)
    basis_fd_train = fd_train.to_basis(bss_fpca)
    basis_fd_test = fd_test.to_basis(bss_fpca)
    # basis_fd = fd.to_basis(representation.basis.BSpline(n_basis=80, order=4))
    fpca_basis = fda.preprocessing.dim_reduction.feature_extraction.FPCA(n_components=n_rep)
    # Get FPCs
    # fpca_basis_fd = fpca_basis.fit(fd)
    fpca_basis = fpca_basis.fit(basis_fd_train)
    # fpca_basis.components_.plot()
    # fpca_basis.explained_variance_

    # Get FPC scores
    fpc_scores_test = fpca_basis.transform(basis_fd_test)
    fpc_scores_test_niter.append(fpc_scores_test)
    FPCA_pred = fpca_basis.inverse_transform(fpc_scores_test)._evaluate(tpts_fd)[:,:,0]
    FPCA_pred_test_niter.append(FPCA_pred)

    # Get mean function
    # fpca_basis.mean_.plot()
    # fpca_mean = fpca_basis.mean_.to_grid().numpy()
    #fpca_basis.singular_values_`

    # fpca_basis.components_[0].plot(label = 'FPC1-FPCA')
    # plt.plot(tpts, pc1, label='FPC1-FAE-encoder')
    # plt.plot(tpts, pc1_hat, label="FPC1-FAE-decoder")
    # plt.xlabel('time grid')
    # plt.title(f"Basis#={n_basis}, FPC#={n_rep}, lamb={lamb}")
    # plt.legend()
    # plt.show()
    # plt.close()

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
    # Evaluate the classifier with the test set
    # FPCA_classifier.predict(fpc_scores_test)
    # Classification accuracy on the test set
    classification_FPCA_test_niter.append(FPCA_classifier.score(fpc_scores_test, TestLabel))
    # Classification accuracy on the training set
    classification_FPCA_train_niter.append(FPCA_classifier.score(fpc_scores_train, TrainLabel))

    ## Clustering
    optimal_n_cluster = len(np.unique(label)) #len(set(label))
    kmeans_par = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 0}
    kmeans_labels_FPCA = KMeans(n_clusters=optimal_n_cluster, **kmeans_par).fit_predict(fpc_scores_all)
    # for i in range(optimal_n_cluster):
    #     no_FPCA = np.count_nonzero(kmeans_labels_FPCA == i)
    acc_list_FPCA = []
    label_list_FPCA = []
    for j in range(optimal_n_cluster):
        ind = indices(label, lambda x: x==np.unique(label)[j])
        acc_list_FPCA.append(acc(kmeans_labels_FPCA[ind], optimal_n_cluster))
        label_list_FPCA.append(most_frequent(kmeans_labels_FPCA[ind], optimal_n_cluster))
    clustering_FPCA_acc_niter.append(acc_list_FPCA)
    clustering_FPCA_acc_mean_niter.append(mean(acc_list_FPCA))
    clustering_FPCA_acc_sd_niter.append(std(acc_list_FPCA))

# Print for result tables
print("--- FPCA Results --- \n"
      f"Train Pred Acc Mean: {mean(FPCA_pred_train_acc_mean_niter):.4f}; "
      f"Train Pred Acc SD: {std(FPCA_pred_train_acc_mean_niter):.4f}; \n"
      f"Test Pred Acc Mean: {mean(FPCA_pred_test_acc_mean_niter):.4f}; "
      f"Test Pred Acc SD: {std(FPCA_pred_test_acc_mean_niter):.4f}; \n"
      f"Train Classification Acc Mean: {mean(classification_FPCA_train_niter):.4f}; "
      f"Train Classification Acc SD: {std(classification_FPCA_train_niter):.4f}; \n"
      f"Test Classification Acc Mean: {mean(classification_FPCA_test_niter):.4f}; "
      f"Test Classification Acc SD: {std(classification_FPCA_test_niter):.4f}; \n"
      f"Clustering Acc Mean (by clusters): {np.around(mean(clustering_FPCA_acc_niter, axis=0), 4)};\n" 
      #[round(i, 4) for i in mean(clustering_FPCA_acc_niter, axis=0)] or [f"{num:.4f}" for num in mean(clustering_FPCA_acc_niter, axis=0)]
      f"Overall Clustering Acc Mean: {mean(clustering_FPCA_acc_mean_niter):.4f};")

# Plot of raw curves vs. recovered curves for the i-th iteration
i=1
TestData = x[[j for j in range(len(x)) if j not in FPCA_train_no_niter[i]]]
input_plt = TestData.detach().numpy()
plt.figure(4, figsize=(10, 20))
plt.subplot(211)
for m in range(0, len(input_plt)):
# for m in id_plt:
    plt.plot(tpts, input_plt[m])
plt.title("Input Curves")
plt.subplot(212)
for m in range(0, len(FPCA_pred_test_niter[i])):
# for m in id_plt:
    plt.plot(tpts, FPCA_pred_test_niter[i][m])
plt.title("Output Curves")
plt.show()

