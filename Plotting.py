import pandas as pd
import numpy as np
from numpy import *
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
import itertools

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

# Plot of Input (Observed Curves) & Output Curves (Predicted Curves)
i=1
TestData = x[[i for i in range(len(x)) if i not in train_no[i]]]
input_plt = TestData.detach().numpy()
FAE_pred_plt = FAE_pred_test_niter[i].detach().numpy()
AE_pred_plt = AE_pred_test_niter[i].detach().numpy()
FPCA_pred_plt = FPCA_pred_test_niter[i].detach().numpy()

plt.figure(3, figsize=(20, 20))
plt.subplot(221)
for m in range(0, len(input_plt)):
# for m in id_plt:
    plt.plot(tpts, input_plt[m], "b")
plt.title("Input Curves")
plt.subplot(222)
for m in range(0, len(AE_pred_plt)):
# for m in id_plt:
    plt.plot(tpts, AE_pred_plt[m])
plt.title("Output Curves (AE)")
plt.subplot(223)
for m in range(0, len(FPCA_pred_plt)):
# for m in id_plt:
    plt.plot(tpts, FPCA_pred_plt[m])
plt.title("Output Curves (FPCA)")
plt.subplot(224)
for m in range(0, len(FAE_pred_plt)):
# for m in id_plt:
    plt.plot(tpts, FAE_pred_plt[m])
plt.title("Output Curves (FAE)")
plt.show()


# Curves of raw, FAE-recoverd, FPCA-recoverd, AE-recovered for some selected subjects
plt.ioff()
pdf = PdfPages("Datasets/ElNino/ElNino_2layers(20-10-5)_nonlinear(Softplus)_+AE2.pdf")
# pdf = matplotlib.backends.backend_pdf.PdfPages("Datasets/tecator/tecator_2layer(50-40-5)_nbasis80_nfpcabasis10_linear_0.2Test.pdf")
for i in range(len(input_plt)): ## will open an empty extra figure :(\
    fig = plt.figure()
    plt.plot(tpts, input_plt[i], label="Raw")
    plt.plot(tpts, FAE_pred_plt[i], label="FAE-pred")
    plt.plot(tpts, FPCA_pred_plt[i], label="FPCA-pred")
    plt.plot(tpts, AE_pred_plt[i], label="FAE-pred")
    plt.legend()
    plt.title(label=f"Observation #{i+1}")
    # plt.show()
    plt.close()
    pdf.savefig(fig)
pdf.close()
plt.ion()

# Plot of clustering result
raw_color = ["b", "g", "r","y"] # 0-b, 1-g, 2-r, 3-y
# AE_color = ["g", "b", "r","y"] #1-2-0(2)-3
# FPCA_color = ["r", "y", "b", "g"] #2-3-0-1
# FAE_color = ["y", "b", "g", "r"] #1-2-3-0
AE_color = [None] * 4
FPCA_color = [None] * 4
FAE_color = [None] * 4
for i in range(len(raw_color)):
    AE_color[label_list_AE[i]] = raw_color[i]
    FPCA_color[label_list_FPCA[i]] = raw_color[i]
    FAE_color[label_list_FAE[i]] = raw_color[i]

AE_all = AE_pred_all_niter[niter-1].detach().numpy()
FAE_all = FAE_pred_all_niter[niter-1].detach().numpy()
FPCA_all = FPCA_pred_all_niter[niter-1]

plt.figure(1, figsize=(20, 20))
plt.subplot(221)
for m in range(0, len(raw)):
# for m in id_plt:
    plt.plot(tpts, raw[m], raw_color[label[m]-1])
plt.title("Original Labelling")
plt.subplot(222)
for m in range(0, len(AE_all)):
# for m in id_plt:
    plt.plot(tpts, AE_all[m], AE_color[kmeans_labels_AE[m]])
plt.title("AE-representation Labelling")
plt.subplot(223)
for m in range(0, len(FPCA_all)):
# for m in id_plt:
    plt.plot(tpts, FPCA_all[m], FPCA_color[kmeans_labels_FPCA[m]])
plt.title("FPCA-representation Labelling")
plt.subplot(224)
for m in range(0, len(FAE_all)):
# for m in id_plt:
    plt.plot(tpts, FAE_all[m], FAE_color[kmeans_labels_FAE[m]])
plt.title("FAE-representation Labelling")
plt.show()

##########################################
#### Plot Simulated Data Sets
os.chdir('C:/Users/Sidi/Desktop/FAE/FAE')
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import DataGenerator_NN
from DataGenerator_NN import *
import Functions
from Functions import *

sim_x, sim_x_noise, sim_labels, sim_reps = DataGenerateor_NN(n_sample=1000, n_class=3, n_rep=5, class_weight=[.3,.4,.3],
                                                    n_basis = 20, basis_type = "BSpline", decoder_hidden = [10],
                                                    time_grid = np.linspace(0,1,51),activation_function = nn.Sigmoid(),
                                                    noise=1)
os.chdir('C:/Users/Sidi/Desktop/FAE/FAE')
n_class=3
time_grid = np.linspace(0,1,51)
with PdfPages("Datasets/Simulations/sim_x_NN_sim6.pdf") as pdf:
    # Plot simulated curves
    sim_x_plt = sim_x.detach().numpy()
    sim_x_noise_plt = sim_x_noise.detach().numpy()
    plt.figure(1)
    for i in range(0, len(sim_x_plt)):
    # for m in id_plt:
        plt.plot(time_grid, sim_x_plt[i])
    plt.title("Simulated Curves")
    pdf.savefig()
    # plt.show()
    plt.close()

    plt.figure(2)
    for i in range(0, len(sim_x_noise_plt)):
    # for m in id_plt:
        plt.plot(time_grid, sim_x_noise_plt[i])
    plt.title("Simulated Curves with Noise")
    pdf.savefig()
    # plt.show()
    plt.close()

    label_list = []
    # Plot simulated curves by class
    for j in range(n_class):
        label_list.append([i for i in range(len(sim_labels)) if sim_labels[i] == j])
        sim_x_label_temp = sim_x_noise[label_list[j]].detach().numpy()
        plt.figure(j)
        for i in range(0, len(sim_x_label_temp)):
        # for m in id_plt:
            plt.plot(time_grid, sim_x_label_temp[i])
        plt.title(f"Simulated Curves with Noise - Group {j + 1}/{n_class}")
        pdf.savefig()
        # plt.show()
        plt.close()

with PdfPages("Datasets/Simulations/sim_reps_NN_sim6.pdf") as pdf:
    com_list = list(itertools.combinations(range(shape(sim_reps)[1]),2))
    for d in range(len(com_list)):
        rep1 = sim_reps[:,com_list[d][0]]
        rep2 = sim_reps[:,com_list[d][1]]
        # plt.figure(1)
        plt.scatter(rep1, rep2, c = sim_labels)
        # plt.legend(handles=unique(sim_labels), loc="upper right", title="Class")
        plt.title(f"Scatterplot of Reps {com_list[d][0]+1} & {com_list[d][1]+1}")
        plt.xlabel(f"Reps {com_list[d][0]+1}")
        plt.ylabel(f"Reps {com_list[d][1]+1}")
        pdf.savefig()
        # plt.show()
        plt.close()


########
mean = [[2,1,2,1,0], [-4,-4,-4,-4,-4], [3,3,3,3,3]]
cov = [[[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]],
       [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]],
       [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]]]
n_class=3
time_grid = np.linspace(0,1,21)
sim_x_dist, sim_x_noise_dist, sim_labels_dist, sim_reps_dist  = DataGenerateor_Dist_NN(n_sample_per_class=400, n_class=3, n_rep=5,
                                                                        mean=mean, cov=cov,
                                                                        n_basis = 10, basis_type = "BSpline",
                                                                        decoder_hidden = [10],
                                                                        time_grid = time_grid,
                                                                        activation_function = nn.Sigmoid(),
                                                                        noise=3)
os.chdir('C:/Users/Sidi/Desktop/FAE/FAE')
with PdfPages("Datasets/Simulations/sim_x_distNN_sim7.pdf") as pdf:
    # Plot simulated curves
    sim_x_dist_plt = sim_x_dist.detach().numpy()
    sim_x_noise_dist_plt = sim_x_noise_dist.detach().numpy()
    time_grid = time_grid
    color = ["y", "b", "g", "r"]

    plt.figure(1)
    for i in range(0, len(sim_x_dist_plt)):
        # for m in id_plt:
        plt.plot(time_grid, sim_x_dist_plt[i], color[sim_labels_dist[i]], alpha=0.1)
    plt.title("Simulated Curves")
    pdf.savefig()
    # plt.show()
    plt.close()
    plt.figure(2)
    for i in range(0, len(sim_x_noise_dist_plt)):
        # for m in id_plt:
        plt.plot(time_grid, sim_x_noise_dist_plt[i], color[sim_labels_dist[i]], alpha=0.1)
    plt.title("Simulated Curves with Noise")
    pdf.savefig()
    # plt.show()
    plt.close()

    label_dist_list = []
    # Plot simulated curves by class
    n_class = 3
    for j in range(n_class):
        label_dist_list.append([i for i in range(len(sim_labels_dist)) if sim_labels_dist[i] == (j + 1)])
        sim_x_label_temp = sim_x_noise_dist[label_dist_list[j]].detach().numpy()
        plt.figure(j)
        for i in range(0, len(sim_x_label_temp)):
            # for m in id_plt:
            plt.plot(time_grid, sim_x_label_temp[i])
        plt.title(f"Simulated Curves with Noise - Group {j + 1}/{n_class}")
        pdf.savefig()
        # plt.show()
        plt.close()


with PdfPages("Datasets/Simulations/sim_reps_distNN_sim7.pdf") as pdf:
    com_list_dist = list(itertools.combinations(range(shape(sim_reps_dist)[1]),2))
    for d in range(len(com_list_dist)):
        rep1 = sim_reps_dist[:,com_list[d][0]]
        rep2 = sim_reps_dist[:,com_list[d][1]]
        plt.scatter(rep1, rep2, c = sim_labels_dist)
        plt.title(f"Scatterplot of Reps {com_list_dist[d][0]+1} & {com_list_dist[d][1]+1}")
        plt.xlabel(f"Reps {com_list_dist[d][0]+1}")
        plt.ylabel(f"Reps {com_list_dist[d][1]+1}")
        pdf.savefig()
        # plt.show()
        plt.close()

sim4_x_dist, sim4_x_noise_dist, sim4_labels_dist, sim4_reps_dist= sim_x_dist, sim_x_noise_dist, sim_labels_dist, sim_reps_dist
sim6_x, sim6_x_noise, sim6_labels, sim6_reps = sim_x, sim_x_noise, sim_labels, sim_reps