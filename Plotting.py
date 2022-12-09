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
import matplotlib.backends.backend_pdf
plt.ioff()
pdf = matplotlib.backends.backend_pdf.PdfPages("Datasets/ElNino/ElNino_2layers(20-10-5)_nonlinear(Softplus)_+AE2.pdf")
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
