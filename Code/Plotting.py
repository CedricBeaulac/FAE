"""
This script contains the code for creating the plots displayed in the manuscript "Autoencoders for Discrete Functional Data Representation Learning and Smoothing".

* Note: this script should be implemented after running the scripts for FAE, AE and FPCA

@author: Sidi Wu
"""

# Import modules
import pandas as pd
import numpy as np
from numpy import *
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mtick
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
import itertools


# The plots showing raw curves, along with FAE-recoverd, FPCA-recoverd, AE-recovered curves for subjects in a random test set (i)

i=8 # randomly select a test set

TestData = x[[j for j in range(len(x)) if j not in AE_train_no_niter[i]]]
input_plt = TestData.detach().numpy()
FPCA_pred_plt = FPCA_pred_test_plot_niter[i]
AE_pred_plt = AE_pred_test_niter[i].detach().numpy()
tpts_FPCA_plot =  torch.tensor(np.arange(0, 1+1/100, 1/100))
tpts_FAE_plot = torch.tensor(np.arange(0, 1+1/200, 1/200))
basis_fc_FAE_plot = torch.from_numpy(bss_revert.evaluate(tpts_FAE_plot, derivative=0)[:, :, 0]).float()
FAE_pred_plt = torch.matmul(FAE_coef_test_niter[i], basis_fc_FAE_plot).detach().numpy()

activation_FAE = "Sigmoid"
activation_AE = "Sigmoid"
true_tpts = time_grid
true_tpts_FAE_plot = diff([min(true_tpts), max(true_tpts)]).item()*tpts_FAE_plot
true_tpts_FPCA_plot = diff([min(true_tpts), max(true_tpts)]).item()*tpts_FPCA_plot

plt.figure(1, figsize=(20, 10))
plt.subplot(221)
for m in range(0, len(input_plt)):
    plt.plot(true_tpts, input_plt[m])
plt.title('"Observed"') # or "Simulated"
# plt.axvspan(0.3, 0.6, alpha=0.5, color='wheat') # for creating shaded area
plt.xticks([])
plt.subplot(222)
for m in range(0, len(FPCA_pred_plt)):
    # for m in id_plt:
    plt.plot(true_tpts_FPCA_plot, FPCA_pred_plt[m])
plt.title("FPCA")
# plt.axvspan(0.3, 0.6, alpha=0.5, color='wheat')  # for creating shaded area
plt.xticks([])
plt.yticks([])
plt.subplot(224)
for m in range(0, len(FAE_pred_plt)):
    # for m in id_plt:
    plt.plot(true_tpts_FAE_plot, FAE_pred_plt[m])
plt.yticks([])
# plt.axvspan(0.3, 0.6, alpha=0.5, color='wheat')  # for creating shaded area
plt.title(f"FAE({activation_FAE})")
plt.subplot(223)
for m in range(0, len(AE_pred_plt)):
    plt.plot(true_tpts, AE_pred_plt[m])
plt.title(f"AE({activation_AE})")
plt.tight_layout(pad=3)
plt.show()
# plt.close()

# The plots showing how prediction error & classification accuracy change with the number of epochs
plt.figure(2, figsize=(16, 6))
plt.subplot(121)
plt.plot([100 * (i + 1) for i in range(len(AE_reg_test_acc_epoch))],
         [mean(AE_reg_test_acc_epoch[i]) for i in range(len(AE_reg_test_acc_epoch))],
         label=f"AE")
plt.plot([100 * (i + 1) for i in range(len(FAE_reg_test_acc_epoch))],
         [mean(FAE_reg_test_acc_epoch[i]) for i in range(len(FAE_reg_test_acc_epoch))],
         label=f"FAE")
# plt.title(r"MSE$_{p}$ vs. Epochs, " + f"{n_rep} representations")
plt.xlabel('Epochs', fontsize = 16)
plt.ylabel("Prediction Error", fontsize = 16)
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.legend(fontsize=14, loc=1)
plt.tight_layout()
plt.subplot(122)
plt.plot([100 * (i + 1) for i in range(len(classification_AE_reg_test_epoch))],
         [mean(classification_AE_reg_test_epoch[i]) for i in range(len(classification_AE_reg_test_epoch))],
         label=f"AE")
plt.plot([100 * (i + 1) for i in range(len(classification_FAE_reg_test_epoch))],
         [mean(classification_FAE_reg_test_epoch[i]) for i in range(len(classification_FAE_reg_test_epoch))],
         label=f"FAE")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.xlabel('Epochs', fontsize  = 16)
plt.ylabel("Classification Accuracy", fontsize = 16)
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.legend(fontsize = 14, loc = 4)
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.4)
plt.show()
# plt.close()