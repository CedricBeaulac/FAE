"""
This script contains the code for importing and pre-processing the simulation data sets
in the manuscript "Functional Autoencoder for Smoothing and Representation Learning".

@author: Sidi Wu
"""

import pandas as pd
import numpy as np
from numpy import *
import os
import torch

os.chdir("~"）

#####################################
### Real application: ElNino data set
#####################################
# Import dataset
x_raw = pd.read_csv('Datasets/RealApplication/ElNino_ERSST.csv')
tpts_raw = pd.read_csv('Datasets/RealApplication/ElNino_ERSST_tpts.csv')
label_table = pd.read_csv('Datasets/RealApplication/ElNino_ERSST_label.csv')
label = label_table.x.to_numpy()
time_grid = np.array(tpts_raw).flatten()

# Pre-process Data sets
# Prepare numpy/tensor data
x_np = np.array(x_raw).astype(float)
x = torch.tensor(x_np).float()
x_mean = torch.mean(x,0)
x = x - torch.mean(x,0)

# Rescale timestamp to [0,1]
tpts_np = np.array(tpts_raw)
tpts_rescale = (tpts_np - min(tpts_np)) / np.ptp(tpts_np)
tpts = torch.tensor(np.array(tpts_rescale))
n_tpts = len(tpts)
