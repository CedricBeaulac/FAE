"""
This script contains the code for importing and pre-processing the El Nino data set in the manuscript "Autoencoders for Discrete Functional Data Representation Learning and Smoothing".

@author: Sidi Wu
"""
import pickle
import os
import torch
import numpy as np
from numpy import *

os.chdir('~/Code')
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import Functions
from Functions import *

os.chdir("~"）

#####################################
### Scenario 1.1
#####################################
# Load simulation data
with open('Datasets/Simulation/sim_s2_data.txt', 'rb') as f:
    dict = pickle.load(f)
s2_x, s2_x_noise, s2_labels, s2_reps = dict[0], dict[1], dict[2], dict[3]
sim_x_dist, sim_x_noise_dist, sim_labels_dist, sim_reps_dist = s2_x.clone(), s2_x_noise.clone(), s2_labels.copy(), s2_reps.copy()
time_grid = np.linspace(0,1,21)

# Import data set
x_raw = sim_x_noise_dist.detach().numpy().copy()
tpts_raw = time_grid.reshape(-1, 1)
label =sim_labels_dist.copy().astype(int64).copy()

# Pre-process data set
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


#####################################
### Scenario 1.2 & Scenario 2.1
#####################################
# Load simulation data
with open('Datasets/Simulation/sim_s1_data.txt', 'rb') as f:
    dict = pickle.load(f)
s1_x, s1_x_noise, s1_labels, s1_reps = dict[0], dict[1], dict[2], dict[3]
sim_x_dist, sim_x_noise_dist, sim_labels_dist, sim_reps_dist = s1_x.clone(), s1_x_noise.clone(), s1_labels.copy(), s1_reps.copy()
time_grid = np.linspace(0,1,51)

# Import data set
x_raw = sim_x_noise_dist.detach().numpy().copy()
tpts_raw = time_grid.reshape(-1, 1)
label =sim_labels_dist.copy().astype(int64).copy()

# Pre-process data set
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


#####################################
### Scenario 2.2
#####################################
# Load simulation data
with open('Datasets/Simulation/sim_s1_data.txt', 'rb') as f:
    dict = pickle.load(f)
s1_x, s1_x_noise, s1_labels, s1_reps = dict[0], dict[1], dict[2], dict[3]
sim_x_dist, sim_x_noise_dist, sim_labels_dist, sim_reps_dist = s1_x.clone(), s1_x_noise.clone(), s1_labels.copy(), s1_reps.copy()
time_grid = np.linspace(0,1,51)

# Import data set
x_raw = sim_x_noise_dist.detach().numpy().copy()
tpts_raw = time_grid
label =sim_labels_dist.copy().astype(int64).copy()

# Pre-process data set
# Create irregular data set
num = 25
true_tpts = np.array(tpts_raw)
true_tpts_rescale = (true_tpts - np.min(true_tpts)) / np.ptp(true_tpts)
tpts_rescale = np.tile(np.array(true_tpts_rescale),(shape(x_raw)[0],1))
x_irr, tpts_irr, nan_ind = random_missing(x_raw, tpts_rescale, num = num)
nan_ind = torch.tensor(nan_ind).float()

# Prepare numpy/tensor data
x_np = np.array(x_irr).astype(float)
x = torch.tensor(x_np).float()
x = x - torch.nanmean(x,0)
x = torch.nan_to_num(x)

# Prepare tensor time
tpts = torch.tensor(tpts_irr)
n_tpts = len(tpts[0])

# Prepare Omega (weight) matrix
Omega = trapezoidal_weights(tpts_irr)
Omega = torch.nan_to_num(Omega)
