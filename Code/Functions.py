"""
This script contains the self-defined functions used for running the existing and proposed methods implemented
in the manuscript "Autoencoders for Discrete Functional Data Representation Learning and Smoothing".

@author: Sidi Wu
"""
import random
from random import seed
import numpy as np
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn

# Function "train_test_split": split the training/test set
def train_test_split(data, label, split_rate, seed_no, missing_ind=None, omega=None):
    """
    :param data: observed curves - n_obs X n_tpts
    :param label: observed label - n_obs X 1
    :param split_rate: 0-1, percentage of training data
    :param seed_no: the seed for splitting
    :param missing_ind: missing data indicator - n_obs X n_tpts
    :param omega: weight matrix for numerical integration - n_obs X n_tpts
    """
    classes = len(np.unique(label))
    train_no = []
    random.seed(seed_no)
    train_seeds = random.sample(range(1000), classes)
    start = 0
    for i in range(0, classes):
        step = len(label[label==np.unique(label)[i]])
        seed(train_seeds[i])
        temp_no = random.sample(range(int(start), int(start+step)), round(step * split_rate))
        train_no.extend(temp_no)
        start += step

    if (missing_ind is not None) and (omega is not None):
        TrainData = data[train_no]
        TrainLabel = label[train_no]
        TrainNan = missing_ind[train_no]
        TrainOmega = omega[train_no]
        if split_rate == 1:
            TestData = data
            TestLabel = label
            TestNan = missing_ind
            TestOmega = omega
        else:
            TestData = data[[i for i in range(len(data)) if i not in train_no]]
            TestLabel = label[[i for i in range(len(label)) if i not in train_no]]
            TestNan = missing_ind[[i for i in range(len(missing_ind)) if i not in train_no]]
            TestOmega = omega[[i for i in range(len(omega)) if i not in train_no]]

        return TrainData, TestData, TrainLabel, TestLabel, TrainNan, TestNan, TrainOmega, TestOmega, train_no
    elif (missing_ind is not None) and (omega is None):
        TrainData = data[train_no]
        TrainLabel = label[train_no]
        TrainNan = missing_ind[train_no]
        if split_rate == 1:
            TestData = data
            TestLabel = label
            TestNan = missing_ind
        else:
            TestData = data[[i for i in range(len(data)) if i not in train_no]]
            TestLabel = label[[i for i in range(len(label)) if i not in train_no]]
            TestNan = missing_ind[[i for i in range(len(missing_ind)) if i not in train_no]]

        return TrainData, TestData, TrainLabel, TestLabel, TrainNan, TestNan, train_no
    elif (missing_ind is None) and (omega is not None):
        TrainData = data[train_no]
        TrainLabel = label[train_no]
        TrainOmega = omega[train_no]
        if split_rate == 1:
            TestData = data
            TestLabel = label
            TestOmega = omega
        else:
            TestData = data[[i for i in range(len(data)) if i not in train_no]]
            TestLabel = label[[i for i in range(len(label)) if i not in train_no]]
            TestOmega = omega[[i for i in range(len(omega)) if i not in train_no]]

        return TrainData, TestData, TrainLabel, TestLabel, TrainOmega, TestOmega, train_no
    else:
        TrainData = data[train_no]
        TrainLabel = label[train_no]
        if split_rate == 1:
            TestData = data
            TestLabel= label
        else:
            TestData = data[[i for i in range(len(data)) if i not in train_no]]
            TestLabel = label[[i for i in range(len(label)) if i not in train_no]]

        return TrainData, TestData, TrainLabel, TestLabel, train_no

# Function "eval_MSE": calculate the MSE lose
def eval_MSE(obs_X, pred_X):
    """
    :param obs_X: observed curves - n_obs X n_tpts
    :param pred_X: predicted curves - n_obs X n_tpts
    :return: MSE loss
    """
    if not torch.is_tensor(obs_X):
        obs_X = torch.tensor(obs_X)
    if not torch.is_tensor(pred_X):
        pred_X = torch.tensor(pred_X)
    loss_fct = nn.MSELoss()
    loss = loss_fct(obs_X, pred_X)
    return loss

# Function "eval_mse_sdse": calculate the MSE & SD of the squared error
def eval_mse_sdse(obs_X, pred_X):
    """
    :param obs_X: observed curves - n_obs X n_tpts
    :param pred_X: predicted curves - n_obs X n_tpts
    :return: mse and sd of the squared error
    """
    if not torch.is_tensor(obs_X):
        obs_X = torch.tensor(obs_X)
    if not torch.is_tensor(pred_X):
        pred_X = torch.tensor(pred_X)
    sd, mean = torch.std_mean(torch.mean(torch.square(obs_X - pred_X), dim=1))
    return mean, sd

# Function "trapezoidal_weights": return the weight matrix according to the inputted time stamp using the trapezoidal rule.
def trapezoidal_weights(tpts):
    """
    :param tpts: observed time stamp - n_obs X n_tpts
    :return: weight for numerical integration
    """
    W = np.nan * np.zeros(shape=shape(tpts))
    for i in range(len(tpts)):
        obs_list = [j for j in range(len(tpts[i])) if str(tpts[i][j]) != 'nan']
        temp = [x for x in tpts[i] if str(x) != 'nan']
        temp_diff = [temp[j] - temp[j - 1] for j in range(len(temp)) if j > 0]
        for k in obs_list:
            k_ind = obs_list.index(k)
            if k_ind == 0:
                W[i][k] = temp_diff[0] * 1 / 2
            elif k_ind == len(obs_list) - 1:
                W[i][k] = temp_diff[-1] * 1 / 2
            else:
                W[i][k] = (temp_diff[k_ind - 1] + temp_diff[k_ind]) * 1 / 2
    W = torch.tensor(W).float()
    return W

# Function "random_missing": produce functional observations with missing values at random or customized time points.
def random_missing(data, time, num = 0, fixed_num = True, missing_col = None, tensor_type = False):
    """
    :param data: observed curves - n_obs X n_tpts
    :param time: observed timestamp - n_obs X n_tpts
    :param num: num of missing values per observation
    :param fixed_num: True for a random missing # per obs, False for a fixed missing # per obs
    :param missing_col: customized missing columns (time points)
    :param tensor_type: True for returning tensor type data, False for returning array type data
    :return: irregularly observed data (data_irr), the corresponding time stamp w/ missing values (time_irr), and an indicator matrix for missing values (nan_ind)
    """
    if torch.is_tensor(data):
        data = data.numpy()
    if torch.is_tensor(time):
        time = time.numpy()
    nan_ind = np.ones(shape = shape(data))
    data_irr = data.copy()
    time_irr = time.copy()
    if fixed_num:
        if missing_col is not None:
            missing_list = missing_col.copy()
            for i in range(shape(data)[0]):
                data_irr[i][missing_list] = np.nan
                time_irr[i][missing_list] = np.nan
                nan_ind[i][missing_list] = [0]
        if missing_col is None:
            for i in range(shape(data)[0]):
                missing_list = random.sample(range(1, shape(data)[1]-1), num)
                data_irr[i][missing_list] = np.nan
                time_irr[i][missing_list] = np.nan
                nan_ind[i][missing_list] = [0]
    if not fixed_num:
        num_temp = random.sample(range(1, num), 1)[0]
        for i in range(shape(data)[0]):
            missing_list = random.sample(range(1, shape(data)[1]-1), num_temp)
            data_irr[i][missing_list] = np.nan
            time_irr[i][missing_list] = np.nan
            nan_ind[i][missing_list] = [0]
    if tensor_type:
        data_irr, time_irr, nan_ind = torch.tensor(data_irr).float(), torch.tensor(time_irr).float(), torch.tensor(nan_ind).float()
    return data_irr, time_irr, nan_ind

# # Function "compute_missing_col":
# def compute_missing_col(x, missing_col):
#     output = np.zeros(shape=(len(x), len(missing_col)))
#     for i in range(len(missing_col)):
#         output[:,i] = torch.mean(x_missing[:,[missing_col[i]-1, missing_col[i]+1]], axis=1)
#     return output

