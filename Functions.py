"""
This script contains self-defined functions ...
"""
import random
import numpy as np
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn

# Function "train_test_split": Split training/test set
def train_test_split(data, label, split_rate, seed_no):
    # TrainData, TestData = torch.utils.data.random_split(x, [round(len(x) * split.rate), (len(x)-round(len(x) * split.rate))])
    # train_no = random.sample(range(0, len(x)), round(len(x) * split.rate))
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

    TrainData = data[train_no]
    TrainLabel = label[train_no]
    if split.rate == 1:
        TestData = data
        TestLabel= label
    else:
        TestData = data[[i for i in range(len(data)) if i not in train_no]]
        TestLabel = label[[i for i in range(len(label)) if i not in train_no]]

    return TrainData, TestData, TrainLabel, TestLabel, train_no

# Function "eval_MSE": calculate the MSE
def eval_MSE(obs_X, pred_X):
    if not torch.is_tensor(obs_X):
        obs_X = torch.tensor(obs_X)
    if not torch.is_tensor(pred_X):
        pred_X = torch.tensor(pred_X)
    loss_fct = nn.MSELoss()
    loss = loss_fct(obs_X, pred_X)
    return loss

# Function "eval_mse_sdse": calculate the MSE & SDSE
def eval_mse_sdse(obs_X, pred_X):
    if not torch.is_tensor(obs_X):
        obs_X = torch.tensor(obs_X)
    if not torch.is_tensor(pred_X):
        pred_X = torch.tensor(pred_X)
    sd, mean = torch.std_mean(torch.mean(torch.square(obs_X - pred_X), dim=1))
    return mean, sd

# Function "most_frequent":
def most_frequent(label_list, optimal_n_cluster):
    counter = 0
    for i in range(optimal_n_cluster):
        no = np.count_nonzero(label_list == i)
        if no > counter:
            counter = no
            num = i
    return num

# Function "acc":
def acc(label_list, optimal_n_cluster):
    most_cluster = most_frequent(label_list, optimal_n_cluster)
    most_no = np.count_nonzero(label_list == most_cluster)
    prob = most_no/len(label_list)
    return prob

# Function "indices":
def indices(list, filter=lambda x: bool(x)):
    return [i for i,x in enumerate(list) if filter(x)]
