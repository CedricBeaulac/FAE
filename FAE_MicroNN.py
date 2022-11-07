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
from skfda.representation import basis as basis
from skfda.exploratory.visualization import FPCAPlot
# from skfda.exploratory.visualization import FPCAPlot
# from skfda.preprocessing.dim_reduction import FPCA
# from skfda.representation.basis import BSpline, Fourier, Monomial
import scipy
from scipy.interpolate import BSpline
import ignite
#import os
import os
import sklearn
from sklearn.decomposition import PCA
import random

#################################################
# FAE
# Encoder: a added layer for fd representation
# Decoder:
#################################################
