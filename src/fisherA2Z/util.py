import numpy as np
import pandas as pd;
from collections import namedtuple
from pprint import pprint;
import sys
from copy import deepcopy;
import pickle;
import pyccl as ccl
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KernelDensity
from scipy.stats import iqr
from itertools import permutations
from collections import OrderedDict
from itertools import combinations
import random
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')
from fisherA2Z.fisher import Fisher, marginalize, plot_contours
from scipy.interpolate import CubicSpline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class Wrapper:
    def __init__(self, ccl_cls, cosmic_shear):
        if cosmic_shear:
            self.ccl_cls = ccl_cls[:15]
        else:
            self.ccl_cls = ccl_cls

def centroid_shift_lcdm(unbiased, biased):
    cl_unbiased = unbiased.ccl_cls
    cl_biased = biased.ccl_cls
    param_length = len(unbiased.param_order)
    diff_cl = (cl_biased - cl_unbiased).flatten()
    bias_vec = []
    for i, param in enumerate(unbiased.param_order[:param_length]):
        if i not in [3,4]:
            bias_vec.append(diff_cl @
                unbiased.invcov @ unbiased.derivs_sig[param].flatten())
    bias_vec = np.array(bias_vec)
    slice_ = list(range(0,3))+list(range(5,param_length))
    para_bias = np.linalg.inv(unbiased.fisher[:,slice_][slice_,:]).dot(bias_vec) 
    para_bias = {unbiased.param_order[i]: para_bias[i] for i in range(len(para_bias))}
    return para_bias


def centroid_shift_lcdm_cosmo(unbiased, biased):
    cl_unbiased = unbiased.ccl_cls
    cl_biased = biased.ccl_cls

    slice_ = list(range(0,3))+list(range(5,11))+list(range(26,len(unbiased.param_order)))
    
    diff_cl = (cl_biased - cl_unbiased).flatten()
    bias_vec = []
    for i, param in enumerate(np.array(unbiased.param_order)[slice_]):

        bias_vec.append(diff_cl @
                unbiased.invcov @ unbiased.derivs_sig[param].flatten())
    bias_vec = np.array(bias_vec)
    
    para_bias = np.linalg.inv(unbiased.fisher[:,slice_][slice_,:]).dot(bias_vec) 
    para_bias = {unbiased.param_order[i]: para_bias[i] for i in range(len(para_bias))}
    return para_bias

def centroid_shift(unbiased, biased):
    cl_unbiased = unbiased.ccl_cls
    cl_biased = biased.ccl_cls
    
    diff_cl = (cl_biased - cl_unbiased).flatten()
    bias_vec = []
    for i, param in enumerate(unbiased.param_order[:36]):
        bias_vec.append(unbiased.derivs_sig[param].flatten()@ unbiased.invcov @ diff_cl)
    bias_vec = np.array(bias_vec)
    para_bias = np.linalg.inv(unbiased.fisher).dot(bias_vec) 
    para_bias = {unbiased.param_order[i]: para_bias[i] for i in range(len(para_bias))}
    return para_bias

def centroid_shift_cosmo(unbiased, biased):
    cl_unbiased = unbiased.ccl_cls
    cl_biased = biased.ccl_cls

    slice_ = list(range(0,11))+list(range(26,36))
    
    diff_cl = (cl_biased - cl_unbiased).flatten()
    bias_vec = []
    for i, param in enumerate(np.array(unbiased.param_order)[slice_]):

        bias_vec.append(diff_cl @
                unbiased.invcov @ unbiased.derivs_sig[param].flatten())
    bias_vec = np.array(bias_vec)
    
    para_bias = np.linalg.inv(unbiased.fisher[:,slice_][slice_,:]).dot(bias_vec) 
    para_bias = {unbiased.param_order[i]: para_bias[i] for i in range(7)}
    return para_bias

def shift(results, param, obj, cosmic_shear = False, marginalize = False):
    if marginalize == True:
        return centroid_shift(obj, Wrapper(np.array(results), cosmic_shear = cosmic_shear))[param]
    else:
        return centroid_shift_cosmo(obj, Wrapper(np.array(results), cosmic_shear = cosmic_shear))[param]

def get_s8_shift(omega_m_shift, sigma_8_shift, obj):
    omega_m = omega_m_shift+obj.vals['omega_m']
    sigma_8 = sigma_8_shift+obj.vals['sigma_8']
    s8 = sigma_8 * np.sqrt(omega_m/0.3)
    s8_shift = s8 - obj.vals['sigma_8'] * np.sqrt(obj.vals['omega_m']/0.3)
    return s8_shift



