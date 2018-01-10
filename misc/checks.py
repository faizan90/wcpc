'''
Created on Dec 30, 2017

@author: Faizan
'''
import numpy as np


def check_nans_finite(in_arr):
    return np.all(np.isfinite(in_arr))


def check_nats(in_arr):
    return np.any(np.isnat(in_arr))
