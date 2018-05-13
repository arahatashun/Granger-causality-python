#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for run irregular lasso parallel processing
"""
import numpy as np
from irregular_lasso import irregular_lasso
from multiprocessing import Pool


def solve_loop(cell_array, alpha):
    """solve irregular lasso in parallel

    :param cell_array:one cell for each time series. Each cell is a 2xT matrix.
        First row contains the values and the second row contains SORTED time stamps.
        The first time series is the target time series which is predicted.
    :param alpha: The regularization parameter in Lasso
    """
    total_features = len(cell_array)
    cause = np.zeros((total_features, total_features, 3))
    argument_for_process = []
    for i in range(total_features):
        order = [i] + list(range(i)) + list(range(i + 1, total_features))
        new_cell = [cell_array[i] for i in order]
        argument_for_process.append((new_cell, i, total_features, alpha))
    pool = Pool()
    output = pool.map(wrap_worker, argument_for_process)
    for i in range(total_features):
        cause[i, :, :] = output[i][0]
    return cause


def process_worker(new_cell, i, n, alpha):
    cause_tmp, aic, bic = irregular_lasso(new_cell, alpha)
    index = list(range(1, i + 1)) + [0] + list(range(i + 1, n))
    return cause_tmp[index, :], aic, bic


def wrap_worker(arg):
    """wrapper function"""
    return process_worker(*arg)
