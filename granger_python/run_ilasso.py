#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for run irregular lasso parallel processing
"""
import numpy as np
from irregular_lasso import irregular_lasso
from multiprocessing import Pool
from tqdm import tqdm
from i_lasso import ilasso

def solve_loop(cell_array, alpha, sigma, lag_len, dt):
    """solve irregular lasso in parallel

    :param cell_array:one cell for each time series. Each cell is a 2xT matrix.
        First row contains the values and the second row contains SORTED time stamps.
        The first time series is the target time series which is predicted.
    :param alpha: The regularization parameter in Lasso
    :param sigma:kernel parameter. Here Gaussian kernel Bandwidth
    :param lag_len:Length of studied lag
    :param dt:the  average  length  of  the  sampling  intervals for the target time series
    """
    total_features = len(cell_array)
    cause = np.zeros((total_features, total_features, lag_len))
    argument_for_process = []
    for i in range(total_features):
        order = [i] + list(range(i)) + list(range(i + 1, total_features))
        new_cell = [cell_array[i] for i in order]
        argument_for_process.append(
            (new_cell, i, total_features, alpha, sigma, lag_len, 1))
    pool = Pool()
    outputs = []
    pbar = tqdm(total=total_features)
    for _, output in enumerate(pool.imap_unordered(wrap_worker, argument_for_process)):
        pbar.update()
        outputs.append(output)
    pbar.close()
    for i in range(total_features):
        j = outputs[i][3]
        cause[j, :, :] = outputs[i][0]
    return cause


def process_worker(new_cell, i, n, alpha, sigma, lag_len, dt):
    cause_tmp, aic, bic = ilasso(new_cell, alpha, sigma, lag_len, dt)
    index = list(range(1, i + 1)) + [0] + list(range(i + 1, n))
    return cause_tmp[index, :], aic, bic, i


def wrap_worker(arg):
    """wrapper function"""
    return process_worker(*arg)
