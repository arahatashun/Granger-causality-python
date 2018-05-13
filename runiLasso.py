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
    """solve irrgegular lasso in parallel

    :param cell_array:one cell for each time series. Each cell is a 2xT matrix.
    First row contains the values and the second row contains SORTED time stamps.
    The first time series is the target time series which is predicted.
    """
    N = len(cell_array)
    cause = np.zeros((N, N, 3))
    argu_for_process = []
    for i in range(N):
        order = [i] + list(range(i)) + list(range(i + 1, N))
        new_cell = [cell_array[i] for i in order]
        argu_for_process.append((new_cell, i, N, alpha))
    pool = Pool()
    output = pool.map(wrap_worker, argu_for_process)
    for i in range(N):
        cause[i, :, :] = output[i][0]
    return cause


def process_worker(new_cell, i, N, alpha):
    cause_tmp, aic, bic = irregular_lasso(new_cell, alpha)
    index = list(range(1, i + 1)) + [0] + list(range(i + 1, N))
    return cause_tmp[index, :],aic,bic


def wrap_worker(arg):
    """wrapper function"""
    return process_worker(*arg)
