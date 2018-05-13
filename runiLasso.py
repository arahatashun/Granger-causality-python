#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for run irregular lasso parallelly
"""
import numpy as np
from irregular_lasso import irregular_lasso
from multiprocessing import Pool


def solve_loop(cell_array, N, alpha):
    cause = np.zeros((N, N, 3))
    argu_for_process = []
    for i in range(N):
        order = [i] + list(range(i)) + list(range(i + 1, N))
        new_cell = [cell_array[i] for i in order]
        argu_for_process.append((new_cell, i, N, alpha))
    p = Pool()
    output = p.map(wrap_worker, argu_for_process)
    for i in range(N):
        cause[i, :, :] = output[i]
    return cause


def process_worker(new_cell, i, N, alpha):
    cause_tmp = irregular_lasso(new_cell, alpha)
    index = list(range(1, i + 1)) + [0] + list(range(i + 1, N))
    return cause_tmp[index, :]


def wrap_worker(arg):
    return process_worker(*arg)
