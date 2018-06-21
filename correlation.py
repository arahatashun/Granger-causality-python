#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for collilation slotting
"""
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import copy


def calc_each_cor(cell_a, cell_b, sigma, lag, dt):
    """
    Calculate correlation between two variables using correlation slotting.
    Note that the observations have to be standardized to zero mean and unit variance before the analysis.

    :param cell_a:one cell for each time series. Each cell is a 2xT matrix.
    First row contains the values and the second row contains SORTED time stamps.
    :param cell_b:
    :param sigma: Kernel Parameter
    :param lag: length of lag
    :param dt: max (dta dtb)
    :return correlation
    """
    cell_a[0, :] = (cell_a[0, :] - cell_a[0, :].mean(axis=0)) / cell_a[0, :].std(axis=0)
    cell_b[0, :] = (cell_b[0, :] - cell_b[0, :].mean(axis=0)) / cell_b[0, :].std(axis=0)
    # number of index of time of explained variable
    N1 = cell_a[0, :].shape[0]
    denominator = 0
    numerator = 0
    # for loop for stored time stamp
    for i in range(N1):
        x = cell_a[0, i]
        ti = cell_a[1, i]
        # reduce kernel length in order to reduce complexity of calculation
        # kernel is used as window function
        # time_match is a cell_list[time_match][1, :] nearest to tij
        time_match = np.searchsorted(cell_b[1, :], ti)
        kernel_length = 5  # half of kernel length

        start = time_match - kernel_length if time_match - kernel_length > 0 else 0
        end = time_match + kernel_length if time_match + kernel_length < len(cell_b[1, :]) - 1 else len(cell_b[1, :])
        tij = np.broadcast_to(ti, (len(cell_b[1, start:end]), ti.size))
        t_select = cell_b[1, start:end].T
        t_select = t_select.reshape(len(t_select), 1)
        y_select = cell_b[0, start:end].T
        kernel_b = np.abs(np.abs(tij - t_select) - lag * dt)
        assert kernel_b.shape == t_select.shape, print('tselect', t_select.shape, "kernel_b", kernel_b.shape)
        exponent = -(np.multiply(kernel_b, kernel_b) / (2*sigma**2))
        assert np.isfinite(exponent).all() == 1, str(exponent)
        Kernel = np.exp(exponent)/(np.sqrt(2*np.pi*sigma))
        assert np.isfinite(Kernel).all() == 1, str(Kernel)
        denominator_tmp = np.sum(Kernel)
        numerator_tmp= np.sum(x * y_select * Kernel)
        assert np.all(x * y_select < 1), 'x:'+str(x)+' y:'+str(y_select)+' x*y_select:'+str(x*y_select)
        assert abs(numerator_tmp)/abs(denominator_tmp) <= 1, ' numerator:'+str(numerator)+' denominator:'+str(denominator)+' x*y:'+str(x * y_select)
        denominator += denominator_tmp
        numerator += numerator_tmp

    correlation = numerator / denominator
    #print("correlation", correlation)
    return correlation


def calc_cor(cell_array, lag_len=0):
    num_features = len(cell_array)
    argument_for_process = []
    for i in range(num_features):
        for j in np.arange(i, num_features):
            dtx = (cell_array[i][1][-1] - cell_array[i][1][0]) / len(cell_array[i][0])
            dty = (cell_array[j][1][-1] - cell_array[j][1][0]) / len(cell_array[j][0])
            assert dtx > 0, "index:" + str(i)
            assert dty > 0, "index:" + str(j)
            dt = max(dtx, dty)
            sigma = dt / 4  # Comparison of correlation analysis techniques for irregularly sampled time series
            new_cell = [cell_array[i], cell_array[j]]
            argument_for_process.append((new_cell, i, j, sigma, lag_len, dt))
    pool = Pool()
    output_list = []
    pbar = tqdm(total=num_features)
    for _, output in enumerate(pool.imap_unordered(wrap_worker, argument_for_process)):
        pbar.update()
        output_list.append(output)
    pbar.close()
    correlation_matrix = np.zeros((num_features, num_features))
    for i in range(len(output_list)):
        correlation = output_list[i][0]
        x = output_list[i][1]
        y = output_list[i][2]
        correlation_matrix[x, y] = correlation
        correlation_matrix[y, x] = correlation
    correlation_matrix = normalize_cor_mat(correlation_matrix)
    print(correlation_matrix)
    return correlation_matrix


def process_worker(new_cell, i, j, sigma, lag_len, dt):
    correlation = calc_each_cor(new_cell[0], new_cell[1], sigma, lag_len, dt)
    return correlation, i, j


def wrap_worker(arg):
    """wrapper function"""
    return process_worker(*arg)


def normalize_cor_mat(cor_mat):
    num_features = cor_mat.shape[0]
    sum = 0
    for i in range(num_features):
        sum += cor_mat[i,i]
    norm = sum/num_features
    return cor_mat/norm

