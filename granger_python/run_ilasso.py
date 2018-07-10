#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for run irregular lasso parallel processing
"""
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from igrouplasso import igrouplasso
from ilasso import ilasso
from tqdm import tqdm
import pyprind

def solve_loop(cell_array, alpha, lag_len, cv=False, group=False):
    """solve irregular lasso in parallel

    :param cell_array:one cell for each time series. Each cell is a 2xT matrix.
        First row contains the values and the second row contains SORTED time stamps.
        The first time series is the target time series which is predicted.
    :param alpha: The regularization parameter in Lasso
    :param lag_len:Length of studied lag
    :param cv:whether or not do cross validation
    :param group:group lasso
    """
    total_features = len(cell_array)
    cause = np.zeros((total_features, total_features, lag_len))
    argument_for_process = []
    for i in range(total_features):
        num_of_element = len(cell_array[i][0])
        avg_dt = (cell_array[i][1][-1] - cell_array[i][1][0]) / num_of_element
        assert avg_dt > 0, "avg_dt:" + str(avg_dt) + ", num_elem:" + str(num_of_element) + ", index:" + str(i)
        sigma = avg_dt / 4  # Comparison of correlation analysis techniques for irregularly sampled time series
        order = [i] + list(range(i)) + list(range(i + 1, total_features))
        new_cell = [cell_array[i] for i in order]
        argument_for_process.append((new_cell, i, total_features, alpha, sigma, lag_len, avg_dt, cv, group))
    pool = Pool()
    outputs = []
    bar = pyprind.ProgBar(total_features, bar_char='█')
    for _, output in enumerate(pool.imap_unordered(wrap_worker, argument_for_process)):
        bar.update()
        outputs.append(output)
    print(bar)
    if cv is False:
        aic = np.zeros(total_features)
        bic = np.zeros(total_features)
        for i in range(total_features):
            j = outputs[i][3]
            cause[j, :, :] = outputs[i][0]
            aic[j] = outputs[i][1]
            bic[j] = outputs[i][2]
        aic = np.sum(aic, axis=None) / total_features
        bic = np.sum(bic, axis=None) / total_features
        print("alpha:", alpha, ", lag:", lag_len, ",AIC:", aic, ",BIC", bic)
        return cause, aic, bic
    else:
        aic = np.zeros(total_features)
        bic = np.zeros(total_features)
        error = 0
        for i in range(total_features):
            j = outputs[i][3]
            cause[j, :, :] = outputs[i][0]
            aic[j] = outputs[i][1]
            bic[j] = outputs[i][2]
            error += outputs[i][4]
        aic = np.sum(aic, axis=None) / total_features
        bic = np.sum(bic, axis=None) / total_features
        print("alpha:", alpha, ", lag:", lag_len, ",AIC:", aic, ",BIC", bic, ",Error", error)
        return cause, aic, bic, error


def process_worker(new_cell, i, n, alpha, sigma, lag_len, dt, cv, group):
    if cv is False:
        if group is False:
            cause_tmp, aic, bic = ilasso(new_cell, alpha, sigma, lag_len, dt, cv)
            index = list(range(1, i + 1)) + [0] + list(range(i + 1, n))
            return cause_tmp[index, :], aic, bic, i
        if group is True:
            cause_tmp, aic, bic = igrouplasso(new_cell, alpha, sigma, lag_len, dt, cv)
            index = list(range(1, i + 1)) + [0] + list(range(i + 1, n))
            return cause_tmp[index, :], aic, bic, i
    else:
        if group is False:
            cause_tmp, aic, bic, error = ilasso(new_cell, alpha, sigma, lag_len, dt, cv)
            index = list(range(1, i + 1)) + [0] + list(range(i + 1, n))
            return cause_tmp[index, :], aic, bic, i, error
        if group is True:
            cause_tmp, aic, bic, error = igrouplasso(new_cell, alpha, sigma, lag_len, dt, cv)
            index = list(range(1, i + 1)) + [0] + list(range(i + 1, n))
            return cause_tmp[index, :], aic, bic, i, error


def wrap_worker(arg):
    """wrapper function"""
    return process_worker(*arg)


def test_solve(cell_array, alpha, lag_len):
    total_features = len(cell_array)
    i = 0
    num_of_element = len(cell_array[i][0])
    avg_dt = (cell_array[i][1][-1] - cell_array[i][1][0]) / num_of_element
    assert avg_dt > 0
    sigma = avg_dt / 4  # Comparison of correlation analysis techniques for irregularly sampled time series
    order = [i] + list(range(i)) + list(range(i + 1, total_features))
    new_cell = [cell_array[i] for i in order]
    igrouplasso(new_cell, alpha, sigma, lag_len, avg_dt)
    return None


def search_optimum_lambda(cell_array, lambda_min, lambda_max, lag_len, grid=20):
    """ search optimum lambda

    :param cell_array:
    :param lambda_max:
    :param lag_len:
    :return:
    """
    cv_error = []
    lambda_list = []
    lambda_exponent = np.linspace(np.log10(lambda_min), np.log10(lambda_max), grid)
    for i, j in enumerate(lambda_exponent):
        print(i)
        _, _, _, error = solve_loop(cell_array, 10 ** j, lag_len, cv=True, group=False)
        lambda_list.append(j)
        cv_error.append(error)

    optimum = lambda_list[np.argmin(cv_error)]

    optimum = 10 ** optimum
    print("Optimum lambda", optimum)
    plt.scatter(lambda_list, cv_error)
    plt.show()
    return optimum
