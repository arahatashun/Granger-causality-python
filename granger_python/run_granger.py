#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for run irregular lasso parallel processing
"""
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import pyprind
from igrouplasso import igrouplasso
from ilasso import ilasso
from glg import GLG


def solve_parallel(cell_array, alpha, lag_len, group=False):
    """solve Granger lasso in parallel

    :param cell_array:one cell for each time series. Each cell is a 2xT matrix.
        First row contains the values and the second row contains SORTED time stamps.
        The first time series is the target time series which is predicted.
    :param alpha: The regularization parameter in Lasso single value
    :param lag_len:Length of studied lag
    :param group:group lasso
    """
    total_features = len(cell_array)
    cause = np.zeros((total_features, total_features, lag_len))
    # Make argument for process
    argument_for_process = []
    for i in range(total_features):
        num_of_element = len(cell_array[i][0])
        avg_dt = (cell_array[i][1][-1] - cell_array[i][1][0]) / num_of_element
        assert avg_dt > 0, "avg_dt:" + str(avg_dt) + ", num_elem:" + str(num_of_element) + ", index:" + str(i)
        sigma = avg_dt / 4  # Comparison of correlation analysis techniques for irregularly sampled time series
        order = [i] + list(range(i)) + list(range(i + 1, total_features))
        new_cell = [cell_array[i] for i in order]
        argument_for_process.append((new_cell, i, total_features, alpha, sigma, lag_len, avg_dt, group))
    # start multiprocessing
    pool = Pool()
    outputs = []
    bar = pyprind.ProgBar(total_features, width=60, bar_char='#', title='PROGRESS')
    for _, output in enumerate(pool.imap_unordered(wrap_worker, argument_for_process)):
        bar.update()
        outputs.append(output)
    print(bar)
    for i in range(total_features):
        j = outputs[i]['index']
        cause[j, :, :] = outputs[i]['cause']
    print(cause.shape)
    return cause


def process_worker(new_cell, i, n, alpha, sigma, lag_len, dt, group):
    if group is False:
        glg = GLG(new_cell, sigma, lag_len, dt)
        cause_tmp = glg.calculate(alpha)
        index = list(range(1, i + 1)) + [0] + list(range(i + 1, n))
        dict = {'cause': cause_tmp[index, :], 'index': i}
        return dict
    if group is True:
        cause_tmp = igrouplasso(new_cell, alpha, sigma, lag_len, dt)
        index = list(range(1, i + 1)) + [0] + list(range(i + 1, n))
        dict = {'cause': cause_tmp[index, :], 'index': i}
        return dict



def wrap_worker(arg):
    """wrapper function"""
    return process_worker(*arg)


def search_optimum_lambda(cell_array, lambda_min, lambda_max, lag_len, group=False, grid=20):
    """ search optimum lambda parallel

    :param cell_array:
    :param lambda_max:
    :param lag_len:
    :param group:
    :return:
    """
    cv_error = np.zeros(grid)
    lambda_exponent = np.linspace(np.log10(lambda_min), np.log10(lambda_max), grid)
    total_features = len(cell_array)
    #for loop for featuers
    bar = pyprind.ProgBar(total_features*grid, width=60, bar_char='#', title='PROGRESS')
    for i in range(total_features):
        num_of_element = len(cell_array[i][0])
        avg_dt = (cell_array[i][1][-1] - cell_array[i][1][0]) / num_of_element
        assert avg_dt > 0, "avg_dt:" + str(avg_dt) + ", num_elem:" + str(num_of_element) + ", index:" + str(i)
        sigma = avg_dt / 4  # Comparison of correlation analysis techniques for irregularly sampled time series
        order = [i] + list(range(i)) + list(range(i + 1, total_features))
        new_cell = [cell_array[i] for i in order]
        if group is False:
            glg: object = GLG(new_cell, sigma, lag_len, avg_dt)
        argument_for_process = [(glg, 10 ** lambda_exponent[i], i) for i in range(grid)]
        # Parallel calculation for cross validation
        pool = Pool()
        for _, output in enumerate(pool.imap_unordered(cv_wrap_worker, argument_for_process)):
            bar.update()
            cv_error[output['index']] += output['error']

    print(bar)
    optimum = lambda_exponent[np.argmin(cv_error)]
    optimum = 10 ** optimum
    print("Optimum lambda", optimum)
    plt.scatter(lambda_exponent, cv_error)
    plt.show()
    #plt.savefig("lambda_search.png")
    return optimum


def cv_process_worker(glg, alpha, i):
    result, error = glg.crossvalidate(alpha)
    dict = {'cause': result, 'error': error, 'index': i}
    return dict


def cv_wrap_worker(arg):
    """wrapper function"""
    return cv_process_worker(*arg)

