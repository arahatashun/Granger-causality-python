#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for irregular Group lasso Granger
"""
import sys

import numpy as np
from numpy import linalg as LA

sys.path.append('../../pylearn-parsimony')
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.functions.nesterov.gl as gl


# @profile
def igrouplasso(cell_list, alpha, sigma, lag_len, dt, cv):
    """
    Learning temporal dependency among irregular time series ussing Lasso (or its variants)
    NOTE:Target is one variable.

    M. T. Bahadori and Yan Liu, "Granger Causality Analysis in Irregular Time Series", (SDM 2012)
    :param cell_list:one cell for each time series. Each cell is a 2xT matrix.
    First row contains the values and the second row contains SORTED time stamps.
    The first time series is the target time series which is predicted.
    :param alpha:The regularization parameter in Lasso
    :param sigma:Kernel parameter. Here Gaussian Kernel Bandwidth
    :param lag_len: Length of studied lag
    :param dt:Delta t denotes the  average  length  of  the  sampling  intervals for the target time series
    :param cv:cross validation
    :return (tuple) tuple containing:
        result: The NxL coefficient matrix.
    """

    # for test
    # sigma = 0.1
    # alpha = 1e-2
    # dt = 1

    # index of last time which is less than lag_len*dt　- 1
    B = np.argmax(cell_list[0][1, :] > lag_len * dt)
    assert B >= 0, " lag_len DT error"
    # number of index of time of explained variable
    N1 = cell_list[0][1].shape[0]
    # number of features
    P = len(cell_list)
    # Build the matrix elements
    Am = np.zeros((N1 - B, P * lag_len))  # explanatory variables
    bm = cell_list[0][0, B:N1 + 1].reshape((N1 - B, 1))
    # for loop for stored time stamp
    for i in range(B, N1):
        # ti is defined by explained variable
        ti = np.linspace((cell_list[0][1, i] - lag_len * dt), cell_list[0][1, i] - dt, num=lag_len)
        # for loop for features
        for j in range(P):
            assert len(ti) == lag_len, str(len(ti)) + str(lag_len) + "length does not match"
            """
            tij = np.broadcast_to(ti, (len(cell_list[j][1, :]), ti.size))
            tSelect = np.broadcast_to(cell_list[j][1, :],
                                      (lag_len, cell_list[j][1, :].size)).T
            ySelect = np.broadcast_to(cell_list[j][0, :],
                                      (lag_len, cell_list[j][0, :].size)).T
            """
            # reduce kernel length in order to reduce complexity of calculation
            # kernel is used as window function
            # time_match is a cell_list[time_match][1, :] nearest to tij
            time_match = np.searchsorted(cell_list[j][1, :], cell_list[0][1, i])
            kernel_length = 25  # half of kernel length
            start = time_match - kernel_length if time_match - kernel_length > 0 else 0
            end = time_match + kernel_length if time_match + kernel_length < len(cell_list[j][1, :]) - 1 else len(
                cell_list[j][1, :])
            tij = np.broadcast_to(ti, (len(cell_list[j][1, start:end]), ti.size))
            tSelect = np.broadcast_to(cell_list[j][1, start:end],
                                      (lag_len, cell_list[j][1, start:end].size)).T
            ySelect = np.broadcast_to(cell_list[j][0, start:end],
                                      (lag_len, cell_list[j][0, start:end].size)).T
            exponent = -(np.multiply((tij - tSelect), (tij - tSelect)) / 2*sigma**2)
            # assert np.isfinite(exponent).all() == 1, str(exponent)
            Kernel = np.exp(exponent)
            # assert np.isfinite(Kernel).all() == 1, str(Kernel)
            with np.errstate(divide='ignore'):
                ker_sum = np.sum(Kernel, axis=0)
                numerator = np.sum(np.multiply(ySelect, Kernel), axis=0)
                # assert np.isfinite(numerator).all() ==1,str(numerator)
                # assert np.isfinite(ker_sum).all() ==1,str(ker_sum)
                tmp = np.divide(numerator, ker_sum)
                tmp[ker_sum == 0] = 1
                # assert (np.isfinite(tmp)).all() == 1,str(tmp)+str(ker_sum)
            """
            if np.sum(Kernel, axis=0).any() == 0:
                print("kernel zero" + str(np.sum(Kernel, axis=0)))
                print(tmp)
            """
            Am[i - B, (j * lag_len):(j + 1) * lag_len] = tmp

    # assert (np.isfinite(Am)).all() == True,str(Am)
    # Solving Lasso using a solver; here the 'GLMnet' package
    if cv == False:
        k = 0.0  # l2 ridge regression coefficient
        l = 0.0  # l1 lasso coefficient
        g = alpha  # group lasso coefficient
        # NOTE SLICEの向き

        lag_group = [np.arange(i, P * lag_len, lag_len) for i in range(lag_len)]
        groups = [lag_group[:i + 1] for i in range(lag_len)]
        for i in range(lag_len):
            ar_num = len(groups[i])
            tmp = groups[i][0]
            for j in range(ar_num - 1):
                tmp = np.append(tmp, groups[i][j + 1])
            groups[i] = tmp
        # print(len(groups))
        # print(Am.shape)
        A = gl.linear_operator_from_groups(P * lag_len, groups)
        estimator = estimators.LinearRegressionL1L2GL(
            k, l, g, A=A,
            algorithm=algorithms.proximal.FISTA(),
            algorithm_params=dict(max_iter=5000))
        res = estimator.fit(Am, bm)
        weight = res.beta

        # Computing the BIC and AIC metrics
        bic = LA.norm(Am @ weight - bm) ** 2 - np.log(N1 - B) * np.sum(
            weight == 0) / 2
        aic = LA.norm(Am @ weight - bm) ** 2 - 2 * np.sum(weight == 0) / 2

        # weight_shape_before = weight.shape
        # weight_shape_after = weight[np.logical_not(np.isnan(weight))].shape
        # assert np.isnan(weight).all() == False

        # Reformatting the output
        result = np.zeros((P, lag_len))
        for i in range(P):
            result[i, :] = weight[i * lag_len:(i + 1) * lag_len].ravel()

        return result, aic, bic
    else:
        last_index = int((N1 - B) * 0.7)
        Am_train = Am[:last_index]
        bm_train = bm[:last_index]
        Am_test = Am[last_index:]
        bm_test = bm[last_index:]

        k = 0.0  # l2 ridge regression coefficient
        l = 0.0  # l1 lasso coefficient
        g = alpha  # group lasso coefficient
        lag_group = [np.arange(i, P * lag_len, lag_len) for i in range(lag_len)]
        groups = [lag_group[:i + 1] for i in range(lag_len)]
        for i in range(lag_len):
            ar_num = len(groups[i])
            tmp = groups[i][0]
            for j in range(ar_num - 1):
                tmp = np.append(tmp, groups[i][j + 1])
            groups[i] = tmp
        A = gl.linear_operator_from_groups(P * lag_len, groups)
        estimator = estimators.LinearRegressionL1L2GL(
            k, l, g, A=A,
            algorithm=algorithms.proximal.FISTA(),
            algorithm_params=dict(max_iter=5000))
        res = estimator.fit(Am_train, bm_train)
        weight = res.beta

        test_error = LA.norm(Am_test @ weight - bm_test) ** 2 / (N1 - B - last_index)
        # Computing the BIC and AIC metrics
        bic = LA.norm(Am_train @ weight - bm_train) ** 2 - np.log(N1 - B) * np.sum(weight == 0) / 2
        aic = LA.norm(Am_train @ weight - bm_train) ** 2 - 2 * np.sum(weight == 0) / 2

        # weight_shape_before = weight.shape
        # weight_shape_after = weight[np.logical_not(np.isnan(weight))].shape
        # assert np.isnan(weight).all() == False

        # Reformatting the output
        result = np.zeros((P, lag_len))
        for i in range(P):
            result[i, :] = weight[i * lag_len:(i + 1) * lag_len].ravel()

        return result, aic, bic, test_error
