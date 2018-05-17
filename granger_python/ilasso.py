#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for irregular lasso Granger
"""
import numpy as np
import glmnet_python
from glmnet import glmnet
from numpy import linalg as LA


def ilasso(cell_list, alpha, sigma, lag_len, dt):
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
    :return (tuple) tuple containing:
        result: The NxL coefficient matrix.
    """

    # index of last time which is less than lag_len*dt　- 1
    B = np.argmax(cell_list[0][1, :] > lag_len * dt)
    assert B => 0, " lag_len DT error"
    # number of index of time of explained variable
    N1 = cell_list[0][1].shape[0]
    # number of features
    P = len(cell_list)

    # Build the matrix elements
    Am = np.zeros((N1 - B, P * lag_len))  # explanatory variables
    bm = cell_list[0][0, B:N1 + 1].reshape((N1 - B, 1))
    # for loop for stored time stamp
    for i in range(B, N1):
        ti = np.arange((cell_list[0][1, i] - lag_len * dt),
                       (cell_list[0][1, i] - dt) + dt, dt)
        # for loop for features
        for j in range(P):
            assert len(ti) == lag_len, "length does not match"
            tij = np.broadcast_to(ti, (len(cell_list[j][1, :]), ti.size))
            tSelect = np.broadcast_to(cell_list[j][1, :],
                                      (lag_len, cell_list[j][1, :].size)).T
            ySelect = np.broadcast_to(cell_list[j][0, :],
                                      (lag_len, cell_list[j][0, :].size)).T
            # kernel is used as window function??
            Kernel = np.exp(
                -(np.multiply((tij - tSelect), (tij - tSelect)) / sigma))
            Am[i - B, (j * lag_len):(j + 1) * lag_len] = np.divide(
                np.sum(np.multiply(ySelect, Kernel), axis=0),
                np.sum(Kernel, axis=0))

    # Solving Lasso using a solver; here the 'GLMnet' package
    fit = glmnet(x=Am, y=bm, family='gaussian', alpha=1,
                 lambdau=np.array([alpha]))
    weight = fit['beta']  # array of coefficient
    # Computing the BIC and AIC metrics
    # TODO: be implemented
    bic = LA.norm(Am @ weight - bm) ** 2 - np.log(N1 - B) * np.sum(
        weight == 0) / 2
    aic = LA.norm(Am @ weight - bm) ** 2 - 2 * np.sum(weight == 0) / 2
    # Reformatting the output
    result = np.zeros((P, lag_len))
    for i in range(P):
        result[i, :] = weight[i * lag_len:(i + 1) * lag_len].ravel()

    return result, aic, bic
