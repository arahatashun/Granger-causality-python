#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for irregular lasso Granger
"""
import numpy as np
import glmnet_python
from glmnet import glmnet


def ilasso(cell_list, alpha):
    """
    Learning temporal dependency among irregular time series ussing Lasso (or its variants)
    NOTE:Target is one variable.

    M. T. Bahadori and Yan Liu, "Granger Causality Analysis in Irregular Time Series", (SDM 2012)
    :param cell_list:one cell for each time series. Each cell is a 2xT matrix.
    First row contains the values and the second row contains SORTED time stamps.
    The first time series is the target time series which is predicted.
    :param alpha:The regularization parameter in Lasso
    :return (tuple) tuple containing:
        result: The NxL coefficient matrix.
    """
    # Parameters
    L = 3  # Length of studied lag
    # Delta t denotes the  average  length  of  the  sampling  intervals for the target time series
    Dt = 1  # Delta t
    SIG = 1 # Kernel parameter. Here Gaussian Kernel Bandwidth
    # index of first time which is larger than L*Dt
    B = np.argmax(cell_list[0][1, :] > L * Dt)
    # number of index of time of explained variable
    N1 = cell_list[0][1].shape[0]
    # number of features
    P = len(cell_list)

    # Build the matrix elements
    Am = np.zeros((N1 - B, P * L))  # explanatory variables
    bm = cell_list[0][0, B:N1]
    # for loop for stored time stamp
    for i in range(B, N1):
        ti = np.arange((cell_list[0][1, i] - L * Dt),
                       (cell_list[0][1, i] - Dt)+Dt, Dt)
        #for loop for features
        for j in range(P):
            assert len(ti) == L, "length does not match"
            tij = np.broadcast_to(ti, (len(cell_list[j][1,:]), ti.size))
            tSelect = np.broadcast_to(cell_list[j][1, :], (L, cell_list[j][1, :].size)).T
            ySelect = np.broadcast_to(cell_list[j][0, :], (L, cell_list[j][0, :].size)).T
            # kernel is used as window function??
            Kernel = np.exp(-(np.multiply((tij - tSelect),(tij - tSelect)) / SIG))
            Am[i-B,(j*L):(j+1)*L] = np.divide(np.sum(np.multiply(ySelect,Kernel),axis = 0),np.sum(Kernel))


    # Solving Lasso using a solver; here the 'GLMnet' package
    fit = glmnet(x=Am, y=bm, family='gaussian', alpha=1,
                 lambdau=np.array([alpha]))
    weight = fit['beta']  # array of coefficient

    # Computing the BIC and AIC metrics
    # TODO: be implemented

    # Reformatting the output
    result = np.zeros((P,L))
    for i in range(P):
        result[i,:] = weight[i*L:(i+1)*L].T
    # is this needed?
    result = np.fliplr(result)
    return result