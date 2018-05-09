#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for irregular lasso Granger
"""
import numpy as np
import pandas as pd


def iLasso(cell_list, alpha, kernel):
    """
    Learning teporal dependency among irregular time series ussing Lasso (or its variants)
    NOTE:Target is one variable.

    M. T. Bahadori and Yan Liu, "Granger Causality Analysis in Irregular Time Series", (SDM 2012)
    :param cell_list:one cell for each time series. Each cellis a 2xT matrix.
    First row contains the values and the second row contains SORTED time stamps.
    The first time series is the target time series which is predicted.
    :param alpha:The regularization parameter in Lasso
    :param kernel:Gausian Kernel
    :return:
    """
    assert times.shape[0] == values.shape[0]
    "time and values dimension does not match"
    # Parameters
    L = 50  # Length of studied lag
    # Delta t denotes the  average  length  of  the  sampling  intervals for the target time series
    Dt = 1  # Delta t
    SIG = 2  # Kernel parameter. Here Gaussian Kernel Bandwidth
    # index of first time which is L*Dt
    B = np.argmax(cell_list[0][1, :] > L * Dt)
    # number of index of time of explained variable
    N1 = cell_list[0][1, :].shape[2]
    # number of features
    P = len(cell_list)

    # Build the matrix elements
    Am = np.zeros((N1 - B + 1, P * L))  # explanatory variables
    bm = np.zeros((N1 - B + 1, 1))  # a explained variable
    bm = np.copy(cell_list[0][1,B:N1 + 1])
    assert bm.shape == (N1 - B + 1, 1), "bm dimension mismatch"
    # Building the design matrix
    for i in range(B, N1):
        ti = np.arange((cell_list[0][1,i] - L*Dt),(cell_list[0](1, i)-Dt), Dt))
        for j in range(P + 1):
