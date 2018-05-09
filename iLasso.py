#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for irregular lasso Granger
"""
import numpy as np
import pandas as pd


def iLasso(time, values, alpha, kernel):
    """
    Learning teporal dependency among irregular time series ussing Lasso (or its variants)
    NOTE:Target is one variable.

    M. T. Bahadori and Yan Liu, "Granger Causality Analysis in Irregular Time Series", (SDM 2012)
    :param times: array of timestamp
    :param values: numpy array
    :param alpha:The regularization parameter in Lasso
    :param kernel:Gausian Kernel
    :return:
    """
    assert times.shape[0] == values.shape[0] "time and values dimension does not match"
    # Parameters
    L = 50  # Length of studied lag
    Dt = 0.5  # Delta t
    SIG = 2  # Kernel parameter. Here Gaussian Kernel Bandwidth
