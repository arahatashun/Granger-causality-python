#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for irregular lasso Granger
"""
import numpy as np
import pandas as pd


def iLasso(df: pd.DataFrame, alpha, kernel):
    """
    Learning teporal dependency among irregular time series ussing Lasso (or its variants)
    NOTE:Target is one variable.

    M. T. Bahadori and Yan Liu, "Granger Causality Analysis in Irregular Time Series", (SDM 2012)
    :param series: pandas DataFrame
        index must be timestamp
    :param alpha:The regularization parameter in Lasso
    :param kernel:Selects the kernel. Default is Gaussian. Available options
                are Sinc (kernel = Sinc) and Inverse distance (kernel = Dist).
    :return:
    """

    # Parameters
    L = 50  # Length of studied lag
    Dt = 0.5  # Delta t
    SIG = 2  # Kernel parameter. Here Gaussian Kernel Bandwidth

    #Limit the df to the first L *DT
    df_limited =
    Begin = np.sum(df_limited.index < L * DT) # number of elements of 1st variable
