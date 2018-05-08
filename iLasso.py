#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for irregular lasso Granger
"""
import numpy as np

def iLasso(series,alpha,kernel):
    """
    Learning teporal dependency among irregular time series ussing Lasso (or its variants)
    M. T. Bahadori and Yan Liu, "Granger Causality Analysis in Irregular Time Series", (SDM 2012)
    :param series:an Nx1 cell array; one cell for each time series. Each cell
                is a 2xT matrix. First row contains the values and the
                second row contains SORTED time stamps. The first time
               series is the target time series which is predicted.
    :param alpha:The regularization parameter in Lasso
    :param kernel:Selects the kernel. Default is Gaussian. Available options
                are Sinc (kernel = Sinc) and Inverse distance (kernel = Dist).
    :return:
    """

    # Parameters
    L = 50 #Length of studied lag
    Dt = 0.5 #Delta t
    SIG = 2 #Kernel parameter. Here Gaussian Kernel Bandwidth

    B = np.sum(series)
