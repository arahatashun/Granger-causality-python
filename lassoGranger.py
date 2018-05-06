#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for lasso Granger
"""
import numpy as np
from numpy import linalg as LA
import glmnet_python
from glmnet import glmnet, glmnetSet


def lasso_granger(series, P, alpha):
    """Lasso Granger
    A. Arnold, Y. Liu, and N. Abe. Temporal causal modeling with graphical granger methods. In KDD, 200
    :param series: (N,T) matrix
    :param P: length of the lag
    :param alpha: value of the penalization parameter in Lasso
    :return:
    """
    N, T = np.shape(series)
    Am = np.zeros((T - P, P * N))
    bm = np.zeros((T - P, 1))
    for i in np.arange(P + 1, T):
        bm[i - P] = series[1, i]
        Am[i - P, :] = np.fliplr(series[:, i - P:i]).flatten()

    # Lasso using GLMnet
    fit = glmnet(x=Am, y=bm, family='gaussian', alpha=1, lambda_min=np.array([alpha]))
    vals2 = fit['beta']  # array of coefficient

    # Outputting aic metric for variable into (N,P) matrix
    th = 0
    aic = (LA.norm(Am @ vals2 - bm, 2))**2 / (T - P) + np.sum(np.abs(vals2) > th) * 2 / (T - P)

    # Reformatting the results into (N,P) matrix
    n1Coeff = np.zeros((N, P))
    for i in range(N):
        n1Coeff[i, :] = vals2[(i - 1) * P, i * P - 1]
    sumCause = np.sum(np.abs(n1Coeff), axis=1)
    sumCause[sumCause < th] = 0
    cause = sumCause
    return cause
