#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for irregular Group lasso Granger
"""

from numpy import linalg as LA
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import gc
rpy2.robjects.numpy2ri.activate()
grpregOverlap= importr('grpregOverlap')
robjects.r('''
                # create a function `f`
                r_grpregOverlap <- function(X, y, group, alpha, verbose = FALSE) {
                    if (verbose) {
                        cat("I am calling f().\n")
                    }
                   grpregOverlap(X, y, group, penalty ='grLasso',lambda = alpha)
                }
                ''')

# @profile
def igrouplasso(cell_list, alpha, sigma, lag_len, dt, cv = False):
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
            exponent = -(np.multiply((tij - tSelect), (tij - tSelect)) / 2 * sigma ** 2)
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

    gc.collect()
    # assert (np.isfinite(Am)).all() == True,str(Am)
    # Solving Lasso using a solver; here the 'GLMnet' package
    # NOTE SLICEの向き
    groups = []
    for i in range(lag_len):
        for j in range(P):
            # NOTE dype must be float
            groups.append(lag_len * j + np.array([lag_len - i for i in range(i + 1)], dtype=np.float))

    r = robjects.r
    r_vector = [r.c(*groups[i]) for i in range(len(groups))]
    r_group = r.list(*r_vector)
    r_grpregOverlap = robjects.globalenv['r_grpregOverlap']
    if cv == False:
        fit = r_grpregOverlap(Am, bm, r_group, alpha)
        gc.collect()
        del Am
        del bm
        del r_group
        del r_vector
        weight = np.asarray(r.coef(fit))[1:] # remove intercept
        del r
        gc.collect()
        # Computing the BIC and AIC metrics
        """
        bic = LA.norm(Am @ weight - bm) ** 2 - np.log(N1 - B) * np.sum(
            weight == 0) / 2
        aic = LA.norm(Am @ weight - bm) ** 2 - 2 * np.sum(weight == 0) / 2
        """
        # weight_shape_before = weight.shape
        # weight_shape_after = weight[np.logical_not(np.isnan(weight))].shape
        # assert np.isnan(weight).all() == False

        # Reformatting the output
        result = np.zeros((P, lag_len))

        for i in range(P):
            result[i, :] = weight[i * lag_len:(i + 1) * lag_len]

        return result, 0, 0
    else:
        last_index = int((N1 - B) * 0.7)
        Am_train = Am[:last_index]
        bm_train = bm[:last_index]
        Am_test = Am[last_index:]
        bm_test = bm[last_index:]
        fit = r_grpregOverlap(Am_train, bm_train, r_group, alpha)
        gc.collect()
        del Am
        del bm
        del r_group
        del r_vector
        weight = np.asarray(r.coef(fit))[1:]  # remove intercept
        intercept = np.asarray(r.coef(fit))[0]
        del r
        gc.collect()
        test_error = LA.norm(Am_test @ weight - bm_test - intercept) ** 2 / (N1 - B - last_index)
        # Computing the BIC and AIC metrics
        # weight_shape_before = weight.shape
        # weight_shape_after = weight[np.logical_not(np.isnan(weight))].shape
        # assert np.isnan(weight).all() == False

        # Reformatting the output
        result = np.zeros((P, lag_len))
        for i in range(P):
            result[i, :] = weight[i * lag_len:(i + 1) * lag_len].ravel()

        return result, 0, 0, test_error

