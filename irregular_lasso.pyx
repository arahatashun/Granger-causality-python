# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for irregular lasso Granger
implemented in Cython

"""
import numpy as np
cimport numpy as np
cimport cython

np.import_array()
import glmnet_python
from glmnet import glmnet
from numpy import linalg as LA

DTYPE = np.double
ctypedef np.double_t DTYPE_t

def irregular_lasso(cell_list, double alpha, int lag_len):
    """
    Learning temporal dependency among irregular time series ussing Lasso (or its variants)
    NOTE:Target is one variable.

    M. T. Bahadori and Yan Liu, "Granger Causality Analysis in Irregular Time Series", (SDM 2012)
    :param cell_list:one cell for each time series. Each cell is a 2xT matrix.
    First row contains the values and the second row contains SORTED time stamps.
    The first time series is the target time series which is predicted.
    :param alpha:The regularization parameter in Lasso
    :param lag_len:Length of studied lag
    :return (tuple) tuple containing:
        result: The NxL coefficient matrix.
        AIC:
        BIC:
    """
    # Parameters
    # Delta t denotes the  average  length  of  the  sampling  intervals for the target time series
    cdef double dt = 1  # Delta t
    cdef double sigma = 0.1  # kernel parameter. Here Gaussian kernel Bandwidth
    # index of last time which is less than lag_len*dt - 1
    cdef int begin = np.argmax(cell_list[0][1, :] > lag_len * dt)
    # assert begin > 0, " lag_len dT error"
    # number of index of time of explained variable
    cdef int len_of_explained = cell_list[0][1].shape[0]
    # number of features
    cdef int num_of_features = len(cell_list)

    # Build the matrix elements
    cdef np.ndarray[DTYPE_t, ndim=2] a_matrix = np.zeros(
        (len_of_explained - begin, num_of_features * lag_len))  # explanatory variables
    cdef np.ndarray[DTYPE_t, ndim=2] b_vector = cell_list[0][0,
                                                begin:len_of_explained + 1].reshape(
        (len_of_explained - begin, 1))
    # for loop for stored time stamp
    cdef i
    cdef j
    for i in range(begin, len_of_explained):
        ti = np.arange((cell_list[0][1, i] - lag_len * dt),
                       (cell_list[0][1, i] - dt) + dt, dt)
        # for loop for features
        for j in range(num_of_features):
            assert len(ti) == lag_len, "length does not match"
            tij = np.broadcast_to(ti, (len(cell_list[j][1, :]), ti.size))
            t_select = np.broadcast_to(cell_list[j][1, :],
                                       (lag_len, cell_list[j][1, :].size)).T
            y_select = np.broadcast_to(cell_list[j][0, :],
                                       (lag_len, cell_list[j][0, :].size)).T
            # kernel is used as window function??
            kernel = np.exp(
                -(np.multiply((tij - t_select), (tij - t_select)) / sigma))
            a_matrix[i - begin, (j * lag_len):(j + 1) * lag_len] = np.divide(
                np.sum(np.multiply(y_select, kernel), axis=0),
                np.sum(kernel, axis=0))

    # Solving Lasso using a solver; here the 'GLMnet' package
    fit = glmnet(x=a_matrix, y=b_vector, family='gaussian', alpha=1,
                 lambdau=np.array([alpha]))
    weight = fit['beta']  # array of coefficient
    # Computing the BIC and AIC metrics
    cdef double bic = LA.norm(a_matrix @ weight - b_vector) ** 2 - np.log(
        len_of_explained - begin) * np.sum(
        weight == 0) / 2
    cdef double aic = LA.norm(a_matrix @ weight - b_vector) ** 2 - 2 * np.sum(
        weight == 0) / 2
    # Reformatting the output
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros(
        (num_of_features, lag_len))
    for i in range(num_of_features):
        result[i, :] = weight[i * lag_len:(i + 1) * lag_len].ravel()

    return result, aic, bic
