# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for irregular lasso Granger
implemented in Cython

Example:
    import pyximport; pyximport.install()
    from irregular_lasso import irregular_lasso
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

def irregular_lasso(cell_list, alpha):
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
    cdef int L = 3  # Length of studied lag
    # Delta t denotes the  average  length  of  the  sampling  intervals for the target time series
    cdef double Dt = 1  # Delta t
    cdef double SIG = 0.1  # Kernel parameter. Here Gaussian Kernel Bandwidth
    # index of last time which is less than L*Dt - 1
    cdef int B = np.argmax(cell_list[0][1, :] > L * Dt)
    # assert B > 0, " L DT error"
    # number of index of time of explained variable
    cdef int N1 = cell_list[0][1].shape[0]
    # number of features
    cdef int P = len(cell_list)

    # Build the matrix elements
    cdef np.ndarray[DTYPE_t, ndim=2] Am = np.zeros((N1 - B, P * L))  # explanatory variables
    cdef np.ndarray[DTYPE_t, ndim=2] bm = cell_list[0][0, B:N1 + 1].reshape((N1 - B, 1))
    # for loop for stored time stamp
    cdef i
    cdef j
    for i in range(B, N1):
        ti = np.arange((cell_list[0][1, i] - L * Dt),
                       (cell_list[0][1, i] - Dt) + Dt, Dt)
        # for loop for features
        for j in range(P):
            assert len(ti) == L, "length does not match"
            tij = np.broadcast_to(ti, (len(cell_list[j][1, :]), ti.size))
            tSelect = np.broadcast_to(cell_list[j][1, :], (L, cell_list[j][1, :].size)).T
            ySelect = np.broadcast_to(cell_list[j][0, :],(L, cell_list[j][0, :].size)).T
            # kernel is used as window function??
            Kernel = np.exp( -(np.multiply((tij - tSelect), (tij - tSelect)) / SIG))
            Am[i - B, (j * L):(j + 1) * L] = np.divide(
                np.sum(np.multiply(ySelect, Kernel), axis=0),
                np.sum(Kernel, axis=0))

    # Solving Lasso using a solver; here the 'GLMnet' package
    fit = glmnet(x=Am, y=bm, family='gaussian', alpha=1,
                 lambdau=np.array([alpha]))
    weight = fit['beta']  # array of coefficient
    # Computing the BIC and AIC metrics
    cdef double BIC = LA.norm(Am @ weight - bm) ** 2 - np.log(N1 - B) * np.sum(
        weight == 0) / 2
    cdef double AIC = LA.norm(Am @ weight - bm) ** 2 - 2 * np.sum(
        weight == 0) / 2
    # Reformatting the output
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros((P, L))
    for i in range(P):
        result[i, :] = weight[i * L:(i + 1) * L].ravel()

    return result
