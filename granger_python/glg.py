#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for Generalized lasso Granger
"""
import numpy as np
import glmnet_python
from glmnet import glmnet
from numpy import linalg as LA
import traceback

class GLG:
    def __init__(self, cell_list, sigma, lag_len, dt, index):
        """ starts pre processing
        :param cell_list:one cell for each time series. Each cell is a 2xT matrix.
        First row contains the values and the second row contains SORTED time stamps.
        The first time series is the target time series which is predicted.
        :param alpha:The regularization parameter in Lasso
        :param sigma:Kernel parameter. Here Gaussian Kernel Bandwidth
        :param lag_len: Length of studied lag
        :param dt:Delta t denotes the  average  length  of  the  sampling  intervals for the target time series
        """
        self.index = index
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
            self.Am = Am
            self.bm = bm
            self.lag_len = lag_len
            self.P = P
            self.N1 = N1
            self.B = B

    def calculate(self,alpha):
        Am = self.Am
        bm = self.bm
        try:
            fit = glmnet(x=Am, y=bm, family='gaussian', alpha=1, lambdau=np.array([alpha]))
        except:
            print('------------------')
            print("index:",self.index)
            print('------------------')
            traceback.print_exc()
            raise Exception('glmnet error')

        weight = fit['beta']  # array of coefficient
        # Reformatting the output
        result = np.zeros((self.P, self.lag_len))
        for i in range(self.P):
            result[i, :] = weight[i * self.lag_len:(i + 1) * self.lag_len].ravel()

        return result

    def crossvalidate(self,alpha):
        Am = self.Am
        bm = self.bm
        last_index = int((self.N1 - self.B) * 0.7)
        Am_train = Am[:last_index]
        bm_train = bm[:last_index]
        Am_test = Am[last_index:]
        bm_test = bm[last_index:]
        fit = glmnet(x=Am_train, y=bm_train, family='gaussian', alpha=1, lambdau=np.array([alpha]))
        weight = fit['beta']  # array of coefficient
        error = LA.norm(Am_test @ weight - bm_test) ** 2 / (self.N1 - self.B - last_index)

        # Reformatting the output
        result = np.zeros((self.P, self.lag_len))
        for i in range(self.P):
            result[i, :] = weight[i * self.lag_len:(i + 1) * self.lag_len].ravel()

        return result, error
