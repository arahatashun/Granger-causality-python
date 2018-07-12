#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for Hierarchical Generalized lasso Granger
"""
from numpy import linalg as LA
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import gc
from glg import GLG

class HGLG(GLG):
    def __init__(self, cell_list, sigma, lag_len, dt):
        """ starts pre processing
            :param cell_list:one cell for each time series. Each cell is a 2xT matrix.
            First row contains the values and the second row contains SORTED time stamps.
            The first time series is the target time series which is predicted.
            :param alpha:The regularization parameter in Lasso
            :param sigma:Kernel parameter. Here Gaussian Kernel Bandwidth
            :param lag_len: Length of studied lag
            :param dt:Delta t denotes the  average  length  of  the  sampling  intervals for the target time series
        """
        super().__init__(cell_list, sigma, lag_len, dt)
        groups = []
        for i in range(self.lag_len):
            for j in range(self.P):
                # NOTE dype must be float
                groups.append(self.lag_len * j + np.array([self.lag_len - i for i in range(i + 1)], dtype=np.float))
        r = robjects.r
        r_vector = [r.c(*groups[i]) for i in range(len(groups))]
        self.r_group = r.list(*r_vector)

    def calculate(self, alpha):
        """override function """
        grpregOverlap = importr('grpregOverlap')
        r = robjects.r
        robjects.r('''
                        # create a function `f`
                        r_grpregOverlap <- function(X, y, group, alpha, verbose = FALSE) {
                            if (verbose) {
                                cat("I am calling f().\n")
                            }
                           grpregOverlap(X, y, group, penalty ='grLasso',lambda = alpha)
                        }
                        ''')
        r_grpregOverlap = robjects.globalenv['r_grpregOverlap']
        fit = r_grpregOverlap(self.Am, self.bm, self.r_group, alpha)
        gc.collect()
        del self.Am
        del self.bm
        del grpregOverlap
        weight = np.asarray(r.coef(fit))[1:]  # remove intercept
        gc.collect()


        # Reformatting the output
        result = np.zeros((self.P, self.lag_len))

        for i in range(self.P):
            result[i, :] = weight[i * self.lag_len:(i + 1) * self.lag_len]

        return result

    def crossvalidate(self,alpha):
        r = robjects.r
        Am = self.Am
        bm = self.bm
        last_index = int((self.N1 - self.B) * 0.7)
        Am_train = Am[:last_index]
        bm_train = bm[:last_index]
        Am_test = Am[last_index:]
        bm_test = bm[last_index:]
        grpregOverlap = importr('grpregOverlap')
        robjects.r('''
                                # create a function `f`
                                r_grpregOverlap <- function(X, y, group, alpha, verbose = FALSE) {
                                    if (verbose) {
                                        cat("I am calling f().\n")
                                    }
                                   grpregOverlap(X, y, group, penalty ='grLasso',lambda = alpha)
                                }
                                ''')
        r_grpregOverlap = robjects.globalenv['r_grpregOverlap']
        fit = r_grpregOverlap(Am_test, bm_test, self.r_group, alpha)
        gc.collect()
        del grpregOverlap
        weight = np.asarray(r.coef(fit))[1:]  # remove intercept
        intercept = np.asarray(r.coef(fit))[0:]
        del r
        gc.collect()
        error = LA.norm(Am_test @ weight - bm_test-intercept) ** 2 / (self.N1 - self.B - last_index)

        # Reformatting the output
        result = np.zeros((self.P, self.lag_len))
        for i in range(self.P):
            result[i, :] = weight[i * self.lag_len:(i + 1) * self.lag_len].ravel()

        return result, error