#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for demonstration
"""
import numpy as np
import matplotlib.pyplot as plt
from lassoGranger import lasso_granger


def gen_synth(N, T, sig):
    """generate simulation data

    :param N: number of time series (a multiple of 4)
    :param T: length of the time series
    :param sig: variance of the noise process
    :return:(tuple): tuple containing:

        series: times series data
        A: Kronecker tensor product
    """
    assert N % 4 == 0, "N must be a multiple of 4"
    K = np.array([[0.9, 0, 0, 0], [1, 0.9, 0, 0], [1, 0, 0.9, 0], [1, 0, 0, 0.9]])
    A = np.kron(np.eye(int(N / 4)), K)
    series = np.zeros((N, T))
    series[:, 0] = np.random.randn(N)
    for t in range(T - 1):
        series[:, t + 1] = A @ series[:, t] + sig * np.random.randn(N)

    return series, A


def main():
    # generate synthetic data set
    N = 20  # number of time series
    T = 101  # length of time series
    sig = 0.2
    series, A = gen_synth(N, T, sig)
    # Run Lasso-Granger
    alpha = 1e-2
    L = 1  # only one lag for analysis
    cause = np.zeros((N, N))
    for i in range(N):
        index = [i] + list(range(i)) + list(range(i + 1, N))
        cause_tmp = lasso_granger(series[index, :], L, alpha)
        index = list(range(1, i+1)) + [0] + list(range(i+1, N))
        cause[i, :] = cause_tmp[index]

    fig, axs = plt.subplots(1, 2)
    ax1 = axs[0]
    ax2 = axs[1]
    ax1.spy(A)
    ax1.set_title('Ground Truth')
    ax2.spy(cause)
    ax2.set_title('Inferred Causality')
    plt.show()

if __name__ == '__main__':
    main()
