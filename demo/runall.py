#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for demonstration
"""
import numpy as np


def genSynth(N, T, sig):
    """generate simulation data

    :param N: number of time series (a multiple of 4)
    :param T: length of the time series
    :param sig: variance of the noise process
    :return:
    """
    assert N % 4 == 0, "N must be a multiple of 4"
    K = np.array([[0.9, 0, 0, 0], [1, 0.9, 0, 0], [1, 0, 0.9, 0], [1, 0, 0, 0.9]])
    A = np.kron(np.eye(int(N / 4)), K)
    series = np.zeros((N, T))
    series[:, 0] = np.random.randn(N)
    for t in range(T-1):
        series[:, t + 1] = A @ series[:, t] + sig * np.random.randn(N)

    return series, A


if __name__ == '__main__':
    genSynth(4, 10, 0.1)
