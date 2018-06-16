#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata

import sys

sys.path.append('../granger_python')
import numpy as np
from run_ilasso import solve_loop
import pandas as pd
import matplotlib.pyplot as plt

cell = np.load('data.npy')
alpha = 1e-7
cause, aic, bic = solve_loop(cell, alpha, 3, cv=False, group=True)
np.savetxt('result3.csv', cause[:, :, 0], delimiter=',')
np.savetxt('result2.csv', cause[:, :, 1], delimiter=',')
np.savetxt('result1.csv', cause[:, :, 2], delimiter=',')

def mat_ans():
    #read matlab answer
    df = pd.read_csv('mat_ans_1.csv', sep=',', header=None)
    ans1 = df.values
    df = pd.read_csv('mat_ans_2.csv', sep=',', header=None)
    ans2 = df.values
    df = pd.read_csv('mat_ans_3.csv', sep=',', header=None)
    ans3 = df.values
    return [ans1, ans2, ans3]

def gen_ans():
    N = 20
    K1 = np.array([[0.3, 0, 0, 0], [0, 0.3, 0, 0], [0, 0, 0.3, 0], [0, 0, 0, 0.3]])
    K2 = np.array([[0.3, 0, 0, 0], [0, 0.3, 0, 0], [0, 0, 0.3, 0], [0, 0, 0, 0.3]])
    K3 = np.array([[0.3, 0, 0, 0], [1, 0.3, 0, 0], [1, 0, 0.3, 0], [1, 0, 0, 0.3]])

    A1 = np.kron(np.eye(int(N / 4)), K1)
    A2 = np.kron(np.eye(int(N / 4)), K2)
    A3 = np.kron(np.eye(int(N / 4)), K3)
    return [A1,A2,A3]

ans = gen_ans()
fig, axs = plt.subplots(3, 2)
for i in range(3):
    ax1 = axs[i, 0]
    ax1.spy(ans[i])
    #ax1.matshow(ans[i], cmap=plt.cm.Blues)
    ax1.set_title('Ground Truth')
    ax2 = axs[i, 1]
    cause[:, :, 2 - i][cause[:, :, 2 - i] > 0.25] = 1
    cause[:, :, 2 - i][cause[:, :, 2 - i] < 1] = 0
    #ax2.spy(cause[:, :, 2 - i])
    #ax2.matshow(cause[:, :, 2 - i], cmap=plt.cm.Blues)
    ax2.set_title('Inferred Causality')

plt.show()
