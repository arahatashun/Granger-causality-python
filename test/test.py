#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata

import sys
sys.path.append('../granger_python')
import numpy as np
from run_ilasso import solve_loop

cell = np.load('data.npy')
alpha = 1e-2
cause = solve_loop(cell, alpha, 0.1, 3)

np.savetxt('result3.csv', cause[:, :, 0], delimiter=',')
np.savetxt('result2.csv', cause[:, :, 1], delimiter=',')
np.savetxt('result1.csv', cause[:, :, 2], delimiter=',')
