#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""
Code for compare graph structure
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import matplotlib.axes

def f_score(a, b, threshold=0.1):
    """ Precision Recall F1 Score

    :param a: train
    :param b: test
    :return:
    """
    a = np.copy(a)
    b = np.copy(b)
    a[a < threshold] = 0
    a[a >= threshold] = 1
    b[b < threshold] = 0
    b[b >= threshold] = 1
    true_positive_mat = np.zeros_like(a)
    b_tmp = np.copy(b)
    b_tmp[b < threshold] = -1
    true_positive_mat[b_tmp == a] = 1
    true_positive = np.sum(true_positive_mat)
    precison = true_positive / np.sum(b)
    recall = true_positive / np.sum(a)
    score = 2 * precison * recall / (precison + recall)
    diff_mat = np.zeros_like(a)
    diff_mat[a >= threshold] += 1
    diff_mat[b >= threshold] += 2

    return precision, recall, score, diff_mat


def generate_cmap(colors):
    """自分で定義したカラーマップを返す
    https://qiita.com/kenmatsu4/items/fe8a2f1c34c8d5676df8
    """
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)


def get_sp_cmap():
    """color map for sparse matrix

    :return:
    """
    return generate_cmap(['white', 'blue', 'red', 'black'])
