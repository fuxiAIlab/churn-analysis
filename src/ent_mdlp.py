# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import pandas as pd
import math
import time
from functools import reduce


def calc_entropy(x):
    x = pd.Series(x)
    ent1 = 0
    p = x.value_counts() / len(x)
    for i in range(len(p)):
        e = -p.iloc[i] * math.log(p.iloc[i])
        ent1 += e
    return ent1


def cut_index(x, y, x_quantile):
    n = len(x)
    x = pd.Series(x)
    entropy = 9999
    cut_d = None
    if x_quantile is not None:
        n_quantile = len(x_quantile)
        if n_quantile == 0:
            return None
        x_quantile = pd.Series(x_quantile)
    # 寻找最佳切分点
        for i in range(n_quantile):
            # if x_quantile.iloc[i+1] != x_quantile.iloc[i]:
            wCutX = x[x < x_quantile.iloc[i]]
            wn = len(wCutX) / n
            e1 = wn *calc_entropy(y[:len(wCutX)])
            e2 = (1 - wn) * calc_entropy(y[len(wCutX):])
            val = e1 + e2
            if val < entropy:
                entropy = val
                cut_d = np.searchsorted(x, x_quantile[i])
        if cut_d is None:
            return None
        else:
            return cut_d, entropy
    else:
        for i in range(n - 1):
            if x.iloc[i+1] != x.iloc[i]:
                wCutX = x[x < x.iloc[i+1]]
                wn = len(wCutX) / n
                e1 = wn * calc_entropy(y[:len(wCutX)])
                e2 = (1 - wn) * calc_entropy(y[len(wCutX):])
                val = e1 + e2
                if val < entropy:
                    entropy = val
                    cut_d = i
        if cut_d is None:
            return None
        else:
            return cut_d, entropy


def cut_stop(cut_d, y, entropy):
    n = len(y)
    es = calc_entropy(y)
    gain = es - entropy
    left = len(set(y[:cut_d]))
    right = len(set(y[cut_d:]))
    length_y = len(set(y))
    if cut_d is None or length_y == 0:
        return None
    else:
        delta = math.log(3 ** length_y - 2) - (length_y * calc_entropy(y) - left * calc_entropy(y[:cut_d])
                                               - right * calc_entropy(y[cut_d:]))
        cond = math.log(n - 1) / n + delta / n
        if gain < cond:
            return None
        else:
            return gain


def cut_points(x, y, quantile=None):
    dx = x.sort_values()
    dx_quantile = None
    if quantile and len(x.value_counts()) < quantile:
        quantile = None
    if quantile:
        dx_quantile = np.percentile(dx, np.arange(0, 100, 100/quantile))
    dy = pd.Series(y, index=dx.index)
    depth = 0

    def gr(low, upp, depth=depth, x1_quantile=None):
        x = dx[low: upp]
        y = dy[low: upp]
        n = len(y)
        k = cut_index(x, y, x_quantile=x1_quantile)
        if k is None:
            return None
        else:
            cut_d = k[0]
            entropy = k[1]
            gain = cut_stop(cut_d, y, entropy)
            if gain is None:
                return None
            else:
                return [cut_d, depth + 1]

    def part(low=0, upp=len(dx), cut_td1=[], depth=depth):
        x1 = dx[low: upp]
        y1 = dy[low: upp]
        x1_quantile = None
        if quantile and len(x1) > 0:
            x1_quantile = [dx for dx in dx_quantile if x1.iloc[0] < dx < x1.iloc[-1]]
            # print('x1:', x1.iloc[0])
            # print('x1last:', x1.iloc[-1])
            # print('len():', len(x1_quantile))
        n = len(x1)
        k = gr(low, upp, depth=depth, x1_quantile=x1_quantile)
        if n < 2 or k is None:
            return cut_td1
        else:
            cut_x = k[0]
            depth += 1
            cut_td1.append(low + cut_x)
            cut_td1.sort()
            return part(low, low + cut_x, cut_td1, depth) + part(cut_x + low, upp, cut_td1, depth)

    res1 = part(low=0, upp=len(dx), cut_td1=[], depth=depth)
    cut_dx = []
    if not res1:
        return None
    # 去重
    func = lambda x, y: x if y in x else x + [y]
    res = reduce(func, [[], ] + res1)
    res = pd.Series(res)
    for i in res.values:
        k = round((x.sort_values().values[i] + x.sort_values().values[i+1])/2, 6)
        cut_dx.append(k)
    return cut_dx


def mdlp(X, y, continuous_features=None, quantile=None):
    if not continuous_features:
        continuous_features = range(X.shape[1])
    cut_p = {}
    cut_ps = {}
    for i in continuous_features:
        # i = 190
        x = X.iloc[:, i]
        print('index:{}, feature:{}, value_count:{}'.format(i, X.columns[i], len(x.value_counts())))
        start = time.time()
        cuts1 = cut_points(x, y, quantile)
        end = time.time()
        if cuts1 is None:
            # cuts1 = 'ALL'
            cuts1 = [max(x)]
        cuts = [[min(x)], cuts1, [max(x)]]
        cut_ps[i] = cuts
        cut_p[i] = cuts1
        # cut_ps.append(cuts)
        # cut_p.append(cuts1)
        print('time:{},cut_num:{}'.format(end-start, len(cuts1)))
        # print(cuts)
        # print(cuts1)
        # print(X[X.columns[i]])
        # a = np.searchsorted(cuts1, X.values[:, i])
        # print(a)
    return cut_p, cut_ps


def mdlpx(X, y, continuous_features=None, quantile=None):
    if not continuous_features:
        continuous_features = range(X.shape[1])
    cut, cut_all = mdlp(X, y, continuous_features, quantile=quantile)
    dic = {}
    dic_all = {}
    output = X.copy()
    for i in continuous_features:
        print(i, X.columns[i])
        dic[X.columns[i]] = cut[i]
        dic_all[X.columns[i]] = cut_all[i]
        output[X.columns[i]] = np.searchsorted(cut[i], X[X.columns[i]])
    return output, dic, dic_all











