#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: linjiexin
"""

import pandas as pd
import numpy as np
# import sklearn as sk
import math


def tree_grow(dataframe, target, min_leaf, min_dec_gini):
    tree = {}  # renew a tree
    is_not_leaf = (len(dataframe) > min_leaf)
    if is_not_leaf:
        fea, sp, gd = best_split_col(dataframe, target)
        if gd > min_dec_gini:
            tree['fea'] = fea
            tree['val'] = sp
            #dataframe.drop(fea,axis=1) #1116 modified
            l, r = dataSplit(dataframe, fea, sp)
            l.drop(fea, axis=1)
            r.drop(fea, axis=1)
            tree['left'] = tree_grow(l, target, min_leaf, min_dec_gini)
            tree['right'] = tree_grow(r, target, min_leaf, min_dec_gini)
        else:  # return a leaf
            return leaf(dataframe[target])
    else:
        return leaf(dataframe[target])

    return tree


def leaf(class_lable):
    tmp = {}
    for i in class_lable:
        if i in tmp:
            tmp[i] += 1
        else:
            tmp[i] = 1
    s = pd.Series(tmp)
    s.sort(ascending=False)

    return s.index[0]


def gini_cal(class_lable):
    p_1 = sum(class_lable) / len(class_lable)
    p_0 = 1 - p_1
    gini = 1 - (pow(p_0, 2) + pow(p_1, 2))

    return gini


def dataSplit(dataframe, split_fea, split_val):
    left_node = dataframe[dataframe[split_fea] <= split_val]
    right_node = dataframe[dataframe[split_fea] > split_val]

    return left_node, right_node


def best_split_col(dataframe, target_name):
    best_fea = ''  # modified 1116
    best_split_point = 0
    col_list = list(dataframe.columns)
    col_list.remove(target_name)
    gini_0 = gini_cal(dataframe[target_name])
    n = len(dataframe)
    gini_dec = -99999999
    for col in col_list:
        node = dataframe[[col, target_name]]
        unique = node.groupby(col).count().index
        for split_point in unique:  # unique value
            left_node, right_node = dataSplit(node, col, split_point)
            if len(left_node) > 0 and len(right_node) > 0:
                gini_col = gini_cal(left_node[target_name]) * (len(left_node) / n) + gini_cal(
                    right_node[target_name]) * (len(right_node) / n)
                if (gini_0 - gini_col) > gini_dec:
                    gini_dec = gini_0 - gini_col  # decrease of impurity
                    best_fea = col
                    best_split_point = split_point
                    # print(col,split_point,gini_0-gini_col)

    return best_fea, best_split_point, gini_dec


def model_prediction(model, row):  # row is a df

    fea = model['fea']
    val = model['val']
    left = model['left']
    right = model['right']
    if row[fea].tolist()[0] <= val:  # get the value
        branch = left
    else:
        branch = right
    if ('dict' in str(type(branch))):
        prediction = model_prediction(branch, row)
    else:
        prediction = branch

    return prediction
