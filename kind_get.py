#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd

'''
transaction数据特征值重写
'''

'''
params_set = ['amt_src1', 'trans_type1', 'geo2', 'geo1']

operation_round1 = pd.read_csv('../data/transaction_round1_new_2.csv', encoding='ISO-8859-1')
operation_TRAIN = pd.read_csv('../data/transaction_train_new_2.csv', encoding='ISO-8859-1')

for param in params_set:

    A = operation_round1[param]
    # A = list(A[:])
    A = set(A)

    B = operation_TRAIN[param]
    # B = list(B[:])
    B = set(B)

    all_set = A | B
    print(len(all_set))
    str_all = ' '.join(all_set)

    path_out = '../data_temp/' + param + '_new' + '_transaction' + '.txt'
    with open(path_out, 'w') as f:
        f.write(str_all)
'''

'''
transaction_round1 = pd.read_csv('../data/transaction_round1_new.csv')

geo_code_train = transaction_round1['geo_code'].fillna(-1)

geo_qian2_train = []
geo_qian1_train = []

for i in geo_code_train:
    if i != -1:
        geo_qian2_train.append(i[:2])
        geo_qian1_train.append(i[:1])
    else:
        geo_qian2_train.append(i)
        geo_qian1_train.append(i)

transaction_round1['geo2'] = geo_qian2_train
transaction_round1['geo1'] = geo_qian1_train

pd.DataFrame(transaction_round1).to_csv('../data/transaction_round1_new_2.csv', index=False)
'''

'''
operation数据特征值重写
'''


params_set = ['mode', 'geo2', 'geo1']

operation_round1 = pd.read_csv('../data/operation_round1_new_2.csv', encoding='ISO-8859-1')
operation_TRAIN = pd.read_csv('../data/operation_train_new_2.csv', encoding='ISO-8859-1')

for param in params_set:
    A = operation_round1[param]
    # A = list(A[:])
    A = set(A)

    B = operation_TRAIN[param]
    # B = list(B[:])
    B = set(B)

    all_set = A | B
    print(len(all_set))
    str_all = ' '.join(list(all_set))

    path_out = '../data_temp/' + param + '_new' + '_operation' + '.txt'
    with open(path_out, 'w') as f:
        f.write(str_all)

