#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd


'''
operation数据特征值重写
'''

operation_round1 = pd.read_csv('../data/operation_round1_new.csv')
operation_TRAIN = pd.read_csv('../data/operation_train_new.csv')

operation_TRAIN['device2'] = operation_TRAIN['device2'].fillna(0)
TRAIN_device2 = operation_TRAIN['device2']

delete_list1 = []
for i in range(len(TRAIN_device2)):
    if TRAIN_device2[i] != 0:
        temp = TRAIN_device2[i].split(' ')[0]
        if 'IPHONE' == temp:
            delete_list1.append(i)

new_operation_TRAIN = operation_TRAIN.drop(delete_list1, axis=0)
A = new_operation_TRAIN['device1']
A = set(A)
print('A: ', len(A))

operation_round1['device2'] = operation_round1['device2'].fillna(0)
round1_device2 = operation_round1['device2']

delete_list2 = []
for i in range(len(round1_device2)):
    if round1_device2[i] != 0:
        temp = round1_device2[i].split(' ')[0]
        if 'IPHONE' == temp:
            delete_list2.append(i)

new_operation_round1 = operation_round1.drop(delete_list2, axis=0)
B = new_operation_round1['device1']
B = set(B)
print('B: ', len(B))

all_set = A | B

print('all_set: ',len(all_set))

str_all = ' '.join(list(all_set))

path_out = '../data_temp/' + 'device2' + '_new' + '.txt'
with open(path_out, 'w') as f:
    f.write(str_all)
