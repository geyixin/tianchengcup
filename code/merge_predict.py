#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
import numpy as np

data_transaction_predict = pd.read_csv('../data_temp/transaction_round1_predict_new_10_10.csv')
data_operation_predict = pd.read_csv('../data_temp/operation_round1_predict_new3.csv')

'''
merge_1
'''


data_transaction_predict = data_transaction_predict.groupby(by='UID')['Tag'].mean()
data_operation_predict = data_operation_predict.groupby(by='UID')['Tag'].mean()
data_transaction_predict = pd.concat([data_transaction_predict],axis=1)
data_operation_predict = pd.concat([data_operation_predict], axis=1)
extra_ID = set(data_operation_predict.index) - set(data_transaction_predict.index)
data_operation_predict = data_operation_predict['Tag'].loc[np.array(list(extra_ID))]
data_operation_predict = pd.concat([data_operation_predict],axis=1)
data_merge_predict = pd.concat([data_transaction_predict, data_operation_predict])


'''
merge_2
'''

'''
data_merge_predict = pd.concat([data_transaction_predict, data_operation_predict])

data_merge_predict_list1 = []

tag_transaction_predict1 = data_merge_predict['Tag']


for i in tag_transaction_predict1:
    if i < 0.5:
        data_merge_predict_list1.append(0)
    else:
        data_merge_predict_list1.append(1)
data_merge_predict['Tag'] = data_merge_predict_list1

data_merge_predict2 = data_merge_predict.groupby(by='UID')['Tag'].mean()
'''

# temp = pd.read_csv('../data_temp/transaction_round1_predict_new4_2.csv')
#
# tag_transaction_predict2 = temp['Tag']
#
# data_merge_predict_list2 = []
#
# for i in tag_transaction_predict2:
#     if i < 0.5:
#         data_merge_predict_list2.append(0)
#     else:
#         data_merge_predict_list2.append(1)
# temp['Tag'] = data_merge_predict_list2

'''
merge_3
'''

# data_transaction_predict = data_transaction_predict.groupby(by='UID')['Tag'].mean()
# data_operation_predict = data_operation_predict.groupby(by='UID')['Tag'].mean()

# data_merge_predict = pd.concat([data_transaction_predict, data_operation_predict], axis=0)
# data_merge_predict = data_merge_predict.groupby(by='UID')['Tag'].mean()
#
data_merge_predict.sort_index(axis=0).to_csv('../data_temp/merge_by_trans_10_10_and_ope_10_10.csv')


# data_operation_predict = pd.read_csv('../data_temp/merge_by_trans_new2_and_ope_wmf_1.csv')
#
# data_transaction_predict = data_operation_predict.groupby(by='UID')['Tag'].mean()

# data_transaction_predict.sort_index(axis=0).to_csv('../data_temp/merge_by_trans_new2_and_ope_wmf_2.csv')