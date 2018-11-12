#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
from sklearn.linear_model import RandomizedLogisticRegression as RLR
import numpy as np
from sklearn.linear_model import LogisticRegression as LR

data_transaction = pd.read_csv('../data_temp/transaction_train_with_tag_new.csv')
data_transaction_round1_origin = pd.read_csv('../data_temp/transaction_round1_temp_new.csv')

data_transaction['trans_type2'] = data_transaction['trans_type2'].fillna(0)
data_transaction_round1_origin['trans_type2'] = data_transaction_round1_origin['trans_type2'].fillna(0)

x2_train = data_transaction.iloc[:,1:8].as_matrix()
y2_train = data_transaction.iloc[:,-1].as_matrix()

# rlr1 = RLR()
# rlr1.fit(x2_train, y2_train)
# res1 = rlr1.get_support()
# print(res1)
# x1 = data_operation[np.array(data_operation.iloc[:,:2].columns)[res1]].as_matrix()

lr2 = LR()
lr2.fit(x2_train, y2_train)
print('transaction score：', lr2.score(x2_train, y2_train))

x2_test = data_transaction_round1_origin.iloc[:,1:8].as_matrix()
y2_predict = lr2.predict_proba(x2_test)
y2_predict = pd.DataFrame(y2_predict)
y2_predict.to_csv('../data_temp/y2_predict.csv', index=False)
print(y2_predict)
# save1 = pd.DataFrame({'ID': data_operation_round1_origin['UID'], 'PROB': operation_predict})
# save1.to_csv('../data_temp/operation_round1_predict.csv', index=False)
#
# x2 = data_transaction.iloc[:,:6].as_matrix()
# y2 = data_transaction.iloc[:,6].as_matrix()
#
# rlr2 = RLR()
# rlr2.fit(x2, y2)
# res2 = rlr2.get_support()
# print(res2)
#
# x2 = data_transaction[np.array(data_transaction.iloc[:,:6].columns)[res2]].as_matrix()
# lr2 = LR()
# lr2.fit(x2, y2)
# print('2 score：', lr2.score(x2, y2))
#
# data_transaction_train = data_transaction_round1[np.array(data_transaction_round1.iloc[:,:6].columns)[res2]].as_matrix()
# transaction_predict = lr2.predict_proba(data_transaction_train)
#
# print(transaction_predict)


























