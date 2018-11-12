#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

'''
transaction数据 训练+预测
'''

'''
data_transaction = pd.read_csv('../data_temp/transaction_train_with_tag_new_10_10.csv')
data_transaction_round1_origin = pd.read_csv('../data_temp/transaction_round1_temp_new_10_10.csv')

x2_train = data_transaction.drop(['UID', 'Tag', 'geo1', 'market_type', 'Is_geo', 'channel', 'Is_apple',
                                  'trans_type2'], axis=1)
y2_train = data_transaction.iloc[:,-1]
x2_test = data_transaction_round1_origin.drop(['UID', 'geo1', 'market_type', 'Is_geo', 'channel',
                                               'Is_apple', 'trans_type2'], axis=1)

lgb_train = lgb.Dataset(x2_train, y2_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': { 'auc'},
    'num_leaves': 36,
    'learning_rate': 0.01,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.7,
    'random_state': 2018,
}
print('Start training...')
gbm = lgb.train(params, lgb_train, num_boost_round=950)
print('Start predict...')
predict2 = gbm.predict(x2_test)

# for i in range(len(predict2)):
#     if predict2[i] >= 0.5:
#         predict2[i] = 1
#     else:
#         predict2[i] = 0
# acc = accuracy_score(predict2, np.array(data_transaction['Tag']))
# print(acc)
# plt.figure()
# lgb.plot_importance(gbm)
# plt.show()
#
save2 = pd.DataFrame({'UID': data_transaction_round1_origin['UID'], 'Tag': predict2})
save2.to_csv('../data_temp/transaction_round1_predict_new_10_10.csv', index=False)
'''

'''
operation数据 训练+预测
'''


data_operation = pd.read_csv('../data_temp/operation_train_with_tag_new_10_10.csv')
data_operation_round1_origin = pd.read_csv('../data_temp/operation_round1_temp_10_10.csv')

x1_train = data_operation.drop(['UID', 'Tag', 'Is_geo', 'Is_apple', 'success', 'geo1'], axis=1)
y1_train = data_operation.iloc[:,-1]
x1_test = data_operation_round1_origin.drop(['UID', 'Is_geo', 'Is_apple', 'success', 'geo1'], axis=1)

# print(x1_train.head(3))
# # print(y1_train.head(3))
# # print(x1_test.head(3))

lgb_train = lgb.Dataset(x1_train, y1_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': { 'auc'},
    'num_leaves': 36,
    'learning_rate': 0.001,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.7,
    'random_state':2018,
}

print('Start training...')
gbm = lgb.train(params, lgb_train, num_boost_round=950)
print('Start predict...')
predict1 = gbm.predict(x1_test)

lgb.plot_importance(gbm)
plt.show()

save1 = pd.DataFrame({'UID': data_operation_round1_origin['UID'], 'Tag': predict1})
save1.to_csv('../data_temp/operation_round1_predict_new_10_10.csv', index=False)










