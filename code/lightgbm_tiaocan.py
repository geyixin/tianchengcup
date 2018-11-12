#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import metrics
import numpy as np

'''
transaction数据 训练+预测
'''


data_transaction = pd.read_csv('../data_temp/transaction_train_with_tag_new3.csv')
data_transaction_round1_origin = pd.read_csv('../data_temp/transaction_round1_temp_new3.csv')

train_xy = data_transaction
train, val = train_test_split(train_xy, test_size=0.2)

y2_train = train.Tag
x2_train = train.drop(['UID', 'Tag'], axis=1)

y2_val = val.Tag
x2_val = val.drop(['UID', 'Tag'], axis=1)

online_test = data_transaction_round1_origin
online_test_x2 = online_test.drop(['UID'], axis=1)

lgb_train = lgb.Dataset(x2_train, y2_train, free_raw_data=False)
lgb_eval = lgb.Dataset(x2_val, y2_val, reference=lgb_train, free_raw_data=False)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
}

min_error = float('Inf')
best_params = {}

for num_leaves in range(20, 50, 5):
    for max_depth in range(2, 8, 1):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth

        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=2018,
                            nfold=3,
                            metrics=['binary_error'],
                            early_stopping_rounds=10,
                            verbose_eval=True
                            )

        mean_error = pd.Series(cv_results['binary_error-mean']).min()
        boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

        if mean_error < min_error:
            min_error = mean_error
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth

params['num_leaves'] = best_params['num_leaves']
params['max_depth'] = best_params['max_depth']

for max_bin in range(50, 260, 5):
    for min_data_in_leaf in range(2, 24, 2):
        params['max_bin'] = max_bin
        params['min_data_in_leaf'] = min_data_in_leaf

        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=42,
            nfold=3,
            metrics=['binary_error'],
            early_stopping_rounds=3,
            verbose_eval=True
        )

        mean_error = pd.Series(cv_results['binary_error-mean']).min()
        boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

        if mean_error < min_error:
            min_error = mean_error
            best_params['max_bin'] = max_bin
            best_params['min_data_in_leaf'] = min_data_in_leaf

params['min_data_in_leaf'] = best_params['min_data_in_leaf']
params['max_bin'] = best_params['max_bin']

for feature_fraction in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for bagging_fraction in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for bagging_freq in range(0, 15, 5):
            params['feature_fraction'] = feature_fraction
            params['bagging_fraction'] = bagging_fraction
            params['bagging_freq'] = bagging_freq

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=42,
                nfold=3,
                metrics=['binary_error'],
                early_stopping_rounds=3,
                verbose_eval=True
            )

            mean_error = pd.Series(cv_results['binary_error-mean']).min()
            boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

            if mean_error < min_error:
                min_error = mean_error
                best_params['feature_fraction'] = feature_fraction
                best_params['bagging_fraction'] = bagging_fraction
                best_params['bagging_freq'] = bagging_freq

params['feature_fraction'] = best_params['feature_fraction']
params['bagging_fraction'] = best_params['bagging_fraction']
params['bagging_freq'] = best_params['bagging_freq']

# for lambda_l1 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     for lambda_l2 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#         for min_split_gain in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#             params['lambda_l1'] = lambda_l1
#             params['lambda_l2'] = lambda_l2
#             params['min_split_gain'] = min_split_gain
#
#             cv_results = lgb.cv(
#                 params,
#                 lgb_train,
#                 seed=42,
#                 nfold=3,
#                 metrics=['binary_error'],
#                 early_stopping_rounds=3,
#                 verbose_eval=True
#             )
#
#             mean_error = pd.Series(cv_results['binary_error-mean']).min()
#             boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
#
#             if mean_error < min_error:
#                 min_error = mean_error
#                 best_params['lambda_l1'] = lambda_l1
#                 best_params['lambda_l2'] = lambda_l2
#                 best_params['min_split_gain'] = min_split_gain
#
# params['lambda_l1'] = best_params['lambda_l1']
# params['lambda_l2'] = best_params['lambda_l2']
# params['min_split_gain'] = best_params['min_split_gain']

print('final params: ', params)

gbm = lgb.train(
          params,
          lgb_train,
          valid_sets=lgb_eval,
          num_boost_round=2000,
          early_stopping_rounds=50,
          )

preds_online = gbm.predict(online_test_x2, num_iteration=gbm.best_iteration)  # 输出概率
online = online_test[['UID']]
online['P'] = preds_online
online.rename(columns={'P': 'Tag'}, inplace=True)           # 更改列名

# online.to_csv('../data_temp/transaction_round1_predict_new_2.csv', index=False)

joblib.dump(lgb, 'lgb.pkl')

df = pd.DataFrame(x2_train.columns.tolist(), columns=['feature'])
df['importance'] = list(lgb.feature_importance())
df = df.sort_values(by='importance', ascending=False)
df.to_csv('../data_temp/feature_score_1.csv', index=None, encoding='gbk')

'''
operation数据 训练+预测
'''

'''
data_operation = pd.read_csv('../data_temp/operation_train_with_tag_new2.csv')
data_operation_round1_origin = pd.read_csv('../data_temp/operation_round1_temp_new2.csv')

data_operation['success'] = data_operation['success'].fillna(0)
data_operation_round1_origin['success'] = data_operation_round1_origin['success'].fillna(0)

train_xy = data_operation
train, val = train_test_split(train_xy, test_size=0.2, random_state=21)

y2_train = train.Tag
x2_train = train.drop(['UID', 'Tag'], axis=1)

y2_val = val.Tag
x2_val = val.drop(['UID', 'Tag'], axis=1)

online_test = data_operation_round1_origin
online_test_x2 = online_test.drop(['UID'], axis=1)

lgb_train = lgb.Dataset(x2_train, y2_train, free_raw_data=False)
lgb_eval = lgb.Dataset(x2_val, y2_val, reference=lgb_train, free_raw_data=False)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
}

min_error = float('Inf') 
best_params = {}

for num_leaves in range(20,200,5):
    for max_depth in range(3,8,1):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth

        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=2018,
                            nfold=3,
                            metrics=['binary_error'],
                            early_stopping_rounds=10,
                            verbose_eval=True
                            )

        mean_error = pd.Series(cv_results['binary_error-mean']).min()
        boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

        if mean_error < min_error:
            min_error = mean_error
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth

params['num_leaves'] = best_params['num_leaves']
params['max_depth'] = best_params['max_depth']

params['learning_rate'] = 0.01
gbm = lgb.train(
          params,                     # 参数字典
          lgb_train,                  # 训练集
          valid_sets=lgb_eval,        # 验证集
          num_boost_round=2000,       # 迭代次数
          early_stopping_rounds=50    # 早停次数
          )
preds_online = gbm.predict(online_test_x2, num_boost_round=950)  # 输出概率
# preds_online = gbm.predict(online_test_x2, num_boost_round=950, num_iteration=gbm.best_iteration)  # 输出概率
online = online_test[['UID']]
online['P'] = preds_online
online.rename(columns={'P': 'Tag'}, inplace=True)           # 更改列名
online.to_csv('../data_temp/operation_round1_predict_new_2.csv', index=False)
'''










