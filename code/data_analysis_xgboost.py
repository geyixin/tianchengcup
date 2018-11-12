#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import accuracy_score


'''
transaction数据 训练+预测
'''

data_transaction = pd.read_csv('../data_temp/transaction_train_with_tag_new2.csv')
data_transaction_round1_origin = pd.read_csv('../data_temp/transaction_round1_temp_new2.csv')

x2_train = data_transaction.drop(['UID', 'Tag'], axis=1)
y2_train = data_transaction.iloc[:,-1]
x2_test = data_transaction_round1_origin.drop(['UID'], axis=1)

model = xgb.XGBRegressor(
    max_depth=10,
    learning_rate=0.1,
    n_estimators=10,
    silent=True,
    objective='reg:linear',
    nthread=-1,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    seed=1440,
    missing=None
)

model.fit(x2_train, y2_train, eval_metric='rmse', verbose=True, early_stopping_rounds=100)
ans = model.predict(x2_test)

save2 = pd.DataFrame({'UID': data_transaction_round1_origin['UID'], 'Tag': ans})
save2.to_csv('../data_temp/xgboost_transaction_predict_1.csv', index=False)


