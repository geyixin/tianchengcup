#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold


def tpr_weight_funtion(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer - 0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer - 0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer - 0.01).idxmin()]
    return 'TC_AUC', 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3, True


op_train = pd.read_csv('../data/operation_train_new.csv')
trans_train = pd.read_csv('../data/transaction_train_new.csv')

op_test = pd.read_csv('../data/operation_round1_new.csv')
trans_test = pd.read_csv('../data/transaction_round1_new.csv')
y = pd.read_csv('../data/tag_train_new.csv')
sub = pd.read_csv('../data/sub.csv')


def get_feature(op, trans, label):
    for feature in op.columns[2:]:
        label = label.merge(op.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
        label = label.merge(op.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')

    for feature in trans.columns[2:]:
        if trans_train[feature].dtype == 'object':
            label = label.merge(trans.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
            label = label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')
        else:
            label = label.merge(trans.groupby(['UID'])[feature].count().reset_index(), on='UID', how='left')
            label = label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(), on='UID', how='left')
            label = label.merge(trans.groupby(['UID'])[feature].max().reset_index(), on='UID', how='left')
            label = label.merge(trans.groupby(['UID'])[feature].min().reset_index(), on='UID', how='left')
            label = label.merge(trans.groupby(['UID'])[feature].sum().reset_index(), on='UID', how='left')
            label = label.merge(trans.groupby(['UID'])[feature].mean().reset_index(), on='UID', how='left')
            label = label.merge(trans.groupby(['UID'])[feature].std().reset_index(), on='UID', how='left')
    return label


train = get_feature(op_train, trans_train, y).fillna(-1)
test = get_feature(op_test, trans_test, sub).fillna(-1)

train = train.drop(['UID', 'Tag'], axis=1).fillna(-1)
label = y['Tag']

test_id = test['UID']
test = test.drop(['UID', 'Tag'], axis=1).fillna(-1)

lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, reg_alpha=3, reg_lambda=5, max_depth=-1,
                               n_estimators=5000, objective='binary', subsample=0.9, colsample_bytree=0.77,
                               subsample_freq=1, learning_rate=0.05,
                               random_state=1000, n_jobs=16, min_child_weight=4, min_child_samples=5, min_split_gain=0)
skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []

oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test_id.shape[0])

for index, (train_index, test_index) in enumerate(skf.split(train, label)):
    lgb_model.fit(train.iloc[train_index], label.iloc[train_index], verbose=50,
                  eval_set=[(train.iloc[train_index], label.iloc[train_index]),
                            (train.iloc[test_index], label.iloc[test_index])], early_stopping_rounds=30)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    oof_preds[test_index] = lgb_model.predict_proba(train.iloc[test_index], num_iteration=lgb_model.best_iteration_)[:,
                            1]

    test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
    sub_preds += test_pred / 5
    # print('test mean:', test_pred.mean())
    # predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred

m = tpr_weight_funtion(y_predict=oof_preds, y_true=label)
print(m[1])
sub = pd.read_csv('../data/sub.csv')
sub['Tag'] = sub_preds
sub.to_csv('../data_temp/baseline_%s.csv' % str(m), index=False)