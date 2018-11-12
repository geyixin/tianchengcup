#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
from sklearn.naive_bayes import GaussianNB

train_dataSet = pd.read_csv('../data_temp/transaction_train_with_tag_new3.csv')
test_dataSet = pd.read_csv('../data_temp/transaction_round1_temp_new3.csv')

x_train = train_dataSet.drop(['UID', 'Tag'], axis=1)

y_train = train_dataSet.iloc[:,-1]

x_test = test_dataSet.drop(['UID'], axis=1)

clf = GaussianNB().fit(x_train, y_train)

predict1 = clf.predict(x_train)

save1 = pd.DataFrame({'UID': train_dataSet['UID'], 'Tag': predict1})
save1.to_csv('../data_temp/bayes_predict_1.csv', index=False)