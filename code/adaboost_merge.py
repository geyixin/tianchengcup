#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from numpy import array
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

data_transaction_predict = pd.read_csv('../data_temp/transaction_round1_predict_new4.csv')
data_operation_predict = pd.read_csv('../data_temp/operation_round1_predict_new4.csv')

x2_train = data_transaction_predict.drop(['Tag'], axis=1)
y2_train = data_transaction_predict.drop(['UID'], axis=1)

clf1 = GaussianNB()
clf2 = DecisionTreeClassifier(random_state=1)
clf3 = RandomForestClassifier()

clf = VotingClassifier(estimators=[('gnb', clf1), ('dt', clf2), ('rf', clf3)], voting='hard')

clf.fit(x2_train, y2_train)
predict = clf.predict(x2_train)


