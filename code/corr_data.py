#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd

data_transaction = pd.read_csv('../data_temp/transaction_train_with_tag_new_10_10.csv', index_col='UID')
cor = data_transaction.corr()

path_out = '../data_temp/' + 'data_transaction_corr' + '.csv'

cor.to_csv(path_out)