#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd

trans = pd.read_csv('../data/transaction_train_new.csv', index_col='UID')
opera = pd.read_csv('../data/operation_train_new.csv', index_col='UID')

tag = pd.read_csv('../data/tag_train_new.csv', index_col='UID')

opera_with_tag = pd.concat([opera, pd.Series(tag['Tag'], index=opera.index)], axis=1)
trans_with_tag = pd.concat([trans, pd.Series(tag['Tag'], index=trans.index)], axis=1)

pd.DataFrame(opera_with_tag).to_csv('../data_temp/opera_with_tag.csv')
pd.DataFrame(trans_with_tag).to_csv('../data_temp/trans_with_tag.csv')