#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'


import pandas as pd

data_tag = pd.read_csv('../data_temp/merge_predict_new4.csv', index_col='UID')

tag_list = []

tag = data_tag['Tag']

for i in tag:
    if i <= 0.15:
        tag_list.append(0)
    elif i >= 0.9:
        tag_list.append(1)
    else:
        tag_list.append(i)

data_tag['Tag'] = tag_list

pd.DataFrame(data_tag).to_csv('../data_temp/change_1.csv')

