#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd

data_tag = pd.read_csv('../data/tag_train_new.csv', index_col='UID')

# print(data_trans_type1)
# data_operation.info(memory_usage='deep')

'''
transaction数据处理
'''

'''
data_transaction_train = pd.read_csv('../data/transaction_train_new_2.csv', index_col='UID', encoding='ISO-8859-1')
data_transaction_round1 = pd.read_csv('../data/transaction_round1_new_2.csv', index_col='UID', encoding='ISO-8859-1')

# unique = data_transaction_round1['device_code3'].unique()
# print(len(unique))

round1_time = data_transaction_round1['time']
round1_time_list = []
for i in round1_time:
    round1_time_list.append(int(i.split(':')[0]))
data_transaction_round1['time'] = round1_time_list

train_time = data_transaction_train['time']
train_time_list = []
for i in train_time:
    train_time_list.append(int(i.split(':')[0]))
data_transaction_train['time'] = train_time_list

data_transaction_train['device_code'] = data_transaction_train['device_code3'].fillna(0)
train_device_code = data_transaction_train['device_code']
train_device_code_list = []
for i in train_device_code:
    if i == 0:
        train_device_code_list.append(0)
    else:
        train_device_code_list.append(1)
data_transaction_train['Is_apple'] = train_device_code_list

data_transaction_round1['device_code'] = data_transaction_round1['device_code3'].fillna(0)
round1_device_code = data_transaction_round1['device_code']
round1_device_code_list = []
for i in round1_device_code:
    if i == 0:
        round1_device_code_list.append(0)
    else:
        round1_device_code_list.append(1)
data_transaction_round1['Is_apple'] = round1_device_code_list

# print(data_transaction_round1['device_code'].head(30))

data_transaction_round1['market_type'] = data_transaction_round1['market_type'].fillna(0)
data_transaction_train['market_type'] = data_transaction_train['market_type'].fillna(0)

data_transaction_train['trans_type2'] = data_transaction_train['trans_type2'].fillna(0)
data_transaction_round1['trans_type2'] = data_transaction_round1['trans_type2'].fillna(0)

with open('../data_temp/amt_src1_new_transaction.txt', 'r') as f:
    data_amt_src1 = f.read()
with open('../data_temp/trans_type1_new_transaction.txt', 'r') as f:
    data_trans_type1 = f.read()
with open('../data_temp/geo1_new_transaction.txt', 'r') as f:
    geo1 = f.read()
with open('../data_temp/geo2_new_transaction.txt', 'r') as f:
    geo2 = f.read()

data_amt_src1 = data_amt_src1.split(' ')
data_trans_type1 = data_trans_type1.split(' ')
geo1 = geo1.split(' ')
geo2 = geo2.split(' ')

data_transaction_train['geo_code'] = data_transaction_train['geo_code'].fillna(0)
train_geo = data_transaction_train['geo_code']
train_geo_list = []
for i in train_geo:
    if i == 0:
        train_geo_list.append(0)
    else:
        train_geo_list.append(1)
data_transaction_train['Is_geo'] = train_geo_list

data_transaction_round1['geo_code'] = data_transaction_round1['geo_code'].fillna(0)
round1_geo = data_transaction_round1['geo_code']
round1_geo_list = []
for i in round1_geo:
    if i == 0:
        round1_geo_list.append(0)
    else:
        round1_geo_list.append(1)
data_transaction_round1['Is_geo'] = round1_geo_list

for i in range(len(data_amt_src1)):
    data_transaction_train.amt_src1[data_transaction_train['amt_src1'] == data_amt_src1[i]] = i+1
    data_transaction_round1.amt_src1[data_transaction_round1['amt_src1'] == data_amt_src1[i]] = i + 1

for j in range(len(data_trans_type1)):
    data_transaction_train.trans_type1[data_transaction_train['trans_type1'] == data_trans_type1[j]] = j+1
    data_transaction_round1.trans_type1[data_transaction_round1['trans_type1'] == data_trans_type1[j]] = j + 1

for i in range(len(geo1)):
    data_transaction_train.geo1[data_transaction_train['geo1'] == geo1[i]] = i+1
    data_transaction_round1.geo1[data_transaction_round1['geo1'] == geo1[i]] = i + 1

for j in range(len(geo2)):
    data_transaction_train.geo2[data_transaction_train['geo2'] == geo2[j]] = j+1
    data_transaction_round1.geo2[data_transaction_round1['geo2'] == geo2[j]] = j + 1

data_transaction_train_with_tag = \
    pd.concat([data_transaction_train[['channel', 'day', 'trans_amt', 'amt_src1', 'trans_type1', 'trans_type2', 'bal',
                                       'time', 'Is_apple', 'market_type', 'Is_geo', 'geo1', 'geo2']],
               pd.Series(data_tag['Tag'], index=data_transaction_train.index)], axis=1)
data_transaction_train_with_tag.columns = ['channel', 'day', 'trans_amt', 'amt_src1', 'trans_type1', 'trans_type2',
                                           'bal', 'time', 'Is_apple', 'market_type', 'Is_geo', 'geo1', 'geo2'] + ['Tag']

data_transaction_round1 = data_transaction_round1[['channel', 'day', 'trans_amt', 'amt_src1', 'trans_type1',
                                                   'trans_type2', 'bal', 'time', 'Is_apple', 'market_type',
                                                   'Is_geo', 'geo1', 'geo2']]

# print(data_transaction_round1.head(1))
# print(data_transaction_train_with_tag.head(1))

pd.DataFrame(data_transaction_train_with_tag).to_csv('../data_temp/transaction_train_with_tag_new_10_10.csv')
pd.DataFrame(data_transaction_round1).to_csv('../data_temp/transaction_round1_temp_new_10_10.csv')
'''

'''
operation数据处理
'''


data_operation_train = pd.read_csv('../data/operation_train_new_2.csv', index_col='UID', encoding='ISO-8859-1')
data_operation_round1 = pd.read_csv('../data/operation_round1_new_2.csv', index_col='UID', encoding='ISO-8859-1')

# unique = data_operation_train['device1'].unique()
# print(unique)

data_operation_train['success'] = data_operation_train['success'].fillna(0)

with open('../data_temp/mode_new_operation.txt', 'r') as f:
    data_mode = f.read()
with open('../data_temp/geo1_new_operation.txt', 'r') as f:
    geo1 = f.read()
with open('../data_temp/geo2_new_operation.txt', 'r') as f:
    geo2 = f.read()

data_mode = data_mode.split(' ')
geo1 = geo1.split(' ')
geo2 = geo2.split(' ')

data_operation_train['success'] = data_operation_train['success'].fillna(0)
data_operation_round1['success'] = data_operation_round1['success'].fillna(0)

round1_time = data_operation_round1['time']
round1_time_list = []
for i in round1_time:
    round1_time_list.append(int(i.split(':')[0]))
data_operation_round1['time'] = round1_time_list

train_time = data_operation_train['time']
train_time_list = []
for i in train_time:
    train_time_list.append(int(i.split(':')[0]))
data_operation_train['time'] = train_time_list

data_operation_train['version'] = data_operation_train['version'].fillna(0)
train_version = data_operation_train['version']
train_version_list = []
for i in train_version:
    if i == 0:
        train_version_list.append(0)
    else:
        train_version_list.append(int(i.split('.')[0]))
data_operation_train['version'] = train_version_list

data_operation_round1['version'] = data_operation_round1['version'].fillna(0)
round1_version = data_operation_round1['version']
round1_version_list = []
for i in round1_version:
    if i == 0:
        round1_version_list.append(0)
    else:
        round1_version_list.append(int(i.split('.')[0]))
data_operation_round1['version'] = round1_version_list

data_operation_train['geo_code'] = data_operation_train['geo_code'].fillna(0)
train_geo = data_operation_train['geo_code']
train_geo_list = []
for i in train_geo:
    if i == 0:
        train_geo_list.append(0)
    else:
        train_geo_list.append(1)
data_operation_train['Is_geo'] = train_geo_list

data_operation_round1['geo_code'] = data_operation_round1['geo_code'].fillna(0)
round1_geo = data_operation_round1['geo_code']
round1_geo_list = []
for i in round1_geo:
    if i == 0:
        round1_geo_list.append(0)
    else:
        round1_geo_list.append(1)
data_operation_round1['Is_geo'] = round1_geo_list

data_operation_train['device_code'] = data_operation_train['device_code3'].fillna(0)
train_device_code = data_operation_train['device_code']
train_device_code_list = []
for i in train_device_code:
    if i == 0:
        train_device_code_list.append(0)
    else:
        train_device_code_list.append(1)
data_operation_train['Is_apple'] = train_device_code_list

data_operation_round1['device_code'] = data_operation_round1['device_code3'].fillna(0)
round1_device_code = data_operation_round1['device_code']
round1_device_code_list = []
for i in round1_device_code:
    if i == 0:
        round1_device_code_list.append(0)
    else:
        round1_device_code_list.append(1)
data_operation_round1['Is_apple'] = round1_device_code_list

for i in range(len(data_mode)):
    # print(data_mode[i])
    data_operation_train['mode'][data_operation_train['mode'] == data_mode[i]] = i + 1
    data_operation_round1['mode'][data_operation_round1['mode'] == data_mode[i]] = i + 1

for i in range(len(geo1)):
    data_operation_train.geo1[data_operation_train['geo1'] == geo1[i]] = i+1
    data_operation_round1.geo1[data_operation_round1['geo1'] == geo1[i]] = i + 1

for j in range(len(geo2)):
    data_operation_train.geo2[data_operation_train['geo2'] == geo2[j]] = j+1
    data_operation_round1.geo2[data_operation_round1['geo2'] == geo2[j]] = j + 1

data_operation_train_with_tag = \
    pd.concat([data_operation_train[['day', 'mode', 'success', 'os', 'time', 'version', 'Is_apple', 'Is_geo',
                                     'geo1', 'geo2']],
               pd.Series(data_tag['Tag'], index=data_operation_train.index)], axis=1)
data_operation_train_with_tag.columns = ['day', 'mode', 'success', 'os', 'time', 'version', 'Is_apple',
                                         'Is_geo', 'geo1', 'geo2'] + ['Tag']

data_operation_round1 = data_operation_round1[['day', 'mode', 'success', 'os', 'time', 'version',
                                               'Is_apple', 'Is_geo', 'geo1', 'geo2']]

# print(data_operation_round1.head(1))
# print(data_operation_train_with_tag.head(1))

pd.DataFrame(data_operation_train_with_tag).to_csv('../data_temp/operation_train_with_tag_new_10_10.csv')
pd.DataFrame(data_operation_round1).to_csv('../data_temp/operation_round1_temp_10_10.csv')



'''
下面这两个不一样
'''

# print(type(data_transaction_round1[['device1']]))   # <class 'pandas.core.frame.DataFrame'>
# print(data_transaction_round1[['device1']].head(1))
# '''
#       device1
# UID
# 31117     NaN
# '''
# print(type(data_transaction_round1['device1']))     # <class 'pandas.core.series.Series'>
# print(data_transaction_round1['device1'].head(1))
# '''
# UID
# 31117    NaN
# Name: device1, dtype: object
