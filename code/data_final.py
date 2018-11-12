#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd

data1 = pd.read_csv('../data/operation_round1.csv')
data2 = pd.read_csv('../data/transaction_round1.csv')

ID1 = set(data1['UID'])
ID2 = set(data2['UID'])

print(len(ID1 | ID2))