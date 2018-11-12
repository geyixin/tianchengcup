#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'geyixin'

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = pd.read_excel('../data/exercise_table.xlsx')
print(data.head(10))

data = pd.get_dummies(data)

print(data.head(10))