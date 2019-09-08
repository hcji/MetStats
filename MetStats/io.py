# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:00:42 2019

@author: yn
"""

import numpy as np
from pandas import read_csv
from sklearn.datasets.base import Bunch

# csv = 'Data/example_1.csv'
def load_csv(csv):
    content = read_csv(csv)
    feature_names = list(content.columns[2:])
    data = np.array(content.iloc[:,2:])
    filename = csv
    target_names = list(set(content.iloc[:,1]))
    target = np.array([target_names.index(t) for t in content.iloc[:,1]])
    sample_names = content.iloc[:,0]
    return Bunch(data=data, target=target, target_names=target_names, 
                 feature_names=feature_names, filename=filename, sample_names=sample_names)