# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:19:49 2019

@author: yn
"""

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

def missing_imputer(data, method='KNN', refine_thres=0.5):
    #  Remove features with too many missing values
    X = []
    for i in range(data.data.shape[1]):
        x = data.data[:,i]
        if len(np.where(np.isnan(x))[0]) > refine_thres*len(x):
            continue
        else:
            X.append(x)
    X = np.array(X).transpose()
    # Estimate the remaining missing values
    if method not in ['mean', 'median', 'most_frequent', 'half-minimum', 'zero', 'KNN', 'RandomForest', 'ExtraTrees', 'BayesianRidge']:
        raise IOError('invalid method')
    if method in ['mean', 'median', 'most_frequent']:
        imp = SimpleImputer(missing_values=np.nan, strategy=method)
        X = imp.fit_transform(X)
    elif method == 'zero':
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        X = imp.fit_transform(X)
    elif method in ['KNN', 'RandomForest', 'ExtraTrees', 'BayesianRidge']:
        estimators = {
                'BayesianRidge': BayesianRidge(),
                'RandomForest': RandomForestRegressor(n_estimators=50),
                'ExtraTrees': ExtraTreesRegressor(n_estimators=50),
                'KNN': KNeighborsRegressor(n_neighbors=min(15, X.shape[0]-1))
                }
        imp = IterativeImputer(estimator = estimators[method])
        X = imp.fit_transform(X)
    else:
        hms = 0.5 * np.nanmin(X, 0)
        for i in range(X.shape[1]):
            x = X[:, i]
            x[np.isnan(x)] = hms[i]
            X[:, i] = x
    data.X = X
    return data
        

def scaler(data, method='auto-scale'):
    X = data.data
    scl = {'center': preprocessing.StandardScaler(with_std=False),
           'auto-scale': preprocessing.StandardScaler(),
           'min-max': preprocessing.MinMaxScaler(),
           'max-abs': preprocessing.MaxAbsScaler(),
           'robust-scale': preprocessing.RobustScaler(),
           'quantile': preprocessing.QuantileTransformer(),
           'L2-normalize': preprocessing.Normalizer()
        }
    X = scl[method].fit_transform(X)
    data.data = X
    return data
    
