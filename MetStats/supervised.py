# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 09:03:30 2019

@author: yn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

# from MetStats.io import load_csv
# data = load_csv('Data/example_1.csv')

class PLSDA:
    def __init__(self, data, ncomp):
        self.X = data.data
        self.y = data.target
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.ncomp = ncomp
        self.dummy = pd.get_dummies(data.target)
        plsda = PLSRegression(n_components=ncomp)
        self.res = plsda.fit(self.X, self.dummy)
        
    def get_scores(self):
        return self.res.x_scores_
    
    def get_loadings(self):
        loadings = pd.DataFrame(self.res.x_loadings_)
        loadings.index = self.feature_names
        return loadings
    
    def get_vips(self):
        # from https://github.com/scikit-learn/scikit-learn/issues/7050
        t = self.res.x_scores_
        w = self.res.x_weights_
        q = self.res.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
            vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
        return pd.DataFrame({'feature':self.feature_names, 'VIP': vips})
    
    def vips_plot(self, topN=10):
        vips = self.get_vips()
        order = np.argsort(-vips)[range(topN)]
        x_axis = np.array(self.feature_names)[order]
        y_axis = vips[order]
        plt.plot(x_axis, y_axis)
        plt.figure()
    
    def scores_plot(self):
        X_r = self.res.x_scores_
        for i, target_name in enumerate(self.target_names):
            plt.scatter(X_r[self.y==i, 0], X_r[self.y==i, 1], alpha=.8, lw=2, label=target_name)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.figure()

        