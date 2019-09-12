# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:26:41 2019

@author: hcji
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc
colors = sns.color_palette("husl", 10)

# from MetStats.io import load_csv
# data = load_csv('Data/example_1.csv')

class Univeriate:
    def __init__(self, data, equal_var=True):
        self.X = data.data
        self.y = data.target
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        
    def t_test(self):
        X = self.X
        y = self.y
        nclass = len(list(set(y)))
        if nclass > 1:
            res = {}
            for i in range(nclass):
                grp1 = X[y==i,:]
                grp2 = X[y!=i,:]
                pvals = []
                for j in range(X.shape[1]):
                    sta, pval = stats.ttest_ind(grp1[:,j], grp2[:,j])
                    pvals.append(pval)
                resi = pd.DataFrame({'feature_names':self.feature_names, 'p_values':pvals})
                res[self.target_names[i]] = resi
        else:
            for i in range(nclass):
                pvals = []
                for j in range(X.shape[1]):
                    sta, pval = stats.ttest_1samp(X[:,j])
                    pvals.append(pval)
                res = pd.DataFrame({'feature_names':self.feature_names, 'p_values':pvals})            
        return res
    
    def box_plot(self, feature = 0):
        X = self.X
        y = self.y
        if type(feature) is not int:
            feature = self.feature_names[feature].index(feature)
        nclass = len(list(set(y)))
        values = {}
        for i in range(nclass):
            values[self.target_names[i]] = X[y==i, feature]
        boxplot = plt.boxplot(values.values(), labels=values.keys(), patch_artist=True)
        for j, patch in enumerate(boxplot['boxes']):
            patch.set_color(colors[j]) 
        plt.show()
        
    def roc_analysis(self):
        X = self.X
        y = self.y
        nclass = len(list(set(y)))
        if nclass > 1:
            res = {}
            for i in range(nclass):
                grp1 = X[y==i,:]
                grp2 = X[y!=i,:]
                aucs = []
                for j in range(X.shape[1]):
                    tpr, fdr, _ = roc_curve(grp1[:,j], grp2[:,j])
                    aucs.append(auc(tpr, fdr))
                resi = pd.DataFrame({'feature_names':self.feature_names, 'aucs':aucs})
                res[self.target_names[i]] = resi
        else:
            raise IOError('y must contain two classes or more')
        return res
    
    def roc_plot(self, feature = 0):
        # unfinished
    