# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:16:05 2019

@author: yn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import decomposition
from sklearn import manifold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from MetStats.utils import confidence_ellipse
colors = sns.color_palette("husl", 10)

# from MetStats.io import load_csv
# data = load_csv('Data/example_1.csv')

class PCA:
    def __init__(self, data, ncomp):
        self.X = data.data
        self.y = data.target
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.ncomp = ncomp
        pca = decomposition.PCA(n_components=self.ncomp)
        self.res = pca.fit(self.X)
    
    def get_scores(self):
        X_r = self.res.transform(self.X)
        return X_r
    
    def get_loadings(self):
        loadings = pd.DataFrame(self.res.components_.T * np.sqrt(self.res.explained_variance_))
        loadings.index = self.feature_names
        return loadings
    
    def scores_plot(self, pc1=0, pc2=1):
        X_r = self.res.transform(self.X)
        Exr = [e/sum(self.res.explained_variance_) for e in self.res.explained_variance_]
        fig, ax = plt.subplots(figsize=(6, 6))
        for i, target_name in enumerate(self.target_names):
            ax.scatter(X_r[self.y==i, 0], X_r[self.y==i, 1], alpha=.8, lw=2, label=target_name, color=colors[i])
            confidence_ellipse(X_r[self.y==i, 0], X_r[self.y==i, 1], ax, edgecolor=colors[i], linestyle=':')           
        plt.xlabel('PC ' + str(pc1) + ' (' + str(round(Exr[pc1] * 100, 2)) + ' %)')
        plt.ylabel('PC ' + str(pc2) + ' (' + str(round(Exr[pc2] * 100, 2)) + ' %)')
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.figure()
        
    def explanation_plot(self):
        Exr = [e/sum(self.res.explained_variance_) for e in self.res.explained_variance_]
        Sxr = np.cumsum(Exr)
        plt.plot(np.arange(1, len(Exr)+1), Exr, label='explanined rate')
        plt.plot(np.arange(1, len(Exr)+1), Sxr, label='total_explanined_rate')
        plt.xlabel('Number of PC')
        plt.ylabel('Rate')
        plt.figure()
        
    def parameters(self):
        print('component number is {}'.format(self.ncomp))
        

class TSNE:
    def __init__(self, data, ncomp):
        self.X = data.data
        self.y = data.target
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.ncomp = ncomp
        self.tsne = manifold.TSNE(n_components=self.ncomp)
    
    def get_scores(self):
        X_r = self.tsne.fit_transform(self.X)
        return X_r
    
    def scores_plot(self, pc1=0, pc2=1):
        X_r = self.tsne.fit_transform(self.X)
        for i, target_name in enumerate(self.target_names):
            plt.scatter(X_r[self.y==i, pc1], X_r[self.y==i, pc2], alpha=.8, lw=2, label=target_name)
        plt.xlabel('PC ' + str(pc1))
        plt.ylabel('PC ' + str(pc2))
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.figure()        

    def parameters(self):
        print('component number is {}'.format(self.ncomp))
        