# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 09:03:30 2019

@author: yn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
from random import shuffle
from tqdm import tqdm

# from MetStats.io import load_csv
# data = load_csv('Data/example_1.csv')

class PLSDA:
    def __init__(self, data, ncomp):
        self.X = data.data
        self.y = data.target
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.ncomp = ncomp
        self.dummy = np.array(pd.get_dummies(data.target))
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
        plt.xlabel('features')
        plt.ylabel('variable importance')
        plt.figure()
    
    def scores_plot(self):
        X_r = self.res.x_scores_
        for i, target_name in enumerate(self.target_names):
            plt.scatter(X_r[self.y==i, 0], X_r[self.y==i, 1], alpha=.8, lw=2, label=target_name)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.figure()

    def LOO_test(self):
        loo = LeaveOneOut()
        y_hat = np.zeros(self.dummy.shape)
        for i, (train_index, test_index) in enumerate(loo.split(self.X)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.dummy[train_index], self.dummy[test_index]
            plsda = PLSRegression(n_components=self.ncomp).fit(X_train, y_train)
            y_hat[i,:] = plsda.predict(X_test)
        pred = np.array([np.argmax(r) for r in y_hat])
        print('classification report: \n')
        print(metrics.classification_report(self.y, pred))
        confusion = pd.DataFrame(metrics.confusion_matrix(self.y, pred))
        confusion.index = self.target_names
        confusion.columns = self.target_names
        acc = metrics.accuracy_score(self.y, pred)
        return {'accuracy': acc, 'confusion': confusion}
            
    def permutation_test(self, maxiter=100):
        corrs = []
        accs = []
        for j in tqdm(range(maxiter+1)):
            if j == 0:
                y_prem = self.y
            else:
                y_prem = np.random.permutation(self.y)
            dummy = np.array(pd.get_dummies(y_prem))
            corrs.append(metrics.accuracy_score(self.y, y_prem))
            loo = LeaveOneOut()
            y_hat = np.zeros(dummy.shape)
            for i, (train_index, test_index) in enumerate(loo.split(self.X)):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = dummy[train_index], dummy[test_index]
                plsda = PLSRegression(n_components=self.ncomp).fit(X_train, y_train)
                y_hat[i,:] = plsda.predict(X_test)      
            pred = np.array([np.argmax(r) for r in y_hat])
            accs.append(metrics.accuracy_score(y_prem, pred))
        plt.scatter(corrs, accs)
        plt.xlabel('correlation')
        plt.ylabel('accuracy score')
        plt.figure()        
        
        
class RandomForest:
    def __init__(self, data, n_estimators=500, max_depth=7):
        self.X = data.data
        self.y = data.target
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.res = self.clf.fit(self.X, self.y)
            
    def get_OOB_error(self):
        oob_error = []
        for i in range(1, self.n_estimators + 1):
            clf = self.clf
            clf.set_params(n_estimators=i)
            clf.fit(self.X, self.y)
            oob_error.append(1 - clf.oob_score_)
        plt.plot(np.array(range(1, self.n_estimators + 1)), np.array(oob_error))
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
        plt.legend(loc="upper right")
        plt.show()
            