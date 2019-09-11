# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 09:03:30 2019

@author: yn
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn import metrics
from tqdm import tqdm
from MetStats.utils import confidence_ellipse
colors = sns.color_palette("husl", 10)

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
        self.plsda = PLSRegression(n_components=ncomp)
        self.plsda.fit(self.X, self.dummy)
        
    def get_scores(self):
        return self.plsda.x_scores_
    
    def get_loadings(self):
        loadings = pd.DataFrame(self.plsda.x_loadings_)
        loadings.index = self.feature_names
        return loadings
    
    def get_vips(self):
        # from https://github.com/scikit-learn/scikit-learn/issues/7050
        t = self.plsda.x_scores_
        w = self.plsda.x_weights_
        q = self.plsda.y_loadings_
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
        order = np.argsort(-np.array(vips['VIP']))[range(topN)]
        x_axis = np.array(vips['feature'])[order]
        y_axis = vips['VIP'][order]
        plt.plot(x_axis, y_axis)
        plt.xlabel('features')
        plt.ylabel('variable importance')
        plt.figure()
    
    def scores_plot(self):
        X_r = self.plsda.x_scores_
        fig, ax = plt.subplots(figsize=(6, 6))
        for i, target_name in enumerate(self.target_names):
            ax.scatter(X_r[self.y==i, 0], X_r[self.y==i, 1], alpha=.8, lw=2, label=target_name, color=colors[i])
            confidence_ellipse(X_r[self.y==i, 0], X_r[self.y==i, 1], ax, edgecolor=colors[i], linestyle=':')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.figure()

    def LOO_test(self):
        loo = LeaveOneOut()
        y_hat = np.zeros(self.dummy.shape)
        for i, (train_index, test_index) in enumerate(tqdm(loo.split(self.X))):
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
        self.clf.fit(self.X, self.y)
            
    def get_OOB_error(self):
        oob_error = []
        for i in tqdm(range(1, self.n_estimators + 1)):
            clf = self.clf
            clf.set_params(n_estimators=i, oob_score=True)
            clf.fit(self.X, self.y)
            oob_error.append(1 - clf.oob_score_)
        plt.plot(np.array(range(1, self.n_estimators + 1)), np.array(oob_error))
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
        plt.legend(loc="upper right")
        plt.show()
        return pd.DataFrame({'estimators':list(range(1, self.n_estimators + 1)), 'oob_error': oob_error})
    
    def get_vips(self):
        vips = self.clf.feature_importances_
        return pd.DataFrame({'feature':self.feature_names, 'VIP': vips})
    
    def vips_plot(self, topN=10):
        vips = self.get_vips()
        order = np.argsort(-np.array(vips['VIP']))[range(topN)]
        x_axis = np.array(vips['feature'])[order]
        y_axis = vips['VIP'][order]
        plt.plot(x_axis, y_axis)
        plt.xlabel('features')
        plt.ylabel('variable importance')
        plt.figure()

    def LOO_test(self):
        loo = LeaveOneOut()
        y_hat = []
        for i, (train_index, test_index) in enumerate(tqdm(loo.split(self.X))):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            clf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth).fit(X_train, y_train)
            y_hat.append(clf.predict(X_test))
        pred = y_hat
        print('classification report: \n')
        print(metrics.classification_report(self.y, pred))
        confusion = pd.DataFrame(metrics.confusion_matrix(self.y, pred))
        confusion.index = self.target_names
        confusion.columns = self.target_names
        acc = metrics.accuracy_score(self.y, pred)
        return {'accuracy': acc, 'confusion': confusion}        
   
    
class SVM:
    def __init__(self, data, C=1.0, kernel='linear', degree=3):
        self.X = data.data
        self.y = data.target
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.clf = svm.SVC(C=C, kernel=kernel, degree=degree, gamma='scale')
        self.clf.fit(self.X, self.y)        
    
    def LOO_test(self):
        loo = LeaveOneOut()
        y_hat = []
        for i, (train_index, test_index) in enumerate(tqdm(loo.split(self.X))):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            clf = svm.SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma='scale').fit(X_train, y_train)
            y_hat.append(clf.predict(X_test))
        pred = y_hat
        print('classification report: \n')
        print(metrics.classification_report(self.y, pred))
        confusion = pd.DataFrame(metrics.confusion_matrix(self.y, pred))
        confusion.index = self.target_names
        confusion.columns = self.target_names
        acc = metrics.accuracy_score(self.y, pred)
        return {'accuracy': acc, 'confusion': confusion}
    
    def get_vips(self, max_iter=100, test_size=0.2):
        # variable importance based on mean decrease accuracy
        acc_prem = np.zeros((max_iter, self.X.shape[1]))
        for i in tqdm(range(self.X.shape[1])):
            acc_premi = []
            for j in range(max_iter):
                newX = self.X.copy()
                newX[:,i] = np.random.permutation(newX[:,i])
                X_train, X_test, y_train, y_test = train_test_split(newX, self.y, test_size=test_size)
                clf = svm.SVC(C=self.C, kernel=self.kernel, degree=self.degree, gamma='scale')
                clf.fit(X_train, y_train)
                y_hat = clf.predict(X_test)
                acc_premi.append(metrics.accuracy_score(y_test, y_hat))
            acc_prem[:,i] = acc_premi
        vips = 1 - acc_prem.mean(axis=0)
        return pd.DataFrame({'feature':self.feature_names, 'VIP': vips})

    def vips_plot(self, topN=10):
        vips = self.get_vips()
        order = np.argsort(-np.array(vips['VIP']))[range(topN)]
        x_axis = np.array(vips['feature'])[order]
        y_axis = vips['VIP'][order]
        plt.plot(x_axis, y_axis)
        plt.xlabel('features')
        plt.ylabel('variable importance')
        plt.figure()
        

class MLP:
    def __init__(self, data, hidden_layer_sizes=(100, 100, ), activation='relu', solver='adam', alpha=0.0001):
        self.X = data.data
        self.y = data.target
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
    
    def LOO_test(self):
        loo = LeaveOneOut()
        y_hat = []
        for i, (train_index, test_index) in enumerate(tqdm(loo.split(self.X))):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            clf = MLPClassifier(hidden_layer_sizes = self.hidden_layer_sizes, 
                                activation = self.activation, solver = self.solver,
                                alpha = self.alpha).fit(X_train, y_train)
            y_hat.append(clf.predict(X_test))
        pred = y_hat
        print('classification report: \n')
        print(metrics.classification_report(self.y, pred))
        confusion = pd.DataFrame(metrics.confusion_matrix(self.y, pred))
        confusion.index = self.target_names
        confusion.columns = self.target_names
        acc = metrics.accuracy_score(self.y, pred)
        return {'accuracy': acc, 'confusion': confusion}
    
    def get_vips(self, max_iter=100, test_size=0.2):
        # variable importance based on mean decrease accuracy
        acc_prem = np.zeros((max_iter, self.X.shape[1]))
        for i in tqdm(range(self.X.shape[1])):
            acc_premi = []
            for j in range(max_iter):
                newX = self.X.copy()
                newX[:,i] = np.random.permutation(newX[:,i])
                X_train, X_test, y_train, y_test = train_test_split(newX, self.y, test_size=test_size)
                clf = MLPClassifier(hidden_layer_sizes = self.hidden_layer_sizes, 
                                    activation = self.activation, solver = self.solver,
                                    alpha = self.alpha)
                clf.fit(X_train, y_train)
                y_hat = clf.predict(X_test)
                acc_premi.append(metrics.accuracy_score(y_test, y_hat))
            acc_prem[:,i] = acc_premi
        vips = 1 - acc_prem.mean(axis=0)
        return pd.DataFrame({'feature':self.feature_names, 'VIP': vips})

    def vips_plot(self, topN=10):
        vips = self.get_vips()
        order = np.argsort(-np.array(vips['VIP']))[range(topN)]
        x_axis = np.array(vips['feature'])[order]
        y_axis = vips['VIP'][order]
        plt.plot(x_axis, y_axis)
        plt.xlabel('features')
        plt.ylabel('variable importance')
        plt.figure()


class XGB:
    def __init__(self, data, max_depth=3, learning_rate=0.1, n_estimators=100, importance_type='gain'):
        self.X = data.data
        self.y = data.target
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.importance_type = importance_type
        self.clf = XGBClassifier(max_depth = self.max_depth, 
                                 learning_rate = self.learning_rate,
                                 n_estimators = self.n_estimators,
                                 importance_type = self.importance_type).fit(self.X, self.y) 
    
    def LOO_test(self):
        loo = LeaveOneOut()
        y_hat = []
        for i, (train_index, test_index) in enumerate(tqdm(loo.split(self.X))):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            clf = XGBClassifier(max_depth = self.max_depth, 
                                learning_rate = self.learning_rate,
                                n_estimators = self.n_estimators).fit(X_train, y_train)
            y_hat.append(clf.predict(X_test))
        pred = y_hat
        print('classification report: \n')
        print(metrics.classification_report(self.y, pred))
        confusion = pd.DataFrame(metrics.confusion_matrix(self.y, pred))
        confusion.index = self.target_names
        confusion.columns = self.target_names
        acc = metrics.accuracy_score(self.y, pred)
        return {'accuracy': acc, 'confusion': confusion}
    
    def get_vips(self, max_iter=100, test_size=0.2):
        vips = self.clf.feature_importances_
        return pd.DataFrame({'feature':self.feature_names, 'VIP': vips})

    def vips_plot(self, topN=10):
        vips = self.get_vips()
        order = np.argsort(-np.array(vips['VIP']))[range(topN)]
        x_axis = np.array(vips['feature'])[order]
        y_axis = vips['VIP'][order]
        plt.plot(x_axis, y_axis)
        plt.xlabel('features')
        plt.ylabel('variable importance')
        plt.figure()
    