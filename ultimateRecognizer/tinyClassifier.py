#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX

from __future__ import division
from sklearn import preprocessing, svm, metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sio
import numpy as np

clf = KNeighborsClassifier(n_neighbors=13)
std_scaler = StandardScaler() 

class TinyClassifier(object):
    def __init__(self):
        super(TinyClassifier, self).__init__()
        self.expected = None
        self.predicted = None
        self.X = []

    def train(self, X, y): #entrenar
        global clf
        clf.fit(X,y)
    
    def predict(self, X): #predecir/probar/testear 
        global clf,std_scaler, predict_results 
        x = std_scaler.transform(X) 
        self.predicted = clf.predict(x)
        return self.predicted

    
    def splitData(self, X, y):
        global pca, std_scaler
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        X_train = std_scaler.fit_transform(X_train)
        X_test = std_scaler.transform(X_test) 
        return X_train, X_test, y_train, y_test

    def getGlobalAccuracy(self, y_test, y_predicted):
        global clf
        result = cross_validation.cross_val_score(clf,y_test, y_predicted, cv=5, scoring='accuracy')
        result = result.mean()
        result *= 100.0
        print 'accuracy score: %0.2f' % result
        return result

    def print_metrics(self, expected, predicted): #salida por consola
        global clf
        print("Classification report for classifier %s:\n%s\n"
              % (clf, metrics.classification_report(expected, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
