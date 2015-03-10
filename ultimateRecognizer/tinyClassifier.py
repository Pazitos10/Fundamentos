#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX

from __future__ import division
from sklearn import metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier 

std_scaler = StandardScaler() 



class TinyClassifier(object):
    def __init__(self):
        super(TinyClassifier, self).__init__()
        # self.expected = None
        # self.X = []
        self.predicted = None

    def extra_trees_clf(self):
        return ExtraTreesClassifier( n_estimators=4,
                                    criterion='gini',
                                    max_features=0.2,
                                    n_jobs=4,
                                    random_state=1)

     
    def dummy_clf(self): 
        return DummyClassifier(strategy='most_frequent',random_state=0)
     
     
    def support_vector_clf(self):
        return SVC(kernel='linear')

    def k_neighbors_clf(self):
        return KNeighborsClassifier(n_neighbors=15)


    def train(self, clf, X, y): #entrenar
        clf.fit(X,y)
    
    def predict(self, clf, X): #predecir/probar/testear 
        self.predicted = clf.predict(X)
        return self.predicted
    
    def splitData(self, X, y):
#        global std_scaler
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#        X_train = std_scaler.fit_transform(X_train)
#        X_test = std_scaler.transform(X_test) 
        return X_train, X_test, y_train, y_test

    def getGlobalAccuracy(self, clf, y_test, y_predicted):
        result = cross_validation.cross_val_score(clf,y_test, y_predicted, cv=5, scoring='accuracy')
        result = result.mean()
        result *= 100.0
        print 'accuracy score: %0.2f' % result
        return result

    def print_metrics(self,clf, expected, predicted): #salida por consola
        print("Classification report for classifier %s:\n%s\n"
              % (clf, metrics.classification_report(expected, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
