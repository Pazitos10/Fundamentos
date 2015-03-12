#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX

from __future__ import division
from sklearn import metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
import numpy as np




class TinyClassifier(object):
    def __init__(self):
        super(TinyClassifier, self).__init__() 

    def extra_trees_clf(self):
        #Defaults => n_estimators=4, criterion='gini',max_features=0.2,n_jobs=4,random_state=1
        return ExtraTreesClassifier(max_features='auto',
                                    n_jobs=-1,
                                    random_state=1)

    def dummy_clf(self): 
        return DummyClassifier(strategy='most_frequent',random_state=np.random.randint(0,9))
          
    def support_vector_clf(self):
        #default kernel= 'linear'
        return SVC(kernel='poly')

    def k_neighbors_clf(self):
        return KNeighborsClassifier(n_neighbors=9, weights='distance')

    def train(self, clf, X, y): #entrenar
        clf.fit(X,y)
    
    def predict(self, clf, X): #predecir/probar/testear 
        return clf.predict(X)
    
    def splitData(self, X, y):
        return train_test_split(X, y, test_size=0.2)

    def getGlobalAccuracy(self, clf, y_test, y_predicted):
        scores = cross_validation.cross_val_score(clf,y_test, y_predicted, cv=5, scoring='accuracy')
        result = scores.mean()
        std_desv = scores.std()
        result *= 100.0
        saved_results = open('results.txt', 'a+')
        #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        line = '\n\n %s \t| Accuracy: %0.2f %% (+/- %0.2f)\n' % (clf.__class__.__name__,result,std_desv)
        line += '-'*100+'\n'
        saved_results.write(line)
        saved_results.close()
        return result

    def save_metrics(self,clf, expected, predicted): #salida por archivo
        results = open('results.txt', 'a+')
        matrix = metrics.confusion_matrix(expected, predicted)
        report = metrics.classification_report(expected, predicted)
        line = "Classification report for classifier %s:\n%s\n" % (clf,report)
        line += "Confusion matrix:\n%s" % matrix
        results.write(line)
        results.close()
        return matrix
