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
from utils import remove_file


class TinyClassifier(object):
    def __init__(self):
        super(TinyClassifier, self).__init__()

    def extra_trees_clf(self):
        return ExtraTreesClassifier(max_features='auto',
                                    n_jobs=-1,
                                    random_state=1)

    def dummy_clf(self):
        return DummyClassifier( strategy='most_frequent',
                                random_state=np.random.randint(0,9))

    def support_vector_clf(self):
        return SVC(gamma=0.001, probability=True)

    def k_neighbors_clf(self):
        return KNeighborsClassifier(n_neighbors=9, algorithm='auto')

    def train(self, clf, X, y): #entrenar
        clf.fit(X,y)

    def predict(self, clf, X): #predecir
        predicted = clf.predict(X)[0]
        probs = {   i : (float("{0:.2f}".format(prob*100))) \
                    for i, prob in enumerate(clf.predict_proba(X)[0])}
        return predicted, probs

    def splitData(self, X, y):
        return train_test_split(X, y, test_size=0.1)

    def getGlobalAccuracy(self, clf, y_test, y_predicted):
        scores = cross_validation.cross_val_score(clf,y_test, y_predicted, cv=5, scoring='accuracy')
        result = scores.mean()
        std_desv = scores.std()
        result *= 100.0
        remove_file('results.txt')
        saved_results = open('results.txt', 'a+')
        line = '\n\n %s \t| Accuracy: %0.2f %% (+/- %0.2f)\n' % (clf.__class__.__name__,result,std_desv)
        line += '-'*100+'\n'
        saved_results.write(line)
        saved_results.close()
        return result

    def save_metrics(self,clf, expected, predicted): #salida por archivo
        remove_file('results.txt')
        results = open('results.txt', 'a+')
        matrix = metrics.confusion_matrix(expected, predicted)
        report = metrics.classification_report(expected, predicted)
        line = "Classification report for classifier %s:\n%s\n" % (clf,report)
        line += "Confusion matrix:\n%s" % matrix
        results.write(line)
        results.close()
        return matrix
