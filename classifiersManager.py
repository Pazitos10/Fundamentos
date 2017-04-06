#!/usr/bin/python
# -*- coding: utf-8 -*-
# Authors: pazitos10, SinX
# Description: 
#  This file holds the Classifiers Manager Model. 
#  This class, basically handles multiple classifiers instances and
#  wrap some of its behaviours to be generic.
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import remove_file


class ClassifiersManager(object):
    def __init__(self):
        super(ClassifiersManager, self).__init__()
        self.classifiers = [
            self.dummy_clf(),
            self.support_vector_clf(),
            self.extra_trees_clf(),
            self.k_neighbors_clf()
        ]
        self.pca = PCA(svd_solver='randomized', n_components=20)
        self.std_scaler = StandardScaler()

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

    def train(self, X, y): #entrenar
        for clf in self.classifiers:
            clf.fit(X, y)

    def predict(self, X): #predecir
        predicted = {}
        probs = []
        for clf in self.classifiers:
            x = self.pca.transform([X])
            x = self.std_scaler.transform(x)
            
            pred = clf.predict(x)[0]
            pred_proba = clf.predict_proba(x)[0]
            
            #pred = clf.predict(X.reshape(1, -1))[0]
            #pred_proba = clf.predict_proba(X.reshape(1,-1))[0]
            print('clf name: '+clf.__class__.__name__)
            print('pred: ', pred)
            print('pred_proba: ', pred_proba)
            print('*'*20)

            pred_proba = [float("{0:.2f}".format(prob*100)) for prob in pred_proba]
            predicted.update({clf.__class__.__name__: pred})              
            probs.append(pred_proba)
        return predicted, probs

    def splitData(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        X_train = self.pca.fit_transform(X_train)
        X_test = self.pca.transform(X_test)
        X_train = self.std_scaler.fit_transform(X_train)
        X_test = self.std_scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def getGlobalAccuracy(self, clf, y_test, y_predicted):
        scores = cross_val_score(clf,y_test, y_predicted, cv=5, scoring='accuracy')
        result = scores.mean()
        std_desv = scores.std()
        result *= 100.0
        remove_file('results.txt')
        saved_results = open('results.txt', 'a+')
        line = '\n\n %s \t| Accuracy: %0.2f %% (+/- %0.2f)\n' % (clf.__class__.__name__, result, std_desv)
        line += '-'*100+'\n'
        saved_results.write(line)
        saved_results.close()
        return result

    def save_metrics(self,clf, expected, predicted): #salida por archivo
        remove_file('metrics.txt')
        results = open('metrics.txt', 'a+')
        matrix = metrics.confusion_matrix(expected, predicted)
        report = metrics.classification_report(expected, predicted)
        line = "Classification report for classifier %s:\n%s\n" % (clf,report)
        line += "Confusion matrix:\n%s" % matrix
        results.write(line)
        results.close()
        return matrix