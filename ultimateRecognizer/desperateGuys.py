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

#clf = svm.SVC(kernel='sigmoid',gamma=0.019, probability=False)

clf = KNeighborsClassifier(n_neighbors=13)
std_scaler = StandardScaler() 

""" Usando sklean proveer a un clasificador svm.SVC los datos ubicados en los .mat
    si eso es posible, podemos mezclar la interfaz de pygame con esto quedando re copado
"""


class DesperateGuysClassifier(object):
    """docstring for DesperateGuysClassifier"""
    def __init__(self):
        super(DesperateGuysClassifier, self).__init__()
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
        #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=(test_proportion/100), random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        X_train = std_scaler.fit_transform(X_train)
        X_test = std_scaler.transform(X_test) 
        return X_train, X_test, y_train, y_test

    def getGlobalAccuracy(self, y_test, y_predicted):
        #calcula precision por cada muestra y luego calcula la precision global
        global clf
        result = metrics.accuracy_score(y_predicted,y_test)
        result *= 100.0
        print 'accuracy score: %0.2f' % result
        return result

    def print_metrics(self, expected, predicted): #salida por consola
        global clf
        print("Classification report for classifier %s:\n%s\n"
              % (clf, metrics.classification_report(expected, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
        
    def getExpected(self):
        return self.expected

    def getPredicted(self):
        return self.predicted

    def setPredicted(self, predicted):
        self.predicted = predicted

    def getX(self):
        return self.X


# #como usarlo: Cargar datos
# mat_contents = sio.loadmat('newX.mat')
# Xs = mat_contents['X']
# mat_contents = sio.loadmat('newy.mat')
# ys = mat_contents['y'].ravel()

# n_samples = len(Xs) #numero de muestras

# # Enseñarle con fit, previamente dividir los datos .. por ejemplo a la mitad
# classifier.fit(Xs[:n_samples / 2], ys[:n_samples / 2])

# # Testear con predict con el resto de los datos
# expected = ys[n_samples / 2:] #esperados 
# predicted = classifier.predict(Xs[n_samples / 2:]) #obtenidos



