#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX


from tinyClassifier import TinyClassifier
from sklearn import datasets
from matplotlib import pyplot as plt
import os
from utils import remove_file

methods = [ 'Dummy Classifier',
            'Support Vector Classifier',
            'KNeighbors Classifier',
            'Extra Trees Classifier']
predictions = []
accuracy = []
proba = []

def get_results(predict_data):
    tiny_clf, data , digits = init()
    X_train, X_test, y_train, y_test = tiny_clf.splitData(data, digits.target)
    classifiers = get_classifiers(tiny_clf)
    map((lambda clf: tiny_clf.train(clf, X_train, y_train)), classifiers) #entrenan todos con los mismos datos
    map((lambda clf: just_predict(tiny_clf,clf,predict_data)), classifiers)

    import json
    remove_file("res.txt")
    f = open("res.txt","w")
    f.write(json.dumps(proba))
    f.close()
    
    return methods,predictions,proba

def just_predict(tiny_clf,clf,data):
    predicted, probs = tiny_clf.predict(clf, data)
    predictions.append(predicted)
    clf_name = clf.__class__.__name__.title()
    proba.append(probs)


    
def predict_and_show_results(tiny_clf, clf, X,expected, data,target):
    predicted, probs = tiny_clf.predict(clf, X)
    #matrix = tiny_clf.save_metrics(clf, expected, predicted)
    acc = tiny_clf.getGlobalAccuracy(clf, data, target)
    predictions.append(predicted[0])
    accuracy.append(acc)
    clf_name = clf.__class__.__name__.title()
    #plot_confusion_matrix(matrix, acc, "Confusion matrix", clf_name)

def get_classifiers(tiny_clf):
    dmc = tiny_clf.dummy_clf()
    svc = tiny_clf.support_vector_clf()
    knn = tiny_clf.k_neighbors_clf()
    etc = tiny_clf.extra_trees_clf()
    return [dmc,svc,knn,etc]

def plot_confusion_matrix(matrix,acc, title, clf, method=None ):
    plt.imshow(matrix, cmap=plt.cm.binary, interpolation='nearest')
    plt.title(title+' - %s  Acc: %s %%' % (clf,str('%.2f' % acc)))
    plt.colorbar()
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.savefig(title+' %s.png' % clf,bbox_inches='tight')
    #plt.show() 
    plt.clf() 


def main():
    tiny_clf, data , digits = init()
    X_train, X_test, y_train, y_test = tiny_clf.splitData(data, digits.target)
    classifiers = get_classifiers(tiny_clf)
    map((lambda clf: tiny_clf.train(clf, X_train, y_train)), classifiers) #entrenan todos con los mismos datos   
    map((lambda clf: predict_and_show_results(  tiny_clf, 
                                                    clf, 
                                                    X_test, 
                                                    y_test, 
                                                    data, 
                                                    digits.target)), classifiers)
    print "Done!. . ."

def init():
    remove_file('results.txt')
    tiny_clf = TinyClassifier()
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return tiny_clf, data , digits


if __name__ == '__main__':
    main()