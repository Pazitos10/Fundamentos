#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX


from tinyClassifier import TinyClassifier
from sklearn import datasets



def main():
    tiny_clf = TinyClassifier()
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X_train, X_test, y_train, y_test = tiny_clf.splitData(data, digits.target)

    dmc = tiny_clf.dummy_clf()
    svc = tiny_clf.support_vector_clf()
    knn = tiny_clf.k_neighbors_clf()
    etc = tiny_clf.extra_trees_clf()
    classifiers = [dmc,svc,knn,etc]

    map((lambda clf: tiny_clf.train(clf, X_train, y_train)), classifiers) #entrenan todos con los mismos datos

    expected = y_test
    for clf in classifiers:
        print "-"*20
        predicted = tiny_clf.predict(clf, X_test)
        tiny_clf.print_metrics(clf, expected, predicted)
        tiny_clf.getGlobalAccuracy(clf, data, digits.target)



if __name__ == '__main__':
    main()