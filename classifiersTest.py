#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX
import urllib, os, io, json, base64 
import numpy as np
from PIL import Image, ImageOps
from sklearn import datasets
from utils import remove_file
from classifiersManager import ClassifiersManager

path = 'static/img/img.png'
path_converted = 'static/img/grey.png'
path_bw = 'static/img/bw.png'
path_bw_s = 'static/img/bw_s.png'

def process_data(data, is_a_path=False):
    remove_file(path_bw_s)
    if not is_a_path: #Consulta string de datos de html
        remove_file(path)
        data = data.split(',')[1]
        f = io.BytesIO(base64.b64decode(data))
        im_a = image_as_array(f, guardar=True)
    else:
        im_a = image_as_array(path) # path es igual a data
    print(im_a)
    return get_results(im_a.flatten())

def image_as_array(f, guardar=False):
    img = Image.open(f)
    if guardar:
        img.save(path)
    gray = img.convert('L')
    gray.thumbnail((8,8), Image.LANCZOS)
    gray.save(path_bw_s)
    gray = ImageOps.invert(gray)
    gray.save(path_bw)
    #im_a = np.array(gray)
   
    im_a = np.asarray(gray, dtype=float)
    im_a = list(map((lambda x: abs(x//16) ), im_a))
    im_a = np.asarray(im_a)
    return im_a

def get_results(predict_data):
    clf_mngr = ClassifiersManager()
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    
    X_train, X_test, y_train, y_test = clf_mngr.splitData(data, digits.target)

    clf_mngr.train(X_train, y_train) #entrenan todos con los mismos datos
    predicted, probs = clf_mngr.predict(predict_data)
    
    save_probs(probs)    
    return predicted, probs

def save_probs(probs):
    remove_file("res.txt")
    f = open("res.txt","w")
    f.write(json.dumps(probs))
    f.close()