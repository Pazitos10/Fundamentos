#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX
import urllib, os, io, json, base64 
import numpy as np
from PIL import Image, ImageOps
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from utils import remove_file, gray2bw
from classifiersManager import ClassifiersManager
from time import sleep

path = 'static/img/img.png'
path_inverted = 'static/img/inverted.png'

clf_mngr = ClassifiersManager()
min_max_scaler = MinMaxScaler((0, 16))
rescaler = MinMaxScaler((0, 255))

def process_data(data):
    if data != '' and data != path:
        remove_file(path_inverted)
        remove_file(path)
        data = data.split(',')[1]
        f = io.BytesIO(base64.b64decode(data))
        img = Image.open(f)
        img.save(path)
    else:
        img = Image.open(path)

    im_a = get_img_arr(img)
    return get_results(im_a.flatten())

def get_img_arr(img):
    #print(np.asarray(filename))
    #number = Image.open(filename)
    number = img.convert('L')
    number = ImageOps.invert(number)
    number.thumbnail((8,8), Image.LANCZOS)
    number = np.asarray(number, dtype=np.float64).reshape(1, -1)
    min_max_scaler.fit(number.reshape(-1, 1))
    number = min_max_scaler.transform(number.reshape(1, -1))
    rescale_and_save(np.copy(number))
    return np.trunc(number)

def rescale_and_save(number_array):
    rescaler.fit(number_array.reshape(-1, 1))
    rescaled_number = rescaler.transform(number_array.reshape(1, -1))
    inverted = Image.fromarray(np.uint8(rescaled_number.reshape(8,8)))
    inverted = inverted.resize((150,150))
    inverted.save(path_inverted)

def get_results(predict_data):
    global clf_mngr
    digits = datasets.load_digits()
    n_samples = len(digits.images)

    #binarized = list(map(lambda digit: gray2bw(digit), digits.images))

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