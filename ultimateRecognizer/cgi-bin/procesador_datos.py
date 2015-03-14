#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX
from PIL import Image
from cStringIO import StringIO
from base64 import decodestring
import cgi, cgitb 
import re
import numpy as np
import urllib
from utils import remove_file

STANDARD_SIZE = (380,380)

path = '/home/bruno/Dev/FTI/ultimateRecognizer/statics/img/img.png'
form = cgi.FieldStorage() 
datos = form.getvalue('matriz_canvas')
anterior = datos



f = StringIO(urllib.urlopen(datos).read())
img = Image.open(f)
remove_file(path)
img.save(path)
img = img.getdata()
img = img.resize((8,8), Image.ANTIALIAS)
img = map(list, img)
img = np.array(img)
s = img.shape[0] * img.shape[1]
img_wide = img.reshape(1, s)


print "Content-type:text/html\r\n\r\n"
print " "
print "<meta http-equiv=\"refresh\" content=\"0;url=../html/results.html\" >"

