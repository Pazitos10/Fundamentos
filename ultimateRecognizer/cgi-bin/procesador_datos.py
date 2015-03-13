#!/usr/bin/python
# Import modules for CGI handling 
import os
from PIL import Image
from StringIO import StringIO
from base64 import decodestring
import cgi, cgitb 
import re
import numpy as np
width = 200
height = 200
# Create instance of FieldStorage 
form = cgi.FieldStorage() 


# Get data from fields
datos = form.getvalue('matriz_canvas')

img = Image(datos)
img = img.getdata()
img = img.resize((8,8))
img = map(list, img)
img = np.array(img)
s = img.shape[0] * img.shape[1]
img_wide = img.reshape(1, s)


#import ipdb
#ipdb.set_trace()

print "Content-type:text/html\r\n\r\n"
print '<html>'
print '<head>'
print '<title>Hello Word - First CGI Program</title>'
print '</head>'
print '<body>'
print 'tipo: %s - valor: %s' % (str(type(img_wide[0] )), len(img_wide[0]) )
print '</body></html>'