#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX
from PIL import Image, ImageOps
from cStringIO import StringIO
import cgi, cgitb 
import urllib
from utils import remove_file
from tests import get_results
import os
import Cookie
import numpy as np

path = '../statics/img/img.png'
path_converted = '../statics/img/grey.png'
path_bw = '../statics/img/bw.png'
path_bw_s = '../statics/img/bw_s.png'


def main():
    global path

    THUMB_SIZE = (8,8)
    hay_results = 'false'
    form = cgi.FieldStorage() 
    datos = form.getvalue('matriz_canvas')

    
    if hay_datos():
        remove_file(path_bw_s)
        remove_file(path_bw)
        if bool(datos):
            remove_file(path) #habilitar esta linea si se hacen las comprobaciones siguientes
            #remove_file(path_converted) #habilitar esta linea si se hacen las comprobaciones siguientes
            #remove_file(path_bw)
            f = StringIO(urllib.urlopen(datos).read())
            im_a = acciones_comunes(f, guardar=True)
        else:
            f = path
            im_a = acciones_comunes(f, guardar=False)

#        im_a = map((lambda x: (x//16)+1 ), im_a)
        #new_im = np.asarray(im_a)

        hay_results = predecir(im_a.flatten()) # llamamos a predecir respetando formato de los datos


        salida(hay_results)

def acciones_comunes(f, guardar=False):
    global path, path_bw_s
    img = Image.open(f)
    if guardar:
        img.save(path)
    gray = img.convert('L')
    gray.thumbnail((8,8), Image.LANCZOS)
    gray.save(path_bw_s)
    gray = ImageOps.invert(gray)
    gray.save(path_bw)
    im_a = np.array(gray)
    return im_a


def hay_datos():
    cookie_string=os.environ.get('HTTP_COOKIE')
    c=Cookie.SimpleCookie()
    c.load(cookie_string)
    if c['hay_datos']: #el valor de la cookie no es relevante, solo importa la existencia
        result = True
    else:
        result = False
    return result

def predecir(datos):
    import operator
    methods,predictions,probs = get_results(datos)
    tabla1=''
    tabla2=''
    proba = [v for v in probs]

    for method,pred in zip(methods,predictions):
        tabla1+='<tr><td>%s</td><td>%s</td></tr>' % (method,pred)

    tabla2+='<tr>'
    for method in methods:
        tabla2+='<td><strong> %s %% </strong></td>' % method
    tabla2+='</tr>'

    for value in proba:
        tabla2 +='<td>'
        max_val = max(value, key=value.get)
        for label, prob in value.iteritems():
            if label == max_val:
                tabla2+='<strong style="color:green">%s: %s </strong><br>'  % (label, prob)
            else:
                tabla2+='%s: %s <br>'  % (label, prob)
        tabla2 +='</td>'

    result_path = "../html/resultados_cargados.html"
    remove_file(result_path)
    archivo_resultados = open(result_path,"w+")
    archivo_resultados.write((open("../html/resultados.html").read()) % (tabla1,tabla2))
    archivo_resultados.close()
    return 'true'
    
def salida(hay_results):

    print "Content-type:text/html\r\n\r\n"
    print " "
    print "<meta http-equiv=\"Set-Cookie\" content=\"hay_results=%s; path=/; expires=null\" >" % hay_results
    print "<meta http-equiv=\"refresh\" content=\"0;url=../html/base.html\" >"


if __name__ == '__main__':
    main()