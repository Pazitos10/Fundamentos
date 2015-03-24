#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX
from PIL import Image
from cStringIO import StringIO
import cgi, cgitb 
import urllib
from utils import remove_file
from tests import get_results
import os
import Cookie
import numpy as np
from pygame import surfarray


def main():
    THUMB_SIZE = (8,8)
    hay_results = 'false'
    form = cgi.FieldStorage() 
    datos = form.getvalue('matriz_canvas')

    if hay_datos():
        #obtenemos datos de imagen procesada
        path = '../statics/img/img.png'
        path_converted = '../statics/img/grey.png'
        path_bw = '../statics/img/bw.png'
        path_bw_s = '../statics/img/bw_s.png'

        remove_file(path) #habilitar esta linea si se hacen las comprobaciones siguientes
        remove_file(path_converted) #habilitar esta linea si se hacen las comprobaciones siguientes
        remove_file(path_bw)
        remove_file(path_bw_s)

        f = StringIO(urllib.urlopen(datos).read())
        img = Image.open(f)
        img.save(path)
        gray = img.convert('L')
        gray.save(path_converted)
        bw = gray.point(lambda x: 255 if x<50 else 0, '1')
        bw.save(path_bw)
        bw_s = bw.resize((8,8), Image.LANCZOS)
        bw_s.save(path_bw_s)
        new_im = np.asarray(bw_s, dtype=float)

        hay_results = predecir(new_im.flatten()) # llamamos a predecir respetando formato de los datos

        salida(hay_results)

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
    methods,predictions,accuracy = get_results(datos)
    tabla=''
    for method,pred,acc in zip(methods,predictions,accuracy):
        tabla+='<tr><td>%s</td><td>%s</td><td>%s</td></tr>' % (method,pred,acc)

    result_path = "../html/resultados_cargados.html"
    remove_file(result_path)
    archivo_resultados = open(result_path,"w+")
    archivo_resultados.write((open("../html/tabla_results.html").read()) % (tabla))
    archivo_resultados.close()
    return 'true'
    
def salida(hay_results):

    print "Content-type:text/html\r\n\r\n"
    print " "
    print "<meta http-equiv=\"Set-Cookie\" content=\"hay_results=%s; path=/; expires=null\" >" % hay_results
    print "<meta http-equiv=\"refresh\" content=\"0;url=../html/base.html\" >"


if __name__ == '__main__':
    main()