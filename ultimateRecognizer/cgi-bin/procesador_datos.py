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
        f = StringIO(urllib.urlopen(datos).read())
        img = Image.open(f)
        gray = img.convert('L')
        gray = gray.resize((8,8), Image.LANCZOS)
        bw = gray.point(lambda x: 0 if x<210 else 255, '1')
        new_im = np.asarray(bw)

        hay_results = predecir(new_im.flatten()) # llamamos a predecir respetando formato de los datos

        salida(hay_results)
        #VERIFICACIONES
        path = '../statics/img/img.png'
        path_converted = '../statics/img/num_bw.png' 
        remove_file(path) #habilitar esta linea si se hacen las comprobaciones siguientes
        remove_file(path_converted) #habilitar esta linea si se hacen las comprobaciones siguientes
        img.save(path)
        gray.save(path_converted)

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