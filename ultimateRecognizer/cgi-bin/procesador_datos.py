#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX
from PIL import Image
from cStringIO import StringIO
import cgi, cgitb 
import urllib
from utils import remove_file
from tests import methods,predictions,accuracy,main
import Cookie


def main():
    THUMB_SIZE = 8, 8
    hay_results = 'false'
    form = cgi.FieldStorage() 
    datos = form.getvalue('matriz_canvas')

    if hay_datos():
        #obtenemos datos de imagen procesada
        f = StringIO(urllib.urlopen(datos).read())
        img = Image.open(f)
        img_copy = img 
        img_copy.thumbnail(THUMB_SIZE, Image.ANTIALIAS)        
        converted = img_copy.convert('LA') #convertimos a 8 bits - blanco y negro
        p_img = converted.getdata()
        p_img = p_img.resize(THUMB_SIZE)
        p_img = map(list, p_img)
        p_img = np.array(p_img)
        s = p_img.shape[0] * p_img.shape[1]
        p_img_wide = p_img.reshape(1, s)

        hay_results = predecir(p_img_wide[0]) # llamamos a predecir respetando formato de los datos

        salida(hay_results)
        #VERIFICACIONES
        #path = '/home/bruno/Dev/FTI/ultimateRecognizer/statics/img/img.png'
        #path_converted = '../statics/img/8b-thumbnail.png' 
        #remove_file(path) #habilitar esta linea si se hacen las comprobaciones siguientes
        #remove_file(path_converted) #habilitar esta linea si se hacen las comprobaciones siguientes
        #converted.save(path_converted) #para comprobar que se guarda el thumbnail y en 8 bits
        #img.save(path) # para comprobar que se guarda la imagen en dimensiones originales 380x380

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
    methods,predictions,accuracy = main(predict_data=datos)
    tabla=''
    for method,pred,acc in zip(methods,predictions,accuracy):
        tabla+='<td>%s</td><td>%s</td><td>%s</td>' % (method,pred,acc)
    archivo_resultados = open("../html/resultados_cargados.html","w+")
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