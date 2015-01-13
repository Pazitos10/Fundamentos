#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import pygame.font, pygame.event, pygame.draw
import scipy.optimize as sc
from pygame.locals import *
import heapq

hubo_cambio = False
total = 0
Xentrenamiento, Xprueba, yentrenamiento, yprueba = [], [], [], []
num_ocultas = 25
num_entradas = 900
num_lables = 10
pantalla = None

def dividirDatos(X, y):
    """Divide la muestra de datos en un conjunto de entrenamiento (80%) y uno de prueba (20%)"""
    
    tamanio1 = X.shape[0] * 0.8
    tamanio2 = X.shape[0] * 0.2
    Xentrenamiento = np.zeros((tamanio1,X.shape[1]))
    Xprueba = np.zeros((tamanio2,X.shape[1]))
    yentrenamiento = np.zeros((tamanio1,1))
    yprueba = np.zeros((tamanio2,1))
    for i, v in enumerate(np.random.permutation(len(y))):
        #print(i, y[v], len(X[v]))
        
        try:
            Xentrenamiento[i] = X[v]
            yentrenamiento[i] = y[v]
        except:
            Xprueba[i-tamanio1] = X[v]
            yprueba[i-tamanio1] = y[v]
    return Xentrenamiento, Xprueba, yentrenamiento, yprueba
    
def calcularImagen(fondo, pantalla, Theta1, Theta2, ancho_de_linea):
    """Corta y redimensiona la entrada"""
    
    global hubo_cambio
    superficie_de_foco = pygame.surfarray.array3d(fondo)
    foco = abs(1-superficie_de_foco/255)
    foco = np.mean(foco, 2) 
    x = []
    xaxis = np.sum(foco, axis=1)
    for i, v in enumerate(xaxis):
        if v > 0:
            x.append(i)
            break
    for i, v in enumerate(xaxis[ : :-1]):
        if v > 0:
            x.append(len(xaxis)-i)
            break
    
    y = []
    yaxis = np.sum(foco, axis=0)
    for i, v in enumerate(yaxis):
        if v > 0:
            y.append(i)
            break
    for i, v in enumerate(yaxis[ : :-1]):
        if v > 0:
            y.append(len(yaxis)-i)
            break

    try:
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        limite = foco.shape[0]     #limite o borde de la forma
        if dx > dy:
            d = dx-dy
            y0t = y[0] - d//2
            y1t = y[1] + d//2+d%2
            if y0t < 0: y0t = y[0]; y1t = y[1] + d
            if y1t > limite: y0t = y[0] - d; y1t = y[1]
            y[0], y[1] = y0t, y1t
        else:
            d = dy-dx
            x0t = x[0] - d//2
            x1t = x[1] + d//2+d%2
            if x0t < 0: x0t = x[0]; x1t = x[1] + d
            if x1t > limite: x0t = x[0] - d; x1t = x[1]
            x[0], x[1] = x0t, x1t 
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        hubo_cambio = True
        superficie_recortada =  pygame.Surface((dx,dy))
        superficie_recortada.blit(fondo,(0,0),(x[0],y[0],x[1],y[1]), special_flags=BLEND_RGBA_MAX)
        fondo_escalado = pygame.transform.smoothscale(superficie_recortada, (30, 30))
        imagen = pygame.surfarray.array3d(fondo_escalado)
        imagen = abs(1-imagen/253)
        imagen = np.mean(imagen, 2) 
        imagen = np.matrix(imagen.ravel())
        dibujarPixeles(imagen, pantalla)
        (valor, prob), (valor2, prob2) = probabilidad(Theta1,Theta2,imagen)
        prob = round(prob,1)
        prob2 = round(prob2, 1)
        label_estadisticas = mostrarEstadisticas(ancho_de_linea, valor, prob)
        label_estadisticas_2 = mostrarEstadisticasPequenia(ancho_de_linea, valor2, prob2)
        (x,y) = pantalla.get_size()
        pantalla.blit(label_estadisticas, (17, y-90))
        pantalla.blit(label_estadisticas_2, (20, y-38))
    except:
        imagen = np.zeros((30,30))

    return imagen
    
def sigmoid(z):
    """Calcula la funcion sigmoid"""
    
    return 1/(1+np.power(np.e,-z))

def probabilidad(Theta1, Theta2, X):
    """Clasifica el numero y calcula la probabilidad"""
    
    X = np.append(np.ones(shape=(X.shape[0],1)),X,axis=1)
    input = Theta1*np.matrix(X.transpose())
    capa_oculta = sigmoid(input)
    capa_oculta = np.append(np.ones(shape=(1,capa_oculta.shape[1])),capa_oculta,axis=0)
    prob = sigmoid(Theta2*capa_oculta)
    l0 = np.array(prob.ravel())[0]
    l1 = heapq.nlargest(2, l0)
    prob2 = l1[1]
    estima2 = int(np.where(l0==l1[1])[0]+1)
    estima2 = estima2 if estima2<10 else 0
    number = int(prob.argmax(0).transpose())
    estima = number+1 if number<9 else 0
    print (estima, float(prob[number])*100), (estima2, prob2*100)
    return (estima, float(prob[number])*100), (estima2, prob2*100)



def probabilidadesParaDibujar(Theta1, Theta2, X):
    """Calcula las probabilidades de prediccion"""
    X = np.append(np.ones(shape=(X.shape[0],1)),X,axis=1)
    input = Theta1*np.matrix(X.transpose())
    capa_oculta = sigmoid(input)
    capa_oculta = np.append(np.ones(shape=(1,capa_oculta.shape[1])),capa_oculta,axis=0)
    proba = sigmoid(Theta2*capa_oculta)
    numbers = proba.argmax(0).transpose()+1
    return numbers

def obtenerPrecision(prob, y):
    """Calcula la precision de la prediccion"""
    total = 0
    for i in range(len(prob)):
        if int(prob[i]) == int(y[i]):
            total += 1
        elif int(prob[i]) == 10 and int(y[i]) == 0:
            total += 1
    return round((total/len(prob))*100,2)

def obtener_tecla():
    """Obtiene el evento de la tecla presionada"""
    while 1:
        event = pygame.event.poll()
        if event.type == KEYDOWN:
            return event.key
        else:
            pass

def mostrar_caja(pantalla, mensaje):
    "Muestra un mensaje en una caja dentro de la pantalla"
    
    fuente = pygame.font.Font(None,120)
    pygame.draw.rect(pantalla, (0,0,0),
                   ((pantalla.get_width() / 2) - 100,
                    (pantalla.get_height()) - 170,
                    70,90), 0)
    if len(mensaje) != 0:
        pantalla.blit(fuente.render(mensaje, 1, (255,255,255)),
                ((pantalla.get_width() / 2) - 110, (pantalla.get_height()) -168))
        pygame.display.flip()


def consultar(pantalla, consulta):
    """crea un input box para ingresar el valor correcto de y"""
    
    pygame.font.init()
    current_string = str()
    mostrar_caja(pantalla, consulta + " " + current_string+"")
    while 1:
        inkey = obtener_tecla()
        if inkey == K_BACKSPACE:
            current_string = current_string[0:-1]
        elif inkey == K_RETURN:
            break
        elif inkey == K_MINUS:
            current_string.append("_")
        elif inkey <= 127:
            current_string+= (chr(inkey))
        mostrar_caja(pantalla, consulta + " " + current_string+"")
    return current_string
    
def probarTeclas(datos):
    """prueba distintas entradas de teclado"""
    
    (evento, fondo, color, ancho_de_linea, continuar, pantalla, imagen) = datos
    
    if evento.key == pygame.K_q:
        continuar = False
    elif evento.key == pygame.K_c:
        fondo.fill((255, 255, 255))
        dibujarPixeles(np.zeros((30,30)), pantalla)
    elif evento.key == pygame.K_s:
        global hubo_cambio
        try:
            contenido_matriz = sio.loadmat('newX.mat')
            X = contenido_matriz['X']
            X = np.append(X,imagen, axis=0)
        except:
            X = imagen
        respuesta = np.matrix(int(consultar(pantalla, "")))
        try:
            contenido_matriz = sio.loadmat('newy.mat')
            y = contenido_matriz['y']
            y = np.append(y,respuesta, axis=0)
        except:
            y = respuesta
        if hubo_cambio:
            sio.savemat('newX.mat', {'X': X})
            sio.savemat('newy.mat', {'y': y})
            hubo_cambio = False

        fondo.fill((255, 255, 255))
        dibujarPixeles(np.zeros((30,30)), pantalla)

    elif evento.key == pygame.K_t:
        pantalla.fill((255, 255, 255))
        fuente1 = pygame.font.SysFont("Verdana", 55)
        fuente2 = pygame.font.SysFont("Verdana", 17)
        fuente3 = pygame.font.SysFont("Verdana", 9)
        pantalla.blit(fuente1.render("Please wait!", 1, ((0, 0, 0))), (365, 90))
        pantalla.blit(fuente2.render("Neural network training in progress...", 1, ((50, 50, 50))), (368, 150))
        pantalla.blit(fuente3.render("Depending on the training data size this could take long time", 1, ((80, 80, 80))), (370, 190))
        pygame.display.flip()
        global Xentrenamiento; global Xprueba; global yentrenamiento; global yprueba
        contenido_matriz = sio.loadmat('newX.mat')
        Xs = contenido_matriz['X']
        contenido_matriz = sio.loadmat('newy.mat')
        ys = contenido_matriz['y']
        Xentrenamiento, Xprueba, yentrenamiento, yprueba = dividirDatos(Xs,ys)
        
        rndInit = inicializacionAleatoria(25*901+10*26)
        respuesta =  sc.fmin_cg(calculateJ, rndInit, calculateGrad, maxiter=100,  disp=True, callback=callback)
        Theta1 = np.reshape(respuesta[:num_hidden*(num_input+1)], (num_hidden,-1))
        Theta2 = np.reshape(respuesta[num_hidden*(num_input+1):], (num_lables,-1))

        precision = obtenerPrecision(probabilidadesParaDibujar(Theta1, Theta2, Xtest), ytest)
        sio.savemat('scaledTheta.mat', {'t': respuesta, 'acc': precision})
        pantalla.fill((0, 0, 0))
        fondo.fill((255, 255, 255))
    elif evento.key == pygame.K_v:
        dibujarEstadisticas(pantalla)

    datos = (evento, fondo, color, ancho_de_linea, continuar)
    return datos


def mostrarEstadisticas(ancho_de_linea, valor, probabilidad):
    """ Muestra las estadisticas actuales """
    
    fuente = pygame.font.SysFont("Verdana", 50)
    estadisticas = "Estimate: %s    P: %s" % (valor, probabilidad)
    return fuente.render(estadisticas+"%", 1, ((255, 255, 255)))


def mostrarEstadisticasPequenia(ancho_de_linea, valor, probabilidad):

    fuente = pygame.font.SysFont("Verdana", 25)
    estadisticas = "Second estimate: %s (%s" % (valor, probabilidad)
    return fuente.render(estadisticas+"%)", 1, ((235, 235, 235)))

def dibujarPixeles(A, pantalla):  
    """dibuja una imagen de 30x30 a partir de una entrada""" 
  
    A = A.ravel()
    A = (255-A*255).transpose()
    tamanio = 25
    for x in range(tamanio):
        for y in range(tamanio):
            z=x*30+y
            c = int(A[z])
            pygame.draw.rect(pantalla,(c,c,c),(x*11+385,15+y*11,11,11))



def dibujarEstadisticas(pantalla):  
    """Dibuja las estadisticas en la pantalla"""

    contenido_mat = sio.loadmat('newX.mat')
    Xs = contenido_mat['X']
    contenido_mat = sio.loadmat('newy.mat')
    ys = contenido_mat['y']
    Xtrain, Xtest, ytrain, ytest = splitData(Xs,ys)
    contenido_mat = sio.loadmat('scaledTheta.mat')
    acc = float(contenido_mat['acc'])
    y = ys.ravel().tolist()

    myFont = pygame.font.SysFont("Verdana", 24)
    myFont2 = pygame.font.SysFont("Verdana", 18)
    myFont3 = pygame.font.SysFont("Verdana", 16)
    pygame.draw.rect(pantalla,(255,255,255),(370,0,730,360))
    pantalla.blit(myFont.render("Muestras: %d" % (Xs.shape[0]), 1, ((0, 0, 0))), (400, 30))
    pantalla.blit(myFont.render("Precision: %s" % str(acc)+"%", 1, ((0, 0, 0))), (400, 60))
    pantalla.blit(myFont3.render("DISTRIBUCION DE LA MUESTRA:", 1, ((0, 0, 0))), (400, 100))
    pantalla.blit(myFont2.render("Total 0 = %s" % (y.count(0)), 1, ((0, 0, 0))), (400, 120))
    for i in range(9):
        pantalla.blit(myFont2.render("Total %s = %s" % (i+1, y.count(i+1)), 1, ((0, 0, 0))), (400, 140+i*20))


def inicializacionAleatoria(i, epsilon=0.12):
    """Para la ruptura de la simetria inicializar thetas al azar"""
    return np.random.rand(i,1)*2*epsilon-epsilon


def gradienteSigmoide(z):
    """Gradient of sigmoid function"""
    return np.multiply(sigmoid(z),(1-sigmoid(z)));


def unirVectores(v1, v2):
    return np.append(np.ravel(v1), np.ravel(v2))


def backProp(p, numero_de_entrada, numero_oculto, numero_de_etiquetas, X, valor_de_y, l=0.2):
    """Backpropagation algorithm of neural network"""
    
    Theta1 = np.reshape(p[:numero_oculto*(numero_de_entrada+1)], (numero_oculto,-1))
    Theta2 = np.reshape(p[numero_oculto*(numero_de_entrada+1):], (numero_de_etiquetas,-1))
    m = len(X)
    delta1 = 0
    delta2 = 0
    for t in range(m):

        a1 = np.matrix(np.append([1],X[t],axis=1)).transpose()
        z2 = Theta1*a1
        a2 = np.append(np.ones(shape=(1,z2.shape[1])), sigmoid(z2),axis=0)
        z3 = Theta2*a2
        a3 = sigmoid(z3)
        w = np.zeros((numero_de_etiquetas,1))
        w[int(valor_de_y[t])-1] = 1
        d3 = (a3-w)
        d2 = np.multiply(Theta2[:,1:].transpose()*d3, gradienteSigmoide(z2))
        delta1 += d2*a1.transpose()
        delta2 += d3*a2.transpose()
        
    
    Theta1_grad = (1/m)*delta1 + (l/m)*np.append(np.zeros(shape=(Theta1.shape[0],1)), Theta1[:,1:], axis=1);
    Theta2_grad = (1/m)*delta2 + (l/m)*np.append(np.zeros(shape=(Theta2.shape[0],1)), Theta2[:,1:], axis=1);
   
    return unirVectores(Theta1_grad, Theta2_grad)

def J(theta, numero_de_entrada, numero_oculto, numero_de_etiquetas,X, valor_de_y, l=0.2):
    """Cost funtion"""
    
    Theta1 = np.reshape(theta[:numero_oculto*(numero_de_entrada+1)], (numero_oculto,-1))
    Theta2 = np.reshape(theta[numero_oculto*(numero_de_entrada+1):], (numero_de_etiquetas,-1))
    m = len(X)
    X = np.append(np.ones(shape=(X.shape[0],1)),X,axis=1)
    J = 0
    for i in range(m):
        x = np.matrix(X[i])
        w = np.zeros((10,1))
        w[int(valor_de_y[i])-1] = 1
        hx = sigmoid(Theta2*np.append([[1]], sigmoid(Theta1*x.transpose()), axis=0))
        J += sum(-w.transpose()*np.log(hx)-(1-w).transpose()*np.log(1-hx))
    J = J/m
    J += (l/(2*m))*(sum(sum(Theta1[:,1:]**2)) + sum(sum(Theta2[:,1:]**2)))    
    return float(J)

def calculateGrad(p):
    """Metodo Backpropagation para optimizar"""
    
    return backProp(p, 900, 25, 10, Xentrenamiento,yentrenamiento)
    
def calculateJ(p):
    """Metodo de costo para optimizar"""
    return J(p, 900, 25, 10, Xentrenamiento, yentrenamiento)

def callback(p):
    """Actualiza la GUI mientras se entrena la red neuronal"""
    global total
    global pantalla
    pygame.event.get()
    total += 1
    fuente = pygame.font.SysFont("Verdana", 110)
    puntos= []
    if total >= 9:
        total = 1
        pygame.draw.rect(pantalla,(255,255,255),(355,220,400,250))
        puntos = []
    for i in range(total):
        puntos.append(".")
    pantalla.blit(fuente.render("".join(puntos), 1, ((150, 150, 150))), (355, 140))
    pygame.display.flip()


def main():
    global pantalla
    pygame.init()
    pantalla = pygame.display.set_mode((730, 450))
    pygame.display.set_caption("Reconocimiento de caracteres")
    
    fondo = pygame.Surface((360,360))
    fondo.fill((255, 255, 255))
    fondo2 = pygame.Surface((360,360))
    fondo2.fill((255, 255, 255))
    
    reloj = pygame.time.Clock()
    sigue = True
    linea_de_inicio = (0, 0)
    color_de_dibujo = (255, 0, 0)
    ancho_de_linea = 15
    
    inputTheta = sio.loadmat('scaledTheta.mat')
    theta = inputTheta['t']
    numero_oculto = 25
    numero_de_entrada = 900
    numero_de_etiquetas = 10

    Theta1 = np.reshape(theta[:numero_oculto*(numero_de_entrada+1)], (numero_oculto,-1))
    Theta2 = np.reshape(theta[numero_oculto*(numero_de_entrada+1):], (numero_de_etiquetas,-1))

    pygame.display.update()
    imagen = None
            
    while sigue:
        
        reloj.tick(30)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                sigue = False
            elif evento.type == pygame.MOUSEMOTION:
                linea_de_fin = pygame.mouse.get_pos()
                if pygame.mouse.get_pressed() == (1, 0, 0):
                    pygame.draw.line(fondo, color_de_dibujo, linea_de_inicio, linea_de_fin, ancho_de_linea)
                linea_de_inicio = linea_de_fin
            elif evento.type == pygame.MOUSEBUTTONUP:
                pantalla.fill((0, 0, 0))
                pantalla.blit(fondo2, (370, 0))
                #w = threading.Thread(name='worker', target=worker)
                imagen = calcularImagen(fondo, pantalla, Theta1, Theta2, ancho_de_linea)

            elif evento.type == pygame.KEYDOWN:
                misDatos = (evento, fondo, color_de_dibujo, ancho_de_linea, sigue, pantalla, imagen)
                misDatos = probarTeclas(misDatos)
                (evento, fondo, color_de_dibujo, ancho_de_linea, sigue) = misDatos
        
        
        pantalla.blit(fondo, (0, 0))
        pygame.display.flip()

        
if __name__ == "__main__":
    main()