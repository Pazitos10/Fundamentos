# -*- coding: utf-8 -*-
from random import randint
 
import numpy as np
from matplotlib import pyplot as plt
 
from net import  HopfieldNetwork
from trainers import hebbian_training

import pygame
 
#Define los patrones iniciales para todas las letras
a_pattern = np.array([[0, 0, 1, 0, 0],
                      [0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1]])
 
b_pattern = np.array([[1, 1, 1, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 0]])
 
c_pattern = np.array([[1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1]])

l_pattern = np.array([[1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1]])

z_pattern = np.array([[1, 1, 1, 1, 1],
                      [0, 0, 0, 1, 1],
                      [0, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 0],
                      [1, 1, 0, 0, 0],
                      [1, 1, 1, 1, 1]])



# cambia 0 por -1 en los arreglos
a_pattern *= 2
a_pattern -= 1
 
b_pattern *= 2
b_pattern -= 1
 
c_pattern *= 2
c_pattern -= 1


l_pattern *= 2
l_pattern -= 1


z_pattern *= 2
z_pattern -= 1

#flatten unifica todas las filas+columnas 
input_patterns = np.array([a_pattern.flatten(), b_pattern.flatten(), c_pattern.flatten(), l_pattern.flatten()])

#Create the neural network and train it using the training patterns
network = HopfieldNetwork(35)
 
hebbian_training(network, input_patterns)
 
#Create the test patterns by using the training patterns and adding some noise to them
#and use the neural network to denoise them 
a_test = np.array([[-1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1]
                      ])
#modifica algunas celdas del patron original para generar una prueba (modificarb)

# Definimos algunos colores
NEGRO    = (   0,   0,   0)
BLANCO    = ( 255, 255, 255)
VERDE    = (   0, 255,   0)
ROJO      = ( 255,   0,   0)
 

largo  = 20
alto = 20
 
# Establecemos el margen entre las celdas.
margen = 5

# Creamos un array bidimensional. Un array bidimensional
# no es más que una lista de listas.
grid = []
for fila in range(7):
    # Añadimos un array vacío que contendrá cada celda 
    # en esta fila
    grid.append([])
    for columna in range(5):
        grid[fila].append(0) # Añade una celda
 
# Inicializamos pygame
pygame.init()
  
# Establecemos el alto y largo de la pantalla
dimensiones = [255, 255]
pantalla = pygame.display.set_mode(dimensiones)
 
# Establecemos el título de la pantalla.
pygame.display.set_caption("Retículas y Matrices")
 
#Iteramos hasta que el usuario pulse el botón de salir.
hecho = False
 
# Lo usamos para establecer cuán rápido de refresca la pantalla.
reloj = pygame.time.Clock()

# -------- Bucle Principal del Programa-----------
while hecho == False:
    for evento in pygame.event.get(): 
        if evento.type == pygame.QUIT: 
            hecho = True
        elif evento.type == pygame.MOUSEMOTION: 
            if pygame.mouse.get_pressed() == (1,0,0):
                # El usuario presiona el ratón. Obtiene su posición.
                pos = pygame.mouse.get_pos()
                # Cambia las coordenadas x/y de la pantalla por coordenadas reticulares
                columna = pos[0] // (largo + margen)
                fila = pos[1] // (alto + margen)
                # Establece esa ubicación a cero
                grid[fila][columna] = 1
                #print("Click ", pos, "Coordenadas de la retícula: ", fila, columna)
                a_test[fila][columna] = 1
    # Establecemos el fondo de pantalla.
    pantalla.fill(NEGRO)
 
    # Dibujamos la retícula
    for fila in range(7):
        for columna in range(5):
            color = BLANCO
            if grid[fila][columna] == 1:
                color = VERDE
            pygame.draw.rect(pantalla,
                             color,
                             [(margen+largo)*columna+margen,
                              (margen+alto)*fila+margen,
                              largo,
                              alto])
     

    # Limitamos a 20 fotogramas por segundo.
    reloj.tick(20)
 
    # Avanzamos y actualizamos la pantalla con lo que hemos dibujado.
    pygame.display.flip()

a_test = a_test.flatten()     
a_result = network.run(a_test,30)
 
a_result.shape = (7, 5)
a_test.shape = (7, 5)

 
b_test =  b_pattern.flatten()
 
for i in range(4):
    p = randint(0, 34)
    b_test[p] *= -1
     
b_result = network.run(b_test)
 
b_result.shape = (7, 5)
b_test.shape = (7, 5)
 
c_test =  c_pattern.flatten()
 
for i in range(4):
    p = randint(0, 34)
    c_test[p] *= -1
     
c_result = network.run(c_test)
 
c_result.shape = (7, 5)
c_test.shape = (7, 5)
 
#Show the results
plt.subplot(3, 2, 1)
plt.imshow(a_test, interpolation="nearest")
plt.subplot(3, 2, 2)
plt.imshow(a_result, interpolation="nearest")
 
plt.subplot(3, 2, 3)
plt.imshow(b_test, interpolation="nearest")
plt.subplot(3, 2, 4)
plt.imshow(b_result, interpolation="nearest")
 
plt.subplot(3, 2, 5)
plt.imshow(c_test, interpolation="nearest")
plt.subplot(3, 2, 6)
plt.imshow(c_result, interpolation="nearest")


 
plt.show()