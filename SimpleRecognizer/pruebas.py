# -*- coding: utf-8 -*-
import pygame
import numpy as np
from net import  HopfieldNetwork
from trainers import hebbian_training

from matplotlib import pyplot as plt

#Define los patrones iniciales para todas las letras
 
c_pattern = np.array([[1, 1, 1, 1, 1,1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 1, 1, 1, 1,1,1,1,1,1]])

l_pattern = np.array([[1, 0, 0, 0, 0,0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 0, 0, 0, 0,0,0,0,0,0],
                      [1, 1, 1, 1, 1,1,1,1,1,1]])

z_pattern = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0 ,0 ,0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])



# cambia 0 por -1 en los arreglos
c_pattern *= 2
c_pattern -= 1

l_pattern *= 2
l_pattern -= 1


z_pattern *= 2
z_pattern -= 1


#flatten unifica todas las filas+columnas 
input_patterns = np.array([c_pattern.flatten(), z_pattern.flatten(), l_pattern.flatten(), ])
print input_patterns

#Create the neural network and train it using the training patterns
network = HopfieldNetwork(100)
 
hebbian_training(network, input_patterns)
 
# Definimos algunos colores
NEGRO    = (   0,   0,   0)
BLANCO    = ( 255, 255, 255)
VERDE    = (   0, 255,   0)
ROJO      = ( 255,   0,   0)
 

largo  = 20
alto = 20
 
# Establecemos el margen entre las celdas.
margen = 5

a_test = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      ])
 
# Creamos un array bidimensional. Un array bidimensional
# no es más que una lista de listas.
grid = []
for fila in range(10):
    # Añadimos un array vacío que contendrá cada celda 
    # en esta fila
    grid.append([])
    for columna in range(10):
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
    for fila in range(10):
        for columna in range(10):
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
a_result = network.run(a_test)
a_result.shape = (10, 10)
a_test.shape = (10, 10)

plt.subplot(3, 2, 1)
plt.imshow(a_test, interpolation="nearest")
plt.subplot(3, 2, 2)
plt.imshow(a_result, interpolation="nearest")

plt.show()
 
     
# Pórtate bien con el IDLE.
pygame.quit()