#!/usr/bin/python
# -*- coding: utf-8 -*-
#Authors: pazitos10, SinX


from pygame.locals import *
import pygame.font, pygame.event, pygame.draw
import scipy.io as sio
import numpy as np 
from desperateGuys import DesperateGuysClassifier 
screen = None
dgc = DesperateGuysClassifier()
acc = 0.0
Xtrain, Xtest, ytrain, ytest = [], [], [], []


def calculateImage(background, screen, lineWidth):
    """Crop and resize the input"""
    global changed
    focusSurface = pygame.surfarray.array3d(background)
    focus = abs(1-focusSurface/255)
    focus = np.mean(focus, 2) 
    x = []
    xaxis = np.sum(focus, axis=1)
    for i, v in enumerate(xaxis):
        if v > 0:
            x.append(i)
            break
    for i, v in enumerate(xaxis[ : :-1]):
        if v > 0:
            x.append(len(xaxis)-i)
            break
    
    y = []
    yaxis = np.sum(focus, axis=0)
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
        bound = focus.shape[0]      
        if dx > dy:
            d = dx-dy
            y0t = y[0] - d//2
            y1t = y[1] + d//2+d%2
            if y0t < 0: y0t = y[0]; y1t = y[1] + d
            if y1t > bound: y0t = y[0] - d; y1t = y[1]
            y[0], y[1] = y0t, y1t
        else:
            d = dy-dx
            x0t = x[0] - d//2
            x1t = x[1] + d//2+d%2
            if x0t < 0: x0t = x[0]; x1t = x[1] + d
            if x1t > bound: x0t = x[0] - d; x1t = x[1]
            x[0], x[1] = x0t, x1t 
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        changed = True
        crop_surf =  pygame.Surface((dx,dy))
        crop_surf.blit(background,(0,0),(x[0],y[0],x[1],y[1]), special_flags=BLEND_RGBA_MAX)
        scaledBackground = pygame.transform.smoothscale(crop_surf, (30, 30))
            
        image = pygame.surfarray.array3d(scaledBackground)
        image = abs(1-image/253)
        image = np.mean(image, 2) 
        image = np.matrix(image.ravel())
        drawPixelated(image, screen)
        
        myLabel = predictAndShowStats(lineWidth, image)
        (x,y) = screen.get_size()
        screen.blit(myLabel, (17, y-90))
        
    except:
        image = np.zeros((30,30))
    return image

def predictAndShowStats(lineWidth, drawnImageData):
    """ shows the current statistics """
    global ytest, dgc
    predicted = dgc.predict(drawnImageData)
    print "predicted", predicted
    myFont = pygame.font.SysFont("Verdana", 50)
    stats = "Predicted : %s" % predicted[0]
    statSurf = myFont.render(stats, 1, ((255, 255, 255)))
    return statSurf


def checkKeys(myData):
    """test for various keyboard inputs"""

    (event, background, drawColor, lineWidth, keepGoing, screen, image) = myData
    
    if event.key == pygame.K_q:
        keepGoing = False
    elif event.key == pygame.K_c:
        background.fill((255, 255, 255))
        drawPixelated(np.zeros((30,30)), screen)
    elif event.key == pygame.K_s:
        global changed
        try:
            mat_contents = sio.loadmat('newX.mat')
            X = mat_contents['X']
            X = np.append(X,image.getA(), axis=0)
        except:
            X = image.getA()
        answer = np.matrix(int(ask(screen, "")))
        try:
            mat_contents = sio.loadmat('newy.mat')
            y = mat_contents['y']
            y = np.append(y,answer, axis=0)
        except:
            y = answer
        if changed:
            sio.savemat('newX.mat', {'X': X})
            sio.savemat('newy.mat', {'y': y})
            changed = False

        background.fill((255, 255, 255))
        drawPixelated(np.zeros((30,30)), screen)

    elif event.key == pygame.K_t:
        screen.fill((255, 255, 255))
        myFont1 = pygame.font.SysFont("Verdana", 55)
        myFont2 = pygame.font.SysFont("Verdana", 17)
        screen.blit(myFont1.render("Please wait!", 1, ((0, 0, 0))), (365, 90))
        screen.blit(myFont2.render("Neural network training in progress...", 1, ((50, 50, 50))), (368, 150))
        pygame.display.flip()
        
        simulateTraining() #entrenamiento
        
        screen.fill((0, 0, 0))
        background.fill((255, 255, 255))
    elif event.key == pygame.K_v:
        drawStatistics(screen)

    myData = (event, background, drawColor, lineWidth, keepGoing)
    return myData

def simulateTraining():
    global Xtrain; global Xtest; global ytrain; global ytest; global dgc; global acc
    mat_contents = sio.loadmat('newX.mat')
    Xs = mat_contents['X']
    mat_contents = sio.loadmat('newy.mat')
    y = mat_contents['y']
    ys = y.ravel()
    Xtrain, Xtest, ytrain, ytest = dgc.splitData(Xs,ys)
    dgc.train(Xtrain,ytrain.ravel()) # entrena
    expected = ytest
    predicted = dgc.predict(Xtest)
    acc = "%.2f" % (dgc.getGlobalAccuracy(Xs, ys))


def display_box(screen, message):
    "Print a message in a box on the screen"
    
    fontobject = pygame.font.Font(None,120)
    pygame.draw.rect(screen, (0,0,0),
                   ((screen.get_width() / 2) - 100,
                    (screen.get_height()) - 170,
                    70,90), 0)
    if len(message) != 0:
        screen.blit(fontobject.render(message, 1, (255,255,255)),
                ((screen.get_width() / 2) - 110, (screen.get_height()) -168))
        pygame.display.flip()

def get_key():
    """Get key event"""
    while 1:
        event = pygame.event.poll()
        if event.type == KEYDOWN:
            return event.key
        else:
            pass


def ask(screen, question):
    """create input box for entering correct value of y"""
    
    pygame.font.init()
    current_string = str()
    display_box(screen, question + " " + current_string+"")
    while 1:
        inkey = get_key()
        if inkey == K_BACKSPACE:
            current_string = current_string[0:-1]
        elif inkey == K_RETURN:
            break
        elif inkey == K_MINUS:
            current_string.append("_")
        elif inkey <= 127:
            current_string+= (chr(inkey))
        display_box(screen, question + " " + current_string+"")
    return current_string


def drawPixelated(A, screen):  
    """Draw 30x30 image of input""" 
    
    A = A.ravel()
    A = (255-A*255).transpose()
    size = 30
    for x in range(size):
        for y in range(size):
            z=x*30+y
            c = int(A[z])
            pygame.draw.rect(screen,(c,c,c),(x*11+385,15+y*11,11,11))



def drawStatistics(screen):  
    """Draw statistics about training set"""
    global dgc, acc
    mat_contents = sio.loadmat('newX.mat')
    Xs = mat_contents['X']
    mat_contents = sio.loadmat('newy.mat')
    ys = mat_contents['y']
    Xtrain, Xtest, ytrain, ytest = dgc.splitData(Xs,ys)
    #mat_contents = sio.loadmat('scaledTheta.mat')
    accuracy = acc

    y = ys.ravel().tolist()

    myFont = pygame.font.SysFont("Verdana", 24)
    myFont2 = pygame.font.SysFont("Verdana", 18)
    myFont3 = pygame.font.SysFont("Verdana", 16)
    pygame.draw.rect(screen,(255,255,255),(370,0,730,360))
    screen.blit(myFont.render("Samples: %d" % (Xs.shape[0]), 1, ((0, 0, 0))), (400, 30))
    screen.blit(myFont.render("Accuracy: %s" % str(accuracy)+"%", 1, ((0, 0, 0))), (400, 60))
    screen.blit(myFont3.render("SAMPLE DISTRIBUTION:", 1, ((0, 0, 0))), (400, 100))
    screen.blit(myFont2.render("Count 0 = %s" % (y.count(0)), 1, ((0, 0, 0))), (400, 120))
    for i in range(9):
        screen.blit(myFont2.render("Count %s = %s" % (i+1, y.count(i+1)), 1, ((0, 0, 0))), (400, 140+i*20))

def main():
    global screen;
    pygame.init()
    screen = pygame.display.set_mode((730, 450))
    pygame.display.set_caption("Handwriting recognition")
    
    background = pygame.Surface((360,360))
    background.fill((255, 255, 255))
    background2 = pygame.Surface((360,360))
    background2.fill((255, 255, 255))
    
    clock = pygame.time.Clock()
    keepGoing = True
    lineStart = (0, 0)
    drawColor = (255, 0, 0)
    lineWidth = 15

    simulateTraining() #entrenamiento inicial

    pygame.display.update()
    image = None

    while keepGoing:
        
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keepGoing = False
            elif event.type == pygame.MOUSEMOTION:
                lineEnd = pygame.mouse.get_pos()
                if pygame.mouse.get_pressed() == (1, 0, 0):
                    pygame.draw.line(background, drawColor, lineStart, lineEnd, lineWidth)
                lineStart = lineEnd
            elif event.type == pygame.MOUSEBUTTONUP:
                screen.fill((0, 0, 0))
                screen.blit(background2, (370, 0))
                #w = threading.Thread(name='worker', target=worker)
                image = calculateImage(background, screen, lineWidth)

            elif event.type == pygame.KEYDOWN:
                myData = (event, background, drawColor, lineWidth, keepGoing, screen, image)
                myData = checkKeys(myData)
                (event, background, drawColor, lineWidth, keepGoing) = myData
        
        
        screen.blit(background, (0, 0))
        pygame.display.flip()



if __name__ == '__main__':
    main()
