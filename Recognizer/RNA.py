import scipy.io as sio
import numpy as np
import heapq
import math

class RedNeuronal():

    def __init__(self, nro_de_entradas, capas_ocultas, nro_de_salidas):
        self.numero_de_entradas = nro_de_entradas
        self.capas_ocultas = capas_ocultas
        self.numero_de_salidas = nro_de_salidas
        print "cree la red neuronal"

    def sigmoide(self,z):
        """Calcula la funcion sigmoide"""
        return 1/(1+np.power(np.e,-z))

    def gradientesigmoide(self,z):
        """Gradient of sigmoide function"""
        return np.multiply(self.sigmoide(z),(1-self.sigmoide(z)));


    def unirVectores(self,v1, v2):
        return np.append(np.ravel(v1), np.ravel(v2))


    def probabilidad(self,Theta1, Theta2, X):
        """Clasifica el numero y calcula la probabilidad"""
        X = np.append(np.ones(shape=(X.shape[0],1)),X,axis=1)
        input = Theta1*np.matrix(X.transpose())
        capas_ocultas = self.sigmoide(input)
        capas_ocultas = np.append(np.ones(shape=(1,capas_ocultas.shape[1])),capas_ocultas,axis=0)
        prob = self.sigmoide(Theta2*capas_ocultas)
        l0 = np.array(prob.ravel())[0]
        l1 = heapq.nlargest(2, l0)
        prob2 = l1[1]
        estima2 = int(np.where(l0==l1[1])[0]+1)
        estima2 = estima2 if estima2<10 else 0
        number = int(prob.argmax(0).transpose())
        estima = number+1 if number<9 else 0
        return (estima, float(prob[number])*100), (estima2, prob2*100)


    def inicializacionAleatoria(self,i, epsilon=0.12):
        """Para la ruptura de la simetria inicializar thetas al azar"""
        return np.random.rand(i,1)*2*epsilon-epsilon


    def backProp(self,p, X, valor_de_y, l=0.2):
        """Backpropagation algorithm of neural network"""
        Theta1 = np.reshape(p[:self.capas_ocultas*(self.numero_de_entradas+1)], (self.capas_ocultas,-1))
        Theta2 = np.reshape(p[self.capas_ocultas*(self.numero_de_entradas+1):], (self.numero_de_salidas,-1))
        m = len(X)
        delta1 = 0
        delta2 = 0
        for t in range(m):
            
            a1 = np.matrix(np.append([1],X[t],axis=1)).transpose()
            z2 = Theta1*a1
            a2 = np.append(np.ones(shape=(1,z2.shape[1])), self.sigmoide(z2),axis=0)
            z3 = Theta2*a2
            a3 = self.sigmoide(z3)
            w = np.zeros((self.numero_de_salidas,1))
            w[int(valor_de_y[t])-1] = 1
            d3 = (a3-w)
            d2 = np.multiply(Theta2[:,1:].transpose()*d3, self.gradientesigmoide(z2))
            delta1 += d2*a1.transpose()
            delta2 += d3*a2.transpose()
            
        
        Theta1_grad = (1/m)*delta1 + (l/m)*np.append(np.zeros(shape=(Theta1.shape[0],1)), Theta1[:,1:], axis=1);
        Theta2_grad = (1/m)*delta2 + (l/m)*np.append(np.zeros(shape=(Theta2.shape[0],1)), Theta2[:,1:], axis=1);
       
        return self.unirVectores(Theta1_grad, Theta2_grad)

    def J(self,theta, X, valor_de_y, l=0.2):
        """Cost funtion"""
        
        Theta1 = np.reshape(theta[:self.capas_ocultas*(self.numero_de_entradas+1)], (self.capas_ocultas,-1))
        Theta2 = np.reshape(theta[self.capas_ocultas*(self.numero_de_entradas+1):], (self.numero_de_salidas,-1))
        m = len(X)
        X = np.append(np.ones(shape=(X.shape[0],1)),X,axis=1)
        J = 0
        for i in range(m):
            x = np.matrix(X[i])
            w = np.zeros((10,1))
            w[int(valor_de_y[i])-1] = 1
            hx = self.sigmoide(Theta2*np.append([[1]], self.sigmoide(Theta1*x.transpose()), axis=0))
            J += sum(-w.transpose()*np.log(hx)-(1-w).transpose()*np.log(1-hx))
        J = J/m
        J += (l/(2*m))*(sum(sum(Theta1[:,1:]**2)) + sum(sum(Theta2[:,1:]**2)))    
        return float(J)

    def calculateGrad(self,p, Xentrenamiento, yentrenamiento):
        """Metodo Backpropagation para optimizar"""
        return self.backProp(p, Xentrenamiento,yentrenamiento)
        
    def calculateJ(self,p, Xentrenamiento, yentrenamiento):
        """Metodo de costo para optimizar"""
        return self.J(p, Xentrenamiento, yentrenamiento)