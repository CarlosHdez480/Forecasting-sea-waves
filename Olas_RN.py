# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 13:41:42 2017

@author: Carlos H
"""

#Autor:Carlos Hernández Hernández

#Cargamos paquetes y librerías
import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

secuencia=[] #Secuencia de datos sobre el que cargaremos los datos de oleaje

#Leemos datos del archivo csv con los datos de oleaje
olas=pd.read_csv("Olas desplazamiento Z 1.csv", header=0)

#Normalizamos la secuencia entre 0 y 1    
secuencia=np.asarray(olas["Desplazamiento"])
secuencia=secuencia/np.amax(secuencia)

#Graficamos los primeros 64 datos de oleaje
plt.plot(secuencia[:64])
plt.xlabel('Tiempo (s)')
plt.ylabel('Altura Olas')
plt.title('Tren de oleaje')
plt.show()

longitud_muestra=10 #Elegido arbitrariamente
muestras_vector=[]
etiquetas_vector=[]
 
for i in range(len(secuencia)-longitud_muestra):
     muestras_vector.append(secuencia[i:i+longitud_muestra])
     etiquetas_vector.append([secuencia[i+longitud_muestra]])

intervalos = [[0, 1.0] for i in range(longitud_muestra)] #intervalos de valores de entrada  

#Entrenamos y creamos la red neuronal    
red = nl.net.newff(intervalos,[15, 1],[nl.trans.TanSig(), nl.trans.TanSig()])
error = red.train(muestras_vector, etiquetas_vector, epochs=600, show=50, goal=0.01)

resultados = []
offset = 20
width = 120
error = []

#Simulamos la red y obtenemos la nueva serie pronosticada.
for i in range(len(secuencia)-longitud_muestra):
    resultados.append(red.sim([secuencia[i:i+longitud_muestra]])[0][0])
    error.append(abs(secuencia[longitud_muestra+i-1] - resultados[i]))
        
#Sumatorio de errores.
error_absoluto = np.sum(error)
print ("Error absoluto: ", error_absoluto)
    
offset = 170

#Representamos con el error.
plt.figure('2')
plt.plot(resultados[0+offset:width+offset], 'r-', secuencia[longitud_muestra-1+offset:longitud_muestra+width-1+offset], 'g-', error[0+offset:width+offset], 'b-')
plt.xlabel('Rojo:valor real - Verde:valor pronosticado - Azul:error')
plt.ylabel('Valores')
plt.title('Resultados')
plt.grid(True)
plt.show()

#Representamos sin el error.
plt.figure('3')
plt.plot(resultados[0+offset:width+offset], 'r-', secuencia[longitud_muestra-1+offset:longitud_muestra+width-1+offset], 'g-')
plt.xlabel('Rojo:valor real - Verde:valor pronosticado - Azul:error')
plt.xlim(0,64)
plt.ylabel('Valores')
plt.title('Resultados')
plt.grid(True)
plt.show()

#Restaría añadir la componente dinámica del problema en la que se reescriben datos
# y se reentrena el modelo para predecir y estimar continuamente los nuevos estados
# de oleaje.


