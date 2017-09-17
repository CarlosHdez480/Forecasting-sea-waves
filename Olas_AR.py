# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:50:52 2017

@author: Carlos H
"""
#Autor:Carlos Hernández Hernández

# Importando modulos, paquetes y librerías necesarios
import numpy as np
import pandas as pd
import statsmodels.api as sm 
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt

#Ajustamos parámetros iniciales para que no emplee más de dos decimales.
pd.set_option('precision', 2)

#Cargamos los datos de oleaje
olas=pd.read_csv('Olas desplazamiento Z 1.csv', header=0)
olas=olas[:700]
olas.index = pd.DatetimeIndex(end=pd.datetime.today(), periods=len(olas), freq='1D')
#olas.index= para añadir la componente temporal de los datos de oleaje.
#No es necesario si los datos ya son leidos con componente temporal que sería lo ideal.
#Los datos ya se recogan con la componente temporal en el Data Frame.

#Calculamos función de autocorrelación, para conocer orden del modelo.
lag_acf = acf(olas, nlags=20)
lag_pacf = pacf(olas, nlags=20, method='ols')

#Presentamos función de autocorrelación.
plt.figure('1')
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(olas)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(olas)),linestyle='--',color='gray')
plt.title('Función de autocorrelación')

#Presentamos función de autocorrelación parcial.
plt.figure('2')
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(olas)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(olas)),linestyle='--',color='gray')
plt.title('Función de autocorrelación parcial')
plt.tight_layout()

#Ajustamos el modelo y calculamos los valores pronosticados.
modelo = sm.tsa.ARIMA(np.float32(olas['Desplazamiento']), order=(32, 0, 0))  
resultados = modelo.fit(disp=-1)  
olas['Pronostico'] = resultados.fittedvalues  

#Representamos la serie temporal con los valores relaes y pronosticados.
plt.figure('2')
plot = olas[['Desplazamiento', 'Pronostico']].plot(figsize=(10, 8))
plot.set_xlim(pd.Timestamp('2016-04-28'),pd.Timestamp('2016-07-27'))
plot.set_ylim(-90,90)

#Restaría añadir la componente dinámica del problema en la que se reescriben datos
# y se reentrena el modelo para predecir y estimar continuamente los nuevos estados
# de oleaje.