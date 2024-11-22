import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Importamos los datos de la misma librería de scikit-learn
diabetes = load_diabetes()
print(diabetes)
print()

#verifico la información del dataset
print("Información del dataset:")
print(diabetes.keys())
print()

#Verifico las características del dataset
print("Características del dataset:")
print(diabetes.DESCR)
print()

#Verifico la cantidad de datos que hay en el dataset
print('Cantidad de datos:', diabetes.data.shape)
print()

#Verifico la información de las columnas
print('Nombres de las columnas:', diabetes.feature_names)

#Seleccionamos multiples columnas del dataset que son las que contienen las variables independientes involucradas
X_multiple = diabetes.data[:,[2, 3, 8]]
print(X_multiple)

#Defino los datos correspondientes a las etiquetas target
y_multiple = diabetes.target

#Separo los datos en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y_multiple, test_size=0.2, random_state=42)

#Definimos el algoritmo de regresion lineal multiple
lr_multiple = LinearRegression()

#Entrenamos el modelo con los datos de entrenamiento
lr_multiple.fit(X_train, y_train)

#Devuelve la puntuación del modelo en los datos de prueba
print(f'Exactitud promedio prueba:{lr_multiple.score(X_test, y_test)}')

#Realizo una predicción del modelo
Y_pred_multiple = lr_multiple.predict(X_test)
print('VALOR ESPERADO',y_test[:30])
print('Predicción',Y_pred_multiple[:30])

print('Datos del modelo de regresión lineal multiple:')
print('Coeficientes:', lr_multiple.coef_)
print('Intercepto:', lr_multiple.intercept_)






