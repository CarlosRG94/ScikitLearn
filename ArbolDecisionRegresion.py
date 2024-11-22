import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

#Importamos los datos de la misma librería de scikit-learn
boston =fetch_california_housing()
print(boston)
print()

#verifico la información del dataset
print("Información del dataset:")
print(boston.keys())
print()

#Verifico las características del dataset
print("Características del dataset:")
print(boston.DESCR)
print()

#Verifico la cantidad de datos que hay en el dataset
print('Cantidad de datos:', boston.data.shape)
print()

#Verifico la información de las columnas
print('Nombres de las columnas:', boston.feature_names)

#Seleccionamos la columna 3 del dataset que es la que contiene la media de habitaciones por vivienda, porque al usar una regresion lineal simple,
#solo hay una variable independiente involucrada y=ax+b donde x es la variable independiente y y es la variable dependiente
X_adr = boston.data[:, np.newaxis, 2]

#Defino los datos correspondientes a las etiquetas target
y_adr = boston.target

#Graficamos los datos correspondientes
plt.scatter(X_adr, y_adr)
plt.xlabel('Número de habitaciones')
plt.ylabel('Valor medio')
plt.xlim(0, 20)
plt.ylim(0, 10)
plt.show()

#Separo los datos en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_adr, y_adr, test_size=0.2, random_state=42)

#Defino el algoritmo de Arbol de Decisión
adr = DecisionTreeRegressor(max_depth=40)

#Entreno el algoritmo con los datos de entrenamiento
adr.fit(X_train, y_train)

#Realizo una prediccion con los datos de prueba
Y_pred = adr.predict(X_test)
print('VALOR ESPERADO',y_test[:30])
print('Prediccion',Y_pred[:30])

#Graficamos los datos de prueba vs las predicciones para verificar la calidad del modelo
X_grid = np.arange(min(X_test), max(X_test), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test)
plt.plot(X_grid, adr.predict(X_grid), color='red', linewidth=3)
plt.xlim(0, 20)
plt.ylim(0, 10)
plt.show()

#Devuelve la precision del modelo en los datos de entrenamiento
print(f'Exactitud promedio prueba:{adr.score(X_train, y_train)}')





