import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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

#Seleccionamos la columna 4 del dataset que contiene la masa corporal
X_bar = diabetes.data[:, np.newaxis, 2]
#Defino los datos correspondientes a las etiquetas target
y_bar = diabetes.target 

#Graficamos los datos correspondientes
plt.scatter(X_bar, y_bar)
plt.show()

#Separo los datos en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_bar, y_bar, test_size=0.2, random_state=42)

#Defino el algoritmo de Bosque Aleatorio de Regresión
bar = RandomForestRegressor(n_estimators=200, max_depth=10)

#Entreno el algoritmo con los datos de entrenamiento
bar.fit(X_train, y_train)

#Realizo una predicción con los datos de prueba
Y_pred = bar.predict(X_test)
print('Valor esperado',y_test[:20])
print('Predicción',Y_pred[:20])

#Graficamos los datos de prueba vs las predicciones para verificar la calidad del modelo
X_grid = np.arange(min(X_test), max(X_test), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test)
plt.plot(X_grid, bar.predict(X_grid), color='red', linewidth=3)
plt.show()

#Precision del modelo con los datos de entrenamiento
print('Precisión del modelo', bar.score(X_train, y_train))

# Nuevo valor de entrada (ejemplo: masa corporal = 0.05)
nuevo_valor = np.array([[0.05]])

# Realizar la predicción
prediccion = bar.predict(nuevo_valor)

# Mostrar el resultado de la predicción
print("Predicción para el nuevo valor de masa corporal (0.05):", prediccion[0])

