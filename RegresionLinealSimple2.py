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

#Seleccionamos la columna 3 del dataset que es la que contiene la media de habitaciones por vivienda, porque al usar una regresion lineal simple,
#solo hay una variable independiente involucrada y=ax+b donde x es la variable independiente y y es la variable dependiente
X = diabetes.data[:, np.newaxis,2]

#Defino los datos correspondientes a las etiquetas target
y = diabetes.target

#Graficamos los datos correspondientes
plt.scatter(X, y)
plt.xlabel('Indice de masa corporal')
plt.ylabel('Valor medio')
plt.show()

#Separo los datos en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Defino el algoritmo de regresión lineal
lr = LinearRegression()

#Entreno el algoritmo con los datos de entrenamiento
lr.fit(X_train, y_train)

#Devuelve la puntuación del modelo en los datos de prueba
print(f'Exactitud promedio prueba:{lr.score(X_test, y_test)}')

#Realizo una predicción con los datos de prueba
Y_pred = lr.predict(X_test)
print('VALOR ESPERADO',y_test[:30])
print('Prediccion',Y_pred[:30])

#Grafico los valores reales vs las predicciones para verificar la calidad del modelo
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)  # Línea diagonal perfecta
plt.xlabel('Indice de masa corporal')
plt.ylabel('Valor medio')
plt.title('Comparación de valores reales vs predicciones')
plt.show()

print()
print('Datos del modelo regresión lineal:')
print('Valor de la pendiente (a):', lr.coef_)
print('Valor del intercepto (b):', lr.intercept_)
print('La ecuación del modelo es: y =', lr.coef_, 'x +', lr.intercept_)