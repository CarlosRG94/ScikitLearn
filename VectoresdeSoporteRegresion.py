import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVR

#Cargamos los datos del dataset real estate.csv con numpy.loadtxt
data = np.loadtxt('dataset\Real estate.csv', delimiter=',', skiprows=1)

X = data[:, 1:-1]
Y = data[:, -1]

print("Dimensiones de X: ", X.shape)
print("Dimensiones de Y: ", Y.shape)
print("Primeras 5 filas de X:",X[:5])
print("Primeras 5 filas de Y", Y[:5])

#Seleccionamos solo la columna 2 del dataset que es la edad de la casa
X_svr = X[:,np.newaxis,1]

#Renombramos los targets que son los precios de las casas
Y_svr = Y

#Graficamos los datos correspondientes
plt.scatter(X_svr,Y_svr)
plt.xlabel('Edad de la casa')
plt.ylabel('Precio de la casa')
plt.show()

#Separo los datos en entrenamiento y prueba para probar los algoritmos
X_train, X_test, Y_train, Y_test = train_test_split(X_svr, Y_svr, test_size=0.2, random_state=42)

#Defino el algoritmo de Support Vector Regression, le incluimos algunos parámetros aunque si se deja vacio automáticamente se eligen los mejores
svr = SVR(kernel='linear', C=1.0, epsilon=0.2)

#Entreno el algoritmo con los datos de entrenamiento
svr.fit(X_train, Y_train)

#Realizo una predicción con los datos de prueba
Y_pred = svr.predict(X_test)
print('Predicción',Y_pred[:15])
print('Valor esperado',Y_test[:15])

#Grafico los valores reales vs las predicciones para verificar la calidad del modelo
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.show()

print('Datos del modelo de Support Vector Regression:')
print('Precision del modelo:', svr.score(X_train, Y_train))

# Nuevo valor de entrada (ejemplo: edad de la casa 15 años)
nuevo_valor = np.array([[15]])

# Realizar la predicción
prediccion = svr.predict(nuevo_valor)

# Mostrar el resultado de la predicción
print("Predicción para el nuevo valor de masa corporal (0.05):", prediccion[0])
