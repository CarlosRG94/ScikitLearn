import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.preprocessing import StandardScaler

#Se importa el dataset de cáncer de seno
dataset = load_breast_cancer()
print(dataset)

#Verifico la información contenida en el dataset
print('Información del dataset:', dataset.keys())

#Verifico las características del dataset
print('Características del dataset:', dataset.DESCR)

#Divido el dataset en características y etiquetas
X = dataset.data 
Y = dataset.target

#Contamolas ocurrencias de cada clase
print('Cantidad de cada clase:', np.bincount(Y))
valores, cuentas = np.unique(Y, return_counts=True)
for valor, cuenta in zip(valores, cuentas):
    print(f'Clase {valor}: {cuenta} ocurrencias')

#Separamos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Defino el algoritmo de Naive Bayes Gaussiano
algoritmo = GaussianNB(priors = None, var_smoothing = 1e-09)

#Entreno el algoritmo con los datos de entrenamiento
algoritmo.fit(X_train, y_train)

#Realizo una predicción con los datos de prueba
y_pred = algoritmo.predict(X_test)
print('Predicciones:', y_pred[:20])
print('Etiquetas reales:', y_test[:20])

#Verifico la matriz de confusión
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:', matriz)

# Calculamos la precisión del modelo, que es los verdaderos positivos sobre el total de predicciones positivas
precision = precision_score(y_test, y_pred)
print('Precisión:', precision)


