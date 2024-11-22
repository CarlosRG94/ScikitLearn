from sklearn.datasets import load_iris
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier

#Cargamos los datos del dataset iris
iris = load_iris()
print(iris)

#Verificamos las información del dataset
print('Información del dataset:', iris.keys())

#Verificamos las características del dataset
print('Características del dataset:', iris.DESCR)

#Mostramos el nombre de las características del dataset
print('Características:', iris.feature_names)

#Seleccionamos todas las columnas del dataset
X = iris.data 

#Selección de las etiquetas target
Y = iris.target 

#Contamos las ocurrencias de cada clase
print('Cantidad de cada clase:', np.bincount(Y))
valores, cuentas = np.unique(Y, return_counts=True)

# Mostrar el resultado, que muestra que el 0 es Setosa, el 1 es Versicolour y el 2 es Virginica, y que los datos están balanceados porque hay 50 y 50 y 50
for valor, cuenta in zip(valores, cuentas):
    print(f'Clase {valor}: {cuenta} ocurrencias')

#Separamos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Defino el algoritmo de K vecinos más cercanos
knn = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p = 2)

#Entreno el algoritmo con los datos de entrenamiento
knn.fit(X_train, y_train)

#Realizo una predicción con los datos de prueba
y_pred = knn.predict(X_test)
print('Predicciones:', y_pred)
print('Etiquetas reales:', y_test)

#Verifico la matriz de confusión
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:', matriz)

# 2. Reporte de Clasificación (Precisión, Recall y F1-Score para cada clase)
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 3. Exactitud (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud (Accuracy): {accuracy:.2f}")

#Prueba con un iris de tamaño(5, 3, 1, 0.5)
prueba = np.array([[7.7, 3. , 6.1, 2.3]])
prediccion = knn.predict(prueba)
print(f"Predicción para un iris de tamaño {prueba} es de la clase {prediccion[0]}")
if (prediccion == 0):
    print("El iris es Setosa")
elif (prediccion == 1):
    print("El iris es Versicolour")
elif (prediccion == 2):
    print("El iris es Virginica")


