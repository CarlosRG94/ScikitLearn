import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Cargamos los datos del dataset de cáncer de seno
dataset = load_breast_cancer()

# Verifico la información del dataset
print('Información del dataset:', dataset.keys())

# Verifico las características del dataset
print('Características del dataset:', dataset.DESCR)

# Seleccionamos todas las columnas del dataset
X = dataset.data

#Defino las etiquetas target de cada clase
Y = dataset.target 

# Comprobamos cuantos datos hay de cada clase para verificar que los datos están balanceados
print('Cantidad de cada clase:', np.bincount(Y))
muestras, cuentas = np.unique(Y, return_counts=True)

# Mostramos el resultado, que muestra que el 0 es maligno y el 1 es benigno, y que los datos están balanceados porque hay 357 y 212
for muestra, cuenta in zip(muestras, cuentas):
    print(f'Clase {muestra}: {cuenta} muestras')
    
# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Defino el algoritmo de árbol de decisión
algoritmo = DecisionTreeClassifier(criterion = 'entropy', max_depth=80)

#Entreno el modelo con los datos de entrenamiento
algoritmo.fit(X_train, y_train)

#Realizo una predicción con los datos de prueba
y_pred = algoritmo.predict(X_test)
print('Predicciones:', y_pred[:20])
print('Etiqueta reales:', y_test[:20])

#Verifico la matriz de confusión
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:', matriz)

#Calculamos la precisión del modelo
precision = precision_score(y_test, y_pred)
print('Precision:', precision)

#Calculamos la sensibilidad del modelo, que es los verdaderos positivos sobre el total de verdaderos positivos
sensibilidad = recall_score(y_test, y_pred)
print('Sensibilidad:', sensibilidad)


