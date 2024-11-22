from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score

#Importamos los datos de la librería de scikit-learn
dataset = load_breast_cancer()
print(dataset)

#Verifico la información del dataset
print('Información del dataset:',dataset.keys())


#Verifico las características del dataset
print('Características del dataset:',dataset.DESCR)

#Seleccionamos todas las columnas del dataset
X = dataset.data

#Defino los datos correspondientes a las etiquetas target
Y = dataset.target

# Contar las ocurrencias de cada clase
print('Cantidad de cada clase:', np.bincount(Y))
valores, cuentas = np.unique(Y, return_counts=True)

# Mostrar el resultado, que muestra que el 0 son malignos y el 1 son benignos, y que los datos están balanceados porque hay 212 y 357
for valor, cuenta in zip(valores, cuentas):
    print(f'Clase {valor}: {cuenta} ocurrencias')
    
# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Escalamos todos los datos
escalar = StandardScaler()
X_train = escalar.fit_transform(X_train)
X_test = escalar.transform(X_test)
    
#Defino el algoritmo de regresión logística
algoritmo = LogisticRegression()

# Entreno el algoritmo con los datos de entrenamiento
algoritmo.fit(X_train, y_train)

# Realizo una predicción con los datos de prueba
y_pred = algoritmo.predict(X_test)
print('Predicciones:', y_pred[:20])
print('Datos reales:', y_test[:20])

#Todas estas verificaciones se suelen hacer cuando los datos no estan balanceados, aunque en este caso los datos están balanceados, pero vamos a confirmar
#Verifico la matriz de confusión
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:', matriz)

# Calculamos la precisión del modelo, que es los verdaderos positivos sobre el total de predicciones positivas
precision = precision_score(y_test, y_pred)
print('Precisión:', precision)
#Calculamos la exactitud del modelo, que es el número de predicciones correctas realizadas sobre el total
exactitud = accuracy_score(y_test, y_pred)
print('Exactitud:', exactitud)

#Calculamos la sensibilidad del modelo, que es los verdaderos positivos sobre el total de verdaderos positivos
sensibilidad = recall_score(y_test, y_pred)
print('Sensibilidad:', sensibilidad)

#Calculamos el puntaje F1 del modelo, que es la media armónica de la precisión y la sensibilidad
f1 = f1_score(y_test, y_pred)
print('Puntaje F1:', f1)

#Calculamos la curva ROC AUC del modelo
roc_auc = roc_auc_score(y_test, y_pred)
print('Curva ROC AUC:', roc_auc)
