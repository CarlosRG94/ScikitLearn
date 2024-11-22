import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

#Cargamos la data con la libreria numpy
data = np.loadtxt('dataset\winequality-red.csv',delimiter=';',skiprows=1)

X = data[:, :-1]
Y= data[:,-1]

print("Dimensiones de X: ", X.shape)
print("Dimensiones de Y: ", Y.shape)
print("Primeras 5 filas de X:",X[:5])
print("Primeras 5 filas de Y", Y[:5])

#Seleccionamos solo la columna 1 del dataset
X_p = X[:,np.newaxis, 0]

#Renombramos los targets
Y_p = Y

#Graficamos los datos correspondientes
plt.scatter(X_p,Y_p)
plt.show()

#Separamos los datos de entrenamiento y prueba
X_train_p, X_test_p, Y_train_p, Y_test_p = train_test_split(X_p,Y_p, test_size=0.2, random_state=42)

#Se define el grado del polinomio
poli_reg = PolynomialFeatures(degree=2)

#Se transforman las características existentes en características de mayor grado
X_train_poli = poli_reg.fit_transform(X_train_p)
X_test_poli = poli_reg.fit_transform(X_test_p)

#Estandarización de los datos
scaler = StandardScaler()
X_train_poli = scaler.fit_transform(X_train_poli)
X_test_poli = scaler.transform(X_test_poli)

#Defino el algoritmo a utilizar
pr = LinearRegression()

#Entreno el modelo
pr.fit(X_train_poli, Y_train_p)

#Realizamos las predicciones
Y_pred_pr = pr.predict(X_test_poli)
print('Resultado esperado: ', Y_test_p[:20])
print('Resultado de la predicción: ', Y_pred_pr[:20])

#Graficamos los datos junto con el modelo
plt.scatter(X_test_p, Y_test_p)
plt.plot(X_test_p, Y_pred_pr, color='red', linewidth=3)
plt.show()

print('Datos del modelo de regresión polinomial')
print('Valor de la pendiente o el coeficiente"a":',pr.coef_)
print('Valor de la intersección o coeficiente "b":',pr.intercept_)
print('Precisión del modelo:', pr.score(X_train_poli, Y_train_p))


