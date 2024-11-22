import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

#cargar la data con x e y de los balanceados que son los dos arreglos que contiene
X = np.load('balanceados.npz')['X']
Y = np.load('balanceados.npz')['Y']

#imprimir cantidad del arreglo
print(X.shape)
print(Y.shape)

#partición 60 train 40 rest
x_train, x_rest, y_train, y_rest = train_test_split(X, Y, test_size=0.4, random_state=123)
#partición rest en 2 mitades
x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.5, random_state=321)

#Verificación 
print('Tamaños:')
print('\tDataset original:', X.shape, Y.shape)
print('\tEntrenamiento:', x_train.shape, y_train.shape)
print('\tValidación:', x_val.shape, y_val.shape)
print('\tPrueba:', x_test.shape, y_test.shape)

print('Proporciones categorías (0s/1s):')
print(f'\tDataset original:{np.sum(Y==0)/len(Y)}/{np.sum(Y==1)/len(Y)}')
print(f'\tEntrenamiento:{np.sum(y_train==0)/len(y_train)}/{np.sum(y_train==1)/len(y_train)}')
print(f'\tValidación:{np.sum(y_val==0)/len(y_val)}/{np.sum(y_val==1)/len(y_val)}')
print(f'\tPrueba:{np.sum(y_test==0)/len(y_test)}/{np.sum(y_test==1)/len(y_test)}')

#valores minimos y máximos
print(f'x_train: {x_train.min(axis=0)}/{x_train.max(axis=0)}')
print(f'x_val: {x_val.min(axis=0)}/{x_val.max(axis=0)}')
print(f'x_train: {x_test.min(axis=0)}/{x_test.max(axis=0)}')

#Instanciar pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(-1,1))),
    ('classifier', RandomForestClassifier())
])

#Entrenar la pipeline
pipeline.fit(x_train, y_train)

#Evaluar la pipeline
print(pipeline.score(x_test, y_test))

#y generar predicciones
print(pipeline.predict(x_test))