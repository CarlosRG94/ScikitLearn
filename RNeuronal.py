import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import keras 
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter

#Cargamos el dataset y mostramos las 5 primeras filas
dataset = pd.read_csv('dataset\Churn_Modelling.csv')
print(dataset.describe(include='all'))
print(dataset.head())

#Seleccionamos las columnas desde la 3 hasta la 12 porque las 3 primeras son datos que no necesitamos y la 13 es el target
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
print(X)
print(y)

#Contamos los targets que tienen el dataset y vemos que está desbalanceado
conteo = dataset['Exited'].value_counts()
print(conteo)

#Transformamos las columnas que tienen los tados en letras
labelencoder_X1 = LabelEncoder()
X[:,1] = labelencoder_X1.fit_transform(X[:,1])
X[:,2] = labelencoder_X1.fit_transform(X[:,2])
print(X)

#Transformamos la columna de paises en 3 columnas damis cada una representa a un pais de los 3 que tenemos
#Borramos la primera columna que por descarte cuando las otras dos sean 0 esta sera la primera aqui por ejemplo sera francia
transformer = ColumnTransformer(
    transformers =[
        ("Churn_Modelling",OneHotEncoder(categories='auto'),[1])
        ], remainder = 'passthrough'
)
X = transformer.fit_transform(X)
print(X)
X = X[:,1:]
print (X.shape)
#Separamos la data en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Aplicar SMOTE para sobremuestrear la clase minoritaria en el conjunto de entrenamiento
smote = SMOTE(random_state=0)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Distribución después de SMOTE:", Counter(y_train_smote))


#Escalamos(normalizamos) los datos para que no haya datos con valores tan distintos como 100000 y 1
sX = StandardScaler()
X_train_smote = sX.fit_transform(X_train_smote)
X_test = sX.transform(X_test)
print(X_train_smote)

#Construimos nuestra red neuronal
#Lo primero que tenemos que hacer es crear un clasificador, que va a ser como el constructor que encapsule el modelo construyendo capa por capa
clf = keras.Sequential()

#Primera capa
#Se añade un capa pera prevenir el sobreajuste(hace que se use un menor porcentaje de neuronas) 
clf.add(keras.layers.Dense(units = 32,kernel_initializer = "uniform", activation = "relu", input_dim = 11))
clf.add(keras.layers.Dropout(0.1))

clf.add(keras.layers.Dense(units = 64,kernel_initializer = "uniform", activation = "relu"))
clf.add(keras.layers.Dropout(0.1))  

clf.add(keras.layers.Dense(units = 64,kernel_initializer = "uniform", activation = "relu"))
clf.add(keras.layers.Dropout(0.1))

clf.add(keras.layers.Dense(units = 64,kernel_initializer = "uniform", activation = "relu"))
clf.add(keras.layers.Dropout(0.1))

clf.add(keras.layers.Dense(units = 64,kernel_initializer = "uniform", activation = "relu"))
clf.add(keras.layers.Dropout(0.1))

#Tercera capa
clf.add(keras.layers.Dense(units = 1,kernel_initializer = "uniform", activation = "sigmoid"))

#Compilador de la RNA
clf.compile(optimizer= keras.optimizers.Adam(learning_rate = 0.0005), loss= "binary_crossentropy", metrics=["accuracy"])
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_smote), y=y_train_smote)
class_weight_dict = dict(enumerate(class_weights))

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

clf.fit(X_train_smote, y_train_smote, batch_size=32, epochs=120, validation_data=(X_test, y_test), callbacks=[early_stopping])

#Creamos un metodo para mostrar graficamente la matriz de confusión

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else: 
        print('Confusion matrix, without normalization')
        
    print(cm)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#Predecimos los datos con nuestros datos de prueba
y_pred = clf.predict(X_test)
print(y_pred[:30])

y_pred = (y_pred>0.5)
y_test = (y_test>0.5)
 
# Contar los True
num_true = sum(y_test)

# Contar los False
num_false = len(y_test) - num_true

print(f"Número de True: {num_true}")
print(f"Número de False: {num_false}") 

# Contar los True
num_truepred = sum(y_pred)

# Contar los False
num_falsepred = len(y_pred) - num_truepred

print(f"Número de Truepred: {num_truepred}")
print(f"Número de Falsepred: {num_falsepred}") 

print(y_pred[:30])
print(y_test[:30])

#Calculamos la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

#Calculamos la exactitud del modelo, que es el número de predicciones correctas realizadas sobre el total
exactitud = accuracy_score(y_test, y_pred)
print('Exactitud:', exactitud)

#Mostramos la grafica
plot_confusion_matrix(cm, ['Se queda ', 'Se va'], title='Matriz de confusión')


