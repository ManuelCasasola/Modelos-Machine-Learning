from sklearn.model_selection import train_test_split;import tensorflow as tf
from tensorflow import keras
from funciones import carga_datos

# Carga de datos
url = 'datosPob.xls'
df = carga_datos(url)

# Seleccionamos la variable objetivo y las características
y = df['2013']
X = df['2020']

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X.values.reshape(-1, 1), y, test_size=0.2)

# Modelo Red Neuronal TensorFlow
# Creamos una instancia de un modelo secuencial
modelo_tf=tf.keras.Sequential()

# Agregamos capas densas a la red neuronal
modelo_tf.add(tf.keras.layers.Dense(32,activation='relu',input_shape=(X_train.shape[1],)))
modelo_tf.add(tf.keras.layers.Dense(16,activation='relu'))
modelo_tf.add(tf.keras.layers.Dense(1))

# Compilamos el modelo
modelo_tf.compile(optimizer='adam',loss='mean_squared_error')

# Entrenamos el modelo
training=modelo_tf.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test))

# Evaluamos el modelo
loss=modelo_tf.evaluate(X_test,y_test)
print(f'Loss: {loss}')
