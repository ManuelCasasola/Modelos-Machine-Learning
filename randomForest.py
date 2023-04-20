from funciones import carga_datos;from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Carga de datos
url = 'datosPob.xls'
df = carga_datos(url)

# Seleccionamos la variable objetivo y las características
y = df['2013']
X = df['2020']

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X.values.reshape(-1, 1), y, test_size=0.2)

# Creamos y entrenamos el modelo de bosque aleatorio
modelo_rf = RandomForestRegressor()
modelo_rf.fit(X_train, y_train)

# Evaluamos el modelo en el conjunto de prueba
score_rf = modelo_rf.score(X_test, y_test)
print("Modelo RandomForest")
print("R^2:", score_rf)