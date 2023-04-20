from funciones import carga_datos;from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB;from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt;import pandas as pd

url = 'datosPob.xls'
df = carga_datos(url)

# Modelo Naive Bayes

# Seleccionamos la variable objetivo y las características
y = df['2013']
X = df['2020']

# Divide los valores continuos en 2 intervalos y asigna una etiqueta de clase a cada intervalo
y_class = pd.cut(y, bins=2, labels=[0, 1])

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X.values.reshape(-1, 1), y_class, test_size=0.2)

# Creamos el modelo
model = GaussianNB()

# Entrenamos el modelo
model.fit(X_train, y_train)

# Realizamos predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calculamos la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Visualizamos la curva ROC
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.show()