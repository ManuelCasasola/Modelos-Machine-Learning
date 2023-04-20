from funciones import carga_datos;from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier;from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt;import seaborn as sns
import pandas as pd

url = 'datosPob.xls'
df = carga_datos(url)

# Modelo GradientBoosting

# Seleccionamos la variable objetivo y las características
y = df['2013']
X = df['2020']

# Transforma la variable objetivo en etiquetas de clase discretas
y_class = pd.cut(y, bins=2, labels=[0, 1])

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X.values.reshape(-1, 1), y_class, test_size=0.2)

# Creamos el modelo GradientBoosting
model = GradientBoostingClassifier()

# Entrenamos el modelo
model.fit(X_train, y_train)

# Realizamos predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calculamos la matriz de confusión
mc = confusion_matrix(y_test, y_pred)

# Visualizamos la matriz de confusión y observamos los resultados del modelo de clasificacion.
sns.heatmap(mc, annot=True, fmt='d')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.show()