from funciones import carga_datos;from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression;from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV;from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer;from sklearn.metrics import mean_squared_error


url = 'datosPob.xls'
df = carga_datos(url)

# Modelo de Regresión

# Seleccionamos la variable objetivo y las características
y = df['2013']
X = df['2020']

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X.values.reshape(-1, 1), y, test_size=0.2)

# Creamos y entrenamos el modelo de regresión lineal
modelo = LinearRegression(fit_intercept=True)
modelo.fit(X_train, y_train)
print(modelo.intercept_)

# Evaluamos el modelo en el conjunto de prueba
score = modelo.score(X_test, y_test)
print("Modelo de Regresión Lineal")
print("R^2:", score)

def fit_model(X, y):
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    regressor = DecisionTreeRegressor()
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}
    scoring_fnc = make_scorer(score)
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)
    grid = grid.fit(X, y)
    return grid.best_estimator_

best_model = fit_model(X_train, y_train)

# Hacemos predicciones en el conjunto de prueba
y_pred = best_model.predict(X_test)

# Calculamos el error cuadrático medio en el conjunto de prueba
ecm = mean_squared_error(y_test, y_pred)
print("Error Cuadrático Medio")
print(f'ECM: {ecm}')