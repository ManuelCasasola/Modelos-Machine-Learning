import pandas as pd;import numpy as np;import seaborn as sns;import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ;from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor;from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV;from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error;from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras


def carga_datos(url):
    """Carga los datos desde una URL y elimina los valores nulos."""
    df = pd.read_excel(url)
    df = df.fillna(df.mean())
    return df

def fit_model(X, y):
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    regressor = DecisionTreeRegressor()
    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}
    modelo = LinearRegression(fit_intercept=True)
    scoring_fnc = make_scorer(modelo)
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)
    grid = grid.fit(X, y)
    return grid.best_estimator_




