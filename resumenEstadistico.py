from funciones import carga_datos; import numpy as np
import seaborn as sns;import matplotlib.pyplot as plt

url = 'datosPob.xls'
df = carga_datos(url)

# Mostramos los datos gráficamente.
sns.pairplot(df,height=2.5)
plt.tight_layout()
plt.show()

# Calculamos una serie de valores estadísticos para cada año.
years = ['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
for year in years:
    data = df[year]
    min_value = np.amin(data)
    max_value = np.amax(data)
    mean_value = np.mean(data)
    median_value = np.median(data)
    print(f'Año: {year}')
    print(f'Min: {min_value}')
    print(f'Max: {max_value}')
    print(f'Media: {mean_value}')
    print(f'Mediana: {median_value}')
    print('')

print(df.describe())


# Matriz de Correlación
matriz_correlacion = df.corr()
sns.set(font_scale=1.5)
mapa_calor = sns.heatmap(matriz_correlacion, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15})
plt.show()
