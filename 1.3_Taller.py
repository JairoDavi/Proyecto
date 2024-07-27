import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Cargar el dataset
data = pd.read_csv('data/metadata.csv')

def matriz_densidad(data):
    print("Matriz de Densidad:")
    sns.pairplot(data, kind="kde")
    plt.suptitle("Matriz de Densidad")
    plt.show()
    print("La matriz de densidad muestra la distribución de cada variable del dataset y las relaciones bivariadas entre ellas utilizando estimaciones de densidad kernel (KDE).")

def matriz_box_plot(data):
    print("Matriz de Box Plot:")
    numeric_cols = data.select_dtypes(include=np.number).columns
    data[numeric_cols].plot(kind='box', subplots=True, layout=(len(numeric_cols)//3 + 1, 3), figsize=(15, 10), sharex=False, sharey=False)
    plt.suptitle("Matriz de Box Plot")
    plt.show()
    print("La matriz de box plot muestra la distribución de cada variable numérica del dataset, incluyendo la mediana, cuartiles y posibles outliers.")

def matriz_correlacion(data):
    print("Matriz de Correlación:")
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Matriz de Correlación")
    plt.show()
    print("La matriz de correlación muestra las relaciones lineales entre las variables numéricas del dataset. Los valores cercanos a 1 o -1 indican una fuerte correlación positiva o negativa, respectivamente.")

def matriz_dispersión(data):
    print("Matriz de Dispersión:")
    sns.pairplot(data)
    plt.suptitle("Matriz de Dispersión")
    plt.show()
    print("La matriz de dispersión muestra gráficos de dispersión para cada par de variables numéricas del dataset, permitiendo observar relaciones y patrones.")

def describir_columnas(data):
    print("\nDescripción de las columnas del dataset:")
    print(data.describe(include='all'))

def resultados_obtenidos():
    print("\nResultados obtenidos:")
    print("Los gráficos generados proporcionan información visual sobre la distribución y las relaciones entre las variables en el dataset, ayudando a identificar patrones y posibles outliers.")

# Ejecutar funciones
#matriz_densidad(data)
matriz_box_plot(data)
#matriz_correlacion(data)
#matriz_dispersión(data)
#describir_columnas(data)
resultados_obtenidos()
