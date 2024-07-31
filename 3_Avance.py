import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def load_data():
    """Carga datos desde un archivo CSV y limpia la columna 'finding'"""
    df = pd.read_csv('data/metadata.csv')
    
    # Limpieza de datos
    df['finding'] = df['finding'].str.strip()  # Elimina espacios en blanco
    df['finding'] = df['finding'].replace({'': np.nan})  # Reemplaza cadenas vacías por NaN
    
    # Imprimir los valores únicos en la columna 'finding'
    print("Valores únicos en la columna 'finding':")
    print(df['finding'].unique())
    
    X = df.drop('finding', axis=1).values
    Y = df['finding'].values
    
    return X, Y

def preprocesar_datos(df):
    """Preprocesar datos para el modelo"""
    le = LabelEncoder()
    df['finding'] = le.fit_transform(df['finding'])
    return df

def density_matrix(df):
    """Generar la matriz de densidad"""
    print("Generando la matriz de densidad...")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    df.plot(ax=ax, kind='density', subplots=True, layout=(6, 6), sharex=False)
    plt.tight_layout()
    plt.show()
    print("La matriz de densidad muestra la distribución de cada variable en el dataset, permitiendo observar la forma y dispersión de los datos.")

def box_plot_matrix(df):
    """Generar la matriz de box plot"""
    print("Generando la matriz de box plot...")
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    df.plot(ax=ax, kind='box', subplots=True, layout=(6, 6), sharex=False)
    plt.tight_layout()
    plt.show()
    print("La matriz de box plot proporciona una vista de la dispersión y posibles valores atípicos de cada variable, mostrando la mediana y los cuartiles.")

def correlation_matrix(df):
    """Generar la matriz de correlación"""
    print("Generando la matriz de correlación...")
    numeric_data = df.select_dtypes(include=[np.float64, np.int64])
    correlations = numeric_data.corr(method='pearson')
    plt.figure(figsize=(12, 12))
    plt.title('Matriz de correlación')
    sns.heatmap(correlations, vmax=1, square=True, annot=True, cmap='viridis')
    plt.show()
    print("La matriz de correlación muestra la relación lineal entre las variables. Los valores cercanos a 1 o -1 indican una fuerte correlación positiva o negativa, respectivamente.")

def dispersion_matrix(df):
    """Generar la matriz de dispersión"""
    print("Generando la matriz de dispersión...")
    numeric_data = df.select_dtypes(include=[np.float64, np.int64])
    plt.rcParams['figure.figsize'] = (12, 12)
    scatter_matrix(numeric_data)
    plt.show()
    print("La matriz de dispersión permite observar la relación entre cada par de variables en el dataset, mostrando patrones y posibles correlaciones visuales.")



def mae():
    """Calcular el Error Medio Absoluto (MAE)"""
    X, Y = load_data()

    # Imprimir los primeros valores de Y antes de la conversión
    print("Valores de Y antes de la conversión:")
    print(Y[:10])  # Imprime los primeros 10 valores de Y

    # Convertir 'N' y 'Y' a 0 y 1 respectivamente
    conversion_dict = {'N': 0, 'Y': 1}
    Y = np.array([conversion_dict.get(val, np.nan) for val in Y])

    # Imprimir los primeros valores de Y después de la conversión
    print("Valores de Y después de la conversión:")
    print(Y[:10])  # Imprime los primeros 10 valores de Y

    # Verificar y manejar valores NaN en Y
    if np.isnan(Y).all():
        raise ValueError("Todos los valores en Y son NaN después de la conversión.")

    # Manejar valores faltantes en X
    col_nan = np.isnan(X).all(axis=0)
    X = X[:, ~col_nan]

    # Verificar si X aún tiene columnas
    if X.shape[1] == 0:
        raise ValueError("Todas las columnas de características en X han sido eliminadas debido a valores NaN.")

    # Imputar valores faltantes en X
    imputer_X = SimpleImputer(strategy='mean')
    X = imputer_X.fit_transform(X)

    # Imputar valores faltantes en Y
    imputer_Y = SimpleImputer(strategy='most_frequent')
    Y = imputer_Y.fit_transform(Y.reshape(-1, 1)).ravel()

    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

    # Entrenar el modelo de Regresión Logística
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, Y_train)

    # Predecir y calcular MAE
    predicted = model.predict(X_test)
    mae_value = mean_absolute_error(Y_test, predicted)
    print(f"MAE: {mae_value:.2f}")

def mse():
    """Calcular el Error Cuadrático Medio (MSE)"""
    X, Y = load_data()

    # Convertir 'N' y 'Y' a 0 y 1 respectivamente
    conversion_dict = {'N': 0, 'Y': 1}
    Y = np.array([conversion_dict.get(val, np.nan) for val in Y])

    # Verificar y manejar valores NaN en Y
    if np.isnan(Y).all():
        raise ValueError("Todos los valores en Y son NaN después de la conversión.")

    # Manejar valores faltantes en X
    col_nan = np.isnan(X).all(axis=0)
    X = X[:, ~col_nan]

    # Verificar si X aún tiene columnas
    if X.shape[1] == 0:
        raise ValueError("Todas las columnas de características en X han sido eliminadas debido a valores NaN.")

    # Imputar valores faltantes en X
    imputer_X = SimpleImputer(strategy='mean')
    X = imputer_X.fit_transform(X)

    # Imputar valores faltantes en Y
    imputer_Y = SimpleImputer(strategy='most_frequent')
    Y = imputer_Y.fit_transform(Y.reshape(-1, 1)).ravel()

    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

    # Entrenar el modelo de Regresión Logística
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, Y_train)

    # Predecir y calcular MSE
    predicted = model.predict(X_test)
    mse_value = mean_squared_error(Y_test, predicted)
    print(f"MSE: {mse_value:.2f}")

def r2():
    """Calcular el R2 Score"""
    X, Y = load_data()

    # Convertir 'N' y 'Y' a 0 y 1 respectivamente
    conversion_dict = {'N': 0, 'Y': 1}
    Y = np.array([conversion_dict.get(val, np.nan) for val in Y])

    # Verificar y manejar valores NaN en Y
    if np.isnan(Y).all():
        raise ValueError("Todos los valores en Y son NaN después de la conversión.")

    # Manejar valores faltantes en X
    col_nan = np.isnan(X).all(axis=0)
    X = X[:, ~col_nan]

    # Verificar si X aún tiene columnas
    if X.shape[1] == 0:
        raise ValueError("Todas las columnas de características en X han sido eliminadas debido a valores NaN.")

    # Imputar valores faltantes en X
    imputer_X = SimpleImputer(strategy='mean')
    X = imputer_X.fit_transform(X)

    # Imputar valores faltantes en Y
    imputer_Y = SimpleImputer(strategy='most_frequent')
    Y = imputer_Y.fit_transform(Y.reshape(-1, 1)).ravel()

    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

    # Entrenar el modelo de Regresión Logística
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, Y_train)

    # Predecir y calcular R2 Score
    predicted = model.predict(X_test)
    r2_value = r2_score(Y_test, predicted)
    print(f"R2: {r2_value:.2f}")

def menu():
    """Mostrar menú de opciones"""
    while True:
        print("\nMenú:")
        print("1. Mostrar la matriz de densidad")
        print("2. Mostrar la matriz de box plot")
        print("3. Mostrar la matriz de correlación")
        print("4. Mostrar la matriz de dispersión")
        print("5. Calcular MAE")
        print("6. Calcular MSE")
        print("7. Calcular R2")
        print("0. Salir")

        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            df = pd.read_csv('data/metadata.csv')  # Asegúrate de ajustar el nombre del archivo
            density_matrix(df)
        elif opcion == "2":
            df = pd.read_csv('data/metadata.csv')  # Asegúrate de ajustar el nombre del archivo
            box_plot_matrix(df)
        elif opcion == "3":
            df = pd.read_csv('data/metadata.csv')  # Asegúrate de ajustar el nombre del archivo
            correlation_matrix(df)
        elif opcion == "4":
            df = pd.read_csv('data/metadata.csv')  # Asegúrate de ajustar el nombre del archivo
            dispersion_matrix(df)
        elif opcion == "5":
            mae()
        elif opcion == "6":
            mse()
        elif opcion == "7":
            r2()
        elif opcion == "0":
            print("Saliendo...")
            break
        else:
            print("Opción no válida, por favor selecciona una opción válida.")

# Ejecutar menú
menu()
