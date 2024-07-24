import pandas as pd

# Leer el archivo metadata.csv
metadata = pd.read_csv('data/metadata.csv')

def resumen_datos(df):
    """Resumen de los datos"""
    print("Resumen de los datos:")
    print(df.describe(include='all'))
    print("\n")

def primeras_filas(df):
    """Primeras filas del DataFrame"""
    print("Primeras filas del DataFrame:")
    print(df.head())
    print("\n")

def dimensiones(df):
    """Dimensiones del DataFrame"""
    print("Dimensiones del DataFrame:")
    print(df.shape)
    print("\n")

def tipos_datos(df):
    """Tipos de datos de cada columna"""
    print("Tipos de datos de cada columna:")
    print(df.dtypes)
    print("\n")

def cantidad_datos_por_clase(df):
    """Cantidad de datos por clase"""
    print("Cantidad de datos por clase en 'finding':")
    print(df['finding'].value_counts())
    print("\n")

def distribucion_clases(df):
    """Distribución entre clases (proporciones)"""
    print("Distribución entre clases en 'finding' (proporciones):")
    print(df['finding'].value_counts(normalize=True))
    print("\n")

def correlacion(df):
    """Correlación entre columnas numéricas"""
    print("Correlación entre columnas numéricas:")
    print(df.corr())
    print("\n")

def sesgo(df):
    """Sesgo en datos numéricos"""
    print("Sesgo en columnas numéricas:")
    print(df.skew(numeric_only=True))
    print("\n")

def distribucion(df):
    """Distribución de los datos"""
    print("Distribución de las columnas numéricas:")
    print(df.describe())
    print("\n")

# Ejecutar las funciones
#resumen_datos(metadata)
#primeras_filas(metadata)
dimensiones(metadata)
#tipos_datos(metadata)
#cantidad_datos_por_clase(metadata)
#distribucion_clases(metadata)
#correlacion(metadata)
#sesgo(metadata)
#distribucion(metadata)
