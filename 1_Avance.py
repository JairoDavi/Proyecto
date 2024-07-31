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

def mostrar_menu():
    print("Menú de opciones:")
    print("1. Resumen de los datos")
    print("2. Primeras filas del DataFrame")
    print("3. Dimensiones del DataFrame")
    print("4. Tipos de datos de cada columna")
    print("5. Cantidad de datos por clase")
    print("6. Distribución entre clases (proporciones)")
    print("7. Correlación entre columnas numéricas")
    print("8. Sesgo en datos numéricos")
    print("9. Distribución de las columnas numéricas")
    print("0. Salir")

if __name__ == "__main__":
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción: ")
        
        if opcion == "1":
            resumen_datos(metadata)
        elif opcion == "2":
            primeras_filas(metadata)
        elif opcion == "3":
            dimensiones(metadata)
        elif opcion == "4":
            tipos_datos(metadata)
        elif opcion == "5":
            cantidad_datos_por_clase(metadata)
        elif opcion == "6":
            distribucion_clases(metadata)
        elif opcion == "7":
            correlacion(metadata)
        elif opcion == "8":
            sesgo(metadata)
        elif opcion == "9":
            distribucion(metadata)
        elif opcion == "0":
            print("Saliendo del menú.")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")


def main():
    print("Este es el contenido de 1_Avance.py")

if __name__ == "__main__":
    main()
