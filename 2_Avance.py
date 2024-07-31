import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Leer el archivo metadata.csv
metadata = pd.read_csv('data/metadata.csv')

# Preprocesar datos
def preprocesar_datos(df):
    """Preprocesar datos para el modelo"""
    # Convertir etiquetas a números
    le = LabelEncoder()
    df['finding'] = le.fit_transform(df['finding'])
    
    # Seleccionar características y etiquetas
    X = df[['age', 'leukocyte_count', 'neutrophil_count', 'lymphocyte_count']]  # Ejemplo de características
    y = df['finding']
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Calcular porcentaje de exactitud
def porcentaje_exactitud():
    """Calcular porcentaje de exactitud"""
    X_train, X_test, y_train, y_test = preprocesar_datos(metadata)
    
    # Entrenar un modelo simple (por ejemplo, RandomForest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Porcentaje de exactitud: {accuracy * 100:.2f}%")

# Calcular porcentaje de acierto
def porcentaje_acierto():
    """Calcular porcentaje de acierto"""
    X_train, X_test, y_train, y_test = preprocesar_datos(metadata)
    
    # Entrenar un modelo simple (por ejemplo, RandomForest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Porcentaje de acierto:")
    print(f"Porcentaje real: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Mostrar matriz de confusión
def matriz_confusion():
    """Mostrar matriz de confusión"""
    X_train, X_test, y_train, y_test = preprocesar_datos(metadata)
    
    # Entrenar un modelo simple (por ejemplo, RandomForest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Clase 1', 'Clase 2', ..., 'Clase 21'], yticklabels=['Clase 1', 'Clase 2', ..., 'Clase 21'])  # Reemplaza con las etiquetas reales
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusión')
    plt.show()

# Mostrar reporte de clasificación
def reporte_clasificacion():
    """Mostrar reporte de clasificación"""
    X_train, X_test, y_train, y_test = preprocesar_datos(metadata)
    
    # Entrenar un modelo simple (por ejemplo, RandomForest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Asegúrate de que target_names tenga todas las clases
    target_names = [f'Clase {i}' for i in range(21)]  # Reemplaza con nombres reales si los tienes
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("Reporte de Clasificación:")
    print(report)

# Menú interactivo
def mostrar_menu():
    print("\nMenú de opciones:")
    print("1. Calcular porcentaje de exactitud")
    print("2. Calcular porcentaje de acierto")
    print("3. Mostrar matriz de confusión")
    print("4. Mostrar reporte de clasificación")
    print("0. Salir")

if __name__ == "__main__":
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción: ")
        
        if opcion == "1":
            porcentaje_exactitud()
        elif opcion == "2":
            porcentaje_acierto()
        elif opcion == "3":
            matriz_confusion()
        elif opcion == "4":
            reporte_clasificacion()
        elif opcion == "0":
            print("Saliendo del menú.")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")
