![](https://itq.edu.ec/wp-content/uploads/2023/02/Recurso-6.png)
>Instituto Superior Tecnologico Quito

----

| Integrantes  | Docente | Carrera |
| :------------ | |:------------:| | :------------ :| 
| David Catucuamba   | Ing. Sebastian Landazuri | Desarrollo de Software |
| Jonathan Toca     |        **Proyecto**              |  **Nivel**                     |  
|                            | Clasificacion de imagenes Medicas |      5to                  |

----
- El Instituto Superior Tecnologico Quito junto con sus estudiantes forman proyectos por cada modulo conforme la materia por ende en esta ocasion ponemos a disposicion el Proyecto Final de un modelo de IA para clasificar imagenes medicas .  ；

----
**Tecnologias utilizadas para la realizacion del Proyecto**


![vsc](https://github.com/user-attachments/assets/d2d79876-cf33-457f-87a2-61a3db520a38)

- Visual Studio Code (VS Code) es un editor de código fuente desarrollado por Microsoft.  Es gratuito y de código abierto, disponible para Windows, macOS y Linux. VS Code está diseñado para ser ligero y altamente extensible, y ofrece numerosas características que facilitan el desarrollo de software en una amplia variedad de lenguajes de programación
-  La versión más actual de Visual Studio Code (VS Code) es la 1.80, lanzada en junio de 2024.

![python](https://github.com/user-attachments/assets/066436c2-6d34-46c4-8098-67abd2dc489a)

-  Python es un lenguaje de programación de alto nivel, interpretado y de propósito general.
- Python es uno de los lenguajes de programación más populares para la creación de inteligencia artificial (IA) debido a su simplicidad, extensibilidad y la amplia gama de bibliotecas y frameworks disponibles.

----
**Instalacion de Bibliotecas**
-  TensorFlow 2.0
- Keras
- numpy
- pandas
- matplotlib
- seaborn


###Links

----

`<GitHub>` : <https://github.com/JairoDavi/Proyecto>
>GitHub

----
# Funcionamiento del Proyecto
- El proyecto de Clasificación de Imágenes Médicas tiene como objetivo desarrollar un modelo de inteligencia artificial (IA) que pueda clasificar imágenes médicas en diferentes categorías. 

----
## Dataset
- El proyecto utiliza  archivos CSV (metadata.csv)() que contiene datos de imágenes médicas. Este archivo incluye información relevante como la edad del paciente, el conteo de leucocitos, neutrófilos y linfocitos, y la etiqueta de diagnóstico 

----
## Entrenamiento del Modelo
- Se utiliza un clasificador basado en RandomForestClassifier, que es un algoritmo de aprendizaje automático popular para tareas de clasificación. Este modelo se entrena con el conjunto de datos de entrenamiento para aprender a clasificar las imágenes médicas en diferentes categorías.

----
## Evaluación del Modelo
- Una vez entrenado el modelo, se evalúa utilizando el conjunto de datos de prueba. Se calculan métricas de rendimiento como el porcentaje de exactitud, el porcentaje de acierto y la matriz de confusión.



----
#Desarrollo del Proyecto 

###Cantidad de datos por clase

    """Cantidad de datos por clase"""
    print("Cantidad de datos por clase en 'finding':")
    print(df['finding'].value_counts())
    print("\n")
# Distribucion de clases

    """Distribución entre clases (proporciones)"""
    print("Distribución entre clases en 'finding' (proporciones):")
    print(df['finding'].value_counts(normalize=True))
    print("\n")
# Correlacion


    """Correlación entre columnas numéricas"""
    print("Correlación entre columnas numéricas:")
    print(df.corr())
    print("\n")
# Sesgo

    """Sesgo en datos numéricos"""
    print("Sesgo en columnas numéricas:")
    print(df.skew(numeric_only=True))
    print("\n")
	
# Distribucion

    """Distribución de los datos"""
    print("Distribución de las columnas numéricas:")
    print(df.describe())
    print("\n")



#Leer el archivo metadata.csv
metadata = pd.read_csv('data/imagenesmedicas/metadata.csv')

# Preprocesar datos

    """Preprocesar datos para el modelo"""
    # Convertir etiquetas a números
    le = LabelEncoder()
    df['finding'] = le.fit_transform(df['finding'])
    
    # Seleccionar características y etiquetas
    X = df[['age', 'leukocyte_count', 'neutrophil_count', 'lymphocyte_count']]  # Ejemplo de características
    y = df['finding']
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Calcular porcentaje de exactitud

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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Pneumonia', 'Pneumonia'], yticklabels=['No Pneumonia', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusión')
    plt.show()
![matriz de confusion](https://github.com/user-attachments/assets/4c525ff7-03a1-4984-aed9-6edf9d46c002)

# Mostrar reporte de clasificación
def reporte_clasificacion():
    """Mostrar reporte de clasificación"""
    X_train, X_test, y_train, y_test = preprocesar_datos(metadata)
    
    # Entrenar un modelo simple (por ejemplo, RandomForest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, labels=np.unique(y_test))
    print("Reporte de Clasificación:")
    print(report)

----
## Conclusion del Proyecto
- Este proyecto destaca cómo la inteligencia artificial puede mejorar los diagnósticos médicos, y sugiere que futuras mejoras podrían fortalecer aún más su efectividad.
