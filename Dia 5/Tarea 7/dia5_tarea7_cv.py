"""
Docstring general del script:

Este script construye y entrena un modelo de clasificación binaria de imágenes
utilizando transfer learning con la arquitectura ResNet50 preentrenada en ImageNet.

El flujo principal incluye:
- Preprocesamiento y aumento de datos mediante ImageDataGenerator
- Carga de datos desde directorios estructurados
- Uso de ResNet50 como base convolucional congelada
- Adición de capas superiores para clasificación binaria
- Entrenamiento del modelo
- Evaluación en datos de validación

Componentes principales:
- Generadores de datos de entrenamiento y validación
- Modelo base preentrenado (ResNet50)
- Capas personalizadas de pooling y salida
- Compilación y entrenamiento del modelo
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# --------------------------------------------------
# Configuración del generador de datos con aumento y normalización
# --------------------------------------------------

# ImageDataGenerator aplica transformaciones en tiempo real para aumentar la diversidad del dataset
datagen = ImageDataGenerator(
    rescale=1./255,              # Normaliza los valores de píxeles al rango [0, 1]
    rotation_range=20,           # Rotación aleatoria de imágenes hasta 20 grados
    width_shift_range=0.2,       # Desplazamiento horizontal aleatorio
    height_shift_range=0.2,      # Desplazamiento vertical aleatorio
    shear_range=0.2,             # Transformación de corte (shear)
    zoom_range=0.2,              # Zoom aleatorio en imágenes
    horizontal_flip=True,        # Volteo horizontal aleatorio
    validation_split=0.2         # División automática entre entrenamiento y validación
)

# --------------------------------------------------
# Carga del conjunto de entrenamiento desde directorio
# --------------------------------------------------

train_data = datagen.flow_from_directory(
    "PATH To DATASET",          # Ruta del dataset organizado por clases
    target_size=(224, 224),     # Redimensiona imágenes a 224x224 píxeles
    batch_size=32,              # Número de imágenes por lote
    class_mode="binary",        # Clasificación binaria (dos clases)
    subset="training"           # Subconjunto de entrenamiento
)

# --------------------------------------------------
# Carga del conjunto de validación desde directorio
# --------------------------------------------------

val_data = datagen.flow_from_directory(
    "PATH To DATASET",          # Misma ruta del dataset original
    target_size=(224, 224),     # Redimensionamiento consistente
    batch_size=32,              # Tamaño del lote
    class_mode="binary",        # Clasificación binaria
    subset="validation"         # Subconjunto de validación
)

# --------------------------------------------------
# Carga del modelo base preentrenado ResNet50
# --------------------------------------------------

# Se carga ResNet50 con pesos entrenados en ImageNet
# include_top=False elimina la capa fully connected original
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# --------------------------------------------------
# Congelación de capas del modelo base
# --------------------------------------------------

# Se deshabilita el entrenamiento de las capas convolucionales preentrenadas
for layer in base_model.layers:
    layer.trainable = False

# --------------------------------------------------
# Construcción de la cabeza del modelo (clasificador)
# --------------------------------------------------

# GlobalAveragePooling reduce la dimensionalidad de los mapas de características
x = GlobalAveragePooling2D()(base_model.output)

# Capa densa final con activación sigmoide para clasificación binaria
output = Dense(1, activation="sigmoid")(x)

# Definición del modelo completo combinando base + clasificador
model = Model(inputs=base_model.input, outputs=output)

# --------------------------------------------------
# Compilación del modelo
# --------------------------------------------------

model.compile(
    optimizer="adam",                    # Optimizador Adam
    loss="binary_crossentropy",          # Función de pérdida para clasificación binaria
    metrics=['accuracy']                 # Métrica de evaluación
)

# --------------------------------------------------
# Entrenamiento del modelo
# --------------------------------------------------

history = model.fit(
    train_data,              # Datos de entrenamiento
    validation_data=val_data,# Datos de validación
    epochs=10                # Número de épocas de entrenamiento
)

# --------------------------------------------------
# Evaluación del modelo
# --------------------------------------------------

val_loss, val_accuracy = model.evaluate(val_data)

# Impresión del resultado final de validación
print(f"Validation Exactitud {val_accuracy}")