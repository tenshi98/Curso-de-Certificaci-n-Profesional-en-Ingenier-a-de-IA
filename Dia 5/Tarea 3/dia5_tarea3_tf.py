"""
DOCSTRING:
Este script implementa transfer learning usando el modelo preentrenado MobileNetV2 en TensorFlow/Keras
para una tarea de clasificación multiclase con 5 clases.

El flujo general del código es:

1. Carga del modelo MobileNetV2 preentrenado en ImageNet sin la capa superior (include_top=False).
2. Congelación de todas las capas del modelo base para evitar su entrenamiento inicial.
3. Construcción de una nueva cabeza de clasificación compuesta por:
   - GlobalAveragePooling2D para reducir mapas de características a un vector.
   - Capa Dense con activación softmax para clasificación de 5 clases.
4. Definición de un generador de datos con aumento de imágenes y división automática de validación.
5. Carga de imágenes desde directorios organizados por clase mediante flow_from_directory.
6. Compilación del modelo con Adam y función de pérdida categorical_crossentropy.
7. Entrenamiento del modelo utilizando los generadores de datos.

Parámetros esperados:
- PATH_TO_DATASET: ruta al dataset estructurado en carpetas por clase.

Retorno:
- Historial de entrenamiento (history) con métricas de accuracy y loss.

Excepciones:
- Puede fallar si el dataset no está organizado correctamente o si las clases no coinciden con el output de 5 clases.
"""

# Importación de librerías necesarias
import tensorflow as tf

# Carga del modelo MobileNetV2 preentrenado
from tensorflow.keras.applications import MobileNetV2

# Capas necesarias para la cabeza de clasificación
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Construcción del modelo funcional
from tensorflow.keras.models import Model

# Corrector: ImageDataGenerator (no ImageDatagenerator)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Carga del modelo base sin capa superior
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Congelación de todas las capas del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Construcción de la cabeza de clasificación
x = GlobalAveragePooling2D()(base_model.output)

# Capa de salida para 5 clases
output = Dense(5, activation="softmax")(x)

# Modelo final
model = Model(inputs=base_model.input, outputs=output)


# Definición de aumento de datos y partición de validación
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shape_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)


# Generador de entrenamiento
train_data = datagen.flow_from_directory(
    "PATH_TO_DATASET",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

# Generador de validación
val_data = datagen.flow_from_directory(
    "PATH_TO_DATASET",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)


# Compilación del modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Entrenamiento del modelo
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    steps_per_epoch=len(train_data),
    validation_steps=len(val_data)
)