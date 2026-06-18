"""
DOCSTRING:
Este script implementa transfer learning utilizando la arquitectura ResNet50 preentrenada en ImageNet
como extractor de características, sobre la cual se añade una cabeza de clasificación personalizada para
un problema de clasificación de 5 clases.

El flujo del código incluye:
1. Carga del modelo base ResNet50 sin la capa de clasificación superior (include_top=False).
2. Congelación de todas las capas del modelo base para evitar su actualización durante el entrenamiento inicial.
3. Construcción de una nueva cabeza de clasificación compuesta por:
   - Flatten para vectorizar las características convolucionales.
   - Capa densa intermedia con activación ReLU.
   - Capa de salida con activación softmax para clasificación multiclase.
4. Compilación del modelo con función de pérdida categorical_crossentropy.
5. Preparación de datos mediante ImageDataGenerator con normalización y partición de validación.
6. Entrenamiento inicial del modelo con capas congeladas.
7. Descongelamiento parcial de las últimas capas del modelo base para fine-tuning.
8. Recompilación del modelo con una tasa de aprendizaje reducida.
9. Evaluación final del modelo sobre el conjunto de validación.

Parámetros esperados:
- PATH_TO_DATASET: ruta al dataset estructurado en carpetas por clase.

Retorno:
- Historial de entrenamiento (history)
- Métricas de evaluación (val_loss, val_accuracy)

Excepciones:
- Puede fallar si el dataset no está estructurado correctamente o si las clases no coinciden con output_dim.
"""

# Importación de librerías necesarias
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Carga del modelo preentrenado ResNet50 sin la capa superior de clasificación
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Congelación de todas las capas del modelo base para transfer learning inicial
for layer in base_model.layers:
    layer.trainable = False

# Construcción de la cabeza de clasificación personalizada
x = Flatten()(base_model.output)
# Capa densa intermedia para aprendizaje de representaciones
x = Dense(256, activation='relu')(x)
# Capa de salida para clasificación en 5 clases
output = Dense(5, activation="softmax")(x)

# Definición del modelo completo
model = Model(inputs=base_model.input, outputs=output)

# Compilación del modelo con función de pérdida multiclase
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

# Resumen de la arquitectura del modelo
model.summary()

# Preparación de datos con aumento y validación dividida automáticamente
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Generador de datos de entrenamiento
train_data = datagen.flow_from_directory(
    "PATH_TO_DATASET",
    target_size=(224, 224),
    batch_size=32,
    class_node="categorical",
    subset="training"
)

# Generador de datos de validación
val_data = datagen.flow_from_directory(
    "PATH_TO_DATASET",
    target_size=(224, 224),
    batch_size=32,
    class_node="categorical",
    subset="validation"
)

# Entrenamiento del modelo con capas congeladas
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    steps_per_epoch=len(train_data),
    validation_steps=len(val_data)
)

# Descongelación parcial de las últimas capas del modelo base para fine-tuning
for layer in base_model.layers[-5:]:
    layer.trainable = True

# Recompilación del modelo con learning rate reducido para ajuste fino
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Evaluación final del modelo sobre el conjunto de validación
val_loss, val_accuracy = model.evaluate(val_data)