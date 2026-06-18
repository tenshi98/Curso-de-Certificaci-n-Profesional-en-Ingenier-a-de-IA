"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: CNN CON CIFAR-10 (TENSORFLOW / KERAS)
--------------------------------------------------

Descripción general:
Este script implementa un modelo de red neuronal convolucional (CNN) utilizando TensorFlow/Keras
para la clasificación del dataset CIFAR-10.

Incluye:
- Carga del dataset CIFAR-10
- Normalización de imágenes
- One-hot encoding de etiquetas
- Aumento de datos (data augmentation)
- Definición de un modelo CNN profundo con BatchNormalization y Dropout
- Entrenamiento del modelo con generador de datos
- Evaluación final del modelo
- Visualización de métricas de entrenamiento

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Dataset:
   - CIFAR-10: 60,000 imágenes RGB (32x32) en 10 clases
   - Separación en entrenamiento y prueba

2. Preprocesamiento:
   - Normalización de píxeles a rango [0, 1]
   - Conversión de etiquetas a formato one-hot

3. Data Augmentation:
   - Rotación aleatoria
   - Desplazamiento horizontal y vertical
   - Flip horizontal

4. Modelo CNN:
   - 2 bloques convolucionales
   - BatchNormalization para estabilizar entrenamiento
   - MaxPooling para reducción dimensional
   - Dropout para regularización
   - Fully connected layer con 512 neuronas
   - Capa de salida softmax para clasificación multiclase

5. Entrenamiento:
   - Optimizador: Adam
   - Función de pérdida: categorical_crossentropy
   - Entrenamiento con generador de datos aumentado

6. Evaluación:
   - Evaluación sobre conjunto de test
   - Métrica: accuracy

7. Visualización:
   - Curvas de accuracy (entrenamiento vs validación)
   - Curvas de loss (entrenamiento vs validación)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# --------------------------------------------------
# CARGA DEL DATASET CIFAR-10
# --------------------------------------------------
# x_train: imágenes de entrenamiento (50,000)
# y_train: etiquetas de entrenamiento
# x_test: imágenes de prueba (10,000)
# y_test: etiquetas de prueba
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# --------------------------------------------------
# NORMALIZACIÓN DE IMÁGENES
# --------------------------------------------------
# Escala los valores de píxeles de [0, 255] a [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# --------------------------------------------------
# ONE-HOT ENCODING DE ETIQUETAS
# --------------------------------------------------
# Convierte etiquetas enteras en vectores binarios de 10 clases
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# --------------------------------------------------
# DATA AUGMENTATION
# --------------------------------------------------
# Generador de imágenes aumentadas para mejorar generalización del modelo
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Ajuste del generador al dataset de entrenamiento
datagen.fit(x_train)

# --------------------------------------------------
# DEFINICIÓN DEL MODELO CNN
# --------------------------------------------------
def create_model():
    """
    Crea un modelo CNN secuencial para clasificación de CIFAR-10.

    Retorno:
    model (tf.keras.Model): modelo compilado de red neuronal convolucional
    """

    model = models.Sequential()

    # --------------------------------------------------
    # CAPA DE ENTRADA
    # --------------------------------------------------
    # Define forma de entrada (32x32 imágenes RGB)
    model.add(layers.Input(shape=(32, 32, 3)))

    # --------------------------------------------------
    # BLOQUE CONVOLUCIONAL 1
    # --------------------------------------------------
    # Primera convolución + BatchNormalization
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())

    # Segunda convolución + BatchNormalization
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())

    # Reducción espacial
    model.add(layers.MaxPooling2D((2, 2)))

    # Regularización para evitar overfitting
    model.add(layers.Dropout(0.25))

    # --------------------------------------------------
    # BLOQUE CONVOLUCIONAL 2
    # --------------------------------------------------
    # Incremento de filtros para extraer características más complejas
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())

    # Reducción espacial
    model.add(layers.MaxPooling2D((2, 2)))

    # Regularización adicional
    model.add(layers.Dropout(0.25))

    # --------------------------------------------------
    # CAPAS FULLY CONNECTED
    # --------------------------------------------------
    # Aplanamiento de feature maps
    model.add(layers.Flatten())

    # Capa densa principal
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())

    # Dropout para reducir sobreajuste
    model.add(layers.Dropout(0.5))

    # --------------------------------------------------
    # CAPA DE SALIDA
    # --------------------------------------------------
    # Softmax para clasificación multiclase (10 clases)
    model.add(layers.Dense(10, activation='softmax'))

    return model

# Creación del modelo
model = create_model()

# --------------------------------------------------
# COMPILACIÓN DEL MODELO
# --------------------------------------------------
# Optimizer: Adam (adaptativo)
# Loss: categorical_crossentropy para clasificación multiclase
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------------------------------
# ENTRENAMIENTO DEL MODELO
# --------------------------------------------------
# Uso de generador de datos aumentados
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=20,
    validation_data=(x_test, y_test),
    steps_per_epoch=x_train.shape[0] // 64
)

# --------------------------------------------------
# EVALUACIÓN DEL MODELO
# --------------------------------------------------
# Evaluación final sobre conjunto de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

print(f"Test Exactitud: {test_accuracy:.2f}")

# --------------------------------------------------
# VISUALIZACIÓN DE RESULTADOS
# --------------------------------------------------
# Gráfico de accuracy (entrenamiento vs validación)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

# Gráfico de loss (entrenamiento vs validación)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()