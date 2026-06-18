"""
DOCSTRING GENERAL DEL SCRIPT
------------------------------------------------------------
Título:
    Entrenamiento de una Red Neuronal Convolucional (CNN) en CIFAR-10 con TensorFlow/Keras

Descripción:
    Este script implementa un flujo completo de deep learning para clasificación de imágenes
    utilizando el dataset CIFAR-10 y una red neuronal convolucional (CNN) en TensorFlow.

    El proceso incluye:
    - Carga del dataset CIFAR-10
    - Normalización de imágenes a rango [0, 1]
    - One-hot encoding de etiquetas
    - Definición de un modelo CNN con capas convolucionales, pooling, dropout y fully connected
    - Entrenamiento del modelo con validación
    - Evaluación en conjunto de prueba
    - Visualización de métricas de entrenamiento (accuracy y loss)

Entradas:
    - CIFAR-10 dataset (imágenes RGB 32x32 en 10 clases)

Salidas:
    - Modelo entrenado CNN
    - Métricas de accuracy y loss
    - Gráficos de evolución del entrenamiento

Dependencias:
    - tensorflow
    - matplotlib

------------------------------------------------------------
"""

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ------------------------------------------------------------
# CARGA Y PREPROCESAMIENTO DEL DATASET
# ------------------------------------------------------------

# Load CIFAR-10 dataset (train/test split)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values from [0, 255] to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert class labels to one-hot encoded vectors
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Display dataset shapes
print(f"Training Data Shape: {X_train.shape}, Label Shapes: {y_train.shape}")
print(f"Test Data Shape: {X_test.shape}, Label Shapes: {y_test.shape}")

# ------------------------------------------------------------
# DEFINICIÓN DEL MODELO CNN
# ------------------------------------------------------------

# Sequential CNN architecture
model = Sequential([
    # First convolutional block
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),

    # Second convolutional block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Flatten feature maps into vector
    Flatten(),

    # Fully connected layer
    Dense(128, activation='relu'),

    # Regularization layer to reduce overfitting
    Dropout(0.5),

    # Output layer (10 classes)
    Dense(10, activation='softmax')
])

# Display model architecture summary
model.summary()

# ------------------------------------------------------------
# COMPILACIÓN DEL MODELO
# ------------------------------------------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------------------------------------------
# ENTRENAMIENTO DEL MODELO
# ------------------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# ------------------------------------------------------------
# EVALUACIÓN DEL MODELO
# ------------------------------------------------------------

# Evaluate performance on unseen test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Display final accuracy
print(f"Test Exactitud: {test_accuracy:.4f}")

# ------------------------------------------------------------
# VISUALIZACIÓN DE MÉTRICAS
# ------------------------------------------------------------

import matplotlib.pyplot as plt

# Plot training vs validation accuracy
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training vs validation loss
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()