"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: CLASIFICACIÓN DE SENTIMIENTOS CON RNN (KERAS + IMDB)
--------------------------------------------------

Descripción general:
Este script implementa un modelo de red neuronal recurrente (RNN) utilizando TensorFlow/Keras
para la clasificación binaria de sentimientos en el dataset IMDB.

El objetivo es clasificar reseñas de películas como positivas (1) o negativas (0).

El flujo incluye:
- Carga del dataset IMDB
- Preprocesamiento de secuencias con padding
- Definición de un modelo secuencial con Embedding + SimpleRNN
- Entrenamiento del modelo
- Evaluación en conjunto de prueba

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Dataset:
   - IMDB: conjunto de reseñas de películas etiquetadas (positivo/negativo)
   - Representación como secuencias de índices de palabras

2. Preprocesamiento:
   - Limitación del vocabulario a 10,000 palabras más frecuentes
   - Padding de secuencias a longitud fija (200 tokens)

3. Modelo:
   - Embedding layer para representación vectorial de palabras
   - SimpleRNN para procesamiento secuencial
   - Capa densa con activación sigmoide para clasificación binaria

4. Entrenamiento:
   - Optimizer: Adam
   - Loss function: binary_crossentropy
   - Métrica: accuracy
   - Validación usando 20% del dataset de entrenamiento

5. Evaluación:
   - Evaluación final en conjunto de test
   - Métricas: loss y accuracy

--------------------------------------------------
COMPONENTES DEL MODELO
--------------------------------------------------

- Embedding:
  Convierte índices de palabras en vectores densos de dimensión 128.

- SimpleRNN:
  Procesa secuencias de embeddings y genera un vector representativo final.

- Dense:
  Capa de salida con activación sigmoide para clasificación binaria.

"""

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# --------------------------------------------------
# CONFIGURACIÓN DEL VOCABULARIO Y LONGITUD DE SECUENCIA
# --------------------------------------------------
vocab_size = 10000  # Número máximo de palabras a considerar
max_len = 200       # Longitud fija de las secuencias

# --------------------------------------------------
# CARGA DEL DATASET IMDB
# --------------------------------------------------
# X_train, X_test: secuencias de índices de palabras
# y_train, y_test: etiquetas binarias (0 = negativo, 1 = positivo)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# --------------------------------------------------
# PADDING DE SECUENCIAS
# --------------------------------------------------
# Ajusta todas las secuencias a longitud fija max_len
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

# Impresión de forma de datos procesados
print(f"Training Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}")

# --------------------------------------------------
# DEFINICIÓN DEL MODELO RNN (KERAS SEQUENTIAL)
# --------------------------------------------------
model = Sequential([
    # Capa de embedding: convierte palabras en vectores densos
    Embedding(input_dim=vocab_size, output_dim=128),

    # Capa RNN simple con activación tanh
    SimpleRNN(128, activation='tanh', return_sequences=False),

    # Capa de salida para clasificación binaria
    Dense(1, activation='sigmoid')
])

# --------------------------------------------------
# COMPILACIÓN DEL MODELO
# --------------------------------------------------
# Optimizer: Adam
# Loss: binary_crossentropy para clasificación binaria
# Métrica: accuracy
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Mostrar arquitectura del modelo
model.summary()

# --------------------------------------------------
# ENTRENAMIENTO DEL MODELO
# --------------------------------------------------
# validation_split: 20% del dataset de entrenamiento se usa como validación
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# --------------------------------------------------
# EVALUACIÓN DEL MODELO
# --------------------------------------------------
# Evaluación final sobre conjunto de test
loss, accuracy = model.evaluate(X_test, y_test)

# Impresión de métricas finales
print(f"Test Loss: {loss:.4f}, Test Exactitud: {accuracy:.4f}")