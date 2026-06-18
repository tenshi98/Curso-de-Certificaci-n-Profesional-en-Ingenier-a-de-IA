"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: COMPARACIÓN RNN vs LSTM vs GRU (KERAS + IMDB)
--------------------------------------------------

Descripción general:
Este script implementa y compara tres arquitecturas de redes neuronales recurrentes
utilizando TensorFlow/Keras para la clasificación binaria de sentimientos en el dataset IMDB:

- SimpleRNN
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

El objetivo es evaluar el desempeño de cada arquitectura en la tarea de análisis de sentimiento
(clasificación de reseñas como positivas o negativas).

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Dataset:
   - IMDB: conjunto de reseñas de películas etiquetadas como positivas o negativas
   - Representación como secuencias de índices de palabras
   - Vocabulario limitado a las 10,000 palabras más frecuentes

2. Preprocesamiento:
   - Padding de secuencias a longitud fija (200 tokens)
   - Homogeneización de la entrada para modelos recurrentes

3. Modelos:
   - SimpleRNN: arquitectura recurrente básica
   - LSTM: red recurrente con memoria de largo plazo
   - GRU: variante optimizada de LSTM con menor complejidad computacional
   - Todos los modelos comparten:
     Embedding → capa recurrente → Dense(sigmoid)

4. Entrenamiento:
   - Optimizer: Adam
   - Loss function: binary_crossentropy
   - Métrica: accuracy
   - Validación automática con 20% del dataset de entrenamiento

5. Evaluación:
   - Evaluación independiente de cada modelo en el conjunto de test
   - Comparación de loss y accuracy entre arquitecturas

--------------------------------------------------
COMPONENTES DEL MODELO
--------------------------------------------------

- Embedding:
  Convierte palabras (índices) en vectores densos de dimensión 128.

- SimpleRNN:
  Modelo recurrente básico con memoria limitada.

- LSTM:
  Arquitectura con compuertas de entrada, olvido y salida,
  adecuada para dependencias largas en secuencias.

- GRU:
  Variante simplificada de LSTM con menor costo computacional
  y desempeño competitivo en tareas NLP.

- Dense:
  Capa de salida con activación sigmoide para clasificación binaria.

"""

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense

# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------
vocab_size = 10000  # Tamaño máximo del vocabulario
max_len = 200       # Longitud fija de las secuencias

# --------------------------------------------------
# CARGA DEL DATASET IMDB
# --------------------------------------------------
# X_train, X_test: secuencias de palabras codificadas como enteros
# y_train, y_test: etiquetas binarias (0 = negativo, 1 = positivo)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# --------------------------------------------------
# PADDING DE SECUENCIAS
# --------------------------------------------------
# Normaliza todas las secuencias a longitud fija max_len
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

# Inspección de forma del dataset procesado
print(f"Training Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}")

# --------------------------------------------------
# MODELO 1: SIMPLE RNN
# --------------------------------------------------
rnn_model = Sequential([
    # Embedding de palabras a vectores densos
    Embedding(input_dim=vocab_size, output_dim=128),

    # RNN básica
    SimpleRNN(128, activation='tanh', return_sequences=False),

    # Salida binaria
    Dense(1, activation='sigmoid')
])

rnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

rnn_model.summary()

rnn_history = rnn_model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test, y_test)

print(f"RNN Test Loss: {rnn_loss:.4f}, Test Exactitud: {rnn_accuracy:.4f}")

# --------------------------------------------------
# MODELO 2: LSTM
# --------------------------------------------------
lstm_model = Sequential([
    # Embedding de palabras
    Embedding(input_dim=vocab_size, output_dim=128),

    # LSTM con memoria de largo plazo
    LSTM(128, activation='tanh', return_sequences=False),

    # Salida binaria
    Dense(1, activation='sigmoid')
])

lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

lstm_model.summary()

lstm_history = lstm_model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test)

print(f"LSTM Test Loss: {lstm_loss:.4f}, Test Exactitud: {lstm_accuracy:.4f}")

# --------------------------------------------------
# MODELO 3: GRU
# --------------------------------------------------
gru_model = Sequential([
    # Embedding de palabras
    Embedding(input_dim=vocab_size, output_dim=128),

    # GRU (optimizado y eficiente)
    GRU(128, activation='tanh', return_sequences=False),

    # Salida binaria
    Dense(1, activation='sigmoid')
])

gru_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

gru_model.summary()

gru_history = gru_model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

gru_loss, gru_accuracy = gru_model.evaluate(X_test, y_test)

print(f"GRU Test Loss: {gru_loss:.4f}, Test Exactitud: {gru_accuracy:.4f}")