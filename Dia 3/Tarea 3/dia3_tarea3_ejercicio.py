"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: COMPARACIÓN RNN vs LSTM (KERAS + IMDB)
--------------------------------------------------

Descripción general:
Este script implementa y compara dos modelos de redes neuronales recurrentes utilizando
TensorFlow/Keras para la clasificación binaria de sentimientos en el dataset IMDB:

- Un modelo basado en SimpleRNN
- Un modelo basado en LSTM

El objetivo es evaluar el desempeño de ambas arquitecturas en la tarea de análisis de sentimiento.

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Dataset:
   - IMDB: conjunto de reseñas de películas etiquetadas como positivas o negativas
   - Representación como secuencias de índices de palabras
   - Vocabulario limitado a las 10,000 palabras más frecuentes

2. Preprocesamiento:
   - Padding de secuencias a longitud fija (200 tokens)
   - Asegura uniformidad en la entrada del modelo

3. Modelos:
   - RNN (SimpleRNN): modelo recurrente básico
   - LSTM: variante avanzada con memoria de largo plazo
   - Ambos usan Embedding + capa recurrente + Dense(sigmoid)

4. Entrenamiento:
   - Optimizer: Adam
   - Loss function: binary_crossentropy
   - Métrica: accuracy
   - Validación automática con 20% del dataset de entrenamiento

5. Evaluación:
   - Evaluación independiente para cada modelo en el conjunto de test
   - Comparación de loss y accuracy entre RNN y LSTM

--------------------------------------------------
COMPONENTES DEL MODELO
--------------------------------------------------

- Embedding:
  Convierte palabras (índices) en vectores densos de dimensión 128.

- SimpleRNN:
  Procesa secuencias de forma recurrente, manteniendo una memoria limitada.

- LSTM:
  Variante de RNN con celdas de memoria y compuertas,
  mejorando el manejo de dependencias largas en el texto.

- Dense:
  Capa de salida con activación sigmoide para clasificación binaria.

"""

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense

# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------
vocab_size = 10000  # Número máximo de palabras del vocabulario
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
# Normaliza todas las secuencias a longitud fija max_len
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

# Inspección de forma del dataset
print(f"Training Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}")

# --------------------------------------------------
# MODELO 1: RNN (SimpleRNN)
# --------------------------------------------------
rnn_model = Sequential([
    # Embedding de palabras a vectores densos
    Embedding(input_dim=vocab_size, output_dim=128),

    # RNN simple (memoria limitada)
    SimpleRNN(128, activation='tanh', return_sequences=False),

    # Capa de salida binaria
    Dense(1, activation='sigmoid')
])

# Compilación del modelo RNN
rnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Resumen del modelo RNN
rnn_model.summary()

# Entrenamiento del modelo RNN
rnn_history = rnn_model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# Evaluación del modelo RNN
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test, y_test)

print(f"RNN Test Loss: {rnn_loss:.4f}, Test Exactitud: {rnn_accuracy:.4f}")

# --------------------------------------------------
# MODELO 2: LSTM
# --------------------------------------------------
lstm_model = Sequential([
    # Embedding de palabras a vectores densos
    Embedding(input_dim=vocab_size, output_dim=128),

    # LSTM con memoria de largo plazo
    LSTM(128, activation='tanh', return_sequences=False),

    # Capa de salida binaria
    Dense(1, activation='sigmoid')
])

# Compilación del modelo LSTM
lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Resumen del modelo LSTM
lstm_model.summary()

# Entrenamiento del modelo LSTM
lstm_history = lstm_model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# Evaluación del modelo LSTM
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test)

print(f"LSTM Test Loss: {lstm_loss:.4f}, Test Exactitud: {lstm_accuracy:.4f}")