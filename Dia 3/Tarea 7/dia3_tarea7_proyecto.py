"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: COMPARACIÓN RNN vs LSTM vs GRU (KERAS + IMDB)
--------------------------------------------------

Descripción general:
Este script implementa y compara tres arquitecturas de redes neuronales recurrentes
utilizando TensorFlow/Keras para la clasificación binaria de sentimientos en el dataset IMDB.

Los modelos evaluados son:
- SimpleRNN
- LSTM
- GRU

El objetivo es comparar su desempeño en términos de precisión (accuracy) en la tarea
de análisis de sentimiento.

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Dataset:
   - IMDB: conjunto de reseñas de películas etiquetadas como positivas o negativas
   - Representación como secuencias de índices de palabras
   - Vocabulario limitado a las 10,000 palabras más frecuentes

2. Preprocesamiento:
   - Padding de secuencias a longitud fija (100 tokens)
   - Normalización de longitud de entrada para redes recurrentes

3. Modelos:
   - SimpleRNN: red recurrente básica con memoria limitada
   - LSTM: red con memoria de largo plazo y compuertas internas
   - GRU: variante optimizada de LSTM con menor complejidad computacional
   - Todos los modelos siguen la arquitectura:
     Embedding → Capa recurrente → Dense(sigmoid)

4. Entrenamiento:
   - Optimizer: Adam
   - Loss function: binary_crossentropy
   - Métrica: accuracy
   - Entrenamiento independiente para cada modelo
   - Validación automática con 20% del dataset de entrenamiento

5. Evaluación:
   - Evaluación final en conjunto de test
   - Comparación de accuracy entre los tres modelos

6. Visualización:
   - Gráfico de la accuracy de entrenamiento por época para cada modelo

--------------------------------------------------
COMPONENTES DEL MODELO
--------------------------------------------------

- Embedding:
  Convierte índices de palabras en vectores densos de 128 dimensiones.

- SimpleRNN:
  Modelo recurrente básico con memoria limitada y menor capacidad de retención.

- LSTM:
  Arquitectura con compuertas que mejora el manejo de dependencias largas.

- GRU:
  Variante simplificada de LSTM con menor costo computacional y rendimiento competitivo.

- Dense:
  Capa de salida con activación sigmoide para clasificación binaria.

--------------------------------------------------
FLUJO GENERAL DEL SCRIPT
--------------------------------------------------

1. Carga del dataset IMDB
2. Padding de secuencias a longitud fija
3. Definición de tres modelos (RNN, LSTM, GRU)
4. Compilación de cada modelo
5. Entrenamiento independiente de cada modelo
6. Evaluación en conjunto de test
7. Comparación de accuracy en gráfico
"""

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM, GRU

# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------
vocab_size = 10000  # Tamaño máximo del vocabulario
max_len = 100       # Longitud fija de secuencias

# --------------------------------------------------
# CARGA DEL DATASET IMDB
# --------------------------------------------------
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# --------------------------------------------------
# PADDING DE SECUENCIAS
# --------------------------------------------------
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

# Inspección de dimensiones
print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# --------------------------------------------------
# MODELO 1: SIMPLE RNN
# --------------------------------------------------
rnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    SimpleRNN(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])

rnn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

rnn_model.summary()

# --------------------------------------------------
# MODELO 2: LSTM
# --------------------------------------------------
lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

lstm_model.summary()

# --------------------------------------------------
# MODELO 3: GRU
# --------------------------------------------------
gru_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    GRU(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])

gru_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

gru_model.summary()

# --------------------------------------------------
# ENTRENAMIENTO DE MODELOS
# --------------------------------------------------
history_rnn = rnn_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=64,
    verbose=1
)

history_lstm = lstm_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=64,
    verbose=1
)

history_gru = gru_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=64,
    verbose=1
)

# --------------------------------------------------
# EVALUACIÓN DE MODELOS
# --------------------------------------------------
loss_rnn, accuracy_rnn = rnn_model.evaluate(X_test, y_test, verbose=0)
loss_lstm, accuracy_lstm = lstm_model.evaluate(X_test, y_test, verbose=0)
loss_gru, accuracy_gru = gru_model.evaluate(X_test, y_test, verbose=0)

print(f"RNN Test Exactitud: {accuracy_rnn:.4f}")
print(f"LSTM Test Exactitud: {accuracy_lstm:.4f}")
print(f"GRU Test Exactitud: {accuracy_gru:.4f}")

# --------------------------------------------------
# VISUALIZACIÓN DE ACCURACY
# --------------------------------------------------
import matplotlib.pyplot as plt

plt.plot(history_rnn.history['accuracy'], label="RNN Training Accuracy")
plt.plot(history_lstm.history['accuracy'], label="LSTM Training Accuracy")
plt.plot(history_gru.history['accuracy'], label="GRU Training Accuracy")

plt.title("Training Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()