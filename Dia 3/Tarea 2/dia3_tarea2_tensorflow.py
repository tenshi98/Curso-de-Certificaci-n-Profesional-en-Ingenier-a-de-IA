"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: CLASIFICACIÓN DE TEXTO CON RNN (KERAS + IMDB)
--------------------------------------------------

Descripción general:
Este script implementa un modelo de red neuronal recurrente (RNN) utilizando TensorFlow/Keras
para la clasificación binaria de sentimientos en el dataset IMDB.

El objetivo es determinar si una reseña de película es positiva (1) o negativa (0).

El flujo del script incluye:
- Carga del dataset IMDB
- Preprocesamiento de secuencias mediante padding
- Definición de un modelo secuencial con Embedding + SimpleRNN
- Entrenamiento del modelo
- Evaluación en conjunto de prueba

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Dataset:
   - IMDB: conjunto de reseñas de películas etiquetadas como positivas o negativas
   - Representación en forma de secuencias de índices de palabras
   - Limitación del vocabulario a las 10,000 palabras más frecuentes

2. Preprocesamiento:
   - Padding de secuencias a longitud fija (200 tokens)
   - Asegura que todas las entradas tengan la misma dimensión

3. Modelo:
   - Embedding: convierte índices de palabras en vectores densos
   - SimpleRNN: procesa secuencias de embeddings
   - Dense (sigmoid): salida binaria (0 o 1)

4. Entrenamiento:
   - Optimizer: Adam
   - Loss function: binary_crossentropy
   - Métrica: accuracy
   - Validación automática con 20% del dataset de entrenamiento

5. Evaluación:
   - Evaluación final en conjunto de test
   - Métricas: loss y accuracy

--------------------------------------------------
COMPONENTES DEL MODELO
--------------------------------------------------

- Embedding:
  Transforma palabras en representaciones vectoriales densas de dimensión 128.

- SimpleRNN:
  Procesa secuencias temporalmente y genera una representación final.

- Dense:
  Capa de salida con activación sigmoide para clasificación binaria.

"""

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# --------------------------------------------------
# CONFIGURACIÓN DE VOCABULARIO Y LONGITUD DE SECUENCIA
# --------------------------------------------------
vocab_size = 10000  # Número máximo de palabras consideradas
max_len = 200       # Longitud fija de cada secuencia

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
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

# --------------------------------------------------
# INSPECCIÓN DE DATOS
# --------------------------------------------------
# Muestra las dimensiones del dataset procesado
print(f"Training Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}")

# --------------------------------------------------
# DEFINICIÓN DEL MODELO RNN (KERAS SEQUENTIAL)
# --------------------------------------------------
model = Sequential([
    # Capa Embedding: convierte índices de palabras en vectores densos
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
# Loss: binary_crossentropy (clasificación binaria)
# Métrica: accuracy
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Resumen de arquitectura del modelo
model.summary()

# --------------------------------------------------
# ENTRENAMIENTO DEL MODELO
# --------------------------------------------------
# validation_split: 20% de los datos de entrenamiento usados para validación
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
loss, accuracy = model.evaluate(X_test, y_test)

# Impresión de resultados finales
print(f"Test Loss: {loss:.4f}, Test Exactitud: {accuracy:.4f}")