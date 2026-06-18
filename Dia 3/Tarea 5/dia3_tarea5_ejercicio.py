"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: LSTM CON Y SIN GloVe (KERAS + IMDB)
--------------------------------------------------

Descripción general:
Este script implementa un modelo de clasificación de sentimientos utilizando el dataset IMDB
y compara dos enfoques de embeddings:

- LSTM con embeddings preentrenados GloVe (congelados)
- LSTM con embeddings aprendidos desde cero

El objetivo es analizar el impacto del uso de embeddings preentrenados en tareas de NLP.

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Dataset:
   - IMDB: conjunto de reseñas de películas etiquetadas como positivas o negativas
   - Representación inicial como secuencias de índices de palabras

2. Preprocesamiento:
   - Decodificación opcional de reseñas a texto (para inspección)
   - Padding de secuencias a longitud fija (200 tokens)
   - Limitación del vocabulario a 10,000 palabras

3. Embeddings:
   - GloVe (Global Vectors for Word Representation)
   - Dimensión: 100
   - Embeddings preentrenados cargados desde archivo externo
   - Matriz de embeddings construida manualmente
   - Embedding congelado (trainable=False) en el primer modelo

4. Modelos:
   - LSTM con GloVe embeddings
   - LSTM sin embeddings preentrenados (aprendidos desde cero)
   - Ambos usan arquitectura: Embedding → LSTM → Dense(sigmoid)

5. Entrenamiento:
   - Optimizer: Adam
   - Loss function: binary_crossentropy
   - Métrica: accuracy
   - Validación automática con 20% del dataset de entrenamiento

6. Evaluación:
   - Evaluación independiente de ambos modelos en el conjunto de test
   - Comparación de accuracy entre modelos

7. Visualización:
   - Gráfico de barras comparando desempeño de ambos modelos

--------------------------------------------------
COMPONENTES DEL MODELO
--------------------------------------------------

- Embedding (GloVe):
  Representaciones vectoriales preentrenadas que capturan semántica del lenguaje.
  Se cargan desde archivo externo y se asignan a palabras del vocabulario.

- Embedding (trainable):
  Representación aprendida durante el entrenamiento del modelo.

- LSTM:
  Red recurrente con memoria de largo plazo para capturar dependencias en texto.

- Dense:
  Capa de salida con activación sigmoide para clasificación binaria.

--------------------------------------------------
NOTA SOBRE DATOS EXTERNOS
--------------------------------------------------

- El archivo "glove.6B.100d.txt" debe estar disponible en el entorno de ejecución.
- Su ausencia impide la construcción de la matriz de embeddings GloVe.

--------------------------------------------------
FLUJO GENERAL DEL SCRIPT
--------------------------------------------------

1. Carga dataset IMDB
2. Construcción de embeddings GloVe
3. Creación de matriz de embeddings
4. Entrenamiento LSTM con GloVe (no entrenable)
5. Entrenamiento LSTM sin GloVe
6. Evaluación de ambos modelos
7. Visualización comparativa de accuracy
"""

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------
vocab_size = 10000  # Tamaño máximo del vocabulario
max_len = 200       # Longitud fija de las secuencias

# --------------------------------------------------
# CARGA DEL DATASET IMDB
# --------------------------------------------------
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# --------------------------------------------------
# DECODIFICACIÓN DE REVIEWS (INSPECCIÓN OPCIONAL)
# --------------------------------------------------
# Convierte índices de palabras a texto legible (solo muestra ejemplos)
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

decoded_reviews = [
    " ".join([reverse_word_index.get(i - 3, "?") for i in review])
    for review in X_train[:5]
]

# --------------------------------------------------
# PADDING DE SECUENCIAS
# --------------------------------------------------
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

# Inspección de dimensiones del dataset
print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# --------------------------------------------------
# CARGA DE EMBEDDINGS GloVe
# --------------------------------------------------
embedding_index = {}
glove_file = "glove.6B.100d.txt"

# Lectura del archivo de embeddings preentrenados
with open(glove_file, "r", encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

print(f"Loaded {len(embedding_index)} word vectors")

# --------------------------------------------------
# CONSTRUCCIÓN DE MATRIZ DE EMBEDDING
# --------------------------------------------------
embedding_dim = 100

# Matriz donde cada fila representa el vector de una palabra del vocabulario
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Asignación de vectores GloVe al vocabulario IMDB
for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# --------------------------------------------------
# MODELO 1: LSTM CON GloVe
# --------------------------------------------------
model = Sequential([
    # Embedding preentrenado (congelado)
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False
    ),

    # LSTM para modelado secuencial
    LSTM(128, activation='tanh', return_sequences=False),

    # Capa de salida binaria
    Dense(1, activation='sigmoid')
])

# Compilación del modelo GloVe
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Entrenamiento del modelo con GloVe
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=64,
    verbose=1
)

# Evaluación del modelo con GloVe
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# --------------------------------------------------
# MODELO 2: LSTM SIN GloVe
# --------------------------------------------------
lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),

    LSTM(128, activation='tanh', return_sequences=False),

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
    batch_size=64,
    validation_split=0.2
)

lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test)

print(f"LSTM without GloVe Test Exactitud: {lstm_accuracy:.4f}")
print(f"LSTM model with GloVe Test Exactitud: {accuracy:.4f}")

# --------------------------------------------------
# VISUALIZACIÓN COMPARATIVA
# --------------------------------------------------
import matplotlib.pyplot as plt

models = ['LSTM', 'LSTM GloVe']
accuracies = [lstm_accuracy, accuracy]

plt.bar(models, accuracies, color=['blue', 'green'])
plt.title('Comparison of LSTM with and without word embeddings')
plt.ylabel('Accuracy')
plt.show()