"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: CLASIFICACIÓN DE SENTIMIENTOS CON RNN (PYTORCH + IMDB)
--------------------------------------------------

Descripción general:
Este script implementa un modelo de red neuronal recurrente (RNN) en PyTorch para la
clasificación de sentimientos en el dataset IMDB.

El objetivo es predecir si una reseña de película es positiva (1) o negativa (0).

El flujo incluye:
- Carga del dataset IMDB desde Keras
- Preprocesamiento con padding de secuencias
- Conversión de datos a tensores PyTorch
- Definición de un modelo RNN con embedding
- Entrenamiento del modelo
- Evaluación del modelo en conjunto de prueba

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Dataset:
   - IMDB: conjunto de reseñas de películas con etiquetas binarias
   - Representación como secuencias de índices de palabras

2. Preprocesamiento:
   - Limitación del vocabulario a 10,000 palabras
   - Padding de secuencias a longitud fija (200)
   - Conversión a tensores PyTorch

3. Modelo:
   - Embedding layer: transforma palabras en vectores densos
   - RNN (vanilla): procesa secuencias de embeddings
   - Fully connected: genera predicción binaria
   - Activación sigmoide para salida probabilística

4. Entrenamiento:
   - Loss: Binary Cross Entropy (BCELoss)
   - Optimizer: Adam
   - Entrenamiento por épocas usando DataLoader

5. Evaluación:
   - Loss sobre conjunto de test
   - Accuracy basada en umbral de decisión

--------------------------------------------------
COMPONENTES DEL MODELO
--------------------------------------------------

- Embedding:
  Convierte índices de palabras en representaciones vectoriales densas.

- RNN:
  Procesa la secuencia completa y produce un estado oculto final.

- Linear:
  Proyecta el estado oculto a una única salida (probabilidad).

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------
vocab_size = 10000  # Tamaño máximo del vocabulario
max_len = 200       # Longitud fija de las secuencias

# --------------------------------------------------
# CARGA DEL DATASET IMDB
# --------------------------------------------------
# X_train, X_test: secuencias de enteros (palabras indexadas)
# y_train, y_test: etiquetas binarias (0 o 1)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# --------------------------------------------------
# PADDING DE SECUENCIAS
# --------------------------------------------------
# Normaliza todas las secuencias a longitud fija max_len
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

# --------------------------------------------------
# CONVERSIÓN A TENSORES PYTORCH
# --------------------------------------------------
train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# --------------------------------------------------
# DEFINICIÓN DEL MODELO RNN
# --------------------------------------------------
class RNNModel(nn.Module):
    """
    Modelo de red neuronal recurrente para clasificación de texto IMDB.

    Arquitectura:
    - Embedding(vocab_size → embedding_dim)
    - RNN(hidden_dim)
    - Linear(hidden_dim → 1)
    - Sigmoid para clasificación binaria
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()

        # Capa de embedding: representación densa de palabras
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Capa RNN (vanilla)
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        # Capa fully connected de salida
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Propagación hacia adelante del modelo.

        Parámetros:
        x (Tensor): batch de secuencias de palabras

        Retorno:
        Tensor: probabilidad de clase positiva
        """

        # Conversión de índices a embeddings
        embedded = self.embedding(x)

        # Procesamiento secuencial con RNN
        output, hidden = self.rnn(embedded)

        # Uso del último estado oculto para clasificación
        return torch.sigmoid(self.fc(hidden.squeeze(0)))

# Instanciación del modelo
model = RNNModel(
    vocab_size=10000,
    embedding_dim=128,
    hidden_dim=128,
    output_dim=1
)

# --------------------------------------------------
# FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
# --------------------------------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------------
# FUNCIÓN DE ENTRENAMIENTO
# --------------------------------------------------
def train_rnn(model, train_loader, criterion, optimizer, epochs=5):
    """
    Entrena el modelo RNN sobre el dataset IMDB.

    Parámetros:
    model (nn.Module): modelo a entrenar
    train_loader (DataLoader): batches de entrenamiento
    criterion: función de pérdida (BCELoss)
    optimizer: optimizador Adam
    epochs (int): número de iteraciones completas

    Retorno:
    None (imprime pérdida promedio por época)
    """

    model.train()

    for epoch in range(epochs):

        epoch_loss = 0

        # Iteración por batches
        for X_batch, y_batch in train_loader:

            # Reinicio de gradientes
            optimizer.zero_grad()

            # Predicción del modelo
            predictions = model(X_batch).squeeze(1)

            # Cálculo de pérdida
            loss = criterion(predictions, y_batch.float())

            # Backpropagation
            loss.backward()

            # Actualización de parámetros
            optimizer.step()

            epoch_loss += loss.item()

        # Promedio de pérdida por época
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

# Entrenamiento del modelo
train_rnn(model, train_loader, criterion, optimizer)

# --------------------------------------------------
# FUNCIÓN DE EVALUACIÓN
# --------------------------------------------------
def evaluate_rnn(model, X_test, y_test):
    """
    Evalúa el modelo RNN en el conjunto de prueba.

    Parámetros:
    model (nn.Module): modelo entrenado
    X_test (array): secuencias de prueba
    y_test (array): etiquetas reales

    Retorno:
    None (imprime loss y accuracy)
    """

    model.eval()

    with torch.no_grad():

        # Predicción sobre conjunto de test
        predictions = model(torch.tensor(X_test)).squeeze(1)

        # Cálculo de pérdida
        loss = criterion(predictions, torch.tensor(y_test).float())

        # Cálculo de accuracy con umbral 0 (nota: clasificación binaria)
        accuracy = (
            (predictions > 0) == torch.tensor(y_test).float()
        ).float().mean().item()

    print(f"Test Loss: {loss.item():.4f}, Test Exactitud: {accuracy:.4f}")