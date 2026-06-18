"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: CLASIFICACIÓN CIFAR-10 CON CNN (PYTORCH)
--------------------------------------------------

Descripción general:
Este script implementa un pipeline completo de aprendizaje profundo utilizando PyTorch
para la clasificación de imágenes del dataset CIFAR-10.

El flujo incluye:
- Carga del dataset CIFAR-10
- Aplicación de transformaciones de normalización
- Creación de DataLoaders para entrenamiento y prueba
- Definición de una red neuronal convolucional (CNN)
- Entrenamiento del modelo
- Evaluación de precisión sobre el conjunto de test

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Transformaciones:
   - Conversión de imágenes a tensores
   - Normalización de píxeles al rango [-1, 1]

2. Dataset:
   - CIFAR-10 con 10 clases de imágenes RGB
   - Separación en conjunto de entrenamiento y prueba

3. Modelo CNN:
   - 2 capas convolucionales
   - 3 capas fully connected
   - Función de activación ReLU
   - MaxPooling para reducción espacial

4. Entrenamiento:
   - Función de pérdida: CrossEntropyLoss
   - Optimizador: Adam

5. Evaluación:
   - Cálculo de accuracy en conjunto de test
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --------------------------------------------------
# TRANSFORMACIONES DE ENTRADA
# --------------------------------------------------
# Convierte imágenes a tensores y normaliza valores de píxeles al rango [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalización por canal RGB
])

# --------------------------------------------------
# CARGA DEL DATASET CIFAR-10
# --------------------------------------------------
# Dataset de entrenamiento (50,000 imágenes)
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Dataset de prueba (10,000 imágenes)
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# --------------------------------------------------
# DATA LOADERS
# --------------------------------------------------
# batch_size: número de imágenes por iteración
# shuffle=True en entrenamiento para mejorar generalización
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# shuffle=False en test para mantener orden consistente
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Impresión de tamaños de dataset
print(f"Training Data Size: {len(train_dataset)}")
print(f"Test Data Size: {len(test_dataset)}")

import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# DEFINICIÓN DEL MODELO CNN
# --------------------------------------------------
class CNN(nn.Module):
    """
    Red neuronal convolucional para clasificación de imágenes CIFAR-10.

    Arquitectura:
    - Conv2D(3 → 6 filtros, kernel 5x5)
    - MaxPool 2x2
    - Conv2D(6 → 16 filtros, kernel 5x5)
    - MaxPool 2x2
    - Fully Connected: 16*5*5 → 120
    - Fully Connected: 120 → 84
    - Fully Connected: 84 → 10 clases
    """

    def __init__(self):
        super(CNN, self).__init__()

        # Capa convolucional inicial (RGB → 6 mapas de características)
        self.conv1 = nn.Conv2d(3, 6, 5)

        # Capa de reducción espacial (submuestreo)
        self.pool = nn.MaxPool2d(2, 2)

        # Segunda capa convolucional
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Capas completamente conectadas
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Propagación hacia adelante del modelo.

        Parámetros:
        x (Tensor): lote de imágenes de entrada

        Retorno:
        Tensor: logits de salida para 10 clases
        """

        # Primera convolución + ReLU + pooling
        x = self.pool(torch.relu(self.conv1(x)))

        # Segunda convolución + ReLU + pooling
        x = self.pool(torch.relu(self.conv2(x)))

        # Aplanamiento del tensor para capas densas
        x = x.view(-1, 16 * 5 * 5)

        # Capas fully connected con activación ReLU
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Capa de salida (sin activación para CrossEntropyLoss)
        x = self.fc3(x)

        return x

# Instanciación del modelo
model = CNN()
print(model)

import torch.optim as optim

# --------------------------------------------------
# FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
# --------------------------------------------------
# CrossEntropyLoss: adecuada para clasificación multiclase
criterion = nn.CrossEntropyLoss()

# Optimizador Adam con tasa de aprendizaje definida
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------------
# FUNCIÓN DE ENTRENAMIENTO
# --------------------------------------------------
def train_model(model, train_loader, criterion, optimizer, epochs=1):
    """
    Entrena el modelo CNN usando batches de CIFAR-10.

    Parámetros:
    model (nn.Module): modelo a entrenar
    train_loader (DataLoader): datos de entrenamiento
    criterion: función de pérdida
    optimizer: algoritmo de optimización
    epochs (int): número de épocas de entrenamiento

    Retorno:
    None (imprime pérdida promedio por época)
    """

    # Modo entrenamiento (activa dropout/batchnorm si existieran)
    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        # Iteración por batches
        for images, labels in train_loader:

            # Reinicio de gradientes
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Cálculo de pérdida
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Actualización de parámetros
            optimizer.step()

            # Acumulación de pérdida
            running_loss += loss.item()

        # Promedio de pérdida por época
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Ejecución del entrenamiento
train_model(model, train_loader, criterion, optimizer)

# --------------------------------------------------
# FUNCIÓN DE EVALUACIÓN
# --------------------------------------------------
def evaluate_model(model, test_loader):
    """
    Evalúa el modelo entrenado sobre el conjunto de prueba.

    Parámetros:
    model (nn.Module): modelo entrenado
    test_loader (DataLoader): datos de prueba

    Retorno:
    None (imprime exactitud del modelo)
    """

    # Modo evaluación (desactiva comportamientos de entrenamiento)
    model.eval()

    correct = 0
    total = 0

    # Desactiva cálculo de gradientes para inferencia
    with torch.no_grad():

        for images, labels in test_loader:

            # Predicción del modelo
            outputs = model(images)

            # Clase con mayor score
            _, predicted = torch.max(outputs, 1)

            # Conteo total de muestras
            total += labels.size(0)

            # Conteo de predicciones correctas
            correct += (predicted == labels).sum().item()

    # Cálculo de exactitud final
    print(f"Test Exactitud: {100 * correct / total:.2f}%")

# Ejecución de evaluación
evaluate_model(model, test_loader)