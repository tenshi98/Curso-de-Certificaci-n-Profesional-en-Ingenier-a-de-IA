"""
--------------------------------------------------
DOCUMENTACIÓN DEL MÓDULO: ENTRENAMIENTO DE CNN CON CIFAR-10 (PYTORCH)
--------------------------------------------------

Descripción general:
Este script implementa un flujo completo de aprendizaje profundo utilizando PyTorch para la
clasificación de imágenes del dataset CIFAR-10. Incluye:

- Carga y preprocesamiento del dataset CIFAR-10
- Definición de transformaciones de entrada
- Construcción de DataLoaders para entrenamiento y prueba
- Definición de una red neuronal convolucional (CNN)
- Entrenamiento del modelo
- Evaluación del desempeño en el conjunto de prueba

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Transformaciones:
   - Conversión de imágenes a tensores
   - Normalización con media y desviación estándar (0.5, 0.5, 0.5)

2. Dataset:
   - CIFAR-10: conjunto de imágenes RGB de 10 clases
   - Separación en conjunto de entrenamiento y prueba

3. Modelo:
   - CNN con dos capas convolucionales
   - Tres capas completamente conectadas (fully connected)
   - Función de activación ReLU
   - MaxPooling para reducción de dimensionalidad

4. Entrenamiento:
   - Función de pérdida: CrossEntropyLoss
   - Optimizador: SGD con momentum
   - Bucle de entrenamiento por épocas

5. Evaluación:
   - Cálculo de exactitud (accuracy) sobre conjunto de prueba
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --------------------------------------------------
# TRANSFORMACIONES DE LOS DATOS DE ENTRADA
# --------------------------------------------------
# Convierte imágenes a tensores y normaliza los valores de píxeles
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --------------------------------------------------
# CARGA DEL DATASET CIFAR-10
# --------------------------------------------------
# Dataset de entrenamiento (60,000 imágenes)
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Dataset de prueba (10,000 imágenes)
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# --------------------------------------------------
# CREACIÓN DE DATA LOADERS
# --------------------------------------------------
# Batch size define cuántas muestras se procesan por iteración
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Impresión de tamaños de dataset
print(f"Training Data Size: {len(train_dataset)}")
print(f"Test Data Size: {len(test_dataset)}")

# --------------------------------------------------
# DEFINICIÓN DEL MODELO CNN
# --------------------------------------------------
class CNN(nn.Module):
    """
    Red neuronal convolucional (CNN) para clasificación de imágenes CIFAR-10.

    Arquitectura:
    - Conv2D(3 → 6 filtros, kernel 5x5)
    - MaxPooling 2x2
    - Conv2D(6 → 16 filtros, kernel 5x5)
    - MaxPooling 2x2
    - Fully Connected: 16*5*5 → 120
    - Fully Connected: 120 → 84
    - Fully Connected: 84 → 10 clases
    """

    def __init__(self):
        super(CNN, self).__init__()

        # Primera capa convolucional: entrada RGB (3 canales)
        self.conv1 = nn.Conv2d(3, 6, 5)

        # Capa de pooling para reducción espacial
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
        Tensor: logits de salida para cada una de las 10 clases
        """

        # Conv1 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv1(x)))

        # Conv2 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))

        # Aplanado de tensor para capas fully connected
        x = x.view(-1, 16 * 5 * 5)

        # Capas densas con activación ReLU
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Capa de salida (sin activación para CrossEntropyLoss)
        x = self.fc3(x)

        return x

# Instanciación del modelo
model = CNN()
print(model)

# --------------------------------------------------
# DEFINICIÓN DE FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
# --------------------------------------------------
# CrossEntropyLoss: adecuada para clasificación multiclase
criterion = nn.CrossEntropyLoss()

# Optimizador SGD con tasa de aprendizaje y momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Alternativa comentada:
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# --------------------------------------------------
# FUNCIÓN DE ENTRENAMIENTO
# --------------------------------------------------
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    """
    Entrena el modelo CNN utilizando los datos de entrenamiento.

    Parámetros:
    model (nn.Module): modelo a entrenar
    train_loader (DataLoader): iterador de datos de entrenamiento
    criterion: función de pérdida
    optimizer: algoritmo de optimización
    epochs (int): número de iteraciones completas sobre el dataset

    Retorno:
    None
    """

    # Configura el modelo en modo entrenamiento
    model.train()

    # Iteración por épocas
    for epoch in range(epochs):

        running_loss = 0.0

        # Iteración sobre batches
        for images, labels in train_loader:

            # Reinicia gradientes acumulados
            optimizer.zero_grad()

            # Forward pass: predicción del modelo
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
        print(f"Epoch {epoch+1}, Loss; {running_loss/len(train_loader):.4f}")

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
    None (imprime la exactitud del modelo)
    """

    # Configura el modelo en modo evaluación
    model.eval()

    correct = 0
    total = 0

    # Desactiva cálculo de gradientes para inferencia
    with torch.no_grad():

        for images, labels in test_loader:

            # Predicción del modelo
            outputs = model(images)

            # Clase con mayor probabilidad
            _, predicted = torch.max(outputs, 1)

            # Conteo de muestras totales
            total += labels.size(0)

            # Conteo de predicciones correctas
            correct += (predicted == labels).sum().item()

    # Cálculo de exactitud final
    print(f"Test Exactitud: {100 * correct / total:.2f}%")

# Ejecución de evaluación
evaluate_model(model, test_loader)