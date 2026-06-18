"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: CNN MEJORADA CON CIFAR-10 (PYTORCH)
--------------------------------------------------

Descripción general:
Este script implementa una red neuronal convolucional (CNN) mejorada en PyTorch para la
clasificación del dataset CIFAR-10.

El pipeline incluye:
- Carga del dataset CIFAR-10
- Data augmentation para entrenamiento
- Normalización de imágenes
- Definición de una CNN con Batch Normalization y Dropout
- Cálculo dinámico de dimensiones de capas fully connected
- Entrenamiento del modelo
- Evaluación en conjunto de prueba
- Visualización de la curva de pérdida

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Transformaciones:
   - Train: RandomHorizontalFlip + RandomCrop + Normalización
   - Test: solo normalización
   - Objetivo: mejorar generalización del modelo

2. Dataset:
   - CIFAR-10: 60,000 imágenes RGB (32x32) en 10 clases
   - Separación en entrenamiento y prueba

3. Modelo CNN:
   - 2 capas convolucionales
   - Batch Normalization en cada bloque conv
   - MaxPooling para reducción dimensional
   - Dropout para regularización
   - Cálculo dinámico del tamaño de entrada a fully connected

4. Entrenamiento:
   - Loss: CrossEntropyLoss
   - Optimizer: Adam
   - 20 épocas de entrenamiento
   - Registro de pérdida por época

5. Evaluación:
   - Accuracy sobre conjunto de test

6. Visualización:
   - Curva de pérdida de entrenamiento por época

--------------------------------------------------
COMPONENTE PRINCIPAL DEL MODELO
--------------------------------------------------

- Clase EnhancedCNN:
  Red neuronal convolucional con normalización por batch, dropout
  y cálculo automático de dimensionalidad de salida convolucional.

- Método _calculate_conv_output:
  Determina el tamaño del tensor de salida de las capas convolucionales
  utilizando un tensor ficticio (dummy input), evitando cálculos manuales.

"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --------------------------------------------------
# TRANSFORMACIONES CON DATA AUGMENTATION (ENTRENAMIENTO)
# --------------------------------------------------
# Aplicación de transformaciones aleatorias para aumentar variabilidad del dataset
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Volteo horizontal aleatorio
    transforms.RandomCrop(32, padding=4),  # Recorte aleatorio con padding
    transforms.ToTensor(),  # Conversión a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalización por canal
])

# --------------------------------------------------
# TRANSFORMACIONES PARA TEST (SIN AUGMENTATION)
# --------------------------------------------------
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --------------------------------------------------
# CARGA DEL DATASET CIFAR-10
# --------------------------------------------------
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

# --------------------------------------------------
# DATA LOADERS
# --------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Impresión de tamaños de dataset
print(f"Training Data Size: {len(train_dataset)}")
print(f"Test Data Size: {len(test_dataset)}")

# --------------------------------------------------
# DEFINICIÓN DEL MODELO CNN MEJORADO
# --------------------------------------------------
class EnhancedCNN(nn.Module):
    """
    Red neuronal convolucional mejorada para clasificación CIFAR-10.

    Características:
    - Batch Normalization
    - Dropout para regularización
    - Cálculo dinámico de dimensión de entrada a capas fully connected
    """

    def __init__(self):
        super(EnhancedCNN, self).__init__()

        # Capa convolucional 1 (RGB → 6 filtros)
        self.conv1 = nn.Conv2d(3, 6, 5)

        # Batch normalization para estabilizar activaciones
        self.bn1 = nn.BatchNorm2d(6)

        # Capa convolucional 2
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Batch normalization segunda capa
        self.bn2 = nn.BatchNorm2d(16)

        # Max pooling para reducción espacial
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout para evitar overfitting
        self.dropout = nn.Dropout(0.5)

        # --------------------------------------------------
        # CÁLCULO DINÁMICO DEL TAMAÑO DE SALIDA CONV
        # --------------------------------------------------
        self._calculate_conv_output()

        # Capas fully connected basadas en tamaño calculado
        self.fc1 = nn.Linear(self.conv_output_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def _calculate_conv_output(self):
        """
        Calcula automáticamente el tamaño de salida de las capas convolucionales
        utilizando un tensor dummy con el tamaño de entrada CIFAR-10 (3x32x32).

        Este valor es necesario para definir correctamente la primera capa fully connected.
        """

        # Tensor ficticio con tamaño de imagen CIFAR-10
        dummy_input = torch.zeros(1, 3, 32, 32)

        # Forward pass sin gradientes para obtener dimensión de salida
        with torch.no_grad():
            output = self.pool(
                F.relu(
                    self.bn2(
                        self.conv2(
                            F.relu(
                                self.bn1(
                                    self.conv1(dummy_input)
                                )
                            )
                        )
                    )
                )
            )

        # Número total de elementos del tensor de salida
        self.conv_output_size = output.numel()

    def forward(self, x):
        """
        Propagación hacia adelante del modelo.

        Parámetros:
        x (Tensor): batch de imágenes CIFAR-10

        Retorno:
        Tensor: logits de salida para 10 clases
        """

        # Bloque convolucional 1
        x = F.relu(self.bn1(self.conv1(x)))

        # Bloque convolucional 2 + pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Aplanamiento para fully connected
        x = x.view(x.size(0), -1)

        # Capas densas
        x = F.relu(self.fc1(x))

        # Regularización dropout
        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        # Capa de salida (logits)
        x = self.fc3(x)

        return x

# Instanciación del modelo
model = EnhancedCNN()
print(model)

# --------------------------------------------------
# FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Registro de pérdida por época
training_loss = []

# --------------------------------------------------
# FUNCIÓN DE ENTRENAMIENTO
# --------------------------------------------------
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    """
    Entrena el modelo CNN mejorado.

    Parámetros:
    model (nn.Module): modelo a entrenar
    train_loader (DataLoader): datos de entrenamiento
    criterion: función de pérdida
    optimizer: optimizador (Adam)
    epochs (int): número de épocas

    Retorno:
    None
    """

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

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

            running_loss += loss.item()

        # Promedio de pérdida por época
        epoch_loss = running_loss / len(train_loader)
        training_loss.append(epoch_loss)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# Entrenamiento del modelo
train_model(model, train_loader, criterion, optimizer)

# --------------------------------------------------
# FUNCIÓN DE EVALUACIÓN
# --------------------------------------------------
def evaluate_model(model, test_loader):
    """
    Evalúa el modelo entrenado en el conjunto de test.

    Parámetros:
    model (nn.Module): modelo entrenado
    test_loader (DataLoader): datos de prueba

    Retorno:
    None (imprime accuracy)
    """

    model.eval()

    correct = 0
    total = 0

    # Inferencia sin gradientes
    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            # Predicción clase con mayor score
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print(f"Test Exactitud: {100 * correct / total:.2f}%")

# Evaluación del modelo
evaluate_model(model, test_loader)

# --------------------------------------------------
# VISUALIZACIÓN DE LA PÉRDIDA
# --------------------------------------------------
import matplotlib.pylab as plt

plt.plot(training_loss, label="Training Loss")
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()