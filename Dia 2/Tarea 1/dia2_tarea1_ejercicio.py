"""
DOCSTRING GENERAL DEL SCRIPT
------------------------------------------------------------
Título:
    Definición de modelos CNN en TensorFlow y PyTorch para clasificación de imágenes

Descripción:
    Este script muestra la creación de modelos de redes neuronales convolucionales (CNN)
    simples utilizando dos frameworks distintos: TensorFlow (Keras) y PyTorch.

    Incluye además (comentado):
    - Carga y visualización del dataset CIFAR-10 usando torchvision
    - Visualización de imágenes y sus etiquetas
    - Inspección de valores de píxeles de una imagen

    Luego se definen dos modelos:
    - Un modelo CNN en TensorFlow Keras
    - Un modelo CNN en PyTorch (clase personalizada)

Entradas:
    - No hay entradas activas en el flujo actual (dataset está comentado)

Salidas:
    - Definición de modelos CNN
    - Mensajes de confirmación en consola

Dependencias:
    - matplotlib
    - torchvision
    - numpy
    - tensorflow
    - torch
    - torch.nn

------------------------------------------------------------
"""

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

# ------------------------------------------------------------
# CARGA Y VISUALIZACIÓN DEL DATASET (COMENTADO)
# ------------------------------------------------------------

# Define transformation to convert images into tensors
transform = transforms.ToTensor()

# Load CIFAR-10 dataset (currently commented)
# train_dataset = datasets.CIFAR10(
#     root='./data',
#     train=True,
#     transform=transform,
#     download=True
# )

# Visualize sample images (currently commented)
# fig, axes = plt.subplots(1, 5, figsize=(12,3))
# for i in range(5):
#     image, label = train_dataset[i]
#     axes[i].imshow(image.permute(1, 2, 0))
#     axes[i].axis('off')
#     axes[i].set_title(f"Label: {label}")
# plt.show()

# Display pixel values of first image (commented)
# image, label = train_dataset[0]
# print(f"Label: {label}")
# print(f"Image Shape: {image.shape}")
# print("Pixel Values:")
# print(image)

# ------------------------------------------------------------
# MODELO CNN EN TENSORFLOW
# ------------------------------------------------------------

import tensorflow as tf

# Define a sequential CNN model using Keras API
model = tf.keras.Sequential([
    # Convolutional layer extracting 32 feature maps
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),

    # Downsampling layer reducing spatial dimensions
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten feature maps into 1D vector
    tf.keras.layers.Flatten(),

    # Fully connected hidden layer
    tf.keras.layers.Dense(128, activation="relu"),

    # Output layer for 10-class classification
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile model with optimizer and loss function
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Confirmation message
print("Tensorflow CNN Model is ready")

# ------------------------------------------------------------
# MODELO CNN EN PYTORCH
# ------------------------------------------------------------

import torch.nn as nn

# Define a simple CNN architecture in PyTorch
class SimpleCNN(nn.Module):
    """
    Simple CNN model for image classification.

    Architecture:
        - Conv2D layer (3 -> 32 filters)
        - ReLU activation
        - MaxPooling layer (2x2)
        - Fully connected layer (32 * 15 * 15 -> 128)
        - Fully connected output layer (128 -> 10 classes)

    Methods:
        forward(x): Defines forward propagation pass

    Note:
        This model assumes input images of shape (3, 32, 32)
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layer extracting low-level features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)

        # Max pooling reduces spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layer
        self.fc1 = nn.Linear(32 * 15 * 15, 128)

        # Output layer for classification into 10 classes
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Apply convolution + activation
        x = F.relu(self.conv1(x))

        # Apply pooling
        x = self.pool(x)

        # Flatten tensor for dense layers
        x = x.view(-1, 32 * 15 * 15)

        # Fully connected layer with activation
        x = F.relu(self.fc1(x))

        # Output layer (logits)
        x = self.fc2(x)

        return x

# Confirmation message
print("PyTorch CNN model ready")