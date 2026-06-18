"""
DOCSTRING GENERAL DEL SCRIPT
------------------------------------------------------------
Título:
    Ejemplo de Max Pooling y Average Pooling en NumPy, TensorFlow y PyTorch

Descripción:
    Este script demuestra el funcionamiento de operaciones de pooling en redes neuronales
    convolucionales, utilizando implementaciones en NumPy (SciPy), TensorFlow y PyTorch.

    El flujo del código incluye:
    - Creación de un feature map sintético
    - Aplicación de Max Pooling y Average Pooling usando SciPy
    - Implementación de MaxPooling2D y AveragePooling2D en TensorFlow
    - Implementación de MaxPool2d y AvgPool2d en PyTorch
    - Definición de modelos CNN simples que incorporan capas de pooling

Entradas:
    - Feature map sintético (matriz 4x4)
    - Tensores derivados del feature map en TensorFlow y PyTorch

Salidas:
    - Resultados de max pooling y average pooling en NumPy
    - Tensores resultantes de pooling en TensorFlow
    - Tensores resultantes de pooling en PyTorch

Dependencias:
    - numpy
    - matplotlib
    - scipy
    - tensorflow
    - torch

------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, uniform_filter

# ------------------------------------------------------------
# FEATURE MAP SINTÉTICO
# ------------------------------------------------------------

# Create a sample feature map (2D matrix representing activations)
feature_map = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 1],
    [7, 8, 9, 2],
    [0, 1, 2, 3]
])

# ------------------------------------------------------------
# POOLING CON SCIPY (NUMPY)
# ------------------------------------------------------------

# Max pooling approximation using maximum filter
max_pooled = maximum_filter(feature_map, size=2, mode='constant')

# Average pooling approximation using uniform filter
avg_pooled = uniform_filter(feature_map, size=2, mode='constant')

# ------------------------------------------------------------
# POOLING CON TENSORFLOW
# ------------------------------------------------------------

import tensorflow as tf

# Reshape feature map to match TensorFlow input format
input_tensor = tf.constant(
    feature_map.reshape(1, 4, 4, 1),
    dtype=tf.float32
)

# Max Pooling layer (2x2 window, stride 2)
max_pool = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=2,
    padding='valid'
)

# Apply max pooling
max_pooled_tensor = max_pool(input_tensor)

# Average Pooling layer (2x2 window, stride 2)
avg_pool = tf.keras.layers.AveragePooling2D(
    pool_size=(2, 2),
    strides=2,
    padding='valid'
)

# Apply average pooling
avg_pooled_tensor = avg_pool(input_tensor)

# Display TensorFlow results
print(f"Max Pooled Tensor:\n{tf.squeeze(max_pooled_tensor).numpy()}")
print(f"Average Pooled Tensor:\n{tf.squeeze(avg_pooled_tensor).numpy()}")

print("\n\n\n")

# ------------------------------------------------------------
# POOLING CON PYTORCH
# ------------------------------------------------------------

import torch 
import torch.nn as nn

# Convert feature map to PyTorch tensor with batch and channel dimensions
input_tensor = torch.tensor(feature_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Max Pooling layer (2x2)
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
max_pooled_tensor = max_pool(input_tensor)

# Average Pooling layer (2x2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
avg_pooled_tensor = avg_pool(input_tensor)

# Display PyTorch results
print(f"Max Pooled Tensor:\n{max_pooled_tensor.squeeze().numpy()}")
print(f"Average Pooled Tensor:\n{avg_pooled_tensor.squeeze().numpy()}")

# ------------------------------------------------------------
# MODELO CNN EN TENSORFLOW CON POOLING
# ------------------------------------------------------------

# Sequential CNN model with pooling layers
model_tf = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),

    # Convolution + Max Pooling
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convolution + Average Pooling
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.AveragePooling2D((2, 2))
])

# ------------------------------------------------------------
# MODELO CNN EN PYTORCH CON POOLING
# ------------------------------------------------------------

class SimpleCNN(torch.nn.Module):
    """
    CNN simple con capas de convolución y pooling.

    Arquitectura:
        - Conv2d (3 -> 32)
        - MaxPool2d
        - Conv2d (32 -> 64)
        - AvgPool2d

    Nota:
        Contiene un error tipográfico intencional en kernel_size (kernal_size)
        según el código original proporcionado.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolution layer 1
        self.conv1 = nn.Conv2d(3, 32, kernal_size=3)

        # Max pooling layer
        self.pool1 = nn.MaxPool2d(2, 2)

        # Convolution layer 2
        self.conv2 = nn.Conv2d(32, 64, kernal_size=3)

        # Average pooling layer
        self.pool2 = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # Apply conv + ReLU + pooling sequence
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        return x