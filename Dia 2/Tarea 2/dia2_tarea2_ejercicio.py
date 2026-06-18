"""
DOCSTRING GENERAL DEL SCRIPT
------------------------------------------------------------
Título:
    Ejemplo de operaciones de convolución en imágenes con NumPy, TensorFlow y PyTorch

Descripción:
    Este script demuestra el concepto de convolución en procesamiento de imágenes,
    aplicando filtros manuales y capas convolucionales en distintos frameworks.

    Incluye tres niveles de implementación:
    1. Convolución manual usando SciPy sobre una imagen aleatoria en escala de grises.
    2. Convolución usando una capa Conv2D de TensorFlow.
    3. Convolución usando una capa Conv2d de PyTorch.

    Además, se muestran variaciones del comportamiento de convoluciones
    al modificar kernel, padding y stride.

Entradas:
    - Imagen sintética generada aleatoriamente (NumPy)
    - Tensores aleatorios en TensorFlow y PyTorch

Salidas:
    - Imágenes filtradas (edge detection y blur) en NumPy
    - Tensores de salida de capas convolucionales en TensorFlow
    - Tensores de salida de capas convolucionales en PyTorch

Dependencias:
    - numpy
    - matplotlib
    - scipy
    - tensorflow
    - torch

------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

# ------------------------------------------------------------
# CONVOLUCIÓN MANUAL CON KERNELS (NUMPY + SCIPY)
# ------------------------------------------------------------

# Load a sample grayscale image (synthetic random image)
image = np.random.rand(10, 10)

# Edge detection kernel (high-pass filter)
edge_detection_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])

# Blur kernel (low-pass filter)
blur_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9

# Apply convolution using SciPy
edge_detected_image = convolve(image, edge_detection_kernel)
blurred_image = convolve(image, blur_kernel)

# ------------------------------------------------------------
# CONVOLUCIÓN CON TENSORFLOW
# ------------------------------------------------------------

import tensorflow as tf

# Create a random input tensor (batch, height, width, channels)
image_tensor = tf.random.normal([1, 10, 10, 1])

# Define a Conv2D layer
conv_layer = tf.keras.layers.Conv2D(
    filters=1,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same'
)

# Apply convolution layer
output_tensor = conv_layer(image_tensor)

# Display tensor shapes
print(f"Original Shape: {image_tensor.shape}")
print(f"Output Shape: {output_tensor.shape}")

# ------------------------------------------------------------
# CONVOLUCIÓN CON PYTORCH
# ------------------------------------------------------------

import torch
import torch.nn as nn

# Create a random input tensor (batch, channels, height, width)
image_tensor_pt = torch.randn(1, 1, 10, 10)

# Define a Conv2d layer
conv_layer_pt = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=1
)

# Apply convolution
output_tensor_pt = conv_layer_pt(image_tensor_pt)

# Display tensor shapes
print(f"Original Shape: {image_tensor_pt.shape}")
print(f"Output Shape: {output_tensor_pt.shape}")

# ------------------------------------------------------------
# VARIACIÓN: KERNEL MÁS GRANDE (TENSORFLOW)
# ------------------------------------------------------------

conv_layer_large_kernel = tf.keras.layers.Conv2D(
    filters=1,
    kernel_size=(5, 5),
    strides=(1, 1),
    padding="same"
)

output_large_kernel = conv_layer_large_kernel(image_tensor)

print(f"Large Kernel Output Shape: {output_large_kernel.shape}")

# ------------------------------------------------------------
# VARIACIÓN: STRIDE EN PYTORCH
# ------------------------------------------------------------

conv_layer_stride_2 = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=2,
    padding=1
)

output_stride_2 = conv_layer_stride_2(image_tensor_pt)

print(f"Stride Output Shape: {output_stride_2.shape}")