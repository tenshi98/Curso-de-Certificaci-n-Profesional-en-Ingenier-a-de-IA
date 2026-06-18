"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: POSITIONAL ENCODING + TRANSFORMER EN PYTORCH
--------------------------------------------------

Descripción general:
Este script implementa y visualiza el mecanismo de Positional Encoding utilizado en Transformers,
junto con una versión simplificada de un modelo Transformer en PyTorch que incorpora:

- Embeddings
- Positional Encoding (fijo y learnable)
- Multi-Head Attention
- Feed Forward Network
- Layer Normalization

El objetivo es mostrar cómo los Transformers incorporan información de orden en secuencias.

--------------------------------------------------
PARTE 1: POSITIONAL ENCODING (NUMPY)
--------------------------------------------------

1. Función: positional_encoding(seq_len, embed_dim)

Propósito:
- Generar vectores que representan la posición de cada token en la secuencia
- Permitir que el modelo capture orden secuencial

--------------------------------------------------
2. Cálculo matemático
--------------------------------------------------

Se construye una matriz basada en la fórmula original del Transformer:

PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

Componentes:
- pos: posición en la secuencia
- i: dimensión del embedding
- angle_rates: factor de escalamiento exponencial

--------------------------------------------------
3. Visualización
--------------------------------------------------

- Se utiliza un heatmap (pcolormesh) para visualizar:
  - posiciones (eje Y)
  - dimensiones del embedding (eje X)
- Permite observar patrones sinusoidales alternados

--------------------------------------------------
PARTE 2: TRANSFORMER CON POSITIONAL ENCODING (PYTORCH)
--------------------------------------------------

Clase: TransformerWithPositionalEncoding

--------------------------------------------------
1. Componentes del modelo
--------------------------------------------------

- Embedding:
  Convierte índices de tokens en vectores densos

- Positional Encoding:
  Matriz fija aprendida como parámetro del modelo
  Se suma directamente al embedding

- MultiheadAttention:
  Módulo de atención multi-cabeza de PyTorch

- Feed Forward Network (FFN):
  Red completamente conectada aplicada por token:
  Linear → ReLU → Linear

- LayerNorm:
  Normalización para estabilizar entrenamiento

--------------------------------------------------
2. Flujo del forward pass
--------------------------------------------------

Input tokens
   ↓
Embedding + Positional Encoding
   ↓
Multi-Head Attention
   ↓
Residual + LayerNorm
   ↓
Feed Forward Network
   ↓
Residual + LayerNorm
   ↓
Output

--------------------------------------------------
3. Positional Encoding en el modelo
--------------------------------------------------

x = self.embedding(x) + self.positional_encoding

- Se suma directamente la información posicional
- Permite que el modelo distinga orden de tokens

--------------------------------------------------
PARTE 3: POSITIONAL ENCODING APRENDIBLE
--------------------------------------------------

Clase: LearnablePositionalEncoding

--------------------------------------------------
1. Descripción:
- Alternativa al encoding sinusoidal fijo
- Representa posiciones como parámetros entrenables

2. Características:
- nn.Parameter inicializado en cero
- Se ajusta durante entrenamiento
- Aprende representaciones óptimas de posición

--------------------------------------------------
COMPARACIÓN CONCEPTUAL
--------------------------------------------------

Positional Encoding fijo:
- Determinístico
- Basado en funciones sinusoidales
- No entrenable

Positional Encoding aprendible:
- Parámetro del modelo
- Ajustado por gradiente descendente
- Más flexible pero menos interpretable

--------------------------------------------------
SALIDAS DEL SCRIPT
--------------------------------------------------

1. pos_encoding:
   Matriz (seq_len, embed_dim) con valores sinusoidales

2. Visualización:
   Heatmap del encoding posicional

3. model:
   Transformer simplificado con encoding fijo

4. learnable_pe:
   Módulo de encoding posicional entrenable

--------------------------------------------------
NOTA CONCEPTUAL
--------------------------------------------------

El positional encoding es esencial porque:
- Transformers no tienen recurrencia
- No capturan orden de forma natural
- Esta técnica introduce información de posición explícita

--------------------------------------------------
USO PRINCIPAL
--------------------------------------------------

- NLP (traducción, clasificación)
- Vision Transformers (ViT)
- Modelos secuenciales en general
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# POSITIONAL ENCODING (NUMPY)
# --------------------------------------------------
def positional_encoding(seq_len, embed_dim):
    """
    Genera matriz de positional encoding sinusoidal.

    Parámetros:
    seq_len (int): longitud de la secuencia
    embed_dim (int): dimensión del embedding

    Retorno:
    np.array: matriz (seq_len, embed_dim)
    """

    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(embed_dim)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (i // 2)) / np.float32(embed_dim)
    )

    angle_rads = pos * angle_rates

    pos_encoding = np.zeros(angle_rads.shape)

    # Sine en dimensiones pares
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Cosine en dimensiones impares
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return pos_encoding

# --------------------------------------------------
# GENERACIÓN DE POSITIONAL ENCODING
# --------------------------------------------------
seq_len = 50
embed_dim = 16

pos_encoding = positional_encoding(seq_len, embed_dim)

# --------------------------------------------------
# VISUALIZACIÓN
# --------------------------------------------------
plt.figure(figsize=(10, 6))
plt.pcolormesh(pos_encoding, cmap='viridis')
plt.colorbar()
plt.title("Positional Encoding")
plt.xlabel("Embedding Dimension")
plt.ylabel("Position")
plt.show()

# --------------------------------------------------
# TRANSFORMER CON POSITIONAL ENCODING
# --------------------------------------------------

import torch
import torch.nn as nn

class TransformerWithPositionalEncoding(nn.Module):
    """
    Transformer simplificado con positional encoding fijo.
    """

    def __init__(self, embed_dim, seq_len, num_heads, ff_dim):
        super(TransformerWithPositionalEncoding, self).__init__()

        # Embedding de tokens
        self.embedding = nn.Embedding(seq_len, embed_dim)

        # Positional encoding fijo como parámetro
        self.positional_encoding = nn.Parameter(
            torch.tensor(
                positional_encoding(seq_len, embed_dim),
                dtype=torch.float32
            )
        )

        # Multi-head attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads
        )

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # Normalización
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass del Transformer.

        Parámetros:
        x (Tensor): secuencia de tokens

        Retorno:
        Tensor: representación transformada
        """

        # Embedding + positional encoding
        x = self.embedding(x) + self.positional_encoding

        # Self-attention
        attn_output, _ = self.multihead_attention(x, x, x)

        # Residual + normalization
        x = self.norm1(x + attn_output)

        # Feed forward
        ffn_output = self.ffn(x)

        # Residual + normalization
        x = self.norm2(x + ffn_output)

        return x

# --------------------------------------------------
# INSTANCIACIÓN DEL MODELO
# --------------------------------------------------
embed_dim = 16
seq_len = 50
num_heads = 4
ff_dim = 64

model = TransformerWithPositionalEncoding(
    embed_dim,
    seq_len,
    num_heads,
    ff_dim
)

print(model)

# --------------------------------------------------
# POSITIONAL ENCODING APRENDIBLE
# --------------------------------------------------

class LearnablePositionalEncoding(nn.Module):
    """
    Positional encoding entrenable como parámetro del modelo.
    """

    def __init__(self, seq_len, embed_dim):
        super(LearnablePositionalEncoding, self).__init__()

        self.positional_encoding = nn.Parameter(
            torch.zeros(seq_len, embed_dim)
        )

    def forward(self, x):
        return x + self.positional_encoding

# Instancia del encoding aprendible
learnable_pe = LearnablePositionalEncoding(seq_len, embed_dim)

print(learnable_pe)