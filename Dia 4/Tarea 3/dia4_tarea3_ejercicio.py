"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: SELF-ATTENTION + MULTI-HEAD ATTENTION (NUMPY + PYTORCH)
--------------------------------------------------

Descripción general:
Este script combina dos implementaciones del mecanismo de atención:

1. Una versión básica de Self-Attention implementada con NumPy
2. Una implementación de Multi-Head Attention en PyTorch con visualización

El objetivo es:
- Comprender el cálculo matemático de attention
- Implementar multi-head attention desde cero
- Visualizar los pesos de atención

--------------------------------------------------
PARTE 1: SELF-ATTENTION (NUMPY)
--------------------------------------------------

1. Generación de datos:
   - Se crean matrices aleatorias para:
     - Query (Q)
     - Key (K)
     - Value (V)

2. Cálculo de scores:
   - Fórmula:
     Q · K^T / sqrt(d_k)
   - Se aplica escalamiento para estabilizar gradientes

3. Softmax:
   - Convierte scores en probabilidades
   - Normaliza importancia relativa entre tokens

4. Context vector:
   - Suma ponderada de values usando attention weights
   - Representa la salida contextualizada

--------------------------------------------------
PARTE 2: MULTI-HEAD ATTENTION (PYTORCH)
--------------------------------------------------

Clase: MultiHeadAttention(nn.Module)

--------------------------------------------------
1. Inicialización del módulo
--------------------------------------------------

Parámetros:
- embed_dim: dimensión total del embedding
- num_heads: número de cabezas de atención
- head_dim: embed_dim / num_heads

Componentes:
- self.query: proyección lineal de queries
- self.key: proyección lineal de keys
- self.value: proyección lineal de values
- self.out: proyección final de salida

--------------------------------------------------
2. Forward Pass
--------------------------------------------------

Paso 1: Proyecciones lineales
- Se transforma la entrada en Q, K, V

Paso 2: Reshape para multi-head
- (batch, seq, embed_dim)
  → (batch, heads, seq, head_dim)

Paso 3: Cálculo de atención
- scores = Q · K^T / sqrt(head_dim)
- softmax para obtener pesos de atención

Paso 4: Context computation
- Se multiplica atención por values
- Se concatena heads nuevamente

Paso 5: Proyección final
- Se aplica capa lineal de salida

--------------------------------------------------
SALIDAS DEL MODELO
--------------------------------------------------

- attention_weights:
  Tensor con forma:
  (batch_size, num_heads, seq_len, seq_len)

- context vector:
  Tensor con información contextual agregada

--------------------------------------------------
PARTE 3: VISUALIZACIÓN
--------------------------------------------------

- Se visualiza la matriz de atención de un head específico
- Usa matplotlib heatmap (matshow)
- Permite interpretar:
  - qué tokens prestan atención a otros tokens
  - patrones de dependencia dentro de la secuencia

--------------------------------------------------
NOTA CONCEPTUAL
--------------------------------------------------

Este código implementa la fórmula general:

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

Multi-Head Attention extiende esto mediante:
- múltiples subespacios de representación
- atención paralela en distintas “cabezas”
- concatenación final de resultados

--------------------------------------------------
LIMITACIONES DEL EJEMPLO
--------------------------------------------------

- No incluye masking (causal/padding)
- No incluye dropout
- No forma parte de un Transformer completo
- Usa datos aleatorios (no entrenamiento real)

--------------------------------------------------
USO EDUCATIVO
--------------------------------------------------

Este script es útil para:
- Entender atención desde cero
- Visualizar comportamiento de heads
- Introducción a Transformers
"""

import numpy as np

# --------------------------------------------------
# GENERACIÓN DE DATOS (QUERY, KEY, VALUE)
# --------------------------------------------------
def generate_data(seq_len, embed_dim):
    """
    Genera matriz aleatoria para simulación de embeddings.

    Parámetros:
    seq_len (int): longitud de la secuencia
    embed_dim (int): dimensión del embedding

    Retorno:
    np.array: matriz (seq_len, embed_dim)
    """
    np.random.seed(42)
    return np.random.rand(seq_len, embed_dim)

sequence_length = 4
embedding_dim = 3

query = generate_data(sequence_length, embedding_dim)
key = generate_data(sequence_length, embedding_dim)
value = generate_data(sequence_length, embedding_dim)

# --------------------------------------------------
# SELF-ATTENTION (NUMPY)
# --------------------------------------------------

# Cálculo de scores escalados
scores = np.dot(query, key.T) / np.sqrt(embedding_dim)

# Función softmax
def softmax(x):
    """
    Convierte scores en distribución de probabilidad.

    Parámetros:
    x (np.array): matriz de scores

    Retorno:
    np.array: probabilidades normalizadas
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

# Pesos de atención
attention_weights = softmax(scores)

# Vector de contexto
context = np.dot(attention_weights, value)

# --------------------------------------------------
# MULTI-HEAD ATTENTION (PYTORCH)
# --------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Implementación básica de Multi-Head Attention en PyTorch.
    """

    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Validación de divisibilidad
        assert embed_dim % num_heads == 0, \
            "Embedding dimension must be divisible by number of heads."

        # Proyecciones lineales
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Capa final de salida
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass de multi-head attention.

        Parámetros:
        x (Tensor): entrada (batch, seq_len, embed_dim)

        Retorno:
        tuple:
            - output transformado
            - attention weights por cabeza
        """

        batch_size = x.size(0)

        # --------------------------------------------------
        # PROYECCIONES LINEALES
        # --------------------------------------------------
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # --------------------------------------------------
        # CÁLCULO DE SCORES
        # --------------------------------------------------
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Softmax sobre última dimensión
        attention_weights = F.softmax(scores, dim=-1)

        # --------------------------------------------------
        # CONTEXT VECTOR
        # --------------------------------------------------
        context = torch.matmul(attention_weights, v)

        # Reorganización de dimensiones
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Proyección final
        return self.out(context), attention_weights

# --------------------------------------------------
# PRUEBA DEL MODELO
# --------------------------------------------------

seq_len, embed_dim = 4, 8
x = torch.rand(1, seq_len, embed_dim)

mha = MultiHeadAttention(embed_dim, num_heads=2)

context, attn_weights = mha(x)

print("Attention Weights:\n", attn_weights)
print("Context Vector:\n", context)

# --------------------------------------------------
# VISUALIZACIÓN
# --------------------------------------------------

import matplotlib.pyplot as plt

plt.matshow(attn_weights[0, 0].detach().numpy(), cmap='viridis')
plt.colorbar()
plt.title("Attention Weights (Head 1)")
plt.show()