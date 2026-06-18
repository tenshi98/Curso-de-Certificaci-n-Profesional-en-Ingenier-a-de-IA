"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: MECANISMO DE SELF-ATTENTION (NUMPY)
--------------------------------------------------

Descripción general:
Este script implementa una versión simplificada del mecanismo de atención (Attention Mechanism)
utilizado en modelos de Deep Learning como Transformers.

El objetivo es calcular representaciones contextuales (context vectors) a partir de:
- Queries (Q)
- Keys (K)
- Values (V)

El proceso simula cómo un modelo asigna importancia relativa a diferentes elementos de entrada.

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Queries (Q):
   - Representan las consultas que se hacen sobre la información disponible
   - Dimensión: (n_queries, d_model)

2. Keys (K):
   - Representan las características de los elementos a comparar
   - Dimensión: (n_keys, d_model)

3. Values (V):
   - Representan la información real que será agregada ponderadamente
   - Dimensión: (n_keys, d_value)

--------------------------------------------------
FLUJO DEL MECANISMO DE ATENCIÓN
--------------------------------------------------

1. Cálculo de scores:
   - Se calcula el producto punto entre Queries y Keys transpuestas
   - Esto mide la similitud entre cada query y cada key

2. Normalización (Softmax):
   - Convierte los scores en probabilidades
   - Asegura que los pesos sumen 1 por fila
   - Destaca las relaciones más relevantes

3. Cálculo del contexto:
   - Se realiza una suma ponderada de los Values
   - Los pesos vienen de la matriz de atención (attention weights)

--------------------------------------------------
SALIDAS DEL SCRIPT
--------------------------------------------------

- Attention Weights:
  Matriz que indica la importancia relativa de cada key para cada query

- Context Vector:
  Representación final combinada que contiene información relevante
  de los values ponderada por la atención

--------------------------------------------------
NOTA CONCEPTUAL
--------------------------------------------------

Este código representa una versión simplificada del mecanismo de atención:

Attention(Q, K, V) = softmax(Q · K^T) · V

No incluye:
- Escalamiento por sqrt(d_k)
- Multi-head attention
- Masking (causal o padding)

"""

import numpy as np

# --------------------------------------------------
# DEFINICIÓN DE QUERIES, KEYS Y VALUES
# --------------------------------------------------
# Q: Representa consultas
queries = np.array([[1, 0, 1], [0, 1, 1]])

# K: Representa claves
keys = np.array([
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1]
])

# V: Representa valores asociados a cada key
values = np.array([
    [10, 0],
    [0, 10],
    [5, 5]
])

# --------------------------------------------------
# CÁLCULO DE SCORE DE ATENCIÓN
# --------------------------------------------------
# Producto punto entre queries y keys
scores = np.dot(queries, keys.T)

# --------------------------------------------------
# FUNCIÓN SOFTMAX
# --------------------------------------------------
def softmax(x):
    """
    Normaliza un vector o matriz en probabilidades.

    Parámetros:
    x (np.array): matriz de scores

    Retorno:
    np.array: probabilidades normalizadas
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

# --------------------------------------------------
# CÁLCULO DE PESOS DE ATENCIÓN
# --------------------------------------------------
attention_weights = softmax(scores)

# --------------------------------------------------
# CÁLCULO DEL VECTOR DE CONTEXTO
# --------------------------------------------------
context = np.dot(attention_weights, values)

# --------------------------------------------------
# SALIDA DEL MODELO
# --------------------------------------------------
print("Attention Weights: \n", attention_weights)
print("Context Vector:\n", context)