"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: SELF-ATTENTION (PYTORCH + VISUALIZACIÓN)
--------------------------------------------------

Descripción general:
Este script implementa un mecanismo básico de atención (attention mechanism)
utilizando PyTorch.

El objetivo es calcular:
- Matriz de pesos de atención (attention weights)
- Vector de contexto (context vector)
y visualizar cómo se distribuye la atención entre elementos.

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Queries (Q):
   - Representan las consultas que se comparan contra las keys
   - Dimensión: (n_queries, d_model)

2. Keys (K):
   - Representan los elementos con los que se compara cada query
   - Dimensión: (n_keys, d_model)

3. Values (V):
   - Representan la información que será agregada según la atención
   - Dimensión: (n_keys, d_value)

--------------------------------------------------
FLUJO DEL MECANISMO DE ATENCIÓN
--------------------------------------------------

1. Cálculo de scores:
   - Se calcula el producto punto entre queries y keys transpuestas
   - Mide la similitud entre cada query y cada key

2. Normalización (Softmax):
   - Se aplica softmax sobre la última dimensión
   - Convierte los scores en probabilidades (pesos de atención)

3. Cálculo del contexto:
   - Se realiza una suma ponderada de los values
   - Cada query genera su propio vector de contexto

--------------------------------------------------
VISUALIZACIÓN
--------------------------------------------------

- Se utiliza matplotlib para visualizar la matriz de atención
- La gráfica tipo heatmap permite observar:
  - Qué keys reciben mayor atención por cada query
  - Distribución de importancia entre elementos

--------------------------------------------------
SALIDAS DEL SCRIPT
--------------------------------------------------

- Attention Weights:
  Matriz de probabilidades que indica la relevancia relativa
  entre queries y keys

- Context Vector:
  Representación final agregada según la atención

- Heatmap:
  Visualización gráfica de la matriz de atención

--------------------------------------------------
NOTA CONCEPTUAL
--------------------------------------------------

Este código representa una forma básica de attention:

Attention(Q, K, V) = softmax(Q · K^T) · V

No incluye:
- Escalamiento por sqrt(d_k)
- Multi-head attention
- Masking
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --------------------------------------------------
# DEFINICIÓN DE QUERIES, KEYS Y VALUES
# --------------------------------------------------
queries = torch.tensor([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0]
])

keys = torch.tensor([
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0]
])

values = torch.tensor([
    [10.0, 0.0],
    [0.0, 10.0],
    [5.0, 5.0]
])

# --------------------------------------------------
# CÁLCULO DE SCORES DE ATENCIÓN
# --------------------------------------------------
# Producto punto entre queries y keys
scores = torch.matmul(queries, keys.T)

# --------------------------------------------------
# NORMALIZACIÓN CON SOFTMAX
# --------------------------------------------------
# Convierte scores en distribución de probabilidad
attention_weights = F.softmax(scores, dim=-1)

# --------------------------------------------------
# CÁLCULO DEL VECTOR DE CONTEXTO
# --------------------------------------------------
# Suma ponderada de values según atención
context = torch.matmul(attention_weights, values)

# --------------------------------------------------
# SALIDA DEL MODELO
# --------------------------------------------------
print("Attention Weights: \n", attention_weights)
print("Context Vector:\n", context)

# --------------------------------------------------
# VISUALIZACIÓN DE ATENCIÓN
# --------------------------------------------------
# Heatmap de la matriz de atención
plt.matshow(attention_weights)
plt.colorbar()

plt.title("Attention Weights")

plt.show()