"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: BLOQUE ENCODER TIPO TRANSFORMER (KERAS)
--------------------------------------------------

Descripción general:
Este script implementa un bloque simplificado de un Encoder Transformer utilizando
TensorFlow/Keras Functional API.

El objetivo es construir una arquitectura básica basada en:
- Multi-Head Self-Attention
- Conexiones residuales (Residual Connections)
- Normalización de capa (Layer Normalization)
- Feed-Forward Network (FFN)

Además, el modelo se visualiza mediante plot_model.

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Input Layer:
   - Recibe una secuencia con dimensión (None, input_dim)
   - None indica longitud variable de la secuencia

2. Multi-Head Attention:
   - Implementa atención auto-regresiva entre elementos de la secuencia
   - Parámetros:
     - num_heads: número de cabezas de atención
     - key_dim: dimensión de cada cabeza
   - Permite capturar múltiples relaciones en paralelo

3. Residual Connection (Add):
   - Suma la entrada original con la salida de atención
   - Ayuda a estabilizar el entrenamiento en redes profundas

4. Layer Normalization:
   - Normaliza activaciones por capa
   - Mejora estabilidad y convergencia

5. Feed-Forward Network (FFN):
   - Red completamente conectada aplicada a cada posición
   - Estructura:
     Dense(ff_dim, relu) → Dense(input_dim)

6. Segunda Residual Connection:
   - Combina salida del FFN con la salida del bloque de atención

7. Segunda Layer Normalization:
   - Normalización final del bloque encoder

--------------------------------------------------
FLUJO DEL ENCODER BLOCK
--------------------------------------------------

Input
  ↓
Multi-Head Attention
  ↓
Residual + Add
  ↓
Layer Normalization
  ↓
Feed Forward Network
  ↓
Residual + Add
  ↓
Layer Normalization
  ↓
Output

--------------------------------------------------
SALIDA DEL SCRIPT
--------------------------------------------------

- encoder_block:
  Modelo Keras funcional que representa un bloque Transformer Encoder

- plot_model:
  Genera una imagen del grafo del modelo:
  "transformer_encoder.png"

--------------------------------------------------
NOTA CONCEPTUAL
--------------------------------------------------

Este bloque es una versión simplificada del encoder de Transformer:

- No incluye positional encoding
- No incluye masking
- No incluye dropout
- No es un Transformer completo (solo un bloque encoder)

Sin embargo, representa la estructura base de:
"Attention is All You Need"

--------------------------------------------------
PARÁMETROS DEL MODELO
--------------------------------------------------

- input_dim:
  Dimensión del embedding de entrada (ej: 64)

- num_heads:
  Número de cabezas de atención (ej: 8)

- ff_dim:
  Dimensión interna del feed-forward network (ej: 128)
"""

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Add, MultiHeadAttention

# --------------------------------------------------
# BLOQUE ENCODER TRANSFORMER SIMPLIFICADO
# --------------------------------------------------
def transformer_encoder(input_dim, num_heads, ff_dim):
    """
    Construye un bloque Encoder tipo Transformer.

    Parámetros:
    input_dim (int): dimensión de entrada del embedding
    num_heads (int): número de cabezas de atención
    ff_dim (int): dimensión de la red feed-forward

    Retorno:
    Model: bloque encoder como modelo Keras funcional
    """

    # Entrada de la secuencia (longitud variable)
    inputs = Input(shape=(None, input_dim))

    # --------------------------------------------------
    # MULTI-HEAD SELF ATTENTION
    # --------------------------------------------------
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=input_dim
    )(inputs, inputs)

    # Residual connection
    attention_output = Add()([inputs, attention_output])

    # Normalización
    attention_output = LayerNormalization()(attention_output)

    # --------------------------------------------------
    # FEED FORWARD NETWORK
    # --------------------------------------------------
    ff_output = Dense(ff_dim, activation='relu')(attention_output)
    ff_output = Dense(input_dim)(ff_output)

    # Residual connection
    outputs = Add()([attention_output, ff_output])

    # Normalización final
    outputs = LayerNormalization()(outputs)

    return Model(inputs, outputs)

# --------------------------------------------------
# CREACIÓN DEL BLOQUE ENCODER
# --------------------------------------------------
encoder_block = transformer_encoder(
    input_dim=64,
    num_heads=8,
    ff_dim=128
)

# --------------------------------------------------
# VISUALIZACIÓN DEL MODELO
# --------------------------------------------------
plot_model(
    encoder_block,
    show_shapes=True,
    to_file="transformer_encoder.png"
)