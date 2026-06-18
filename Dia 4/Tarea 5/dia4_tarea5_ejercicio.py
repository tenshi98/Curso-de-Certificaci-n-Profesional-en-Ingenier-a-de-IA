"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: GENERACIÓN DE TEXTO CON GPT-2 (HUGGING FACE)
--------------------------------------------------

Descripción general:
Este script utiliza un modelo preentrenado GPT-2 (Generative Pretrained Transformer 2)
para generar texto a partir de un prompt inicial.

El objetivo es:
- Cargar un modelo causal de lenguaje
- Tokenizar una entrada textual
- Generar una secuencia de texto autoregresiva
- Decodificar la salida a lenguaje natural

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Modelo GPT-2:
   - Tipo: AutoModelForCausalLM
   - Arquitectura: Transformer decoder autoregresivo
   - Entrenado para predecir el siguiente token

2. Tokenizer:
   - Convierte texto en IDs de tokens
   - Maneja vocabulario subword (Byte Pair Encoding)
   - Permite codificación y decodificación del texto

3. Entrada del modelo:
   - input_ids: representación numérica del texto
   - attention_mask: indica tokens válidos vs padding

4. Generación de texto:
   - Método: model.generate()
   - Generación autoregresiva token a token
   - Basado en probabilidad del siguiente token

--------------------------------------------------
FLUJO DEL SCRIPT
--------------------------------------------------

1. Carga del modelo GPT-2 preentrenado
2. Carga del tokenizer asociado
3. Tokenización del prompt inicial
4. Creación de attention mask
5. Generación de secuencia con parámetros definidos
6. Decodificación del output a texto legible

--------------------------------------------------
PARÁMETROS IMPORTANTES DE GENERACIÓN
--------------------------------------------------

- max_length:
  Longitud máxima de la secuencia generada

- num_return_sequences:
  Número de secuencias generadas en paralelo

- pad_token_id:
  Token usado para padding (GPT-2 no lo define por defecto)

- attention_mask:
  Indica qué tokens deben ser atendidos

--------------------------------------------------
NOTA SOBRE GPT-2
--------------------------------------------------

- Modelo autoregresivo (causal)
- No bidireccional como BERT
- Genera texto de izquierda a derecha
- Basado en Transformer decoder

--------------------------------------------------
CASO DE USO
--------------------------------------------------

Este tipo de modelo se utiliza para:
- Generación de texto creativo
- Chatbots
- Continuación de historias
- Asistentes de escritura
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------
# CARGA DEL MODELO Y TOKENIZER GPT-2
# --------------------------------------------------
gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# --------------------------------------------------
# TEXTO DE ENTRADA
# --------------------------------------------------
input_text = "Once upon a time"

# Codificación del texto en IDs de tokens
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# --------------------------------------------------
# ATTENTION MASK
# --------------------------------------------------
# Marca tokens válidos (no padding)
attention_mask = input_ids.ne(tokenizer.pad_token_id)

# --------------------------------------------------
# CONFIGURACIÓN DE PAD TOKEN
# --------------------------------------------------
# GPT-2 no define pad_token_id por defecto, se maneja fallback
pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

# --------------------------------------------------
# GENERACIÓN DE TEXTO
# --------------------------------------------------
output = gpt_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=50,
    num_return_sequences=1,
    pad_token_id=pad_token_id
)

# --------------------------------------------------
# DECODIFICACIÓN DE SALIDA
# --------------------------------------------------
print("Generated Text:", tokenizer.decode(output[0], skip_special_tokens=True))