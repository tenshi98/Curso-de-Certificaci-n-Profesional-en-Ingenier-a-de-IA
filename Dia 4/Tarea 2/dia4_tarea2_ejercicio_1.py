"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: USO DE BERT CON HUGGING FACE TRANSFORMERS
--------------------------------------------------

Descripción general:
Este script utiliza un modelo preentrenado BERT (Bidirectional Encoder Representations from Transformers)
a través de la librería Hugging Face Transformers para obtener representaciones contextuales de texto.

El objetivo es:
- Tokenizar una oración de entrada
- Procesarla con BERT
- Obtener los embeddings contextuales (hidden states)

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Tokenizer (BERT Tokenizer):
   - Convierte texto en tokens compatibles con el modelo BERT
   - Aplica subword tokenization (WordPiece)
   - Genera tensores PyTorch como entrada

2. Modelo BERT:
   - Modelo preentrenado "bert-base-uncased"
   - Arquitectura Transformer bidireccional
   - Genera representaciones contextuales para cada token

3. Entrada del modelo:
   - Texto convertido a tokens
   - Incluye automáticamente:
     - input_ids
     - attention_mask (implícito en el tokenizer)

4. Salida del modelo:
   - last_hidden_state: tensor con embeddings contextuales por token

--------------------------------------------------
FLUJO DEL SCRIPT
--------------------------------------------------

1. Carga del tokenizer preentrenado BERT
2. Carga del modelo BERT preentrenado
3. Tokenización del texto de entrada
4. Conversión a tensores PyTorch
5. Forward pass a través del modelo
6. Extracción de embeddings contextuales

--------------------------------------------------
SALIDAS DEL SCRIPT
--------------------------------------------------

- Hidden States Shape:
  Tensor con forma:
  (batch_size, sequence_length, hidden_size)

  Donde:
  - batch_size: número de textos procesados simultáneamente
  - sequence_length: número de tokens generados por el tokenizer
  - hidden_size: dimensión del embedding (768 en bert-base-uncased)

--------------------------------------------------
NOTA CONCEPTUAL
--------------------------------------------------

BERT genera embeddings contextuales, lo que significa que:
- Cada palabra tiene una representación distinta según su contexto
- No es un embedding estático como Word2Vec o GloVe

--------------------------------------------------
MODELO UTILIZADO
--------------------------------------------------

- bert-base-uncased:
  - 12 capas Transformer
  - 768 dimensiones ocultas
  - Lowercase text (uncased)
  - Preentrenado en BookCorpus + Wikipedia

--------------------------------------------------
USO PRINCIPAL
--------------------------------------------------

Este tipo de modelo se utiliza en tareas como:
- Clasificación de texto
- QA (Question Answering)
- NER (Named Entity Recognition)
- Embeddings semánticos
"""

from transformers import BertTokenizer, BertModel

# --------------------------------------------------
# CARGA DE TOKENIZER Y MODELO PREENTRENADO
# --------------------------------------------------
# Tokenizer convierte texto en tokens compatibles con BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Modelo BERT preentrenado para embeddings contextuales
model = BertModel.from_pretrained("bert-base-uncased")

# --------------------------------------------------
# TEXTO DE ENTRADA
# --------------------------------------------------
text = "Transformers are powerful models for NLP tasks"

# --------------------------------------------------
# TOKENIZACIÓN
# --------------------------------------------------
# Convierte texto a tensores PyTorch (input_ids, attention_mask)
inputs = tokenizer(text, return_tensors='pt')

# --------------------------------------------------
# FORWARD PASS EN BERT
# --------------------------------------------------
# Obtiene representaciones contextuales del texto
outputs = model(**inputs)

# --------------------------------------------------
# SALIDA DEL MODELO
# --------------------------------------------------
# last_hidden_state: embeddings por token
print("Hidden States Shape:", outputs.last_hidden_state.shape)