"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: USO DE BERT CON TENSORFLOW (HUGGING FACE TRANSFORMERS)
--------------------------------------------------

Descripción general:
Este script utiliza un modelo preentrenado BERT (bert-base-uncased)
a través de la librería Hugging Face Transformers en su implementación para TensorFlow.

El objetivo es:
- Tokenizar una oración de entrada
- Procesarla con un modelo BERT en TensorFlow
- Obtener representaciones contextuales (hidden states)

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Tokenizer (BERT Tokenizer):
   - Convierte texto en tokens compatibles con BERT
   - Usa WordPiece tokenization
   - Genera tensores en formato TensorFlow (tf.Tensor)

2. Modelo BERT (TFBertModel):
   - Versión TensorFlow del modelo BERT preentrenado
   - Arquitectura Transformer bidireccional
   - Devuelve embeddings contextuales por token

3. Entrada del modelo:
   - Texto convertido en:
     - input_ids
     - attention_mask
   - Formato TensorFlow (return_tensors='tf')

4. Salida del modelo:
   - last_hidden_state:
     Representación contextual de cada token

--------------------------------------------------
FLUJO DEL SCRIPT
--------------------------------------------------

1. Carga del tokenizer preentrenado "bert-base-uncased"
2. Carga del modelo BERT en TensorFlow (TFBertModel)
3. Tokenización del texto de entrada
4. Conversión automática a tensores TensorFlow
5. Forward pass del modelo BERT
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

BERT produce embeddings contextuales, lo que significa:
- Cada token tiene una representación dependiente del contexto
- No son embeddings estáticos (como Word2Vec o GloVe)
- Captura relaciones bidireccionales en el texto

--------------------------------------------------
MODELO UTILIZADO
--------------------------------------------------

- bert-base-uncased:
  - 12 capas Transformer encoder
  - 768 dimensiones ocultas
  - Lowercase (uncased)
  - Preentrenado en BookCorpus + Wikipedia

--------------------------------------------------
DIFERENCIA CON PYTORCH VERSION
--------------------------------------------------

- Este script usa TensorFlow (TFBertModel)
- En lugar de tensores PyTorch, utiliza tf.Tensor
- Misma arquitectura y mismos pesos preentrenados

--------------------------------------------------
USO PRINCIPAL
--------------------------------------------------

Aplicaciones típicas:
- Clasificación de texto
- Embeddings semánticos
- Question Answering (QA)
- Named Entity Recognition (NER)
"""

from transformers import BertTokenizer, TFBertModel

# --------------------------------------------------
# CARGA DEL TOKENIZER Y MODELO BERT (TENSORFLOW)
# --------------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertModel.from_pretrained("bert-base-uncased")

# --------------------------------------------------
# TEXTO DE ENTRADA
# --------------------------------------------------
text = "Transformers are powerful models for NLP tasks"

# --------------------------------------------------
# TOKENIZACIÓN (TENSORFLOW)
# --------------------------------------------------
inputs = tokenizer(text, return_tensors='tf')

# --------------------------------------------------
# INFERENCIA CON BERT
# --------------------------------------------------
outputs = model(**inputs)

# --------------------------------------------------
# SALIDA DEL MODELO
# --------------------------------------------------
print("Hidden States Shape:", outputs.last_hidden_state.shape)