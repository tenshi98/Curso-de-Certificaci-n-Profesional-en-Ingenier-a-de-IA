"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: SUMARIZACIÓN DE TEXTO CON T5 (SEQ2SEQ - TRANSFORMERS)
--------------------------------------------------

Descripción general:
Este script implementa un flujo completo de NLP para sumarización de texto
utilizando un modelo preentrenado T5 (t5-small) con Hugging Face Transformers.

El objetivo es:
- Cargar un dataset de diálogo (SAMSum)
- Preprocesar texto para tarea de summarization
- Tokenizar entradas y salidas (seq2seq)
- Preparar entrenamiento con Trainer (Hugging Face)
- Generar una predicción de resumen con beam search

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Dataset:
   - Fuente: "samsum"
   - Tipo: conversaciones de chat + resumen humano
   - Tarea: abstractive summarization

2. Modelo:
   - t5-small (Text-To-Text Transfer Transformer)
   - Arquitectura: Encoder-Decoder (Seq2Seq)
   - Entrenado para múltiples tareas NLP mediante formato textual

3. Tokenizer:
   - Convierte texto a tokens compatibles con T5
   - Maneja input/output en formato seq2seq

--------------------------------------------------
CONFIGURACIÓN DEL ENTORNO
--------------------------------------------------

Variable:
- KMP_DUPLICATE_LIB_OK = "True"
  Evita conflictos relacionados con bibliotecas paralelas de OpenMP

--------------------------------------------------
PREPROCESAMIENTO DEL DATASET
--------------------------------------------------

Función: tokenize_function(examples)

Propósito:
- Preparar entradas y salidas para entrenamiento seq2seq

--------------------------------------------------
1. PREPARACIÓN DE INPUTS
--------------------------------------------------

inputs = ["summarize: " + doc]

- Se añade prefijo "summarize:"
- T5 utiliza prompts textuales para definir la tarea

--------------------------------------------------
2. TOKENIZACIÓN DE ENTRADAS
--------------------------------------------------

tokenizer(
    inputs,
    max_length=512,
    truncation=True,
    padding="max_length"
)

- Limita longitud de entrada a 512 tokens
- Padding asegura tamaño uniforme

--------------------------------------------------
3. TOKENIZACIÓN DE TARGETS (SUMARIOS)
--------------------------------------------------

with tokenizer.as_target_tokenizer():

- Indica que se están tokenizando etiquetas (output)

labels = tokenizer(
    examples["summary"],
    max_length=150,
    truncation=True,
    padding="max_length"
)

--------------------------------------------------
4. SALIDA DEL TOKENIZER
--------------------------------------------------

model_inputs["labels"] = labels["input_ids"]

- Se agregan labels para entrenamiento supervisado seq2seq

--------------------------------------------------
CONFIGURACIÓN DE ENTRENAMIENTO
--------------------------------------------------

TrainingArguments:

- output_dir:
  Directorio de salida para checkpoints

- eval_strategy:
  Evaluación al final de cada época

- save_strategy:
  Guardado de modelo por época

- learning_rate:
  Tasa de aprendizaje (2e-5 recomendado para Transformers)

- batch_size:
  Tamaño de lote para entrenamiento y evaluación

- num_train_epochs:
  Número de ciclos completos de entrenamiento

- weight_decay:
  Regularización para evitar overfitting

- load_best_model_at_end:
  Carga el mejor modelo según evaluación final

--------------------------------------------------
TRAINER (HUGGING FACE)
--------------------------------------------------

Trainer:
- Maneja entrenamiento completo automáticamente
- Controla forward/backward pass
- Maneja evaluación y checkpoints

Nota:
- En este script el entrenamiento está comentado (#trainer.train())

--------------------------------------------------
INFERENCIA (GENERACIÓN DE RESUMEN)
--------------------------------------------------

1. Selección de dispositivo:
   device = "mps" (Apple GPU) o CPU

2. Entrada:
   Texto de ejemplo sobre Transformers

3. Tokenización:
   Se agrega prompt "summarize:"

4. Generación:
   model.generate()

Parámetros:
- max_length=150: longitud máxima del resumen
- num_beams=4: beam search para mejorar calidad
- early_stopping=True: detiene generación al finalizar secuencia

--------------------------------------------------
FLUJO DEL SCRIPT
--------------------------------------------------

1. Carga dataset SAMSum
2. Carga modelo T5-small
3. Tokeniza inputs (dialogue) y outputs (summary)
4. Prepara dataset para seq2seq
5. Define TrainingArguments
6. Inicializa Trainer
7. (Opcional) Entrena modelo
8. Ejecuta inferencia de resumen
9. Decodifica salida

--------------------------------------------------
METRICAS (COMENTADAS)
--------------------------------------------------

Se incluye evaluación opcional con ROUGE:
- ROUGE mide similitud entre resumen generado y referencia humana
- Métrica estándar en summarization

--------------------------------------------------
CASO DE USO
--------------------------------------------------

- Sumarización automática de textos
- Asistentes de lectura
- Compresión de información
- NLP generativo basado en encoder-decoder

--------------------------------------------------
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import evaluate
import torch
import os

# --------------------------------------------------
# CONFIGURACIÓN DEL ENTORNO
# --------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# --------------------------------------------------
# CARGA DEL DATASET
# --------------------------------------------------
dataset = load_dataset("samsum")

# --------------------------------------------------
# MODELO Y TOKENIZER
# --------------------------------------------------
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# --------------------------------------------------
# TOKENIZACIÓN SEQ2SEQ
# --------------------------------------------------
def tokenize_function(examples):
    """
    Prepara inputs y labels para entrenamiento de summarization.

    Parámetros:
    examples (dict): batch con 'dialogue' y 'summary'

    Retorno:
    dict: tokens listos para entrenamiento seq2seq
    """

    # Prefijo de tarea para T5
    inputs = ["summarize: " + doc for doc in examples["dialogue"]]

    # Tokenización de inputs
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Tokenización de targets (resúmenes)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=150,
            truncation=True,
            padding="max_length"
        )

    # Asignación de labels para entrenamiento
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Aplicación del preprocesamiento
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# --------------------------------------------------
# CONFIGURACIÓN DE TRAINER
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True
)

# --------------------------------------------------
# INICIALIZACIÓN DEL TRAINER
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer
)

# --------------------------------------------------
# ENTRENAMIENTO (COMENTADO)
# --------------------------------------------------
# trainer.train()

# --------------------------------------------------
# CONFIGURACIÓN DE DISPOSITIVO
# --------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# --------------------------------------------------
# INFERENCIA DE SUMARIZACIÓN
# --------------------------------------------------
sample_text = "The Transformer model has revolutionized NLP by enabling parallel processing of sequences."

inputs = tokenizer(
    "summarize: " + sample_text,
    return_tensors="pt",
    max_length=512,
    truncation=True
).to(device)

outputs = model.generate(
    inputs["input_ids"],
    max_length=150,
    num_beams=4,
    early_stopping=True
)

print(
    "Generated Summary: ",
    tokenizer.decode(outputs[0], skip_special_tokens=True)
)

# --------------------------------------------------
# MÉTRICAS (COMENTADAS)
# --------------------------------------------------
# metric = evaluate.load("rouge")
# predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
# references = [tokenizer.decode(r, skip_special_tokens=True) for r in tokenized_datasets["validation"]["summary"]]
# results = metric.compute(predictions=predictions, references=references)
# print(results)