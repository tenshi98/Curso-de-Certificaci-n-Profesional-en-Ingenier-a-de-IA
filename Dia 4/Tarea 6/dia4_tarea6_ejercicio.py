"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: FINE-TUNING DE ROBERTA PARA CLASIFICACIÓN DE TEXTO (AG NEWS)
--------------------------------------------------

Descripción general:
Este script realiza el fine-tuning de un modelo preentrenado RoBERTa
para una tarea de clasificación de texto multiclase utilizando el dataset AG News.

El objetivo principal es:
- Cargar un dataset de noticias
- Tokenizar el texto con RoBERTa
- Ajustar un modelo preentrenado para clasificación (4 clases)
- Entrenar el modelo con Hugging Face Trainer
- Evaluar el rendimiento final

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Dataset:
   - Fuente: "ag_news"
   - Tarea: clasificación de noticias en 4 categorías
   - Contiene texto (noticia) y etiqueta (label)

2. Tokenizer:
   - Modelo: roberta-base
   - Convierte texto en tokens compatibles con RoBERTa
   - Aplica truncamiento y padding a longitud fija (128 tokens)

3. Modelo:
   - AutoModelForSequenceClassification
   - Base: roberta-base
   - Adaptado a 4 clases de salida

--------------------------------------------------
PREPROCESAMIENTO DEL DATASET
--------------------------------------------------

Función: tokenize_function(examples)

Propósito:
- Convertir texto en IDs de tokens
- Normalizar longitud mediante padding y truncation

Transformaciones aplicadas:
- remove_columns(["text"]): elimina texto original
- rename_column("label", "labels"): compatibilidad con Trainer
- set_format("torch"): convierte a tensores PyTorch

--------------------------------------------------
CONFIGURACIÓN DE ENTRENAMIENTO
--------------------------------------------------

TrainingArguments:

- output_dir:
  Directorio donde se guardan checkpoints y resultados

- eval_strategy:
  Evaluación al final de cada época

- learning_rate:
  Tasa de aprendizaje (2e-5 típico para Transformers)

- batch_size:
  Tamaño de lote para entrenamiento y evaluación

- num_train_epochs:
  Número de iteraciones completas sobre el dataset

- weight_decay:
  Regularización para evitar overfitting

- save_steps:
  Frecuencia de guardado de checkpoints

--------------------------------------------------
HUGGING FACE TRAINER
--------------------------------------------------

Trainer:
- Abstracción que maneja:
  - Loop de entrenamiento
  - Evaluación
  - Optimización
  - Logging

Entradas:
- model: RoBERTa ajustado para clasificación
- args: configuración de entrenamiento
- train_dataset: datos de entrenamiento tokenizados
- eval_dataset: datos de evaluación
- processing_class: tokenizer asociado

--------------------------------------------------
FLUJO DEL PROGRAMA
--------------------------------------------------

1. Cargar dataset AG News
2. Cargar tokenizer RoBERTa
3. Cargar modelo preentrenado para clasificación
4. Tokenizar dataset completo
5. Formatear dataset para PyTorch
6. Configurar hiperparámetros de entrenamiento
7. Inicializar Trainer
8. Entrenar modelo (fine-tuning)
9. Evaluar modelo en test set
10. Imprimir métricas finales

--------------------------------------------------
NOTA IMPORTANTE
--------------------------------------------------

- El modelo RoBERTa se inicializa preentrenado y se ajusta al dataset
- Este proceso se conoce como "fine-tuning"
- Permite adaptar modelos generales a tareas específicas

--------------------------------------------------
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# --------------------------------------------------
# CARGA DEL DATASET
# --------------------------------------------------
# Dataset de noticias con 4 categorías
dataset = load_dataset("ag_news")

# --------------------------------------------------
# CARGA DE MODELO Y TOKENIZER
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Modelo RoBERTa adaptado a clasificación multiclase (4 clases)
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=4
)

# --------------------------------------------------
# TOKENIZACIÓN DEL DATASET
# --------------------------------------------------
def tokenize_function(examples):
    """
    Convierte texto en tokens compatibles con RoBERTa.

    Parámetros:
    examples (dict): batch de ejemplos con campo 'text'

    Retorno:
    dict: tokens (input_ids, attention_mask, etc.)
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Aplicación de tokenización en batch
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# --------------------------------------------------
# PREPARACIÓN DEL DATASET PARA TRAINER
# --------------------------------------------------

# Eliminación de texto original
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# Renombrar etiqueta para compatibilidad con Trainer
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Conversión a tensores PyTorch
tokenized_datasets.set_format("torch")

# Separación de datasets
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# --------------------------------------------------
# CONFIGURACIÓN DE ENTRENAMIENTO
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500
)

# --------------------------------------------------
# INICIALIZACIÓN DEL TRAINER
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer
)

# --------------------------------------------------
# ENTRENAMIENTO DEL MODELO
# --------------------------------------------------
trainer.train()

# --------------------------------------------------
# EVALUACIÓN DEL MODELO
# --------------------------------------------------
results = trainer.evaluate()

# Impresión de métricas finales
print("Evaluation Results:", results)