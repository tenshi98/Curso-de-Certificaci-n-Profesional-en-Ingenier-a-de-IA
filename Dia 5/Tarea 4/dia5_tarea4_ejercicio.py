"""
DOCSTRING:
Este script implementa un pipeline completo de fine-tuning de un modelo BERT
para clasificación de sentimiento binario utilizando el dataset IMDB.

El flujo del código es el siguiente:

1. Carga del dataset IMDB desde la librería datasets.
2. Inicialización de un tokenizer preentrenado BERT (bert-base-uncased).
3. Tokenización del texto con truncamiento y padding a longitud fija (128 tokens).
4. Limpieza y reestructuración del dataset:
   - Eliminación de la columna de texto original.
   - Renombrado de la columna "label" a "labels" para compatibilidad con Trainer.
   - Conversión del dataset a formato tensorial de PyTorch.
5. Carga de un modelo BERT preentrenado adaptado a clasificación con 2 etiquetas.
6. Definición de los argumentos de entrenamiento mediante TrainingArguments:
   - Configuración de epochs, batch size, learning rate y estrategia de evaluación.
7. Inicialización del objeto Trainer de Hugging Face:
   - Encargado de gestionar entrenamiento y evaluación.
8. Entrenamiento del modelo sobre el dataset de entrenamiento.
9. Evaluación final del modelo sobre el conjunto de test.

Parámetros importantes:
- max_length=128: longitud máxima de secuencia de entrada.
- num_labels=2: clasificación binaria (sentimiento positivo/negativo).

Retorno:
- Modelo fine-tuneado de BERT.
- Métricas de evaluación en el dataset de test.

Excepciones:
- Puede fallar si no hay memoria suficiente para entrenar BERT en CPU.
"""

# pip install transformers datasets

# Importación de librerías necesarias
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Carga del dataset IMDB
dataset = load_dataset("imdb")

# Inicialización del tokenizer BERT preentrenado
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Función de tokenización del dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Aplicación de tokenización al dataset completo
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Eliminación de texto original (no necesario para entrenamiento)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# Renombrado de etiqueta para compatibilidad con Trainer
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Conversión del dataset a formato tensorial PyTorch
tokenized_datasets.set_format("torch")

# Separación en train y test
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Carga del modelo BERT para clasificación binaria
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Definición de parámetros de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2
)

# Inicialización del Trainer de Hugging Face
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer
)

# Entrenamiento del modelo
trainer.train()

# Evaluación del modelo entrenado
results = trainer.evaluate()
print("Evaluation Results:", results)