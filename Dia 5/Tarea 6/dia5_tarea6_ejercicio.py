"""
DOCSTRING:
Este script implementa un pipeline de clasificación de texto biomédico utilizando el dataset
PubMed RCT 20k y un modelo preentrenado BioBERT de Hugging Face Transformers.

El flujo del código es el siguiente:

1. Carga del dataset PubMed RCT (versión 20k) desde Hugging Face Datasets.
2. Visualización de una muestra del dataset de entrenamiento.
3. Inicialización de un tokenizer preentrenado BERT (bert-base-uncased).
4. Preprocesamiento del texto mediante tokenización:
   - Truncamiento a 512 tokens
   - Padding para igualar longitudes
5. Tokenización del dataset completo mediante dataset.map.
6. Renombrado de la columna "label" a "labels" para compatibilidad con Trainer.
7. Conversión del dataset a formato tensorial para PyTorch.
8. Carga de un modelo preentrenado BioBERT para clasificación multiclase (5 clases).
9. Definición de hiperparámetros de entrenamiento mediante TrainingArguments:
   - Learning rate
   - Batch size
   - Número de épocas
   - Weight decay
10. Inicialización del Trainer de Hugging Face:
    - Maneja automáticamente entrenamiento y evaluación del modelo.
11. Entrenamiento del modelo sobre el dataset PubMed RCT.
12. Evaluación del modelo entrenado sobre el conjunto de validación.
13. Implementación de una función simple de data augmentation basada en sinónimos:
    - Reemplazo de palabras clave médicas por sinónimos predefinidos.

Parámetros importantes:
- max_length=512: longitud máxima de secuencia para textos biomédicos largos.
- num_labels=5: clasificación multiclase del dataset RCT.
- model: BioBERT (dmis-lab/biobert-base-cased-v1.1).

Retorno:
- Modelo BioBERT fine-tuneado.
- Métricas de evaluación del Trainer.
- Ejemplo de datos aumentados (data augmentation).

Excepciones:
- Puede fallar si el dataset PubMed RCT no está disponible o no descargado correctamente.
- Puede fallar si no hay memoria suficiente para entrenar BioBERT.

Notas:
- El parámetro "padding='mex_length'" contiene un error tipográfico, pero no se modifica según las reglas de documentación.
"""

# pip install transformers datasets

# Importación del dataset desde Hugging Face
from datasets import load_dataset

# Carga del dataset PubMed RCT 20k
dataset = load_dataset("pubmed_rct", "20k_rct")

# Visualización de una muestra del dataset
print(dataset["train"][0])

# Importación del tokenizer BERT
from transformers import AutoTokenizer

# Inicialización del tokenizer preentrenado
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Función de preprocesamiento y tokenización
def preprocess_data(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="mex_length",
        max_length=512
    )

# Aplicación del tokenizer al dataset completo
tokenized_datasets = dataset.map(preprocess_data, batched=True)

# Renombrado de columna para compatibilidad con Trainer
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Conversión a formato tensorial PyTorch
tokenized_datasets.set_format("torch")

# Importación del modelo de clasificación
from transformers import AutoModelForSequenceClassification

# Carga del modelo BioBERT preentrenado para clasificación multiclase
model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    num_labels=5
)

# Definición de argumentos de entrenamiento
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

# Inicialización del Trainer
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

# Entrenamiento del modelo
trainer.train()

# Evaluación del modelo
results = trainer.evaluate()
print("Evaluation Results:", results)

# Función de data augmentation basada en sinónimos simples
import random

def augment_text(text):
    synonyms = {
        "cancer": ["tumor", "malignancy"],
        "study": ["research", "experiment"]
    }
    words = text.split()
    new_words = [
        random.choice(synonyms[word]) if word in synonyms else word
        for word in words
    ]
    return " ".join(new_words)

# Aplicación de augmentación sobre el dataset de entrenamiento
augmented_data = [augment_text(sample["text"]) for sample in dataset["train"]]

# Visualización de ejemplos aumentados
print(augmented_data[:5])