"""
DOCSTRING:
Este script implementa un pipeline de entrenamiento y evaluación de un modelo BERT
para clasificación de sentimientos en el dataset IMDB, utilizando PyTorch de forma manual
y cálculo de métricas como F1-score y BLEU.

El flujo del código es el siguiente:

1. Carga del dataset IMDB desde Hugging Face datasets.
2. División del conjunto de entrenamiento en train/test usando train_test_split.
3. Tokenización del texto con un tokenizer preentrenado (bert-base-uncased):
   - Conversión a input_ids
   - Generación de attention_mask
   - Padding y truncamiento a longitud máxima de 128 tokens
4. Construcción de una clase Dataset personalizada de PyTorch (IMDBDataset):
   - Devuelve diccionarios con input_ids, attention_mask y labels como tensores.
5. Creación de DataLoaders para entrenamiento y evaluación.
6. Carga de un modelo preentrenado BERT para clasificación binaria (num_labels=2).
7. Definición del optimizador Adam.
8. Definición de scheduler de tasa de aprendizaje (linear schedule con warmup):
   - Ajusta dinámicamente la tasa de aprendizaje durante el entrenamiento.
9. Definición de loop de entrenamiento manual en PyTorch:
   - Forward pass
   - Cálculo de pérdida
   - Backpropagation
   - Actualización de parámetros
   - Paso del scheduler
10. Evaluación del modelo utilizando F1-score ponderado (weighted F1).
11. Ejemplo adicional de evaluación de calidad de texto usando BLEU score
   con la librería sacrebleu.

Parámetros importantes:
- max_length=128: longitud máxima de secuencia de entrada.
- num_labels=2: clasificación binaria (sentimiento positivo/negativo).
- batch_size=16: tamaño de batch para entrenamiento y evaluación.
- epochs=3: número de épocas de entrenamiento.

Retorno:
- Modelo BERT entrenado (si se ejecuta train_model).
- F1-score en conjunto de test.
- BLEU score de ejemplo.

Excepciones:
- Puede fallar si no hay GPU disponible (entrenamiento lento en CPU).
- Puede fallar si el modelo no ha sido entrenado antes de evaluación.
"""

# Importación de librerías necesarias
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Carga del dataset IMDB
dataset = load_dataset("imdb")

# División manual del dataset en entrenamiento y validación
train_texts, test_texts, train_labels, test_labels = train_test_split(
    dataset['train']['text'],
    dataset['train']['label'],
    test_size=0.2,
    random_state=42
)

# Inicialización del tokenizer BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Función de tokenización
def tokenize_data(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length
    )
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    }

# Aplicación de tokenización a train y test
train_data = tokenize_data(train_texts, train_labels, tokenizer)
test_data = tokenize_data(test_texts, test_labels, tokenizer)

# Dataset personalizado para PyTorch
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "labels": torch.tensor(self.labels[idx]),
        }

# Creación de datasets y dataloaders
train_dataset = IMDBDataset(train_data)
test_dataset = IMDBDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Carga del modelo BERT para clasificación binaria
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# Definición del optimizador Adam
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Configuración del scheduler de learning rate (linear schedule con warmup)
num_training_steps = len(train_loader) * 3
warmup_steps = int(0.1 * num_training_steps)

scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)

# Configuración de dispositivo (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loop de entrenamiento manual en PyTorch
def train_model():
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)

            # Cálculo de pérdida
            loss = outputs.loss

            # Backpropagation
            loss.backward()

            # Optimización
            optimizer.step()

            # Actualización del scheduler
            scheduler.step()

            # Reset de gradientes
            optimizer.zero_grad()

# Evaluación del modelo usando F1-score
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)

        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

# Cálculo de F1-score ponderado
f1 = f1_score(all_labels, all_preds, average="weighted")
print(f"F1-Score: {f1:.4f}")

# Evaluación adicional con BLEU score (ejemplo ilustrativo)
from sacrebleu import BLEU

references = [["this is a test sentence", "this is a sample sentence"]]
hypotheses = ["This is a test"]

bleu = BLEU()
bleu_score = bleu.corpus_score(hypotheses, references).score

print(f"BLEU Score: {bleu_score:.2f}")