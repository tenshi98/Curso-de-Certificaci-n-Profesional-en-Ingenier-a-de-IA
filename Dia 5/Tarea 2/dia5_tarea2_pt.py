"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: TRANSFER LEARNING CON RESNET50 (PYTORCH) + ENTRENAMIENTO COMPLETO
--------------------------------------------------

Descripción general:
Este script implementa un pipeline completo de transfer learning utilizando
ResNet50 preentrenado en ImageNet, adaptándolo a una nueva tarea de clasificación
multiclase con 5 categorías.

El flujo incluye:
- Carga de modelo preentrenado
- Congelamiento de capas base
- Reemplazo de la capa final
- Preparación de dataset de imágenes
- Entrenamiento supervisado
- Evaluación de validación

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Modelo base:
   - Arquitectura: ResNet50
   - Pesos: ImageNet preentrenado
   - Framework: PyTorch torchvision

2. Transfer Learning:
   - Congelamiento de capas convolucionales
   - Entrenamiento solo de la cabeza (classifier head)
   - Posterior habilitación parcial de layer4

3. Dataset:
   - Formato: ImageFolder
   - Estructura: carpetas por clase
   - Tarea: clasificación multiclase (5 clases)

--------------------------------------------------
FLUJO GENERAL DEL SCRIPT
--------------------------------------------------

1. Carga del modelo ResNet50 preentrenado
2. Congelamiento de todos los parámetros
3. Reemplazo de la última capa fully connected
4. Definición de transformaciones de imagen
5. Carga de datasets de entrenamiento y validación
6. Creación de DataLoaders
7. Definición de loss y optimizador
8. Entrenamiento del modelo
9. Activación parcial de capas profundas
10. Evaluación en validación

--------------------------------------------------
DETALLE DEL CÓDIGO
--------------------------------------------------

--------------------------------------------------
1. CARGA DEL MODELO PREENTRENADO
--------------------------------------------------

model = models.resnet50(pretrained=True)

- Se carga ResNet50 con pesos entrenados en ImageNet
- Permite reutilizar representaciones visuales generales

--------------------------------------------------
2. CONGELAMIENTO DE PARÁMETROS
--------------------------------------------------

for param in model.parameters():
    param.requires_grad = False

Propósito:
- Evitar actualización de pesos del backbone
- Mantener características visuales aprendidas

--------------------------------------------------
3. MODIFICACIÓN DE LA CAPA FINAL
--------------------------------------------------

model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 5),
    nn.Softmax(dim=1)
)

Propósito:
- Adaptar modelo a nueva tarea de 5 clases
- Agregar capa intermedia para mayor capacidad de aprendizaje
- Aplicar Softmax para distribución de probabilidades

--------------------------------------------------
IMPORTANTE (COMPORTAMIENTO REAL)
--------------------------------------------------

- CrossEntropyLoss espera logits SIN softmax explícito
- Sin embargo, en este script se incluye Softmax en la salida

--------------------------------------------------
4. PREPARACIÓN DE DATOS
--------------------------------------------------

Transformaciones:

- Resize(224x224):
  Ajusta imágenes al tamaño esperado por ResNet50

- ToTensor():
  Convierte imagen a tensor PyTorch

- Normalize():
  Normalización basada en ImageNet:
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]

--------------------------------------------------
Dataset:
--------------------------------------------------

datasets.ImageFolder(PATH)

- Estructura esperada:
  root/
    class_1/
    class_2/
    class_3/
    class_4/
    class_5/

- Labels inferidos automáticamente por carpeta

--------------------------------------------------
5. DATALOADERS
--------------------------------------------------

train_loader:
- batch_size=32
- shuffle=True

val_loader:
- batch_size=32
- shuffle=False

--------------------------------------------------
6. LOSS Y OPTIMIZER
--------------------------------------------------

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

- CrossEntropyLoss:
  usada para clasificación multiclase

- Adam:
  optimizador adaptativo

--------------------------------------------------
7. ENTRENAMIENTO
--------------------------------------------------

for epoch in range(10):
    model.train()

    for inputs, labels:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

--------------------------------------------------
Descripción:
- Entrenamiento supervisado por 10 épocas
- Backpropagation sobre capas no congeladas (principalmente fc)

--------------------------------------------------
8. DESCONGELAMIENTO PARCIAL
--------------------------------------------------

for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

Propósito:
- Activar fine-tuning en última capa convolucional
- Ajustar representaciones profundas al nuevo dominio

--------------------------------------------------
9. EVALUACIÓN
--------------------------------------------------

model.eval()

- Desactiva dropout y batchnorm training behavior

Proceso:
- Forward pass sin gradientes
- Predicción con argmax
- Cálculo de accuracy

--------------------------------------------------
SALIDA DEL SCRIPT
--------------------------------------------------

- Modelo ResNet50 adaptado a 5 clases
- Capa fully connected personalizada
- Entrenamiento parcial de backbone (layer4)
- Métrica final: accuracy de validación

--------------------------------------------------
CONCEPTO CLAVE
--------------------------------------------------

Transfer Learning:
- Reutiliza modelos preentrenados en grandes datasets
- Reduce tiempo de entrenamiento y necesidad de datos

Fine-tuning progresivo:
- Primero se entrena solo la cabeza
- Luego se habilitan capas profundas (layer4)

--------------------------------------------------
CASOS DE USO
--------------------------------------------------

- Clasificación de imágenes personalizadas
- Diagnóstico médico por imágenes
- Clasificación industrial
- Visión por computador en datasets pequeños/medianos
"""

import torch 
import torch.utils
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim

# --------------------------------------------------
# CARGA DEL MODELO PREENTRENADO
# --------------------------------------------------
model = models.resnet50(pretrained=True)

# --------------------------------------------------
# CONGELAMIENTO DE PARÁMETROS
# --------------------------------------------------
for param in model.parameters():
    param.requires_grad = False

# --------------------------------------------------
# MODIFICACIÓN DE LA CAPA FINAL
# --------------------------------------------------
num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 5),
    nn.Softmax(dim=1)
)

print(model)

# --------------------------------------------------
# TRANSFORMACIONES DE DATOS
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# CARGA DE DATASETS
# --------------------------------------------------
train_data = datasets.ImageFolder("PATH_TO_FODLER_TRAIN", transform=transform)
val_data = datasets.ImageFolder("PATH_TO_FODLER_VAL", transform=transform)

# --------------------------------------------------
# DATALOADERS
# --------------------------------------------------
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

# --------------------------------------------------
# LOSS Y OPTIMIZER
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------------
# ENTRENAMIENTO
# --------------------------------------------------
for epoch in range(10):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, loss: {loss.item()}")

# --------------------------------------------------
# DESCONGELAMIENTO PARCIAL
# --------------------------------------------------
for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

# --------------------------------------------------
# EVALUACIÓN
# --------------------------------------------------
model.eval()

correct = 0 
total = 0

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Val Acccuracy: {100 * correct/ total:.2f}%")