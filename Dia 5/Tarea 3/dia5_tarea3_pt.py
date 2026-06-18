"""
DOCSTRING:
Este script implementa transfer learning utilizando el modelo preentrenado MobileNetV2 de torchvision
para una tarea de clasificación multiclase con 5 categorías.

El flujo del código incluye:

1. Carga de MobileNetV2 preentrenado en ImageNet.
2. Congelación de todos los parámetros del modelo base para evitar su actualización durante el entrenamiento inicial.
3. Modificación de la capa final del clasificador para adaptarla a 5 clases de salida.
4. Definición de transformaciones de datos para entrenamiento y validación:
   - Aumento de datos en entrenamiento (rotación, recorte aleatorio, flipping).
   - Normalización estándar compatible con ImageNet.
5. Carga de datasets desde directorios estructurados mediante ImageFolder.
6. Creación de DataLoaders para batching y shuffling.
7. Definición de función de pérdida CrossEntropyLoss para clasificación multiclase.
8. Definición del optimizador Adam aplicado a todos los parámetros del modelo.
9. Loop de entrenamiento durante 10 épocas con propagación hacia adelante, cálculo de pérdida
   y actualización de pesos.

Parámetros esperados:
- TRAINING_IMAGE_FOLDER: ruta al dataset de entrenamiento organizado por clases.
- Validation_IMAGE_FOLDER: ruta al dataset de validación organizado por clases.

Retorno:
- Pérdida por época impresa en consola.

Excepciones:
- Puede fallar si las rutas de los datasets no existen o si la estructura de carpetas no es válida.
"""

# Importación de librerías necesarias
import torch
import torch.utils
import torch.utils.data
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim

# Carga del modelo preentrenado MobileNetV2
model = models.mobilenet_v2(pretrained=True)

# Congelación de todos los parámetros del modelo base para transfer learning
for param in model.parameters():
    param.requires_grad = False

# Reemplazo de la capa clasificadora final para adaptación a 5 clases
model.classifier[1] = nn.Linear(model.last_channel, 5)

# Transformaciones de entrenamiento con aumento de datos
train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformaciones de validación sin aumento de datos
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
])

# Carga del dataset de entrenamiento desde estructura de carpetas
train_data = datasets.ImageFolder("TRAINING_IMAGE_FOLDER", transform=train_transform)

# Carga del dataset de validación desde estructura de carpetas
val_data = datasets.ImageFolder("Validation_IMAGE_FOLDER", transform=val_transform)

# Creación de DataLoaders para batching y shuffle
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

# Definición de la función de pérdida para clasificación multiclase
criterion = nn.CrossEntropyLoss()

# Definición del optimizador Adam
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Loop de entrenamiento por épocas
for epoch in range(10):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Cálculo de la pérdida
        loss = criterion(outputs, labels)
        
        # Backpropagation
        loss.backward()
        
        # Actualización de parámetros
        optimizer.step()
    
    # Impresión de pérdida de la última iteración de la época
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")