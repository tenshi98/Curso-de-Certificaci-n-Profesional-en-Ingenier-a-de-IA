"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: TRANSFER LEARNING CON RESNET50 (PYTORCH)
--------------------------------------------------

Descripción general:
Este script implementa un flujo básico de transfer learning utilizando
un modelo preentrenado ResNet50 de torchvision.

El objetivo es:
- Cargar un modelo CNN preentrenado en ImageNet
- Congelar sus parámetros para reutilización como extractor de características
- Modificar la capa final para una nueva tarea de clasificación
- Descongelar parcialmente una parte profunda del modelo para fine-tuning

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Modelo base:
   - Arquitectura: ResNet50
   - Preentrenado en ImageNet
   - Incluye bloques residuales profundos

2. Transfer Learning:
   - Técnica que reutiliza conocimiento de un modelo previamente entrenado
   - Permite entrenar más rápido en nuevos datasets

--------------------------------------------------
FLUJO DEL SCRIPT
--------------------------------------------------

1. Carga del modelo preentrenado
2. Congelamiento de parámetros
3. Modificación de la capa final
4. Descongelamiento parcial de capas profundas

--------------------------------------------------
DETALLE DEL CÓDIGO
--------------------------------------------------

--------------------------------------------------
1. CARGA DEL MODELO RESNET50
--------------------------------------------------

model = models.resnet50(pretrained=True)

- Se carga una red ResNet50 con pesos entrenados en ImageNet
- El modelo ya tiene conocimiento general de características visuales

--------------------------------------------------
2. CONGELAMIENTO DE PARÁMETROS
--------------------------------------------------

for param in model.parameters():
    param.requires_grad = False

Propósito:
- Evitar que los pesos del modelo base se actualicen durante entrenamiento
- Mantener las representaciones aprendidas

Efecto:
- Reduce costo computacional
- Previene sobreajuste en datasets pequeños

--------------------------------------------------
3. MODIFICACIÓN DE LA CAPA FINAL
--------------------------------------------------

num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 10)

Propósito:
- Adaptar el modelo a una nueva tarea de clasificación
- Sustituye la última capa totalmente conectada

Detalle:
- in_features: tamaño de entrada de la capa original
- out_features: 10 clases nuevas

--------------------------------------------------
4. DESCONGELAMIENTO PARCIAL
--------------------------------------------------

for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

Propósito:
- Permitir ajuste fino (fine-tuning) en las últimas capas convolucionales
- "layer4" es la última etapa del backbone ResNet

Efecto:
- Mantiene capas iniciales congeladas (features generales)
- Ajusta capas profundas a la nueva tarea

--------------------------------------------------
CONCEPTO CLAVE
--------------------------------------------------

Transfer Learning:
- Reutiliza conocimiento aprendido en grandes datasets
- Reduce necesidad de datos y tiempo de entrenamiento
- Mejora rendimiento en tareas similares

Fine-tuning parcial:
- Solo ciertas capas se actualizan
- Balance entre generalización y adaptación

--------------------------------------------------
SALIDA DEL SCRIPT
--------------------------------------------------

- Modelo ResNet50 modificado con:
  - Backbone congelado parcialmente
  - Nueva capa fully connected (10 clases)
  - Última etapa (layer4) entrenable

--------------------------------------------------
CASOS DE USO
--------------------------------------------------

- Clasificación de imágenes con pocos datos
- Adaptación de modelos a dominios específicos
- Sistemas de visión por computador
"""

import torch
import torchvision.models as models

# --------------------------------------------------
# CARGA DEL MODELO PREENTRENADO
# --------------------------------------------------
# ResNet50 entrenado en ImageNet
model = models.resnet50(pretrained=True)

# --------------------------------------------------
# CONGELAMIENTO DE PARÁMETROS
# --------------------------------------------------
# Evita actualización de pesos del modelo base
for param in model.parameters():
    param.requires_grad = False

# --------------------------------------------------
# MODIFICACIÓN DE LA CAPA FINAL
# --------------------------------------------------
# Número de características de entrada de la última capa
num_features = model.fc.in_features

# Nueva capa para 10 clases
model.fc = torch.nn.Linear(num_features, 10)

# --------------------------------------------------
# DESCONGELAMIENTO PARCIAL (FINE-TUNING)
# --------------------------------------------------
# Se habilita entrenamiento solo en la última capa convolucional
for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True