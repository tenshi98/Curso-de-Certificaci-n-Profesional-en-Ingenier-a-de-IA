"""
--------------------------------------------------
DOCUMENTACIÓN DEL SCRIPT: FINE-TUNING DE RESNET50 (TENSORFLOW / KERAS)
--------------------------------------------------

Descripción general:
Este script carga un modelo preentrenado ResNet50 desde Keras Applications
y configura sus capas para realizar transfer learning mediante congelamiento
parcial de la red.

El objetivo es:
- Cargar ResNet50 preentrenado en ImageNet
- Inspeccionar su arquitectura (opcional)
- Congelar la mayoría de las capas del modelo
- Dejar solo las últimas capas entrenables para fine-tuning

--------------------------------------------------
COMPONENTES PRINCIPALES
--------------------------------------------------

1. Modelo:
   - Arquitectura: ResNet50
   - Framework: TensorFlow / Keras
   - Pesos: ImageNet preentrenados

2. Transfer Learning:
   - Reutilización de representaciones aprendidas
   - Ajuste parcial del modelo a nueva tarea

--------------------------------------------------
FLUJO DEL SCRIPT
--------------------------------------------------

1. Carga del modelo ResNet50 con pesos ImageNet
2. (Opcional) Inspección de arquitectura del modelo
3. Congelamiento parcial de capas
4. Preparación del modelo para fine-tuning

--------------------------------------------------
DETALLE DEL CÓDIGO
--------------------------------------------------

--------------------------------------------------
1. CARGA DEL MODELO PREENTRENADO
--------------------------------------------------

model = ResNet50(weights="imagenet")

Propósito:
- Instancia un modelo ResNet50 ya entrenado en ImageNet
- Incluye 1000 clases de clasificación originales

--------------------------------------------------
2. INSPECCIÓN DEL MODELO (COMENTADA)
--------------------------------------------------

model.summary()

Propósito:
- Mostrar estructura completa del modelo
- Incluye número de parámetros y capas

--------------------------------------------------
Iteración sobre capas:
--------------------------------------------------

for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}, Trainable: {layer.trainable}")

Propósito:
- Revisar estado de entrenamiento de cada capa
- Permite análisis de qué partes del modelo están activas

--------------------------------------------------
3. CONGELAMIENTO PARCIAL DE CAPAS
--------------------------------------------------

for layer in model.layers[:-10]:
    layer.trainable = False

--------------------------------------------------
Significado:
- Se congelan todas las capas excepto las últimas 10
- Las últimas capas quedan entrenables

--------------------------------------------------
EFECTO DEL CONGELAMIENTO
--------------------------------------------------

Capas congeladas:
- No actualizan sus pesos durante backpropagation
- Mantienen conocimiento aprendido en ImageNet

Capas entrenables:
- Se ajustan al nuevo dataset
- Aprenden representaciones específicas de la nueva tarea

--------------------------------------------------
CONCEPTO CLAVE
--------------------------------------------------

Transfer Learning:
- Reutiliza redes profundas preentrenadas
- Reduce tiempo y datos necesarios
- Mejora rendimiento en datasets pequeños o medianos

Fine-tuning parcial:
- Solo un subconjunto de capas se entrena
- Equilibrio entre conocimiento general y adaptación

--------------------------------------------------
SALIDA DEL SCRIPT
--------------------------------------------------

- Modelo ResNet50 con:
  - Mayoría de capas congeladas
  - Últimas 10 capas entrenables
  - Preparado para ser compilado y entrenado en nueva tarea

--------------------------------------------------
CASOS DE USO
--------------------------------------------------

- Clasificación de imágenes personalizadas
- Adaptación de modelos a dominios específicos
- Transfer learning en visión por computador
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# --------------------------------------------------
# CARGA DEL MODELO PREENTRENADO
# --------------------------------------------------
# ResNet50 con pesos entrenados en ImageNet
model = ResNet50(weights="imagenet")

# --------------------------------------------------
# INSPECCIÓN DEL MODELO (OPCIONAL)
# --------------------------------------------------
# model.summary()

# for i, layer in enumerate(model.layers):
#     print(f"Layer {i}: {layer.name}, Trainable: {layer.trainable}")

# --------------------------------------------------
# CONGELAMIENTO PARCIAL DE CAPAS
# --------------------------------------------------
# Se congelan todas las capas excepto las últimas 10
for layer in model.layers[:-10]:
    layer.trainable = False