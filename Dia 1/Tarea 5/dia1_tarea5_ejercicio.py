"""
DOCSTRING:
---------
Evaluación de un modelo Random Forest utilizando validación cruzada
(K-Fold y Stratified K-Fold) sobre un dataset de detección de fraude.

Descripción:
Este script carga un dataset de transacciones de tarjetas de crédito
y evalúa un modelo de clasificación Random Forest utilizando dos técnicas
de validación cruzada:

- K-Fold Cross Validation.
- Stratified K-Fold Cross Validation.

Se compara el rendimiento del modelo en ambos esquemas mediante accuracy.

Flujo del proceso:
1. Carga del dataset de crédito desde una URL externa.
2. Exploración de la información general del dataset.
3. Análisis de la distribución de la variable objetivo.
4. Separación en variables predictoras (X) y objetivo (y).
5. División en conjunto de entrenamiento y prueba.
6. Definición de K-Fold Cross Validation.
7. Evaluación del modelo Random Forest con K-Fold.
8. Definición de Stratified K-Fold Cross Validation.
9. Evaluación del modelo Random Forest con Stratified K-Fold.
10. Comparación de resultados entre ambas estrategias.

Parámetros:
No recibe parámetros (script ejecutable directamente).

Retorno:
No retorna valores (solo imprime scores de validación cruzada y promedios).

Excepciones:
- URLError: si el dataset no puede descargarse desde la URL.
- ValueError: si los splits de validación cruzada no son válidos.
- KeyError: si la columna 'Class' no existe en el dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# Carga del dataset de crédito
# ------------------------------------------------------------

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

# ------------------------------------------------------------
# Información del dataset
# ------------------------------------------------------------

print("Conjunto de datos Info:\n")

print(df.info)

print("\n Class Distribution:\n")

print(df['Class'].value_counts())

# ------------------------------------------------------------
# Definición de variables
# ------------------------------------------------------------

X = df.drop(columns=['Class'])
y = df['Class']

# ------------------------------------------------------------
# División del dataset
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# Validación cruzada K-Fold
# ------------------------------------------------------------

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf_model = RandomForestClassifier(random_state=42)

scores_kfold = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=kf,
    scoring='accuracy'
)

print(f"K-fold cross validation scores: {scores_kfold}")

print(f"Mean Exactitud (K-Fold): {scores_kfold.mean():.2f}")

# ------------------------------------------------------------
# Validación cruzada Stratified K-Fold
# ------------------------------------------------------------

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_stratified = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=skf,
    scoring='accuracy'
)

print(f"Stratified K-fold cross validation scores: {scores_stratified}")

print(f"Mean Exactitud (Stratified K-Fold): {scores_stratified.mean():.2f}")