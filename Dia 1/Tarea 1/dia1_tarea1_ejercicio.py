"""
DOCSTRING:
---------
Comparación de rendimiento de modelos Random Forest con hiperparámetros por defecto
y configurados manualmente en el dataset Breast Cancer.

Descripción:
Este script entrena dos modelos de clasificación Random Forest sobre el dataset
de cáncer de mama de sklearn:

- Un modelo con hiperparámetros por defecto.
- Un modelo con hiperparámetros ajustados manualmente.

Se evalúa el desempeño de ambos modelos mediante accuracy y reporte de clasificación.

Flujo del proceso:
1. Carga del dataset Breast Cancer desde sklearn.
2. Separación en variables predictoras (X) y objetivo (y).
3. División del dataset en entrenamiento y prueba.
4. Entrenamiento de Random Forest con parámetros por defecto.
5. Predicción y evaluación del modelo base.
6. Entrenamiento de Random Forest con hiperparámetros ajustados.
7. Predicción y evaluación del modelo ajustado.
8. Comparación de resultados mediante accuracy y classification report.

Parámetros:
No recibe parámetros (script ejecutable directamente).

Retorno:
No retorna valores (solo imprime métricas y reportes de evaluación).

Excepciones:
- ValueError: si el dataset no puede dividirse correctamente.
- RuntimeError: si falla el entrenamiento del modelo.
- KeyError: si el dataset no contiene las columnas esperadas.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------
# Carga del dataset Breast Cancer
# ------------------------------------------------------------

data = load_breast_cancer()

# Variables predictoras y objetivo
X, y = data.data, data.target

# ------------------------------------------------------------
# División del dataset en entrenamiento y prueba
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# Información del dataset
# ------------------------------------------------------------

print(f"Feature Names: {data.feature_names}")
print(f"Class Names: {data.target_names}")

# ------------------------------------------------------------
# Modelo Random Forest con hiperparámetros por defecto
# ------------------------------------------------------------

rf_default = RandomForestClassifier(random_state=42)

# Entrenamiento del modelo base
rf_default.fit(X_train, y_train)

# Predicción modelo base
y_predict_default = rf_default.predict(X_test)

# Evaluación modelo base
accuracy_default = accuracy_score(y_test, y_predict_default)

print(f"Default Model Exactitud: {accuracy_default:.4f}")

print("\nInforme de clasificación:\n", classification_report(y_test, y_predict_default))

# ------------------------------------------------------------
# Modelo Random Forest con hiperparámetros ajustados
# ------------------------------------------------------------

rf_tuned = RandomForestClassifier(
    n_estimators=400,
    max_depth=5,
    random_state=42
)

# Entrenamiento del modelo ajustado
rf_tuned.fit(X_train, y_train)

# Predicción modelo ajustado
y_pred_tuned = rf_tuned.predict(X_test)

# Evaluación modelo ajustado
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tuned Model Exactitud: {accuracy_tuned:.4f}")

print("\n Informe de clasificación:\n", classification_report(y_test, y_pred_tuned))