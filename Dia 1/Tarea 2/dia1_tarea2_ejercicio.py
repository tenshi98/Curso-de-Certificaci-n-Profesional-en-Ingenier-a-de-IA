"""
DOCSTRING:
---------
Optimización de hiperparámetros en un modelo Random Forest utilizando Grid Search
y Randomized Search sobre el dataset Iris.

Descripción:
Este script entrena un modelo de clasificación Random Forest sobre el dataset Iris
y compara dos estrategias de optimización de hiperparámetros:

- Grid Search (búsqueda exhaustiva en rejilla).
- Randomized Search (búsqueda aleatoria en un espacio de parámetros).

Se evalúa el rendimiento del mejor modelo encontrado por cada estrategia
utilizando accuracy en el conjunto de prueba.

Flujo del proceso:
1. Carga del dataset Iris desde sklearn.
2. División en conjunto de entrenamiento y prueba.
3. Definición del espacio de hiperparámetros para Grid Search.
4. Ejecución de Grid Search con validación cruzada.
5. Evaluación del mejor modelo de Grid Search.
6. Definición del espacio de distribución de hiperparámetros.
7. Ejecución de Randomized Search con validación cruzada.
8. Evaluación del mejor modelo de Randomized Search.
9. Comparación de resultados entre ambas estrategias.

Parámetros:
No recibe parámetros (script ejecutable directamente).

Retorno:
No retorna valores (solo imprime hiperparámetros óptimos y métricas de accuracy).

Excepciones:
- ValueError: si los datos no pueden dividirse correctamente.
- RuntimeError: si falla la búsqueda de hiperparámetros.
- KeyError: si las métricas o parámetros no están disponibles en el objeto de búsqueda.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# ------------------------------------------------------------
# Carga del dataset Iris
# ------------------------------------------------------------

data = load_iris()

# Variables predictoras y objetivo
X, y = data.data, data.target

# ------------------------------------------------------------
# División del dataset
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
# Configuración Grid Search
# ------------------------------------------------------------

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Ejecución Grid Search
grid_search.fit(X_train, y_train)

# ------------------------------------------------------------
# Evaluación Grid Search
# ------------------------------------------------------------

best_grid_model = grid_search.best_estimator_

y_pred_grid = best_grid_model.predict(X_test)

accuracy_grid = accuracy_score(y_test, y_pred_grid)

print(f"Best Hyperparameters (Grid Search): {grid_search.best_params_}")
print(f"Grid Search Exactitud: {accuracy_grid:.4f}")

# ------------------------------------------------------------
# Configuración Randomized Search
# ------------------------------------------------------------

param_dist = {
    'n_estimators': np.arange(50, 200, 10),
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10, 20]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Ejecución Randomized Search
random_search.fit(X_train, y_train)

# ------------------------------------------------------------
# Evaluación Randomized Search
# ------------------------------------------------------------

best_random_model = random_search.best_estimator_

y_pred_random = best_random_model.predict(X_test)

accuracy_random = accuracy_score(y_test, y_pred_random)

print(f"Best Hyperparameters (Random Search): {random_search.best_params_}")
print(f"Random Search Exactitud: {accuracy_random:.4f}")