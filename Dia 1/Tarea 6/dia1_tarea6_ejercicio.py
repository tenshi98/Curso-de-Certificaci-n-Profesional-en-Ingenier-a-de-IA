"""
DOCSTRING:
---------
Comparación de optimización de hiperparámetros utilizando Grid Search y Randomized Search
sobre modelos Gradient Boosting y SVM en el dataset Iris.

Descripción:
Este script entrena y evalúa dos enfoques de optimización de hiperparámetros:

- Grid Search aplicado a Gradient Boosting Classifier.
- Randomized Search aplicado a Support Vector Machine (SVC).

Se evalúan ambos modelos utilizando accuracy y métricas de clasificación.

Flujo del proceso:
1. Carga del dataset Iris desde sklearn.
2. División en conjuntos de entrenamiento y prueba.
3. Definición de espacio de hiperparámetros para Gradient Boosting.
4. Ejecución de Grid Search con validación cruzada.
5. Selección del mejor modelo de Gradient Boosting.
6. Evaluación del modelo en conjunto de prueba.
7. Definición de distribución de hiperparámetros para SVM.
8. Ejecución de Randomized Search con validación cruzada.
9. Selección del mejor modelo SVM.
10. Evaluación del modelo en conjunto de prueba.
11. Comparación de resultados entre ambas estrategias.

Parámetros:
No recibe parámetros (script ejecutable directamente).

Retorno:
No retorna valores (solo imprime hiperparámetros óptimos, accuracy y reportes).

Excepciones:
- ValueError: si los datos no pueden dividirse o entrenarse correctamente.
- RuntimeError: si GridSearchCV o RandomizedSearchCV falla durante ejecución.
- KeyError: si las métricas o parámetros no existen en los resultados.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
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

print("Conjunto de datos Loaded and Split Successfully")

# ------------------------------------------------------------
# Grid Search - Gradient Boosting
# ------------------------------------------------------------

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_params_grid = grid_search.best_params_
best_score_grid = grid_search.best_score_

print(f"Best Parameters (GridSeachCV): {best_params_grid}")
print(f"Best Cross-Validation Exactitud (GridSearchCV): {best_score_grid:.4f}")

best_grid_model = grid_search.best_estimator_

y_pred_grid = best_grid_model.predict(X_test)

accuracy_grid = accuracy_score(y_test, y_pred_grid)

print(f"Test Exactitud (GridSearchCV): {accuracy_grid:.4f}")

print("\n Informe de clasificación:\n", classification_report(y_test, y_pred_grid))

# ------------------------------------------------------------
# Randomized Search - SVM
# ------------------------------------------------------------

param_dist = {
    'C': np.logspace(-3, 3, 10),
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

random_search = RandomizedSearchCV(
    estimator=SVC(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

print(f"Best Parameters (RandomizedSearchCV): {best_params_random}")
print(f"Bestr Cross-Validation Exactitud (RandomizedSearchCV): {best_score_random:.4f}")

best_random_model = random_search.best_estimator_

y_pred_random = best_random_model.predict(X_test)

accuracy_random = accuracy_score(y_test, y_pred_random)

print(f"Test Exactitud (RandomizedSeachCV): {accuracy_random:.4f}")

print("\n Classsification Report:\n", classification_report(y_test, y_pred_random))