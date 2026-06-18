"""
DOCSTRING:
---------
Optimización avanzada de hiperparámetros en XGBoost utilizando Optuna,
Grid Search y Randomized Search sobre el dataset Breast Cancer.

Descripción:
Este script implementa un flujo completo de entrenamiento y optimización
de un modelo XGBoost para clasificación binaria utilizando tres estrategias:

- Modelo base (baseline) sin optimización.
- Optimización automática con Optuna.
- Búsqueda exhaustiva con Grid Search.
- Búsqueda aleatoria con Randomized Search.

Se utiliza estandarización de características antes del entrenamiento.

Flujo del proceso:
1. Carga del dataset Breast Cancer desde sklearn.
2. Separación en conjuntos de entrenamiento y prueba.
3. Escalado de características con StandardScaler.
4. Entrenamiento de modelo XGBoost baseline.
5. Evaluación del modelo baseline.
6. Definición de función objetivo para Optuna.
7. Ejecución de optimización bayesiana con Optuna.
8. Evaluación del mejor modelo encontrado por Optuna.
9. Definición y ejecución de Grid Search con validación cruzada.
10. Evaluación del mejor modelo de Grid Search.
11. Definición y ejecución de Randomized Search.
12. Evaluación del mejor modelo de Randomized Search.

Parámetros:
No recibe parámetros (script ejecutable directamente).

Retorno:
No retorna valores (solo imprime métricas de accuracy y mejores hiperparámetros).

Excepciones:
- ValueError: si los datos no pueden ser escalados o divididos correctamente.
- RuntimeError: si Optuna o GridSearch fallan durante la optimización.
- KeyError: si los parámetros o métricas no están disponibles en los resultados.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna

# ------------------------------------------------------------
# Carga del dataset Breast Cancer
# ------------------------------------------------------------

data = load_breast_cancer()

# Variables predictoras y objetivo
X, y = data.data, data.target

# ------------------------------------------------------------
# División del dataset
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# Estandarización de características
# ------------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# ------------------------------------------------------------
# Modelo baseline XGBoost
# ------------------------------------------------------------

baseline_model = XGBClassifier(eval_metric='logloss', random_state=42)

# Entrenamiento baseline
baseline_model.fit(X_train, y_train)

# Evaluación baseline
baseline_preds = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_preds)

print(f"Baseline XGBoost Exactitud: {baseline_accuracy:.4f}")

# ------------------------------------------------------------
# Función objetivo para Optuna
# ------------------------------------------------------------

def objective(trial):

    params = {
        # Número de árboles
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),

        # Profundidad máxima del árbol
        'max_depth': trial.suggest_int('max_depth', 3, 100),

        # Tasa de aprendizaje
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),

        # Submuestreo de filas
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),

        # Submuestreo de columnas
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),

        # Regularización gamma
        'gamma': trial.suggest_float('gamma', 0, 5),

        # Regularización L1
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),

        # Regularización L2
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }

    # Modelo XGBoost con hiperparámetros sugeridos
    model = XGBClassifier(eval_metric='logloss', random_state=42, **params)

    model.fit(X_train, y_train)

    # Predicción sobre conjunto de prueba
    preds = model.predict(X_test)

    # Retorno de métrica de evaluación
    accuracy = accuracy_score(y_test, preds)

    return accuracy

# ------------------------------------------------------------
# Optimización con Optuna
# ------------------------------------------------------------

study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=50)

print("Best Hyperparameters:", study.best_params)
print("Best Exactitud: ", study.best_value)

# ------------------------------------------------------------
# Grid Search
# ------------------------------------------------------------

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n\n\nGrid Search Best Parameters: ", grid_search.best_params_)
print("Grid Search Best Exactitud:", grid_search.best_score_)

# ------------------------------------------------------------
# Randomized Search
# ------------------------------------------------------------

param_dist = {
    'n_estimators': [50, 100, 200, 300, 400],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("\n\n\nRandom Search Best Parameters:", random_search.best_params_)
print("Random Search Best Exactitud:", random_search.best_score_)

# ------------------------------------------------------------
# Comentario de resultados (referencial)
# ------------------------------------------------------------

# Baseline XGBoost Exactitud: 0.9561
# Optuna Best Exactitud: 0.9649
# Grid Search Best Exactitud: 0.9758
# Random Search Best Exactitud: 0.9758