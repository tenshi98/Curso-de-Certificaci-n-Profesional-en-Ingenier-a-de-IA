"""
DOCSTRING:
---------
Comparación de modelos de regresión lineal con y sin regularización
utilizando el dataset California Housing.

Descripción:
Este script entrena tres modelos de regresión sobre el dataset de precios
de viviendas de California:

- Regresión lineal estándar (sin regularización).
- Regresión Ridge (regularización L2).
- Regresión Lasso (regularización L1).

Se comparan los modelos utilizando el error cuadrático medio (MSE) y
se muestran los coeficientes aprendidos por cada modelo.

Flujo del proceso:
1. Carga del dataset California Housing desde sklearn.
2. Separación en variables predictoras (X) y objetivo (y).
3. División en conjuntos de entrenamiento y prueba.
4. Entrenamiento de regresión lineal estándar.
5. Evaluación del modelo mediante MSE.
6. Entrenamiento de regresión Ridge.
7. Evaluación del modelo Ridge mediante MSE.
8. Entrenamiento de regresión Lasso.
9. Evaluación del modelo Lasso mediante MSE.
10. Visualización de coeficientes de cada modelo.

Parámetros:
No recibe parámetros (script ejecutable directamente).

Retorno:
No retorna valores (solo imprime métricas de error y coeficientes).

Excepciones:
- ValueError: si los datos no pueden dividirse correctamente.
- RuntimeError: si el modelo no converge (posible en Lasso en ciertos casos).
- KeyError: si las variables del dataset no coinciden con los esperados.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------
# Carga del dataset California Housing
# ------------------------------------------------------------

california = fetch_california_housing()

# Variables predictoras y objetivo
X, y = california.data, california.target

# Nombres de características
feature_names = california.feature_names

# ------------------------------------------------------------
# División del dataset
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# Información del dataset
# ------------------------------------------------------------

print("Feature Names:\n", feature_names)

print("\n Sample Data:\n", pd.DataFrame(X, columns=feature_names).head())

# ------------------------------------------------------------
# Regresión lineal sin regularización
# ------------------------------------------------------------

lr_model = LinearRegression()

# Entrenamiento modelo lineal
lr_model.fit(X_train, y_train)

# Predicción modelo lineal
y_pred = lr_model.predict(X_test)

# Evaluación modelo lineal
mse_lr = mean_squared_error(y_test, y_pred)

print(f"Regresión lineal MSE (No Regularization): {mse_lr:.4f}")

print("Coefficients:\n", lr_model.coef_)

# ------------------------------------------------------------
# Regresión Ridge (L2)
# ------------------------------------------------------------

ridge_model = Ridge(alpha=0.1)

# Entrenamiento Ridge
ridge_model.fit(X_train, y_train)

# Predicción Ridge
y_pred_ridge = ridge_model.predict(X_test)

# Evaluación Ridge
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print(f"Regresión de cresta MSE: {mse_ridge:.4f}")

print("Coefficients:\n", ridge_model.coef_)

# ------------------------------------------------------------
# Regresión Lasso (L1)
# ------------------------------------------------------------

lasso_model = Lasso(alpha=0.1)

# Entrenamiento Lasso
lasso_model.fit(X_train, y_train)

# Predicción Lasso
y_pred_lasso = lasso_model.predict(X_test)

# Evaluación Lasso
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print(f"Regresión Lasso MSE: {mse_lasso:.4f}")

print("Coefficients:\n", lasso_model.coef_)