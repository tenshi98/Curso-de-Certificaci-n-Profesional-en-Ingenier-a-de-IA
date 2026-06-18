"""
DOCSTRING GENERAL DEL SCRIPT
------------------------------------------------------------
Título:
    Pipeline de clasificación para predicción de churn en clientes (Telco)

Descripción:
    Este script realiza un flujo completo de machine learning sobre un dataset de
    clientes de telecomunicaciones para predecir la variable objetivo 'Churn'
    (abandono de cliente).

    El proceso incluye:
    - Carga y exploración del dataset
    - Limpieza de datos (manejo de valores faltantes)
    - Codificación de variables categóricas
    - Escalado de variables numéricas
    - Entrenamiento de un modelo RandomForest inicial
    - Optimización de hiperparámetros mediante RandomizedSearchCV
    - Evaluación del modelo optimizado
    - Validación cruzada del mejor modelo

Entradas:
    - Archivo CSV: "telco-customer-churn.csv"

Salidas:
    - Métricas de accuracy
    - Reporte de clasificación
    - Mejores hiperparámetros del modelo
    - Resultados de validación cruzada

Dependencias:
    - pandas
    - numpy
    - sklearn

------------------------------------------------------------
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ------------------------------------------------------------
# CARGA DEL DATASET
# ------------------------------------------------------------

# Load dataset from local file
df = pd.read_csv("telco-customer-churn.csv")

# Display dataset information structure (column types, nulls, etc.)
print("Conjunto de datos Info:\n")
print(df.info())

# Show class distribution of target variable
print("\n Class Distribution: \n")
print(df['Churn'].value_counts())

# Show sample records from dataset
print("\n Sample Data:\n", df.head())

# ------------------------------------------------------------
# LIMPIEZA DE DATOS
# ------------------------------------------------------------

# Convert 'TotalCharges' to numeric, forcing invalid parsing to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values in 'TotalCharges' using median
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)

# ------------------------------------------------------------
# CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ------------------------------------------------------------

# Initialize label encoder
label_encoder = LabelEncoder()

# Encode all categorical columns except target
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        df[column] = label_encoder.fit_transform(df[column])

# Encode target variable (Churn)
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# ------------------------------------------------------------
# ESCALADO DE VARIABLES NUMÉRICAS
# ------------------------------------------------------------

# Initialize scaler for numeric features
scaler = StandardScaler()

# Define numeric columns to scale
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Apply standardization
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# ------------------------------------------------------------
# DEFINICIÓN DE FEATURES Y TARGET
# ------------------------------------------------------------

X = df.drop(columns=['Churn'])
y = df['Churn']

# ------------------------------------------------------------
# DIVISIÓN DEL DATASET
# ------------------------------------------------------------

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# MODELO BASE (RANDOM FOREST)
# ------------------------------------------------------------

# Initialize Random Forest classifier with default parameters
rf_model = RandomForestClassifier(random_state=42)

# Train model
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Evaluate baseline model accuracy
accuracy_initial = accuracy_score(y_test, y_pred)

print(f"Initial Model Exactitud: {accuracy_initial:.4f}")

# Print classification report for baseline model
print("\n Informe de clasificación: \n", classification_report(y_test, y_pred))

# ------------------------------------------------------------
# OPTIMIZACIÓN DE HIPERPARÁMETROS (RANDOMIZED SEARCH)
# ------------------------------------------------------------

# Define hyperparameter search space
param_dist = {
    'n_estimators': np.arange(50, 200, 10),
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize RandomizedSearchCV with RandomForest
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Execute hyperparameter search
random_search.fit(X_train, y_train)

# Retrieve best parameters found during search
best_params = random_search.best_params_
print(f"Best Parameters (RandomizedSearchCV): {best_params}")

# ------------------------------------------------------------
# MODELO OPTIMIZADO
# ------------------------------------------------------------

# Extract best estimator model
best_model = random_search.best_estimator_

# Predict using optimized model
y_pred_tuned = best_model.predict(X_test)

# Evaluate optimized model accuracy
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tunes Model Exactitud: {accuracy_tuned:.4f}")

# Print classification report for optimized model
print("\n Classification Report (Tuned Model):\n", classification_report(y_test, y_pred_tuned))

# ------------------------------------------------------------
# VALIDACIÓN CRUZADA
# ------------------------------------------------------------

# Perform 5-fold cross-validation on full dataset
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

# Display cross-validation results
print(f"Cross-Validation Exactitud Scores: {cv_scores}")
print(f"Mean Cross-Validation Exactitud: {cv_scores.mean():.4f}")