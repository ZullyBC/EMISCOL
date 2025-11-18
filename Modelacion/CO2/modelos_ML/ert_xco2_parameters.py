# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 12:18:30 2025

@author: Zully JBC

Nota: Este código entrena un modelo de ERT (Extreme Random Forest o Extra Trees) 
para la predicción del xco2, primero identifica los mejores parámetros según los 
datos de entrada, despúes hace una validación cruzada del modelo, imprime las 
métricas del modelo y un plot con la dispersión de los datos, para finalmente 
entrenar el ERT con todos los datos y guardarlo por medio de la libreria joblib. 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import (cross_val_predict, KFold, learning_curve)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer

# 1. Cargar y preprocesar datos
df = pd.read_csv("./datos_post/data_ml_oco2.csv")
df["world_pop"] = df["world_pop"].replace(-99999.0, 0).fillna(0)

features = [
    'lat','lon','t2m', 'd2m', 'sp', 'u10', 'v10', 'co2_carbon', 'srtm', 
    'ndvi', 'land_cover', 'world_pop'
]
X = df[features]
y = df["xco2"]

imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 2. Hiperparámetros para ExtraTrees
param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3, 0.5]
}

scoring = {
    'MSE': 'neg_mean_squared_error',
    'MAE': 'neg_mean_absolute_error'
}

random_search = RandomizedSearchCV(
    estimator=ExtraTreesRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=75,
    cv=5,
    scoring=scoring,
    refit='MSE',
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X, y)

print("Mejores parámetros encontrados:", random_search.best_params_)

best_model = random_search.best_estimator_

# 3. Curva de aprendizaje
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X, y,
    cv=5,
    scoring='neg_mean_squared_error',
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    n_jobs=-1
)

train_rmse = np.sqrt(-train_scores)
val_rmse = np.sqrt(-val_scores)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_rmse.mean(axis=1), 'o-', label='Train RMSE')
plt.plot(train_sizes, val_rmse.mean(axis=1), 'o-', label='Validation RMSE')
plt.xlabel('Proporción de entrenamiento')
plt.ylabel('RMSE (ppm)')
plt.title('Curva de Aprendizaje (ExtraTrees)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Validación cruzada
kf = KFold(n_splits=10, shuffle=True, random_state=42)
y_pred = cross_val_predict(best_model, X, y, cv=kf)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

print("\nMétricas optimizadas:")
print(f"\nR²: {r2:.3f}, RMSE: {rmse:.3f} ppm, MAE: {mae:.3f} ppm")

# 5. Observado vs Predicho
plt.rcParams['font.family'] = ['Times New Roman', 'serif']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 7))
plt.scatter(y, y_pred, alpha=0.5, edgecolors='white', linewidths=0.5, s=40, label='Datos')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='1:1')
m, b = np.polyfit(y, y_pred, 1)
plt.plot([y.min(), y.max()], [m*y.min()+b, m*y.max()+b], 'r-', label=f'y={m:.2f}x+{b:.2f}')

plt.xlabel(r"OCO-2 XCO$_2$ Observado (ppm)", fontsize=12, labelpad=10)
plt.ylabel(r"ERT XCO$_2$ Predicho (ppm)", fontsize=12, labelpad=10)
plt.title("Observado vs Predicho (ERT) - 5-fold CV", fontsize=14, pad=20)

textstr = '\n'.join((f'N = {len(y):,}'.replace(',', '.'),
                     f'R² = {r2:.2f}',
                     f'RMSE = {rmse:.2f} ppm',
                     f'MAE = {mae:.2f} ppm'))
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
               verticalalignment='top', bbox=props, fontsize=11)

plt.legend(loc='lower right', frameon=True, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 6. Importancia de variables
importances = best_model.feature_importances_ * 100
feat_imp = pd.DataFrame({'Variable': features, 'Importancia (%)': importances})
print("\nImportancia de Variables:")
print(feat_imp.sort_values('Importancia (%)', ascending=False).to_string(index=False))


# 8. Guardar modelo entrenado
joblib.dump(best_model, "modelo_ert_xco2.joblib")
print("Modelo guardado como 'modelo_ert_xco2.joblib'")
