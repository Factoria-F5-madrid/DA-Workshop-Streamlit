"""
Entrena un modelo de regresion sobre el dataset California Housing y lo serializa.

A diferencia del modelo de coches (modelo_regresion.pkl), entrenado con 4 filas de
juguete, este modelo se entrena sobre datos reales (~20.600 bloques censales de
California) y predice el valor medio de la vivienda (MedHouseVal, en unidades de
$100.000).

Genera: modelo_california.pkl  -> dict con {model, metrics, features, feature_stats}

Uso:
    python train_california_model.py
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# El .pkl se guarda junto a este script (casas/), no en el CWD.
ARTIFACT_PATH = Path(__file__).resolve().parent / "modelo_california.pkl"


def main():
    print("Cargando California Housing...")
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target  # MedHouseVal en unidades de $100.000

    features = list(X.columns)
    print(f"Filas: {len(X):,} | Features: {features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Entrenando HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.08,
        max_depth=8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --- Evaluacion ---
    y_pred = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print(
        f"R2={metrics['r2']:.3f} | RMSE={metrics['rmse']:.3f} "
        f"| MAE={metrics['mae']:.3f} (x$100k)"
    )

    # --- Estadisticos por feature para defaults/rangos de los sliders ---
    # Usamos percentiles 1-99 para evitar que outliers (AveRooms, AveOccup, etc.)
    # desplacen el slider a rangos sin sentido para el usuario.
    feature_stats = {}
    for col in features:
        feature_stats[col] = {
            "min": float(np.percentile(X[col], 1)),
            "max": float(np.percentile(X[col], 99)),
            "median": float(X[col].median()),
        }

    bundle = {
        "model": model,
        "metrics": metrics,
        "features": features,
        "feature_stats": feature_stats,
    }
    joblib.dump(bundle, ARTIFACT_PATH)
    print(f"Modelo guardado en {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
