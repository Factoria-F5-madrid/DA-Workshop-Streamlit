import streamlit as st
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="California Housing — Workshop",
    layout="wide",
    page_icon="🏠",
)

st.title("🏠 California Housing — Exploración y Predicción")

st.markdown("""
Proyecto de *data analytics* sobre el dataset **California Housing** (~20.640 bloques
censales de California, 1990). La app está organizada en dos páginas que puedes
abrir desde el menú lateral:

- **📊 Explorar** — dashboard interactivo: mapa, métricas y gráficos para entender los datos.
- **🔮 Predecir** — estima el valor medio de una vivienda con un modelo de regresión
  entrenado sobre este mismo dataset.
""")

# --- Métricas del modelo entrenado ---
try:
    bundle = joblib.load(BASE_DIR / "modelo_california.pkl")
    m = bundle["metrics"]

    st.subheader("📈 Modelo de predicción")
    st.caption(
        "HistGradientBoostingRegressor entrenado sobre California Housing "
        f"({m['n_train']:,} filas de entrenamiento / {m['n_test']:,} de prueba)."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{m['r2']:.3f}", help="Proporción de varianza explicada (1.0 = perfecto)")
    c2.metric("RMSE", f"{m['rmse']:.3f}", help="Error cuadrático medio, en unidades de $100.000")
    c3.metric("MAE", f"{m['mae']:.3f}", help="Error absoluto medio, en unidades de $100.000")
except FileNotFoundError:
    st.warning(
        "Aún no existe `modelo_california.pkl`. Ejecuta `python train_california_model.py` "
        "para entrenar el modelo."
    )

st.markdown("---")
st.caption(
    "Workshop de Data Analytics · Dataset: `sklearn.datasets.fetch_california_housing`. "
    "El objetivo (MedHouseVal) está expresado en unidades de $100.000."
)
