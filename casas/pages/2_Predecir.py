import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# El modelo vive en casas/ (un nivel arriba de casas/pages/).
BASE_DIR = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Predecir precio", layout="wide", page_icon="🔮")

st.title("🔮 Predecir valor de vivienda")
st.caption(
    "Ajusta las características del bloque y el modelo estima el valor medio de la "
    "vivienda. Modelo entrenado sobre California Housing."
)


@st.cache_resource
def load_bundle():
    return joblib.load(BASE_DIR / "modelo_california.pkl")


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error(
        "No se encontró `modelo_california.pkl`. Ejecuta "
        "`python train_california_model.py` para generarlo."
    )
    st.stop()

model = bundle["model"]
features = bundle["features"]
stats = bundle["feature_stats"]

# Etiquetas amigables y ayuda por feature (en orden del modelo).
LABELS = {
    "MedInc":     ("Ingreso medio del vecindario (x $10.000)", "Mediana del ingreso de los hogares del bloque."),
    "HouseAge":   ("Antigüedad media de las viviendas (años)", "Edad mediana de las viviendas del bloque."),
    "AveRooms":   ("Habitaciones promedio por hogar", "Número medio de habitaciones por vivienda."),
    "AveBedrms":  ("Dormitorios promedio por hogar", "Número medio de dormitorios por vivienda."),
    "Population": ("Población del bloque", "Personas que viven en el bloque censal."),
    "AveOccup":   ("Ocupantes promedio por hogar", "Personas por vivienda en el bloque."),
    "Latitude":   ("Latitud", "Coordenada norte-sur (California: ~32 a ~42)."),
    "Longitude":  ("Longitud", "Coordenada este-oeste (California: ~-124 a ~-114)."),
}


def slider_for(col):
    s = stats[col]
    label, help_text = LABELS.get(col, (col, None))
    return st.slider(
        label,
        min_value=round(s["min"], 2),
        max_value=round(s["max"], 2),
        value=round(s["median"], 2),
        help=help_text,
    )


# --- Entradas agrupadas ---
st.subheader("Características del bloque")
col_viv, col_vec, col_ubi = st.columns(3)

valores = {}
with col_viv:
    st.markdown("**🏚️ Vivienda**")
    for col in ["HouseAge", "AveRooms", "AveBedrms"]:
        valores[col] = slider_for(col)
with col_vec:
    st.markdown("**👥 Vecindario**")
    for col in ["MedInc", "Population", "AveOccup"]:
        valores[col] = slider_for(col)
with col_ubi:
    st.markdown("**📍 Ubicación**")
    for col in ["Latitude", "Longitude"]:
        valores[col] = slider_for(col)

# Construir el DataFrame respetando el orden de features del modelo.
input_df = pd.DataFrame({col: [valores[col]] for col in features})

# --- Predicción (en vivo) ---
pred_100k = float(model.predict(input_df)[0])
pred_usd = pred_100k * 100_000

st.markdown("---")
res_col, map_col = st.columns([1, 1])

with res_col:
    st.subheader("💰 Valor estimado")
    st.metric("Precio medio de la vivienda", f"${pred_usd:,.0f}")
    st.caption(f"= {pred_100k:.3f} en unidades de $100.000 (objetivo MedHouseVal)")

    m = bundle["metrics"]
    st.caption(
        f"Precisión del modelo — R²={m['r2']:.3f} · RMSE={m['rmse']:.3f} · "
        f"MAE={m['mae']:.3f} (±{m['mae'] * 100_000:,.0f} USD de error típico)."
    )

with map_col:
    st.subheader("📍 Ubicación seleccionada")
    st.map(
        pd.DataFrame({"lat": [valores["Latitude"]], "lon": [valores["Longitude"]]}),
        zoom=5,
        color="#FF4B4B",
    )

with st.expander("Ver datos de entrada enviados al modelo"):
    st.dataframe(input_df, use_container_width=True)
