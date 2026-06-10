import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Rutas ancladas al directorio del script (independiente del CWD de streamlit run).
BASE_DIR = Path(__file__).resolve().parent

st.title('Predicción de Precios de Vehículos')

# --- CLASES PRE Y POST PROCESADO ---

class DummyEncoder():
    def __init__(self):
        self.columns_ = None

    def fit(self, X, y=None):
        X_dummies = pd.get_dummies(X)
        self.columns_ = X_dummies.columns
        return self

    def transform(self, X):
        X_dummies = pd.get_dummies(X)
        for col in self.columns_:
            if col not in X_dummies:
                X_dummies[col] = 0
        return X_dummies[self.columns_]

class PostProcesador:
    def fit(self, y, *_):
        return self

    def round_post(self, y):
        return np.round(y).astype(int)


# --- CARGA DEL MODELO Y POSTPROCESADOR ---

try:
    pipeline = joblib.load(BASE_DIR / 'modelo_regresion.pkl')
    st.success("Modelo cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

try:
    post = joblib.load(BASE_DIR / 'post.pkl')
    st.success("Postprocesador cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el postprocesador: {e}")
    st.stop()

# --- FORMULARIO DE PREDICCIÓN ---

with st.form("prediction_form"):
    st.header("Ingrese los datos del vehículo")


    marca = st.selectbox("Marca", ["ford", "toyota", "honda"])
    color = st.selectbox("Color", ["rojo", "azul", "verde"])

    submit_button = st.form_submit_button("Predecir Precio")

if submit_button:
    try:
        # Crear DataFrame con los datos de entrada
        input_data = pd.DataFrame({
            'marca': [marca],
            'color': [color]
        })

        prediccion = pipeline.predict(input_data)

        # Aplicar postprocesamiento
        prediccion_final = post.round_post(prediccion)[0]

        st.success(f"Precio predicho: ${prediccion_final:,.2f}")

        with st.expander("Detalles de la predicción"):
            st.write(f"**Marca:** {marca}")
            st.write(f"**Color:** {color}")
            st.write(f"**Precio sin redondear:** ${prediccion[0]:,.2f}")

    except Exception as e:
        st.error(f"Error al hacer la predicción: {e}")

# --- INFORMACIÓN ADICIONAL ---

st.markdown("---")
st.markdown("""
**Notas:**
- Este modelo predice precios basado en marca y color del vehículo.
- Los datos de ejemplo incluyen las marcas: ford, toyota, honda.
- Los colores disponibles son: rojo, azul, verde.
- El precio final se redondea al entero más cercano.
""")