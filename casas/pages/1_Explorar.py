import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# Configuración de página
st.set_page_config(page_title="Dashboard California Housing", layout="wide", page_icon="🏠")

# Título
st.title("🏠 Dashboard: Precios de Viviendas en California")

# Cargar datos con cache
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df['Latitude'] = data.data['Latitude']
    df['Longitude'] = data.data['Longitude']
    return df

df = load_data()

# Sidebar con filtros
st.sidebar.header("Filtros")
price_range = st.sidebar.slider(
    "Rango de Precio (x$100k)", 
    float(df['MedHouseVal'].min()), 
    float(df['MedHouseVal'].max()),
    (1.0, 5.0)
)

# Filtrar datos
df_filtered = df[(df['MedHouseVal'] >= price_range[0]) & (df['MedHouseVal'] <= price_range[1])]

# Métricas principales
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Viviendas", f"{len(df_filtered):,}")
with col2:
    st.metric("Precio Promedio", f"${df_filtered['MedHouseVal'].mean():.2f}")
with col3:
    st.metric("Edad Promedio", f"{df_filtered['HouseAge'].mean():.1f} años")
with col4:
    st.metric("Habitaciones Prom.", f"{df_filtered['AveRooms'].mean():.1f}")

# Layout de 2 columnas
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📍 Mapa de Ubicaciones")
    # Mapa interactivo
    map_data = df_filtered[['Latitude', 'Longitude', 'MedHouseVal']].rename(
        columns={'Latitude': 'lat', 'Longitude': 'lon'}
    )
    st.map(map_data, size='MedHouseVal', color='#FF4B4B')

with col_right:
    st.subheader("📊 Distribución de Precios")
    st.bar_chart(df_filtered['MedHouseVal'].value_counts().sort_index().head(20))

# Gráficos adicionales
st.subheader("📈 Análisis Comparativo")
tab1, tab2, tab3 = st.tabs(["Edad vs Precio", "Ingreso vs Precio", "Datos Crudos"])

with tab1:
    chart_data = df_filtered[['HouseAge', 'MedHouseVal']].groupby('HouseAge').mean()
    st.line_chart(chart_data)

with tab2:
    st.scatter_chart(
        df_filtered.sample(1000)[['MedInc', 'MedHouseVal']].rename(
            columns={'MedInc': 'Ingreso Medio', 'MedHouseVal': 'Precio Vivienda'}
        )
    )

with tab3:
    st.dataframe(df_filtered.head(50), use_container_width=True)

# Footer
st.markdown("---")
st.caption(f"Dataset: {len(df):,} viviendas en California | Mostrando: {len(df_filtered):,} viviendas")
