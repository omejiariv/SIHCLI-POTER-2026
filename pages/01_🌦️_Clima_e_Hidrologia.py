# pages/01_☁️_Clima_e_Hidrologia.py

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sqlalchemy import text
import sys
import os

# --- 1. CONFIGURACIÓN DE RUTAS Y PÁGINA ---
st.set_page_config(page_title="Monitor Hidroclimático", page_icon="🌦️", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from modules.db_manager import get_engine
except ImportError:
    from db_manager import get_engine

# --- 2. FUNCIÓN DE CARGA DE DATOS ---
@st.cache_data(ttl=60) # Recarga cada 60 segundos para ver cambios del Admin
def cargar_datos_mapa():
    engine = get_engine()
    datos = {"estaciones": pd.DataFrame(), "predios": pd.DataFrame()}
    
    if engine:
        try:
            with engine.connect() as conn:
                # Cargar Estaciones
                df_est = pd.read_sql(text("SELECT * FROM estaciones WHERE latitud != 0"), conn)
                datos["estaciones"] = df_est
                
                # Cargar Predios
                df_pre = pd.read_sql(text("SELECT * FROM predios WHERE latitud != 0"), conn)
                datos["predios"] = df_pre
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            
    return datos

# --- 3. INTERFAZ PRINCIPAL ---
st.title("🌦️ Monitor Hidroclimático e Intervenciones")
st.markdown("Visualización integrada de la red de monitoreo y predios gestionados.")

# Cargar datos
data = cargar_datos_mapa()
df_estaciones = data["estaciones"]
df_predios = data["predios"]

# Métricas rápidas
c1, c2, c3 = st.columns(3)
c1.metric("Estaciones Activas", len(df_estaciones), delta="En mapa")
c2.metric("Predios Gestionados", len(df_predios), delta="Georreferenciados")
c3.metric("Municipios Cubiertos", df_estaciones['municipio'].nunique())

# --- 4. CONSTRUCCIÓN DEL MAPA ---
st.subheader("🗺️ Visor Territorial")

# Centro inicial (Medellín aprox)
m = folium.Map(location=[6.5, -75.4], zoom_start=9, tiles="CartoDB positron")

# CAPA 1: ESTACIONES (Azul)
fg_est = folium.FeatureGroup(name="🌧️ Estaciones")
for _, row in df_estaciones.iterrows():
    # Crear popup con HTML bonito
    html = f"""
    <b>{row['nom_est']}</b><br>
    ID: {row['id_estacion']}<br>
    Mun: {row['municipio']}<br>
    Elev: {row['elevacion']} msnm
    """
    folium.Marker(
        location=[row['latitud'], row['longitud']],
        tooltip=row['nom_est'],
        popup=folium.Popup(html, max_width=200),
        icon=folium.Icon(color="blue", icon="cloud", prefix="fa")
    ).add_to(fg_est)
fg_est.add_to(m)

# CAPA 2: PREDIOS (Verde)
fg_pred = folium.FeatureGroup(name="🏡 Predios (Fincas)")
for _, row in df_predios.iterrows():
    html_p = f"""
    <b>{row['nombre_predio']}</b><br>
    Prop: {row['propietario']}<br>
    Área: {row['area_ha']} ha<br>
    Vereda: {row['vereda']}
    """
    folium.Marker(
        location=[row['latitud'], row['longitud']],
        tooltip=f"Finca: {row['nombre_predio']}",
        popup=folium.Popup(html_p, max_width=200),
        icon=folium.Icon(color="green", icon="home", prefix="fa")
    ).add_to(fg_pred)
fg_pred.add_to(m)

# Control de Capas
folium.LayerControl().add_to(m)

# Renderizar en Streamlit
st_folium(m, width="100%", height=600)