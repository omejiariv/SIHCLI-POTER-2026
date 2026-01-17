import streamlit as st
import plotly.express as px
import pandas as pd
from PIL import Image
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="SIHCLI-POTER",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TÍTULO Y BIENVENIDA ---
st.title("🌊 Sistema de Información Hidroclimática (SIHCLI-POTER)")
st.markdown("""
**Bienvenido al ecosistema de inteligencia territorial para la seguridad hídrica.**
Esta plataforma integra datos climáticos, hidrológicos y de biodiversidad para soportar la toma de decisiones estratégicas en la cuenca.
""")

st.divider()

# --- DATOS DEL GRÁFICO SUNBURST (Estructura Profunda) ---
data = {
    'id': [
        'SIHCLI-POTER', 
        # NIVEL 1: MÓDULOS PRINCIPALES
        'Clima e Hidrología', 'Aguas Subterráneas', 'Biodiversidad', 'Toma de Decisiones', 'Herramientas',
        
        # NIVEL 2: SUBMÓDULOS
        # Hijos de Clima
        'Precipitación', 'Índices (ENSO)', 'Caudales', 'Temperaturas',
        # Hijos de Aguas Sub
        'Modelo Turc', 'Mapa Recarga', 'Escenarios', 'Balance Hídrico',
        # Hijos de Biodiversidad
        'Monitor GBIF', 'Taxonomía', 'Amenazas IUCN', 'Servicios Ecosistémicos',
        # Hijos de Decisiones
        'Matriz Prioridad', 'Análisis Multicriterio', 'Predios',
        # Hijos de Herramientas
        'Diagnóstico Calidad', 'Detective de Datos',

        # --- NIVEL 3: DESAGREGACIÓN CLIMA (NUEVO) ---
        # Hijos de Precipitación
        'Mapas Isoyetas', 'Series Temporales', 'Análisis de Tendencias', 'Anomalías',
        # Hijos de Índices
        'ONI (Oceanic Niño)', 'SOI (Southern)', 'MEI (Multivariate)',
        # Hijos de Caudales
        'Oferta Hídrica', 'Curvas de Duración', 'Caudales Ecológicos'
    ],
    'parent': [
        '', 
        # Padres Nivel 1
        'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER',
        
        # Padres Nivel 2
        'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología', # Clima
        'Aguas Subterráneas', 'Aguas Subterráneas', 'Aguas Subterráneas', 'Aguas Subterráneas', # Aguas
        'Biodiversidad', 'Biodiversidad', 'Biodiversidad', 'Biodiversidad', # Bio
        'Toma de Decisiones', 'Toma de Decisiones', 'Toma de Decisiones', # Decisiones
        'Herramientas', 'Herramientas', # Herramientas

        # Padres Nivel 3 (Clima)
        'Precipitación', 'Precipitación', 'Precipitación', 'Precipitación',
        'Índices (ENSO)', 'Índices (ENSO)', 'Índices (ENSO)',
        'Caudales', 'Caudales', 'Caudales'
    ],
    'value': [
        100, 
        30, 20, 20, 20, 10, # Ajuste de pesos Nivel 1
        # Nivel 2 (Los valores de Clima se ignoran, se calculan por la suma de sus hijos)
        0, 0, 0, 4, # Precip, Indices, Caudales (0 porque tienen hijos), Temp (4 fijo)
        5, 5, 5, 5, # Aguas
        5, 5, 5, 5, # Bio
        7, 7, 6,    # Decisiones
        5, 5,       # Herramientas
        
        # Nivel 3 (Valores Reales de Clima)
        3, 3, 2, 2, # Precipitación (Total 10)
        3, 3, 2,    # Índices (Total 8)
        3, 3, 2     # Caudales (Total 8)
    ]
}

# --- CREACIÓN DEL GRÁFICO ---
def create_system_map():
    df = pd.DataFrame(data)
    
    fig = px.sunburst(
        df,
        names='id',
        parents='parent',
        values='value',
        color='parent', 
        color_discrete_sequence=px.colors.qualitative.Pastel1, 
        branchvalues='total' 
    )
    
    fig.update_layout(
        title={
            'text': "🗺️ Mapa de Navegación del Sistema",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        font=dict(family="Arial", size=14),
        margin=dict(t=60, l=0, r=0, b=0),
        height=650,
        paper_bgcolor='rgba(0,0,0,0)', 
    )
    
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Módulo: %{parent}<extra></extra>',
        textinfo='label+percent parent'
    )
    
    return fig

# --- LAYOUT PRINCIPAL ---
c1, c2 = st.columns([2, 1])

with c1:
    st.plotly_chart(create_system_map(), use_container_width=True)

with c2:
    st.subheader("📌 Acceso Rápido")
    st.info("Utiliza este gráfico interactivo para entender la estructura del sistema. Haz clic en un sector para hacer zoom.")
    
    st.markdown("### Módulos Destacados")
    
    # --- NUEVO: CLIMA E HIDROLOGÍA ---
    with st.expander("🌦️ Clima e Hidrología"):
        st.write("Tablero de control con series temporales de precipitación, caudales e índices climáticos (ENSO).")
        st.caption("Estado: ✅ Operativo")

    with st.expander("💧 Aguas Subterráneas"):
        st.write("Cálculo de recarga potencial y proyección de escenarios climáticos.")
        st.caption("Estado: ✅ Operativo")
        
    with st.expander("🍃 Biodiversidad"):
        st.write("Conexión con GBIF para monitoreo de especies y amenazas.")
        st.caption("Estado: ✅ Operativo")
        
    with st.expander("🎯 Toma de Decisiones"):
        st.write("Priorización espacial de predios para inversión basada en multicriterio.")
        st.caption("Estado: ✅ Operativo")

# --- FOOTER ACTUALIZADO ---
st.divider()
st.caption("© 2026 omejia CV | SIHCLI-POTER v2.0")