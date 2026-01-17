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

# --- DATOS DEL GRÁFICO SUNBURST (Estructura del Sistema) ---
data = {
    'id': [
        'SIHCLI-POTER', 
        # NIVEL 1
        'Clima e Hidrología', 'Aguas Subterráneas', 'Biodiversidad', 'Toma de Decisiones', 'Herramientas',
        # NIVEL 2
        'Precipitación', 'Índices (ENSO)', 'Caudales',
        'Modelo Turc', 'Mapa Recarga', 'Escenarios', 'Balance Hídrico',
        'Monitor GBIF', 'Taxonomía', 'Amenazas IUCN', 'Servicios Ecosistémicos',
        'Matriz Prioridad', 'Análisis Multicriterio', 'Predios',
        'Diagnóstico Calidad', 'Detective de Datos'
    ],
    'parent': [
        '', 
        'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER',
        'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología',
        'Aguas Subterráneas', 'Aguas Subterráneas', 'Aguas Subterráneas', 'Aguas Subterráneas',
        'Biodiversidad', 'Biodiversidad', 'Biodiversidad', 'Biodiversidad',
        'Toma de Decisiones', 'Toma de Decisiones', 'Toma de Decisiones',
        'Herramientas', 'Herramientas'
    ],
    'value': [
        100, 
        20, 25, 20, 20, 15, 
        6, 7, 7,            
        6, 7, 6, 6,         
        5, 5, 5, 5,         
        7, 7, 6,            
        7, 8                
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