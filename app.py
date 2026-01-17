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
# Aquí definimos la jerarquía: Abuelo -> Padre -> Hijo
data = {
    'id': [
        'SIHCLI-POTER', 
        # --- NIVEL 1: MÓDULOS PRINCIPALES ---
        'Clima e Hidrología', 'Aguas Subterráneas', 'Biodiversidad', 'Toma de Decisiones', 'Herramientas',
        
        # --- NIVEL 2: SUBMÓDULOS (HIJOS) ---
        # Hijos de Clima
        'Precipitación', 'Índices (ENSO)', 'Caudales',
        # Hijos de Aguas Sub
        'Modelo Turc', 'Mapa Recarga', 'Escenarios', 'Balance Hídrico',
        # Hijos de Biodiversidad
        'Monitor GBIF', 'Taxonomía', 'Amenazas IUCN', 'Servicios Ecosistémicos',
        # Hijos de Decisiones
        'Matriz Prioridad', 'Análisis Multicriterio', 'Predios',
        # Hijos de Herramientas (Diagnóstico/Detective)
        'Diagnóstico Calidad', 'Detective de Datos'
    ],
    'parent': [
        '', # Raíz (No tiene padre)
        # Padres Nivel 1
        'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER',
        # Padres Nivel 2 (Conectan con Nivel 1)
        'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología',
        'Aguas Subterráneas', 'Aguas Subterráneas', 'Aguas Subterráneas', 'Aguas Subterráneas',
        'Biodiversidad', 'Biodiversidad', 'Biodiversidad', 'Biodiversidad',
        'Toma de Decisiones', 'Toma de Decisiones', 'Toma de Decisiones',
        'Herramientas', 'Herramientas'
    ],
    'value': [
        100, # Valor Central
        20, 25, 20, 20, 15, # Pesos Nivel 1
        6, 7, 7,            # Clima
        6, 7, 6, 6,         # Aguas
        5, 5, 5, 5,         # Bio
        7, 7, 6,            # Decisiones
        7, 8                # Herramientas
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
        color='parent', # Colorear según el módulo padre
        color_discrete_sequence=px.colors.qualitative.Pastel1, # Paleta profesional y suave
        branchvalues='total' # El tamaño del padre es la suma de los hijos
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
        paper_bgcolor='rgba(0,0,0,0)', # Fondo transparente
    )
    
    # Efecto Hover personalizado
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
    
    with st.expander("💧 Aguas Subterráneas", expanded=True):
        st.write("Cálculo de recarga potencial y proyección de escenarios climáticos.")
        st.caption("Estado: ✅ Operativo")
        
    with st.expander("🍃 Biodiversidad"):
        st.write("Conexión con GBIF para monitoreo de especies y amenazas.")
        st.caption("Estado: ✅ Operativo")
        
    with st.expander("🎯 Toma de Decisiones"):
        st.write("Priorización espacial de predios para inversión basada en multicriterio.")
        st.caption("Estado: ✅ Operativo")

# --- FOOTER ---
st.divider()
st.caption("© 2026 CuencaVerde & Nutresa | SIHCLI-POTER v2.0")