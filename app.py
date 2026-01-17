import streamlit as st
import plotly.express as px
import pandas as pd
import os

# --- 1. CONFIGURACIÓN DE PÁGINA (Debe ser lo primero) ---
st.set_page_config(
    page_title="SIHCLI-POTER",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. TÍTULO Y BIENVENIDA ---
st.title("🌊 Sistema de Información Hidroclimática (SIHCLI-POTER)")
st.markdown("""
**Bienvenido al ecosistema de inteligencia territorial para la seguridad hídrica.**
Esta plataforma integra datos climáticos, hidrológicos y de biodiversidad para soportar la toma de decisiones estratégicas en la cuenca.
""")

st.divider()

# --- 3. DATOS DEL GRÁFICO SUNBURST (Estructura Profunda Corregida) ---
# Se definen las listas por separado para asegurar la integridad de los datos

# A. Identificadores únicos de cada sección
ids = [
    'SIHCLI-POTER', 
    # NIVEL 1: MÓDULOS
    'Clima e Hidrología', 'Aguas Subterráneas', 'Biodiversidad', 'Toma de Decisiones', 'Herramientas',
    
    # NIVEL 2: SUBMÓDULOS
    # Clima (Padres de Nivel 3)
    'Precipitación', 'Índices (ENSO)', 'Caudales', 'Temperaturas',
    # Aguas
    'Modelo Turc', 'Mapa Recarga', 'Escenarios', 'Balance Hídrico',
    # Bio
    'Monitor GBIF', 'Taxonomía', 'Amenazas IUCN', 'Servicios Ecosistémicos',
    # Decisiones
    'Matriz Prioridad', 'Análisis Multicriterio', 'Predios',
    # Herramientas
    'Diagnóstico Calidad', 'Detective de Datos',

    # NIVEL 3: DETALLES CLIMA
    # Hijos de Precipitación
    'Mapas Isoyetas', 'Series Temporales', 'Análisis de Tendencias', 'Anomalías',
    # Hijos de Índices
    'ONI (Oceanic Niño)', 'SOI (Southern)', 'MEI (Multivariate)',
    # Hijos de Caudales
    'Oferta Hídrica', 'Curvas de Duración', 'Caudales Ecológicos'
]

# B. Padres (De quién depende cada ID)
parents = [
    '', # Raíz
    # Padres Nivel 1
    'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER',
    
    # Padres Nivel 2
    'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología', # Clima
    'Aguas Subterráneas', 'Aguas Subterráneas', 'Aguas Subterráneas', 'Aguas Subterráneas', # Aguas
    'Biodiversidad', 'Biodiversidad', 'Biodiversidad', 'Biodiversidad', # Bio
    'Toma de Decisiones', 'Toma de Decisiones', 'Toma de Decisiones', # Decisiones
    'Herramientas', 'Herramientas', # Herramientas

    # Padres Nivel 3 (Dependen de los submódulos de Clima)
    'Precipitación', 'Precipitación', 'Precipitación', 'Precipitación',
    'Índices (ENSO)', 'Índices (ENSO)', 'Índices (ENSO)',
    'Caudales', 'Caudales', 'Caudales'
]

# C. Valores (Peso visual)
# Nota: En Sunburst 'total', el valor del padre debe ser >= suma de hijos
values = [
    100, # SIHCLI (Raíz)
    30, 20, 20, 20, 10, # Nivel 1 (Suman 100)
    
    # Nivel 2 (Clima tiene hijos, su valor se calcula automático o debe coincidir)
    10, 8, 8, 4, # Precip(10), Indices(8), Caudales(8), Temp(4) -> Suma 30 (Correcto)
    5, 5, 5, 5,  # Aguas (Suma 20)
    5, 5, 5, 5,  # Bio (Suma 20)
    7, 7, 6,     # Decisiones (Suma 20)
    5, 5,        # Herramientas (Suma 10)

    # Nivel 3 (Hijos de Clima)
    3, 3, 2, 2,  # Hijos Precipitación (Suman 10)
    3, 3, 2,     # Hijos Índices (Suman 8)
    3, 3, 2      # Hijos Caudales (Suman 8)
]

# --- 4. CREACIÓN DEL GRÁFICO ---
def create_system_map():
    # Verificación de seguridad para evitar pantalla blanca
    if len(ids) != len(parents) or len(ids) != len(values):
        st.error(f"Error de Estructura: IDs({len(ids)}), Parents({len(parents)}), Values({len(values)}) no coinciden.")
        return None

    df = pd.DataFrame(dict(ids=ids, parents=parents, values=values))
    
    fig = px.sunburst(
        df,
        names='ids',
        parents='parents',
        values='values',
        branchvalues='total', # Importante para que los tamaños sean proporcionales reales
        color='parents', # Colorear por módulo padre
        color_discrete_sequence=px.colors.qualitative.Pastel1
    )
    
    fig.update_layout(
        title={
            'text': "🗺️ Mapa de Navegación del Sistema",
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        font=dict(family="Arial", size=14),
        margin=dict(t=60, l=0, r=0, b=0),
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Sección: %{parent}<extra></extra>',
        textinfo='label+percent parent'
    )
    
    return fig

# --- 5. LAYOUT PRINCIPAL ---
c1, c2 = st.columns([2, 1])

with c1:
    fig = create_system_map()
    if fig:
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("📌 Acceso Rápido")
    st.info("Utiliza el gráfico interactivo para explorar la estructura. Haz clic en un sector para hacer zoom.")
    
    st.markdown("### Módulos Destacados")
    
    with st.expander("🌦️ Clima e Hidrología"):
        st.write("Tablero de control con series temporales, análisis de isoyetas, anomalías e índices climáticos (ENSO).")
        st.caption("Estado: ✅ Operativo")

    with st.expander("💧 Aguas Subterráneas"):
        st.write("Cálculo de recarga potencial (Turc), mapas de infiltración y proyección de escenarios.")
        st.caption("Estado: ✅ Operativo")
        
    with st.expander("🍃 Biodiversidad"):
        st.write("Monitor de especies (GBIF), taxonomía y análisis de amenazas IUCN.")
        st.caption("Estado: ✅ Operativo")
        
    with st.expander("🎯 Toma de Decisiones"):
        st.write("Priorización espacial de predios para inversión basada en análisis multicriterio.")
        st.caption("Estado: ✅ Operativo")

# --- FOOTER ---
st.divider()
st.caption("© 2026 omejia CV | SIHCLI-POTER v2.0")