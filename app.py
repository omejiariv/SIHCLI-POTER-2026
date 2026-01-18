import streamlit as st
import plotly.express as px
import pandas as pd
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="SIHCLI-POTER",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. TÍTULO Y BIENVENIDA ---
st.title("🌊 Sistema de Información Hidroclimática (SIHCLI-POTER)")
st.markdown("**Sistema de Información Hidroclimática Integrada para la Gestión Integral del Agua y la Biodiversidad en el Norte de la Región Andina.**")

# --- 3. PESTAÑAS DE INICIO (VISIÓN Y CONTEXTO) ---
tab_pres, tab_clima, tab_cap = st.tabs(["📘 Presentación del Sistema", "🏔️ Climatología Andina", "📖 El Aleph"])

with tab_pres:
    st.markdown("### Origen y Visión")
    st.write("""
    **SIHCLI-POTER** nace de la necesidad imperativa de integrar datos, ciencia y tecnología para la toma de decisiones informadas en el territorio. En un contexto de variabilidad climática creciente, la gestión del recurso hídrico y el ordenamiento territorial requieren herramientas que transformen datos dispersos en conocimiento accionable.

    Este sistema no es solo un repositorio de datos; es un **cerebro analítico** diseñado para procesar, modelar y visualizar la complejidad hidrometeorológica de la región Andina. Su arquitectura modular permite desde el monitoreo en tiempo real hasta la proyección de escenarios de cambio climático a largo plazo.
    """)
    
    st.markdown("### Aplicaciones Clave")
    c_app1, c_app2 = st.columns(2)
    with c_app1:
        st.info("**Gestión del Riesgo:** Alertas tempranas y mapas de vulnerabilidad ante eventos extremos (sequías e inundaciones).")
        st.info("**Planeación Territorial (POT):** Insumos técnicos para la zonificación ambiental y la gestión de cuencas.")
    with c_app2:
        st.success("**Agricultura de Precisión:** Calendarios de siembra basados en pronósticos estacionales y zonas de vida.")
        st.warning("**Investigación:** Base de datos depurada y herramientas estadísticas para estudios académicos.")

with tab_clima:
    st.markdown("### 🏔️ La Complejidad de los Andes")
    st.write("""
    La región Andina presenta uno de los sistemas climáticos más complejos del mundo. La interacción entre la Zona de Convergencia Intertropical (ZCIT), los vientos alisios y la topografía escarpada genera microclimas que cambian en distancias cortas.
    
    **SIHCLI-POTER** está diseñado específicamente para capturar esta variabilidad, integrando estaciones en tierra con modelos satelitales para llenar los vacíos de información en zonas de alta montaña.
    """)

with tab_cap:
    st.markdown("### 📖 El Aleph")
    st.caption("El punto que contiene todos los puntos.")
    st.write("Espacio reservado para documentación profunda, referencias bibliográficas y el marco conceptual del proyecto.")

st.divider()

# --- 4. DATOS DEL GRÁFICO SUNBURST (ESTRUCTURA DEL SISTEMA) ---
# Definimos la jerarquía de navegación
ids = [
    'SIHCLI-POTER', 
    # NIVEL 1: GRANDES ÁREAS
    'Clima e Hidrología', 'Aguas Subterráneas', 'Biodiversidad', 'Toma de Decisiones', 'Isoyetas HD', 'Herramientas',
    
    # NIVEL 2: SUB-COMPONENTES
    # Clima
    'Precipitación', 'Índices (ENSO)', 'Caudales', 'Temperaturas',
    # Isoyetas (Ahora como módulo principal)
    'Escenarios', 'Pronósticos', 'Variabilidad',
    # Aguas
    'Modelo Turc', 'Recarga', 'Balance',
    # Bio
    'GBIF', 'Taxonomía', 'Amenazas',
    # Decisiones
    'Priorización', 'Multicriterio',
    # Herramientas
    'Calidad', 'Auditoría'
]

parents = [
    '', 
    # Hijos de Raíz
    'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER',
    
    # Hijos Clima
    'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología',
    # Hijos Isoyetas
    'Isoyetas HD', 'Isoyetas HD', 'Isoyetas HD',
    # Hijos Aguas
    'Aguas Subterráneas', 'Aguas Subterráneas', 'Aguas Subterráneas',
    # Hijos Bio
    'Biodiversidad', 'Biodiversidad', 'Biodiversidad',
    # Hijos Decisiones
    'Toma de Decisiones', 'Toma de Decisiones',
    # Hijos Herramientas
    'Herramientas', 'Herramientas'
]

values = [
    100, 
    20, 15, 15, 15, 20, 15, # Pesos equilibrados para los módulos principales
    5, 5, 5, 5, # Clima
    7, 7, 6,    # Isoyetas
    5, 5, 5,    # Aguas
    5, 5, 5,    # Bio
    7, 8,       # Decisiones
    7, 8        # Herramientas
]

def create_system_map():
    if len(ids) != len(parents) or len(ids) != len(values): return None
    df = pd.DataFrame(dict(ids=ids, parents=parents, values=values))
    fig = px.sunburst(
        df, names='ids', parents='parents', values='values', branchvalues='total',
        color='parents', color_discrete_sequence=px.colors.qualitative.Pastel1
    )
    fig.update_layout(
        title={'text': "🗺️ Mapa de Navegación del Sistema", 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        font=dict(family="Arial", size=14), margin=dict(t=60, l=0, r=0, b=0), height=600, paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_traces(hovertemplate='<b>%{label}</b><br>Sección: %{parent}<extra></extra>', textinfo='label+percent parent')
    return fig

# --- 5. LAYOUT PRINCIPAL (DOS COLUMNAS) ---
c1, c2 = st.columns([1.8, 1.2])

with c1:
    fig = create_system_map()
    if fig: st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("🛠️ Módulos (Aplicaciones Eco-Hidroclimáticas)")
    st.markdown("Acceda a las capacidades analíticas del sistema:")
    
    # 1. ISOYETAS HD
    with st.expander("🗺️ Isoyetas HD (Escenarios & Pronósticos)", expanded=True):
        st.write("""
        **Generador Avanzado de Superficies Climáticas:**
        * ✅ Interpolación RBF Normalizada (Alta Definición).
        * ✅ Análisis de Mínimos y Máximos Históricos.
        * ✅ Mapa de Variabilidad Temporal (Desviación Estándar).
        * ✅ Pronóstico Climático Lineal (2026-2040).
        * ✅ Descargas GIS (Raster/Vector).
        """)
        st.caption("Estado: ✅ Operativo y Calibrado")

    # 2. CLIMA E HIDROLOGÍA
    with st.expander("🌦️ Clima e Hidrología"):
        st.write("""
        **Tablero de Control Hidrometeorológico:**
        * ✅ Monitoreo de series temporales (Precipitación, Nivel, Caudal).
        * ✅ Cálculo de Anomalías e Índices Estandarizados.
        * ✅ Seguimiento de Fenómenos Macroclimáticos (ENSO/ONI).
        * ✅ Análisis de Tendencias (Mann-Kendall).
        """)
        st.caption("Estado: ✅ Operativo")

    # 3. AGUAS SUBTERRÁNEAS
    with st.expander("💧 Aguas Subterráneas"):
        st.write("""
        **Modelación Hidrogeológica Simplificada:**
        * ✅ Balance Hídrico (Método de Turc).
        * ✅ Estimación de Recarga Potencial de Acuíferos.
        * ✅ Escenarios de Infiltración por Cobertura.
        * ✅ Relación Lluvia-Escorrentía.
        """)
        st.caption("Estado: ✅ Operativo")

    # 4. BIODIVERSIDAD
    with st.expander("🍃 Biodiversidad"):
        st.write("""
        **Inteligencia Biológica del Territorio:**
        * ✅ Monitor de Registros Biológicos (Integración GBIF).
        * ✅ Análisis Taxonómico y Funcional.
        * ✅ Filtros por Estado de Amenaza (IUCN / Libros Rojos).
        * ✅ Distribución Espacial de Especies.
        """)
        st.caption("Estado: ✅ Operativo")

    # 5. TOMA DE DECISIONES
    with st.expander("🎯 Toma de Decisiones"):
        st.write("""
        **Herramientas de Planificación Estratégica:**
        * ✅ Matriz de Priorización Espacial.
        * ✅ Análisis Multicriterio (AHP) para Inversiones.
        * ✅ Identificación de Predios Estratégicos.
        * ✅ Reportes de Gestión.
        """)
        st.caption("Estado: ✅ Operativo")

# --- FOOTER ---
st.divider()
st.caption("© 2026 omejia CV | SIHCLI-POTER v3.0 | Plataforma de Inteligencia Territorial")