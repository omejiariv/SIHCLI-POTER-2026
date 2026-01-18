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

# --- 3. PESTAÑAS DE INICIO (NUEVA ESTRUCTURA) ---
# Se define la nueva estructura de pestañas solicitada
tab_pres, tab_modulos, tab_clima, tab_aleph = st.tabs([
    "📘 Presentación del Sistema", 
    "🛠️ Módulos y Capacidades", 
    "🏔️ Climatología Andina", 
    "📖 El Aleph"
])

# --- PESTAÑA 1: PRESENTACIÓN ---
with tab_pres:
    with st.expander("Origen y Visión", expanded=True):
        st.write("""
        **SIHCLI-POTER** nace de la necesidad imperativa de integrar datos, ciencia y tecnología para la toma de decisiones informadas en el territorio. En un contexto de variabilidad climática creciente, la gestión del recurso hídrico y el ordenamiento territorial requieren herramientas que transformen datos dispersos en conocimiento accionable.

        Este sistema no es solo un repositorio de datos; es un **cerebro analítico** diseñado para procesar, modelar y visualizar la complejidad hidrometeorológica de la región Andina. Su arquitectura modular permite desde el monitoreo en tiempo real hasta la proyección de escenarios de cambio climático a largo plazo.
        """)

# --- PESTAÑA 2: MÓDULOS Y CAPACIDADES (NUEVA) ---
with tab_modulos:
    # Sección A: Aplicaciones Clave
    with st.expander("🎯 Aplicaciones Clave", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.info("**Gestión del Riesgo:** Alertas tempranas y mapas de vulnerabilidad ante eventos extremos (sequías e inundaciones).")
            st.info("**Planeación Territorial (POT):** Insumos técnicos para la zonificación ambiental y la gestión de cuencas.")
        with c2:
            st.success("**Agricultura de Precisión:** Calendarios de siembra basados en pronósticos estacionales y zonas de vida.")
            st.warning("**Investigación:** Base de datos depurada y herramientas estadísticas para estudios académicos.")

    # Sección B: Arquitectura del Sistema
    with st.expander("🏗️ Arquitectura del Sistema: Módulos Especializados", expanded=True):
        st.markdown("""
        **SIHCLI-POTER está estructurado en módulos especializados interconectados:**

        * **🚨 Monitoreo (Tiempo Real):**
            * Tablero de control con las últimas lecturas de estaciones telemétricas.
            * Alertas inmediatas de umbrales críticos.
        
        * **🗺️ Distribución Espacial:**
            * Mapas interactivos para visualizar la red de monitoreo.
            * Análisis de cobertura espacial y densidad de datos.
        
        * **🔮 Pronóstico Climático & ENSO:**
            * Integración directa con el **IRI (Columbia University)** para pronósticos oficiales de El Niño/La Niña.
            * Modelos de predicción local (Prophet, SARIMA) y análisis de probabilidades.
        
        * **📉 Tendencias y Riesgo:**
            * Análisis estadístico de largo plazo (Mann-Kendall) para detectar si llueve más o menos que antes.
            * Mapas de vulnerabilidad hídrica interpolados.
        
        * **🛰️ Satélite y Sesgo:**
            * Comparación de datos de tierra vs. reanálisis satelital (ERA5-Land).
            * Herramientas para corregir y rellenar series históricas.
        
        * **🌱 Zonas de Vida y Cobertura:**
            * Cálculo automático de la clasificación de Holdridge.
            * Análisis de uso del suelo y cobertura vegetal.
        """)

# --- PESTAÑA 3: CLIMATOLOGÍA ---
with tab_clima:
    with st.expander("🏔️ La Complejidad de los Andes", expanded=False):
        st.write("""
        La región Andina presenta uno de los sistemas climáticos más complejos del mundo. La interacción entre la Zona de Convergencia Intertropical (ZCIT), los vientos alisios y la topografía escarpada genera microclimas que cambian en distancias cortas.
        
        **SIHCLI-POTER** está diseñado específicamente para capturar esta variabilidad, integrando estaciones en tierra con modelos satelitales para llenar los vacíos de información en zonas de alta montaña.
        """)

# --- PESTAÑA 4: EL ALEPH ---
with tab_aleph:
    with st.expander("📖 Fragmento de 'El Aleph' - Jorge Luis Borges (1945)", expanded=True):
        st.markdown("""
        > *"... Todo lenguaje es un alfabeto de símbolos cuyo ejercicio presupone un pasado que los interlocutores comparten; ¿cómo transmitir a los otros el infinito Aleph, que mi temerosa memoria apenas abarca? (...)*
        >
        > *En la parte inferior del escalón, hacia la derecha, vi una pequeña esfera tornasolada, de casi intolerable fulgor. Al principio la creí giratoria; luego comprendí que ese movimiento era una ilusión producida por los vertiginosos espectáculos que encerraba. El diámetro del Aleph sería de dos o tres centímetros, pero el espacio cósmico estaba ahí, sin disminución de tamaño. Cada cosa (la luna del espejo, digamos) era infinitas cosas, porque yo la veía claramente desde todos los puntos del universo.*
        >
        > *Vi el populoso mar, vi el alba y la tarde, vi las muchedumbres de América, vi una plateada telaraña en el centro de una negra pirámide, vi un laberinto roto (era Londres), vi interminables ojos inmediatos escrutándose en mí como en un espejo, vi todos los espejos del planeta y ninguno me reflejó...*
        >
        > *Vi el engranaje del amor y la modificación de la muerte, vi el Aleph, desde todos los puntos, vi en el Aleph la tierra, y en la tierra otra vez el Aleph y en el Aleph la tierra, vi mi cara y mis vísceras, vi tu cara, y sentí vértigo y lloré, porque mis ojos habían visto ese objeto secreto y conjetural, cuyo nombre usurpan los hombres, pero que ningún hombre ha mirado: el inconcebible universo."*
        """)

st.divider()

# --- 4. DATOS DEL GRÁFICO SUNBURST ---
ids = [
    'SIHCLI-POTER', 
    'Clima e Hidrología', 'Aguas Subterráneas', 'Biodiversidad', 'Toma de Decisiones', 'Isoyetas HD', 'Herramientas',
    'Precipitación', 'Índices (ENSO)', 'Caudales', 'Temperaturas',
    'Escenarios', 'Pronósticos', 'Variabilidad',
    'Modelo Turc', 'Recarga', 'Balance',
    'GBIF', 'Taxonomía', 'Amenazas',
    'Priorización', 'Multicriterio',
    'Calidad', 'Auditoría'
]

parents = [
    '', 
    'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER',
    'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología',
    'Isoyetas HD', 'Isoyetas HD', 'Isoyetas HD',
    'Aguas Subterráneas', 'Aguas Subterráneas', 'Aguas Subterráneas',
    'Biodiversidad', 'Biodiversidad', 'Biodiversidad',
    'Toma de Decisiones', 'Toma de Decisiones',
    'Herramientas', 'Herramientas'
]

values = [100, 20, 15, 15, 15, 20, 15, 5, 5, 5, 5, 7, 7, 6, 5, 5, 5, 5, 5, 5, 7, 8, 7, 8]

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

# --- 5. LAYOUT PRINCIPAL ---
c1, c2 = st.columns([1.8, 1.2])

with c1:
    fig = create_system_map()
    if fig: st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("🛠️ Módulos (Aplicaciones Eco-Hidroclimáticas)")
    st.markdown("Acceda a las capacidades analíticas del sistema:")
    
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

    with st.expander("🌦️ Clima e Hidrología"):
        st.write("""
        **Tablero de Control Hidrometeorológico:**
        * ✅ Monitoreo de series temporales (Precipitación, Nivel, Caudal).
        * ✅ Cálculo de Anomalías e Índices Estandarizados.
        * ✅ Seguimiento de Fenómenos Macroclimáticos (ENSO/ONI).
        * ✅ Análisis de Tendencias (Mann-Kendall).
        """)
        st.caption("Estado: ✅ Operativo")

    with st.expander("💧 Aguas Subterráneas"):
        st.write("""
        **Modelación Hidrogeológica Simplificada:**
        * ✅ Balance Hídrico (Método de Turc).
        * ✅ Estimación de Recarga Potencial de Acuíferos.
        * ✅ Escenarios de Infiltración por Cobertura.
        """)
        st.caption("Estado: ✅ Operativo")

    with st.expander("🍃 Biodiversidad"):
        st.write("""
        **Inteligencia Biológica del Territorio:**
        * ✅ Monitor de Registros Biológicos (Integración GBIF).
        * ✅ Análisis Taxonómico y Funcional.
        * ✅ Filtros por Estado de Amenaza (IUCN).
        """)
        st.caption("Estado: ✅ Operativo")

    with st.expander("🎯 Toma de Decisiones"):
        st.write("""
        **Herramientas de Planificación Estratégica:**
        * ✅ Matriz de Priorización Espacial.
        * ✅ Análisis Multicriterio (AHP) para Inversiones.
        """)
        st.caption("Estado: ✅ Operativo")

# --- FOOTER ---
st.divider()
st.caption("© 2026 omejia CV | SIHCLI-POTER v3.0 | Un Aleph Hidroclimático: Plataforma de Inteligencia Territorial")