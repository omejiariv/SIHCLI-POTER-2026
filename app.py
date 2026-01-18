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
st.markdown("""
**Bienvenido al ecosistema de inteligencia territorial para la seguridad hídrica.**
Esta plataforma integra datos climáticos, hidrológicos y de biodiversidad para soportar la toma de decisiones estratégicas en la cuenca.
""")

st.divider()

# --- 3. DATOS DEL GRÁFICO SUNBURST (ACTUALIZADO v2.0) ---
# Hemos expandido la rama de Precipitación para mostrar la potencia del nuevo módulo

# A. Identificadores únicos
ids = [
    'SIHCLI-POTER', 
    # NIVEL 1: MÓDULOS PRINCIPALES
    'Clima e Hidrología', 'Aguas Subterráneas', 'Biodiversidad', 'Toma de Decisiones', 'Herramientas',
    
    # NIVEL 2: SUBMÓDULOS CLIMA
    'Precipitación', 'Índices (ENSO)', 'Caudales', 'Temperaturas',
    
    # NIVEL 3: DETALLES PRECIPITACIÓN (Aquí está la actualización)
    'Isoyetas HD', 'Series Temporales', 'Anomalías',
    
    # NIVEL 4: CAPACIDADES ISOYETAS (NUEVO)
    'Escenarios (Min/Max)', 'Pronósticos (2026-40)', 'Variabilidad (Desv.Std)', 'Interpolación RBF',

    # NIVEL 2: OTROS MÓDULOS (Manteniendo estructura original)
    'Modelo Turc', 'Mapa Recarga', 'Balance Hídrico', # Aguas
    'Monitor GBIF', 'Taxonomía', 'Amenazas IUCN',     # Bio
    'Matriz Prioridad', 'Análisis Multicriterio',     # Decisiones
    'Diagnóstico Calidad', 'Detective de Datos',      # Herramientas
    
    # NIVEL 3: DETALLES ÍNDICES Y CAUDALES
    'ONI', 'SOI', 'MEI',               # Índices
    'Oferta Hídrica', 'Curvas Duración' # Caudales
]

# B. Padres (Jerarquía)
parents = [
    '', # Raíz
    # Hijos de Raíz
    'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER', 'SIHCLI-POTER',
    
    # Hijos de Clima e Hidrología
    'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología', 'Clima e Hidrología',
    
    # Hijos de Precipitación (Actualizado)
    'Precipitación', 'Precipitación', 'Precipitación',
    
    # Hijos de Isoyetas HD (NUEVO - Mostramos lo que hace el módulo)
    'Isoyetas HD', 'Isoyetas HD', 'Isoyetas HD', 'Isoyetas HD',

    # Hijos de Aguas Subterráneas
    'Aguas Subterráneas', 'Aguas Subterráneas', 'Aguas Subterráneas',
    # Hijos de Biodiversidad
    'Biodiversidad', 'Biodiversidad', 'Biodiversidad',
    # Hijos de Toma de Decisiones
    'Toma de Decisiones', 'Toma de Decisiones',
    # Hijos de Herramientas
    'Herramientas', 'Herramientas',
    
    # Hijos de Índices
    'Índices (ENSO)', 'Índices (ENSO)', 'Índices (ENSO)',
    # Hijos de Caudales
    'Caudales', 'Caudales'
]

# C. Valores (Peso Visual)
values = [
    100, # Raíz
    35, 20, 15, 20, 10, # Nivel 1 (Clima pesa más ahora)
    
    # Clima (Suma 35)
    15, 8, 8, 4, # Precipitación(15), Índices(8), Caudales(8), Temp(4)
    
    # Precipitación (Suma 15)
    10, 3, 2, # Isoyetas HD(10) es el protagonista, Series(3), Anomalías(2)
    
    # Hijos de Isoyetas HD (Suma 10)
    2.5, 2.5, 2.5, 2.5, # Repartido equitativamente
    
    # Otros Módulos (Pesos referenciales)
    7, 7, 6,    # Aguas
    5, 5, 5,    # Bio
    10, 10,     # Decisiones
    5, 5,       # Herramientas
    
    3, 3, 2,    # Índices
    4, 4        # Caudales
]

# --- 4. CREACIÓN DEL GRÁFICO ---
def create_system_map():
    # Validación de integridad
    if len(ids) != len(parents) or len(ids) != len(values):
        st.error(f"Error Estructural: IDs({len(ids)}) vs Parents({len(parents)}) vs Values({len(values)})")
        return None

    df = pd.DataFrame(dict(ids=ids, parents=parents, values=values))
    
    fig = px.sunburst(
        df,
        names='ids',
        parents='parents',
        values='values',
        branchvalues='total',
        color='parents',
        color_discrete_sequence=px.colors.qualitative.Pastel1
    )
    
    fig.update_layout(
        title={
            'text': "🗺️ Mapa de Navegación del Sistema (v2.0)",
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
    st.subheader("📌 Novedades del Sistema")
    st.info("Utiliza el gráfico interactivo para explorar la estructura actualizada.")
    
    st.markdown("### 🚀 Módulo Estrella")
    
    with st.expander("🗺️ Isoyetas HD (Nuevo)", expanded=True):
        st.write("""
        **Generador Avanzado de Escenarios & Pronósticos:**
        * ✅ Interpolación RBF Normalizada.
        * ✅ Análisis de Mínimos y Máximos Históricos.
        * ✅ Mapa de Variabilidad Temporal.
        * ✅ Pronóstico Climático Lineal (2026-2040).
        * ✅ Descargas GIS (Raster/Vector).
        """)
        st.caption("Estado: ✅ Operativo y Calibrado")

    st.markdown("### Otros Módulos")
    with st.expander("🌦️ Clima e Hidrología"):
        st.write("Tablero de control con series temporales e índices climáticos (ENSO).")
    
    with st.expander("💧 Aguas Subterráneas"):
        st.write("Modelo Turc y balance hídrico.")

    with st.expander("🎯 Toma de Decisiones"):
        st.write("Priorización espacial basada en análisis multicriterio.")

# --- FOOTER ---
st.divider()
st.caption("© 2026 omejia CV | SIHCLI-POTER v2.0")