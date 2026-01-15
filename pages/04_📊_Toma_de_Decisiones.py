# Módulo de Soporte a Decisiones

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from sqlalchemy import create_engine
from scipy.interpolate import griddata
import sys
import os

# --- SETUP ---
st.set_page_config(page_title="Matriz de Decisiones", page_icon="🎯", layout="wide")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
except Exception as e:
    st.error(f"Error de sistema: {e}")
    st.stop()

st.title("🎯 Priorización de Áreas de Intervención")

# --- DOCUMENTACIÓN TÉCNICA (NUEVO) ---
with st.expander("📘 Documentación Técnica: Metodología, Ecuaciones y Fuentes", expanded=False):
    st.markdown("""
    ### 1. Marco Conceptual
    Este módulo implementa un **Análisis Multicriterio Espacial (SMCA)** simplificado, diseñado para apoyar la toma de decisiones en la gestión de cuencas. Se basa en la superposición ponderada de dos dimensiones críticas:
    * **Oferta Hídrica (Recarga Potencial):** Identifica zonas clave para el ciclo hidrológico.
    * **Valor Ecosistémico:** Identifica zonas de importancia biológica (basado en altitud y humedad como proxis).

    ### 2. Metodología de Cálculo
    El proceso se ejecuta en tiempo real siguiendo estos pasos:
    1.  **Ingesta Híbrida:** Se georreferencian las estaciones mediante catálogo oficial (CSV) y se consultan sus series históricas de precipitación en base de datos SQL.
    2.  **Interpolación Robusta:** Se genera una superficie continua utilizando un método híbrido:
        * *Linear Interpolation:* Para suavidad en zonas con alta densidad de estaciones.
        * *Nearest Neighbor:* Para rellenar vacíos en los bordes y evitar zonas sin datos (NaN).
    3.  **Modelación Hidro-Climática:** Se aplica el método de Turc para estimar el balance hídrico.
    4.  **Normalización y Ponderación:** Las variables se escalan de 0 a 1 y se combinan según los pesos definidos por el usuario.

    ### 3. Ecuaciones Principales
    
    **A. Estimación de Temperatura (Gradiente Altitudinal)**
    $$ T_{est} = 30 - (0.0065 \times Altitud) $$
    
    **B. Evapotranspiración Real (Fórmula de Turc)**
    Capacidad evaporativa del aire $L(t)$:
    $$ L(t) = 300 + 25T + 0.05T^3 $$
    
    Evapotranspiración Real ($ETR$):
    $$ ETR = \\frac{P}{\\sqrt{0.9 + (\\frac{P}{L(t)})^2}} $$
    *Donde $P$ es la precipitación media anual.*

    **C. Recarga Hídrica Potencial ($R$)**
    $$ R = P - ETR $$

    **D. Índice de Prioridad ($Score$)**
    $$ Score = (R_{norm} \times W_{agua}) + (Bio_{norm} \times W_{bio}) $$

    ### 4. Fuentes de Información
    * **Climatología:** Base de datos SIHCLI (Series históricas IDEAM/EPM procesadas).
    * **Cartografía:** Capas vectoriales de Cuencas y Municipios (Gobernación de Antioquia/IGAC).
    * **Ubicación Estaciones:** Catálogo `mapaCVENSO.csv`.
    """)

st.markdown("---")

# --- 1. SELECTOR ---
ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

# --- 2. PONDERACIÓN ---
with st.sidebar:
    st.divider()
    st.header("⚖️ Criterios de Decisión")
    w_agua = st.slider("💧 Peso: Hídrico", 0, 100, 60, 5)
    w_bio = st.slider("🍃 Peso: Ecosistémico", 0, 100, 40, 5)
    
    total = w_agua + w_bio
    pct_agua = w_agua / (total if total > 0 else 1)
    pct_bio = w_bio / (total if total > 0 else 1)
    
    st.caption(f"**Distribución:** Agua {pct_agua:.0%} | Bio {pct_bio:.0%}")
    st.divider()
    # Umbral por defecto en 0 para asegurar visibilidad inicial
    umbral_prioridad = st.slider("Filtrar Prioridad Alta (%)", 0, 90, 0)

# --- FUNCIÓN DE INTERPOLACIÓN ROBUSTA (HÍBRIDA) ---
def interpolacion_segura(points, values, grid_x, grid_y):
    """
    Combina 'linear' para suavidad y 'nearest' para rellenar bordes.
    Garantiza 0 huecos blancos.
    """
    # 1. Intento Lineal (Suave, pero deja huecos fuera del polígono convexo)
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear')
    
    # 2. Relleno Nearest (Cuadriculado, pero llena todo)
    mask = np.isnan(grid_z0)
    if np.any(mask):
        grid_z1 = griddata(points, values, (grid_x, grid_y), method='nearest')
        # Rellenar los huecos del lineal con el nearest
        grid_z0[mask] = grid_z1[mask]
    
    return grid_z0

# --- 3. MOTOR DE ANÁLISIS ---
if gdf_zona is not None and not gdf_zona.empty:
    engine = create_engine(st.secrets["DATABASE_URL"])
    
    try:
        # A. CARGAR CATÁLOGO (CSV)
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mapaCVENSO.csv')
        
        if os.path.exists(csv_path):
            df_estaciones_all = pd.read_csv(
                csv_path, sep=';', decimal=',', encoding='latin-1', 
                usecols=['Id_estacio', 'Longitud_geo', 'Latitud_geo', 'alt_est', 'Nom_Est']
            )
            df_estaciones_all.rename(columns={'Id_estacio': 'id_estacion', 'Longitud_geo': 'longitude', 'Latitud_geo': 'latitude', 'Nom_Est': 'nombre'}, inplace=True)
            
            # Limpieza
            for col in ['latitude', 'longitude', 'alt_est']:
                df_estaciones_all[col] = pd.to_numeric(df_estaciones_all[col], errors='coerce')
            
            df_estaciones_all.dropna(subset=['latitude', 'longitude'], inplace=True)
            df_estaciones_all['alt_est'] = df_estaciones_all['alt_est'].fillna(altitud_ref)

            # B. FILTRO ESPACIAL (Buffer generoso para asegurar bordes)
            minx, miny, maxx, maxy = gdf_zona.buffer(0.05).total_bounds
            mask = (
                (df_estaciones_all['longitude'] >= minx) & (df_estaciones_all['longitude'] <= maxx) &
                (df_estaciones_all['latitude'] >= miny) & (df_estaciones_all['latitude'] <= maxy)
            )
            df_filtrada = df_estaciones_all[mask].copy()
            
            # C. DATOS PRECIPITACIÓN (SQL)
            if not df_filtrada.empty:
                ids_validos = df_filtrada['id_estacion'].unique()
                ids_sql = ",".join([f"'{str(x)}'" for x in ids_validos])
                
                if ids_sql:
                    q_ppt = f"SELECT id_estacion_fk as id_estacion, AVG(precipitation) * 12 as p_anual FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_sql}) GROUP BY id_estacion_fk"
                    df_ppt = pd.read_sql(q_ppt, engine)
                    
                    df_filtrada['id_estacion'] = df_filtrada['id_estacion'].astype(str)
                    df_ppt['id_estacion'] = df_ppt['id_estacion'].astype(str)
                    
                    df_data = pd.merge(df_filtrada, df_ppt, on='id_estacion', how='inner')
                else:
                    df_data = pd.DataFrame()
            else:
                df_data = pd.DataFrame()

            # --- D. CÁLCULOS ---
            if len(df_data) >= 3:
                with st.spinner("🎨 Generando mapa continuo..."):
                    # 1. Crear Malla (Grid)
                    grid_x, grid_y = np.mgrid[minx:maxx:100j, miny:maxy:100j]
                    
                    # 2. Interpolación Híbrida (Aquí está la magia)
                    points = df_data[['longitude', 'latitude']].values
                    
                    # Lluvia
                    values_p = df_data['p_anual'].values
                    grid_P = interpolacion_segura(points, values_p, grid_x, grid_y)
                    
                    # Altitud
                    values_alt = df_data['alt_est'].values
                    grid_Alt = interpolacion_segura(points, values_alt, grid_x, grid_y)
                    
                    # 3. Modelo Matemático
                    grid_T = 30 - (0.0065 * grid_Alt)
                    L_t = 300 + 25*grid_T + 0.05*(grid_T**3)
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
                        grid_R = grid_P - grid_ETR
                    grid_R = np.nan_to_num(grid_R, nan=0).clip(min=0)
                    
                    # Normalización
                    max_R = np.nanmax(grid_R)
                    norm_R = grid_R / max_R if max_R > 0 else grid_R
                    
                    raw_Bio = (grid_Alt * 0.7) + (grid_P * 0.3)
                    max_B = np.nanmax(raw_Bio)
                    norm_Bio = raw_Bio / max_B if max_B > 0 else raw_Bio
                    
                    # 4. Score Final
                    grid_Score = (norm_R * pct_agua) + (norm_Bio * pct_bio)
                    
                    # Filtro de Umbral
                    mask_score = grid_Score >= (umbral_prioridad / 100.0)
                    grid_Final = np.where(mask_score, grid_Score, np.nan)

                    # 5. Visualización
                    col_map, col_res = st.columns([3, 1])
                    
                    with col_map:
                        fig = go.Figure()
                        
                        # Mapa de Calor (Transpuesto para alinear con Plotly)
                        # Plotly contour espera x, y y z. grid_x es [100,100], grid_y es [100,100]
                        # A veces griddata devuelve shape transpuesto visualmente respecto a lat/lon
                        
                        fig.add_trace(go.Contour(
                            z=grid_Final.T, # Transponer para alinear Lat/Lon correctamente
                            x=np.linspace(minx, maxx, 100),
                            y=np.linspace(miny, maxy, 100),
                            colorscale="RdYlGn", 
                            colorbar=dict(title="Prioridad", len=0.8),
                            hoverinfo='z', name="Prioridad",
                            connectgaps=True,
                            line_smoothing=0.85,
                            contours=dict(coloring='heatmap')
                        ))
                        
                        # Contorno Zona
                        for idx, row in gdf_zona.iterrows():
                            geom = row.geometry
                            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                            for poly in polys:
                                x, y = poly.exterior.xy
                                fig.add_trace(go.Scatter(
                                    x=list(x), y=list(y), mode='lines', 
                                    line=dict(color='black', width=3), hoverinfo='skip', showlegend=False
                                ))
                        
                        # Estaciones
                        fig.add_trace(go.Scatter(
                            x=df_data['longitude'], y=df_data['latitude'],
                            mode='markers', 
                            marker=dict(color='blue', size=6, line=dict(width=1, color='white')),
                            name='Estaciones', text=df_data['nombre'], hoverinfo='text'
                        ))

                        fig.update_layout(
                            title=f"Mapa: {nombre_seleccion}",
                            height=650, margin=dict(l=0,r=0,t=40,b=0),
                            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_res:
                        st.markdown("### 📊 Cobertura")
                        valid_cells = np.count_nonzero(~np.isnan(grid_Final))
                        pct_visible = valid_cells / grid_Final.size
                        
                        st.metric("Área Analizada", f"{pct_visible:.1%}")
                        st.info(f"Estaciones: {len(df_data)}")
                        
                        st.markdown("---")
                        st.markdown("🟢 **Alta Prioridad**")
                        st.markdown("🔴 **Baja Prioridad**")

            else:
                st.warning("⚠️ Datos insuficientes (Mínimo 3 estaciones).")
                st.info("Aumenta el 'Radio Buffer' en la barra lateral.")

        else:
            st.error("CSV no encontrado.")

    except Exception as e:
        st.error(f"Error técnico: {e}")
else:
    st.info("👈 Seleccione una zona.")