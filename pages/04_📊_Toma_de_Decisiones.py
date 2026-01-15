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

# --- FUNCIONES AUXILIARES (NUEVO) ---
@st.cache_data(ttl=3600)
def load_geojson_cached(filename):
    """Carga rápida de GeoJSON."""
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
    if os.path.exists(filepath):
        try:
            gdf = gpd.read_file(filepath)
            if gdf.crs and gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
            return gdf
        except: pass
    return None

def add_context_layers(fig, gdf_zona, bounds_buffer=0.05):
    """Añade capas de contexto (Munis, Cuencas, Predios) con Hover."""
    
    # Definir área de interés ampliada para el recorte
    roi = gdf_zona.buffer(bounds_buffer)

    # 1. MUNICIPIOS
    gdf_muni = load_geojson_cached("MunicipiosAntioquia.geojson")
    if gdf_muni is not None:
        try:
            gdf_clip = gpd.clip(gdf_muni, roi)
            name_col = next((c for c in ['MPIO_CNMBR', 'NOMBRE', 'nombre'] if c in gdf_clip.columns), None)
            for _, row in gdf_clip.iterrows():
                hover_txt = f"Mpio: {row[name_col]}" if name_col else "Municipio"
                geom = row.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                for poly in polys:
                    x, y = poly.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', line=dict(color='rgba(100,100,100,0.5)', width=1),
                        hoverinfo='text', text=hover_txt, showlegend=False
                    ))
        except: pass

    # 2. CUENCAS (Subcuencas)
    gdf_cuenca = load_geojson_cached("SubcuencasAinfluencia.geojson")
    if gdf_cuenca is not None:
        try:
            gdf_clip = gpd.clip(gdf_cuenca, roi)
            name_col = next((c for c in ['N-NSS3', 'SUBC_LBL', 'NOM_CUENCA'] if c in gdf_clip.columns), None)
            for _, row in gdf_clip.iterrows():
                hover_txt = f"Cuenca: {row[name_col]}" if name_col else "Cuenca"
                geom = row.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                for poly in polys:
                    x, y = poly.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', line=dict(color='rgba(0,0,200,0.4)', width=1, dash='dot'),
                        hoverinfo='text', text=hover_txt, showlegend=False
                    ))
        except: pass
        
    # 3. PREDIOS (Ejecutados)
    gdf_predios = load_geojson_cached("PrediosEjecutados.geojson")
    if gdf_predios is not None:
        try:
            # Clip más estricto para predios
            gdf_clip = gpd.clip(gdf_predios, gdf_zona.buffer(0.01))
            for _, row in gdf_clip.iterrows():
                geom = row.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                for poly in polys:
                    x, y = poly.exterior.xy
                    # Relleno sutil para los predios
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', fill='toself', fillcolor='rgba(255, 165, 0, 0.3)',
                        line=dict(color='orange', width=1.5),
                        hoverinfo='text', text="Predio Ejecutado", name='Predios'
                    ))
        except: pass

# --- DOCUMENTACIÓN ---
with st.expander("📘 Documentación Técnica: Metodología y Fuentes", expanded=False):
    st.markdown("""
    * **Metodología:** Análisis Multicriterio Espacial (SMCA) mediante superposición ponderada.
    * **Variables:** Oferta Hídrica (Balance P-ETR Turc) y Valor Ecosistémico (Gradiente Altitudinal + Humedad).
    * **Interpolación:** Método híbrido (Linear + Nearest Neighbor) para generar superficies continuas sin vacíos.
    * **Fuentes:** Climatología SIHCLI (IDEAM/EPM), Cartografía IGAC/Gobernación, Catálogo de Estaciones propio.
    """)
st.divider()

# --- 1. SELECTOR ---
ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

# --- 2. PONDERACIÓN ---
with st.sidebar:
    st.divider()
    st.header("⚖️ Criterios")
    w_agua = st.slider("💧 Peso: Hídrico", 0, 100, 60, 5)
    w_bio = st.slider("🍃 Peso: Ecosistémico", 0, 100, 40, 5)
    total = w_agua + w_bio
    pct_agua = w_agua / (total if total > 0 else 1)
    pct_bio = w_bio / (total if total > 0 else 1)
    st.caption(f"Agua {pct_agua:.0%} | Bio {pct_bio:.0%}")
    st.divider()
    umbral_prioridad = st.slider("Filtrar Prioridad Alta (%)", 0, 90, 0)

# --- FUNCIÓN INTERPOLACIÓN ---
def interpolacion_segura(points, values, grid_x, grid_y):
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear')
    mask = np.isnan(grid_z0)
    if np.any(mask):
        grid_z1 = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z0[mask] = grid_z1[mask]
    return grid_z0

# --- 3. MOTOR DE ANÁLISIS ---
if gdf_zona is not None and not gdf_zona.empty:
    engine = create_engine(st.secrets["DATABASE_URL"])
    try:
        # A. CARGAR CATÁLOGO CSV
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mapaCVENSO.csv')
        if os.path.exists(csv_path):
            df_estaciones_all = pd.read_csv(csv_path, sep=';', decimal=',', encoding='latin-1', usecols=['Id_estacio', 'Longitud_geo', 'Latitud_geo', 'alt_est', 'Nom_Est'])
            df_estaciones_all.rename(columns={'Id_estacio': 'id_estacion', 'Longitud_geo': 'longitude', 'Latitud_geo': 'latitude', 'Nom_Est': 'nombre'}, inplace=True)
            for col in ['latitude', 'longitude', 'alt_est']: df_estaciones_all[col] = pd.to_numeric(df_estaciones_all[col], errors='coerce')
            df_estaciones_all.dropna(subset=['latitude', 'longitude'], inplace=True)
            df_estaciones_all['alt_est'] = df_estaciones_all['alt_est'].fillna(altitud_ref)

            # B. FILTRO ESPACIAL
            minx, miny, maxx, maxy = gdf_zona.buffer(0.05).total_bounds
            mask = ((df_estaciones_all['longitude'] >= minx) & (df_estaciones_all['longitude'] <= maxx) & (df_estaciones_all['latitude'] >= miny) & (df_estaciones_all['latitude'] <= maxy))
            df_filtrada = df_estaciones_all[mask].copy()
            
            # C. DATOS PRECIPITACIÓN SQL
            if not df_filtrada.empty:
                ids_validos = df_filtrada['id_estacion'].unique()
                ids_sql = ",".join([f"'{str(x)}'" for x in ids_validos])
                if ids_sql:
                    q_ppt = f"SELECT id_estacion_fk as id_estacion, AVG(precipitation) * 12 as p_anual FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_sql}) GROUP BY id_estacion_fk"
                    df_ppt = pd.read_sql(q_ppt, engine)
                    df_filtrada['id_estacion'] = df_filtrada['id_estacion'].astype(str)
                    df_ppt['id_estacion'] = df_ppt['id_estacion'].astype(str)
                    df_data = pd.merge(df_filtrada, df_ppt, on='id_estacion', how='inner')
                else: df_data = pd.DataFrame()
            else: df_data = pd.DataFrame()

            # --- D. CÁLCULOS ---
            if len(df_data) >= 3:
                with st.spinner("Generando mapa de prioridades..."):
                    gx, gy = np.mgrid[minx:maxx:120j, miny:maxy:120j] # Mayor resolución
                    points = df_data[['longitude', 'latitude']].values
                    grid_P = interpolacion_segura(points, df_data['p_anual'].values, gx, gy)
                    grid_Alt = interpolacion_segura(points, df_data['alt_est'].values, gx, gy)
                    
                    # Modelo
                    grid_T = 30 - (0.0065 * grid_Alt)
                    L_t = 300 + 25*grid_T + 0.05*(grid_T**3)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
                        grid_R = grid_P - grid_ETR
                    grid_R = np.nan_to_num(grid_R, nan=0).clip(min=0)
                    max_R = np.nanmax(grid_R)
                    norm_R = grid_R / max_R if max_R > 0 else grid_R
                    raw_Bio = (grid_Alt * 0.7) + (grid_P * 0.3)
                    max_B = np.nanmax(raw_Bio)
                    norm_Bio = raw_Bio / max_B if max_B > 0 else raw_Bio
                    grid_Score = (norm_R * pct_agua) + (norm_Bio * pct_bio)
                    mask_score = grid_Score >= (umbral_prioridad / 100.0)
                    grid_Final = np.where(mask_score, grid_Score, np.nan)

                    # VISUALIZACIÓN
                    col_map, col_res = st.columns([3, 1])
                    with col_map:
                        fig = go.Figure()
                        # 1. Mapa de Calor
                        fig.add_trace(go.Contour(
                            z=grid_Final.T, x=np.linspace(minx, maxx, 120), y=np.linspace(miny, maxy, 120),
                            colorscale="RdYlGn", colorbar=dict(title="Prioridad", len=0.7),
                            hoverinfo='z', name="Prioridad", connectgaps=True, contours=dict(coloring='heatmap'), opacity=0.8
                        ))
                        
                        # 2. CAPAS DE CONTEXTO (NUEVO: Munis, Cuencas, Predios con Hover)
                        add_context_layers(fig, gdf_zona, bounds_buffer=0.05)

                        # 3. Límite Zona Seleccionada (Negro fuerte)
                        for idx, row in gdf_zona.iterrows():
                            geom = row.geometry
                            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                            for poly in polys:
                                x, y = poly.exterior.xy
                                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='black', width=2.5), hoverinfo='skip', showlegend=False))
                        
                        # 4. Estaciones
                        fig.add_trace(go.Scatter(x=df_data['longitude'], y=df_data['latitude'], mode='markers', marker=dict(color='blue', size=5, line=dict(width=0.5, color='white')), name='Estaciones', text=df_data['nombre'], hoverinfo='text'))
                        
                        fig.update_layout(
                            title=f"Mapa: {nombre_seleccion}", height=700, margin=dict(l=0,r=0,t=40,b=0),
                            xaxis=dict(visible=False, showgrid=False), yaxis=dict(visible=False, showgrid=False, scaleanchor="x"),
                            plot_bgcolor='white', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.6)")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_res:
                        st.subheader("Resultados")
                        valid = np.count_nonzero(~np.isnan(grid_Final))
                        st.metric("Cobertura Análisis", f"{valid/grid_Final.size:.1%}")
                        st.caption(f"Basado en {len(df_data)} estaciones.")
                        st.success("Pasa el mouse sobre el mapa para ver detalles de municipios, cuencas y predios.")

            else: st.warning("⚠️ Datos insuficientes (mínimo 3 estaciones). Aumenta el Radio Buffer.")
        else: st.error("CSV no encontrado.")
    except Exception as e: st.error(f"Error técnico: {e}")
else: st.info("👈 Seleccione una zona.")