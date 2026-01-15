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

# --- SETUP INICIAL ---
st.set_page_config(page_title="Matriz de Decisiones", page_icon="🎯", layout="wide")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
except Exception as e:
    st.error(f"Error crítico de importación: {e}")
    st.stop()

st.title("🎯 Priorización de Áreas de Intervención")

# --- FUNCIONES DE CAPAS (Optimizadas) ---
@st.cache_data(ttl=3600)
def get_clipped_context_layers(gdf_zona_bounds):
    """
    Carga y recorta las capas pesadas UNA SOLA VEZ por zona seleccionada.
    Usamos bounds (tupla) como key del caché porque los dataframes no son hashables fácilmente.
    """
    layers_data = {'municipios': None, 'cuencas': None, 'predios': None}
    
    # Reconstruir zona desde bounds para el clip (aproximado pero rápido)
    minx, miny, maxx, maxy = gdf_zona_bounds
    from shapely.geometry import box
    roi_poly = box(minx, miny, maxx, maxy)
    roi_gdf = gpd.GeoDataFrame(geometry=[roi_poly], crs="EPSG:4326")
    
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Cargar Municipios
    try:
        muni_path = os.path.join(base_dir, "MunicipiosAntioquia.geojson")
        if os.path.exists(muni_path):
            gdf = gpd.read_file(muni_path)
            if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
            layers_data['municipios'] = gpd.clip(gdf, roi_gdf)
    except: pass

    # Cargar Cuencas
    try:
        cuenca_path = os.path.join(base_dir, "SubcuencasAinfluencia.geojson")
        if os.path.exists(cuenca_path):
            gdf = gpd.read_file(cuenca_path)
            if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
            layers_data['cuencas'] = gpd.clip(gdf, roi_gdf)
    except: pass
    
    # Cargar Predios
    try:
        predios_path = os.path.join(base_dir, "PrediosEjecutados.geojson")
        if os.path.exists(predios_path):
            gdf = gpd.read_file(predios_path)
            if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
            layers_data['predios'] = gpd.clip(gdf, roi_gdf)
    except: pass

    return layers_data

# --- INTERPOLACIÓN ---
def interpolacion_segura(points, values, grid_x, grid_y):
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear')
    mask = np.isnan(grid_z0)
    if np.any(mask):
        grid_z1 = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z0[mask] = grid_z1[mask]
    return grid_z0

# --- INTERFAZ ---
with st.expander("📘 Metodología", expanded=False):
    st.write("Análisis Multicriterio Espacial (SMCA).")

ids, nombre, alt_ref, gdf_zona = selectors.render_selector_espacial()

with st.sidebar:
    st.divider()
    st.header("⚖️ Criterios")
    w_agua = st.slider("💧 Peso: Hídrico", 0, 100, 60, 5)
    w_bio = st.slider("🍃 Peso: Ecosistémico", 0, 100, 40, 5)
    pct_agua = w_agua / (w_agua + w_bio if w_agua + w_bio > 0 else 1)
    pct_bio = w_bio / (w_agua + w_bio if w_agua + w_bio > 0 else 1)
    st.divider()
    umbral = st.slider("Filtrar Prioridad Alta (%)", 0, 90, 0)

if gdf_zona is not None and not gdf_zona.empty:
    engine = create_engine(st.secrets["DATABASE_URL"])
    try:
        # A. Cargar Datos Estaciones
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mapaCVENSO.csv')
        if os.path.exists(csv_path):
            df_est = pd.read_csv(csv_path, sep=';', decimal=',', encoding='latin-1', usecols=['Id_estacio', 'Longitud_geo', 'Latitud_geo', 'alt_est', 'Nom_Est'])
            df_est.rename(columns={'Id_estacio': 'id_estacion', 'Longitud_geo': 'longitude', 'Latitud_geo': 'latitude', 'Nom_Est': 'nombre'}, inplace=True)
            for c in ['latitude', 'longitude', 'alt_est']: df_est[c] = pd.to_numeric(df_est[c], errors='coerce')
            df_est.dropna(subset=['latitude', 'longitude'], inplace=True)
            df_est['alt_est'] = df_est['alt_est'].fillna(alt_ref)
            
            # Buffer de zona
            minx, miny, maxx, maxy = gdf_zona.total_bounds
            # Pequeño margen para atrapar estaciones borde
            margin = 0.05
            mask = ((df_est['longitude']>=minx-margin) & (df_est['longitude']<=maxx+margin) & 
                    (df_est['latitude']>=miny-margin) & (df_est['latitude']<=maxy+margin))
            df_filt = df_est[mask].copy()
            
            if not df_filt.empty:
                ids_v = df_filt['id_estacion'].unique()
                ids_s = ",".join([f"'{str(x)}'" for x in ids_v])
                
                if ids_s:
                    q_ppt = f"SELECT id_estacion_fk as id_estacion, AVG(precipitation)*12 as p_anual FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY id_estacion_fk"
                    df_ppt = pd.read_sql(q_ppt, engine)
                    df_filt['id_estacion'] = df_filt['id_estacion'].astype(str)
                    df_ppt['id_estacion'] = df_ppt['id_estacion'].astype(str)
                    df_data = pd.merge(df_filt, df_ppt, on='id_estacion', how='inner')
                else: df_data = pd.DataFrame()
            else: df_data = pd.DataFrame()

            # B. Cálculos y Mapa
            if len(df_data) >= 3:
                with st.spinner("Procesando territorio..."):
                    # Grid
                    gx, gy = np.mgrid[minx:maxx:100j, miny:maxy:100j]
                    pts = df_data[['longitude', 'latitude']].values
                    
                    grid_P = interpolacion_segura(pts, df_data['p_anual'].values, gx, gy)
                    grid_Alt = interpolacion_segura(pts, df_data['alt_est'].values, gx, gy)
                    
                    # Modelo Turc Simplificado Vectorizado
                    grid_T = 30 - (0.0065 * grid_Alt)
                    L_t = 300 + 25*grid_T + 0.05*(grid_T**3)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
                    grid_R = (grid_P - grid_ETR).clip(min=0)
                    
                    # Normalización
                    max_R = np.nanmax(grid_R)
                    norm_R = grid_R / max_R if max_R > 0 else grid_R
                    raw_Bio = (grid_Alt * 0.7) + (grid_P * 0.3)
                    max_B = np.nanmax(raw_Bio)
                    norm_Bio = raw_Bio / max_B if max_B > 0 else raw_Bio
                    
                    grid_Final = (norm_R * pct_agua) + (norm_Bio * pct_bio)
                    grid_Final = np.where(grid_Final >= (umbral/100.0), grid_Final, np.nan)

                    # Graficar
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        fig = go.Figure()
                        
                        # 1. Mapa Calor
                        fig.add_trace(go.Contour(
                            z=grid_Final.T, x=np.linspace(minx, maxx, 100), y=np.linspace(miny, maxy, 100),
                            colorscale="RdYlGn", colorbar=dict(title="Prioridad"),
                            hoverinfo='z', connectgaps=True, contours=dict(coloring='heatmap'), opacity=0.7
                        ))

                        # 2. Contexto (Cargado seguro)
                        context_layers = get_clipped_context_layers(tuple(gdf_zona.total_bounds))
                        
                        if context_layers['municipios'] is not None:
                            for _, row in context_layers['municipios'].iterrows():
                                if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                                    x, y = row.geometry.exterior.xy if row.geometry.geom_type == 'Polygon' else row.geometry.geoms[0].exterior.xy
                                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='gray', width=0.5), hoverinfo='text', text="Municipio", showlegend=False))

                        if context_layers['cuencas'] is not None:
                             for _, row in context_layers['cuencas'].iterrows():
                                if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                                    x, y = row.geometry.exterior.xy if row.geometry.geom_type == 'Polygon' else row.geometry.geoms[0].exterior.xy
                                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='blue', width=0.5, dash='dot'), hoverinfo='text', text="Cuenca", showlegend=False))
                        
                        if context_layers['predios'] is not None:
                             for _, row in context_layers['predios'].iterrows():
                                if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
                                    x, y = row.geometry.exterior.xy if row.geometry.geom_type == 'Polygon' else row.geometry.geoms[0].exterior.xy
                                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='orange', width=1.5), hoverinfo='text', text="Predio", showlegend=False))

                        # 3. Límite Zona
                        for _, row in gdf_zona.iterrows():
                            geom = row.geometry
                            if geom.geom_type == 'Polygon':
                                x, y = geom.exterior.xy
                                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='black', width=2), hoverinfo='skip', showlegend=False))
                            elif geom.geom_type == 'MultiPolygon':
                                for poly in geom.geoms:
                                    x, y = poly.exterior.xy
                                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='black', width=2), hoverinfo='skip', showlegend=False))

                        fig.update_layout(height=650, margin=dict(l=0,r=0,t=20,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"), plot_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with c2:
                         st.metric("Cobertura", f"{np.count_nonzero(~np.isnan(grid_Final))/grid_Final.size:.1%}")

            else: st.warning("Datos insuficientes.")
        else: st.error("CSV no encontrado.")
    except Exception as e: st.error(f"Error: {e}")
else: st.info("👈 Seleccione una zona.")