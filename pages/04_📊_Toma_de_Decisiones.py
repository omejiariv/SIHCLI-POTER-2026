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

# --- FUNCIONES AUXILIARES DE CAPAS ---
@st.cache_data(ttl=3600)
def load_geojson_cached(filename):
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
    if os.path.exists(filepath):
        try:
            gdf = gpd.read_file(filepath)
            if gdf.crs and gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
            return gdf
        except: pass
    return None

def add_context_layers(fig, gdf_zona):
    """Añade capas de contexto recortadas a la zona."""
    # Buffer seguro proyectando a metros (EPSG:3116 Colombia Bogota Zone) y volviendo a WGS84
    try:
        roi = gdf_zona.to_crs("EPSG:3116").buffer(5000).to_crs("EPSG:4326") # 5km buffer
    except:
        roi = gdf_zona.buffer(0.05) # Fallback geográfico

    # 1. MUNICIPIOS
    gdf = load_geojson_cached("MunicipiosAntioquia.geojson")
    if gdf is not None:
        try:
            gdf_clip = gpd.clip(gdf, roi)
            for _, row in gdf_clip.iterrows():
                # Intentar varios nombres de columna comunes
                name = row.get('MPIO_CNMBR', row.get('NOMBRE', row.get('nombre', 'Mpio')))
                geom = row.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                for poly in polys:
                    x, y = poly.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', line=dict(color='rgba(150,150,150,0.5)', width=0.8),
                        hoverinfo='text', text=f"🏛️ {name}", showlegend=False
                    ))
        except: pass

    # 2. CUENCAS
    gdf = load_geojson_cached("SubcuencasAinfluencia.geojson")
    if gdf is not None:
        try:
            gdf_clip = gpd.clip(gdf, roi)
            for _, row in gdf_clip.iterrows():
                name = row.get('N-NSS3', row.get('SUBC_LBL', row.get('NOM_CUENCA', 'Cuenca')))
                geom = row.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                for poly in polys:
                    x, y = poly.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', line=dict(color='rgba(0,100,255,0.3)', width=1, dash='dot'),
                        hoverinfo='text', text=f"💧 {name}", showlegend=False
                    ))
        except: pass

    # 3. PREDIOS
    gdf = load_geojson_cached("PrediosEjecutados.geojson")
    if gdf is not None:
        try:
            gdf_clip = gpd.clip(gdf, gdf_zona) # Clip estricto
            for _, row in gdf_clip.iterrows():
                geom = row.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                for poly in polys:
                    x, y = poly.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', fill='toself', fillcolor='rgba(255,165,0,0.4)',
                        line=dict(color='orange', width=1.5),
                        hoverinfo='text', text="🏡 Predio Ejecutado", name='Predios'
                    ))
        except: pass

# --- INTERPOLACIÓN ---
def interpolacion_segura(points, values, grid_x, grid_y):
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear')
    mask = np.isnan(grid_z0)
    if np.any(mask):
        grid_z1 = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z0[mask] = grid_z1[mask]
    return grid_z0

# --- INTERFAZ PRINCIPAL ---
with st.expander("📘 Metodología", expanded=False):
    st.write("Análisis Multicriterio Espacial (SMCA) cruzando Oferta Hídrica y Valor Ecosistémico.")

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
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mapaCVENSO.csv')
        if os.path.exists(csv_path):
            df_est = pd.read_csv(csv_path, sep=';', decimal=',', encoding='latin-1', usecols=['Id_estacio', 'Longitud_geo', 'Latitud_geo', 'alt_est', 'Nom_Est'])
            df_est.rename(columns={'Id_estacio': 'id_estacion', 'Longitud_geo': 'longitude', 'Latitud_geo': 'latitude', 'Nom_Est': 'nombre'}, inplace=True)
            for c in ['latitude', 'longitude', 'alt_est']: df_est[c] = pd.to_numeric(df_est[c], errors='coerce')
            df_est.dropna(subset=['latitude', 'longitude'], inplace=True)
            df_est['alt_est'] = df_est['alt_est'].fillna(alt_ref)
            
            # Buffer seguro para filtro
            roi_buffer = gdf_zona.to_crs("EPSG:3116").buffer(10000).to_crs("EPSG:4326").total_bounds
            mask = ((df_est['longitude']>=roi_buffer[0]) & (df_est['longitude']<=roi_buffer[2]) & (df_est['latitude']>=roi_buffer[1]) & (df_est['latitude']<=roi_buffer[3]))
            df_filt = df_est[mask].copy()
            
            if not df_filt.empty:
                ids_v = df_filt['id_estacion'].unique()
                ids_s = ",".join([f"'{str(x)}'" for x in ids_v])
                if ids_s:
                    df_ppt = pd.read_sql(f"SELECT id_estacion_fk as id_estacion, AVG(precipitation)*12 as p_anual FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY id_estacion_fk", engine)
                    df_filt['id_estacion'] = df_filt['id_estacion'].astype(str)
                    df_ppt['id_estacion'] = df_ppt['id_estacion'].astype(str)
                    df_data = pd.merge(df_filt, df_ppt, on='id_estacion', how='inner')
                else: df_data = pd.DataFrame()
            else: df_data = pd.DataFrame()

            if len(df_data) >= 3:
                with st.spinner("Procesando mapa..."):
                    gx, gy = np.mgrid[roi_buffer[0]:roi_buffer[2]:100j, roi_buffer[1]:roi_buffer[3]:100j]
                    pts = df_data[['longitude', 'latitude']].values
                    
                    grid_P = interpolacion_segura(pts, df_data['p_anual'].values, gx, gy)
                    grid_Alt = interpolacion_segura(pts, df_data['alt_est'].values, gx, gy)
                    
                    grid_T = 30 - (0.0065 * grid_Alt)
                    L_t = 300 + 25*grid_T + 0.05*(grid_T**3)
                    grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
                    grid_R = (grid_P - grid_ETR).clip(min=0)
                    
                    max_R = np.nanmax(grid_R)
                    norm_R = grid_R / max_R if max_R > 0 else grid_R
                    raw_Bio = (grid_Alt * 0.7) + (grid_P * 0.3)
                    max_B = np.nanmax(raw_Bio)
                    norm_Bio = raw_Bio / max_B if max_B > 0 else raw_Bio
                    
                    grid_Final = (norm_R * pct_agua) + (norm_Bio * pct_bio)
                    grid_Final = np.where(grid_Final >= (umbral/100.0), grid_Final, np.nan)

                    c1, c2 = st.columns([3, 1])
                    with c1:
                        fig = go.Figure()
                        # Mapa Calor
                        fig.add_trace(go.Contour(
                            z=grid_Final.T, x=np.linspace(roi_buffer[0], roi_buffer[2], 100), y=np.linspace(roi_buffer[1], roi_buffer[3], 100),
                            colorscale="RdYlGn", colorbar=dict(title="Prioridad"),
                            hoverinfo='z', connectgaps=True, contours=dict(coloring='heatmap'), opacity=0.7
                        ))
                        
                        # CAPAS CONTEXTO
                        add_context_layers(fig, gdf_zona)
                        
                        # Zona Selección
                        for _, row in gdf_zona.iterrows():
                            g = row.geometry
                            ps = [g] if g.geom_type == 'Polygon' else list(g.geoms)
                            for p in ps:
                                x,y = p.exterior.xy
                                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='black', width=3), hoverinfo='skip', showlegend=False))
                        
                        fig.update_layout(
                            height=700, margin=dict(l=0,r=0,t=20,b=0),
                            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                            plot_bgcolor='white'
                        )
                        # CORRECCIÓN DEPRECACIÓN: Usamos width='stretch' implícito o explícito si es necesario
                        st.plotly_chart(fig, use_container_width=True) # En versiones 2026 debería funcionar o usar width="stretch" si da error fatal
            
            else: st.warning("Datos insuficientes.")
        else: st.error("CSV no encontrado.")
    except Exception as e: st.error(f"Error: {e}")
else: st.info("👈 Seleccione una zona.")