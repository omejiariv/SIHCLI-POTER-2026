# Módulo de Soporte a Decisiones (CORREGIDO)

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from scipy.interpolate import griddata
import sys
import os

# --- SETUP INICIAL ---
st.set_page_config(page_title="Matriz de Decisiones", page_icon="🎯", layout="wide")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
    # Intentamos importar get_engine para no crear conexiones extra
    try:
        from modules.db_manager import get_engine
    except:
        def get_engine(): return create_engine(st.secrets["DATABASE_URL"])

except Exception as e:
    st.error(f"Error crítico de importación: {e}")
    st.stop()

st.title("🎯 Priorización de Áreas de Intervención")

# --- FUNCIONES DE CAPAS (Optimizadas) ---
@st.cache_data(ttl=3600)
def get_clipped_context_layers(gdf_zona_bounds):
    """Carga y recorta las capas pesadas UNA SOLA VEZ."""
    layers_data = {'municipios': None, 'cuencas': None, 'predios': None}
    
    minx, miny, maxx, maxy = gdf_zona_bounds
    from shapely.geometry import box
    roi_poly = box(minx, miny, maxx, maxy)
    roi_gdf = gpd.GeoDataFrame(geometry=[roi_poly], crs="EPSG:4326")
    
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Lista de archivos a cargar
    files = {
        'municipios': "MunicipiosAntioquia.geojson",
        'cuencas': "SubcuencasAinfluencia.geojson",
        'predios': "PrediosEjecutados.geojson"
    }

    for key, fname in files.items():
        try:
            fpath = os.path.join(base_dir, fname)
            if os.path.exists(fpath):
                gdf = gpd.read_file(fpath)
                if gdf.crs and gdf.crs.to_string() != "EPSG:4326": 
                    gdf = gdf.to_crs("EPSG:4326")
                layers_data[key] = gpd.clip(gdf, roi_gdf)
        except: pass

    return layers_data

# --- INTERPOLACIÓN ---
def interpolacion_segura(points, values, grid_x, grid_y):
    # Interpolación Lineal
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear')
    # Rellenar huecos (NaN) con Nearest Neighbor para evitar bordes vacíos
    mask = np.isnan(grid_z0)
    if np.any(mask):
        grid_z1 = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z0[mask] = grid_z1[mask]
    return grid_z0

# --- INTERFAZ ---
with st.expander("📘 Metodología", expanded=False):
    st.write("Análisis Multicriterio Espacial (SMCA) basado en datos en tiempo real.")

ids, nombre, alt_ref, gdf_zona = selectors.render_selector_espacial()

with st.sidebar:
    st.divider()
    st.header("⚖️ Criterios")
    w_agua = st.slider("💧 Peso: Hídrico", 0, 100, 60, 5)
    w_bio = st.slider("🍃 Peso: Ecosistémico", 0, 100, 40, 5)
    
    # Normalización de pesos
    suma = w_agua + w_bio
    pct_agua = w_agua / suma if suma > 0 else 0.5
    pct_bio = w_bio / suma if suma > 0 else 0.5
    
    st.divider()
    umbral = st.slider("Filtrar Prioridad Alta (%)", 0, 90, 0)

# --- LÓGICA PRINCIPAL ---
if gdf_zona is not None and not gdf_zona.empty:
    engine = get_engine()
    
    try:
        # A. CARGAR DATOS (CAMBIO CLAVE: Usamos DB, NO CSV)
        # Consultamos la tabla 'estaciones' directamente. Es más seguro.
        q_est = text("SELECT id_estacion, nombre, latitud, longitud, altitud FROM estaciones")
        df_est = pd.read_sql(q_est, engine)
        
        if not df_est.empty:
            # Renombrar para compatibilidad con el resto del script
            df_est = df_est.rename(columns={
                'latitud': 'latitude', 
                'longitud': 'longitude', 
                'altitud': 'alt_est'
            })
            
            # Limpieza numérica
            for c in ['latitude', 'longitude', 'alt_est']: 
                df_est[c] = pd.to_numeric(df_est[c], errors='coerce')
            
            df_est = df_est.dropna(subset=['latitude', 'longitude'])
            df_est['alt_est'] = df_est['alt_est'].fillna(alt_ref)
            
            # Buffer y Recorte Espacial
            minx, miny, maxx, maxy = gdf_zona.total_bounds
            margin = 0.05
            mask = (
                (df_est['longitude'] >= minx-margin) & (df_est['longitude'] <= maxx+margin) & 
                (df_est['latitude'] >= miny-margin) & (df_est['latitude'] <= maxy+margin)
            )
            df_filt = df_est[mask].copy()
            
            if not df_filt.empty:
                # Preparar IDs para consulta SQL
                # Usamos TRIM para asegurar que no fallen los cruces
                ids_v = df_filt['id_estacion'].astype(str).str.strip().unique()
                
                if len(ids_v) > 0:
                    # Formatear IDs para SQL: 'ID1','ID2','ID3'
                    ids_s = ",".join([f"'{x}'" for x in ids_v])
                    
                    # CONSULTA DE LLUVIA (CORREGIDA A NUEVA BD)
                    # Usamos 'valor' en lugar de 'precipitation' y 'precipitacion' en lugar de 'precipitacion_mensual'
                    q_ppt = text(f"""
                        SELECT id_estacion, AVG(valor)*12 as p_anual 
                        FROM precipitacion 
                        WHERE id_estacion IN ({ids_s}) 
                        GROUP BY id_estacion
                    """)
                    
                    df_ppt = pd.read_sql(q_ppt, engine)
                    
                    # Merge (Cruce)
                    df_filt['id_estacion'] = df_filt['id_estacion'].astype(str).str.strip()
                    df_ppt['id_estacion'] = df_ppt['id_estacion'].astype(str).str.strip()
                    
                    df_data = pd.merge(df_filt, df_ppt, on='id_estacion', how='inner')
                else:
                    df_data = pd.DataFrame()
            else:
                df_data = pd.DataFrame()

            # B. CÁLCULOS Y MAPA
            if len(df_data) >= 3:
                with st.spinner("Modelando prioridades..."):
                    # Grid
                    gx, gy = np.mgrid[minx:maxx:100j, miny:maxy:100j]
                    pts = df_data[['longitude', 'latitude']].values
                    
                    # Interpolaciones
                    grid_P = interpolacion_segura(pts, df_data['p_anual'].values, gx, gy)
                    grid_Alt = interpolacion_segura(pts, df_data['alt_est'].values, gx, gy)
                    
                    # Modelo Turc Simplificado
                    grid_T = np.maximum(5, 30 - (0.0065 * grid_Alt))
                    L_t = 300 + 25*grid_T + 0.05*(grid_T**3)
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
                    
                    grid_R = (grid_P - grid_ETR).clip(min=0)
                    
                    # Normalización (0-1) para SMCA
                    max_R = np.nanmax(grid_R)
                    norm_R = grid_R / max_R if (max_R > 0) else grid_R
                    
                    # Proxy de Biodiversidad (Altura + Lluvia)
                    raw_Bio = (grid_Alt * 0.7) + (grid_P * 0.3)
                    max_B = np.nanmax(raw_Bio)
                    norm_Bio = raw_Bio / max_B if (max_B > 0) else raw_Bio
                    
                    # Matriz Final
                    grid_Final = (norm_R * pct_agua) + (norm_Bio * pct_bio)
                    grid_Final = np.where(grid_Final >= (umbral/100.0), grid_Final, np.nan)

                    # --- VISUALIZACIÓN ---
                    c1, c2 = st.columns([3, 1])
                    
                    with c1:
                        fig = go.Figure()
                        
                        # 1. Capa Raster (Heatmap)
                        fig.add_trace(go.Contour(
                            z=grid_Final.T, x=np.linspace(minx, maxx, 100), y=np.linspace(miny, maxy, 100),
                            colorscale="RdYlGn", 
                            colorbar=dict(title="Prioridad (0-1)"),
                            hoverinfo='z', 
                            connectgaps=True, 
                            contours=dict(coloring='heatmap'), 
                            opacity=0.7
                        ))

                        # 2. Capas de Contexto
                        context_layers = get_clipped_context_layers(tuple(gdf_zona.total_bounds))
                        
                        # Función auxiliar para pintar polígonos
                        def add_poly_layer(gdf, color, name, width=1, dash=None):
                            if gdf is not None and not gdf.empty:
                                for _, row in gdf.iterrows():
                                    geom = row.geometry
                                    if geom.geom_type in ['Polygon', 'MultiPolygon']:
                                        polys = [geom] if geom.geom_type == 'Polygon' else geom.geoms
                                        for poly in polys:
                                            x, y = poly.exterior.xy
                                            line_dict = dict(color=color, width=width)
                                            if dash: line_dict['dash'] = dash
                                            fig.add_trace(go.Scatter(
                                                x=list(x), y=list(y), 
                                                mode='lines', 
                                                line=line_dict, 
                                                hoverinfo='skip', 
                                                name=name, 
                                                showlegend=False
                                            ))

                        add_poly_layer(context_layers['municipios'], 'gray', "Municipio", 0.5)
                        add_poly_layer(context_layers['cuencas'], 'blue', "Cuenca", 0.5, 'dot')
                        add_poly_layer(context_layers['predios'], 'orange', "Predio", 1.5)

                        # 3. Límite de Zona Seleccionada
                        add_poly_layer(gdf_zona, 'black', "Zona", 2)

                        fig.update_layout(
                            height=650, 
                            margin=dict(l=0,r=0,t=20,b=0), 
                            xaxis=dict(visible=False), 
                            yaxis=dict(visible=False, scaleanchor="x"), 
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with c2:
                        valid_cells = np.count_nonzero(~np.isnan(grid_Final))
                        pct_cobertura = valid_cells / grid_Final.size
                        st.metric("Cobertura de Análisis", f"{pct_cobertura:.1%}")
                        st.info(f"Estaciones usadas: {len(df_data)}")
                        with st.expander("Ver Datos Base"):
                            st.dataframe(df_data[['id_estacion', 'nombre', 'p_anual', 'alt_est']])

            else:
                st.warning(f"⚠️ Datos insuficientes en esta zona. Se encontraron {len(df_data)} estaciones con datos (mínimo 3 requeridas para interpolar).")
                st.write("Intenta seleccionar una Cuenca más grande o aumentar el área.")
        else:
            st.error("No se encontraron estaciones en la Base de Datos.")
            
    except Exception as e:
        st.error(f"Error técnico: {e}")
else:
    st.info("👈 Seleccione una zona en el menú lateral.")