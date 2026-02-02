# modules/selectors.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from modules import db_manager
from modules.config import Config

def render_selector_espacial():
    """
    Selector espacial con opción manual de columna de nombres.
    """
    engine = db_manager.get_engine()
    
    st.sidebar.header("📍 Filtros Geográficos")
    
    # 1. MODO DE AGREGACIÓN
    modo = st.sidebar.radio(
        "Nivel de Agregación:",
        ["Por Cuenca", "Por Municipio", "Departamento (Antioquia)"],
        index=0
    )
    
    gdf_zona = None
    nombre_zona = "Antioquia"
    altitud_ref = 1500
    
    try:
        # --- A. POR CUENCA ---
        if modo == "Por Cuenca":
            try:
                gdf_cuencas = gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry")
                
                # --- RECUPERADO: Selector de Columna de Nombres ---
                cols_texto = [c for c in gdf_cuencas.columns if c not in ['geometry', 'gid', 'objectid']]
                default_idx = 0
                if 'subc_lbl' in cols_texto: default_idx = cols_texto.index('subc_lbl')
                elif 'nombre' in cols_texto: default_idx = cols_texto.index('nombre')
                
                col_nom = st.sidebar.selectbox("📂 Columna de Nombres:", cols_texto, index=default_idx)
                
                if col_nom:
                    lista = sorted(gdf_cuencas[col_nom].astype(str).unique().tolist())
                    sel = st.sidebar.selectbox("Seleccione Cuenca:", lista)
                    if sel:
                        nombre_zona = sel
                        gdf_zona = gdf_cuencas[gdf_cuencas[col_nom] == sel]

            except Exception as e:
                st.sidebar.warning(f"Error cargando cuencas: {e}")
                return [], "", 0, None

        # --- B. POR MUNICIPIO ---
        elif modo == "Por Municipio":
            try:
                gdf_mun = gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry")
                
                cols_texto = [c for c in gdf_mun.columns if c not in ['geometry', 'gid']]
                default_idx = 0
                if 'mpio_cnmbr' in cols_texto: default_idx = cols_texto.index('mpio_cnmbr')
                
                col_nom = st.sidebar.selectbox("📂 Columna de Nombres:", cols_texto, index=default_idx)
                
                if col_nom:
                    lista = sorted(gdf_mun[col_nom].astype(str).unique().tolist())
                    sel = st.sidebar.selectbox("Seleccione Municipio:", lista)
                    if sel:
                        nombre_zona = sel
                        gdf_mun = gdf_mun.to_crs("EPSG:4326")
                        gdf_zona = gdf_mun[gdf_mun[col_nom] == sel]
            except:
                st.sidebar.warning("Error en tabla municipios.")

        # --- C. DEPARTAMENTO ---
        else:
            from shapely.geometry import box
            gdf_zona = gpd.GeoDataFrame(
                {'nombre': ['Antioquia']}, 
                geometry=[box(-77.5, 5.0, -73.5, 9.0)], 
                crs="EPSG:4326"
            )

        # --- 2. FILTRAR ESTACIONES ---
        ids_estaciones = []
        if gdf_zona is not None and not gdf_zona.empty:
            
            # Auto-corrección de CRS (Metros a Grados)
            if gdf_zona.crs and gdf_zona.crs.to_string() != "EPSG:4326":
                 gdf_zona = gdf_zona.to_crs("EPSG:4326")
            
            # Slider de Buffer
            buff_km = st.sidebar.slider("Radio Buffer (km):", 0, 50, 5)
            buff_deg = buff_km / 111.0 
            
            minx, miny, maxx, maxy = gdf_zona.total_bounds
            
            # Consulta SQL optimizada
            q_est = text(f"""
                SELECT id_estacion, nombre, latitud, longitud, altitud 
                FROM estaciones 
                WHERE longitud BETWEEN {minx - buff_deg} AND {maxx + buff_deg} 
                AND latitud BETWEEN {miny - buff_deg} AND {maxy + buff_deg}
            """)
            
            df_est = pd.read_sql(q_est, engine)
            
            if not df_est.empty:
                gdf_ptos = gpd.GeoDataFrame(
                    df_est, 
                    geometry=gpd.points_from_xy(df_est.longitud, df_est.latitud), 
                    crs="EPSG:4326"
                )
                
                # Spatial Join
                zona_buffered = gdf_zona.copy()
                if buff_deg > 0:
                    zona_buffered['geometry'] = zona_buffered.geometry.buffer(buff_deg)
                
                est_in = gdf_ptos[gdf_ptos.geometry.within(zona_buffered.unary_union)]
                
                if not est_in.empty:
                    ids_estaciones = est_in['id_estacion'].astype(str).str.strip().tolist()
                    altitud_ref = est_in['altitud'].mean()
                    st.sidebar.success(f"📍 Estaciones encontradas: {len(ids_estaciones)}")
                else:
                    st.sidebar.warning("0 estaciones en el área exacta.")
            else:
                st.sidebar.warning("0 estaciones en el cuadrante.")

    except Exception as e:
        st.sidebar.error(f"Error selector: {e}")
        
    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona