# modules/selectors.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from modules import db_manager
from modules.config import Config

def render_selector_espacial():
    """
    Selector espacial UNIFICADO y conectado a Base de Datos.
    Retorna: ids_estaciones, nombre_zona, altitud_ref, gdf_zona_seleccionada
    """
    engine = db_manager.get_engine()
    
    st.sidebar.header("📍 Filtros Geográficos")
    
    # --- 1. MODO DE AGREGACIÓN ---
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
            # Intentar leer geom o geometry
            try:
                gdf_cuencas = gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry")
            except:
                st.sidebar.warning("Tabla 'cuencas' no tiene columna geométrica estándar.")
                return [], "", 0, None

            # Detectar columna de nombre
            col_nom = next((c for c in gdf_cuencas.columns if c in ['nombre', 'nombre_cuenca', 'subc_lbl']), None)
            
            if col_nom:
                lista = sorted(gdf_cuencas[col_nom].astype(str).unique().tolist())
                sel = st.sidebar.selectbox("Seleccione Cuenca:", lista)
                if sel:
                    nombre_zona = sel
                    gdf_zona = gdf_cuencas[gdf_cuencas[col_nom] == sel]
            else:
                st.sidebar.error("No se encontró columna de nombre en cuencas.")

        # --- B. POR MUNICIPIO ---
        elif modo == "Por Municipio":
            try:
                gdf_mun = gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry")
            except:
                st.sidebar.warning("Tabla 'municipios' con problemas.")
                return [], "", 0, None

            col_nom = next((c for c in gdf_mun.columns if c in ['nombre', 'nombre_municipio', 'mpio_cnmbr']), None)
            
            if col_nom:
                lista = sorted(gdf_mun[col_nom].astype(str).unique().tolist())
                sel = st.sidebar.selectbox("Seleccione Municipio:", lista)
                if sel:
                    nombre_zona = sel
                    gdf_zona = gdf_mun[gdf_mun[col_nom] == sel]

        # --- C. DEPARTAMENTO ---
        else:
            # Crear un cuadro delimitador para Antioquia si no hay shape
            from shapely.geometry import box
            gdf_zona = gpd.GeoDataFrame(
                {'nombre': ['Antioquia']}, 
                geometry=[box(-77.5, 5.0, -73.5, 9.0)], 
                crs="EPSG:4326"
            )

        # --- 2. FILTRAR ESTACIONES (CRÍTICO: NOMBRES NUEVOS) ---
        ids_estaciones = []
        if gdf_zona is not None and not gdf_zona.empty:
            # Asegurar CRS
            if gdf_zona.crs and gdf_zona.crs.to_string() != "EPSG:4326":
                gdf_zona = gdf_zona.to_crs("EPSG:4326")
            
            minx, miny, maxx, maxy = gdf_zona.total_bounds
            
            # Margen de seguridad (buffer visual)
            buff = 0.05 
            
            # CONSULTA BLINDADA (Usa latitud/longitud de la BD nueva)
            q_est = text(f"""
                SELECT id_estacion, nombre, latitud, longitud, altitud 
                FROM estaciones 
                WHERE longitud BETWEEN {minx - buff} AND {maxx + buff} 
                AND latitud BETWEEN {miny - buff} AND {maxy + buff}
            """)
            
            df_est = pd.read_sql(q_est, engine)
            
            if not df_est.empty:
                # Convertir a GeoDataFrame
                gdf_ptos = gpd.GeoDataFrame(
                    df_est, 
                    geometry=gpd.points_from_xy(df_est.longitud, df_est.latitud), 
                    crs="EPSG:4326"
                )
                
                # Spatial Join: Quedarse solo con las que están DENTRO del polígono
                est_in = gdf_ptos[gdf_ptos.geometry.within(gdf_zona.unary_union)]
                
                if not est_in.empty:
                    ids_estaciones = est_in['id_estacion'].astype(str).str.strip().tolist()
                    altitud_ref = est_in['altitud'].mean()
                
                st.sidebar.success(f"📍 Estaciones en zona: {len(ids_estaciones)}")
            else:
                st.sidebar.warning("No hay estaciones en este recuadro geográfico.")
                
    except Exception as e:
        st.sidebar.error(f"Error en selector: {e}")
        
    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona