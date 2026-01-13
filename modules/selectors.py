# modules/selectors.py
import streamlit as st
import geopandas as gpd
import pandas as pd
from modules import data_processor
from modules.config import Config
from shapely import wkt

def render_selector_espacial():
    """
    Renderiza la barra lateral de selección espacial universal.
    Mapeo corregido: [0]=Estaciones, [4]=Cuencas.
    """
    st.sidebar.header("📍 Filtros Geográficos")
    
    # 1. CARGAR DATOS (Mapeo por Índices Corregido)
    try:
        all_data = data_processor.load_and_process_all_data()
        
        # --- MAPEO EXACTO SEGÚN TU DIAGNÓSTICO ---
        _stations = all_data[0]   # Índice 0: Estaciones
        _municipios = all_data[1] # Índice 1: Municipios
        _subcuencas = all_data[4] # Índice 4: CUENCAS (171 filas)
        
        # --- BLINDAJE Y CONVERSIÓN DE GEOMETRÍAS ---
        
        # A. Estaciones
        if isinstance(_stations, gpd.GeoDataFrame):
            gdf_stations = _stations
        else:
            # Si perdió el formato Geo, lo recuperamos
            if 'longitude' in _stations.columns:
                gdf_stations = gpd.GeoDataFrame(
                    _stations, 
                    geometry=gpd.points_from_xy(_stations.longitude, _stations.latitude),
                    crs="EPSG:4326"
                )
            else:
                gdf_stations = gpd.GeoDataFrame(_stations)

        # B. Cuencas (La clave del éxito)
        if isinstance(_subcuencas, gpd.GeoDataFrame):
            gdf_subcuencas = _subcuencas
        else:
            # Conversión de emergencia si llega como texto WKT
            if not _subcuencas.empty and 'wkt' in _subcuencas.columns:
                try:
                    _subcuencas['geometry'] = _subcuencas['wkt'].apply(wkt.loads)
                    gdf_subcuencas = gpd.GeoDataFrame(_subcuencas, geometry='geometry')
                    gdf_subcuencas.set_crs("EPSG:4326", inplace=True)
                except:
                    gdf_subcuencas = gpd.GeoDataFrame()
            else:
                gdf_subcuencas = gpd.GeoDataFrame(_subcuencas) # Fallback

        # Asegurar CRS para operaciones espaciales
        if gdf_stations.crs is None: gdf_stations.set_crs("EPSG:4326", inplace=True)
        if gdf_subcuencas.crs is None and not gdf_subcuencas.empty: gdf_subcuencas.set_crs("EPSG:4326", inplace=True)

        # C. Municipios
        gdf_municipios = _municipios

    except Exception as e:
        st.sidebar.error(f"Error cargando datos: {e}")
        return [], "Error", 1500, None

    # 2. SELECTOR DE MODO
    opciones_modo = ["📍 Por Estación", "🏙️ Por Municipio", "🌍 Por Región"]
    
    if not gdf_subcuencas.empty:
        opciones_modo.append("⛰️ Por Cuenca")
        
    modo = st.sidebar.radio("Nivel de Agregación:", opciones_modo)
    st.sidebar.divider()
    
    # Variables de salida
    ids_out = []
    nombre_out = ""
    altitud_out = 1500
    gdf_area_out = None 

    # --- LÓGICA DE FILTRADO ---
    
    if modo == "📍 Por Estación":
        col_id = 'id_estacion' if 'id_estacion' in gdf_stations.columns else 'codigo'
        opciones = gdf_stations[Config.STATION_NAME_COL] + " [" + gdf_stations[col_id] + "]"
        sel_str = st.sidebar.selectbox("Seleccione Estación:", options=opciones)
        
        if sel_str:
            cod_sel = sel_str.split("[")[-1].replace("]", "")
            row = gdf_stations[gdf_stations[col_id] == cod_sel].iloc[0]
            
            ids_out = [cod_sel]
            nombre_out = row[Config.STATION_NAME_COL]
            altitud_out = row[Config.ALTITUDE_COL] if pd.notnull(row[Config.ALTITUDE_COL]) else 1500
            gdf_area_out = gdf_stations[gdf_stations[col_id] == cod_sel]

    elif modo == "🏙️ Por Municipio":
        lista = sorted(gdf_stations[Config.MUNICIPALITY_COL].dropna().unique())
        sel = st.sidebar.selectbox("Seleccione Municipio:", options=lista)
        
        if sel:
            subset = gdf_stations[gdf_stations[Config.MUNICIPALITY_COL] == sel]
            col_id = 'id_estacion' if 'id_estacion' in subset.columns else 'codigo'
            ids_out = subset[col_id].tolist()
            nombre_out = f"Municipio de {sel}"
            altitud_out = subset[Config.ALTITUDE_COL].mean()
            
            # Intentar geometría
            if not gdf_municipios.empty:
                cols = gdf_municipios.columns
                col_nom = next((c for c in cols if 'nomb' in c.lower()), cols[0])
                gdf_area_out = gdf_municipios[gdf_municipios[col_nom] == sel]

    elif modo == "🌍 Por Región":
        lista = sorted(gdf_stations[Config.REGION_COL].dropna().unique())
        sel = st.sidebar.selectbox("Seleccione Región:", options=lista)
        
        if sel:
            subset = gdf_stations[gdf_stations[Config.REGION_COL] == sel]
            col_id = 'id_estacion' if 'id_estacion' in subset.columns else 'codigo'
            ids_out = subset[col_id].tolist()
            nombre_out = f"Región {sel}"
            altitud_out = subset[Config.ALTITUDE_COL].mean()

    elif modo == "⛰️ Por Cuenca":
        # Buscar nombre de columna flexiblemente (nombre, Name, cuenca...)
        cols = gdf_subcuencas.columns
        col_nom = next((c for c in cols if 'nomb' in c.lower() or 'cuenca' in c.lower()), cols[0])
        lista = sorted(gdf_subcuencas[col_nom].astype(str).unique())
        
        sel = st.sidebar.selectbox("Seleccione Cuenca:", options=lista)
        
        if sel:
            gdf_cuenca_sel = gdf_subcuencas[gdf_subcuencas[col_nom] == sel]
            gdf_area_out = gdf_cuenca_sel
            
            # Spatial Join Robusto
            if gdf_stations.crs != gdf_cuenca_sel.crs:
                gdf_stations = gdf_stations.to_crs(gdf_cuenca_sel.crs)
                
            estaciones_dentro = gpd.sjoin(gdf_stations, gdf_cuenca_sel, predicate='within')
            
            if not estaciones_dentro.empty:
                col_id = 'id_estacion' if 'id_estacion' in estaciones_dentro.columns else 'codigo'
                ids_out = estaciones_dentro[col_id].tolist()
                altitud_out = estaciones_dentro[Config.ALTITUDE_COL].mean()
                st.sidebar.success(f"✅ {len(ids_out)} estaciones.")
            else:
                st.sidebar.warning("⚠️ Sin estaciones monitoreadas en esta cuenca.")
            
            nombre_out = f"Cuenca {sel}"

    if pd.isna(altitud_out): altitud_out = 1500
    
    return ids_out, nombre_out, altitud_out, gdf_area_out