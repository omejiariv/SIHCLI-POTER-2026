# modules/selectors.py
import streamlit as st
import geopandas as gpd
import pandas as pd
from modules import data_processor
from modules.config import Config

def render_selector_espacial():
    """
    Renderiza la barra lateral de selección espacial universal.
    Integra Estaciones, Municipios, Regiones y CUENCAS (vía cruce espacial).
    """
    st.sidebar.header("📍 Filtros Geográficos")
    
    # 1. CARGAR DATOS CENTRALIZADOS
    try:
        # Recibimos la tupla y extraemos los 3 primeros elementos
        all_data = data_processor.load_and_process_all_data()
        
        gdf_stations = all_data[0]   # Estaciones
        gdf_municipios = all_data[1] # Municipios
        gdf_subcuencas = all_data[2] # Cuencas
        
    except Exception as e:
        st.sidebar.error(f"Error cargando datos: {e}")
        return [], "Error Datos", 1500, None

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

    # --- LÓGICA DE FILTRADO CORREGIDA (id_estacion) ---
    
    if modo == "📍 Por Estación":
        # CORRECCIÓN AQUÍ: Usamos 'id_estacion' en lugar de 'codigo'
        opciones = gdf_stations[Config.STATION_NAME_COL] + " [" + gdf_stations['id_estacion'] + "]"
        sel_str = st.sidebar.selectbox("Seleccione Estación:", options=opciones)
        
        if sel_str:
            cod_sel = sel_str.split("[")[-1].replace("]", "")
            # CORRECCIÓN AQUÍ
            row = gdf_stations[gdf_stations['id_estacion'] == cod_sel].iloc[0]
            
            ids_out = [cod_sel]
            nombre_out = row[Config.STATION_NAME_COL]
            altitud_out = row[Config.ALTITUDE_COL] if pd.notnull(row[Config.ALTITUDE_COL]) else 1500
            # CORRECCIÓN AQUÍ
            gdf_area_out = gdf_stations[gdf_stations['id_estacion'] == cod_sel]

    elif modo == "🏙️ Por Municipio":
        lista = sorted(gdf_stations[Config.MUNICIPALITY_COL].dropna().unique())
        sel = st.sidebar.selectbox("Seleccione Municipio:", options=lista)
        
        if sel:
            subset = gdf_stations[gdf_stations[Config.MUNICIPALITY_COL] == sel]
            # CORRECCIÓN AQUÍ
            ids_out = subset['id_estacion'].tolist()
            nombre_out = f"Municipio de {sel}"
            altitud_out = subset[Config.ALTITUDE_COL].mean()
            
            # Buscar geometría del municipio
            if not gdf_municipios.empty:
                # Buscamos columna de nombre flexiblemente
                cols = gdf_municipios.columns
                col_nom = next((c for c in cols if 'nomb' in c.lower() or 'muni' in c.lower()), cols[0])
                gdf_area_out = gdf_municipios[gdf_municipios[col_nom] == sel]

    elif modo == "🌍 Por Región":
        lista = sorted(gdf_stations[Config.REGION_COL].dropna().unique())
        sel = st.sidebar.selectbox("Seleccione Región:", options=lista)
        
        if sel:
            subset = gdf_stations[gdf_stations[Config.REGION_COL] == sel]
            # CORRECCIÓN AQUÍ
            ids_out = subset['id_estacion'].tolist()
            nombre_out = f"Región {sel}"
            altitud_out = subset[Config.ALTITUDE_COL].mean()

    elif modo == "⛰️ Por Cuenca":
        col_nom_cuenca = next((c for c in gdf_subcuencas.columns if 'nomb' in c.lower() or 'cuenca' in c.lower()), gdf_subcuencas.columns[0])
        lista = sorted(gdf_subcuencas[col_nom_cuenca].astype(str).unique())
        
        sel = st.sidebar.selectbox("Seleccione Cuenca:", options=lista)
        
        if sel:
            gdf_cuenca_sel = gdf_subcuencas[gdf_subcuencas[col_nom_cuenca] == sel]
            gdf_area_out = gdf_cuenca_sel
            
            if gdf_stations.crs != gdf_cuenca_sel.crs:
                gdf_stations = gdf_stations.to_crs(gdf_cuenca_sel.crs)
                
            estaciones_dentro = gpd.sjoin(gdf_stations, gdf_cuenca_sel, predicate='within')
            
            if not estaciones_dentro.empty:
                # CORRECCIÓN AQUÍ
                ids_out = estaciones_dentro['id_estacion'].tolist()
                altitud_out = estaciones_dentro[Config.ALTITUDE_COL].mean()
                st.sidebar.success(f"✅ {len(ids_out)} estaciones encontradas.")
            else:
                st.sidebar.warning("⚠️ Esta cuenca no contiene estaciones monitoreadas.")
            
            nombre_out = f"Cuenca {sel}"

    if pd.isna(altitud_out): altitud_out = 1500
    
    return ids_out, nombre_out, altitud_out, gdf_area_out