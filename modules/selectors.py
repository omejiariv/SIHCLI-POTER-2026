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
    
    Retorna:
        - ids_seleccionados (list): Códigos de estaciones filtradas.
        - nombre_seleccion (str): Etiqueta para títulos (ej: "Cuenca Río Grande").
        - altitud_ref (float): Altitud promedio de la selección.
        - gdf_area_interes (GeoDataFrame): La geometría del área seleccionada (para mapas).
    """
    st.sidebar.header("📍 Filtros Geográficos")
    
    # 1. CARGAR DATOS CENTRALIZADOS
    try:
        # CORRECCIÓN: Recibimos todo en una tupla y sacamos lo que necesitamos por índice
        # Esto evita el error si en el futuro agregas más cosas al procesador de datos.
        all_data = data_processor.load_and_process_all_data()
        
        gdf_stations = all_data[0]   # El primero siempre son las estaciones
        gdf_municipios = all_data[1] # El segundo son municipios
        gdf_subcuencas = all_data[2] # El tercero son cuencas
        
        # El resto (predios, enso, etc.) no lo necesitamos aquí, así que lo ignoramos.
        
    except Exception as e:
        st.sidebar.error(f"Error cargando datos espaciales: {e}")
        return [], "Error Datos", 1500, None
    # 2. SELECTOR DE MODO
    opciones_modo = ["📍 Por Estación", "🏙️ Por Municipio", "🌍 Por Región"]
    
    # Solo mostramos opción Cuenca si la capa existe
    if not gdf_subcuencas.empty:
        opciones_modo.append("⛰️ Por Cuenca")
        
    modo = st.sidebar.radio("Nivel de Agregación:", opciones_modo)
    st.sidebar.divider()
    
    # Variables de salida
    ids_out = []
    nombre_out = ""
    altitud_out = 1500
    gdf_area_out = None # Para guardar el polígono seleccionado

    # --- LÓGICA DE FILTRADO ---
    
    if modo == "📍 Por Estación":
        # Usamos el formato Nombre [Codigo]
        opciones = gdf_stations[Config.STATION_NAME_COL] + " [" + gdf_stations['codigo'] + "]"
        sel_str = st.sidebar.selectbox("Seleccione Estación:", options=opciones)
        
        if sel_str:
            # Extraer el código del string
            cod_sel = sel_str.split("[")[-1].replace("]", "")
            row = gdf_stations[gdf_stations['codigo'] == cod_sel].iloc[0]
            
            ids_out = [cod_sel]
            nombre_out = row[Config.STATION_NAME_COL]
            altitud_out = row[Config.ALTITUDE_COL] if pd.notnull(row[Config.ALTITUDE_COL]) else 1500
            # El área es el punto mismo (buffer pequeño opcional)
            gdf_area_out = gdf_stations[gdf_stations['codigo'] == cod_sel]

    elif modo == "🏙️ Por Municipio":
        lista = sorted(gdf_stations[Config.MUNICIPALITY_COL].dropna().unique())
        sel = st.sidebar.selectbox("Seleccione Municipio:", options=lista)
        
        if sel:
            subset = gdf_stations[gdf_stations[Config.MUNICIPALITY_COL] == sel]
            ids_out = subset['codigo'].tolist()
            nombre_out = f"Municipio de {sel}"
            altitud_out = subset[Config.ALTITUDE_COL].mean()
            # Intentamos buscar la geometría del municipio
            if not gdf_municipios.empty:
                # Asumimos que hay una columna nombre, ajusta si se llama distinto
                col_nom_mun = next((c for c in gdf_municipios.columns if 'nomb' in c.lower()), None)
                if col_nom_mun:
                    gdf_area_out = gdf_municipios[gdf_municipios[col_nom_mun] == sel]

    elif modo == "🌍 Por Región":
        lista = sorted(gdf_stations[Config.REGION_COL].dropna().unique())
        sel = st.sidebar.selectbox("Seleccione Región:", options=lista)
        
        if sel:
            subset = gdf_stations[gdf_stations[Config.REGION_COL] == sel]
            ids_out = subset['codigo'].tolist()
            nombre_out = f"Región {sel}"
            altitud_out = subset[Config.ALTITUDE_COL].mean()

    elif modo == "⛰️ Por Cuenca":
        # Buscamos la columna de nombre en la capa de cuencas
        col_nom_cuenca = next((c for c in gdf_subcuencas.columns if 'nomb' in c.lower() or 'cuenca' in c.lower()), gdf_subcuencas.columns[0])
        lista = sorted(gdf_subcuencas[col_nom_cuenca].astype(str).unique())
        
        sel = st.sidebar.selectbox("Seleccione Cuenca:", options=lista)
        
        if sel:
            # 1. Obtener la geometría de la cuenca seleccionada
            gdf_cuenca_sel = gdf_subcuencas[gdf_subcuencas[col_nom_cuenca] == sel]
            gdf_area_out = gdf_cuenca_sel
            
            # 2. CRUCE ESPACIAL (Spatial Join): Estaciones DENTRO de la Cuenca
            # Aseguramos proyecciones iguales
            if gdf_stations.crs != gdf_cuenca_sel.crs:
                gdf_stations = gdf_stations.to_crs(gdf_cuenca_sel.crs)
                
            estaciones_dentro = gpd.sjoin(gdf_stations, gdf_cuenca_sel, predicate='within')
            
            if not estaciones_dentro.empty:
                ids_out = estaciones_dentro['codigo'].tolist()
                altitud_out = estaciones_dentro[Config.ALTITUDE_COL].mean()
                st.sidebar.success(f"✅ {len(ids_out)} estaciones encontradas.")
            else:
                st.sidebar.warning("⚠️ Esta cuenca no contiene estaciones monitoreadas.")
            
            nombre_out = f"Cuenca {sel}"

    # Validación final de altitud
    if pd.isna(altitud_out): altitud_out = 1500
    
    return ids_out, nombre_out, altitud_out, gdf_area_out