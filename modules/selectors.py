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
        all_data = data_processor.load_and_process_all_data()
        
        # Recuperamos los datos crudos
        _stations = all_data[0]
        _municipios = all_data[1]
        _subcuencas = all_data[2]
        
        # --- BLOQUE DE SANITIZACIÓN GEOGRÁFICA (EL BLINDAJE) ---
        # 1. Asegurar que Estaciones sea GeoDataFrame
        if isinstance(_stations, gpd.GeoDataFrame):
            gdf_stations = _stations
        else:
            # Si es DataFrame normal, lo convertimos usando lat/lon
            if 'longitude' in _stations.columns and 'latitude' in _stations.columns:
                gdf_stations = gpd.GeoDataFrame(
                    _stations, 
                    geometry=gpd.points_from_xy(_stations.longitude, _stations.latitude),
                    crs="EPSG:4326"
                )
            else:
                gdf_stations = gpd.GeoDataFrame(_stations) # Fallback vacío

        # 2. Asegurar que Cuencas sea GeoDataFrame
        if isinstance(_subcuencas, gpd.GeoDataFrame):
            gdf_subcuencas = _subcuencas
        else:
            # Si tiene columna geometry pero no es GeoDataFrame
            if 'geometry' in _subcuencas.columns and not _subcuencas.empty:
                try:
                    from shapely import wkt
                    # A veces llega como texto (WKT), intentamos convertir
                    if _subcuencas['geometry'].dtype == 'object':
                        # Verificamos si es string antes de aplicar wkt.loads
                        first_val = _subcuencas['geometry'].iloc[0]
                        if isinstance(first_val, str):
                            _subcuencas['geometry'] = _subcuencas['geometry'].apply(wkt.loads)
                            
                    gdf_subcuencas = gpd.GeoDataFrame(_subcuencas, geometry='geometry')
                    # Asignamos CRS si no tiene (asumimos WGS84 por defecto)
                    if gdf_subcuencas.crs is None:
                        gdf_subcuencas.set_crs("EPSG:4326", inplace=True)
                except Exception as e:
                    print(f"Error convirtiendo geometría cuencas: {e}")
                    gdf_subcuencas = gpd.GeoDataFrame() # Fallback
            else:
                 gdf_subcuencas = gpd.GeoDataFrame() # Fallback

        # Pasamos municipios directo (no es crítico para el cruce espacial de cuencas)
        gdf_municipios = _municipios
        
    except Exception as e:
        st.sidebar.error(f"Error cargando/procesando datos: {e}")
        return [], "Error Datos", 1500, None

    # 2. SELECTOR DE MODO
    opciones_modo = ["📍 Por Estación", "🏙️ Por Municipio", "🌍 Por Región"]
    
    # Solo mostramos Cuenca si logramos convertirla a GeoDataFrame válido
    if not gdf_subcuencas.empty and isinstance(gdf_subcuencas, gpd.GeoDataFrame):
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
        # Usamos 'id_estacion' (nombre real en BD)
        if 'id_estacion' in gdf_stations.columns:
            col_id = 'id_estacion' 
        else:
            col_id = 'codigo' # Fallback por si acaso

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
            
            if not gdf_municipios.empty:
                cols = gdf_municipios.columns
                col_nom = next((c for c in cols if 'nomb' in c.lower() or 'muni' in c.lower()), cols[0])
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
        col_nom_cuenca = next((c for c in gdf_subcuencas.columns if 'nomb' in c.lower() or 'cuenca' in c.lower()), gdf_subcuencas.columns[0])
        lista = sorted(gdf_subcuencas[col_nom_cuenca].astype(str).unique())
        
        sel = st.sidebar.selectbox("Seleccione Cuenca:", options=lista)
        
        if sel:
            gdf_cuenca_sel = gdf_subcuencas[gdf_subcuencas[col_nom_cuenca] == sel]
            gdf_area_out = gdf_cuenca_sel
            
            # CRUCE ESPACIAL SEGURO
            # Ahora estamos seguros de que ambos son GeoDataFrames con .crs
            if gdf_stations.crs is None:
                gdf_stations.set_crs("EPSG:4326", inplace=True)
            if gdf_cuenca_sel.crs is None:
                gdf_cuenca_sel.set_crs("EPSG:4326", inplace=True) # Asumimos WGS84 si falta
                
            if gdf_stations.crs != gdf_cuenca_sel.crs:
                gdf_stations = gdf_stations.to_crs(gdf_cuenca_sel.crs)
                
            estaciones_dentro = gpd.sjoin(gdf_stations, gdf_cuenca_sel, predicate='within')
            
            if not estaciones_dentro.empty:
                col_id = 'id_estacion' if 'id_estacion' in estaciones_dentro.columns else 'codigo'
                ids_out = estaciones_dentro[col_id].tolist()
                altitud_out = estaciones_dentro[Config.ALTITUDE_COL].mean()
                st.sidebar.success(f"✅ {len(ids_out)} estaciones.")
            else:
                st.sidebar.warning("⚠️ Sin estaciones en esta cuenca.")
            
            nombre_out = f"Cuenca {sel}"

    if pd.isna(altitud_out): altitud_out = 1500
    
    return ids_out, nombre_out, altitud_out, gdf_area_out