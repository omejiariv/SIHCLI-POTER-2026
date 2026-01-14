import streamlit as st
import pandas as pd
import geopandas as gpd
import os

def render_selector_espacial():
    st.sidebar.header("📍 Filtros Geográficos")
    
    # 1. Selector de Nivel
    nivel = st.sidebar.radio(
        "Nivel de Agregación:",
        ["Por Cuenca", "Por Municipio", "Departamento (Antioquia)"]
    )
    
    ids_seleccionados = []
    nombre_seleccion = ""
    altitud_ref = 1500
    gdf_zona = None
    
    # Rutas a archivos (Aseguramos ruta absoluta)
    base_dir = os.path.dirname(os.path.dirname(__file__)) # Sube un nivel desde modules/
    path_cuencas = os.path.join(base_dir, 'data', 'SubcuencasAinfluencia.geojson')
    path_munis = os.path.join(base_dir, 'data', 'MunicipiosAntioquia.geojson')
    
    try:
        if nivel == "Por Cuenca":
            if os.path.exists(path_cuencas):
                # Cargar GeoJSON (Cachear si es pesado, por ahora directo)
                gdf_all = gpd.read_file(path_cuencas)
                
                # Buscar columna de nombre
                name_col = next((c for c in ['nombre', 'Name', 'NOM_CUENCA', 'subcuenca', 'SBC_CNMBR'] if c in gdf_all.columns), None)
                
                if name_col:
                    opciones = sorted(gdf_all[name_col].astype(str).unique())
                    sel = st.sidebar.selectbox("Seleccione Cuenca:", opciones)
                    
                    if sel:
                        nombre_seleccion = f"Cuenca {sel}"
                        gdf_zona = gdf_all[gdf_all[name_col] == sel]
                else:
                    st.sidebar.error("No se encontró columna de nombre en el archivo de cuencas.")
            else:
                st.sidebar.error(f"Archivo no encontrado: {path_cuencas}")

        elif nivel == "Por Municipio":
            if os.path.exists(path_munis):
                gdf_all = gpd.read_file(path_munis)
                name_col = next((c for c in ['MPIO_CNMBR', 'nombre', 'NOMBRE', 'municipio'] if c in gdf_all.columns), None)
                
                if name_col:
                    opciones = sorted(gdf_all[name_col].astype(str).unique())
                    sel = st.sidebar.selectbox("Seleccione Municipio:", opciones)
                    
                    if sel:
                        nombre_seleccion = f"Mpio. {sel}"
                        gdf_zona = gdf_all[gdf_all[name_col] == sel]
            else:
                st.sidebar.error(f"Archivo no encontrado: {path_munis}")

        elif nivel == "Departamento (Antioquia)":
            nombre_seleccion = "Antioquia"
            if os.path.exists(path_munis):
                # Usamos la unión de municipios como Antioquia si no hay shape departamental
                gdf_all = gpd.read_file(path_munis)
                gdf_zona = gdf_all.dissolve() # Fusionar todo
            else:
                st.sidebar.warning("No se pudo cargar geometría de Antioquia.")

        # Configuración Espacial Común (Buffer)
        if gdf_zona is not None and not gdf_zona.empty:
            # Asegurar CRS WGS84
            if gdf_zona.crs and gdf_zona.crs != "EPSG:4326":
                gdf_zona = gdf_zona.to_crs("EPSG:4326")
                
            buffer_km = st.sidebar.slider("Radio Buffer (km):", 0, 50, 0)
            if buffer_km > 0:
                # Proyectar a metros temporalmente para buffer
                gdf_metros = gdf_zona.to_crs("EPSG:3116") # Magna Sirgas
                gdf_metros['geometry'] = gdf_metros.buffer(buffer_km * 1000)
                gdf_zona = gdf_metros.to_crs("EPSG:4326")
                st.sidebar.success(f"Zona ampliada +{buffer_km}km")

    except Exception as e:
        st.sidebar.error(f"Error en selector: {e}")
    
    return ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona