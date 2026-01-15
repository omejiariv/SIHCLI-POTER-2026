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
    
    # Rutas a archivos
    base_dir = os.path.dirname(os.path.dirname(__file__))
    path_cuencas = os.path.join(base_dir, 'data', 'SubcuencasAinfluencia.geojson')
    path_munis = os.path.join(base_dir, 'data', 'MunicipiosAntioquia.geojson')
    
    try:
        # --- LÓGICA CUENCAS ---
        if nivel == "Por Cuenca":
            if os.path.exists(path_cuencas):
                gdf_all = gpd.read_file(path_cuencas)
                
                # LISTA AMPLIADA DE POSIBLES NOMBRES DE COLUMNA
                posibles_nombres = [
                    'nombre', 'Name', 'NAME', 'NOM_CUENCA', 'subcuenca', 'SBC_CNMBR', 
                    'NOMBRE_SUB', 'NOMBRE', 'nom_subcue', 'Cuenca', 'CUENCA', 'Label'
                ]
                
                # Buscar la primera coincidencia
                name_col = next((c for c in posibles_nombres if c in gdf_all.columns), None)
                
                # Si falla, usar la primera columna de texto como fallback
                if not name_col:
                    object_cols = gdf_all.select_dtypes(include=['object']).columns
                    if not object_cols.empty:
                        name_col = object_cols[0]
                
                if name_col:
                    # Ordenar y limpiar
                    opciones = sorted(gdf_all[name_col].astype(str).unique())
                    sel = st.sidebar.selectbox("Seleccione Cuenca:", opciones)
                    
                    if sel:
                        nombre_seleccion = f"Cuenca {sel}"
                        gdf_zona = gdf_all[gdf_all[name_col] == sel]
                else:
                    st.sidebar.error(f"Error: No se identificó columna de nombre. Columnas disponibles: {list(gdf_all.columns)}")
            else:
                st.sidebar.error(f"Archivo no encontrado: {path_cuencas}")

        # --- LÓGICA MUNICIPIOS ---
        elif nivel == "Por Municipio":
            if os.path.exists(path_munis):
                gdf_all = gpd.read_file(path_munis)
                
                # Buscar nombre
                posibles_nombres = ['MPIO_CNMBR', 'nombre', 'NOMBRE', 'municipio', 'MPIO_NOMBRE']
                name_col = next((c for c in posibles_nombres if c in gdf_all.columns), None)
                
                if name_col:
                    opciones = sorted(gdf_all[name_col].astype(str).unique())
                    sel = st.sidebar.selectbox("Seleccione Municipio:", opciones)
                    
                    if sel:
                        nombre_seleccion = f"Mpio. {sel}"
                        gdf_zona = gdf_all[gdf_all[name_col] == sel]
                else:
                    st.sidebar.error("Columna de nombre no encontrada en Municipios.")
            else:
                st.sidebar.error(f"Archivo no encontrado: {path_munis}")

        # --- LÓGICA DEPTO ---
        elif nivel == "Departamento (Antioquia)":
            nombre_seleccion = "Antioquia"
            if os.path.exists(path_munis):
                gdf_all = gpd.read_file(path_munis)
                gdf_zona = gdf_all.dissolve() # Disolver para obtener el contorno de Antioquia
            else:
                st.sidebar.warning("No se pudo cargar geometría de Antioquia.")

        # --- CONFIGURACIÓN ESPACIAL ---
        if gdf_zona is not None and not gdf_zona.empty:
            if gdf_zona.crs and gdf_zona.crs != "EPSG:4326":
                gdf_zona = gdf_zona.to_crs("EPSG:4326")
                
            buffer_km = st.sidebar.slider("Radio Buffer (km):", 0, 50, 0)
            if buffer_km > 0:
                gdf_metros = gdf_zona.to_crs("EPSG:3116")
                gdf_metros['geometry'] = gdf_metros.buffer(buffer_km * 1000)
                gdf_zona = gdf_metros.to_crs("EPSG:4326")
                st.sidebar.success(f"Zona ampliada +{buffer_km}km")

    except Exception as e:
        st.sidebar.error(f"Error crítico en selector: {e}")
    
    return ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona