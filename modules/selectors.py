import streamlit as st
import pandas as pd
import geopandas as gpd
import os

# --- OPTIMIZACIÓN DE VELOCIDAD (CACHÉ) ---
# Esta función carga los archivos UNA sola vez y los guarda en memoria.
@st.cache_data(ttl=3600, show_spinner=False)
def load_geodata_cached(file_path):
    if os.path.exists(file_path):
        try:
            # Leemos el archivo
            gdf = gpd.read_file(file_path)
            # Optimizamos proyecciones de una vez
            if gdf.crs and gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            return gdf
        except Exception as e:
            return None
    return None

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
            # Usamos la carga optimizada
            gdf_all = load_geodata_cached(path_cuencas)
            
            if gdf_all is not None:
                # LISTA DE CANDIDATOS (Agrega aquí el nombre si lo ves en la lista de abajo)
                posibles_nombres = [
                    'SUBC_LBL', # Nombres técnicos comunes
                    'subcuenca', 
                    'NOMBRE_SUB', 'nom_subcue', 'Cuenca', 'CUENCA', 'Label'
                ]
                
                # Buscar coincidencia exacta
                name_col = next((c for c in posibles_nombres if c in gdf_all.columns), None)
                
                # --- DIAGNÓSTICO DE COLUMNAS ---
                # Si sigue saliendo mal, descomenta la siguiente línea para ver las columnas:
                # st.sidebar.write("Columnas encontradas:", list(gdf_all.columns))
                
                if name_col:
                    opciones = sorted(gdf_all[name_col].astype(str).unique())
                    sel = st.sidebar.selectbox("Seleccione Cuenca:", opciones)
                    
                    if sel:
                        nombre_seleccion = f"Cuenca {sel}"
                        gdf_zona = gdf_all[gdf_all[name_col] == sel]
                else:
                    # SI FALLA: Muestra las columnas disponibles para que sepas cuál es
                    st.sidebar.error("⚠️ No encontré la columna 'Nombre'.")
                    st.sidebar.info(f"Las columnas en el archivo son: {list(gdf_all.columns)}")
                    st.sidebar.markdown("**Avísame cuál de estas es el nombre de la cuenca.**")
            else:
                st.sidebar.error(f"Archivo no encontrado o corrupto: {path_cuencas}")

        # --- LÓGICA MUNICIPIOS ---
        elif nivel == "Por Municipio":
            gdf_all = load_geodata_cached(path_munis)
            
            if gdf_all is not None:
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
            gdf_all = load_geodata_cached(path_munis)
            if gdf_all is not None:
                # Dissolve es lento, intentamos cachearlo también si fuera necesario, 
                # pero por ahora lo hacemos al vuelo (solo una vez)
                gdf_zona = gdf_all.dissolve() 
            else:
                st.sidebar.warning("No se pudo cargar geometría de Antioquia.")

        # --- CONFIGURACIÓN ESPACIAL ---
        if gdf_zona is not None and not gdf_zona.empty:
            buffer_km = st.sidebar.slider("Radio Buffer (km):", 0, 50, 0)
            if buffer_km > 0:
                gdf_metros = gdf_zona.to_crs("EPSG:3116")
                gdf_metros['geometry'] = gdf_metros.buffer(buffer_km * 1000)
                gdf_zona = gdf_metros.to_crs("EPSG:4326")
                st.sidebar.success(f"Zona ampliada +{buffer_km}km")

    except Exception as e:
        st.sidebar.error(f"Error crítico en selector: {e}")
    
    return ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona