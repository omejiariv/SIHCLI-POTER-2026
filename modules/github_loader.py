import io
import requests
import streamlit as st
import pandas as pd
# Agregamos GeoPandas aquí para optimizar la carga desde el origen
import geopandas as gpd 

@st.cache_data(ttl=3600)
def load_csv_from_url(url):
    """
    Carga un archivo CSV desde una URL a un DataFrame de Pandas,
    probando varias codificaciones comunes para evitar errores.
    """
    try:
        # 1. Descargamos el contenido del archivo
        response = requests.get(url)
        response.raise_for_status()
        content_bytes = response.content

        # 2. Lista de codificaciones a probar
        encodings_to_try = ["utf-8", "latin1", "cp1252"]
        df = None

        # 3. Bucle para encontrar la codificación correcta
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(io.BytesIO(content_bytes), sep=";", encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            st.error(f"No se pudo decodificar el archivo CSV desde {url}.")
            return None

        output_csv_bytes = df.to_csv(index=False, sep=";").encode("utf-8")
        return io.BytesIO(output_csv_bytes)

    except Exception as e:
        st.error(f"Error al cargar CSV: {url}\nError: {e}")
        return None


@st.cache_data(ttl=3600)
def load_zip_from_url(url, simplify_tolerance=None):
    """
    Descarga un archivo ZIP (shapefile).
    Si se pasa 'simplify_tolerance' (ej. 0.001), devuelve un GeoDataFrame optimizado (más rápido).
    Si no, devuelve el objeto bytes original (comportamiento antiguo).
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Si no pedimos simplificar, devolvemos bytes (comportamiento original)
        if simplify_tolerance is None:
            return io.BytesIO(response.content)
            
        # Si pedimos simplificar, procesamos con GeoPandas aquí mismo
        gdf = gpd.read_file(io.BytesIO(response.content))
        
        if 'geometry' in gdf.columns:
            # Esta línea hace la magia de velocidad
            gdf['geometry'] = gdf.geometry.simplify(tolerance=simplify_tolerance, preserve_topology=True)
            
        return gdf

    except Exception as e:
        st.error(f"Error al descargar ZIP: {url}\nError: {e}")
        return None