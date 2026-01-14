# modules/gbif_connector.py

import requests
import pandas as pd
import geopandas as gpd
import streamlit as st

# Cache para no saturar la API
@st.cache_data(ttl=3600, show_spinner=False)
def get_gbif_occurrences(minx, miny, maxx, maxy, limit=1000):
    """Consulta la API de GBIF para un Bounding Box."""
    api_url = "https://api.gbif.org/v1/occurrence/search"
    
    params = {
        'decimalLatitude': f"{miny},{maxy}",
        'decimalLongitude': f"{minx},{maxx}",
        'hasCoordinate': 'true',
        'limit': limit,
        # Incluimos Especímenes preservados además de observaciones humanas
        'basisOfRecord': 'HUMAN_OBSERVATION,OBSERVATION,MACHINE_OBSERVATION,PRESERVED_SPECIMEN'
    }
    
    try:
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and data['results']:
            df = pd.DataFrame(data['results'])
            
            # Mapeo de columnas
            cols_map = {
                'key': 'gbif_id',
                'scientificName': 'Nombre Científico',
                'vernacularName': 'Nombre Común',
                'kingdom': 'Reino', 'phylum': 'Filo', 'class': 'Clase',
                'order': 'Orden', 'family': 'Familia', 'genus': 'Género',
                'decimalLatitude': 'lat', 'decimalLongitude': 'lon',
                'iucnRedListCategory': 'Amenaza IUCN'
            }
            
            existing_cols = [c for c in cols_map.keys() if c in df.columns]
            df = df[existing_cols].rename(columns=cols_map)
            
            # Limpieza
            if 'Nombre Común' in df.columns:
                df['Nombre Común'] = df['Nombre Común'].fillna(df['Nombre Científico'])
            if 'Amenaza IUCN' in df.columns:
                df['Amenaza IUCN'] = df['Amenaza IUCN'].fillna('NE')
            
            return df
        return pd.DataFrame()
            
    except Exception as e:
        print(f"Error GBIF: {e}")
        return pd.DataFrame()

def get_biodiversity_in_polygon(gdf_zona, limit=2000):
    """
    Obtiene datos de GBIF.
    CRÍTICO: Convierte a EPSG:4326 ANTES de consultar.
    """
    if gdf_zona is None or gdf_zona.empty:
        return gpd.GeoDataFrame()
    
    # --- CORRECCIÓN CLAVE: FORZAR LAT/LON ---
    # GBIF solo entiende EPSG:4326. Si la cuenca viene en metros, convertimos primero.
    gdf_wgs84 = gdf_zona.to_crs("EPSG:4326")
    
    # 1. Bounding Box (Ahora sí en Grados Decimales)
    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    
    # Pequeño buffer (0.01 grados ~ 1km) para asegurar bordes
    buffer = 0.01
    
    # 2. API Call
    df_raw = get_gbif_occurrences(minx-buffer, miny-buffer, maxx+buffer, maxy+buffer, limit)
    
    if df_raw.empty:
        return gpd.GeoDataFrame()
    
    # 3. Convertir a GeoDataFrame
    gdf_points = gpd.GeoDataFrame(
        df_raw, 
        geometry=gpd.points_from_xy(df_raw.lon, df_raw.lat),
        crs="EPSG:4326"
    )
    
    # 4. Clip Espacial (Recorte)
    # Intentamos recortar exactamente con la forma de la cuenca
    try:
        gdf_final = gpd.clip(gdf_points, gdf_wgs84)
        
        # RED DE SEGURIDAD:
        # Si el clip borra todo (a veces pasa por bordes complejos), 
        # devolvemos los puntos del cuadro (gdf_points) para no dejar el mapa vacío.
        if len(gdf_final) > 0:
            return gdf_final
        else:
            print("Clip retornó vacío, mostrando Bounding Box")
            return gdf_points 
            
    except Exception as e:
        print(f"Error en clip: {e}, devolviendo puntos sin recortar")
        return gdf_points