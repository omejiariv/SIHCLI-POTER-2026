# modules/gbif_connector.py

import requests
import pandas as pd
import geopandas as gpd
import streamlit as st

# Cacheamos la consulta (1 hora)
@st.cache_data(ttl=3600, show_spinner=False)
def get_gbif_occurrences(minx, miny, maxx, maxy, limit=1000):
    """
    Consulta la API de GBIF.
    CORRECCIÓN: 'basisOfRecord' ahora se pasa como lista para que requests lo codifique bien (key=val1&key=val2).
    """
    api_url = "https://api.gbif.org/v1/occurrence/search"
    
    # Parámetros corregidos
    params = {
        'decimalLatitude': f"{miny},{maxy}",
        'decimalLongitude': f"{minx},{maxx}",
        'hasCoordinate': 'true',
        'limit': limit,
        # CORRECCIÓN: Lista Python, requests lo convertirá automáticamente
        'basisOfRecord': ['HUMAN_OBSERVATION', 'OBSERVATION', 'MACHINE_OBSERVATION', 'PRESERVED_SPECIMEN'] 
    }
    
    # Identificación para evitar bloqueos
    headers = {
        'User-Agent': 'SIHCLI-App/1.0 (Research Project)'
    }
    
    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=20)
        response.raise_for_status() # Lanza error si hay 404 o 500
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
            
            # Filtrar columnas existentes
            existing_cols = [c for c in cols_map.keys() if c in df.columns]
            df = df[existing_cols].rename(columns=cols_map)
            
            # Limpieza estética
            if 'Nombre Común' in df.columns:
                df['Nombre Común'] = df['Nombre Común'].fillna(df['Nombre Científico'])
            if 'Amenaza IUCN' in df.columns:
                df['Amenaza IUCN'] = df['Amenaza IUCN'].fillna('NE')
            
            return df
        
        # Si la respuesta es válida pero vacía
        return pd.DataFrame()
            
    except Exception as e:
        print(f"Error interno GBIF Connector: {e}")
        return pd.DataFrame()

def get_biodiversity_in_polygon(gdf_zona, limit=2000):
    """
    Obtiene datos y recorta espacialmente.
    Incluye red de seguridad: Si el recorte falla, devuelve el cuadro completo.
    """
    if gdf_zona is None or gdf_zona.empty:
        return gpd.GeoDataFrame()
    
    # Asegurar WGS84
    gdf_wgs84 = gdf_zona.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf_wgs84.total_bounds
    
    # Consultar API
    df_raw = get_gbif_occurrences(minx, miny, maxx, maxy, limit)
    
    if df_raw.empty:
        return gpd.GeoDataFrame()
    
    # Convertir a Geo
    gdf_points = gpd.GeoDataFrame(
        df_raw, 
        geometry=gpd.points_from_xy(df_raw.lon, df_raw.lat),
        crs="EPSG:4326"
    )
    
    # Intentar recorte exacto (Clip)
    try:
        gdf_final = gpd.clip(gdf_points, gdf_wgs84)
        
        # SI EL RECORTE BORRA TODO (A veces pasa en bordes), devolvemos los datos crudos
        if len(gdf_final) > 0:
            return gdf_final
        else:
            return gdf_points 
            
    except Exception:
        # Si falla el proceso de recorte, devolvemos todo
        return gdf_points