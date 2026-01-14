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
        'basisOfRecord': 'HUMAN_OBSERVATION,OBSERVATION,MACHINE_OBSERVATION'
    }
    
    try:
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and data['results']:
            df = pd.DataFrame(data['results'])
            
            # Mapeo de columnas (inglés API -> español App)
            cols_map = {
                'key': 'gbif_id',
                'scientificName': 'Nombre Científico',
                'vernacularName': 'Nombre Común',
                'kingdom': 'Reino', 'phylum': 'Filo', 'class': 'Clase',
                'order': 'Orden', 'family': 'Familia', 'genus': 'Género',
                'decimalLatitude': 'lat', 'decimalLongitude': 'lon',
                'iucnRedListCategory': 'Amenaza IUCN'
            }
            
            # Filtrar solo columnas existentes
            existing_cols = [c for c in cols_map.keys() if c in df.columns]
            df = df[existing_cols].rename(columns=cols_map)
            
            # Limpieza básica
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
    """Obtiene datos de GBIF y los recorta con la forma exacta de la cuenca."""
    if gdf_zona is None or gdf_zona.empty:
        return gpd.GeoDataFrame()
    
    # 1. Bounding Box
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    
    # 2. API Call
    df_raw = get_gbif_occurrences(minx, miny, maxx, maxy, limit)
    
    if df_raw.empty:
        return gpd.GeoDataFrame()
    
    # 3. Convertir a GeoDataFrame
    gdf_points = gpd.GeoDataFrame(
        df_raw, 
        geometry=gpd.points_from_xy(df_raw.lon, df_raw.lat),
        crs="EPSG:4326"
    )
    
    # 4. Clip Espacial (Recorte)
    if gdf_zona.crs != gdf_points.crs:
        gdf_zona = gdf_zona.to_crs(gdf_points.crs)
        
    try:
        gdf_final = gpd.clip(gdf_points, gdf_zona)
        return gdf_final
    except:
        return gdf_points # Si falla el clip, devolvemos los puntos del cuadro