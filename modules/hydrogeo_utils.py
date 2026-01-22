# modules/hydrogeo_utils.py

import pandas as pd
import numpy as np
from . import analysis, config
import io
import rasterio
from rasterio.transform import from_origin

# --- 1. LÓGICA DE CÁLCULO DE SERIES ---
def calcular_serie_recarga(df_lluvia, lat, altitud, ki_suelo=None):
    """
    Calcula el balance hídrico mensual: Lluvia -> ETR -> Excedente -> Recarga/Escorrentía.
    """
    df = df_lluvia.copy()
    
    # Estandarizar fecha
    if config.Config.DATE_COL not in df.columns and 'fecha' in df.columns:
        df = df.rename(columns={'fecha': config.Config.DATE_COL})
    
    df[config.Config.DATE_COL] = pd.to_datetime(df[config.Config.DATE_COL])
    df = df.sort_values(config.Config.DATE_COL)
    
    # IMPORTANTE: Eliminar duplicados de fecha para evitar error en Prophet
    df = df.drop_duplicates(subset=[config.Config.DATE_COL])

    # Estimar Temperatura (Gradiente altitudinal)
    temp_estimada = analysis.estimate_temperature(altitud)
    
    # Calcular ETR y Agua Disponible (Turc)
    # analysis.calculate_water_balance_turc retorna (etr, excedente)
    resultados = df[config.Config.PRECIPITATION_COL].apply(
        lambda p: analysis.calculate_water_balance_turc(p, temp_estimada)
    )
    
    df['etr_mm'] = [x[0] for x in resultados]
    df['agua_disponible_mm'] = [x[1] for x in resultados]
    
    # Calcular Recarga vs Escorrentía usando Ki del suelo
    factor_inf = ki_suelo if pd.notnull(ki_suelo) else 0.15 # Default 15% si no hay dato
    
    df['recarga_mm'] = df['agua_disponible_mm'] * factor_inf
    df['escorrentia_sup_mm'] = df['agua_disponible_mm'] * (1 - factor_inf)
    
    return df[[config.Config.DATE_COL, config.Config.PRECIPITATION_COL, 'etr_mm', 'recarga_mm', 'escorrentia_sup_mm']]

# --- 2. LÓGICA PARA EL MAPA (Datos de todas las estaciones) ---
def obtener_datos_estaciones_recarga(engine):
    """
    Calcula la recarga media ANUAL para todas las estaciones disponibles.
    Retorna un DataFrame listo para interpolar.
    """
    # A. Obtener ubicación y suelos
    q_meta = """
    SELECT e.id_estacion, e.nom_est, e.latitud, e.longitud, e.elevacion, s.infiltracion_ki 
    FROM estaciones e 
    LEFT JOIN suelos s ON ST_Intersects(e.geom, s.geom)
    """
    df_meta = pd.read_sql(q_meta, engine)
    
    # B. Obtener Lluvia Promedio Mensual
    q_lluvia = """
    SELECT id_estacion_fk as id_estacion, AVG(precipitation) as ppt_media_mes
    FROM precipitacion_mensual 
    GROUP BY id_estacion_fk
    """
    df_lluvia = pd.read_sql(q_lluvia, engine)
    
    # C. Unir y Calcular
    df_full = pd.merge(df_meta, df_lluvia, on='id_estacion')
    
    # Cálculo vectorizado (rápido para miles de filas)
    df_full['temp_est'] = 28.0 - (0.006 * df_full['elevacion'])
    
    # Fórmula Turc vectorizada
    L = 300 + 25 * df_full['temp_est'] + 0.05 * (df_full['temp_est']**3)
    denom = np.sqrt(0.9 + (df_full['ppt_media_mes'] / L)**2)
    df_full['etr_est_mes'] = df_full['ppt_media_mes'] / denom
    
    # Recarga
    df_full['ki_final'] = df_full['infiltracion_ki'].fillna(0.15)
    df_full['recarga_mes'] = (df_full['ppt_media_mes'] - df_full['etr_est_mes']) * df_full['ki_final']
    df_full.loc[df_full['recarga_mes'] < 0, 'recarga_mes'] = 0 # Corregir negativos matemáticos
    
    # D. Proyección Anual (x12) para que el mapa tenga sentido
    df_full['recarga_anual'] = df_full['recarga_mes'] * 12
    
    return df_full

# --- 3. GENERADORES DE DESCARGA (Bytes en memoria) ---
def generar_geotiff_bytes(z_grid, bounds, crs_code=4326):
    """Crea un archivo TIFF en memoria RAM."""
    min_x, min_y, max_x, max_y = bounds
    height, width = z_grid.shape
    pixel_width = (max_x - min_x) / width
    pixel_height = (max_y - min_y) / height 
    
    transform = from_origin(min_x, max_y, pixel_width, pixel_height)
    
    memfile = io.BytesIO()
    with rasterio.open(
        memfile, 'w', driver='GTiff',
        height=height, width=width, count=1, dtype='float32',
        crs=f"EPSG:{crs_code}", transform=transform, nodata=-9999
    ) as dst:
        dst.write(z_grid.astype('float32'), 1)
    memfile.seek(0)
    return memfile

def generar_geojson_bytes(df):
    """Crea un archivo GeoJSON en memoria RAM."""
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitud, df.latitud),
        crs="EPSG:4326"
    )
    return gdf.to_json().encode('utf-8')