# modules/hydrogeo_utils.py

import pandas as pd
import numpy as np
from . import analysis, config

# Configuración de Infiltración por defecto (Fallback)
DEFAULT_KI = 0.15 

def calcular_serie_recarga(df_lluvia, lat, altitud, ki_suelo=None):
    """
    Calcula la serie temporal completa de Recarga usando el Balance de Turc (analysis.py).
    
    Retorna: DataFrame con columnas [fecha, precipitation, etr, escorrentia, recarga]
    """
    df = df_lluvia.copy()
    
    # 1. Validar fechas
    if config.Config.DATE_COL not in df.columns and 'fecha' in df.columns:
        df = df.rename(columns={'fecha': config.Config.DATE_COL})
    
    df[config.Config.DATE_COL] = pd.to_datetime(df[config.Config.DATE_COL])
    df = df.sort_values(config.Config.DATE_COL)

    # 2. Estimar Temperatura (usando tu función de analysis.py)
    # Si la estación no tiene sensor de temperatura, la estimamos por gradiente altitudinal
    temp_estimada = analysis.estimate_temperature(altitud)
    
    # 3. Aplicar Balance Hídrico (Turc) fila por fila
    # analysis.calculate_water_balance_turc retorna (ETR, Excedente/Escorrentía_Potencial)
    # Nota: Tu función original devuelve (etr, q), donde q es el agua disponible tras evaporarse.
    
    resultados = df[config.Config.PRECIPITATION_COL].apply(
        lambda p: analysis.calculate_water_balance_turc(p, temp_estimada)
    )
    
    # Desempaquetar resultados
    df['etr_mm'] = [x[0] for x in resultados]
    df['agua_disponible_mm'] = [x[1] for x in resultados] # Esto es (P - ETR)
    
    # 4. Separación Recarga vs Escorrentía Superficial
    # Usamos el Ki (Coeficiente de Infiltración) de la base de datos o un default
    factor_inf = ki_suelo if pd.notnull(ki_suelo) else DEFAULT_KI
    
    df['recarga_mm'] = df['agua_disponible_mm'] * factor_inf
    df['escorrentia_sup_mm'] = df['agua_disponible_mm'] * (1 - factor_inf)
    
    # Limpieza final
    return df[[config.Config.DATE_COL, config.Config.PRECIPITATION_COL, 'etr_mm', 'recarga_mm', 'escorrentia_sup_mm']]

def obtener_datos_estaciones_recarga(engine):
    """
    Obtiene un GeoDataFrame con la recarga PROMEDIO calculada para TODAS las estaciones.
    Vital para el mapa de interpolación.
    """
    # 1. Traer estaciones y sus parámetros de suelo
    q_meta = """
    SELECT e.id_estacion, e.nom_est, e.latitud, e.longitud, e.elevacion, s.infiltracion_ki 
    FROM estaciones e 
    LEFT JOIN suelos s ON ST_Intersects(e.geom, s.geom)
    """
    df_meta = pd.read_sql(q_meta, engine)
    
    # 2. Traer lluvia promedio histórica (agregada en SQL para rapidez)
    q_lluvia = """
    SELECT id_estacion_fk as id_estacion, AVG(precipitation) as ppt_media
    FROM precipitacion_mensual 
    GROUP BY id_estacion_fk
    """
    df_lluvia = pd.read_sql(q_lluvia, engine)
    
    # 3. Cruzar y calcular Recarga Promedio Puntual
    df_full = pd.merge(df_meta, df_lluvia, on='id_estacion')
    
    # Cálculo vectorizado aproximado para el mapa (Turc simplificado con T media)
    # T = 28 - 0.006*h
    df_full['temp_est'] = 28.0 - (0.006 * df_full['elevacion'])
    
    # Turc vectorizado (L)
    L = 300 + 25 * df_full['temp_est'] + 0.05 * (df_full['temp_est']**3)
    denom = np.sqrt(0.9 + (df_full['ppt_media'] / L)**2)
    df_full['etr_est'] = df_full['ppt_media'] / denom
    
    # Recarga = (P - ETR) * Ki
    # Rellenar Ki nulos con 0.15
    df_full['ki_final'] = df_full['infiltracion_ki'].fillna(0.15)
    df_full['recarga_media'] = (df_full['ppt_media'] - df_full['etr_est']) * df_full['ki_final']
    
    # Limpiar negativos (si P < ETR)
    df_full.loc[df_full['recarga_media'] < 0, 'recarga_media'] = 0
    
    return df_full

# ... importaciones existentes ...
import io
import rasterio
from rasterio.transform import from_origin

def generar_geotiff_bytes(z_grid, bounds, crs_code=4326):
    """
    Convierte una matriz numpy (grid) en un archivo GeoTIFF en memoria.
    bounds: [min_lon, min_lat, max_lon, max_lat]
    """
    min_x, min_y, max_x, max_y = bounds
    height, width = z_grid.shape
    
    # Calcular tamaño de pixel
    pixel_width = (max_x - min_x) / width
    pixel_height = (max_y - min_y) / height # Usualmente negativo en transform
    
    # Definir transformación (Ojo: Rasterio usa origen arriba-izquierda)
    transform = from_origin(min_x, max_y, pixel_width, pixel_height)
    
    memfile = io.BytesIO()
    with rasterio.open(
        memfile, 'w', driver='GTiff',
        height=height, width=width,
        count=1, dtype=str(z_grid.dtype),
        crs=f"EPSG:{crs_code}",
        transform=transform,
        nodata=-9999
    ) as dst:
        dst.write(z_grid, 1)
        
    memfile.seek(0)
    return memfile

def generar_geojson_bytes(df_estaciones):
    """Convierte el DataFrame de estaciones a GeoJSON bytes."""
    # Asumiendo que df_estaciones tiene lat/lon
    import geopandas as gpd
    
    if 'geometry' not in df_estaciones.columns:
        gdf = gpd.GeoDataFrame(
            df_estaciones, 
            geometry=gpd.points_from_xy(df_estaciones.longitud, df_estaciones.latitud),
            crs="EPSG:4326"
        )
    else:
        gdf = df_estaciones
        
    return gdf.to_json().encode('utf-8')