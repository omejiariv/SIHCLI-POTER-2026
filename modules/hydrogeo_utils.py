# modules/hydrogeo_utils.py

import pandas as pd
import numpy as np
from . import analysis, config
import io
import rasterio
from rasterio.transform import from_origin

# --- 1. LÓGICA DE BALANCE HÍDRICO (SERIES) ---
def calcular_serie_recarga(df_lluvia, lat, altitud, ki_suelo=None):
    """
    Calcula el balance hídrico mensual secuencial (Lluvia -> ETP -> ETR -> Recarga).
    """
    df = df_lluvia.copy()
    
    # Normalización de fechas
    if config.Config.DATE_COL not in df.columns and 'fecha' in df.columns:
        df = df.rename(columns={'fecha': config.Config.DATE_COL})
    
    df[config.Config.DATE_COL] = pd.to_datetime(df[config.Config.DATE_COL])
    
    # --- CURA PARA EL ERROR DE PROPHET ---
    # Agrupamos por mes exacto para eliminar cualquier duplicado de base de datos
    df = df.groupby(pd.Grouper(key=config.Config.DATE_COL, freq='MS')).mean().reset_index()
    df = df.sort_values(config.Config.DATE_COL)

    # Variables Físicas
    temp_media = 28.0 - (0.006 * float(altitud))
    if temp_media < 5: temp_media = 5

    # ETP (Hargreaves Simplificado)
    etp_mensual = temp_media * 4.5 + 10 
    
    # Balance
    df['etp_potencial'] = etp_mensual
    df['etr_mm'] = np.minimum(df[config.Config.PRECIPITATION_COL], df['etp_potencial'])
    df['agua_disponible_mm'] = df[config.Config.PRECIPITATION_COL] - df['etr_mm']
    
    # Infiltración vs Escorrentía
    ki_final = ki_suelo if pd.notnull(ki_suelo) else 0.15
    df['recarga_mm'] = df['agua_disponible_mm'] * ki_final
    df['escorrentia_sup_mm'] = df['agua_disponible_mm'] * (1 - ki_final)
    
    return df[[config.Config.DATE_COL, config.Config.PRECIPITATION_COL, 'etr_mm', 'recarga_mm', 'escorrentia_sup_mm']]

# --- 2. DATOS PARA EL MAPA (ENRIQUECIDO) ---
def obtener_datos_estaciones_recarga(engine):
    """
    Calcula promedios anuales para el mapa e incluye metadatos para Popups.
    """
    # A. Metadatos (Incluimos Nombre y Municipio)
    q_meta = """
    SELECT e.id_estacion, e.nom_est, e.municipio, e.latitud, e.longitud, e.elevacion, s.infiltracion_ki 
    FROM estaciones e 
    LEFT JOIN suelos s ON ST_Intersects(e.geom, s.geom)
    """
    df_meta = pd.read_sql(q_meta, engine)
    
    # B. Lluvia Promedio Mensual
    q_lluvia = """
    SELECT id_estacion_fk as id_estacion, AVG(precipitation) as ppt_mes
    FROM precipitacion_mensual 
    GROUP BY id_estacion_fk
    """
    df_lluvia = pd.read_sql(q_lluvia, engine)
    
    # C. Merge
    df_full = pd.merge(df_meta, df_lluvia, on='id_estacion')
    
    # D. Cálculo Masivo
    df_full['temp_est'] = 28.0 - (0.006 * df_full['elevacion'])
    df_full.loc[df_full['temp_est'] < 5, 'temp_est'] = 5
    
    # ETP Mes
    df_full['etp_mes'] = df_full['temp_est'] * 4.5 + 10
    
    # ETR Real (promedio mes)
    df_full['etr_real_mes'] = np.minimum(df_full['ppt_mes'], df_full['etp_mes'])
    
    # Recarga Mes
    df_full['ki_final'] = df_full['infiltracion_ki'].fillna(0.15)
    df_full['recarga_mes'] = (df_full['ppt_mes'] - df_full['etr_real_mes']) * df_full['ki_final']
    df_full.loc[df_full['recarga_mes'] < 0, 'recarga_mes'] = 0
    
    # --- VARIABLES ANUALES PARA POPUP ---
    df_full['recarga_anual'] = df_full['recarga_mes'] * 12
    df_full['ppt_anual'] = df_full['ppt_mes'] * 12
    df_full['etr_anual'] = df_full['etr_real_mes'] * 12
    
    # Redondeo para estética
    df_full = df_full.round({'recarga_anual': 0, 'ppt_anual': 0, 'etr_anual': 0, 'elevacion': 0})
    
    return df_full

def calcular_calidad_datos(df, start_date, end_date):
    if df.empty or not start_date or not end_date: return 0, 0, 0
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    total_records = len(df)
    completeness = (total_records / total_months) * 100 if total_months > 0 else 0
    return total_records, total_months, min(100.0, completeness)

def generar_geotiff_bytes(z_grid, bounds, crs_code=4326):
    min_x, min_y, max_x, max_y = bounds
    height, width = z_grid.shape
    pixel_width = (max_x - min_x) / width
    pixel_height = (max_y - min_y) / height 
    transform = from_origin(min_x, max_y, pixel_width, pixel_height)
    memfile = io.BytesIO()
    with rasterio.open(memfile, 'w', driver='GTiff', height=height, width=width, count=1, dtype='float32', crs=f"EPSG:{crs_code}", transform=transform, nodata=-9999) as dst:
        dst.write(z_grid.astype('float32'), 1)
    memfile.seek(0)
    return memfile

def generar_geojson_bytes(df):
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitud, df.latitud), crs="EPSG:4326")
    return gdf.to_json().encode('utf-8')