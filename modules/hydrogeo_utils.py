# modules/hydrogeo_utils.py

import pandas as pd
import numpy as np
from . import analysis, config
import io
import rasterio
from rasterio.transform import from_origin

DEFAULT_KI = 0.15 

def calcular_serie_recarga(df_lluvia, lat, altitud, ki_suelo=None):
    """Calcula balance hídrico con limpieza de fechas extrema."""
    df = df_lluvia.copy()
    
    # 1. Normalización de Columnas
    if config.Config.DATE_COL not in df.columns and 'fecha' in df.columns:
        df = df.rename(columns={'fecha': config.Config.DATE_COL})
    
    # 2. Reconstrucción de Fecha (Aplanadora de duplicados)
    # Convertimos a datetime
    fechas = pd.to_datetime(df[config.Config.DATE_COL])
    # Creamos columna Año-Mes
    df['year'] = fechas.dt.year
    df['month'] = fechas.dt.month
    # Agrupamos por Año-Mes y promediamos la lluvia (elimina cualquier duplicado diario/horario)
    df = df.groupby(['year', 'month'], as_index=False)[config.Config.PRECIPITATION_COL].mean()
    # Reconstruimos la fecha al día 1 del mes
    df[config.Config.DATE_COL] = pd.to_datetime(df[['year', 'month']].assign(DAY=1))
    df = df.sort_values(config.Config.DATE_COL)

    # 3. Física (Hargreaves + Balance)
    temp_media = 28.0 - (0.006 * float(altitud))
    if temp_media < 1: temp_media = 5 # Evitar congelamiento en páramo para ETP

    etp_mensual = temp_media * 4.5 + 10 
    
    df['etp_potencial'] = etp_mensual
    df['etr_mm'] = np.minimum(df[config.Config.PRECIPITATION_COL], df['etp_potencial'])
    df['agua_disponible_mm'] = df[config.Config.PRECIPITATION_COL] - df['etr_mm']
    
    # 4. Infiltración
    ki_final = ki_suelo if pd.notnull(ki_suelo) else DEFAULT_KI
    df['recarga_mm'] = df['agua_disponible_mm'] * ki_final
    df['escorrentia_sup_mm'] = df['agua_disponible_mm'] * (1 - ki_final)
    
    return df[[config.Config.DATE_COL, config.Config.PRECIPITATION_COL, 'etr_mm', 'recarga_mm', 'escorrentia_sup_mm']]

def obtener_datos_estaciones_recarga(engine):
    """
    Obtiene datos usando 'Nearest Neighbor' para garantizar cruce de suelos.
    """
    # 1. METADATOS + SUELOS (Lógica Infalible)
    # Busca el polígono de suelo más cercano (distancia 0 si está dentro)
    q_meta = """
    SELECT 
        e.id_estacion, e.nom_est, e.municipio, e.latitud, e.longitud, e.elevacion,
        s.infiltracion_ki 
    FROM estaciones e
    LEFT JOIN LATERAL (
        SELECT infiltracion_ki 
        FROM suelos s
        ORDER BY e.geom <-> s.geom ASC 
        LIMIT 1
    ) s ON true
    """
    df_meta = pd.read_sql(q_meta, engine)
    
    # 2. LLUVIA HISTÓRICA
    q_lluvia = """
    SELECT id_estacion_fk as id_estacion, AVG(precipitation) as ppt_mes
    FROM precipitacion_mensual 
    GROUP BY id_estacion_fk
    """
    df_lluvia = pd.read_sql(q_lluvia, engine)
    
    # 3. UNIÓN
    df_full = pd.merge(df_meta, df_lluvia, on='id_estacion')
    
    # 4. CÁLCULO
    df_full['temp_est'] = 28.0 - (0.006 * df_full['elevacion'])
    df_full.loc[df_full['temp_est'] < 1, 'temp_est'] = 5
    
    df_full['etp_mes'] = df_full['temp_est'] * 4.5 + 10
    df_full['etr_real_mes'] = np.minimum(df_full['ppt_mes'], df_full['etp_mes'])
    
    # Ki (Si sigue nulo tras el Nearest Neighbor, usamos default)
    df_full['ki_final'] = df_full['infiltracion_ki'].fillna(DEFAULT_KI)
    
    df_full['recarga_mes'] = (df_full['ppt_mes'] - df_full['etr_real_mes']) * df_full['ki_final']
    df_full.loc[df_full['recarga_mes'] < 0, 'recarga_mes'] = 0
    
    # Anuales
    df_full['recarga_anual'] = df_full['recarga_mes'] * 12
    df_full['ppt_anual'] = df_full['ppt_mes'] * 12
    df_full['etr_anual'] = df_full['etr_real_mes'] * 12
    
    return df_full

# --- Helpers ---
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