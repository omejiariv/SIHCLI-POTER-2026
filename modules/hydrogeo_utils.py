# modules/hydrogeo_utils.py

import pandas as pd
import numpy as np
from . import analysis, config
import io
import rasterio
from rasterio.transform import from_origin

# Coeficiente K (Infiltración) por defecto
DEFAULT_KI = 0.15 

def calcular_serie_recarga(df_lluvia, lat, altitud, ki_suelo=None):
    """
    Realiza un Balance Hídrico Mensual Secuencial corregido.
    Garantiza que en meses lluviosos exista excedente (Recarga + Escorrentía).
    """
    df = df_lluvia.copy()
    
    # 1. Limpieza de Fechas
    if config.Config.DATE_COL not in df.columns and 'fecha' in df.columns:
        df = df.rename(columns={'fecha': config.Config.DATE_COL})
    
    df[config.Config.DATE_COL] = pd.to_datetime(df[config.Config.DATE_COL])
    df = df.sort_values(config.Config.DATE_COL)
    df = df.drop_duplicates(subset=[config.Config.DATE_COL])

    # 2. Variable Física: Temperatura Media Estimada (°C)
    # Gradiente térmico andino estándar: 28°C a nivel mar, -0.6°C cada 100m
    temp_media = 28.0 - (0.006 * float(altitud))
    if temp_media < 5: temp_media = 5 # Límite físico páramo

    # 3. Cálculo de ETP (Evapotranspiración Potencial) - Método Hargreaves Simplificado Mensual
    # Factor de radiación extraterrestre (RA) aproximado para trópico (~15 mm/día)
    # ETP estimada ~ 50 - 150 mm/mes dependiendo de T
    # Fórmula empírica robusta: ETP_mes = T_media * K_latitud + Base
    etp_mensual_est = temp_media * 4.5 + 10 # Aproximación calibrada para trópico húmedo
    
    # 4. Balance de Masas (Vectorizado)
    # ETR no puede ser mayor que la lluvia disponible NI mayor que la capacidad de evaporar (ETP)
    df['etp_potencial'] = etp_mensual_est
    
    # ETR Real = min(Lluvia, ETP_Potencial)
    df['etr_mm'] = np.minimum(df[config.Config.PRECIPITATION_COL], df['etp_potencial'])
    
    # Agua Disponible (Excedente) = Lluvia - ETR Real
    # Si llueve 300 y ETP es 100, Excedente es 200.
    df['agua_disponible_mm'] = df[config.Config.PRECIPITATION_COL] - df['etr_mm']
    
    # 5. Repartición del Excedente
    ki_final = ki_suelo if pd.notnull(ki_suelo) else DEFAULT_KI
    
    df['recarga_mm'] = df['agua_disponible_mm'] * ki_final
    df['escorrentia_sup_mm'] = df['agua_disponible_mm'] * (1 - ki_final)
    
    return df[[config.Config.DATE_COL, config.Config.PRECIPITATION_COL, 'etr_mm', 'recarga_mm', 'escorrentia_sup_mm']]

def obtener_datos_estaciones_recarga(engine):
    """
    Calcula Recarga ANUAL para el mapa, aplicando la misma física corregida.
    """
    # A. Metadatos
    q_meta = """
    SELECT e.id_estacion, e.latitud, e.longitud, e.elevacion, s.infiltracion_ki 
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
    
    # D. Cálculo Vectorizado (Misma lógica que arriba)
    df_full['temp_est'] = 28.0 - (0.006 * df_full['elevacion'])
    df_full.loc[df_full['temp_est'] < 5, 'temp_est'] = 5
    
    # ETP Mes
    df_full['etp_mes'] = df_full['temp_est'] * 4.5 + 10
    
    # ETR Real = min(PPT, ETP)
    df_full['etr_real'] = np.minimum(df_full['ppt_mes'], df_full['etp_mes'])
    
    # Excedente
    df_full['excedente'] = df_full['ppt_mes'] - df_full['etr_real']
    
    # Recarga
    df_full['ki_final'] = df_full['infiltracion_ki'].fillna(DEFAULT_KI)
    df_full['recarga_mes'] = df_full['excedente'] * df_full['ki_final']
    
    # Anualizar
    df_full['recarga_anual'] = df_full['recarga_mes'] * 12
    
    return df_full

def calcular_calidad_datos(df, start_date, end_date):
    """Calcula porcentaje de completitud de la serie."""
    if df.empty or not start_date or not end_date:
        return 0, 0, 0
    
    # Total meses teóricos
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    total_records = len(df)
    
    completeness = (total_records / total_months) * 100 if total_months > 0 else 0
    return total_records, total_months, min(100.0, completeness)

# --- Generadores de Archivos (Sin cambios) ---
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