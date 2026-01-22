# modules/hydrogeo_utils.py

import pandas as pd
import numpy as np
import io
import rasterio
from rasterio.transform import from_origin
from sqlalchemy import text
from prophet import Prophet
import geopandas as gpd
import json
from shapely.geometry import shape

# --- 1. MODELO TURC & BALANCE (Lógica Física) ---
def calcular_balance_turc(df_lluvia, altitud, ki):
    """
    Calcula el balance hídrico mensual (Lluvia -> ETP -> Recarga).
    """
    df = df_lluvia.copy()
    
    # Estandarizar nombre fecha
    col_fecha = 'fecha' if 'fecha' in df.columns else 'ds'
    if 'fecha_mes_año' in df.columns: col_fecha = 'fecha_mes_año'
    
    df['ds'] = pd.to_datetime(df[col_fecha])
    
    # 1. Agrupar mensual (Solución Error Duplicados Prophet)
    df = df.set_index('ds').resample('MS')['precipitation'].mean().reset_index()
    df.columns = ['fecha', 'p_mes']
    
    # 2. Variables Físicas
    temp = 30 - (0.0065 * float(altitud))
    if temp < 5: temp = 5 # Límite físico
    
    # 3. ETP (Hargreaves Simplificado para datos mensuales)
    # Turc original es anual, para mensual Hargreaves ajustado es mejor
    etp_mes = temp * 4.5 + 10
    
    # 4. Balance
    df['etr_mm'] = np.minimum(df['p_mes'], etp_mes)
    df['excedente'] = (df['p_mes'] - df['etr_mm']).clip(lower=0)
    
    # 5. Infiltración
    df['recarga_mm'] = df['excedente'] * ki
    df['escorrentia_mm'] = df['excedente'] * (1 - ki)
    
    return df

# --- 2. PROPHET HÍBRIDO (Sin errores de índice) ---
def ejecutar_pronostico_prophet(df_hist, meses_futuros, altitud, ki, ruido=0.0):
    try:
        # Preparar
        df_prophet = df_hist.rename(columns={'fecha': 'ds', 'p_mes': 'y'})
        
        # Modelo
        m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
        m.fit(df_prophet)
        
        future = m.make_future_dataframe(periods=meses_futuros, freq='MS')
        forecast = m.predict(future)
        
        # Unir
        df_final = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                            df_prophet, on='ds', how='left')
        
        # Combinar (Llenar huecos futuros con predicción)
        df_final['p_final'] = df_final['y'].combine_first(df_final['yhat']).clip(lower=0)
        
        # Aplicar Balance a la proyección
        temp_df = pd.DataFrame({'fecha': df_final['ds'], 'precipitation': df_final['p_final']})
        df_bal = calcular_balance_turc(temp_df, altitud, ki)
        
        # Merge Final
        df_res = pd.merge(df_final, df_bal, on='fecha')
        df_res['tipo'] = np.where(df_res['fecha'] <= df_prophet['ds'].max(), 'Histórico', 'Proyección')
        
        return df_res
    except Exception as e:
        print(f"Error Prophet: {e}")
        return pd.DataFrame()

# --- 3. CARGA GIS RÁPIDA (Tu código optimizado) ---
def cargar_capas_gis_optimizadas(engine):
    layers = {}
    if not engine: return layers
    tol = 0.001 
    try:
        with engine.connect() as conn:
            # Suelos
            q_s = text(f"SELECT *, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, {tol})) as gj FROM suelos LIMIT 1000")
            df_s = pd.read_sql(q_s, conn)
            if not df_s.empty:
                df_s['geometry'] = df_s['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                layers['suelos'] = gpd.GeoDataFrame(df_s, geometry='geometry', crs="EPSG:4326")
            
            # Hidro
            q_h = text(f"SELECT *, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, {tol})) as gj FROM zonas_hidrogeologicas LIMIT 1000")
            df_h = pd.read_sql(q_h, conn)
            if not df_h.empty:
                df_h['geometry'] = df_h['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                layers['hidro'] = gpd.GeoDataFrame(df_h, geometry='geometry', crs="EPSG:4326")
                
            # Bocatomas
            q_b = text("SELECT *, ST_AsGeoJSON(geom) as gj FROM bocatomas LIMIT 2000")
            df_b = pd.read_sql(q_b, conn)
            if not df_b.empty:
                df_b['geometry'] = df_b['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                layers['bocatomas'] = gpd.GeoDataFrame(df_b, geometry='geometry', crs="EPSG:4326")
    except: pass
    return layers

# --- 4. DESCARGAS ---
def generar_geotiff(z_grid, bounds):
    min_x, min_y, max_x, max_y = bounds
    height, width = z_grid.shape
    transform = from_origin(min_x, max_y, (max_x-min_x)/width, (max_y-min_y)/height)
    mem = io.BytesIO()
    with rasterio.open(mem, 'w', driver='GTiff', height=height, width=width, count=1, dtype='float32', crs="EPSG:4326", transform=transform, nodata=-9999) as dst:
        dst.write(z_grid.astype('float32'), 1)
    mem.seek(0)
    return mem