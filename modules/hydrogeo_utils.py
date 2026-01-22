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

# --- 1. MODELO TURC & BALANCE ---
def calcular_balance_turc(df_lluvia, altitud, ki):
    """
    Calcula el balance hídrico usando el método de Turc mensualizado.
    """
    df = df_lluvia.copy()
    
    # Estandarizar nombre fecha
    col_fecha = 'fecha' if 'fecha' in df.columns else 'ds'
    if 'fecha_mes_año' in df.columns: col_fecha = 'fecha_mes_año'
    
    # Estandarizar nombre precipitación
    col_p = 'precipitation' if 'precipitation' in df.columns else 'p_mes'
    if 'y' in df.columns: col_p = 'y'
    
    df['ds'] = pd.to_datetime(df[col_fecha])
    
    # 1. Agrupar mensual
    df = df.set_index('ds').resample('MS')[col_p].mean().reset_index()
    df.columns = ['fecha', 'p_mes']
    
    # 2. Variables Físicas
    temp = 30 - (0.0065 * float(altitud))
    if temp < 5: temp = 5 
    
    # 3. ETP (Hargreaves Simplificado)
    l_t = 300 + 25*temp + 0.05*(temp**3)
    if l_t == 0: l_t = 0.001
    
    # 4. Balance
    # ETR Turc mensual aproximado
    df['etr_mm'] = df['p_mes'] / np.sqrt(0.9 + (df['p_mes'] / (l_t/12))**2) 
    
    df['excedente'] = (df['p_mes'] - df['etr_mm']).clip(lower=0)
    df['recarga_mm'] = df['excedente'] * ki
    df['escorrentia_mm'] = df['excedente'] * (1 - ki)
    
    return df

# --- 2. PROPHET HÍBRIDO (CORREGIDO) ---
def ejecutar_pronostico_prophet(df_hist, meses_futuros, altitud, ki, ruido=0.0):
    try:
        # CORRECCIÓN DE NOMBRES DE COLUMNA
        # Detectamos qué columna de lluvia viene
        col_p = 'precipitation' if 'precipitation' in df_hist.columns else 'p_mes'
        
        # Preparar DataFrame para Prophet
        df_prophet = df_hist.rename(columns={'fecha': 'ds', col_p: 'y'})
        # Asegurar que las columnas existan
        df_prophet = df_prophet[['ds', 'y']]
        
        # Modelo
        m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
        m.fit(df_prophet)
        
        future = m.make_future_dataframe(periods=meses_futuros, freq='MS')
        forecast = m.predict(future)
        
        # Unir histórico y proyección
        df_final = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                            df_prophet, on='ds', how='left')
        
        # Combinar serie real con predicción donde no hay dato
        df_final['p_final'] = df_final['y'].combine_first(df_final['yhat']).clip(lower=0)
        
        # Aplicar ruido si se solicita
        if ruido > 0:
            noise = np.random.normal(0, 0.05 * ruido, len(df_final))
            df_final['p_final'] = df_final['p_final'] * (1 + noise)

        # Recalcular Balance sobre la serie proyectada
        temp_df = pd.DataFrame({'fecha': df_final['ds'], 'precipitation': df_final['p_final']})
        df_balance = calcular_balance_turc(temp_df, altitud, ki)
        
        # Merge final
        df_result = pd.merge(df_final, df_balance, left_on='ds', right_on='fecha')
        
        # Definir tipo (Histórico vs Proyección)
        last_date = df_prophet['ds'].max()
        df_result['tipo'] = np.where(df_result['ds'] <= last_date, 'Histórico', 'Proyección')
        
        return df_result
        
    except Exception as e:
        print(f"Error Prophet Detallado: {e}")
        # Retornar DataFrame vacío pero con columnas para evitar KeyError en la página
        return pd.DataFrame(columns=['tipo', 'p_final', 'recarga_mm', 'etr_mm', 'fecha'])

# --- 3. CARGA GIS OPTIMIZADA ---
def cargar_capas_gis_optimizadas(engine):
    layers = {}
    if not engine: return layers
    tol = 0.001 
    try:
        with engine.connect() as conn:
            # SUELOS
            q_s = text(f"SELECT *, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, {tol})) as gj FROM suelos LIMIT 1500")
            df_s = pd.read_sql(q_s, conn)
            if not df_s.empty:
                df_s['geometry'] = df_s['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                layers['suelos'] = gpd.GeoDataFrame(df_s, geometry='geometry', crs="EPSG:4326")
            
            # HIDROGEOLOGIA
            q_h = text(f"SELECT *, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, {tol})) as gj FROM zonas_hidrogeologicas LIMIT 1500")
            df_h = pd.read_sql(q_h, conn)
            if not df_h.empty:
                df_h['geometry'] = df_h['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                layers['hidro'] = gpd.GeoDataFrame(df_h, geometry='geometry', crs="EPSG:4326")
                
            # BOCATOMAS
            q_b = text("SELECT *, ST_AsGeoJSON(geom) as gj FROM bocatomas LIMIT 2000")
            df_b = pd.read_sql(q_b, conn)
            if not df_b.empty:
                df_b['geometry'] = df_b['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                layers['bocatomas'] = gpd.GeoDataFrame(df_b, geometry='geometry', crs="EPSG:4326")
                
    except Exception as e:
        print(f"Error GIS: {e}")
        
    return layers

# --- 4. GENERADORES DE DESCARGA ---
def generar_geotiff(z_grid, bounds):
    min_x, min_y, max_x, max_y = bounds
    height, width = z_grid.shape
    transform = from_origin(min_x, max_y, (max_x-min_x)/width, (max_y-min_y)/height)
    mem = io.BytesIO()
    with rasterio.open(mem, 'w', driver='GTiff', height=height, width=width, count=1, dtype='float32', crs="EPSG:4326", transform=transform, nodata=-9999) as dst:
        dst.write(z_grid.astype('float32'), 1)
    mem.seek(0)
    return mem