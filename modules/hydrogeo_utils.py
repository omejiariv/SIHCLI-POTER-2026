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
    
    # Limpieza de Fechas (Anti-Duplicados)
    if 'fecha' in df.columns:
        df['ds'] = pd.to_datetime(df['fecha'])
    elif 'fecha_mes_año' in df.columns:
        df['ds'] = pd.to_datetime(df['fecha_mes_año'])
    
    # Agrupar mensual para evitar errores
    df = df.set_index('ds').resample('MS')['precipitation'].mean().reset_index()
    df.columns = ['fecha', 'p_mes']
    
    # Variables Físicas
    temp = 30 - (0.0065 * altitud)
    # L de Turc ajustado
    l_t = 300 + 25*temp + 0.05*(temp**3)
    if l_t == 0: l_t = 0.001
    
    # Cálculo Balance
    # ETR = P / sqrt(0.9 + (P/L)^2)
    # Nota: Para meses muy lluviosos, Turc mensual puede subestimar ETR, 
    # pero es consistente con tu formula preferida.
    df['etr_mm'] = df['p_mes'] / np.sqrt(0.9 + (df['p_mes'] / (l_t/12))**2) 
    
    df['excedente'] = (df['p_mes'] - df['etr_mm']).clip(lower=0)
    df['recarga_mm'] = df['excedente'] * ki
    df['escorrentia_mm'] = df['excedente'] * (1 - ki)
    
    return df

# --- 2. PROPHET HÍBRIDO (Tu lógica + Mi limpieza) ---
def ejecutar_pronostico_prophet(df_hist, meses_futuros, altitud, ki, ruido=0.0):
    """
    Ejecuta Prophet y recalcula el balance hídrico sobre la predicción.
    """
    try:
        # Preparar DataFrame para Prophet
        df_prophet = df_hist[['fecha', 'p_mes']].rename(columns={'fecha': 'ds', 'p_mes': 'y'})
        
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
        
        # Aplicar ruido/incertidumbre si se solicita
        if ruido > 0:
            df_final['p_final'] = df_final['p_final'] * (1 + np.random.normal(0, 0.05 * ruido, len(df_final)))

        # Recalcular Balance sobre la serie proyectada
        # Reutilizamos la función de Turc
        temp_df = pd.DataFrame({'fecha': df_final['ds'], 'precipitation': df_final['p_final']})
        df_balance = calcular_balance_turc(temp_df, altitud, ki)
        
        # Merge final
        df_result = pd.merge(df_final, df_balance, on='fecha')
        df_result['tipo'] = np.where(df_result['fecha'] <= df_prophet['ds'].max(), 'Histórico', 'Proyección')
        
        return df_result
        
    except Exception as e:
        print(f"Error Prophet: {e}")
        return pd.DataFrame()

# --- 3. CARGA GIS OPTIMIZADA (Tu código) ---
def cargar_capas_gis_optimizadas(engine, bounds=None):
    layers = {}
    if not engine: return layers
    
    tol = 0.001 # Simplificación para velocidad
    
    try:
        with engine.connect() as conn:
            # SUELOS
            q_suelos = text(f"SELECT *, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, {tol})) as gj FROM suelos LIMIT 2000")
            df_s = pd.read_sql(q_suelos, conn)
            if not df_s.empty:
                df_s['geometry'] = df_s['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                layers['suelos'] = gpd.GeoDataFrame(df_s, geometry='geometry', crs="EPSG:4326")
            
            # HIDROGEOLOGIA
            q_hidro = text(f"SELECT *, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, {tol})) as gj FROM zonas_hidrogeologicas LIMIT 2000")
            df_h = pd.read_sql(q_hidro, conn)
            if not df_h.empty:
                df_h['geometry'] = df_h['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                layers['hidro'] = gpd.GeoDataFrame(df_h, geometry='geometry', crs="EPSG:4326")
                
            # BOCATOMAS (Puntos no se simplifican)
            q_boca = text("SELECT *, ST_AsGeoJSON(geom) as gj FROM bocatomas LIMIT 2000")
            df_b = pd.read_sql(q_boca, conn)
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