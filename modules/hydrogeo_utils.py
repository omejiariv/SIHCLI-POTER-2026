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
import streamlit as st

# ==============================================================================
# 1. MODELO TURC Y PROPHET (Lógica Base)
# ==============================================================================

def calcular_balance_turc(df_lluvia, altitud, ki):
    """Calcula el balance hídrico mensualizado según el método de Turc."""
    df = df_lluvia.copy()
    
    # Normalización de nombres de columnas
    col_fecha = 'ds'
    if 'fecha_mes_año' in df.columns: col_fecha = 'fecha_mes_año'
    elif 'fecha' in df.columns: col_fecha = 'fecha'
        
    col_p = 'precipitation' if 'precipitation' in df.columns else 'p_mes'
    if 'y' in df.columns: col_p = 'y'
    if 'valor' in df.columns: col_p = 'valor' # Soporte para tabla 'precipitacion'

    df['ds'] = pd.to_datetime(df[col_fecha])
    
    # Agrupar por mes (Resample)
    df_monthly = df.set_index('ds').resample('MS')[col_p].mean().reset_index()
    df_monthly.columns = ['fecha', 'p_mes']

    # Variables Físicas
    temp = 30 - (0.0065 * float(altitud))
    if temp < 5: temp = 5

    # Evapotranspiración Potencial
    I_t = 300 + 25*temp + 0.05*(temp**3)
    if I_t == 0: I_t = 0.001

    # Balance
    df_monthly['etr_mm'] = df_monthly['p_mes'] / np.sqrt(0.9 + (df_monthly['p_mes'] / (I_t/12))**2)
    df_monthly['excedente'] = (df_monthly['p_mes'] - df_monthly['etr_mm']).clip(lower=0)
    df_monthly['recarga_mm'] = df_monthly['excedente'] * ki
    df_monthly['escorrentia_mm'] = df_monthly['excedente'] * (1 - ki)

    return df_monthly

@st.cache_data(show_spinner=False)
def ejecutar_pronostico_prophet(df_hist, meses_futuros, altitud, ki, ruido=0.0):
    try:
        df_work = df_hist.copy()
        # Normalización
        if 'fecha_mes_año' in df_work.columns: df_work = df_work.rename(columns={'fecha_mes_año': 'ds'})
        elif 'fecha' in df_work.columns: df_work = df_work.rename(columns={'fecha': 'ds'})
        else: return pd.DataFrame(columns=['tipo', 'p_final', 'recarga_mm', 'etr_mm', 'fecha'])

        col_p = 'precipitation'
        if 'p_mes' in df_work.columns: col_p = 'p_mes'
        elif 'valor' in df_work.columns: col_p = 'valor'
        
        df_work = df_work.rename(columns={col_p: 'y'})
        df_prophet = df_work[['ds', 'y']].sort_values('ds')

        # Modelo
        m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
        m.fit(df_prophet)

        future = m.make_future_dataframe(periods=meses_futuros, freq='MS')
        forecast = m.predict(future)

        df_final = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                            df_prophet, on='ds', how='left')

        df_final['p_final'] = df_final['y'].combine_first(df_final['yhat']).clip(lower=0)

        if ruido > 0:
            noise = np.random.normal(0, 0.05 * ruido, len(df_final))
            df_final['p_final'] = df_final['p_final'] * (1 + noise)

        # Recalcular balance sobre la serie completa (historia + futuro)
        temp_df = pd.DataFrame({'fecha': df_final['ds'], 'precipitation': df_final['p_final']})
        df_balance = calcular_balance_turc(temp_df, altitud, ki)

        df_result = pd.merge(df_final, df_balance, left_on='ds', right_on='fecha')

        last_date_real = df_prophet['ds'].max()
        df_result['tipo'] = np.where(df_result['ds'] <= last_date_real, 'Histórico', 'Proyección')

        return df_result

    except Exception as e:
        print(f"Error Prophet: {e}")
        return pd.DataFrame(columns=['tipo', 'p_final', 'recarga_mm', 'etr_mm', 'fecha'])

# ==============================================================================
# 2. CARGA GIS OPTIMIZADA (Selectivo y Ligero)
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=60)
def cargar_capas_gis_optimizadas(_engine, bounds=None):
    layers = {}
    if not _engine: return layers
    
    config = {
        'suelos': 'suelos',
        'hidro': 'zonas_hidrogeologicas',
        'bocatomas': 'bocatomas'
    }
    
    # Tolerancia muy fina para no perder polígonos pequeños
    tol = 0.0005 
    limit_poly = 2000 

    with _engine.connect() as conn:
        for key, tabla in config.items():
            try:
                if not _engine.dialect.has_table(conn, tabla): continue

                # 1. Detectar columnas reales
                q_cols = text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{tabla}'")
                cols_reales = pd.read_sql(q_cols, conn)['column_name'].tolist()
                
                col_geom = 'geometry' if 'geometry' in cols_reales else ('geom' if 'geom' in cols_reales else None)
                if not col_geom: continue

                # 2. SELECT INTELIGENTE (Excluye la geometría pesada binaria)
                # Esto es clave para que 'litologia' y 'caracteri' viajen rápido
                cols_select = [c for c in cols_reales if c != col_geom]
                cols_sql = ", ".join([f'"{c}"' for c in cols_select])
                
                base_q = f"""
                    SELECT {cols_sql}, 
                    ST_AsGeoJSON(ST_SimplifyPreserveTopology(ST_Transform({col_geom}, 4326), {tol})) as gj 
                    FROM {tabla}
                """
                
                final_q = ""
                # Filtro Espacial
                if bounds is not None:
                    try:
                        minx, miny, maxx, maxy = bounds
                        envelope = f"ST_Transform(ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326), ST_SRID({col_geom}))"
                        final_q = f"{base_q} WHERE ST_Intersects({col_geom}, {envelope}) LIMIT {limit_poly}"
                    except: pass
                else:
                    final_q = f"{base_q} LIMIT 50"

                if final_q:
                    df = pd.read_sql(text(final_q), conn)
                    if not df.empty:
                        df['geometry'] = df['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                        df = df.dropna(subset=['geometry'])
                        if 'gj' in df.columns: df = df.drop(columns=['gj'])
                        
                        layers[key] = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
                        # Normalizar a minúsculas para facilitar tooltips
                        layers[key].columns = [c.lower() for c in layers[key].columns]
                    
            except Exception as e:
                print(f"Error capa {key}: {e}")

    return layers

# ==============================================================================
# 3. CÁLCULO ESTADÍSTICO CACHEADO (Solución a la inestabilidad)
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=600)
def obtener_estadisticas_estaciones(_engine, df_puntos_snapshot):
    """Calcula ETR, Recarga y Lluvia media para todas las estaciones sin bloquear la UI."""
    if df_puntos_snapshot.empty: return df_puntos_snapshot
    
    df_res = df_puntos_snapshot.copy()
    ids = tuple(df_res['id_estacion'].astype(str).tolist())
    
    if not ids: return df_res

    # 1. Detectar tabla de lluvia (precipitacion vs precipitation)
    tabla_lluvia = "precipitacion"
    col_valor = "valor"
    
    # Query de prueba para detectar tabla
    try:
        with _engine.connect() as conn:
            if not _engine.dialect.has_table(conn, tabla_lluvia):
                tabla_lluvia = "precipitation"
                col_valor = "precipitation"
            
            # Query real
            q_text = f"""
                SELECT id_estacion::text, AVG({col_valor}) as p_men, STDDEV({col_valor}) as p_std 
                FROM {tabla_lluvia} 
                WHERE id_estacion::text IN :ids 
                GROUP BY id_estacion
            """
            
            if len(ids) == 1:
                 # Ajuste para tupla unitaria
                 q_single = text(q_text.replace("IN :ids", f" = '{ids[0]}'"))
                 df_stats = pd.read_sql(q_single, conn)
            else:
                 df_stats = pd.read_sql(text(q_text), conn, params={'ids': ids})
                 
    except Exception as e:
        print(f"Error Stats DB: {e}")
        return df_res

    if df_stats.empty: return df_res

    # 2. Merge de datos
    df_res = pd.merge(df_res, df_stats, left_on='id_estacion', right_on='id_estacion', how='left').fillna(0)

    # 3. Cálculo Vectorial (Rápido)
    p_anual = df_res['p_men'] * 12
    t_media = np.maximum(5, 30 - 0.0065 * df_res['alt_est'])
    
    l_t = 300 + 25*t_media + 0.05*(t_media**3)
    denom = np.sqrt(0.9 + (p_anual/l_t)**2)
    etr = np.where(denom > 0, p_anual / denom, 0)
    
    excedente = p_anual - etr
    
    # Asignar resultados
    df_res['p_media'] = df_res['p_men']
    df_res['etr_media'] = etr / 12
    df_res['recarga_calc'] = (excedente * 0.20) / 12  # Asumiendo Ki promedio 0.2 si no hay dato
    df_res['escorrentia_media'] = (excedente * 0.80) / 12
    df_res['std_lluvia'] = df_res['p_std']

    return df_res

# ==============================================================================
# 4. EXPORTADORES
# ==============================================================================
def generar_geotiff(z_grid, bounds):
    min_x, min_y, max_x, max_y = bounds
    height, width = z_grid.shape
    transform = from_origin(min_x, max_y, (max_x - min_x) / width, (max_y - min_y) / height)
    mem = io.BytesIO()
    with rasterio.open(mem, 'w', driver='GTiff', height=height, width=width, count=1, dtype='float32', crs="EPSG:4326", transform=transform, nodata=-9999) as dst:
        dst.write(z_grid.astype('float32'), 1)
    mem.seek(0)
    return mem