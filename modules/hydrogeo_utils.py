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
# 1. MODELO TURC Y PROPHET
# ==============================================================================
def calcular_balance_turc(df_lluvia, altitud, ki):
    """Calcula Turc. Respeta NaNs."""
    df = df_lluvia.copy()
    
    # 1. Normalizar nombres
    col_fecha = next((c for c in ['fecha', 'fecha_mes_año', 'ds'] if c in df.columns), None)
    col_p = next((c for c in ['valor', 'precipitation', 'p_mes', 'lluvia'] if c in df.columns), None)
    
    if not col_fecha or not col_p: return df 

    df['ds'] = pd.to_datetime(df[col_fecha])
    
    # 2. Resample Mensual
    df_monthly = df.set_index('ds').resample('MS')[col_p].mean().reset_index()
    df_monthly.columns = ['fecha', 'p_mes']

    # 3. Variables Físicas
    temp = np.maximum(5, 30 - (0.0065 * float(altitud)))
    I_t = 300 + 25*temp + 0.05*(temp**3)
    if I_t == 0: I_t = 0.001

    # 4. Balance (Operaciones vectorizadas)
    denom = np.sqrt(0.9 + (df_monthly['p_mes'] / (I_t/12))**2)
    # Si denom es 0 o NaN, resultado NaN
    df_monthly['etr_mm'] = np.where(denom > 0, df_monthly['p_mes'] / denom, np.nan)
    
    df_monthly['excedente'] = (df_monthly['p_mes'] - df_monthly['etr_mm']).clip(lower=0)
    df_monthly['recarga_mm'] = df_monthly['excedente'] * ki
    df_monthly['escorrentia_mm'] = df_monthly['excedente'] * (1 - ki)

    return df_monthly

@st.cache_data(show_spinner=False)
def ejecutar_pronostico_prophet(df_hist, meses_futuros, altitud, ki, ruido=0.0):
    try:
        df_work = df_hist.copy()
        
        col_fecha = next((c for c in ['fecha', 'fecha_mes_año', 'ds'] if c in df_work.columns), None)
        col_p = next((c for c in ['valor', 'precipitation', 'p_mes'] if c in df_work.columns), None)
        
        # Estructura vacía de retorno por defecto
        cols_retorno = ['tipo', 'p_final', 'recarga_mm', 'etr_mm', 'fecha', 'yhat_lower', 'yhat_upper']
        df_vacio = pd.DataFrame(columns=cols_retorno)

        if not col_fecha or not col_p: return df_vacio

        # Preparar datos (Prophet no admite NaNs en entrenamiento)
        df_prophet = df_work.rename(columns={col_fecha: 'ds', col_p: 'y'})[['ds', 'y']].dropna()
        
        if len(df_prophet) < 6: return df_vacio # Mínimo datos

        # Modelo
        m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=meses_futuros, freq='MS')
        forecast = m.predict(future)

        # Resultados
        df_final = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], df_prophet, on='ds', how='left')
        df_final['p_final'] = df_final['y'].combine_first(df_final['yhat']).clip(lower=0)

        if ruido > 0:
            noise = np.random.normal(0, 0.05 * ruido, len(df_final))
            df_final['p_final'] = df_final['p_final'] * (1 + noise)

        # Calcular Balance completo
        temp_df = pd.DataFrame({'fecha': df_final['ds'], 'valor': df_final['p_final']})
        df_balance = calcular_balance_turc(temp_df, altitud, ki)

        df_result = pd.merge(df_final, df_balance, left_on='ds', right_on='fecha')
        
        last_date_real = df_prophet['ds'].max()
        df_result['tipo'] = np.where(df_result['ds'] <= last_date_real, 'Histórico', 'Proyección')

        return df_result

    except Exception as e:
        print(f"Error Prophet: {e}")
        return pd.DataFrame(columns=['tipo', 'p_final', 'recarga_mm', 'etr_mm', 'fecha'])

# ==============================================================================
# 2. CARGA GIS "FULL DATA" (Arregla Tooltips vacíos)
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=60)
def cargar_capas_gis_optimizadas(_engine, bounds=None):
    layers = {}
    if not _engine: return layers
    
    config = {'suelos': 'suelos', 'hidro': 'zonas_hidrogeologicas', 'bocatomas': 'bocatomas'}
    tol = 0.0005 
    limit_poly = 3000 

    with _engine.connect() as conn:
        for key, tabla in config.items():
            try:
                if not _engine.dialect.has_table(conn, tabla): continue
                
                # 1. Obtener TODAS las columnas
                q_cols = text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{tabla}'")
                cols_reales = pd.read_sql(q_cols, conn)['column_name'].tolist()
                
                col_geom = 'geometry' if 'geometry' in cols_reales else ('geom' if 'geom' in cols_reales else None)
                if not col_geom: continue

                # 2. Seleccionar TODO excepto la geometría binaria pesada
                cols_select = [c for c in cols_reales if c != col_geom]
                cols_sql = ", ".join([f'"{c}"' for c in cols_select])
                
                base_q = f"SELECT {cols_sql}, ST_AsGeoJSON(ST_SimplifyPreserveTopology(ST_Transform({col_geom}, 4326), {tol})) as gj FROM {tabla}"
                final_q = f"{base_q} LIMIT 50"

                if bounds is not None:
                    try:
                        minx, miny, maxx, maxy = bounds
                        envelope = f"ST_Transform(ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326), ST_SRID({col_geom}))"
                        final_q = f"{base_q} WHERE ST_Intersects({col_geom}, {envelope}) LIMIT {limit_poly}"
                    except: pass

                df = pd.read_sql(text(final_q), conn)
                if not df.empty:
                    df['geometry'] = df['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                    df = df.dropna(subset=['geometry'])
                    if 'gj' in df.columns: df = df.drop(columns=['gj'])
                    
                    layers[key] = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
                    # Normalizar columnas a minúsculas
                    layers[key].columns = [c.lower() for c in layers[key].columns]
            except Exception as e:
                print(f"Error capa {key}: {e}")
    return layers

# ==============================================================================
# 3. CÁLCULO ESTADÍSTICO (Arregla Popups N/D y Mapa Recarga)
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=600)
def obtener_estadisticas_estaciones(_engine, df_puntos_snapshot):
    """Calcula estadísticas intentando varias tablas. Respeta NaNs."""
    if df_puntos_snapshot.empty: return df_puntos_snapshot
    
    df_res = df_puntos_snapshot.copy()
    ids = tuple(df_res['id_estacion'].astype(str).tolist())
    
    # Inicializar con NaN
    for col in ['p_media', 'etr_media', 'recarga_calc', 'escorrentia_media', 'std_lluvia']:
        df_res[col] = np.nan

    if not ids: return df_res

    # --- ESTRATEGIA: Intentar tablas hasta encontrar datos ---
    df_stats = pd.DataFrame()
    
    # Lista de intentos: (Tabla, Columna Valor)
    intentos = [
        ('precipitacion', 'valor'),                # 1. Tabla Nueva
        ('precipitacion_mensual', 'precipitation'), # 2. Tabla Vieja
        ('precipitation', 'precipitation')          # 3. Variante inglés
    ]

    with _engine.connect() as conn:
        for tabla, col_val in intentos:
            try:
                if not _engine.dialect.has_table(conn, tabla): continue
                
                # Query Agregada
                q_text = f"""
                    SELECT id_estacion_fk::text as id_estacion, AVG({col_val}) as p_men, STDDEV({col_val}) as p_std 
                    FROM {tabla} 
                    WHERE id_estacion_fk::text IN :ids 
                    GROUP BY id_estacion_fk
                """
                # Ajuste si la columna ID se llama diferente en la tabla nueva
                if tabla == 'precipitacion':
                     q_text = q_text.replace('id_estacion_fk', 'id_estacion')

                if len(ids) == 1:
                     q_single = text(q_text.replace("IN :ids", f" = '{ids[0]}'"))
                     df_temp = pd.read_sql(q_single, conn)
                else:
                     df_temp = pd.read_sql(text(q_text), conn, params={'ids': ids})
                
                if not df_temp.empty:
                    df_stats = df_temp
                    break # ¡Encontramos datos! Salimos del loop
            except Exception:
                continue

    if df_stats.empty: return df_res # Retorna todo NaN si falló todo

    # Merge
    df_res = pd.merge(df_res, df_stats, left_on='id_estacion', right_on='id_estacion', how='left')

    # Cálculos Vectoriales
    p_anual = df_res['p_men'] * 12
    t_media = np.maximum(5, 30 - 0.0065 * df_res['alt_est'].fillna(1500))
    l_t = 300 + 25*t_media + 0.05*(t_media**3)
    denom = np.sqrt(0.9 + (p_anual/l_t)**2)
    
    etr = np.where(denom > 0, p_anual / denom, np.nan)
    excedente = p_anual - etr
    
    # Asignación final
    df_res['p_media'] = df_res['p_men']
    df_res['etr_media'] = etr / 12
    df_res['recarga_calc'] = (excedente * 0.20) / 12
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