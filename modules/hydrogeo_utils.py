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

# ---------------------------------------------------------
# 1. MODELO TURC & BALANCE (Sin Cambios)
# ---------------------------------------------------------
def calcular_balance_turc(df_lluvia, altitud, ki):
    df = df_lluvia.copy()
    col_fecha = 'ds'
    if 'fecha_mes_año' in df.columns: col_fecha = 'fecha_mes_año'
    elif 'fecha' in df.columns: col_fecha = 'fecha'
        
    col_p = 'precipitation' if 'precipitation' in df.columns else 'p_mes'
    if 'y' in df.columns: col_p = 'y'

    df['ds'] = pd.to_datetime(df[col_fecha])
    df_monthly = df.set_index('ds').resample('MS')[col_p].mean().reset_index()
    df_monthly.columns = ['fecha', 'p_mes']

    temp = 30 - (0.0065 * float(altitud))
    if temp < 5: temp = 5

    I_t = 300 + 25*temp + 0.05*(temp**3)
    if I_t == 0: I_t = 0.001

    df_monthly['etr_mm'] = df_monthly['p_mes'] / np.sqrt(0.9 + (df_monthly['p_mes'] / (I_t/12))**2)
    df_monthly['excedente'] = (df_monthly['p_mes'] - df_monthly['etr_mm']).clip(lower=0)
    df_monthly['recarga_mm'] = df_monthly['excedente'] * ki
    df_monthly['escorrentia_mm'] = df_monthly['excedente'] * (1 - ki)

    return df_monthly

# ---------------------------------------------------------
# 2. PROPHET HÍBRIDO (Sin Cambios)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def ejecutar_pronostico_prophet(df_hist, meses_futuros, altitud, ki, ruido=0.0):
    try:
        df_work = df_hist.copy()
        if 'fecha_mes_año' in df_work.columns:
            df_work = df_work.rename(columns={'fecha_mes_año': 'ds'})
        elif 'fecha' in df_work.columns:
            df_work = df_work.rename(columns={'fecha': 'ds'})
        else:
            return pd.DataFrame(columns=['tipo', 'p_final', 'recarga_mm', 'etr_mm', 'fecha'])

        col_p = 'precipitation' if 'precipitation' in df_work.columns else 'p_mes'
        df_work = df_work.rename(columns={col_p: 'y'})
        df_prophet = df_work[['ds', 'y']].sort_values('ds')

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

        temp_df = pd.DataFrame({'fecha': df_final['ds'], 'precipitation': df_final['p_final']})
        df_balance = calcular_balance_turc(temp_df, altitud, ki)

        df_result = pd.merge(df_final, df_balance, left_on='ds', right_on='fecha')

        last_date_real = df_prophet['ds'].max()
        df_result['tipo'] = np.where(df_result['ds'] <= last_date_real, 'Histórico', 'Proyección')

        return df_result

    except Exception as e:
        return pd.DataFrame(columns=['tipo', 'p_final', 'recarga_mm', 'etr_mm', 'fecha'])

# ---------------------------------------------------------
# 3. CARGA GIS "FULL DATA" (Trae todas las columnas)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def cargar_capas_gis_optimizadas(_engine, bounds=None):
    layers = {}
    if not _engine: return layers
    
    # Solo definimos la tabla, ya no adivinamos columnas porque traeremos TODO (*)
    config = {
        'suelos': 'suelos',
        'hidro': 'zonas_hidrogeologicas',
        'bocatomas': 'bocatomas'
    }
    
    # PARAMETROS DE RENDIMIENTO
    tol = 0.008 # Simplificación fuerte para velocidad
    limit_poly = 300 
    limit_pts = 1000

    with _engine.connect() as conn:
        for key, tabla in config.items():
            try:
                if not _engine.dialect.has_table(conn, tabla): continue

                # 1. Detectar columnas y geometría
                q_cols = text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{tabla}'")
                cols_reales = pd.read_sql(q_cols, conn)['column_name'].tolist()
                
                col_geom = 'geometry' if 'geometry' in cols_reales else ('geom' if 'geom' in cols_reales else None)
                if not col_geom: continue

                # 2. QUERY "FULL DATA"
                # Seleccionamos TODO (*) excepto la geometría pesada original
                # La geometría la traemos procesada como 'gj'
                cols_select = [c for c in cols_reales if c != col_geom]
                cols_sql = ", ".join([f'"{c}"' for c in cols_select]) # Comillas para proteger nombres
                
                base_q = f"""
                    SELECT {cols_sql}, 
                    ST_AsGeoJSON(ST_SimplifyPreserveTopology(ST_Transform({col_geom}, 4326), {tol})) as gj 
                    FROM {tabla}
                """
                
                final_q = ""
                limite = limit_pts if key == 'bocatomas' else limit_poly
                
                # 3. FILTRO ESPACIAL
                if bounds is not None:
                    try:
                        minx, miny, maxx, maxy = bounds
                        envelope = f"ST_Transform(ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326), ST_SRID({col_geom}))"
                        final_q = f"{base_q} WHERE ST_Intersects({col_geom}, {envelope}) LIMIT {limite}"
                    except: pass
                else:
                    final_q = f"{base_q} LIMIT 50" # Fallback ligero

                # 4. EJECUTAR
                if final_q:
                    df = pd.read_sql(text(final_q), conn)
                    if not df.empty:
                        # Convertir a GeoDataFrame
                        df['geometry'] = df['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                        df = df.dropna(subset=['geometry'])
                        # Eliminamos la columna temporal gj para limpiar
                        if 'gj' in df.columns: df = df.drop(columns=['gj'])
                        
                        layers[key] = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
                        # Normalizar nombres de columnas a minúsculas para facilitar tooltips
                        layers[key].columns = [c.lower() for c in layers[key].columns]
                    
            except Exception as e:
                print(f"Error capa {key}: {e}")

    return layers

# ---------------------------------------------------------
# 4. EXPORTADORES (Sin Cambios)
# ---------------------------------------------------------
def generar_geotiff(z_grid, bounds):
    min_x, min_y, max_x, max_y = bounds
    height, width = z_grid.shape
    transform = from_origin(min_x, max_y, (max_x - min_x) / width, (max_y - min_y) / height)
    mem = io.BytesIO()
    with rasterio.open(mem, 'w', driver='GTiff', height=height, width=width, count=1, dtype='float32', crs="EPSG:4326", transform=transform, nodata=-9999) as dst:
        dst.write(z_grid.astype('float32'), 1)
    mem.seek(0)
    return mem

def generar_geojson_isolines(contour_set):
    features = []
    for i, collection in enumerate(contour_set.collections):
        level = contour_set.levels[i]
        for path in collection.get_paths():
            coords = [list(v) for v in path.vertices]
            if len(coords) > 2:
                features.append({
                    "type": "Feature",
                    "properties": {"recarga_mm": float(level)},
                    "geometry": {"type": "LineString", "coordinates": coords}
                })
    return json.dumps({"type": "FeatureCollection", "features": features})