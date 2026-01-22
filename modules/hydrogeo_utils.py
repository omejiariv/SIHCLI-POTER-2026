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

# ---------------------------------------------------------
# 1. MODELO TURC & BALANCE (Lógica Física)
# ---------------------------------------------------------
def calcular_balance_turc(df_lluvia, altitud, ki):
    """
    Calcula el balance hídrico usando el método de Turc mensualizado.
    """
    df = df_lluvia.copy()
    
    # ✅ CORRECCIÓN: Detección inteligente de la columna de fecha
    col_fecha = 'ds' # Default
    if 'fecha_mes_año' in df.columns:
        col_fecha = 'fecha_mes_año'
    elif 'fecha' in df.columns:
        col_fecha = 'fecha'
        
    # ✅ CORRECCIÓN: Detección de precipitación
    col_p = 'precipitation' if 'precipitation' in df.columns else 'p_mes'
    if 'y' in df.columns: col_p = 'y' # Si viene de Prophet

    # Aseguramos formato fecha
    df['ds'] = pd.to_datetime(df[col_fecha])

    # 1. Agrupar mensual (Por si vienen datos diarios)
    # Usamos 'ds' como índice temporal
    df_monthly = df.set_index('ds').resample('MS')[col_p].mean().reset_index()
    df_monthly.columns = ['fecha', 'p_mes'] # Estandarizamos nombres internos

    # 2. Variables Físicas
    # Estimación de temperatura media basada en altitud (Gradiente térmico)
    temp = 30 - (0.0065 * float(altitud))
    if temp < 5: temp = 5 # Límite físico inferior

    # 3. ETP (Hargreaves Simplificado / Turc)
    # I_t: Índice térmico anual aproximado
    I_t = 300 + 25*temp + 0.05*(temp**3)
    if I_t == 0: I_t = 0.001

    # 4. Balance
    # ETR Turc mensual aproximado
    # La fórmula original es anual, aquí se usa una adaptación para paso mensual
    df_monthly['etr_mm'] = df_monthly['p_mes'] / np.sqrt(0.9 + (df_monthly['p_mes'] / (I_t/12))**2)
    
    # Excedente (Lluvia - ETR)
    df_monthly['excedente'] = (df_monthly['p_mes'] - df_monthly['etr_mm']).clip(lower=0)
    
    # Recarga (Infiltración efectiva)
    df_monthly['recarga_mm'] = df_monthly['excedente'] * ki
    
    # Escorrentía superficial
    df_monthly['escorrentia_mm'] = df_monthly['excedente'] * (1 - ki)

    return df_monthly

# ---------------------------------------------------------
# 2. PROPHET HÍBRIDO (Motor de Pronóstico)
# ---------------------------------------------------------
def ejecutar_pronostico_prophet(df_hist, meses_futuros, altitud, ki, ruido=0.0):
    try:
        # ✅ CORRECCIÓN CRÍTICA: Estandarización de nombres antes de entrar a Prophet
        df_work = df_hist.copy()

        # 1. Buscar columna de FECHA (Soporte para 'fecha_mes_año')
        if 'fecha_mes_año' in df_work.columns:
            df_work = df_work.rename(columns={'fecha_mes_año': 'ds'})
        elif 'fecha' in df_work.columns:
            df_work = df_work.rename(columns={'fecha': 'ds'})
        else:
            raise ValueError(f"Columna de fecha no encontrada. Columnas disponibles: {list(df_work.columns)}")

        # 2. Buscar columna de PRECIPITACIÓN
        col_p = 'precipitation' if 'precipitation' in df_work.columns else 'p_mes'
        df_work = df_work.rename(columns={col_p: 'y'})

        # Seleccionar solo lo necesario para Prophet
        df_prophet = df_work[['ds', 'y']].sort_values('ds')

        # Modelo Prophet
        m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
        m.fit(df_prophet)

        # Crear futuro
        future = m.make_future_dataframe(periods=meses_futuros, freq='MS')
        forecast = m.predict(future)

        # Unir histórico y proyección
        df_final = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                            df_prophet, on='ds', how='left')

        # Combinar serie real (y) con predicción (yhat) donde no hay dato real
        df_final['p_final'] = df_final['y'].combine_first(df_final['yhat']).clip(lower=0)

        # Aplicar incertidumbre aleatoria (Ruido) si se solicita
        if ruido > 0:
            noise = np.random.normal(0, 0.05 * ruido, len(df_final))
            df_final['p_final'] = df_final['p_final'] * (1 + noise)

        # Recalcular Balance Hídrico (Turc) sobre toda la serie (Histórica + Futura)
        temp_df = pd.DataFrame({'fecha': df_final['ds'], 'precipitation': df_final['p_final']})
        df_balance = calcular_balance_turc(temp_df, altitud, ki)

        # Merge final para tener fechas, proyecciones y balance físico juntos
        df_result = pd.merge(df_final, df_balance, left_on='ds', right_on='fecha')

        # Definir etiqueta 'tipo' (Histórico vs Proyección)
        last_date_real = df_prophet['ds'].max()
        df_result['tipo'] = np.where(df_result['ds'] <= last_date_real, 'Histórico', 'Proyección')

        return df_result

    except Exception as e:
        print(f"❌ Error en Prophet: {e}")
        # Retornar DataFrame vacío pero con estructura válida para evitar crash en frontend
        return pd.DataFrame(columns=['tipo', 'p_final', 'recarga_mm', 'etr_mm', 'fecha'])

# ---------------------------------------------------------
# 3. CARGA GIS OPTIMIZADA
# ---------------------------------------------------------

def cargar_capas_gis_optimizadas(engine, bounds=None):
    """
    Carga capas GIS recortadas espacialmente al área de interés para evitar colapsos de memoria.
    bounds: tupla (minx, miny, maxx, maxy)
    """
    layers = {}
    if not engine: return layers
    
    # Tolerancia de simplificación (Más alto = polígonos más ligeros)
    tol = 0.003 
    
    # Construir cláusula WHERE espacial
    where_clause = ""
    if bounds:
        minx, miny, maxx, maxy = bounds
        # Agregamos un pequeño buffer (pad) para que no se corte feo en los bordes
        pad = 0.02
        # ST_MakeEnvelope crea un rectángulo con las coordenadas
        where_clause = f"WHERE ST_Intersects(geom, ST_MakeEnvelope({minx-pad}, {miny-pad}, {maxx+pad}, {maxy+pad}, 4326))"
    
    try:
        with engine.connect() as conn:
            # SUELOS - Solo traemos lo que intersecta la vista
            # Limitamos a 500 polígonos por seguridad
            q_s = text(f"""
                SELECT codigo, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, {tol})) as gj 
                FROM suelos 
                {where_clause}
                LIMIT 500
            """)
            df_s = pd.read_sql(q_s, conn)
            if not df_s.empty:
                df_s['geometry'] = df_s['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                layers['suelos'] = gpd.GeoDataFrame(df_s, geometry='geometry', crs="EPSG:4326")

            # HIDROGEOLOGIA
            q_h = text(f"""
                SELECT tipo, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, {tol})) as gj 
                FROM zonas_hidrogeologicas 
                {where_clause}
                LIMIT 500
            """)
            df_h = pd.read_sql(q_h, conn)
            if not df_h.empty:
                df_h['geometry'] = df_h['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                layers['hidro'] = gpd.GeoDataFrame(df_h, geometry='geometry', crs="EPSG:4326")
            
            # BOCATOMAS (Puntos son ligeros, podemos traer más o filtrar igual)
            q_b = text(f"SELECT nom_bocatoma, ST_AsGeoJSON(geom) as gj FROM bocatomas {where_clause} LIMIT 500")
            df_b = pd.read_sql(q_b, conn)
            if not df_b.empty:
                df_b['geometry'] = df_b['gj'].apply(lambda x: shape(json.loads(x)) if x else None)
                layers['bocatomas'] = gpd.GeoDataFrame(df_b, geometry='geometry', crs="EPSG:4326")
                
    except Exception as e:
        print(f"⚠️ Error GIS (Optimizado): {e}")
        
    return layers

# ---------------------------------------------------------
# 4. GENERADORES DE DESCARGA
# ---------------------------------------------------------
def generar_geotiff(z_grid, bounds):
    """
    Genera un archivo GeoTIFF en memoria a partir de una matriz numpy.
    """
    min_x, min_y, max_x, max_y = bounds
    height, width = z_grid.shape
    
    # Definir transformación afín (coordenadas -> píxeles)
    transform = from_origin(min_x, max_y, (max_x - min_x) / width, (max_y - min_y) / height)
    
    mem = io.BytesIO()
    with rasterio.open(
        mem, 'w', driver='GTiff',
        height=height, width=width,
        count=1, dtype='float32',
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999
    ) as dst:
        dst.write(z_grid.astype('float32'), 1)
        
    mem.seek(0)
    return mem