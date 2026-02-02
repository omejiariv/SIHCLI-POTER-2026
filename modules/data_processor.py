# modules/data_processor.py

import geopandas as gpd
import pandas as pd
import streamlit as st
from shapely import wkt
from sqlalchemy import text
from modules.config import Config
from modules.db_manager import get_engine

# --- UTILIDADES ---
def parse_spanish_date_robust(x):
    """Utilidad para fechas en español (legacy support)."""
    if isinstance(x, pd.Timestamp): return x
    if pd.isna(x) or x == "": return pd.NaT
    x = str(x).lower().strip()
    trans = {"ene":"Jan", "feb":"Feb", "mar":"Mar", "abr":"Apr", "may":"May", "jun":"Jun", 
             "jul":"Jul", "ago":"Aug", "sep":"Sep", "oct":"Oct", "nov":"Nov", "dic":"Dec"}
    for es, en in trans.items():
        if es in x: x = x.replace(es, en); break
    try: return pd.to_datetime(x, format="%b-%y")
    except: return pd.to_datetime(x, errors='coerce')

def complete_series(df):
    """Interpolación lineal usando nombres de Config."""
    if df is None or df.empty: return df
    df = df.sort_values(Config.DATE_COL)
    if Config.PRECIPITATION_COL in df.columns:
        df[Config.PRECIPITATION_COL] = df[Config.PRECIPITATION_COL].interpolate(method="linear", limit_direction="both")
    return df

# --- FUNCIONES SQL OPTIMIZADAS ---

def get_lista_estaciones_simple():
    """Trae lista para el selector usando la TABLA NUEVA."""
    engine = get_engine()
    if not engine: return []
    try:
        with engine.connect() as conn:
            # CAMBIO: nom_est -> nombre
            query = text("SELECT id_estacion, nombre, municipio FROM estaciones ORDER BY nombre ASC")
            df = pd.read_sql(query, conn)
            df['label'] = df.apply(lambda x: f"{x['nombre']} [{x['id_estacion']}]", axis=1)
            return df['label'].tolist()
    except Exception as e:
        st.error(f"Error cargando lista: {e}")
        return []

def get_datos_estacion_individual(station_id):
    """Trae datos de lluvia de la TABLA NUEVA."""
    engine = get_engine()
    if not engine: return pd.DataFrame()
    try:
        with engine.connect() as conn:
            # CAMBIO: precipitacion_mensual -> precipitacion
            # CAMBIO: fecha_mes_año -> fecha, precipitation -> valor
            query = text("""
                SELECT fecha, valor 
                FROM precipitacion 
                WHERE id_estacion = :id 
                ORDER BY fecha ASC
            """)
            df = pd.read_sql(query, conn, params={"id": station_id})
            
            # Renombrar para coincidir con Config
            df = df.rename(columns={"fecha": Config.DATE_COL, "valor": Config.PRECIPITATION_COL})
            df[Config.DATE_COL] = pd.to_datetime(df[Config.DATE_COL])
            return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Cargando ecosistema espacial...", ttl=600)
def load_spatial_data():
    """Carga geometrías usando nombres NUEVOS."""
    engine = get_engine()
    gdf_stations = gpd.GeoDataFrame()
    gdf_municipios = gpd.GeoDataFrame()
    gdf_subcuencas = gpd.GeoDataFrame()
    gdf_predios = gpd.GeoDataFrame()

    if not engine: return None, None, None, None

    try:
        # 1. ESTACIONES
        # CAMBIO: Nombres de columnas actualizados
        sql_est = text("SELECT id_estacion, nombre, altitud, municipio, departamento, latitud, longitud FROM estaciones")
        df_est = pd.read_sql(sql_est, engine)
        
        if not df_est.empty:
            gdf_stations = gpd.GeoDataFrame(
                df_est, 
                geometry=gpd.points_from_xy(df_est.longitud, df_est.latitud), 
                crs="EPSG:4326"
            )
            # Renombrar según Config
            gdf_stations = gdf_stations.rename(columns={
                "nombre": Config.STATION_NAME_COL,
                "altitud": Config.ALTITUDE_COL,
                "municipio": Config.MUNICIPALITY_COL,
                "departamento": Config.REGION_COL,
                "latitud": Config.LATITUDE_COL,
                "longitud": Config.LONGITUDE_COL
            })

        # 2. OTRAS GEOMETRÍAS (Asumiendo que estas tablas no cambiaron o son PostGIS directo)
        try:
            gdf_municipios = gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry")
            gdf_subcuencas = gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry")
            gdf_predios = gpd.read_postgis("SELECT * FROM predios", engine, geom_col="geometry")
        except: pass

        return gdf_stations, gdf_municipios, gdf_subcuencas, gdf_predios

    except Exception as e:
        st.warning(f"Alerta GIS: {e}")
        return gdf_stations, gdf_municipios, gdf_subcuencas, gdf_predios