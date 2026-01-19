# modules/data_processor.py

import geopandas as gpd
import pandas as pd
import streamlit as st
from shapely import wkt
from sqlalchemy import text
from modules.config import Config

# Importamos el motor centralizado para no duplicar conexiones
from modules.db_manager import get_engine

# --- UTILIDADES ---
def parse_spanish_date_robust(x):
    """Tu función robusta original para fechas en español."""
    if isinstance(x, pd.Timestamp):
        return x
    if pd.isna(x) or x == "":
        return pd.NaT
    x = str(x).lower().strip()
    trans = {
        "ene": "Jan", "feb": "Feb", "mar": "Mar", "abr": "Apr",
        "may": "May", "jun": "Jun", "jul": "Jul", "ago": "Aug",
        "sep": "Sep", "oct": "Oct", "nov": "Nov", "dic": "Dec",
    }
    for es, en in trans.items():
        if es in x:
            x = x.replace(es, en)
            break
    try:
        return pd.to_datetime(x, format="%b-%y")
    except:
        try:
            return pd.to_datetime(x)
        except:
            return pd.NaT

def complete_series(df):
    """Interpolación lineal para rellenar huecos."""
    if df is None or df.empty:
        return df
    df = df.sort_values(Config.DATE_COL)
    df[Config.PRECIPITATION_COL] = df[Config.PRECIPITATION_COL].interpolate(
        method="linear", limit_direction="both"
    )
    return df

# --- NUEVAS FUNCIONES ÁGILES (SQL) ---

def get_lista_estaciones_simple():
    """
    Obtiene solo ID y Nombre para llenar el selector (Dropdown).
    Súper rápido y liviano via SQL.
    """
    engine = get_engine()
    if not engine: return []
    
    try:
        with engine.connect() as conn:
            # Traemos ID y Nombre concatenados para el usuario
            query = text("""
                SELECT id_estacion, nom_est, municipio 
                FROM estaciones 
                ORDER BY nom_est ASC
            """)
            df = pd.read_sql(query, conn)
            
            # Crear etiqueta bonita: "Aeropuerto [12345]"
            df['label'] = df.apply(
                lambda x: f"{x['nom_est']} [{x['id_estacion']}]", axis=1
            )
            return df['label'].tolist()
    except Exception as e:
        st.error(f"Error cargando lista estaciones: {e}")
        return []

def get_datos_estacion_individual(station_id):
    """
    Trae SOLO los datos de lluvia de una estación específica.
    Esto reemplaza la carga masiva para los gráficos individuales.
    """
    engine = get_engine()
    if not engine: return pd.DataFrame()

    try:
        with engine.connect() as conn:
            # Consulta optimizada con filtro WHERE
            query = text("""
                SELECT fecha_mes_año, precipitation
                FROM precipitacion_mensual
                WHERE id_estacion_fk = :id
                ORDER BY fecha_mes_año ASC
            """)
            # Pandas lee SQL y convierte params de forma segura
            df = pd.read_sql(query, conn, params={"id": station_id})
            
            # Estandarizar nombres de columnas a tu Config
            df = df.rename(columns={
                "fecha_mes_año": Config.DATE_COL,
                "precipitation": Config.PRECIPITATION_COL
            })
            
            # Asegurar datetime (Postgres ya devuelve datetime, pero por seguridad)
            df[Config.DATE_COL] = pd.to_datetime(df[Config.DATE_COL])
            
            return df
    except Exception as e:
        st.error(f"Error cargando datos de estación {station_id}: {e}")
        return pd.DataFrame()

# --- FUNCIÓN "LEGACY" OPTIMIZADA (Para Mapas y Análisis Global) ---

@st.cache_data(show_spinner="Cargando ecosistema espacial...", ttl=600)
def load_spatial_data():
    """
    Carga SOLO las geometrías (mapas), separado de los datos de lluvia masivos.
    """
    engine = get_engine()
    gdf_stations = gpd.GeoDataFrame()
    gdf_municipios = gpd.GeoDataFrame()
    gdf_subcuencas = gpd.GeoDataFrame()
    gdf_predios = gpd.GeoDataFrame()

    if not engine: return None, None, None, None

    try:
        # 1. ESTACIONES (Solo metadatos y ubicación, NO lluvias)
        sql_est = text("SELECT id_estacion, nom_est, alt_est, municipio, depto_region, ST_AsText(geom) as wkt FROM estaciones")
        df_est = pd.read_sql(sql_est, engine)
        
        if "wkt" in df_est.columns and not df_est.empty:
            df_est["geometry"] = df_est["wkt"].apply(lambda x: wkt.loads(x) if x else None)
            gdf_stations = gpd.GeoDataFrame(df_est, geometry="geometry", crs="EPSG:4326")
            # Crear columnas lat/lon explícitas para mapas rápidos
            gdf_stations["latitude"] = gdf_stations.geometry.y
            gdf_stations["longitude"] = gdf_stations.geometry.x
            
            # Renombrar columnas según Config
            gdf_stations = gdf_stations.rename(columns={
                "nom_est": Config.STATION_NAME_COL,
                "alt_est": Config.ALTITUDE_COL,
                "municipio": Config.MUNICIPALITY_COL,
                "depto_region": Config.REGION_COL
            })

        # 2. GEOMETRÍAS (Municipios, Cuencas, Predios)
        sql_geo = text("SELECT nombre, tipo_geometria, ST_AsText(geom) as wkt FROM geometrias")
        df_geo = pd.read_sql(sql_geo, engine)
        
        if not df_geo.empty:
            df_geo["geometry"] = df_geo["wkt"].apply(lambda x: wkt.loads(x) if x else None)
            gdf_all = gpd.GeoDataFrame(df_geo, geometry="geometry", crs="EPSG:4326")
            
            gdf_municipios = gdf_all[gdf_all["tipo_geometria"] == "municipio"]
            gdf_subcuencas = gdf_all[gdf_all["tipo_geometria"].isin(["subcuenca", "cuenca"])]
            gdf_predios = gdf_all[gdf_all["tipo_geometria"] == "predio"]

        return gdf_stations, gdf_municipios, gdf_subcuencas, gdf_predios

    except Exception as e:
        st.warning(f"Error cargando datos espaciales: {e}")
        return None, None, None, None