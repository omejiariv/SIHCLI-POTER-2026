# -------------------------------------------------------------------
# MIGRACIÓN DE DATOS A POSTGRESQL (migracion.py) - VERSION FINAL PRO
# -------------------------------------------------------------------

import pandas as pd
import geopandas as gpd
import os
import sys
import numpy as np
import socket 
from datetime import datetime

# Importaciones de SQLAlchemy
from sqlalchemy import create_engine, text, Column, Integer, String, Float, Date, JSON, Text, Numeric
from sqlalchemy.orm import sessionmaker, declarative_base
from geoalchemy2 import Geometry 

print("Iniciando el script de migracion...")

# --- PASO 1: CONFIGURAR LA CONEXIÓN A LA BASE DE DATOS ---
# --------------------------------------------------------
# Servidor: aws-1-us-east-2 (Pooler IPv4 para cuentas PRO en Ohio)
# Usuario: postgres.ldunpssoxvifemoyeuac
# Contrasena: ******

DATABASE_URL = "postgresql://postgres.ldunpssoxvifemoyeuac:SIHCLI-POTER123*@aws-1-us-east-2.pooler.supabase.com:6543/postgres"

# Imprimimos para verificar (ocultando la clave)
print(f"Intentando conectar a: {DATABASE_URL.replace('SIHCLI-POTER123*', '******')}")
# --------------------------------------------------------

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
except Exception as e:
    print(f"ERROR: No se pudo conectar a la base de datos.")
    print(f"Detalle: {e}")
    sys.exit(1)

# --- PASO 2: DEFINIR EL ESQUEMA (ORM) ---
Base = declarative_base()

class Estacion(Base):
    __tablename__ = "estaciones"
    id_estacion = Column(Text, primary_key=True)
    nom_est = Column(Text, unique=True, nullable=False)
    alt_est = Column(Numeric)
    municipio = Column(Text)
    depto_region = Column(Text)
    geom = Column(Geometry(geometry_type='POINT', srid=4326))
    et_mmy = Column(Numeric)

class Precipitacion(Base):
    __tablename__ = "precipitacion_mensual"
    id_precip = Column(Integer, primary_key=True, autoincrement=True)
    id_estacion_fk = Column(Text, nullable=False)
    fecha_mes_año = Column(Date, nullable=False)
    precipitation = Column(Numeric)
    origin = Column(Text, default='Original')

class IndiceClimatico(Base):
    __tablename__ = "indices_climaticos"
    fecha_mes_año = Column(Date, primary_key=True)
    anomalia_oni = Column(Numeric)
    soi = Column(Numeric)
    iod = Column(Numeric)

class Geometria(Base):
    __tablename__ = "geometrias"
    id_geom = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(Text)
    tipo_geometria = Column(Text, nullable=False)
    geom = Column(Geometry(geometry_type='MULTIPOLYGON', srid=4326))
    metadatos = Column(JSON)

class Raster(Base):
    __tablename__ = "rasters"
    id_raster = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(Text, unique=True, nullable=False)
    tipo_raster = Column(Text, nullable=False)
    ruta_archivo = Column(Text, nullable=False)

# --- PASO 3: FUNCIONES HELPER ---

def parse_spanish_dates(date_series):
    months_es_to_en = {
        'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
        'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
        'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
    }
    date_series_str = date_series.astype(str).str.lower()
    for es, en in months_es_to_en.items():
        date_series_str = date_series_str.str.replace(es, en, regex=False)
    return pd.to_datetime(date_series_str, format='%b-%y', errors='coerce')

def standardize_numeric_column(series):
    series_clean = series.astype(str).str.replace(',', '.', regex=False)
    return pd.to_numeric(series_clean, errors='coerce')

def preparar_geometria(gdf, tipo, col_nombre):
    gdf = gdf.rename(columns={col_nombre: 'nombre', 'geometry': 'geom'})
    gdf['tipo_geometria'] = tipo
    prop_cols = [col for col in gdf.columns if col not in ['geom', 'nombre', 'tipo_geometria']]
    
    if prop_cols:
        gdf['metadatos'] = gdf[prop_cols].to_dict('records')
    else:
        gdf['metadatos'] = None

    columnas_requeridas = ['nombre', 'tipo_geometria', 'geom', 'metadatos']
    for col in columnas_requeridas:
        if col not in gdf.columns:
            if col == 'metadatos':
                gdf['metadatos'] = None
            else:
                raise KeyError(f"Error interno: falta columna '{col}'.")
                
    return gdf[columnas_requeridas]

# --- PASO 4: SCRIPT PRINCIPAL ---

def migrar_datos():
    try:
        print("Conectado. Verificando extension PostGIS...")
        session.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        session.commit()
        
        print("Creando tablas...")
        Base.metadata.create_all(bind=engine)
        print("Tablas verificadas.")

        # --- A. Migrar Estaciones ---
        print("\nIniciando migracion de 'estaciones'...")
        df_estaciones_raw = pd.read_csv("data/mapaCVENSO.csv", sep=";", encoding='latin1')
        df_estaciones_raw.columns = [col.strip().lower() for col in df_estaciones_raw.columns]
        
        lon_col_real = next((col for col in df_estaciones_raw.columns if 'longitud' in col or 'lon' in col), None)
        lat_col_real = next((col for col in df_estaciones_raw.columns if 'latitud' in col or 'lat' in col), None)
        id_col_real = next((col for col in df_estaciones_raw.columns if 'id_estacio' in col), 'id_estacio')
        alt_col_real = next((col for col in df_estaciones_raw.columns if 'alt_est' in col), 'alt_est')
        et_col_real = next((col for col in df_estaciones_raw.columns if 'et_mmy' in col), 'et_mmy')
        
        if not all([lon_col_real, lat_col_real]):
            raise KeyError("Error: No se encontraron columnas de Latitud o Longitud.")

        df_estaciones = df_estaciones_raw.copy()
        df_estaciones[id_col_real] = df_estaciones[id_col_real].astype(str).str.strip()
        df_estaciones[lat_col_real] = standardize_numeric_column(df_estaciones[lat_col_real])
        df_estaciones[lon_col_real] = standardize_numeric_column(df_estaciones[lon_col_real])
        
        if alt_col_real in df_estaciones.columns:
            df_estaciones[alt_col_real] = standardize_numeric_column(df_estaciones[alt_col_real])
        if et_col_real in df_estaciones.columns:
            df_estaciones[et_col_real] = standardize_numeric_column(df_estaciones[et_col_real])
        else:
            df_estaciones[et_col_real] = np.nan
        
        gdf_estaciones = gpd.GeoDataFrame(
            df_estaciones,
            geometry=gpd.points_from_xy(df_estaciones[lon_col_real], df_estaciones[lat_col_real]),
            crs="EPSG:4326"
        )
        
        rename_map = {
            id_col_real: 'id_estacion',
            'nom_est': 'nom_est',
            alt_col_real: 'alt_est',
            'municipio': 'municipio',
            'depto_region': 'depto_region',
            'geometry': 'geom',
            et_col_real: 'et_mmy'
        }
        
        columnas_para_renombrar = {k: v for k, v in rename_map.items() if k in gdf_estaciones.columns}
        gdf_estaciones_sql = gdf_estaciones.rename(columns=columnas_para_renombrar)
        
        columnas_tabla_estacion = [c.name for c in Estacion.__table__.columns]
        for col in columnas_tabla_estacion:
            if col not in gdf_estaciones_sql.columns:
                gdf_estaciones_sql[col] = np.nan
                
        gdf_estaciones_sql = gdf_estaciones_sql[columnas_tabla_estacion]
        gdf_estaciones_sql = gdf_estaciones_sql.set_geometry("geom")

        print(f"Cargando {len(gdf_estaciones_sql)} estaciones...")
        gdf_estaciones_sql.to_postgis('estaciones', engine, if_exists='replace', index=False)
        print("Migracion de 'estaciones' completada.")

        # --- B. Migrar Índices ---
        print("\nIniciando migracion de 'indices_climaticos'...")
        df_precip_raw = pd.read_csv("data/DatosPptnmes_ENSO.csv", sep=";", encoding='latin1')
        df_precip_raw.columns = [col.strip().lower() for col in df_precip_raw.columns]
        
        df_indices = df_precip_raw[['fecha_mes_año', 'anomalia_oni', 'soi', 'iod']].drop_duplicates()
        df_indices['anomalia_oni'] = standardize_numeric_column(df_indices['anomalia_oni'])
        df_indices['soi'] = standardize_numeric_column(df_indices['soi'])
        df_indices['iod'] = standardize_numeric_column(df_indices['iod'])
        
        df_indices = df_indices.dropna(subset=['fecha_mes_año']).drop_duplicates(subset=['fecha_mes_año'])
        
        print(f"Cargando {len(df_indices)} indices...")
        df_indices.to_sql('indices_climaticos', engine, if_exists='replace', index=False)
        print("Migracion de 'indices_climaticos' completada.")

        # --- C. Migrar Geometrías ---
        print("\nIniciando migracion de 'geometrias'...")
        gdf_subcuencas = gpd.read_file("data/SubcuencasAinfluencia.geojson")
        gdf_predios = gpd.read_file("data/PrediosEjecutados.geojson")
        gdf_municipios = gpd.read_file("data/mapaCVENSO.zip")
        
        gdf_subcuencas.columns = [col.strip().lower() for col in gdf_subcuencas.columns]
        gdf_predios.columns = [col.strip().lower() for col in gdf_predios.columns]
        gdf_municipios.columns = [col.strip().lower() for col in gdf_municipios.columns]
        
        gdf_subcuencas_sql = preparar_geometria(gdf_subcuencas, 'subcuenca', 'subc_lbl')
        gdf_predios_sql = preparar_geometria(gdf_predios, 'predio', 'nombre_pre')
        gdf_municipios_sql = preparar_geometria(gdf_municipios, 'municipio', 'municipio')
        
        gdf_geometrias_final = pd.concat([gdf_subcuencas_sql, gdf_predios_sql, gdf_municipios_sql], ignore_index=True)
        gdf_geometrias_final = gdf_geometrias_final.set_geometry("geom")

        print(f"Cargando {len(gdf_geometrias_final)} geometrias...")
        gdf_geometrias_final.to_postgis('geometrias', engine, if_exists='replace', index=False)
        print("Migracion de 'geometrias' completada.")
        
        # --- D. Migrar Rasters ---
        print("\nIniciando migracion de 'rasters'...")
        rasters_data = [
            {"nombre": "DEM Antioquia (Base)", "tipo_raster": "DEM", "ruta_archivo": "data/DemAntioquia_EPSG3116.tif"},
            {"nombre": "Zonas de Vida (Holdridge)", "tipo_raster": "ZonaVida", "ruta_archivo": "data/PPAMAnt.tif"},
            {"nombre": "Cobertura del Suelo", "tipo_raster": "CoberturaSuelo", "ruta_archivo": "data/Cob25m_WGS84.tif"}
        ]
        df_rasters = pd.DataFrame(rasters_data)
        
        print(f"Cargando {len(df_rasters)} rasters...")
        df_rasters.to_sql('rasters', engine, if_exists='replace', index=False)
        print("Migracion de 'rasters' completada.")

        # --- E. Migrar Precipitación (Parquet) ---
        print("\nIniciando migracion de 'precipitacion_mensual' (esto puede tardar)...")
        df_precip_long = pd.read_parquet("data/datos_precipitacion_largos.parquet")
        
        df_precip_sql = df_precip_long.rename(columns={
            'id_estacion': 'id_estacion_fk',
            'precipitacion_mm': 'precipitation'
        })
        df_precip_sql['fecha_mes_año'] = parse_spanish_dates(df_precip_sql['fecha_mes_año'])
        df_precip_sql['precipitation'] = standardize_numeric_column(df_precip_sql['precipitation'])
        df_precip_sql['id_estacion_fk'] = df_precip_sql['id_estacion_fk'].astype(str).str.strip()
        df_precip_sql['origin'] = 'Original'
        
        columnas_tabla_precip = ['id_estacion_fk', 'fecha_mes_año', 'precipitation', 'origin']
        df_precip_sql = df_precip_sql[columnas_tabla_precip].dropna(subset=['fecha_mes_año'])
        
        print(f"Cargando {len(df_precip_sql)} registros de precipitacion (en lotes)...")
        
        df_precip_sql.to_sql(
            'precipitacion_mensual', 
            engine, 
            if_exists='replace', 
            index=False, 
            chunksize=50000,
            method='multi'
        )
        
        print("Migracion de 'precipitacion_mensual' completada.")
        
        print("\n--- MIGRACION COMPLETADA CON EXITO ---")

    except Exception as e:
        print("\n--- ERROR DURANTE LA MIGRACION ---")
        print(e)
        import traceback
        traceback.print_exc()
        session.rollback()
    finally:
        session.close()
        print("Conexion cerrada.")

# --- EJECUTAR EL SCRIPT ---
if __name__ == "__main__":
    
    archivos_necesarios = [
        "data/mapaCVENSO.csv",
        "data/DatosPptnmes_ENSO.csv",
        "data/datos_precipitacion_largos.parquet",
        "data/SubcuencasAinfluencia.geojson",
        "data/PrediosEjecutados.geojson",
        "data/mapaCVENSO.zip"
    ]
    
    archivos_faltantes = [f for f in archivos_necesarios if not os.path.exists(f)]
    
    if archivos_faltantes:
        print("ERROR: Faltan archivos de datos en la carpeta 'data/'.")
        for f in archivos_faltantes:
            print(f"- {f}")
    else:
        print("Archivos encontrados. Iniciando...")
        migrar_datos()