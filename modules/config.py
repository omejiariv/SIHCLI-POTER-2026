import os
import streamlit as st

class Config:
    """
    Configuración centralizada para SIHCLI-POTER.
    Ajustada a la NUEVA estructura de Base de Datos PostgreSQL.
    """

    APP_TITLE = "SIHCLI-POTER"

    # --- MAPEO EXACTO CON BASE DE DATOS NUEVA ---
    DATE_COL = "fecha"              # Antes: fecha_mes_año
    PRECIPITATION_COL = "valor"     # Antes: precipitation

    # Metadatos de Estaciones
    STATION_NAME_COL = "nombre"     # Antes: nom_est
    ALTITUDE_COL = "altitud"        # Antes: alt_est
    MUNICIPALITY_COL = "municipio"
    REGION_COL = "departamento"     # Antes: depto_region

    # Columnas geográficas
    LATITUDE_COL = "latitud"
    LONGITUDE_COL = "longitud"
    
    # Columnas generadas internamente
    YEAR_COL = "año"
    MONTH_COL = "mes"

    # Índices Climáticos
    ENSO_ONI_COL = "anomalia_oni"
    SOI_COL = "soi"
    IOD_COL = "iod"

    # --- RUTAS DE ARCHIVOS Y ASSETS ---
    _MODULES_DIR = os.path.dirname(__file__)
    _PROJECT_ROOT = os.path.abspath(os.path.join(_MODULES_DIR, ".."))

    ASSETS_DIR = os.path.join(_PROJECT_ROOT, "assets")
    DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

    # Archivos de Imagen y Raster
    LOGO_PATH = os.path.join(ASSETS_DIR, "CuencaVerde_Logo.jpg")
    CHAAC_IMAGE_PATH = os.path.join(ASSETS_DIR, "chaac.png")
    LAND_COVER_RASTER_PATH = os.path.join(DATA_DIR, "Cob25m_WGS84.tif")
    DEM_FILE_PATH = os.path.join(DATA_DIR, "DemAntioquia_EPSG3116.tif")
    PRECIP_RASTER_PATH = os.path.join(DATA_DIR, "PPAMAnt.tif")

    # --- TEXTOS ---
    WELCOME_TEXT = """
    **Sistema de Información Hidroclimática del Norte de la Región Andina**
    Esta plataforma integra datos históricos, análisis estadísticos y modelación espacial.
    """
    QUOTE_TEXT = "El agua es la fuerza motriz de toda la naturaleza."
    QUOTE_AUTHOR = "Leonardo da Vinci"
    CHAAC_STORY = "Chaac es la deidad maya de la lluvia."

    # --- GESTIÓN DE SESIÓN ---
    @staticmethod
    def initialize_session_state():
        keys = [
            "data_loaded", "apply_interpolation", "gdf_stations", "df_long",
            "df_enso", "gdf_municipios", "gdf_subcuencas", "gdf_predios",
            "unified_basin_gdf", "basin_results", "sarima_res", 
            "prophet_res", "res_cuenca", "current_coverage_stats"
        ]
        for k in keys:
            if k not in st.session_state:
                st.session_state[k] = None