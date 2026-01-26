# pages/01_🌦️_Clima_e_Hidrologia.py

import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sqlalchemy import text
import geopandas as gpd
from scipy.interpolate import griddata
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SIHCLI-POTER", page_icon="🌧️", layout="wide")
warnings.filterwarnings("ignore")

# --- 2. IMPORTACIONES ---
try:
    from modules.config import Config
    # NOTA: Cambiamos 'load_and_process_all_data' por 'load_spatial_data'

    from modules.data_processor import complete_series, load_spatial_data, parse_spanish_date_robust
    from modules.reporter import generate_pdf_report
    from modules.db_manager import get_engine # Importamos el motor de base de datos
    
    # Importamos función de tendencias
    try:
        from modules.analysis import calculate_trends_mann_kendall
    except ImportError:
        calculate_trends_mann_kendall = None

    from modules.visualizer import (
        display_advanced_maps_tab, display_anomalies_tab, display_climate_forecast_tab,
        display_climate_scenarios_tab, display_correlation_tab, display_current_filters,
        display_drought_analysis_tab, display_graphs_tab, display_land_cover_analysis_tab,
        display_life_zones_tab, display_realtime_dashboard, display_spatial_distribution_tab,
        display_station_table_tab, display_stats_tab, display_trends_and_forecast_tab,
        display_welcome_tab, display_bias_correction_tab
    )
except Exception as e:
    st.error(f"Error crítico importando módulos: {e}")
    st.stop()


# --- FUNCIÓN HÍBRIDA: ESTRATEGIA "ISOYETAS" PARA MAPAS BASE ---
@st.cache_resource(show_spinner="Cargando cartografía...", ttl=3600)
def load_data_from_db():
    from sqlalchemy import text
    import geopandas as gpd
    import pandas as pd
    import os
    
    # Intentamos conectar a BD (para datos dinámicos como Estaciones/Lluvia)
    try:
        engine = get_engine()
    except:
        from modules.db_manager import get_engine
        engine = get_engine()

    # Inicialización de DataFrames vacíos por seguridad
    gdf_municipios = gpd.GeoDataFrame(columns=['MPIO_CNMBR', 'geometry'])
    gdf_subcuencas = gpd.GeoDataFrame(columns=['SUBC_LBL', 'nom_cuenca', 'geometry'])
    gdf_predios_db = gpd.GeoDataFrame(columns=['NOMBRE_PRE', 'geometry'])
    gdf_stations_db = gpd.GeoDataFrame(columns=[Config.STATION_NAME_COL, 'geometry'])
    df_long = pd.DataFrame(columns=[Config.STATION_NAME_COL, Config.PRECIPITATION_COL, Config.DATE_COL])
    df_enso = pd.DataFrame(columns=[Config.DATE_COL, Config.ENSO_ONI_COL, 'fecha_mes_año'])

    # ==============================================================================
    # 1. MUNICIPIOS: ESTRATEGIA "ISOYETAS" (Lectura de Archivo)
    # ==============================================================================
    # En lugar de SQL, buscamos el archivo físico, que es infalible.
    try:
        # Lista de rutas posibles (igual que en Isoyetas HD)
        rutas_posibles = [
            "data/MunicipiosAntioquia.geojson",
            "MunicipiosAntioquia.geojson",
            "pages/data/MunicipiosAntioquia.geojson"
        ]
        
        archivo_encontrado = None
        for ruta in rutas_posibles:
            if os.path.exists(ruta):
                archivo_encontrado = ruta
                break
        
        # Opción B: Si no está en disco, usar GitHub (Tu preferencia)
        if not archivo_encontrado:
            # Reemplaza TU_USUARIO/TU_REPO por los reales si decides usar esta vía
            archivo_encontrado = "https://raw.githubusercontent.com/TU_USUARIO/TU_REPO/main/data/MunicipiosAntioquia.geojson"

        # Leemos con GeoPandas (Directo y Robustos)
        gdf_municipios = gpd.read_file(archivo_encontrado)
        
        if not gdf_municipios.empty:
            # Asegurar WGS84
            if gdf_municipios.crs and gdf_municipios.crs.to_string() != "EPSG:4326":
                gdf_municipios = gdf_municipios.to_crs("EPSG:4326")
            
            # Normalización de Nombres (Lógica de Isoyetas)
            # Busca columnas comunes sin importar mayúsculas
            col_nombre = next((c for c in gdf_municipios.columns if 'cnmbr' in c.lower() or 'nomb' in c.lower() or 'muni' in c.lower()), None)
            
            if col_nombre:
                gdf_municipios['MPIO_CNMBR'] = gdf_municipios[col_nombre] # Estandarizamos para el mapa
                gdf_municipios['name'] = gdf_municipios[col_nombre]       # Para tooltip

    except Exception as e:
        print(f"⚠️ Aviso: No se pudo cargar Municipios desde archivo ({e}).")

    # ==============================================================================
    # 2. CUENCAS: ESTRATEGIA "ISOYETAS" (Lectura de Archivo)
    # ==============================================================================
    # Aplicamos la misma lógica ganadora para Cuencas
    try:
        rutas_cuencas = [
            "data/SubcuencasAinfluencia.geojson", 
            "SubcuencasAinfluencia.geojson"
        ]
        path_cuencas = next((p for p in rutas_cuencas if os.path.exists(p)), None)
        
        if path_cuencas:
            gdf_subcuencas = gpd.read_file(path_cuencas)
            if gdf_subcuencas.crs and gdf_subcuencas.crs.to_string() != "EPSG:4326":
                gdf_subcuencas = gdf_subcuencas.to_crs("EPSG:4326")
            
            # Buscar nombre (N-NSS3 o similar)
            col_nom_c = next((c for c in gdf_subcuencas.columns if 'nss3' in c.lower() or 'nom' in c.lower() or 'cuenca' in c.lower()), None)
            if col_nom_c:
                gdf_subcuencas['nom_cuenca'] = gdf_subcuencas[col_nom_c]
                gdf_subcuencas['SUBC_LBL'] = gdf_subcuencas[col_nom_c]

        else:
            # Si no hay archivo, intentamos fallback a BD (Lógica antigua)
            if engine:
                q_cue = "SELECT nombre_cuenca, subcuenca, ST_AsBinary(geometry) as geom_wkb FROM cuencas"
                df_c = pd.read_sql(text(q_cue), engine)
                from shapely import wkb
                if not df_c.empty and 'geom_wkb' in df_c.columns:
                    df_c['geometry'] = df_c['geom_wkb'].apply(lambda x: wkb.loads(bytes(x)) if x else None)
                    gdf_subcuencas = gpd.GeoDataFrame(df_c, geometry='geometry', crs="EPSG:4326")
                    gdf_subcuencas['nom_cuenca'] = df_c.get('nombre_cuenca', 'Cuenca')
    except Exception:
        pass

    # ==============================================================================
    # 3. PREDIOS: RESCATE DESDE BD (Lat/Lon)
    # ==============================================================================
    # Predios suele ser dinámico, así que intentamos BD primero
    if engine:
        try:
            q_pre = "SELECT nombre_predio, latitud, longitud FROM predios"
            df_pre = pd.read_sql(text(q_pre), engine)
            df_pre = df_pre.dropna(subset=['latitud', 'longitud'])
            if not df_pre.empty:
                df_pre['latitud'] = pd.to_numeric(df_pre['latitud'], errors='coerce')
                df_pre['longitud'] = pd.to_numeric(df_pre['longitud'], errors='coerce')
                gdf_predios_db = gpd.GeoDataFrame(
                    df_pre, geometry=gpd.points_from_xy(df_pre['longitud'], df_pre['latitud']), crs="EPSG:4326"
                )
                if 'nombre_predio' in gdf_predios_db.columns:
                    gdf_predios_db['NOMBRE_PRE'] = gdf_predios_db['nombre_predio']
        except: pass

    # ==============================================================================
    # 4. DATOS DINÁMICOS (Estaciones, Lluvia, Índices) - SIEMPRE DE BD
    # ==============================================================================
    if engine:
        try:
            # Estaciones
            with engine.connect() as conn:
                df_est = pd.read_sql(text("SELECT * FROM estaciones"), conn)
                # ... (Lógica de limpieza existente) ...
                for c in ['latitud', 'longitud', 'alt_est']:
                    if c in df_est.columns:
                        df_est[c] = pd.to_numeric(df_est[c].astype(str).str.replace(',', '.'), errors='coerce')
                if 'latitud' in df_est.columns: df_est = df_est[df_est['latitud'].abs() > 0.0001]
                
                if not df_est.empty:
                    gdf_stations_db = gpd.GeoDataFrame(df_est, geometry=gpd.points_from_xy(df_est.get('longitud', 0), df_est.get('latitud', 0)), crs="EPSG:4326")
                    if 'nom_est' in df_est.columns: gdf_stations_db[Config.STATION_NAME_COL] = df_est['nom_est']
                    if 'municipio' in df_est.columns: gdf_stations_db['municipality'] = df_est['municipio']
                    gdf_stations_db['latitude'] = df_est.get('latitud', 0)
                    gdf_stations_db['longitude'] = df_est.get('longitud', 0)

                # Lluvia
                q_rain = text("SELECT p.id_estacion_fk, e.nom_est as station_name, p.fecha_mes_año, p.precipitation FROM precipitacion_mensual p JOIN estaciones e ON p.id_estacion_fk = e.id_estacion")
                df_long = pd.read_sql(q_rain, conn)
                if 'fecha_mes_año' in df_long.columns:
                    df_long[Config.DATE_COL] = pd.to_datetime(df_long['fecha_mes_año'])
                    df_long[Config.YEAR_COL] = df_long[Config.DATE_COL].dt.year
                    df_long[Config.MONTH_COL] = df_long[Config.DATE_COL].dt.month
                df_long.rename(columns={"station_name": Config.STATION_NAME_COL, "precipitation": Config.PRECIPITATION_COL}, inplace=True)
                
                # ENSO
                df_enso_raw = pd.read_sql(text("SELECT * FROM indices_climaticos"), conn)
                # ... (Procesamiento ENSO existente) ...
                cols = {c.lower(): c for c in df_enso_raw.columns}
                col_anio = next((k for k in cols if 'a' in k and 'o' in k and len(k)<=5), None)
                col_mes = next((k for k in cols if 'mes' in k or 'month' in k), None)
                if col_anio and col_mes:
                    df_enso_raw['fecha_mes_año'] = pd.to_datetime(dict(year=pd.to_numeric(df_enso_raw[cols[col_anio]], errors='coerce'), month=pd.to_numeric(df_enso_raw[cols[col_mes]], errors='coerce'), day=1), errors='coerce')
                    df_enso = df_enso_raw.dropna(subset=['fecha_mes_año']).sort_values('fecha_mes_año')
                    df_enso[Config.DATE_COL] = df_enso['fecha_mes_año']
                    if 'anomalia_oni' in cols: df_enso[Config.ENSO_ONI_COL] = df_enso[cols['anomalia_oni']]
        except Exception as e:
            print(f"Error cargando datos dinámicos: {e}")

    # Garantías finales
    if Config.STATION_NAME_COL not in df_long.columns: df_long[Config.STATION_NAME_COL] = "Estación"
    if Config.PRECIPITATION_COL not in df_long.columns: df_long[Config.PRECIPITATION_COL] = 0.0
    if Config.ENSO_ONI_COL not in df_enso.columns: df_enso[Config.ENSO_ONI_COL] = 0.0

    return gdf_stations_db, gdf_municipios, df_long, df_enso, gdf_subcuencas, gdf_predios_db


# --- NUEVAS FUNCIONES VISUALES (SIHCLI v2.0) ---
def get_name_from_row_v2(row, type_layer):
    """Ayudante para nombres en capas fantasma."""
    cols = row.index.str.lower()
    if type_layer == 'muni':
        for c in ['mpio_cnmbr', 'nombre', 'municipio']:
            if c in cols: return row[c]
    elif type_layer == 'cuenca':
        for c in ['n-nss3', 'subc_lbl', 'nom_cuenca']:
            if c in cols: return row[c]
    return "Desconocido"

def add_context_layers_ghost(fig, gdf_zona):
    """Añade capas de contexto con estilo 'Fantasma' (Sutil y Punteado)."""
    try:
        if gdf_zona is None or gdf_zona.empty: return
        
        # Buffer simple para asegurar contexto alrededor
        roi = gdf_zona.total_bounds
        # Cargar archivos locales si existen para contexto visual rápido
        path_muni = os.path.join("data", "MunicipiosAntioquia.geojson") 
        
        if os.path.exists(path_muni):
            gdf_m = gpd.read_file(path_muni).to_crs("EPSG:4326")
            # Clip visual rápido
            gdf_c = gdf_m.cx[roi[0]:roi[2], roi[1]:roi[3]]
            
            for _, r in gdf_c.iterrows():
                name = get_name_from_row_v2(r, 'muni')
                geom = r.geometry
                if geom:
                    polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                    for p in polys:
                        x, y = p.exterior.xy
                        fig.add_trace(go.Scatter(
                            x=list(x), y=list(y), mode='lines', 
                            line=dict(width=0.7, color='rgba(100, 100, 100, 0.3)', dash='dot'), 
                            hoverinfo='text', text=f"Mpio: {name}", showlegend=False
                        ))
    except Exception as e: print(f"Error capas fantasma: {e}")

# --- FUNCIÓN AUXILIAR PARA DETECTAR COLUMNAS ---
def get_fuzzy_col(df, keywords):
    """Busca si alguna columna contiene alguna de las palabras clave."""
    if df is None: return None
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in col.lower():
                return col
    return None

def main():
    # --- A. INICIALIZACIÓN ---
    # Ya no necesitamos init_db() aquí cada vez, se maneja en admin
    for k in ["lz_raster_result", "lz_profile", "lz_names", "lz_colors"]:
        if k not in st.session_state:
            st.session_state[k] = None

    # --- B. CARGA DE DATOS (AHORA VIA SQL) ---
    data_loaded = False
    
    # Usamos la nueva función load_data_from_db
    (
        gdf_stations, gdf_municipios, df_long,
        df_enso, gdf_subcuencas, gdf_predios,
    ) = load_data_from_db()

    if gdf_stations is not None and not gdf_stations.empty:
        data_loaded = True
    else:
        st.error("No se pudieron cargar los datos de la Base de Datos. Revisa la conexión.")
        st.stop()

    # --- C. BARRA LATERAL (FILTROS Y NAVEGACIÓN) ---
    with st.sidebar:
        st.title("🎛️ SIHCLI-POTER")
        
        # --- 1. MENÚ DE NAVEGACIÓN ---
        st.markdown("### 🚀 Navegación Principal")
        selected_module = st.radio(
            "Ir a:",
            [
                "🏠 Inicio", 
                "🚨 Monitoreo", 
                "🗺️ Distribución", 
                "📈 Gráficos", 
                "📊 Estadísticas", 
                "🔮 Pronóstico Climático", 
                "📉 Tendencias", 
                "⚠️ Anomalías", 
                "🔗 Correlación", 
                "🌊 Extremos", 
                "🌍 Mapas Avanzados", 
                "🧪 Sesgo", 
                "🌿 Cobertura", 
                "🌱 Zonas Vida", 
                "🌡️ Clima Futuro", 
                "📄 Reporte",
                "✨ Mapas Isoyetas HD"
            ]
        )
        st.markdown("---")

        # --- 2. FILTROS GEOGRÁFICOS ---
        with st.expander("🗺️ Filtros Geográficos", expanded=False):
            col_region = get_fuzzy_col(gdf_stations, ["region", "zon", "cuenca", "dpto"])
            col_muni = get_fuzzy_col(gdf_stations, ["muni", "ciud", "city"])
            col_alt = get_fuzzy_col(gdf_stations, ["alt", "elev", "cota", "height"])

            # A. Regiones
            list_regions = []
            sel_regions = []
            if col_region:
                list_regions = sorted(gdf_stations[col_region].astype(str).unique())
                sel_regions = st.multiselect(f"📍 Región:", list_regions, default=[])

            # B. Municipios
            list_munis = []
            sel_munis = []
            if col_muni:
                if sel_regions and col_region:
                    gdf_temp = gdf_stations[gdf_stations[col_region].isin(sel_regions)]
                else:
                    gdf_temp = gdf_stations
                list_munis = sorted(gdf_temp[col_muni].astype(str).unique())
                sel_munis = st.multiselect(f"🏙️ Municipio:", list_munis, default=[])

            # C. Altitud
            rango_alt = None
            if col_alt:
                try:
                    gdf_stations[col_alt] = pd.to_numeric(gdf_stations[col_alt], errors='coerce')
                    min_a, max_a = float(gdf_stations[col_alt].min()), float(gdf_stations[col_alt].max())
                    if pd.notnull(min_a) and pd.notnull(max_a) and min_a < max_a:
                        rango_alt = st.slider("⛰️ Altitud:", int(min_a), int(max_a), (int(min_a), int(max_a)))
                except: pass

        # --- 3. SELECCIÓN DE ESTACIONES ---
        with st.expander("🌧️ Selección Estaciones", expanded=False):
            mask_geo = pd.Series(True, index=gdf_stations.index)
            if sel_regions and col_region:
                mask_geo &= gdf_stations[col_region].isin(sel_regions)
            if sel_munis and col_muni:
                mask_geo &= gdf_stations[col_muni].isin(sel_munis)
            if rango_alt and col_alt:
                mask_geo &= (gdf_stations[col_alt] >= rango_alt[0]) & (gdf_stations[col_alt] <= rango_alt[1])
                
            stations_avail = gdf_stations.loc[mask_geo, Config.STATION_NAME_COL].unique()
            st.caption(f"Disponibles: {len(stations_avail)}")

            # Optimización: Si son muchas, no seleccionar todas por defecto para no saturar gráficos
            default_sel = stations_avail if len(stations_avail) < 10 else []
            
            if st.checkbox("✅ Seleccionar Todas (Visibles)", value=(len(stations_avail) < 10)):
                stations_for_analysis = st.multiselect("Estaciones:", options=stations_avail, default=stations_avail)
            else:
                stations_for_analysis = st.multiselect("Estaciones:", options=stations_avail, default=default_sel)

        # --- 4. TIEMPO Y LIMPIEZA ---
        with st.expander("⏳ Tiempo y Limpieza", expanded=False):
            if not df_long.empty:
                min_year = int(df_long[Config.YEAR_COL].min())
                max_year = int(df_long[Config.YEAR_COL].max())
            else:
                min_year, max_year = 1980, 2024
            
            year_range = st.slider("📅 Años:", min_year, max_year, (min_year, max_year))

            col_opts1, col_opts2 = st.columns(2)
            with col_opts1:
                ignore_zeros = st.checkbox("🚫 Sin Ceros", value=False)
                apply_interp = st.checkbox("🔄 Interpolación", value=False)
            with col_opts2:
                ignore_nulls = st.checkbox("🚫 Sin Nulos", value=False)
            
            analysis_mode = "Anual"

        # --- 5. GESTIÓN ---
        with st.expander("📂 Gestión", expanded=False):
            if st.button("🔄 Recargar Datos"):
                st.cache_data.clear()
                st.rerun()

    # --- D. PROCESAMIENTO DE DATOS (FILTRADO) ---
    # Filtrado en memoria (Pandas) usando el DataFrame cargado
    # En el futuro, esto se podrá mover a SQL para ser aún más rápido
    mask_base = (
        (df_long[Config.YEAR_COL] >= year_range[0])
        & (df_long[Config.YEAR_COL] <= year_range[1])
        & (df_long[Config.STATION_NAME_COL].isin(stations_for_analysis))
    )
    
    df_monthly_filtered = df_long.loc[mask_base].copy()
    
    if ignore_zeros:
        df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] != 0]
    if ignore_nulls:
        df_monthly_filtered = df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL])

    gdf_filtered = gdf_stations[gdf_stations[Config.STATION_NAME_COL].isin(stations_for_analysis)]

    if apply_interp:
        with st.spinner("Interpolando..."):
            df_monthly_filtered = complete_series(df_monthly_filtered)
    
    df_anual_melted = (
        df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[
            Config.PRECIPITATION_COL
        ].sum().reset_index()
    )

    start_date = pd.to_datetime(f"{year_range[0]}-01-01")
    end_date = pd.to_datetime(f"{year_range[1]}-12-31")

    gdf_coberturas = gdf_predios if gdf_predios is not None else None

    # --- E. CÁLCULO LAZY DE TENDENCIAS ---
    df_trends = None
    if selected_module in ["🌍 Mapas Avanzados", "🌡️ Clima Futuro"]:
        if calculate_trends_mann_kendall is not None and not df_anual_melted.empty:
            try:
                trends_res = calculate_trends_mann_kendall(df_anual_melted)
                if trends_res is not None:
                    df_trends = trends_res['trend_data']
            except: pass

    # --- F. EMPAQUETADO DE ARGUMENTOS ---
    display_args = {
        "df_long": df_monthly_filtered, "df_complete": df_monthly_filtered,
        "gdf_stations": gdf_stations, "gdf_filtered": gdf_filtered,
        "gdf_municipios": gdf_municipios, "gdf_subcuencas": gdf_subcuencas,
        "gdf_predios": gdf_predios, "df_enso": df_enso,
        "stations_for_analysis": stations_for_analysis, "df_anual_melted": df_anual_melted,
        "df_monthly_filtered": df_monthly_filtered, "analysis_mode": analysis_mode,
        "selected_regions": sel_regions, "selected_municipios": sel_munis,
        "selected_months": list(range(1, 13)), "year_range": year_range,
        "start_date": start_date, "end_date": end_date,
        "gdf_coberturas": gdf_coberturas,
        "df_trends": df_trends,
        "interpolacion": "Si" if apply_interp else "No",
        "user_loc": None
    }

    # --- G. RENDERIZADO DEL CONTENIDO ---
    
    # 1. Resumen Superior
    try:
        display_current_filters(
            stations_sel=stations_for_analysis,
            regions_sel=sel_regions,
            munis_sel=sel_munis,
            year_range=year_range,
            interpolacion="Si" if apply_interp else "No",
            df_data=df_monthly_filtered,
            gdf_filtered=gdf_filtered
        )
    except Exception:
        pass

    # 2. Enrutador de Módulos
    if selected_module == "🏠 Inicio":
        display_welcome_tab()
        
    elif selected_module == "🚨 Monitoreo":
        display_realtime_dashboard(df_monthly_filtered, gdf_stations, gdf_filtered)
        
    elif selected_module == "🗺️ Distribución":
        display_spatial_distribution_tab(**display_args)
        
    elif selected_module == "📈 Gráficos":
        display_graphs_tab(**display_args)
        
    elif selected_module == "📊 Estadísticas":
        display_stats_tab(**display_args)
        st.markdown("---")
        display_station_table_tab(**display_args)
        
    elif selected_module == "🔮 Pronóstico Climático":
        display_climate_forecast_tab(**display_args)
        
    elif selected_module == "📉 Tendencias":
        display_trends_and_forecast_tab(**display_args)
        
    elif selected_module == "⚠️ Anomalías":
        display_anomalies_tab(**display_args)
        
    elif selected_module == "🔗 Correlación":
        display_correlation_tab(**display_args)
        
    elif selected_module == "🌊 Extremos":
        display_drought_analysis_tab(**display_args)
        
    elif selected_module == "🌍 Mapas Avanzados":
        display_advanced_maps_tab(**display_args)
        
    elif selected_module == "🧪 Sesgo":
        try:
            display_bias_correction_tab(**display_args)
        except:
            st.info("Módulo Sesgo cargando o no disponible.")
            
    elif selected_module == "🌿 Cobertura":
        display_land_cover_analysis_tab(**display_args)
        
    elif selected_module == "🌱 Zonas Vida":
        display_life_zones_tab(**display_args)
        
    elif selected_module == "🌡️ Clima Futuro":
        display_climate_scenarios_tab(**display_args)
        
    elif selected_module == "📄 Reporte":
        st.header("Reporte PDF")
        if st.button("Generar Reporte"):
            res = {"n_estaciones": len(stations_for_analysis), "rango": f"{year_range}"}
            pdf = generate_pdf_report(df_monthly_filtered, gdf_filtered, res)
            if pdf:
                st.download_button("Descargar", pdf, "reporte.pdf", "application/pdf")

# --- MÓDULO INTEGRADO MEJORADO (Isoyetas HD + RBF) ---
    elif selected_module == "✨ Mapas Isoyetas HD":
        st.header("🗺️ Mapas de Isoyetas de Alta Definición")
        
        # VERIFICACIÓN DE INDENTACIÓN AQUÍ
        if gdf_filtered is not None and not gdf_filtered.empty:
            # Recuperamos los límites de la zona FILTRADA
            minx, miny, maxx, maxy = gdf_filtered.total_bounds
            
            col_iso1, col_iso2 = st.columns([1, 3])
            
            with col_iso1:
                st.subheader("Configuración")
                # Selector de año invertido (del más reciente al más antiguo)
                year_iso = st.selectbox("Seleccionar Año:", range(int(year_range[1]), int(year_range[0])-1, -1))
                
                st.info(f"📍 Estaciones en zona: {len(gdf_filtered)}")
                
                # Control de Suavizado
                suavidad = st.slider("Nivel de Suavizado (RBF):", 0.0, 2.0, 0.5, help="0 = Datos crudos, 2 = Muy suavizado")
            
            with col_iso2:
                try:
                    engine = get_engine()
                    # Obtenemos los IDs de las estaciones FILTRADAS en el sidebar
                    ids_validos = tuple(gdf_filtered[Config.STATION_NAME_COL].unique())
                    
                    # Manejo seguro de tuplas para SQL con 1 solo elemento
                    if len(ids_validos) == 1:
                        ids_sql = f"('{ids_validos[0]}')" 
                    else:
                        ids_sql = str(ids_validos)

                    # Consulta optimizada que respeta los filtros
                    q_iso = text(f"""
                        SELECT e.id_estacion, e.nom_est, ST_X(e.geom::geometry) as lon, ST_Y(e.geom::geometry) as lat,
                               SUM(p.precipitation) as valor
                        FROM precipitacion_mensual p
                        JOIN estaciones e ON p.id_estacion_fk = e.id_estacion
                        WHERE extract(year from p.fecha_mes_año) = :anio
                        AND e.nom_est IN {ids_sql} 
                        GROUP BY e.id_estacion, e.nom_est, e.geom
                    """)
                    
                    with engine.connect() as conn:
                        df_iso = pd.read_sql(q_iso, conn, params={"anio": year_iso})
                    
                    # Filtros de limpieza
                    if ignore_zeros:
                        df_iso = df_iso[df_iso['valor'] > 0]
                    if ignore_nulls:
                        df_iso = df_iso.dropna(subset=['valor'])

                    if len(df_iso) >= 3:
                        with st.spinner(f"Generando superficie RBF para {len(df_iso)} estaciones..."):
                            
                            from scipy.interpolate import Rbf
                            
                            grid_res = 200
                            gx, gy = np.mgrid[minx:maxx:complex(0, grid_res), miny:maxy:complex(0, grid_res)]
                            
                            rbf = Rbf(df_iso['lon'], df_iso['lat'], df_iso['valor'], function='thin_plate', smooth=suavidad)
                            grid_z = rbf(gx, gy)
                            
                            fig_m = go.Figure()
                            
                            # Isoyetas
                            fig_m.add_trace(go.Contour(
                                z=grid_z.T, x=np.linspace(minx, maxx, grid_res), y=np.linspace(miny, maxy, grid_res),
                                colorscale="YlGnBu", 
                                colorbar=dict(title="Lluvia (mm)"),
                                hovertemplate="Lluvia: %{z:.0f} mm<extra></extra>",
                                contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white'), start=0),
                                opacity=0.8, connectgaps=True, line_smoothing=1.3
                            ))
                            
                            # Contexto
                            add_context_layers_ghost(fig_m, gdf_filtered)
                            
                            # Puntos
                            fig_m.add_trace(go.Scatter(
                                x=df_iso['lon'], y=df_iso['lat'], mode='markers',
                                marker=dict(size=6, color='black', line=dict(width=1, color='white')),
                                text=df_iso['nom_est'] + ': ' + df_iso['valor'].round(0).astype(str) + ' mm',
                                hoverinfo='text'
                            ))
                            
                            fig_m.update_layout(
                                title=f"Isoyetas Año {year_iso} (Método RBF)", 
                                height=700, 
                                xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False),
                                margin=dict(l=0,r=0,t=40,b=0), plot_bgcolor='white'
                            )
                            st.plotly_chart(fig_m, use_container_width=True)
                    else:
                        st.warning("⚠️ Datos insuficientes para interpolar (mínimo 3 estaciones con datos).")
                except Exception as e:
                    st.error(f"Error al generar mapa: {e}")
        else:
            st.info("👈 Seleccione una cuenca o región en el menú lateral para empezar.")

    # Ajuste CSS
    st.markdown("""<style>.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }</style>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()