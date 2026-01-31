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
st.set_page_config(page_title="SIHCLI-POTER", page_icon="🌦️", layout="wide")
warnings.filterwarnings("ignore")

# --- 2. IMPORTACIONES ---
try:
    from modules.config import Config
    # Importamos el selector robusto (NUEVO)
    from modules import selectors 
    
    from modules.data_processor import complete_series, load_spatial_data, parse_spanish_date_robust
    from modules.reporter import generate_pdf_report
    from modules.db_manager import get_engine 
    
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


# --- FUNCIÓN UNIFICADA DE CARGA (INTACTA) ---
@st.cache_resource(show_spinner="Consultando Sistema de Información (Nube)...", ttl=3600)
def load_data_from_db():
    from sqlalchemy import text
    import geopandas as gpd
    import pandas as pd
    from modules.config import Config
    
    try:
        engine = get_engine()
    except:
        from modules.db_manager import get_engine
        engine = get_engine()

    gdf_mun = gpd.GeoDataFrame()
    gdf_cuencas = gpd.GeoDataFrame()
    gdf_predios = gpd.GeoDataFrame()
    gdf_est = gpd.GeoDataFrame()
    df_rain = pd.DataFrame(columns=[Config.STATION_NAME_COL, Config.PRECIPITATION_COL, Config.DATE_COL])
    df_enso = pd.DataFrame(columns=[Config.DATE_COL, Config.ENSO_ONI_COL])

    if not engine:
        return gdf_est, gdf_mun, df_rain, df_enso, gdf_cuencas, gdf_predios

    # 1. MUNICIPIOS
    try:
        gdf_mun = gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry")
        if 'nombre_municipio' in gdf_mun.columns:
            gdf_mun['MPIO_CNMBR'] = gdf_mun['nombre_municipio']
        elif 'nombre' in gdf_mun.columns:
            gdf_mun['MPIO_CNMBR'] = gdf_mun['nombre']
    except Exception as e: print(f"⚠️ Error Municipios BD: {e}")

    # 2. CUENCAS
    try:
        gdf_cuencas = gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry")
        if 'nombre_cuenca' in gdf_cuencas.columns:
            gdf_cuencas['SUBC_LBL'] = gdf_cuencas['nombre_cuenca']
            gdf_cuencas['N-NSS3'] = gdf_cuencas['nombre_cuenca']
            gdf_cuencas['nom_cuenca'] = gdf_cuencas['nombre_cuenca']
    except Exception as e: print(f"⚠️ Error Cuencas BD: {e}")

    # 3. PREDIOS
    try:
        gdf_predios = gpd.read_postgis("SELECT * FROM predios", engine, geom_col="geometry")
        if 'nombre_predio' in gdf_predios.columns:
            gdf_predios['NOMBRE_PRE'] = gdf_predios['nombre_predio']
    except Exception: pass

    # 4. DATOS DINÁMICOS
    try:
        # A. Estaciones
        try:
            gdf_est = gpd.read_postgis("SELECT * FROM estaciones", engine, geom_col="geom")
        except:
            df_e = pd.read_sql("SELECT * FROM estaciones", engine)
            if 'latitud' in df_e.columns and 'longitud' in df_e.columns:
                gdf_est = gpd.GeoDataFrame(df_e, geometry=gpd.points_from_xy(df_e.longitud, df_e.latitud), crs="EPSG:4326")

        # B. Lluvia
        q_rain = text("""
            SELECT p.id_estacion_fk, e.nom_est, p.fecha_mes_año, p.precipitation 
            FROM precipitacion_mensual p 
            JOIN estaciones e ON p.id_estacion_fk = e.id_estacion
        """)
        df_rain = pd.read_sql(q_rain, engine)
        
        if not df_rain.empty:
            col_fecha = 'fecha_mes_año'
            col_valor = 'precipitation'
            col_nombre = 'nom_est'
            
            df_rain[Config.DATE_COL] = pd.to_datetime(df_rain[col_fecha])
            df_rain[Config.PRECIPITATION_COL] = pd.to_numeric(df_rain[col_valor], errors='coerce')
            df_rain[Config.STATION_NAME_COL] = df_rain[col_nombre]
            # Mapeo de ID para cruces
            df_rain['id_estacion'] = df_rain['id_estacion_fk'].astype(str)
            
            df_rain[Config.YEAR_COL] = df_rain[Config.DATE_COL].dt.year
            df_rain[Config.MONTH_COL] = df_rain[Config.DATE_COL].dt.month

        # C. Índices
        df_enso_raw = pd.read_sql("SELECT * FROM indices_climaticos", engine)
        if not df_enso_raw.empty:
            if 'fecha_mes_año' in df_enso_raw.columns:
                df_enso_raw[Config.DATE_COL] = pd.to_datetime(df_enso_raw['fecha_mes_año'])
            elif 'año' in df_enso_raw.columns and 'mes' in df_enso_raw.columns:
                df_enso_raw[Config.DATE_COL] = pd.to_datetime(df_enso_raw[['año', 'mes']].assign(DAY=1))
            
            df_enso = df_enso_raw.sort_values(Config.DATE_COL)
            if 'anomalia_oni' in df_enso.columns:
                df_enso[Config.ENSO_ONI_COL] = df_enso['anomalia_oni']

    except Exception as e: print(f"⚠️ Error en datos dinámicos: {e}")

    return gdf_est, gdf_mun, df_rain, df_enso, gdf_cuencas, gdf_predios


# --- FUNCIONES VISUALES AUXILIARES ---
def get_name_from_row_v2(row, type_layer):
    cols = row.index.str.lower()
    if type_layer == 'muni':
        for c in ['mpio_cnmbr', 'nombre', 'municipio']:
            if c in cols: return row[c]
    elif type_layer == 'cuenca':
        for c in ['n-nss3', 'subc_lbl', 'nom_cuenca']:
            if c in cols: return row[c]
    return "Desconocido"

def add_context_layers_ghost(fig, gdf_zona):
    try:
        if gdf_zona is None or gdf_zona.empty: return
        roi = gdf_zona.total_bounds
        path_muni = os.path.join("data", "MunicipiosAntioquia.geojson") 
        
        if os.path.exists(path_muni):
            gdf_m = gpd.read_file(path_muni).to_crs("EPSG:4326")
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

def get_fuzzy_col(df, keywords):
    if df is None: return None
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in col.lower():
                return col
    return None

# ==============================================================================
# MAIN APLICATION
# ==============================================================================
def main():
    
# --- 1. SELECTOR ESPACIAL (NUEVO & UNIFICADO) ---
    # Reemplaza la antigua barra lateral de filtros dispersos
    ids_estaciones, nombre_zona, altitud_ref, gdf_zona = selectors.render_selector_espacial()

    # LÓGICA DE VECINDAD (BUFFER):
    # Si no hay estaciones EXACTAMENTE dentro, buscamos en un radio de 20km.
    if not ids_estaciones and gdf_zona is not None and not gdf_zona.empty:
        with st.spinner(f"🔎 No hay estaciones dentro de {nombre_zona}. Buscando cercanas (20km)..."):
            try:
                # 1. Crear Buffer de 20km (0.2 grados aprox si es WGS84, o 20000m si es proyectado)
                # Detectamos CRS para saber si usar grados o metros
                es_geografico = gdf_zona.crs.is_geographic if gdf_zona.crs else True
                radio_buffer = 0.18 if es_geografico else 20000 # ~20km
                
                buffer_geom = gdf_zona.geometry.buffer(radio_buffer).iloc[0]
                
                # 2. Consultar Estaciones en BD que caigan en el buffer
                engine_temp = get_engine()
                # Traemos geom de todas las estaciones
                gdf_all_est = gpd.read_postgis("SELECT id_estacion, geom FROM estaciones", engine_temp, geom_col="geom")
                
                # 3. Intersección Espacial
                est_cercanas = gdf_all_est[gdf_all_est.geometry.within(buffer_geom)]
                
                if not est_cercanas.empty:
                    ids_estaciones = est_cercanas['id_estacion'].astype(str).unique().tolist()
                    st.toast(f"✅ Se encontraron {len(ids_estaciones)} estaciones cercanas.", icon="📡")
                
            except Exception as e:
                print(f"Error calculando buffer: {e}")

    # VALIDACIÓN FINAL
    # Solo detenemos si DESPUÉS del buffer sigue sin haber nada.
    if not ids_estaciones:
        if gdf_zona is None:
            st.info("👈 Selecciona una Cuenca o Municipio en el menú lateral para comenzar.")
        else:
            st.warning(f"⚠️ No se encontraron estaciones ni dentro ni cerca (20km) de {nombre_zona}.")
        st.stop()


    # --- 2. CARGA DE DATOS ---
    (gdf_stations, gdf_municipios, df_all_rain, df_enso, gdf_subcuencas, gdf_predios) = load_data_from_db()

    # Filtramos los datos globales con la selección del usuario
    if not df_all_rain.empty and ids_estaciones:
        # Filtrar Lluvia
        df_long = df_all_rain[df_all_rain['id_estacion'].isin(ids_estaciones)].copy()
        
        # Filtrar Estaciones (Geometría)
        if gdf_stations is not None and not gdf_stations.empty:
            # Aseguramos que id_estacion sea string para comparar
            gdf_stations['id_estacion'] = gdf_stations['id_estacion'].astype(str)
            gdf_filtered = gdf_stations[gdf_stations['id_estacion'].isin(ids_estaciones)]
        else:
            gdf_filtered = gpd.GeoDataFrame()
    else:
        st.error("No hay datos de lluvia disponibles para esta zona.")
        st.stop()

    if df_long.empty:
        st.warning(f"La zona '{nombre_zona}' no tiene registros históricos de precipitación.")
        st.stop()

    stations_for_analysis = df_long[Config.STATION_NAME_COL].unique().tolist()

    # --- 3. BARRA LATERAL (NAVEGACIÓN Y TIEMPO) ---
    with st.sidebar:
        st.divider()
        st.markdown("### 🚀 Navegación")
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

        # --- [INICIO BLOQUE NUEVO] FILTRO POR REGIÓN ---
        # Detectar columnas que parezcan región en los datos cargados
        col_region = get_fuzzy_col(gdf_filtered, ["region", "subregion", "zona", "dpto"])
        sel_regions = []
        
        if col_region:
            # Extraer regiones únicas
            unique_regions = sorted(gdf_filtered[col_region].astype(str).unique())
            
            # Solo mostramos el filtro si hay más de una región
            if len(unique_regions) > 1:
                with st.expander("📍 Filtrar por Región", expanded=True):
                    sel_regions = st.multiselect("Seleccionar:", unique_regions)
                    
                    if sel_regions:
                        # APLICAR FILTRO EN CASCADA
                        # 1. Filtramos las estaciones geográficas
                        gdf_filtered = gdf_filtered[gdf_filtered[col_region].isin(sel_regions)]
                        
                        # 2. Obtenemos los IDs válidos
                        valid_ids = gdf_filtered['id_estacion'].unique()
                        
                        # 3. Filtramos la tabla de datos climáticos (df_long)
                        df_long = df_long[df_long['id_estacion'].isin(valid_ids)]
                        
                        st.caption(f"✅ Filtrado: {len(valid_ids)} estaciones")
        # --- [FIN BLOQUE NUEVO] ---

        # Filtro de Tiempo (Conservado porque es útil)
        with st.expander("⏳ Tiempo y Limpieza", expanded=False):
            min_year = int(df_long[Config.YEAR_COL].min())
            max_year = int(df_long[Config.YEAR_COL].max())
            year_range = st.slider("📅 Años:", min_year, max_year, (min_year, max_year))

            c1, c2 = st.columns(2)
            ignore_zeros = c1.checkbox("🚫 Sin Ceros", value=False)
            ignore_nulls = c2.checkbox("🚫 Sin Nulos", value=False)
            apply_interp = st.checkbox("🔄 Interpolación", value=False)
            
            analysis_mode = "Anual"

        # Botón recarga
        if st.button("🔄 Recargar Datos"):
            st.cache_data.clear()
            st.rerun()

    # --- 4. PROCESAMIENTO DE DATOS (FILTRO TEMPORAL) ---
    mask_time = (df_long[Config.YEAR_COL] >= year_range[0]) & (df_long[Config.YEAR_COL] <= year_range[1])
    df_monthly_filtered = df_long.loc[mask_time].copy()
    
    if ignore_zeros:
        df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] != 0]
    if ignore_nulls:
        df_monthly_filtered = df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL])

    if apply_interp:
        with st.spinner("Interpolando series..."):
            df_monthly_filtered = complete_series(df_monthly_filtered)
    
    # Agregado Anual para ciertos gráficos
    df_anual_melted = (
        df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[Config.PRECIPITATION_COL]
        .sum().reset_index()
    )

    # Coberturas (Si hay predios cargados)
    gdf_coberturas = gdf_predios if gdf_predios is not None else None

    # --- 5. EMPAQUETADO DE ARGUMENTOS ---
    # Esto asegura que todas las pestañas reciban lo que necesitan
    display_args = {
        "df_long": df_monthly_filtered, "df_complete": df_monthly_filtered,
        "gdf_stations": gdf_stations, "gdf_filtered": gdf_filtered,
        "gdf_municipios": gdf_municipios, "gdf_subcuencas": gdf_subcuencas,
        "gdf_predios": gdf_predios, "df_enso": df_enso,
        "stations_for_analysis": stations_for_analysis, "df_anual_melted": df_anual_melted,
        "df_monthly_filtered": df_monthly_filtered, "analysis_mode": analysis_mode,
        "selected_regions": [], "selected_municipios": [],
        "selected_months": list(range(1, 13)), "year_range": year_range,
        "start_date": pd.to_datetime(f"{year_range[0]}-01-01"), 
        "end_date": pd.to_datetime(f"{year_range[1]}-12-31"),
        "gdf_coberturas": gdf_coberturas,
        "interpolacion": "Si" if apply_interp else "No",
        "user_loc": None,
        # NUEVO: Pasamos la geometría de la zona seleccionada para mapas
        "gdf_zona": gdf_zona 
    }

    # --- 6. RENDERIZADO DE MÓDULOS ---
    
    # Título Dinámico
    st.title(f"🌦️ Análisis: {nombre_zona}")
    
    # Enrutador
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
        st.header("🌍 Modelación Hidrológica Distribuida (Aleph)")
        
        # --- 1. IMPORTACIÓN DE MOTORES ---
        try:
            from modules import hydro_physics as physics
            from modules import visualizer as viz
        except ImportError as e:
            st.error(f"Error cargando módulos: {e}. Verifica que 'hydro_physics.py' exista.")
            st.stop()

        # 2. PREPARACIÓN DE DATOS
        if 'year' not in df_monthly_filtered.columns:
            df_monthly_filtered['year'] = df_monthly_filtered[Config.DATE_COL].dt.year
            
        # Agrupación
        df_annual = df_monthly_filtered.groupby([Config.STATION_NAME_COL, 'year'])[Config.PRECIPITATION_COL].sum().reset_index()
        df_mean = df_annual.groupby(Config.STATION_NAME_COL)[Config.PRECIPITATION_COL].mean().reset_index(name='ppt_media')
        
        # --- MERGE SEGURO ---
        # Unimos los datos calculados (df_mean) con la geometría original (gdf_filtered)
        gdf_calc = gdf_filtered.merge(df_mean, on=Config.STATION_NAME_COL, how='inner')
        
        # IMPORTANTE: Forzar que sea GeoDataFrame inmediatamente después del merge
        if not isinstance(gdf_calc, gpd.GeoDataFrame):
            gdf_calc = gpd.GeoDataFrame(gdf_calc, geometry='geometry')
            
        # Asegurarnos de tener lat/lon explícitos como respaldo para el visualizador
        gdf_calc['latitude'] = gdf_calc.geometry.y
        gdf_calc['longitude'] = gdf_calc.geometry.x

        # --- 3. DEFINICIÓN DEL GRID (LA "HOJA DE PAPEL") ---
        # Usamos los límites de la cuenca si existe, o de las estaciones
        if gdf_zona is not None:
            minx, miny, maxx, maxy = gdf_zona.total_bounds
        else:
            minx, miny, maxx, maxy = gdf_filtered.total_bounds
            
        # Margen del 20% para evitar efectos de borde
        dx, dy = maxx - minx, maxy - miny
        pad_x, pad_y = dx * 0.2, dy * 0.2
        
        # Resolución del Grid (300x300 píxeles para alta definición)
        grid_res = 300
        xi = np.linspace(minx - pad_x, maxx + pad_x, grid_res)
        yi = np.linspace(miny - pad_y, maxy + pad_y, grid_res)
        grid_x, grid_y = np.meshgrid(xi, yi)

        # --- 4. INTERPOLACIÓN INICIAL (LLUVIA) ---
        # Interpolamos la precipitación base (Z_P) desde las estaciones al grid
        # Usamos el motor físico para esto
        with st.spinner("Interpolando Precipitación..."):
            Z_P = physics.interpolar_variable(gdf_calc, 'ppt_media', grid_x, grid_y)

        # 5. EJECUCIÓN MODELO FÍSICO
        paths = {
            'dem': 'DemAntioquia_EPSG3116.tif',
            'cobertura': 'Cob25m_WGS84.tif'
        }
        
        with st.spinner("Calculando Balance Distribuido (Reproyectando Rasters)..."):
            # PASAMOS 'bounds_wgs84' AHORA
            matrices = physics.run_distributed_model(Z_P, grid_x, grid_y, paths, bounds_wgs84)        
        with st.spinner("Calculando Balance Distribuido (Turc + Schosinsky)..."):
            # ¡AQUÍ OCURRE LA MAGIA! El cerebro devuelve todas las matrices listas
            matrices = physics.run_distributed_model(Z_P, grid_x, grid_y, paths)

        # --- 6. MÁSCARA VISUAL (RECORTAR POR CUENCA) ---
        # Creamos una máscara para que el visualizador sepa qué pintar transparente
        mask_inside = None
        if gdf_zona is not None:
            from shapely.ops import unary_union
            from matplotlib import path as mpath
            
            # Unificar geometría
            zona_union = gdf_zona.unary_union
            # Crear máscara booleana con matplotlib path (rápido y preciso)
            polys = [zona_union] if zona_union.geom_type == 'Polygon' else list(zona_union.geoms)
            
            points_flat = np.vstack((grid_x.flatten(), grid_y.flatten())).T
            full_mask = np.zeros(points_flat.shape[0], dtype=bool)
            
            for poly in polys:
                p_path = mpath.Path(list(poly.exterior.coords))
                full_mask = full_mask | p_path.contains_points(points_flat)
            
            mask_inside = full_mask.reshape(grid_x.shape)

        # --- 7. LLAMADA AL VISUALIZADOR ---
        # Le entregamos todo cocinado al "Pintor"
        viz.display_advanced_maps_tab(
            gdf_stations=gdf_calc,
            matrices=matrices,
            grid=(grid_x, grid_y),
            mask=mask_inside,
            gdf_zona=gdf_zona,
            nombre_zona=nombre_zona
        )

    elif selected_module == "🧪 Sesgo":
        try: display_bias_correction_tab(**display_args)
        except: st.info("Módulo Sesgo cargando...")
            
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
            if pdf: st.download_button("Descargar", pdf, "reporte.pdf", "application/pdf")

    # --- MÓDULO ESPECIAL: ISOYETAS HD (Conservado Original) ---
    elif selected_module == "✨ Mapas Isoyetas HD":
        st.header("🗺️ Mapas de Isoyetas de Alta Definición")
        
        if gdf_filtered is not None and not gdf_filtered.empty:
            minx, miny, maxx, maxy = gdf_filtered.total_bounds
            col_iso1, col_iso2 = st.columns([1, 3])
            
            with col_iso1:
                st.subheader("Configuración")
                year_iso = st.selectbox("Seleccionar Año:", range(int(year_range[1]), int(year_range[0])-1, -1))
                st.info(f"📍 Estaciones en zona: {len(gdf_filtered)}")
                suavidad = st.slider("Nivel de Suavizado (RBF):", 0.0, 2.0, 0.5)
            
            with col_iso2:
                try:
                    engine = get_engine()
                    ids_validos = tuple(gdf_filtered['id_estacion'].unique()) # Usamos ID, más seguro
                    if not ids_validos: st.stop()
                    
                    if len(ids_validos) == 1: ids_sql = f"('{ids_validos[0]}')" 
                    else: ids_sql = str(ids_validos)

                    q_iso = text(f"""
                        SELECT e.id_estacion, e.nom_est, ST_X(e.geom::geometry) as lon, ST_Y(e.geom::geometry) as lat,
                               SUM(p.precipitation) as valor
                        FROM precipitacion_mensual p
                        JOIN estaciones e ON p.id_estacion_fk = e.id_estacion
                        WHERE extract(year from p.fecha_mes_año) = :anio
                        AND e.id_estacion IN {ids_sql} 
                        GROUP BY e.id_estacion, e.nom_est, e.geom
                    """)
                    
                    with engine.connect() as conn:
                        df_iso = pd.read_sql(q_iso, conn, params={"anio": year_iso})
                    
                    if ignore_zeros: df_iso = df_iso[df_iso['valor'] > 0]
                    if ignore_nulls: df_iso = df_iso.dropna(subset=['valor'])

                    if len(df_iso) >= 3:
                        with st.spinner(f"Generando superficie RBF..."):
                            from scipy.interpolate import Rbf
                            grid_res = 200
                            gx, gy = np.mgrid[minx:maxx:complex(0, grid_res), miny:maxy:complex(0, grid_res)]
                            rbf = Rbf(df_iso['lon'], df_iso['lat'], df_iso['valor'], function='thin_plate', smooth=suavidad)
                            grid_z = rbf(gx, gy)
                            
                            fig_m = go.Figure()
                            fig_m.add_trace(go.Contour(
                                z=grid_z.T, x=np.linspace(minx, maxx, grid_res), y=np.linspace(miny, maxy, grid_res),
                                colorscale="YlGnBu", colorbar=dict(title="Lluvia (mm)"),
                                contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                                opacity=0.8, connectgaps=True, line_smoothing=1.3
                            ))
                            add_context_layers_ghost(fig_m, gdf_filtered) # Contexto fantasma
                            fig_m.add_trace(go.Scatter(
                                x=df_iso['lon'], y=df_iso['lat'], mode='markers',
                                marker=dict(size=6, color='black', line=dict(width=1, color='white')),
                                text=df_iso['nom_est'] + ': ' + df_iso['valor'].round(0).astype(str) + ' mm',
                                hoverinfo='text'
                            ))
                            fig_m.update_layout(title=f"Isoyetas Año {year_iso}", height=700, margin=dict(l=0,r=0,t=40,b=0))
                            st.plotly_chart(fig_m, use_container_width=True)
                    else:
                        st.warning("⚠️ Datos insuficientes para interpolar.")
                except Exception as e:
                    st.error(f"Error al generar mapa: {e}")
        else:
            st.info("👈 Seleccione una cuenca o región.")

    st.markdown("""<style>.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }</style>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()