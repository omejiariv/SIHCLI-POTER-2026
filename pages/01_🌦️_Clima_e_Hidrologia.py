# app.py

import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import geopandas as gpd
from scipy.interpolate import griddata
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SIHCLI-POTER", page_icon="🌧️", layout="wide")
warnings.filterwarnings("ignore")

# --- 2. IMPORTACIONES ---
try:
    from modules.config import Config
    from modules.data_processor import complete_series, load_and_process_all_data
    from modules.reporter import generate_pdf_report
    
    # Importamos función de tendencias
    try:
        from modules.analysis import calculate_trends_mann_kendall
    except ImportError:
        calculate_trends_mann_kendall = None

    try:
        import modules.db_manager as db_manager
        DB_AVAILABLE = True
    except ImportError:
        DB_AVAILABLE = False

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
        roi = gdf_zona.buffer(0.05)
        path_muni = os.path.join("data", "MunicipiosAntioquia.geojson") 
        path_cuenca = os.path.join("data", "SubcuencasAinfluencia.geojson")
        
        if os.path.exists(path_muni):
            gdf_m = gpd.read_file(path_muni).to_crs("EPSG:4326")
            gdf_c = gpd.clip(gdf_m, roi)
            gdf_c.columns = gdf_c.columns.str.lower()
            for _, r in gdf_c.iterrows():
                name = get_name_from_row_v2(r, 'muni')
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', 
                        line=dict(width=0.7, color='rgba(100, 100, 100, 0.3)', dash='dot'), 
                        hoverinfo='text', text=f"Mpio: {name}", showlegend=False
                    ))
        
        if os.path.exists(path_cuenca):
            gdf_cu = gpd.read_file(path_cuenca).to_crs("EPSG:4326")
            gdf_c = gpd.clip(gdf_cu, roi)
            gdf_c.columns = gdf_c.columns.str.lower()
            for _, r in gdf_c.iterrows():
                name = get_name_from_row_v2(r, 'cuenca')
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', 
                        line=dict(width=0.7, color='rgba(50, 100, 200, 0.3)', dash='dash'), 
                        hoverinfo='text', text=f"Cuenca: {name}", showlegend=False
                    ))
    except Exception as e: print(f"Error capas fantasma: {e}")

def interpolacion_suave(points, values, grid_x, grid_y):
    """Interpolación Cúbica para contornos suaves (HD)."""
    try:
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
        mask = np.isnan(grid_z)
        if np.any(mask): # Rellenar huecos con nearest
            grid_n = griddata(points, values, (grid_x, grid_y), method='nearest')
            grid_z[mask] = grid_n[mask]
        return grid_z
    except:
        return griddata(points, values, (grid_x, grid_y), method='linear')

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
    if DB_AVAILABLE:
        try:
            db_manager.init_db()
        except:
            pass

    for k in ["lz_raster_result", "lz_profile", "lz_names", "lz_colors"]:
        if k not in st.session_state:
            st.session_state[k] = None

    # --- B. CARGA DE DATOS ---
    data_loaded = False
    with st.spinner("Cargando datos..."):
        try:
            (
                gdf_stations, gdf_municipios, df_long,
                df_enso, gdf_subcuencas, gdf_predios,
            ) = load_and_process_all_data()
            data_loaded = True
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            st.stop()

    if not data_loaded or gdf_stations is None or df_long is None:
        st.error("No se pudieron cargar los datos.")
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

            if st.checkbox("✅ Seleccionar Todas", value=True):
                stations_for_analysis = st.multiselect("Estaciones:", options=stations_avail, default=stations_avail)
            else:
                stations_for_analysis = st.multiselect("Estaciones:", options=stations_avail, default=[])

        # --- 4. TIEMPO Y LIMPIEZA ---
        with st.expander("⏳ Tiempo y Limpieza", expanded=False):
            min_year = int(df_long[Config.YEAR_COL].min())
            max_year = int(df_long[Config.YEAR_COL].max())
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

    # 2. Enrutador de Módulos (Corregido y Compatible)
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

    # --- NUEVO MÓDULO INTEGRADO (Isoyetas HD) ---
    elif selected_module == "✨ Mapas Isoyetas HD":
        st.header("🗺️ Mapas de Isoyetas de Alta Definición")
        st.info("Interpolación espacial suavizada (Cubic Spline) con capas de contexto fantasma.")
        
        # Usamos gdf_filtered (que ya está calculado en app.py) para definir el zoom inicial
        if gdf_filtered is not None and not gdf_filtered.empty:
            minx, miny, maxx, maxy = gdf_filtered.total_bounds
            
            # Inicializamos motor de BD localmente para este bloque
            engine = create_engine(st.secrets["DATABASE_URL"])
            
            col_iso1, col_iso2 = st.columns([1, 3])
            
            with col_iso1:
                st.subheader("Configuración")
                year_iso = st.selectbox("Seleccionar Año:", range(2025, 1980, -1))
            
            with col_iso2:
                try:
                    q_iso = text(f"""
                        SELECT e.id_estacion, e.nom_est, ST_X(e.geom::geometry) as lon, ST_Y(e.geom::geometry) as lat,
                               SUM(p.precipitation) as valor
                        FROM precipitacion_mensual p
                        JOIN estaciones e ON p.id_estacion_fk = e.id_estacion
                        WHERE extract(year from p.fecha_mes_año) = :anio
                        AND ST_X(e.geom::geometry) BETWEEN :minx AND :maxx 
                        AND ST_Y(e.geom::geometry) BETWEEN :miny AND :maxy
                        GROUP BY e.id_estacion, e.nom_est, e.geom
                    """)
                    
                    df_iso = pd.read_sql(q_iso, engine, params={"anio": year_iso, "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})
                    
                    if len(df_iso) >= 3:
                        with st.spinner(f"Interpolando datos de {len(df_iso)} estaciones..."):
                            gx, gy = np.mgrid[minx:maxx:200j, miny:maxy:200j]
                            grid_z = interpolacion_suave(df_iso[['lon', 'lat']].values, df_iso['valor'].values, gx, gy)
                            
                            fig_m = go.Figure()
                            
                            # 1. Isoyetas Suaves
                            fig_m.add_trace(go.Contour(
                                z=grid_z.T, x=np.linspace(minx, maxx, 200), y=np.linspace(miny, maxy, 200),
                                colorscale="YlGnBu", 
                                colorbar=dict(title="mm/año"),
                                hovertemplate="Lluvia: %{z:.0f} mm<extra></extra>",
                                contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                                opacity=0.8, connectgaps=True, line_smoothing=1.3
                            ))
                            
                            # 2. Capas Fantasma (Usamos gdf_filtered como zona base)
                            add_context_layers_ghost(fig_m, gdf_filtered)
                            
                            # 3. Puntos Estaciones
                            fig_m.add_trace(go.Scatter(
                                x=df_iso['lon'], y=df_iso['lat'], mode='markers',
                                marker=dict(size=6, color='black', line=dict(width=1, color='white')),
                                text=df_iso['nom_est'] + ': ' + df_iso['valor'].round(0).astype(str) + ' mm',
                                hoverinfo='text'
                            ))
                            
                            fig_m.update_layout(
                                title=f"Isoyetas Año {year_iso}", 
                                height=650, 
                                xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False),
                                margin=dict(l=0,r=0,t=40,b=0), plot_bgcolor='white'
                            )
                            st.plotly_chart(fig_m, use_container_width=True)
                    else:
                        st.warning("⚠️ Datos insuficientes en esta zona/año para interpolar.")
                except Exception as e:
                    st.error(f"Error al generar mapa: {e}")
        else:
            st.info("👈 Seleccione estaciones en el menú lateral para generar el mapa.")

    # Ajuste CSS para Tabs internas
    st.markdown("""<style>.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }</style>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()