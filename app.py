# modules/app.py

import warnings
import pandas as pd
import streamlit as st

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SIHCLI-POTER", page_icon="🌧️", layout="wide")
warnings.filterwarnings("ignore")

# --- 2. IMPORTACIONES ---
try:
    from modules.config import Config
    from modules.data_processor import complete_series, load_and_process_all_data
    from modules.reporter import generate_pdf_report
    
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

# --- FUNCIÓN AUXILIAR PARA DETECTAR COLUMNAS ---
def get_fuzzy_col(df, keywords):
    """Busca si alguna columna contiene alguna de las palabras clave."""
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

    # --- C. FILTROS AVANZADOS (BARRA LATERAL) ---
    with st.sidebar:
        st.title("🎛️ Panel de Control")
        
        # 1. GESTIÓN DE DATOS
        with st.expander("📂 Gestión de Datos", expanded=False):
            st.info("Fuente: GitHub (Automático)")
            if st.button("🔄 Recargar Datos"):
                st.cache_data.clear()
                st.rerun()
            if st.checkbox("Carga Manual", value=False):
                st.file_uploader("Subir CSV", type=["csv", "txt"])

        # 2. FILTROS GEOGRÁFICOS
        # Detectar columnas automáticamente (Región/Zona y Altitud/Elevación)
        col_region = get_fuzzy_col(gdf_stations, ["region", "zon", "cuenca", "dpto"])
        col_muni = get_fuzzy_col(gdf_stations, ["muni", "ciud", "city"])
        col_alt = get_fuzzy_col(gdf_stations, ["alt", "elev", "cota", "height"])

        with st.expander("🗺️ Filtros Geográficos", expanded=True):
            # A. Regiones
            list_regions = []
            sel_regions = []
            if col_region:
                # Normalizamos a string para evitar errores de tipo
                list_regions = sorted(gdf_stations[col_region].astype(str).unique())
                sel_regions = st.multiselect(f"📍 Región ({col_region}):", list_regions, default=[])
            else:
                st.warning("⚠️ No se detectó columna de Región.")
                # Descomenta esto si quieres ver los nombres reales para depurar:
                # st.caption(f"Cols: {list(gdf_stations.columns)}")

            # B. Municipios (Cascada)
            list_munis = []
            sel_munis = []
            if col_muni:
                # Filtramos GDF temporalmente para la cascada
                if sel_regions and col_region:
                    gdf_temp = gdf_stations[gdf_stations[col_region].isin(sel_regions)]
                else:
                    gdf_temp = gdf_stations
                
                list_munis = sorted(gdf_temp[col_muni].astype(str).unique())
                sel_munis = st.multiselect(f"🏙️ Municipio ({col_muni}):", list_munis, default=[])

            # C. Altitud
            rango_alt = None
            if col_alt:
                try:
                    # Convertir a numérico forzado por si hay texto sucio
                    gdf_stations[col_alt] = pd.to_numeric(gdf_stations[col_alt], errors='coerce')
                    min_a = float(gdf_stations[col_alt].min())
                    max_a = float(gdf_stations[col_alt].max())
                    
                    # Solo mostramos si hay datos válidos
                    if pd.notnull(min_a) and pd.notnull(max_a) and min_a < max_a:
                        rango_alt = st.slider(
                            "⛰️ Altitud (msnm):", 
                            int(min_a), int(max_a), (int(min_a), int(max_a))
                        )
                except Exception as e:
                    st.caption(f"Error en filtro altitud: {e}")
            else:
                st.warning("⚠️ No se detectó columna de Altitud.")

        # 3. SELECCIÓN DE ESTACIONES
        with st.expander("🌧️ Selección de Estaciones", expanded=True):
            mask_geo = pd.Series(True, index=gdf_stations.index)
            
            # Aplicar filtros geo acumulativos
            if sel_regions and col_region:
                mask_geo &= gdf_stations[col_region].isin(sel_regions)
            if sel_munis and col_muni:
                mask_geo &= gdf_stations[col_muni].isin(sel_munis)
            if rango_alt and col_alt:
                mask_geo &= (gdf_stations[col_alt] >= rango_alt[0]) & (gdf_stations[col_alt] <= rango_alt[1])
                
            stations_avail = gdf_stations.loc[mask_geo, Config.STATION_NAME_COL].unique()
            
            st.markdown(f"**Disponibles:** {len(stations_avail)}")

            # Botón maestro Seleccionar/Deseleccionar
            if st.checkbox("✅ Seleccionar Todas", value=True):
                stations_for_analysis = st.multiselect(
                    "Estaciones:", options=stations_avail, default=stations_avail
                )
            else:
                stations_for_analysis = st.multiselect(
                    "Estaciones:", options=stations_avail, default=[]
                )

        # 4. TIEMPO Y LIMPIEZA
        with st.expander("⏳ Tiempo y Limpieza", expanded=True):
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

    # --- D. APLICAR FILTROS (PROCESAMIENTO) ---
    mask_base = (
        (df_long[Config.YEAR_COL] >= year_range[0])
        & (df_long[Config.YEAR_COL] <= year_range[1])
        & (df_long[Config.STATION_NAME_COL].isin(stations_for_analysis))
    )
    
    df_monthly_filtered = df_long.loc[mask_base].copy()
    
    # Limpieza
    if ignore_zeros:
        df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] != 0]
    if ignore_nulls:
        df_monthly_filtered = df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL])

    # Filtrar GDF final
    gdf_filtered = gdf_stations[gdf_stations[Config.STATION_NAME_COL].isin(stations_for_analysis)]

    if apply_interp:
        with st.spinner("Interpolando..."):
            df_monthly_filtered = complete_series(df_monthly_filtered)
    
    df_anual_melted = (
        df_monthly_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL])[
            Config.PRECIPITATION_COL
        ]
        .sum()
        .reset_index()
    )

    start_date = pd.to_datetime(f"{year_range[0]}-01-01")
    end_date = pd.to_datetime(f"{year_range[1]}-12-31")

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
    }

    # --- E. RENDERIZADO ---
    try:
        display_current_filters(
            stations_sel=stations_for_analysis, regions_sel=sel_regions,
            munis_sel=sel_munis, year_range=year_range,
            interpolacion="Si" if apply_interp else "No", df_data=df_monthly_filtered,
        )
    except:
        pass

    tab_titles = [
        "🏠 Inicio", "🚨 Monitoreo", "🗺️ Distribución", "📈 Gráficos", 
        "📊 Estadísticas", "🔮 Pronóstico Climático", "📉 Tendencias", 
        "⚠️ Anomalías", "🔗 Correlación", "🌊 Extremos", 
        "🌍 Mapas Avanzados", "🧪 Sesgo", "🌿 Cobertura", 
        "🌱 Zonas Vida", "🌡️ Clima Futuro", "📄 Reporte"
    ]

    tabs = st.tabs(tab_titles)

    with tabs[0]: display_welcome_tab()
    with tabs[1]: display_realtime_dashboard(df_monthly_filtered, gdf_stations, gdf_filtered)
    with tabs[2]: display_spatial_distribution_tab(user_loc=None, interpolacion="Si" if apply_interp else "No", **display_args)
    with tabs[3]: display_graphs_tab(**display_args)
    with tabs[4]: 
        display_stats_tab(**display_args)
        st.markdown("---")
        display_station_table_tab(**display_args)
    with tabs[5]: display_climate_forecast_tab(**display_args)
    with tabs[6]: display_trends_and_forecast_tab(**display_args)
    with tabs[7]: display_anomalies_tab(**display_args)
    with tabs[8]: display_correlation_tab(**display_args)
    with tabs[9]: display_drought_analysis_tab(**display_args)
    with tabs[10]: display_advanced_maps_tab(**display_args)
    with tabs[11]:
        try:
            display_bias_correction_tab(**display_args)
        except:
            st.info("Módulo Sesgo cargando...")
    with tabs[12]: display_land_cover_analysis_tab(**display_args)
    with tabs[13]: display_life_zones_tab(**display_args)
    with tabs[14]: display_climate_scenarios_tab(**display_args)
    with tabs[15]:
        st.header("Reporte PDF")
        if st.button("Generar Reporte"):
            res = {"n_estaciones": len(stations_for_analysis), "rango": f"{year_range}"}
            pdf = generate_pdf_report(df_monthly_filtered, gdf_filtered, res)
            if pdf:
                st.download_button("Descargar", pdf, "reporte.pdf", "application/pdf")

    st.markdown("""<style>.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }</style>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()