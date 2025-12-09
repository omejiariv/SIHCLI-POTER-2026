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
    
    # Intentamos importar db_manager
    try:
        import modules.db_manager as db_manager
        DB_AVAILABLE = True
    except ImportError:
        DB_AVAILABLE = False

    from modules.visualizer import (
        display_advanced_maps_tab,
        display_anomalies_tab,
        display_climate_forecast_tab,
        display_climate_scenarios_tab,
        display_correlation_tab,
        display_current_filters,
        display_drought_analysis_tab,
        display_graphs_tab,
        display_land_cover_analysis_tab,
        display_life_zones_tab,
        display_realtime_dashboard,
        display_spatial_distribution_tab,
        display_station_table_tab,
        display_stats_tab,
        display_trends_and_forecast_tab,
        display_welcome_tab,
    )
except Exception as e:
    st.error(f"Error crítico importando módulos: {e}")
    st.stop()


def main():
    # --- A. INICIALIZACIÓN ---
    if DB_AVAILABLE:
        try:
            db_manager.init_db()
        except Exception:
            pass

    for k in ["lz_raster_result", "lz_profile", "lz_names", "lz_colors"]:
        if k not in st.session_state:
            st.session_state[k] = None

    # --- B. CARGA DE DATOS ---
    data_loaded = False
    with st.spinner("Cargando datos..."):
        try:
            (
                gdf_stations,
                gdf_municipios,
                df_long,
                df_enso,
                gdf_subcuencas,
                gdf_predios,
            ) = load_and_process_all_data()
            data_loaded = True
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            st.stop()

    if not data_loaded or gdf_stations is None or df_long is None:
        st.error("No se pudieron cargar los datos. Revise la conexión.")
        st.stop()

    # --- C. FILTROS (BARRA LATERAL / SIDEBAR) ---
    with st.sidebar:
        st.header("🎛️ Filtros")
        
        # 1. ESTACIONES
        st.markdown("##### Estaciones")
        if Config.STATION_NAME_COL in df_long.columns:
            lista_estaciones = df_long[Config.STATION_NAME_COL].unique()
            stations_for_analysis = st.multiselect(
                "Seleccione Estación(es):",
                options=lista_estaciones,
                default=lista_estaciones
            )
        else:
            st.error(f"Columna {Config.STATION_NAME_COL} no encontrada.")
            stations_for_analysis = []

        st.markdown("---") # Separador visual

        # 2. RANGO DE AÑOS
        st.markdown("##### Rango de Años")
        min_year = int(df_long[Config.YEAR_COL].min())
        max_year = int(df_long[Config.YEAR_COL].max())
        
        year_range = st.slider(
            "Seleccione periodo:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

        st.markdown("---") # Separador visual

        # 3. OPCIONES
        st.markdown("##### Opciones")
        apply_interp = st.checkbox(
            "Aplicar Interpolación", value=False, key="apply_interpolation"
        )
        analysis_mode = "Anual"

    # --- D. APLICAR FILTROS ---
    mask_base = (
        (df_long[Config.YEAR_COL] >= year_range[0])
        & (df_long[Config.YEAR_COL] <= year_range[1])
        & (df_long[Config.STATION_NAME_COL].isin(stations_for_analysis))
    )
    
    df_monthly_filtered = df_long.loc[mask_base].copy()
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

    # Variables vacías (legacy support)
    sel_regions = []
    sel_munis = []
    selected_months = list(range(1, 13))

    display_args = {
        "df_long": df_monthly_filtered,
        "df_complete": df_monthly_filtered,
        "gdf_stations": gdf_stations,
        "gdf_filtered": gdf_filtered,
        "gdf_municipios": gdf_municipios,
        "gdf_subcuencas": gdf_subcuencas,
        "gdf_predios": gdf_predios,
        "df_enso": df_enso,
        "stations_for_analysis": stations_for_analysis,
        "df_anual_melted": df_anual_melted,
        "df_monthly_filtered": df_monthly_filtered,
        "analysis_mode": analysis_mode,
        "selected_regions": sel_regions,
        "selected_municipios": sel_munis,
        "selected_months": selected_months,
        "year_range": year_range,
        "start_date": start_date,
        "end_date": end_date,
    }

    # --- E. RENDERIZADO ---
    try:
        display_current_filters(
            stations_sel=stations_for_analysis,
            regions_sel=sel_regions,
            munis_sel=sel_munis,
            year_range=year_range,
            interpolacion="Si" if apply_interp else "No",
            df_data=df_monthly_filtered,
        )
    except Exception:
        pass

    tab_titles = [
        "🏠 Inicio", "🚨 Monitoreo", "🗺️ Distribución", "📈 Gráficos", 
        "📊 Estadísticas", "🔮 Pronóstico Climático", "📉 Tendencias", 
        "⚠️ Anomalías", "🔗 Correlación", "🌊 Extremos", 
        "🌍 Mapas Avanzados", "🧪 Sesgo", "🌿 Cobertura", 
        "🌱 Zonas Vida", "🌡️ Clima Futuro", "📄 Reporte"
    ]

    tabs = st.tabs(tab_titles)

    with tabs[0]:
        display_welcome_tab()
    
    with tabs[1]:
        display_realtime_dashboard(df_monthly_filtered, gdf_stations, gdf_filtered)
    
    with tabs[2]:
        display_spatial_distribution_tab(
            user_loc=None, 
            interpolacion="Si" if apply_interp else "No", 
            **display_args
        )
    
    with tabs[3]:
        display_graphs_tab(**display_args)
    
    with tabs[4]:
        display_stats_tab(**display_args)
        st.markdown("---")
        display_station_table_tab(**display_args)

    with tabs[5]:
        display_climate_forecast_tab(**display_args)
    
    with tabs[6]:
        display_trends_and_forecast_tab(**display_args)
    
    with tabs[7]:
        display_anomalies_tab(**display_args)
    
    with tabs[8]:
        display_correlation_tab(**display_args)
    
    with tabs[9]:
        display_drought_analysis_tab(**display_args)
    
    with tabs[10]:
        display_advanced_maps_tab(**display_args)

    with tabs[11]:
        try:
            from modules.visualizer import display_bias_correction_tab
            display_bias_correction_tab(**display_args)
        except Exception:
            st.info("Módulo Sesgo cargando...")

    with tabs[12]:
        display_land_cover_analysis_tab(**display_args)
    
    with tabs[13]:
        display_life_zones_tab(**display_args)
    
    with tabs[14]:
        display_climate_scenarios_tab(**display_args)

    with tabs[15]:
        st.header("Reporte PDF")
        if st.button("Generar Reporte"):
            res = {
                "n_estaciones": len(stations_for_analysis), 
                "rango": f"{year_range}"
            }
            pdf = generate_pdf_report(df_monthly_filtered, gdf_filtered, res)
            if pdf:
                st.download_button(
                    "Descargar", pdf, "reporte.pdf", "application/pdf"
                )

    st.markdown(
        """<style>.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }</style>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
