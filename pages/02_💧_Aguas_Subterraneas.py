# pages/02_💧_Aguas_Subterraneas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
import geopandas as gpd
import os
import sys


import folium
from folium.features import DivIcon
from folium import plugins
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata
# ----------------------------------------

# --- IMPORTACIÓN DE MÓDULOS (BLINDADA) ---
try:
    from modules import db_manager, hydrogeo_utils, selectors
    from modules.config import Config
    
    # Módulos opcionales con manejo de fallo
    try: from modules import land_cover
    except ImportError: land_cover = None
        
    try: from modules import analysis
    except ImportError: analysis = None
        
except ImportError as e:
    st.error(f"Error importando módulos del sistema: {e}")
    st.stop()

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

if st.sidebar.button("🧹 Limpiar Memoria y Recargar"):
    st.cache_data.clear()
    st.rerun()

# --- 1. SELECTOR ESPACIAL (CONECTADO AL SELECTOR ARREGLADO) ---
ids_estaciones, nombre_zona, altitud_ref, gdf_zona = selectors.render_selector_espacial()
engine = db_manager.get_engine()

# --- 2. PARÁMETROS ECO-HIDROLÓGICOS ---
st.sidebar.divider()
st.sidebar.header("🎛️ Parámetros del Modelo")

RUTA_RASTER = "data/Cob25m_WGS84.tif"

modo_params = st.sidebar.radio(
    "Fuente de Coberturas:", 
    ["Automático (Satélite)", "Manual (Simulación)"],
    horizontal=True
)

# Valores por defecto
pct_bosque, pct_agricola, pct_pecuario, pct_agua, pct_urbano = 40.0, 20.0, 30.0, 5.0, 5.0

# Lógica de Coberturas
if modo_params == "Automático (Satélite)" and gdf_zona is not None and land_cover:
    with st.sidebar.status("🛰️ Analizando territorio..."):
        try:
            stats_raw = land_cover.calcular_estadisticas_zona(gdf_zona, RUTA_RASTER)
            p_bosque, p_agricola, p_pecuario, p_agua, p_urbano = land_cover.agrupar_coberturas_turc(stats_raw)
            
            if stats_raw:
                st.sidebar.success("✅ Datos extraídos del satélite")
                pct_bosque, pct_agricola, pct_pecuario, pct_agua, pct_urbano = p_bosque, p_agricola, p_pecuario, p_agua, p_urbano
                
                # Visualización rápida en sidebar
                st.sidebar.progress(int(pct_bosque), text=f"Bosque: {pct_bosque:.0f}%")
                st.sidebar.progress(int(pct_pecuario + pct_agricola), text=f"Agro: {(pct_pecuario+pct_agricola):.0f}%")
            else:
                st.sidebar.warning("⚠️ Sin datos raster en la zona. Usando valores manuales.")
        except Exception as e:
            st.sidebar.error(f"Error procesando raster: {e}")
else:
    if modo_params == "Automático (Satélite)" and not land_cover:
        st.sidebar.warning("Módulo land_cover no disponible.")
        
    pct_bosque = st.sidebar.number_input("% Bosque", 0, 100, 40)
    pct_agricola = st.sidebar.number_input("% Agrícola", 0, 100, 20)
    pct_pecuario = st.sidebar.number_input("% Pecuario", 0, 100, 30)
    pct_agua = st.sidebar.number_input("% Agua/Humedal", 0, 100, 5)
    pct_urbano = max(0, 100 - (pct_bosque + pct_agricola + pct_pecuario + pct_agua))
    st.sidebar.metric("% Urbano / Otro", f"{pct_urbano}%")

# --- FACTORES HIDROGEOLÓGICOS ---
st.sidebar.subheader("🌱 Suelo (Infiltración)")
tipo_suelo = st.sidebar.select_slider(
    "Textura Dominante:",
    options=["Arcilloso (Baja)", "Franco-Arcilloso", "Franco (Media)", "Franco-Arenoso", "Arenoso (Alta)"],
    value="Franco (Media)"
)
mapa_factores_suelo = {"Arcilloso (Baja)": 0.6, "Franco-Arcilloso": 0.8, "Franco (Media)": 1.0, "Franco-Arenoso": 1.2, "Arenoso (Alta)": 1.35}
factor_suelo = mapa_factores_suelo[tipo_suelo]

st.sidebar.subheader("🪨 Geología (Recarga)")
tipo_geo = st.sidebar.select_slider(
    "Permeabilidad del Acuífero:",
    options=["Muy Baja (Granitos)", "Baja", "Media (Sedimentarias)", "Alta", "Muy Alta (Aluvial/Kárstico)"],
    value="Media (Sedimentarias)"
)
mapa_kg = {"Muy Baja (Granitos)": 0.3, "Baja": 0.5, "Media (Sedimentarias)": 0.7, "Alta": 0.85, "Muy Alta (Aluvial/Kárstico)": 0.95}
kg_factor = mapa_kg[tipo_geo]

# Cálculo de Coeficientes Ponderados
kc_ponderado = ((pct_bosque * 1.0) + (pct_agricola * 0.85) + (pct_pecuario * 0.80) + (pct_agua * 1.05) + (pct_urbano * 0.40)) / 100.0
ki_cobertura = ((pct_bosque * 0.50) + (pct_agricola * 0.30) + (pct_pecuario * 0.30) + (pct_agua * 0.90) + (pct_urbano * 0.05)) / 100.0
ki_final = max(0.01, min(0.95, ki_cobertura * factor_suelo))

c1, c2 = st.sidebar.columns(2)
c1.metric("Infiltración Est.", f"{(ki_final*100):.0f}%")
c2.metric("Recarga Potencial", f"{(kg_factor*100):.0f}%")

st.sidebar.divider()
meses_futuros = st.sidebar.slider("Horizonte Pronóstico", 12, 60, 24)
ruido = st.sidebar.slider("Factor Incertidumbre", 0.0, 1.0, 0.1)

# --- LÓGICA DE DATOS ---
if gdf_zona is not None:
    
    # 1. Recuperar Estaciones (Consulta Geoespacial si faltan IDs)
    if not ids_estaciones:
        if gdf_zona.crs and gdf_zona.crs.to_string() != "EPSG:4326":
            gdf_zona = gdf_zona.to_crs("EPSG:4326")
            
        minx, miny, maxx, maxy = gdf_zona.total_bounds
        buff = 0.05
        
        # Consulta usando columnas corregidas (latitud/longitud)
        q_geo = text(f"""
            SELECT id_estacion, nombre, latitud, longitud, altitud, municipio 
            FROM estaciones 
            WHERE longitud BETWEEN {minx-buff} AND {maxx+buff} 
            AND latitud BETWEEN {miny-buff} AND {maxy+buff}
        """)
        df_puntos = pd.read_sql(q_geo, engine)
        
        if not df_puntos.empty:
            ids_estaciones = df_puntos['id_estacion'].astype(str).tolist()
    else:
        # Consulta por IDs específicos
        ids_fmt = ",".join([f"'{x}'" for x in ids_estaciones])
        q = text(f"SELECT id_estacion, nombre, latitud, longitud, altitud, municipio FROM estaciones WHERE id_estacion IN ({ids_fmt})")
        df_puntos = pd.read_sql(q, engine)

    if df_puntos.empty:
        st.warning("❌ No se encontraron estaciones en esta zona.")
        st.stop()

    # 2. Procesamiento Hidrológico
    with st.spinner("Procesando balance hídrico y recarga..."):
        
        # Obtener datos de lluvia
        # Priorizamos la tabla 'precipitacion' nueva
        ids_fmt = ",".join([f"'{x}'" for x in ids_estaciones])
        q_rain = text(f"""
            SELECT id_estacion, fecha, valor 
            FROM precipitacion 
            WHERE id_estacion IN ({ids_fmt})
            ORDER BY fecha ASC
        """)
        df_raw = pd.read_sql(q_rain, engine)
        
        # Ejecutar Modelo Prophet (Pronóstico)
        df_res = pd.DataFrame()
        if not df_raw.empty:
            # Asegurar tipos
            df_raw['id_estacion'] = df_raw['id_estacion'].astype(str)
            df_raw['fecha'] = pd.to_datetime(df_raw['fecha'])
            
            alt_calc = altitud_ref if altitud_ref else df_puntos['altitud'].mean()
            
            # Llamada al núcleo hidrogeológico
            df_res = hydrogeo_utils.ejecutar_pronostico_prophet(
                df_raw, meses_futuros, alt_calc, ki_final, ruido, kg=kg_factor, kc=kc_ponderado
            )

    st.markdown(f"### Análisis: {nombre_zona}")

    # ==============================================================================
    # 1. PANEL SUPERIOR DE INDICADORES
    # ==============================================================================
    if not df_res.empty:
        df_hist = df_res[df_res['tipo'] == 'Histórico']
        
        if not df_hist.empty:
            # --- A. CÁLCULO DE ÁREA ---
            area_km2 = 0
            try:
                # Limpieza de nombre para búsqueda SQL
                if isinstance(nombre_zona, list): n_busq = str(nombre_zona[0])
                else: n_busq = str(nombre_zona)
                
                n_busq = n_busq.replace("['", "").replace("']", "").strip()
                
                # Buscar en Cuencas
                q_area = text("SELECT area_km2 FROM cuencas WHERE nombre_cuenca ILIKE :n OR CAST(subc_lbl AS TEXT) ILIKE :n LIMIT 1")
                df_a = pd.read_sql(q_area, engine, params={'n': f"%{n_busq}%"})
                
                if not df_a.empty:
                    area_km2 = df_a.iloc[0]['area_km2']
                else:
                    # Buscar en Municipios
                    q_mun = text("SELECT area_km2 FROM municipios WHERE nombre_municipio ILIKE :n LIMIT 1")
                    df_m = pd.read_sql(q_mun, engine, params={'n': f"%{n_busq}%"})
                    if not df_m.empty: area_km2 = df_m.iloc[0]['area_km2']
            except: pass
            
            if area_km2 <= 0: area_km2 = 10.0 # Valor por defecto seguro

            # --- B. CÁLCULOS AGREGADOS ---
            # Promedios mensuales * 12 = Anuales
            p_med = df_hist['p_final'].mean() * 12
            etr_med = df_hist['etr_mm'].mean() * 12
            rec_med = df_hist['recarga_mm'].mean() * 12
            inf_med = df_hist['infiltracion_mm'].mean() * 12
            esc_med = df_hist['escorrentia_mm'].mean() * 12
            
            # Caudales (m3/s)
            # Q = (Lluvia_mm * Area_km2 * 1000) / (365 * 24 * 3600)
            segundos_anio = 31536000
            q_base_m3s = (rec_med * area_km2 * 1000) / segundos_anio
            q_medio_m3s = (esc_med * area_km2 * 1000) / segundos_anio
            
            # Estadísticas Extremas (usando analysis.py si existe)
            q_min_50a, q_eco = 0, 0
            if analysis:
                try:
                    serie_p = df_hist.set_index('fecha')['p_final']
                    # Coeficiente escorrentía directo aprox
                    c_dir = (esc_med - rec_med) / p_med if p_med > 0 else 0.3
                    
                    stats = analysis.calculate_hydrological_statistics(
                        serie_p, runoff_coeff=c_dir, area_km2=area_km2, q_base_m3s=q_base_m3s
                    )
                    q_min_50a = stats.get("Q_Min_50a", 0)
                    q_eco = stats.get("Q_Ecologico_Q95", 0)
                except: pass

            # --- C. VISUALIZACIÓN DE MÉTRICAS (10 COLUMNAS) ---
            st.markdown("##### 💧 Balance Hídrico y Oferta Subterránea (Promedios Multianuales)")
            cols = st.columns(10)
            
            def fmt(v, u=""): return f"{v:,.0f} {u}"
            
            cols[0].metric("📏 Área", f"{area_km2:,.1f} km²")
            cols[1].metric("🌧️ Lluvia", fmt(p_med, "mm/año"))
            cols[2].metric("☀️ ETR", fmt(etr_med, "mm/año"))
            cols[3].metric("🌱 Infiltración", fmt(inf_med, "mm/año"))
            cols[4].metric("💧 Recarga", fmt(rec_med, "mm/año"), help="Oferta hídrica subterránea")
            cols[5].metric("🌊 Escorrentía", fmt(esc_med, "mm/año"))
            cols[6].metric("⚖️ Q. Medio", f"{q_medio_m3s:.2f} m³/s")
            cols[7].metric("📉 Q. Min 50a", f"{q_min_50a:.3f} m³/s", delta_color="inverse", help="Caudal mínimo probable (Tr=50 años)")
            cols[8].metric("🐟 Q. Ecológico", f"{q_eco:.3f} m³/s", help="Caudal ambiental (Q95)")
            cols[9].metric("📡 Estaciones", len(df_puntos))

    st.divider()

    # ==============================================================================
    # 2. PREPARACIÓN DE DATOS ESPACIALES (CRÍTICO: Define df_mapa_stats)
    # ==============================================================================
    # Inicializamos df_mapa_stats con los metadatos básicos
    df_mapa_stats = df_puntos.copy()
    
    # Si hay datos de lluvia, enriquecemos los puntos
    if 'df_raw' in locals() and not df_raw.empty:
        try:
            # 1. Agrupar lluvia histórica por estación
            grp = df_raw.groupby('id_estacion')['valor'].agg(['mean', 'std']).reset_index()
            grp.columns = ['id_estacion', 'p_media', 'std_lluvia']
            
            # 2. Unir con df_mapa_stats
            # Aseguramos tipos string para el join
            df_mapa_stats['id_estacion'] = df_mapa_stats['id_estacion'].astype(str)
            grp['id_estacion'] = grp['id_estacion'].astype(str)
            
            df_mapa_stats = pd.merge(df_mapa_stats, grp, on='id_estacion', how='left')
            
            # 3. Calcular Balance Puntual (Turc) para el mapa
            # T = 28 - 0.006*h
            df_mapa_stats['temp'] = 28 - (0.006 * df_mapa_stats['altitud'])
            # L = 300 + 25T + 0.05T^3
            df_mapa_stats['L_turc'] = 300 + 25*df_mapa_stats['temp'] + 0.05*(df_mapa_stats['temp']**3)
            
            # ETR y Recarga
            def calc_etr(row):
                if pd.isna(row['p_media']) or row['L_turc'] == 0: return 0
                return row['p_media'] / np.sqrt(0.9 + (row['p_media']/row['L_turc'])**2)

            df_mapa_stats['etr_media'] = df_mapa_stats.apply(calc_etr, axis=1)
            
            # Factores globales (del sidebar)
            factor_recarga = ki_final * kg_factor
            df_mapa_stats['recarga_calc'] = (df_mapa_stats['p_media'] - df_mapa_stats['etr_media']) * factor_recarga
            df_mapa_stats['escorrentia_media'] = df_mapa_stats['p_media'] - df_mapa_stats['etr_media'] - df_mapa_stats['recarga_calc']
            
        except Exception as e:
            st.warning(f"Advertencia calculando mapa: {e}")

    # ==============================================================================
    # 3. PESTAÑAS DE ANÁLISIS
    # ==============================================================================
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Serie Completa", "🗺️ Mapa Contexto", "💧 Mapa Recarga", "📥 Descargas"])

    # --- GUÍA TÉCNICA ENRIQUECIDA ---
    with st.expander("📘 Guía Técnica: Metodología, Ecuaciones e Interpretación", expanded=False):
        t1, t2, t3 = st.tabs(["🧮 Ecuaciones", "⚙️ Modelo Estocástico", "📖 Interpretación"])
        
        with t1:
            st.markdown(r"""
            #### 1. Balance Hídrico de Largo Plazo
            Se utiliza el método de **Turc Modificado** para estimar la oferta hídrica en cuencas tropicales.
            
            **Ecuación Fundamental:**
            $$P = ETR + E_s + R + \Delta S$$
            
            Donde:
            * **$P$ (Precipitación):** Entrada total de agua al sistema (mm/año).
            * **$ETR$ (Evapotranspiración Real):** Agua que retorna a la atmósfera por evaporación del suelo y transpiración de plantas. Se calcula en función de la Temperatura ($T$) y la Lluvia ($P$):
                $$ETR = \frac{P}{\sqrt{0.9 + (\frac{P}{L})^2}} \quad \text{donde} \quad L = 300 + 25T + 0.05T^3$$
            * **$R$ (Recarga Potencial):** Fracción del agua que se infiltra profundamente y alimenta el acuífero.
            * **$E_s$ (Escorrentía):** Flujo superficial rápido hacia los cauces.
            """)
            
        with t2:
            st.markdown(r"""
            #### 2. Análisis de Extremos (Caudales)
            Para la gestión del riesgo y concesiones, no basta con el promedio. Analizamos los extremos usando distribuciones de probabilidad:
            
            * **📉 Caudales Mínimos (Sequías):** Se ajustan a una distribución **Log-Normal de 2 Parámetros**.
                * **$Q_{Min}^{50a}$:** El caudal mínimo esperado una vez cada 50 años (crítico para abastecimiento).
                * **$Q_{95}$ (Ecológico):** El caudal que es superado el 95% del tiempo (garantía de sostenibilidad biótica).
            
            * **📈 Caudales Máximos (Crecientes):** Se ajustan a una distribución de **Gumbel (Valores Extremos Tipo I)**.
                * Permite estimar cotas de inundación para periodos de retorno de 2.33, 5, 10, 50 y 100 años.
            """)
            
        with t3:
            st.info("""
            **Interpretación de Resultados:**
            * **Rendimiento Hídrico ($m^3/ha-año$):** Indica cuánta agua produce cada hectárea de la cuenca. Zonas boscosas suelen tener mayor rendimiento de regulación (Recarga).
            * **Recarga vs. Escorrentía:** Una cuenca sana busca maximizar la Recarga (línea azul oscura en la gráfica) y moderar la Escorrentía superficial, reduciendo el riesgo de erosión e inundaciones.
            """)

    # --- TAB 1: SERIE TEMPORAL DETALLADA ---
    with tab1:
        if not df_res.empty:
            # 1. Agrupar promedio regional (Mes a Mes)
            df_avg = df_res.groupby(['fecha', 'tipo'])[[
                'p_final', 'etr_mm', 'infiltracion_mm', 'recarga_mm', 
                'escorrentia_mm', 'yhat_upper', 'yhat_lower'
            ]].mean().reset_index().sort_values('fecha')

            # 2. Cálculo de Rendimiento Hídrico (m³/ha)
            # Factor 10: 1 mm = 10 m3/ha
            # .clip(lower=0) asegura que no haya rendimientos negativos (físicamente imposibles)
            df_avg['rendimiento_m3ha'] = ((df_avg['escorrentia_mm'] + df_avg['recarga_mm']) * 10).clip(lower=0)

            # 3. Gráfica Multivariable
            fig = go.Figure()
            df_hist = df_avg[df_avg['tipo'] == 'Histórico']
            df_fut = df_avg[df_avg['tipo'] == 'Proyección']
            
            # --- Variables de Entrada/Salida ---
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['p_final'], name='🌧️ Lluvia Histórica', line=dict(color='#95a5a6', width=1), visible='legendonly'))
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['etr_mm'], name='☀️ ETR', line=dict(color='#e67e22', width=1, dash='dot')))
            
            # --- Variables del Suelo ---
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['infiltracion_mm'], name='🌱 Infiltración Pot.', line=dict(color='#2ecc71', width=1.5)))
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['escorrentia_mm'], name='🌊 Escorrentía', line=dict(color='#27ae60', width=1.5)))
            
            # --- Variables Objetivo (Recarga) ---
            # Recarga Real (Histórica)
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['recarga_mm'], name='💧 Recarga Real', line=dict(color='#2980b9', width=3), fill='tozeroy'))
            
            # PROYECCIONES (Futuro)
            if not df_fut.empty:
                # Lluvia Proyectada (Nueva)
                fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['p_final'], name='🌧️ Lluvia Proyectada', line=dict(color='#bdc3c7', width=1, dash='dot')))
                
                # Recarga Proyectada
                fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['recarga_mm'], name='🔮 Recarga Proy.', line=dict(color='#00d2d3', width=2, dash='dot')))
                
                # Incertidumbre
                fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_upper'], showlegend=False, line=dict(width=0), hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_lower'], name='Incertidumbre', fill='tonexty', line=dict(width=0), fillcolor='rgba(0,210,211,0.1)', hoverinfo='skip'))

            # --- Eje Secundario: Rendimiento Hídrico ---
            fig.add_trace(go.Scatter(
                x=df_hist['fecha'], y=df_hist['rendimiento_m3ha'], 
                name='🚜 Rendimiento (m³/ha)', 
                line=dict(color='#8e44ad', width=1),
                yaxis='y2', opacity=0.3
            ))

            # Layout con doble eje
            fig.update_layout(
                title=f"Dinámica Hidrológica Completa: {nombre_zona}",
                height=550, 
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1),
                yaxis=dict(title="Lámina de Agua (mm/mes)"),
                yaxis2=dict(title="Rendimiento (m³/ha)", overlaying='y', side='right', showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📅 Ver Tabla de Datos Completa"):
                st.dataframe(df_avg, use_container_width=True)
        else:
            st.info("Sin datos suficientes para el balance.")


    # --- TAB 2: MAPA DE CONTEXTO ---
    with tab2:
        # CORRECCIÓN DE ERROR: Agregamos key='btn_ctx_uniq' para evitar ID duplicado
        if st.button("🔄 Recargar Mapa Contexto", key="btn_ctx_uniq"): st.rerun()
        
        try:
            pad = 0.05
            min_lat, max_lat = df_puntos['latitud'].min(), df_puntos['latitud'].max()
            min_lon, max_lon = df_puntos['longitud'].min(), df_puntos['longitud'].max()
            
            m = folium.Map(location=[(min_lat+max_lat)/2, (min_lon+max_lon)/2], zoom_start=11, tiles="CartoDB positron")
            m.fit_bounds([[min_lat-pad, min_lon-pad], [max_lat+pad, max_lon+pad]])

            st.markdown("<style>.leaflet-tooltip {white-space: normal !important; max-width: 300px;}</style>", unsafe_allow_html=True)

            # Cargar capas externas (si existen)
            if hasattr(hydrogeo_utils, 'cargar_capas_gis_optimizadas'):
                try: 
                    layers = hydrogeo_utils.cargar_capas_gis_optimizadas(engine, [min_lon-pad, min_lat-pad, max_lon+pad, max_lat+pad])
                    
                    if 'hidro' in layers:
                        folium.GeoJson(layers['hidro'], name="Hidrogeología", 
                            style_function=lambda x: {'color': '#2c3e50', 'weight': 0.5, 'fillOpacity': 0.3}
                        ).add_to(m)
                except: pass

            # Capa Raster (Coberturas)
            if land_cover and os.path.exists(RUTA_RASTER) and gdf_zona is not None:
                try:
                    img_cob, bounds_cob = land_cover.obtener_imagen_folium_coberturas(gdf_zona, RUTA_RASTER)
                    if img_cob is not None:
                        folium.raster_layers.ImageOverlay(img_cob, bounds_cob, opacity=0.6, name="Coberturas").add_to(m)
                except: pass

            # Marcadores Estaciones
            fg = folium.FeatureGroup(name="Estaciones", show=True)
            for _, r in df_mapa_stats.iterrows():
                if pd.notnull(r.get('latitud')) and pd.notnull(r.get('longitud')):
                    # Tooltip seguro
                    p_val = r.get('p_media', 0) * 12
                    r_val = r.get('recarga_calc', 0) * 12
                    
                    html = f"""
                    <b>{r.get('nombre')}</b><br>
                    ID: {r.get('id_estacion')}<br>
                    🌧️ Lluvia: {p_val:,.0f} mm<br>
                    💧 Recarga: {r_val:,.0f} mm
                    """
                    folium.Marker(
                        [r['latitud'], r['longitud']], 
                        popup=folium.Popup(html, max_width=200),
                        icon=folium.Icon(color='blue', icon='tint')
                    ).add_to(fg)
            
            fg.add_to(m)
            folium.LayerControl().add_to(m)
            st_folium(m, width=1400, height=600, key=f"map_ctx_{nombre_zona}")

        except Exception as e:
            st.error(f"Error renderizando mapa: {e}")

    # --- TAB 3: MAPA DE RECARGA (INTERPOLACIÓN) ---
    with tab3:
        st.subheader("Distribución Espacial de la Recarga")
        df_valid = df_mapa_stats.dropna(subset=['recarga_calc'])
        
        if len(df_valid) < 4:
            st.warning("⚠️ Se requieren al menos 4 estaciones con datos para interpolar.")
        else:
            try:
                # Datos para interpolar
                x = df_valid['longitud'].values
                y = df_valid['latitud'].values
                z = df_valid['recarga_calc'].values * 12 # mm/año
                
                # Grid
                pad = 0.05
                xi = np.linspace(x.min()-pad, x.max()+pad, 100)
                yi = np.linspace(y.min()-pad, y.max()+pad, 100)
                Xi, Yi = np.meshgrid(xi, yi)
                
                # Interpolación Linear
                Zi = griddata((x, y), z, (Xi, Yi), method='linear')
                
                # Mapa
                m_iso = folium.Map(location=[y.mean(), x.mean()], zoom_start=11, tiles="CartoDB positron")
                m_iso.fit_bounds([[y.min(), x.min()], [y.max(), x.max()]])
                
                # Renderizar Raster
                if not np.isnan(Zi).all():
                    vmin, vmax = np.nanmin(Zi), np.nanmax(Zi)
                    cmap = cm.get_cmap('Blues')
                    norm_z = (Zi - vmin) / (vmax - vmin)
                    rgba = cmap(norm_z)
                    rgba[np.isnan(Zi), 3] = 0
                    
                    folium.raster_layers.ImageOverlay(
                        image=rgba, bounds=[[yi.min(), xi.min()], [yi.max(), xi.max()]], 
                        opacity=0.7, name="Recarga (Raster)"
                    ).add_to(m_iso)
                
                folium.LayerControl().add_to(m_iso)
                st_folium(m_iso, width=1400, height=600, key=f"map_iso_{nombre_zona}")
                
            except Exception as e:
                st.error(f"Error en interpolación: {e}")

    # --- TAB 4: DESCARGAS ---
    with tab4:
        col1, col2 = st.columns(2)
        if not df_res.empty:
            col1.download_button("⬇️ Descargar Serie Temporal (.csv)", df_res.to_csv(index=False).encode('utf-8'), "balance.csv", "text/csv")
        if not df_mapa_stats.empty:
            col2.download_button("⬇️ Descargar Datos Estaciones (.csv)", df_mapa_stats.to_csv(index=False).encode('utf-8'), "estaciones_recarga.csv", "text/csv")

    # ==============================================================================
    # 4. REPORTE GLOBAL (GENERADOR MAESTRO)
    # ==============================================================================
    st.markdown("---")
    with st.expander("📑 Reporte Maestro de Cuencas (Generación Masiva)", expanded=False):
        st.info("Herramienta administrativa para recalcular todas las cuencas.")
        
        # Selección de columna de nombre
        try:
            q_cols = text("SELECT column_name FROM information_schema.columns WHERE table_name = 'cuencas' AND data_type = 'text'")
            cols_bd = pd.read_sql(q_cols, engine)['column_name'].tolist()
            idx_def = next((i for i, c in enumerate(cols_bd) if c in ['n_nss3', 'subc_lbl', 'nombre']), 0)
            col_nom_rep = st.selectbox("Columna de Nombre:", cols_bd, index=idx_def, key="col_rep_final")
        except: col_nom_rep = 'nombre_cuenca'

        if st.button("🚀 Generar Reporte Completo", key="btn_gen_rep"):
            try:
                import rasterio
                from rasterio.mask import mask
                
                with st.spinner("Procesando todas las cuencas (esto tomará unos minutos)..."):
                    # 1. Cargar Geometrías
                    gdf_all = gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry").to_crs("EPSG:3116")
                    
                    # 2. Cargar Estaciones (Usando coordenadas reparadas)
                    q_est = text("SELECT id_estacion, ST_SetSRID(ST_MakePoint(CAST(longitud AS FLOAT), CAST(latitud AS FLOAT)), 4326) as geometry FROM estaciones WHERE latitud IS NOT NULL")
                    gdf_est = gpd.read_postgis(q_est, engine, geom_col="geometry").to_crs("EPSG:3116")
                    gdf_est['id_estacion'] = gdf_est['id_estacion'].astype(str)
                    
                    # 3. Lluvia Anual
                    df_rain = pd.read_sql("SELECT id_estacion, AVG(valor)*12 as ppt FROM precipitacion GROUP BY id_estacion", engine)
                    df_rain['id_estacion'] = df_rain['id_estacion'].astype(str)
                    
                    # 4. Bucle
                    res = []
                    prog = st.progress(0)
                    for i, row in gdf_all.iterrows():
                        nom = str(row.get(col_nom_rep, f"Cuenca {i}"))
                        area = row.geometry.area / 1e6
                        
                        # Buffer y cruce
                        buf = row.geometry.buffer(20000)
                        est_in = gdf_est[gdf_est.geometry.within(buf)]
                        
                        ppt = 2000
                        if not est_in.empty:
                            ids = est_in['id_estacion'].tolist()
                            vals = df_rain[df_rain['id_estacion'].isin(ids)]['ppt']
                            if not vals.empty: ppt = vals.mean()
                        
                        # Balance simple para reporte
                        temp = max(0, 28 - 0.006 * 1500) # Altitud media aprox 1500
                        L = 300 + 25*temp + 0.05*(temp**3)
                        etr = ppt / np.sqrt(0.9 + (ppt/L)**2) if L>0 else 0
                        recarga = (ppt - etr) * 0.15 # Factor regional medio
                        
                        res.append({
                            "Cuenca": nom, "Area_km2": round(area,2), 
                            "Lluvia_mm": round(ppt,0), "Recarga_mm": round(recarga,0),
                            "Q_Medio_m3s": round((ppt-etr)*area*1000/31536000, 3)
                        })
                        prog.progress((i+1)/len(gdf_all))
                    
                    # Guardar
                    pd.DataFrame(res).to_sql("reporte_cuencas", engine, if_exists='replace', index=False)
                    st.success("Reporte generado correctamente.")
                    st.dataframe(pd.DataFrame(res).head())
                    
            except Exception as e:
                st.error(f"Error en reporte: {e}")