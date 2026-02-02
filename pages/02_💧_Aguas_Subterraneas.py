# pages/02_💧_Aguas_Subterraneas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
import geopandas as gpd
import os
import sys

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
            st.markdown("##### 💧 Balance Hídrico y Oferta Subterránea")
            cols = st.columns(10)
            
            def fmt(v, u=""): return f"{v:,.0f} {u}"
            
            cols[0].metric("📏 Área", f"{area_km2:,.1f} km²")
            cols[1].metric("🌧️ Lluvia", fmt(p_med, "mm"))
            cols[2].metric("☀️ ETR", fmt(etr_med, "mm"))
            cols[3].metric("🌱 Infiltración", fmt(inf_med, "mm"))
            cols[4].metric("💧 Recarga", fmt(rec_med, "mm"), help="Entrada al acuífero")
            cols[5].metric("🌊 Escorrentía", fmt(esc_med, "mm"))
            cols[6].metric("⚖️ Q. Medio", f"{q_medio_m3s:.2f} m³/s")
            cols[7].metric("📉 Q. Min 50a", f"{q_min_50a:.3f}", delta_color="inverse")
            cols[8].metric("🐟 Q. Ecológico", f"{q_eco:.3f}")
            cols[9].metric("📡 Estaciones", len(df_puntos))

    st.divider()

# ==============================================================================
    # 2. PESTAÑAS DE ANÁLISIS DETALLADO
    # ==============================================================================
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Serie Completa", "🗺️ Mapa Contexto", "💧 Mapa Recarga", "📥 Descargas"])

    # --- GUÍA TÉCNICA (Global) ---
    with st.expander("📘 Guía Técnica: Balance Hídrico y Recarga", expanded=False):
        st.markdown(r"""
        **Ecuación Fundamental:** $R = P - ETR - E_s$
        * **$P$:** Precipitación (Entrada).
        * **$ETR$:** Evapotranspiración Real (Pérdida a la atmósfera).
        * **$E_s$:** Escorrentía Superficial (Pérdida por flujo rápido).
        * **$R$:** Recarga Potencial (Infiltración profunda al acuífero).
        
        **Metodología:**
        1. **Turc Modificado:** Para estimar ETR en función de Temperatura y Lluvia.
        2. **Balance de Suelos:** Estimación de Escorrentía usando coeficientes de cobertura ($K_c$) y suelo ($K_s$).
        """)

    # --------------------------------------------------------------------------
    # TAB 1: ANÁLISIS TEMPORAL (BALANCE)
    # --------------------------------------------------------------------------
    with tab1:
        if not df_res.empty:
            # Agrupar por fecha para obtener el promedio regional
            # df_res viene del módulo hydrogeo_utils con nombres estandarizados
            df_avg = df_res.groupby(['fecha', 'tipo'])[[
                'p_final', 'etr_mm', 'infiltracion_mm', 'recarga_mm', 
                'escorrentia_mm', 'yhat_upper', 'yhat_lower'
            ]].mean().reset_index().sort_values('fecha')

            # --- A. GRÁFICA DE BALANCE ---
            df_hist = df_avg[df_avg['tipo'] == 'Histórico']
            df_fut = df_avg[df_avg['tipo'] == 'Proyección']
            
            fig = go.Figure()
            
            # Histórico
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['p_final'], name='Lluvia (Entrada)', line=dict(color='#95a5a6', width=1)))
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['etr_mm'], name='ETR (Salida)', line=dict(color='#e67e22', width=1.5)))
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['recarga_mm'], name='Recarga (Acuífero)', line=dict(color='#2980b9', width=2.5), fill='tozeroy'))
            
            # Proyección
            if not df_fut.empty:
                fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['p_final'], name='Lluvia Proyectada', line=dict(color='#95a5a6', width=1, dash='dot')))
                fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['recarga_mm'], name='Recarga Proyectada', line=dict(color='#00d2d3', width=2, dash='dot')))
                
                # Banda de Incertidumbre
                fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_upper'], showlegend=False, line=dict(width=0)))
                fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_lower'], name='Incertidumbre', fill='tonexty', line=dict(width=0), fillcolor='rgba(0,210,211,0.1)'))
            
            fig.update_layout(
                title=f"Balance Hídrico Regional: {nombre_zona}",
                yaxis_title="Lámina de Agua (mm)",
                height=500, hovermode="x unified",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- B. TABLA DE DATOS ---
            with st.expander("📅 Tabla de Datos Mensuales"):
                st.dataframe(df_avg.style.format("{:.1f}", subset=['p_final', 'etr_mm', 'recarga_mm']), use_container_width=True)

        else:
            st.info("Seleccione una zona con estaciones para ver el balance.")

    # --------------------------------------------------------------------------
    # TAB 2: MAPA DE CONTEXTO
    # --------------------------------------------------------------------------
    with tab2:
        if not df_puntos.empty:
            # Centro del mapa
            lat_center = df_puntos['latitud'].mean()
            lon_center = df_puntos['longitud'].mean()
            
            m = folium.Map(location=[lat_center, lon_center], zoom_start=10, tiles="CartoDB positron")
            
            # Polígono de la Zona (Cuenca/Municipio)
            if gdf_zona is not None:
                # Simplificar para renderizado rápido
                sim_geo = gdf_zona.to_crs("EPSG:4326").geometry.simplify(0.001)
                folium.GeoJson(
                    sim_geo,
                    style_function=lambda x: {'fillColor': '#3498db', 'color': '#2980b9', 'weight': 2, 'fillOpacity': 0.1}
                ).add_to(m)

            # Puntos de Estaciones
            for _, row in df_puntos.iterrows():
                folium.CircleMarker(
                    location=[row['latitud'], row['longitud']],
                    radius=5, color='black', fill=True, fill_color='white',
                    popup=f"<b>{row['nombre']}</b><br>ID: {row['id_estacion']}<br>Alt: {row['altitud']}m"
                ).add_to(m)
            
            st_folium(m, width=None, height=500)
        else:
            st.warning("No hay estaciones para mostrar en el mapa.")

    # --------------------------------------------------------------------------
    # TAB 3: MAPA DE RECARGA (INTERPOLACIÓN)
    # --------------------------------------------------------------------------
    with tab3:
        if not df_res.empty and len(df_puntos) >= 3:
            st.subheader("Distribución Espacial de la Recarga")
            
            # 1. Calcular Recarga Promedio por Estación
            # df_res tiene datos temporales. Necesitamos agrupar por estación.
            # Pero df_res (output de hydrogeo) a veces viene agregado. 
            # Re-calculamos recarga puntual usando los datos crudos filtrados.
            
            # Traemos datos crudos de nuevo para el mapa espacial
            ids_sql = tuple(df_puntos['id_estacion'].astype(str).tolist())
            if ids_sql:
                q_spatial = text(f"""
                    SELECT id_estacion, AVG(valor) as p_media 
                    FROM precipitacion 
                    WHERE id_estacion IN :ids 
                    GROUP BY id_estacion
                """)
                df_spatial_rain = pd.read_sql(q_spatial, engine, params={'ids': ids_sql})
                
                # Unir con coordenadas
                df_map_recarga = pd.merge(df_puntos, df_spatial_rain, on='id_estacion', how='inner')
                
                # Aplicar Modelo Turc Puntual
                # T = 28 - 0.006 * h
                df_map_recarga['temp'] = 28 - (0.006 * df_map_recarga['altitud'])
                # L = 300 + 25T + 0.05T^3
                df_map_recarga['L'] = 300 + 25*df_map_recarga['temp'] + 0.05*(df_map_recarga['temp']**3)
                # ETR = P / sqrt(0.9 + (P/L)^2)
                df_map_recarga['etr'] = df_map_recarga['p_media'] / np.sqrt(0.9 + (df_map_recarga['p_media']/df_map_recarga['L'])**2)
                # Recarga = (P - ETR) * Coeficientes (Simplificado para mapa)
                # Usamos los coeficientes globales del sidebar para el mapa visual
                factor_global = ki_final * kg_factor
                df_map_recarga['recarga_anual'] = (df_map_recarga['p_media'] - df_map_recarga['etr']) * factor_global * 12 # Anualizado
                
                # Graficar Mapa Interpolado (Plotly Contour)
                fig_map = go.Figure()
                
                # Interpolación simple
                grid_x = np.linspace(df_map_recarga['longitud'].min(), df_map_recarga['longitud'].max(), 50)
                grid_y = np.linspace(df_map_recarga['latitud'].min(), df_map_recarga['latitud'].max(), 50)
                
                try:
                    from scipy.interpolate import griddata
                    grid_z = griddata(
                        (df_map_recarga['longitud'], df_map_recarga['latitud']), 
                        df_map_recarga['recarga_anual'], 
                        (grid_x[None, :], grid_y[:, None]), 
                        method='cubic'
                    )
                    
                    fig_map.add_trace(go.Contour(
                        z=grid_z, x=grid_x, y=grid_y,
                        colorscale="Blues",
                        colorbar=dict(title="Recarga (mm/año)"),
                        connectgaps=True
                    ))
                    
                    fig_map.add_trace(go.Scatter(
                        x=df_map_recarga['longitud'], y=df_map_recarga['latitud'],
                        mode='markers+text',
                        text=df_map_recarga['recarga_anual'].round(0).astype(str),
                        textposition="top center",
                        marker=dict(color='black', size=5),
                        name="Estaciones"
                    ))
                    
                    fig_map.update_layout(title="Recarga Potencial Anual Estimada", height=600)
                    st.plotly_chart(fig_map, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"No se pudo generar interpolación (se requieren más puntos distribuidos). Mostrando puntos.")
                    st.dataframe(df_map_recarga[['id_estacion', 'nombre', 'recarga_anual']])
        else:
            st.warning("⚠️ Se necesitan al menos 3 estaciones con datos para generar el mapa de recarga.")

    # --------------------------------------------------------------------------
    # TAB 4: DESCARGAS
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("💾 Exportar Datos")
        if not df_res.empty:
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Serie Temporal (CSV)",
                data=csv,
                file_name=f"balance_hidrico_{nombre_zona}.csv",
                mime="text/csv"
            )
            
        if not df_puntos.empty:
            csv_pts = df_puntos.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Inventario Estaciones (CSV)",
                data=csv_pts,
                file_name=f"estaciones_{nombre_zona}.csv",
                mime="text/csv"
            )


# --------------------------------------------------------------------------
    # TAB 2: MAPA DE CONTEXTO (FOLIUM AVANZADO)
    # --------------------------------------------------------------------------
    with tab2:
        if st.button("🔄 Recargar Mapa Contexto"): st.rerun()
        
        # 1. Configuración del Mapa Base
        try:
            pad = 0.05
            # Usamos df_puntos que ya tiene las coordenadas validadas en la Parte 1
            min_lat, max_lat = df_puntos['latitud'].min(), df_puntos['latitud'].max()
            min_lon, max_lon = df_puntos['longitud'].min(), df_puntos['longitud'].max()
            
            m = folium.Map(
                location=[(min_lat + max_lat)/2, (min_lon + max_lon)/2], 
                zoom_start=11, 
                tiles="CartoDB positron"
            )
            # Ajustar vista a los puntos
            m.fit_bounds([[min_lat - pad, min_lon - pad], [max_lat + pad, max_lon + pad]])
        except:
            m = folium.Map(location=[6.2, -75.5], zoom_start=8)

        st.markdown("<style>.leaflet-tooltip {white-space: normal !important; max-width: 300px !important; font-size:11px;}</style>", unsafe_allow_html=True)

        # 2. Carga de Capas Temáticas (Suelos, Hidro, etc.)
        try:
            bounds_list = [min_lon-pad, min_lat-pad, max_lon+pad, max_lat+pad]
            # Intentamos cargar capas si el modulo existe
            if hasattr(hydrogeo_utils, 'cargar_capas_gis_optimizadas'):
                layers = hydrogeo_utils.cargar_capas_gis_optimizadas(engine, bounds_list)
            else:
                layers = {}
        except Exception as e:
            # st.warning(f"No se pudieron cargar capas de contexto: {e}")
            layers = {}

        # Función auxiliar para tooltips seguros
        def tooltip_ok(gdf, dic):
            if gdf is None or gdf.empty: return None
            cols = [c.lower().strip() for c in gdf.columns]
            f, a = [], []
            for k, v in dic.items():
                # Busca columna que contenga la clave (ej: 'potencial' en 'potencial_recarga')
                match = next((c for c in cols if k.lower() in c), None)
                if match:
                    f.append(match)
                    a.append(v)
            return folium.GeoJsonTooltip(fields=f, aliases=a, localize=True) if f else None

        # --- CAPA 1: COBERTURAS (RASTER) ---
        # Si tienes la lógica de raster en land_cover
        if land_cover and os.path.exists(RUTA_RASTER) and gdf_zona is not None:
            try:
                img_cob, bounds_cob = land_cover.obtener_imagen_folium_coberturas(gdf_zona, RUTA_RASTER)
                if img_cob is not None:
                    folium.raster_layers.ImageOverlay(
                        image=img_cob,
                        bounds=bounds_cob,
                        opacity=0.5,
                        name="Coberturas (Satélite)",
                        zindex=1
                    ).add_to(m)
            except: pass

        # --- CAPA 2: SUELOS ---
        if 'suelos' in layers and not layers['suelos'].empty:
            dic_suelos = {'ucs':'UCS:', 'litolo':'Litología:', 'paisaje':'Paisaje:', 'clima':'Clima:'}
            folium.GeoJson(
                layers['suelos'], 
                name="Suelos", 
                style_function=lambda x: {'color':'orange', 'weight':0.5, 'fillOpacity':0.1},
                tooltip=tooltip_ok(layers['suelos'], dic_suelos)
            ).add_to(m)

        # --- CAPA 3: HIDROGEOLOGÍA ---
        if 'hidro' in layers and not layers['hidro'].empty:
            def get_color_hidro(feature):
                props = feature.get('properties', {})
                # Buscamos claves comunes de potencial
                val = props.get('potencial_') or props.get('potencial') or ''
                txt = str(val).lower().strip()
                if 'muy alto' in txt: return '#006400'
                if 'alto' in txt: return '#32CD32'
                if 'medio' in txt: return '#F1C40F'
                if 'muy bajo' in txt: return '#8B0000'
                if 'bajo' in txt: return '#E67E22'
                return '#85C1E9'

            dic_hidro = {'potencial': 'Potencial:', 'unidad': 'Unidad:', 'sigla': 'Sigla:'}
            folium.GeoJson(
                layers['hidro'], 
                name="Hidrogeología", 
                style_function=lambda f: {'fillColor': get_color_hidro(f), 'color': '#2c3e50', 'weight': 0.5, 'fillOpacity': 0.5},
                tooltip=tooltip_ok(layers['hidro'], dic_hidro)
            ).add_to(m)

        # --- CAPA 4: BOCATOMAS ---
        if 'bocatomas' in layers and not layers['bocatomas'].empty:
            dic_boca = {'nombre': 'Nombre:', 'caudal': 'Q (L/s):', 'tipo': 'Tipo:'}
            folium.GeoJson(
                layers['bocatomas'], 
                name="Bocatomas", 
                marker=folium.CircleMarker(radius=4, color='red', fill=True),
                tooltip=tooltip_ok(layers['bocatomas'], dic_boca)
            ).add_to(m)

        # --- CAPA 5: ESTACIONES (MARCADORES RICOS) ---
        fg_estaciones = folium.FeatureGroup(name="Estaciones (Click)", show=True)
        
        # Iteramos sobre df_mapa_stats que tiene los cálculos, pero aseguramos coordenadas
        # Si df_mapa_stats perdió las coordenadas, las recuperamos de df_puntos
        if not df_mapa_stats.empty:
            # Hacemos un merge seguro por si acaso
            if 'latitud' not in df_mapa_stats.columns:
                df_mapa_stats = pd.merge(df_mapa_stats, df_puntos[['id_estacion', 'latitud', 'longitud']], on='id_estacion', how='left')

            for _, r in df_mapa_stats.iterrows():
                # Coordenadas seguras
                lat = r.get('latitud')
                lon = r.get('longitud')
                
                if pd.notnull(lat) and pd.notnull(lon):
                    # Formateo de valores
                    def fmt(v): return f"{v*12:,.0f} mm" if pd.notnull(v) else "N/D"
                    
                    html = f"""
                    <div style='font-family:sans-serif; width:180px; font-size:12px;'>
                        <b style="color:#2980b9;">{r.get('nombre', 'Estación')}</b><br>
                        <span style="font-size:10px; color:gray;">ID: {r.get('id_estacion')}</span>
                        <hr style="margin: 5px 0;">
                        🌧️ Lluvia: <b>{fmt(r.get('p_media'))}</b><br>
                        ☀️ ETR: {fmt(r.get('etr_media'))}<br>
                        💧 <b>Recarga: {fmt(r.get('recarga_calc'))}</b><br>
                    </div>
                    """
                    
                    folium.Marker(
                        [lat, lon],
                        popup=folium.Popup(html, max_width=200),
                        icon=folium.Icon(color='blue', icon='tint', prefix='fa'),
                        tooltip=r.get('nombre')
                    ).add_to(fg_estaciones)

        fg_estaciones.add_to(m)

        # Controles
        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        
        st_folium(m, width=1400, height=600, key=f"mapa_ctx_{nombre_zona}")


# ==============================================================================
    # PREPARACIÓN DE DATOS ESPACIALES (NECESARIO PARA TAB 2 Y 3)
    # ==============================================================================
    # Calculamos estadísticas puntuales por estación para los mapas
    df_mapa_stats = df_puntos.copy()
    if not df_raw.empty:
        # 1. Agrupar lluvia por estación
        grp = df_raw.groupby('id_estacion')['valor'].agg(['mean', 'std']).reset_index()
        grp.columns = ['id_estacion', 'p_media', 'std_lluvia']
        
        # 2. Unir con metadatos
        df_mapa_stats = pd.merge(df_mapa_stats, grp, on='id_estacion', how='left')
        
        # 3. Calcular Balance Puntual (Turc) para el mapa
        # T = 28 - 0.006*h
        df_mapa_stats['temp'] = 28 - (0.006 * df_mapa_stats['altitud'])
        # L = 300 + 25T + 0.05T^3
        df_mapa_stats['L_turc'] = 300 + 25*df_mapa_stats['temp'] + 0.05*(df_mapa_stats['temp']**3)
        # ETR
        df_mapa_stats['etr_media'] = df_mapa_stats.apply(
            lambda x: x['p_media'] / np.sqrt(0.9 + (x['p_media']/x['L_turc'])**2) if x['p_media']>0 and x['L_turc']>0 else 0, 
            axis=1
        )
        # Recarga = (P - ETR) * Coeficientes
        # Usamos los factores globales definidos en el sidebar
        factor_recarga = ki_final * kg_factor
        df_mapa_stats['recarga_calc'] = (df_mapa_stats['p_media'] - df_mapa_stats['etr_media']) * factor_recarga
        df_mapa_stats['escorrentia_media'] = df_mapa_stats['p_media'] - df_mapa_stats['etr_media'] - df_mapa_stats['recarga_calc']

    # ==============================================================================
    # 2. PESTAÑAS VISUALES
    # ==============================================================================
    tab2, tab3, tab4 = st.tabs(["🗺️ Mapa Contexto", "💧 Mapa Recarga", "📥 Descargas"])

    # --- TAB 2: CONTEXTO (TOOLTIPS RICOS) ---
    with tab2:
        if st.button("🔄 Recargar Mapa Contexto"): st.rerun()
        
        try:
            pad = 0.05
            min_lat, max_lat = df_puntos['latitud'].min(), df_puntos['latitud'].max()
            min_lon, max_lon = df_puntos['longitud'].min(), df_puntos['longitud'].max()
            bounds = [min_lon-pad, min_lat-pad, max_lon+pad, max_lat+pad]
            
            # Cargar capas GIS (si existe la función)
            layers = {}
            if hasattr(hydrogeo_utils, 'cargar_capas_gis_optimizadas'):
                try: layers = hydrogeo_utils.cargar_capas_gis_optimizadas(engine, bounds)
                except: pass

            m = folium.Map(location=[(min_lat+max_lat)/2, (min_lon+max_lon)/2], zoom_start=11, tiles="CartoDB positron")
            m.fit_bounds([[min_lat-pad, min_lon-pad], [max_lat+pad, max_lon+pad]])

            # Estilos CSS para Popups
            st.markdown("<style>.leaflet-tooltip {white-space: normal !important; max-width: 300px !important; font-size:11px;}</style>", unsafe_allow_html=True)

            # --- CAPA DE COBERTURAS (RASTER) ---
            if land_cover and os.path.exists(RUTA_RASTER) and gdf_zona is not None:
                try:
                    img_cob, bounds_cob = land_cover.obtener_imagen_folium_coberturas(gdf_zona, RUTA_RASTER)
                    if img_cob is not None:
                        folium.raster_layers.ImageOverlay(
                            image=img_cob, bounds=bounds_cob, opacity=0.6, name="Coberturas (Satélite)", zindex=1
                        ).add_to(m)
                except: pass

            # --- CAPAS VECTORIALES (Suelos, Hidro, etc) ---
            def style_hidro(feature):
                props = feature.get('properties', {})
                val = str(props.get('potencial_', props.get('potencial', ''))).lower()
                c = '#85C1E9'
                if 'alto' in val: c = '#32CD32'
                elif 'medio' in val: c = '#F1C40F'
                elif 'bajo' in val: c = '#E67E22'
                return {'fillColor': c, 'color': '#2c3e50', 'weight': 0.5, 'fillOpacity': 0.5}

            if 'hidro' in layers:
                folium.GeoJson(layers['hidro'], name="Hidrogeología", style_function=style_hidro).add_to(m)

            if 'bocatomas' in layers:
                folium.GeoJson(layers['bocatomas'], name="Bocatomas", marker=folium.CircleMarker(radius=4, color='red')).add_to(m)

            # --- ESTACIONES (Popups Completos) ---
            fg_estaciones = folium.FeatureGroup(name="Estaciones", show=True)

            for _, r in df_mapa_stats.iterrows():
                # Formateador
                def fmt(val, mult=12): return f"{val*mult:,.0f} mm" if pd.notnull(val) else "N/D"
                
                html = f"""
                <div style='font-family:sans-serif; width:200px; font-size:12px;'>
                    <b style="font-size:13px; color:#2c3e50;">{r.get('nombre', 'Estación')}</b>
                    <hr style='margin:4px 0; border-top: 1px solid #ccc;'>
                    📍 <b>Mun:</b> {r.get('municipio', 'N/A')} <br>
                    ⛰️ <b>Alt:</b> {r.get('altitud', 0):,.0f} m <br>
                    <hr style='margin:4px 0; border-top: 1px dashed #ccc;'>
                    🌧️ <b>Lluvia:</b> {fmt(r.get('p_media'))}<br>
                    💧 <b>Recarga:</b> <b style='color:#0000AA;'>{fmt(r.get('recarga_calc'))}</b><br>
                </div>"""
                
                folium.Marker(
                    [r['latitud'], r['longitud']], 
                    popup=folium.Popup(html, max_width=220), 
                    icon=folium.Icon(color='black', icon='tint'),
                    tooltip=r.get('nombre')
                ).add_to(fg_estaciones)

            fg_estaciones.add_to(m)
            folium.LayerControl().add_to(m)
            plugins.Fullscreen().add_to(m)
            st_folium(m, width=1400, height=600, key=f"ctx_{nombre_zona}")

        except Exception as e:
            st.error(f"Error cargando mapa de contexto: {e}")

    # --- TAB 3: RECARGA (INTERPOLACIÓN) ---
    with tab3:
        st.subheader("Distribución Espacial de la Recarga")
        df_valid = df_mapa_stats.dropna(subset=['recarga_calc'])
        
        if len(df_valid) < 4:
            st.warning("⚠️ Se requieren al menos 4 estaciones con datos válidos para interpolar.")
        else:
            try:
                # Interpolación
                x = df_valid['longitud'].values
                y = df_valid['latitud'].values
                z = df_valid['recarga_calc'].values * 12 # Anualizar
                
                # Crear grid
                pad = 0.05
                xi = np.linspace(x.min()-pad, x.max()+pad, 100)
                yi = np.linspace(y.min()-pad, y.max()+pad, 100)
                Xi, Yi = np.meshgrid(xi, yi)
                
                Zi = griddata((x, y), z, (Xi, Yi), method='linear')
                
                # Mapa Base
                m_iso = folium.Map(location=[y.mean(), x.mean()], zoom_start=11, tiles="CartoDB positron")
                m_iso.fit_bounds([[y.min(), x.min()], [y.max(), x.max()]])
                
                # Capa Raster (Colores)
                if not np.isnan(Zi).all():
                    vmin, vmax = np.nanmin(Zi), np.nanmax(Zi)
                    # Colormap simple
                    try: cmap = plt.get_cmap('Blues')
                    except: cmap = cm.Blues
                    
                    norm_z = (Zi - vmin) / (vmax - vmin)
                    rgba = cmap(norm_z)
                    rgba[np.isnan(Zi), 3] = 0 # Transparencia para NaNs
                    
                    folium.raster_layers.ImageOverlay(
                        image=rgba, 
                        bounds=[[yi.min(), xi.min()], [yi.max(), xi.max()]], 
                        opacity=0.7, 
                        name="Recarga (Raster)"
                    ).add_to(m_iso)

                # Isolíneas (Aprox)
                folium.LayerControl().add_to(m_iso)
                st_folium(m_iso, width=1400, height=600, key=f"iso_{nombre_zona}")
                
            except Exception as e:
                st.error(f"Error en interpolación: {e}")

    # --- TAB 4: FICHA TÉCNICA Y DESCARGAS ---
    with tab4:
        # Ficha Técnica Restaurada
        with st.expander("📘 Ficha Técnica: Modelo Hidrológico", expanded=True):
            st.markdown("""
            ### Metodología
            * **Balance Hídrico:** Método de Turc Modificado para el trópico.
            * **Componentes:** $R = P - ETR - Es$. Donde $R$ es Recarga, $P$ Precipitación, $ETR$ Evapotranspiración y $Es$ Escorrentía.
            * **Estadística:** Ajuste Gumbel (Máximos) y Log-Normal (Mínimos) para proyecciones de retorno.
            """)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if not df_res.empty:
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Descargar Serie Temporal (.csv)", csv, "balance_hidrico.csv", "text/csv")
        with col_d2:
            if not df_mapa_stats.empty:
                csv_est = df_mapa_stats.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Descargar Datos Estaciones (.csv)", csv_est, "estaciones_calculadas.csv", "text/csv")


# ==============================================================================
    # SECCIÓN: REPORTE GLOBAL HIDROLÓGICO (GENERADOR MAESTRO)
    # ==============================================================================
    st.markdown("---")
    with st.expander("📑 Reporte Maestro de Cuencas (Tabla Global)", expanded=False):
        
        st.info("Genera tabla maestra con Modelo Aditivo (Escorrentía Directa + Caudal Base) y Estadísticas Extremas.")

        # 1. VER REPORTE EXISTENTE
        try:
            df_reporte_existente = pd.read_sql("SELECT * FROM reporte_cuencas", engine)
            st.success(f"✅ Reporte disponible en BD ({len(df_reporte_existente)} registros).")
            st.dataframe(df_reporte_existente, use_container_width=True)
            csv_ex = df_reporte_existente.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Descargar Tabla (CSV)", csv_ex, "Reporte_Hidrologico_Global.csv", "text/csv")
            st.markdown("---")
        except:
            st.warning("⚠️ Aún no has generado el reporte.")

        # 2. CONFIGURACIÓN DE COLUMNAS
        st.write("#### ⚙️ Configuración")
        try:
            # Buscamos columnas de texto para usarlas como nombre
            q_cols = text("SELECT column_name FROM information_schema.columns WHERE table_name = 'cuencas' AND data_type = 'text'")
            cols_bd = pd.read_sql(q_cols, engine)['column_name'].tolist()
            # Intentamos adivinar la columna correcta
            idx_def = next((i for i, c in enumerate(cols_bd) if c in ['n_nss3', 'subc_lbl', 'nombre', 'name']), 0)
            col_nombre_reporte = st.selectbox("🏷️ Columna para Nombres:", cols_bd, index=idx_def, key="sel_col_rep_final")
        except:
            col_nombre_reporte = 'nombre_cuenca'

        # 3. BOTÓN DE CÁLCULO MASIVO
        if st.button(f"🚀 Generar Reporte Completo (Puede tardar minutos)"):
            try:
                import rasterio
                from rasterio.mask import mask
            except ImportError:
                st.error("Librería 'rasterio' no instalada. No se pueden procesar DEMs.")
                st.stop()
            
            try:
                # A. CARGAR DATOS (ACTUALIZADO A NUEVA BD)
                with st.spinner("Cargando geometrías y normalizando datos..."):
                    
                    # 1. Cuencas (Polígonos)
                    gdf_all = gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry")
                    # Asegurar CRS Magnas-Sirgas (EPSG:3116) para cálculos métricos correctos
                    if gdf_all.crs and gdf_all.crs.to_string() != "EPSG:3116":
                        gdf_all = gdf_all.to_crs("EPSG:3116")
                    
                    # 2. Estaciones (Puntos) - CORRECCIÓN VITAL
                    # Construimos la geometría desde lat/long reparados para ser infalibles
                    # Usamos CAST para asegurar que sean números
                    q_est_geo = text("""
                        SELECT id_estacion, ST_SetSRID(ST_MakePoint(CAST(longitud AS FLOAT), CAST(latitud AS FLOAT)), 4326) as geometry 
                        FROM estaciones
                        WHERE latitud IS NOT NULL AND longitud IS NOT NULL
                    """)
                    gdf_est = gpd.read_postgis(q_est_geo, engine, geom_col="geometry")
                    
                    if gdf_est.crs and gdf_est.crs.to_string() != "EPSG:3116":
                        gdf_est = gdf_est.to_crs("EPSG:3116")
                    gdf_est['id_estacion'] = gdf_est['id_estacion'].astype(str).str.strip()

                    # 3. Lluvias (Datos)
                    # Promedio Anual por Estación
                    df_rain_anual = pd.read_sql("""
                        SELECT id_estacion, AVG(valor)*12 as ppt_anual 
                        FROM precipitacion 
                        GROUP BY id_estacion
                    """, engine)
                    df_rain_anual['id_estacion'] = df_rain_anual['id_estacion'].astype(str).str.strip()

                    # Serie Mensual Completa
                    df_rain_mensual = pd.read_sql("SELECT id_estacion, fecha, valor FROM precipitacion", engine)
                    df_rain_mensual['fecha'] = pd.to_datetime(df_rain_mensual['fecha'])
                    df_rain_mensual['id_estacion'] = df_rain_mensual['id_estacion'].astype(str).str.strip()

                # B. PREPARAR DEM
                path_dem = "data/DemAntioquia_EPSG3116.tif"
                src_dem = None
                crs_dem_objetivo = None
                
                if os.path.exists(path_dem):
                    src_dem = rasterio.open(path_dem)
                    crs_dem_objetivo = src_dem.crs
                    if not crs_dem_objetivo and src_dem.transform[2] > 4000000:
                        crs_dem_objetivo = "EPSG:9377"

                # C. BUCLE DE PROCESAMIENTO
                progreso = st.progress(0)
                status = st.empty()
                lista_resultados = []
                total = len(gdf_all)
                
                for i, row in gdf_all.iterrows():
                    # Obtener nombre seguro
                    nom = str(row.get(col_nombre_reporte, f"Cuenca {i}"))
                    status.text(f"Procesando {i+1}/{total}: {nom}...")
                    
                    # Geometría Base
                    geom_base = row.geometry
                    area_km2 = geom_base.area / 1e6
                    perim_km = geom_base.length / 1000
                    
                    # --- 1. TOPOGRAFÍA ---
                    alt_min, alt_max, alt_med, pend_med = 0, 0, 0, 0
                    ec_hyp = "N/A"
                    
                    if src_dem and crs_dem_objetivo:
                        try:
                            geom_para_dem = gpd.GeoSeries([geom_base], crs="EPSG:3116").to_crs(crs_dem_objetivo).iloc[0]
                            out_image, _ = mask(src_dem, [geom_para_dem], crop=True, nodata=src_dem.nodata)
                            data = out_image[0]
                            validos = data[(data != src_dem.nodata) & (data > -500)]
                            
                            if validos.size > 0:
                                alt_min, alt_max, alt_med = float(np.min(validos)), float(np.max(validos)), float(np.mean(validos))
                                l_caract = np.sqrt(area_km2 * 1e6)
                                if l_caract > 0: pend_med = ((alt_max - alt_min) / l_caract) * 100

                                try:
                                    hist, bins = np.histogram(validos, bins=50)
                                    areas_acum = np.cumsum(hist[::-1]) / validos.size * 100
                                    z = np.polyfit(areas_acum, bins[:-1][::-1], 3)
                                    ec_hyp = f"H = {z[0]:.2e}A³ + {z[1]:.2e}A² + {z[2]:.2e}A + {z[3]:.0f}"
                                except: pass
                        except: pass

                    # --- 2. HIDROLOGÍA Y BALANCE ---
                    if alt_med == 0: alt_med = 1500
                    temp = max(0, 28 - 0.006 * alt_med)
                    L = 300 + 25*temp + 0.05*(temp**3)

                    # Buffer y Lluvias
                    buffer_geom = geom_base.buffer(20000) 
                    est_in = gdf_est[gdf_est.geometry.within(buffer_geom)]
                    n_est = len(est_in)
                    
                    ppt_cuenca = 0
                    if n_est > 0:
                        ids = est_in['id_estacion'].unique().tolist()
                        ppt_vals = df_rain_anual[df_rain_anual['id_estacion'].isin(ids)]['ppt_anual']
                        if not ppt_vals.empty: ppt_cuenca = ppt_vals.mean()
                    else:
                        ppt_cuenca = 2000 # Fallback regional

                    # Balance Turc
                    etr = ppt_cuenca / np.sqrt(0.9 + (ppt_cuenca/L)**2) if (L>0 and ppt_cuenca>0) else 0
                    etr = min(etr, ppt_cuenca)
                    esc_total_anual = ppt_cuenca - etr 
                    
                    # Desglose Hidrogeológico (Factores regionales)
                    inf = esc_total_anual * 0.30 
                    recarga_mm = inf * 0.50 
                    esc_directa_mm = esc_total_anual - inf 
                    
                    # Caudales
                    q_base_m3s = (recarga_mm * area_km2 * 1000) / 31536000
                    q_medio_total = ((esc_directa_mm * area_km2 * 1000)/31536000) + q_base_m3s
                    
                    c_directo = esc_directa_mm / ppt_cuenca if ppt_cuenca > 0 else 0.3

                    # --- 3. ESTADÍSTICAS AVANZADAS ---
                    ec_fdc = "N/A"
                    stats_ext = {}
                    
                    # Verificamos si existe el módulo analysis
                    has_analysis = 'analysis' in locals() or 'analysis' in globals()
                    
                    if n_est > 0 and ppt_cuenca > 0 and has_analysis:
                        try:
                            ids = est_in['id_estacion'].unique().tolist()
                            s_mensual = df_rain_mensual[df_rain_mensual['id_estacion'].isin(ids)]
                            
                            if not s_mensual.empty:
                                # Agrupar por fecha y promediar 'valor'
                                s_sintetica = s_mensual.groupby('fecha')['valor'].mean()
                                
                                # Estadísticas
                                if analysis:
                                    stats_ext = analysis.calculate_hydrological_statistics(
                                        s_sintetica, 
                                        runoff_coeff=c_directo, 
                                        area_km2=area_km2, 
                                        q_base_m3s=q_base_m3s
                                    )
                                    
                                    # Curva de Duración (FDC)
                                    fdc = analysis.calculate_duration_curve(s_sintetica, runoff_coeff=c_directo, area_km2=area_km2, q_base_m3s=q_base_m3s)
                                    if fdc: ec_fdc = fdc.get("equation", "N/A")
                        except: pass

                    # Índices
                    im = ppt_cuenca / (temp + 10)
                    ifow = (ppt_cuenca**2) / ppt_cuenca if ppt_cuenca > 0 else 0

                    # --- 4. CONSTRUIR FILA ---
                    fila = {
                        "Cuenca": nom,
                        "Área (km²)": round(area_km2, 2),
                        "Perímetro (km)": round(perim_km, 2),
                        "Altitud Media": round(alt_med, 0),
                        "Altitud Máx": round(alt_max, 0),
                        "Altitud Mín": round(alt_min, 0),
                        "Pendiente (%)": round(pend_med, 2),
                        
                        "Lluvia (mm)": round(ppt_cuenca, 0),
                        "ETR (mm)": round(etr, 0),
                        "Infiltración (mm)": round(inf, 0),
                        "Recarga (mm)": round(recarga_mm, 0),
                        "Escorrentía Directa (mm)": round(esc_directa_mm, 0),
                        
                        "Caudal Base (m³/s)": round(q_base_m3s, 3),
                        "Caudal Medio Total (m³/s)": round(q_medio_total, 3),
                        "Estaciones (20km)": n_est,
                        
                        "I. Martonne": round(im, 2),
                        "I. Fournier": round(ifow, 2),
                        "Ec. Hipsométrica": ec_hyp,
                        "Ec. FDC": ec_fdc,

                        # Estadísticas
                        "Desviación Std": round(stats_ext.get("Desviacion_Std", 0), 3),
                        "Q Ecológico (Q95)": round(stats_ext.get("Q_Ecologico_Q95", 0), 3),
                        
                        # Máximos
                        "Q Max 2.33a": round(stats_ext.get("Q_Max_2.33a", 0), 3),
                        "Q Max 50a": round(stats_ext.get("Q_Max_50a", 0), 3),
                        "Q Max 100a": round(stats_ext.get("Q_Max_100a", 0), 3),
                        
                        # Mínimos
                        "Q Min 50a": round(stats_ext.get("Q_Min_50a", 0), 3),
                    }

                    lista_resultados.append(fila)
                    progreso.progress((i+1)/total)

                # GUARDAR EN BD
                df_final = pd.DataFrame(lista_resultados)
                df_final.to_sql("reporte_cuencas", engine, if_exists='replace', index=False)
                
                progreso.empty()
                status.success(f"✅ ¡Reporte Generado Exitosamente! ({len(df_final)} Cuencas procesadas).")
                st.rerun()

            except Exception as e:
                st.error(f"Error crítico durante la generación del reporte: {e}")