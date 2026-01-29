# pages/02_💧_Aguas_Subterraneas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
import geopandas as gpd
from scipy.interpolate import griddata
import folium
from folium.features import DivIcon
from folium import plugins
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# Importaciones de módulos propios
from modules import db_manager, hydrogeo_utils, selectors
from modules import land_cover
# Intentamos importar analysis para las curvas, si falla no rompe la app
try:
    from modules import analysis
except ImportError:
    analysis = None

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

if st.sidebar.button("🧹 Limpiar Memoria y Recargar"):
    st.cache_data.clear()
    st.rerun()

# --- 1. SELECTOR ESPACIAL ---
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

pct_bosque, pct_agricola, pct_pecuario, pct_agua, pct_urbano = 40.0, 20.0, 30.0, 5.0, 5.0

if modo_params == "Automático (Satélite)" and gdf_zona is not None:
    with st.sidebar.status("🛰️ Analizando territorio..."):
        stats_raw = land_cover.calcular_estadisticas_zona(gdf_zona, RUTA_RASTER)
        p_bosque, p_agricola, p_pecuario, p_agua, p_urbano = land_cover.agrupar_coberturas_turc(stats_raw)
    
    if not stats_raw:
        st.sidebar.warning("⚠️ Sin datos raster. Usando default.")
    else:
        st.sidebar.success("✅ Datos extraídos")
        pct_bosque, pct_agricola, pct_pecuario, pct_agua, pct_urbano = p_bosque, p_agricola, p_pecuario, p_agua, p_urbano
        st.sidebar.progress(int(pct_bosque), text=f"Bosque: {pct_bosque:.0f}%")
        st.sidebar.progress(int(pct_pecuario + pct_agricola), text=f"Agropecuario: {(pct_pecuario+pct_agricola):.0f}%")
        st.sidebar.caption(f"Urbano: {pct_urbano:.1f}% | Agua: {pct_agua:.1f}%")
else:
    pct_bosque = st.sidebar.number_input("% Bosque", 0, 100, 40)
    pct_agricola = st.sidebar.number_input("% Agrícola", 0, 100, 20)
    pct_pecuario = st.sidebar.number_input("% Pecuario", 0, 100, 30)
    pct_agua = st.sidebar.number_input("% Agua/Humedal", 0, 100, 5)
    pct_urbano = max(0, 100 - (pct_bosque + pct_agricola + pct_pecuario + pct_agua))
    st.sidebar.metric("% Urbano / Otro", f"{pct_urbano}%")

# --- FACTORES ---
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
    options=["Muy Baja (Granitos/Arcillolitas)", "Baja", "Media (Sedimentarias)", "Alta", "Muy Alta (Aluvial/Kárstico)"],
    value="Media (Sedimentarias)"
)
mapa_kg = {"Muy Baja (Granitos/Arcillolitas)": 0.3, "Baja": 0.5, "Media (Sedimentarias)": 0.7, "Alta": 0.85, "Muy Alta (Aluvial/Kárstico)": 0.95}
kg_factor = mapa_kg[tipo_geo]

kc_ponderado = ((pct_bosque * 1.0) + (pct_agricola * 0.85) + (pct_pecuario * 0.80) + (pct_agua * 1.05) + (pct_urbano * 0.40)) / 100.0
ki_cobertura = ((pct_bosque * 0.50) + (pct_agricola * 0.30) + (pct_pecuario * 0.30) + (pct_agua * 0.90) + (pct_urbano * 0.05)) / 100.0
ki_final = max(0.01, min(0.95, ki_cobertura * factor_suelo))

c1, c2 = st.sidebar.columns(2)
c1.metric("Infiltración", f"{(ki_final*100):.0f}%")
c2.metric("Recarga Real", f"{(kg_factor*100):.0f}%")

st.sidebar.divider()
meses_futuros = st.sidebar.slider("Horizonte", 12, 60, 24)
ruido = st.sidebar.slider("Incertidumbre", 0.0, 1.0, 0.1)

# --- LÓGICA DE DATOS ---
if gdf_zona is not None:
    # 1. Recuperar Estaciones
    if not ids_estaciones:
        minx, miny, maxx, maxy = gdf_zona.total_bounds
        buff = 0.05
        q_geo = text(f"""
            SELECT id_estacion, nom_est, latitud, longitud, alt_est, municipio 
            FROM estaciones 
            WHERE longitud BETWEEN {minx-buff} AND {maxx+buff} 
            AND latitud BETWEEN {miny-buff} AND {maxy+buff}
        """)
        df_puntos = pd.read_sql(q_geo, engine)
        
        if not df_puntos.empty:
            try:
                points = gpd.points_from_xy(df_puntos.longitud, df_puntos.latitud)
                gdf_pts = gpd.GeoDataFrame(df_puntos, geometry=points, crs="EPSG:4326")
                
                if gdf_zona.crs is None: gdf_zona = gdf_zona.set_crs("EPSG:4326")
                else: gdf_zona = gdf_zona.to_crs("EPSG:4326")
                
                df_joined = gpd.sjoin(gdf_pts, gdf_zona, how="inner", predicate="intersects")
                if not df_joined.empty: 
                    df_puntos = df_joined[df_puntos.columns].copy()
            except: pass
            
            ids_estaciones = df_puntos['id_estacion'].tolist()
    else:
        if len(ids_estaciones) == 1:
            q = text(f"SELECT id_estacion, nom_est, latitud, longitud, alt_est, municipio FROM estaciones WHERE id_estacion = '{ids_estaciones[0]}'")
            df_puntos = pd.read_sql(q, engine)
        else:
            q = text("SELECT id_estacion, nom_est, latitud, longitud, alt_est, municipio FROM estaciones WHERE id_estacion IN :ids")
            df_puntos = pd.read_sql(q, engine, params={'ids': tuple(ids_estaciones)})

    if df_puntos.empty:
        st.error("❌ No se encontraron estaciones.")
        st.stop()

    # 2. Estadísticas y Datos
    with st.spinner("Procesando hidrología..."):
        df_mapa_stats = hydrogeo_utils.obtener_estadisticas_estaciones(engine, df_puntos)
        
        df_raw = pd.DataFrame()
        intentos = [
            ('precipitacion', 'id_estacion', 'fecha', 'valor'), 
            ('precipitacion_mensual', 'id_estacion_fk', 'fecha_mes_año', 'precipitation')
        ]
        
        for tbl, col_id, col_f, col_v in intentos:
            try:
                # --- CORRECCIÓN CLAVE AQUÍ ---
                # Agregamos "{col_id} as id_estacion" a la consulta SQL
                if len(ids_estaciones) == 1:
                    q = text(f"SELECT {col_id} as id_estacion, {col_f} as fecha, {col_v} as valor FROM {tbl} WHERE {col_id} = '{ids_estaciones[0]}'")
                    df_temp = pd.read_sql(q, engine)
                else:
                    q = text(f"SELECT {col_id} as id_estacion, {col_f} as fecha, {col_v} as valor FROM {tbl} WHERE {col_id} IN :ids")
                    df_temp = pd.read_sql(q, engine, params={'ids': tuple(ids_estaciones)})
                
                if not df_temp.empty:
                    df_raw = df_temp
                    # Aseguramos que el ID sea string para que coincida con la tabla de puntos
                    df_raw['id_estacion'] = df_raw['id_estacion'].astype(str)
                    break
            except Exception as e:
                continue

    df_res = pd.DataFrame()
    if not df_raw.empty:
        alt_calc = altitud_ref if altitud_ref else df_puntos['alt_est'].mean()
        df_res = hydrogeo_utils.ejecutar_pronostico_prophet(df_raw, meses_futuros, alt_calc, ki_final, ruido, kg=kg_factor, kc=kc_ponderado)

    st.markdown(f"### 📍 {nombre_zona}")
    
    # ==============================================================================
    # SECCIÓN: INDICADORES PRINCIPALES (PANEL SUPERIOR CON 10 COLUMNAS)
    # ==============================================================================
    if not df_res.empty:
        df_hist = df_res[df_res['tipo'] == 'Histórico']
        
        if not df_hist.empty:
            # 1. CÁLCULO DE ÁREA (ROBUSTO Y COMPATIBLE CON BUFFER)
            area_km2 = 0
            try:
                # --- CORRECCIÓN CRÍTICA: Normalizar GeoSeries ---
                # Si gdf_zona viene de un Buffer, es una GeoSeries sin columnas.
                # La convertimos a GeoDataFrame para que el resto del código no falle.
                if isinstance(gdf_zona, gpd.GeoSeries):
                    gdf_zona = gpd.GeoDataFrame(geometry=gdf_zona)

                # Opción A: Si la columna ya viene calculada desde la BD (y no se perdió en el buffer)
                if 'area_km2' in gdf_zona.columns:
                     val = gdf_zona['area_km2'].iloc[0]
                     if val > 0: area_km2 = val

                # Opción B: Calcular desde la geometría (Si A falló, no existe, o es un Buffer)
                if area_km2 == 0:
                    gdf_calc = gdf_zona.copy()
                    
                    # Si no tiene CRS definido, adivinamos por los valores
                    if gdf_calc.crs is None:
                        minx = gdf_calc.total_bounds[0]
                        if -180 <= minx <= 180:
                            gdf_calc.set_crs("EPSG:4326", inplace=True)
                        else:
                            gdf_calc.set_crs("EPSG:3116", inplace=True)
                    
                    # Proyectamos a Metros para medir
                    gdf_metros = gdf_calc.to_crs("EPSG:3116")
                    area_km2 = gdf_metros.area.iloc[0] / 1e6

            except Exception as e:
                # Fallback seguro
                if hasattr(gdf_zona, 'columns') and 'Shape_Area' in gdf_zona.columns:
                     try: area_km2 = gdf_zona['Shape_Area'].iloc[0] / 1e6
                     except: area_km2 = 1.0
                else:
                     area_km2 = 1.0 
            
            # Validación final anti-cero
            if area_km2 <= 0: area_km2 = 1.0

 
            # 2. MEDIAS ANUALES (Balance Hídrico)
            p_med = df_hist['p_final'].mean() * 12
            etr_med = df_hist['etr_mm'].mean() * 12
            rec_med = df_hist['recarga_mm'].mean() * 12
            inf_med = df_hist['infiltracion_mm'].mean() * 12
            esc_med = df_hist['escorrentia_mm'].mean() * 12  # Superficial + Base

            # 3. CAUDALES (Modelo Aditivo)
            # Caudal Base (Aporte Acuífero)
            q_base_m3s = (rec_med * area_km2 * 1000) / 31536000
            
            # Caudal Medio Total
            q_medio_m3s = (esc_med * area_km2 * 1000) / 31536000
            
            # 4. ESTADÍSTICAS EXTREMAS (Q Min 50a y Q Eco)
            q_min_50a = 0
            q_eco = 0
            
            if analysis: 
                try:
                    # Serie temporal
                    serie_temporal = df_hist.set_index('fecha')['p_final']
                    
                    # Coeficiente directo
                    esc_directa = esc_med - rec_med
                    c_directo = esc_directa / p_med if p_med > 0 else 0.3
                    
                    stats_panel = analysis.calculate_hydrological_statistics(
                        serie_temporal, 
                        runoff_coeff=c_directo, 
                        area_km2=area_km2,
                        q_base_m3s=q_base_m3s
                    )
                    q_min_50a = stats_panel.get("Q_Min_50a", 0)
                    q_eco = stats_panel.get("Q_Ecologico_Q95", 0)
                except Exception as e:
                    pass

            # 5. VISUALIZACIÓN (10 COLUMNAS)
            st.markdown("##### 💧 Balance Hídrico y Oferta")
            cols = st.columns(10)
            
            cols[0].metric("📏 Área", f"{area_km2:,.1f} km²")
            cols[1].metric("🌧️ Lluvia", f"{p_med:,.0f} mm")
            cols[2].metric("☀️ ETR", f"{etr_med:,.0f} mm")
            cols[3].metric("🌱 Infiltración", f"{inf_med:,.0f} mm")
            cols[4].metric("💧 Recarga", f"{rec_med:,.0f} mm")
            cols[5].metric("🌊 Escorrentía", f"{esc_med:,.0f} mm")
            
            cols[6].metric("⚖️ Q. Medio", f"{q_medio_m3s:.2f} m³/s")
            cols[7].metric("📉 Q. Min 50a", f"{q_min_50a:.3f} m³/s", delta_color="inverse", help="Caudal mínimo en sequía de 50 años (Log-Normal + Base)")
            cols[8].metric("🐟 Q. Ecológico", f"{q_eco:.3f} m³/s", help="Caudal Q95 (Sostenibilidad)")
            
            n_estaciones = len(df_puntos) if 'df_puntos' in locals() else 0
            cols[9].metric("📡 Estaciones", f"{n_estaciones}")

            st.divider()

            with st.expander("📘 Guía Técnica, Metodología y Fuentes de Información", expanded=False):
                tab_guia1, tab_guia2, tab_guia3 = st.tabs(["📚 Conceptos & Ecuaciones", "🛠️ Metodología", "Fuentes de Datos"])
                
                with tab_guia1:
                    st.markdown(r"""
                    ### 💧 Balance Hídrico Simplificado
                    El modelo se basa en la ecuación fundamental de conservación de masa:
                    
                    $$ P = ETR + E_s + R + \Delta S $$
                    
                    Donde:
                    * $P$: Precipitación (Lluvia).
                    * $ETR$: Evapotranspiración Real (Agua que vuelve a la atmósfera).
                    * $E_s$: Escorrentía Superficial (Agua que corre por ríos/quebradas).
                    * $R$: Recarga (Agua que entra al acuífero).
                    
                    ### 🧠 Factores Clave
                    * **Infiltración ($I$):** Es el agua que logra atravesar la superficie del suelo. Depende de la **Cobertura Vegetal** (Bosques infiltran más que Cemento) y la **Textura del Suelo** (Arenas infiltran más que Arcillas).
                    * **Recarga Real ($R$):** Es la fracción de la infiltración que efectivamente llega al almacenamiento subterráneo profundo, condicionada por la **Geología** (Permeabilidad de la roca).
                    """)
                    
                with tab_guia2:
                    st.markdown("""
                    ### ⚙️ Motor de Cálculo
                    1.  **Climatología:** Se utiliza el método de **Turc Modificado** para estimar la ETR mensual, ajustada por un coeficiente de cultivo ($K_c$) dependiente de la cobertura vegetal satelital.
                    2.  **Proyección:** Se implementa el algoritmo **Facebook Prophet** (Regresión Aditiva Generalizada) para proyectar tendencias climáticas y detectar estacionalidad en la lluvia.
                    3.  **Espacialización:** Los mapas de isoyetas y recarga se generan mediante interpolación lineal o IDW (Inverse Distance Weighting) sobre la red de estaciones activas.
                    
                    ### 🚦 Interpretación del Mapa de Potencial
                    * 🟢 **Muy Alto / Alto:** Zonas estratégicas de recarga. Acuíferos productivos o zonas de alta permeabilidad.
                    * 🟡 **Medio:** Zonas de transición.
                    * 🔴 **Bajo / Muy Bajo:** Zonas impermeables, rocas cristalinas o áreas con baja capacidad de almacenamiento.
                    """)
                    
                with tab_guia3:
                    st.info("Este sistema integra información de múltiples entidades oficiales y académicas.")
                    
                    col_f1, col_f2 = st.columns(2)
                    
                    with col_f1:
                        st.markdown("**🗺️ Información Cartográfica**")
                        st.caption("""
                        * **Potencial Hidrogeológico:** Teresita Betancur V. (Universidad de Antioquia).
                        * **Coberturas de la Tierra:** Corine Land Cover (2020).
                        * **Suelos y Litología:** Secretaría de Agricultura, Gobernación de Antioquia.
                        * **Bocatomas:** Secretaría de Agricultura, Gobernación de Antioquia.
                        """)
                        
                    with col_f2:
                        st.markdown("**🌧️ Red de Monitoreo Hidroclimático**")
                        st.caption("""
                        * **IDEAM:** Instituto de Hidrología, Meteorología y Estudios Ambientales.
                        * **EPM:** Empresas Públicas de Medellín.
                        * **Piragua:** Corantioquia.
                        * **CuencaVerde:** Fondo de Agua.
                        * **Google Earth Engine:** Datos satelitales complementarios (CHIRPS/GOES).
                        """)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Serie Completa", "🗺️ Mapa Contexto", "🌈 Mapa Recarga", "📥 Descargas"])

    # --- TAB 1: GRÁFICO COMPLETO Y TABLA ---
    # --- TAB 1: ANÁLISIS COMPLETO (AGREGADO POR CUENCA) ---
    with tab1:
        if not df_res.empty:
            # --- CORRECCIÓN: AGRUPAR POR FECHA (1 Fila = 1 Mes) ---
            # Colapsamos las 37 estaciones en un solo valor promedio para la zona
            df_avg = df_res.groupby(['fecha', 'tipo'])[[
                'p_final', 'etr_mm', 'infiltracion_mm', 'recarga_mm', 
                'escorrentia_mm', 'yhat_upper', 'yhat_lower'
            ]].mean().reset_index().sort_values('fecha')

            # --- SECCIÓN A: GRÁFICA DE BALANCE (USANDO PROMEDIOS) ---
            df_hist = df_avg[df_avg['tipo'] == 'Histórico']
            df_fut = df_avg[df_avg['tipo'] == 'Proyección']
            
            fig = go.Figure()
            
            # Trazos Históricos (Ahora son el promedio de la zona)
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['p_final'], name='Lluvia Hist. (Media)', line=dict(color='#95a5a6', width=1)))
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['etr_mm'], name='ETR Hist. (Media)', line=dict(color='#e67e22', width=1.5)))
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['escorrentia_mm'], name='Escorrentía Hist.', line=dict(color='#27ae60', width=1.5)))
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['recarga_mm'], name='Recarga Hist.', line=dict(color='#2980b9', width=2), fill='tozeroy'))
            
            # Trazos Proyección
            fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['p_final'], name='Lluvia Proy.', line=dict(color='#95a5a6', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['recarga_mm'], name='Recarga Proy.', line=dict(color='#00d2d3', width=2, dash='dot')))
            
            # Incertidumbre (Solo si existe en la proyección)
            if 'yhat_upper' in df_fut.columns and not df_fut['yhat_upper'].isna().all():
                 fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_upper'], showlegend=False, line=dict(width=0)))
                 fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_lower'], name='Incertidumbre', fill='tonexty', line=dict(width=0), fillcolor='rgba(0,210,211,0.1)'))
            
            fig.update_layout(height=450, hovermode="x unified", title=f"Balance Hídrico Agregado: {nombre_zona}", template="plotly_white")
            
            config_plotly = {
                'toImageButtonOptions': {'format': 'png', 'filename': f'Balance_{nombre_zona}', 'height': 600, 'width': 1200, 'scale': 2},
                'displayModeBar': True
            }
            st.plotly_chart(fig, use_container_width=True, config=config_plotly)

            # --- SECCIÓN B: TABLA DE SERIE TEMPORAL (AGREGADA) ---
            # Esta tabla ahora coincidirá con la gráfica: 1 fila por mes
            with st.expander("📅 Ver Tabla de Datos Mensuales (Promedio de la Zona)", expanded=True):
                df_tabla = df_avg.copy() # Usamos df_avg, no df_res
                
                meses_es = {1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun", 7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"}
                df_tabla['Mes Año'] = df_tabla['fecha'].dt.month.map(meses_es) + " " + df_tabla['fecha'].dt.year.astype(str)
                
                cols_tabla = ['Mes Año', 'p_final', 'etr_mm', 'infiltracion_mm', 'recarga_mm', 'escorrentia_mm', 'tipo']
                cols_existentes = [c for c in cols_tabla if c in df_tabla.columns]
                df_tabla = df_tabla[cols_existentes]
                
                mapa_nombres = {
                    'p_final': 'Lluvia', 'etr_mm': 'ETR', 'infiltracion_mm': 'Infiltración', 
                    'recarga_mm': 'Recarga', 'escorrentia_mm': 'Escorrentía', 'tipo': 'Tipo'
                }
                df_tabla = df_tabla.rename(columns=mapa_nombres)
                
                # Configuración Barras
                cols_num = ['Lluvia', 'ETR', 'Infiltración', 'Recarga', 'Escorrentía']
                cols_num_validas = [c for c in cols_num if c in df_tabla.columns]
                max_val = df_tabla[cols_num_validas].max().max() if cols_num_validas else 100

                cfg_barras = {col: st.column_config.ProgressColumn(f"{col} (mm)", format="%.0f", min_value=0, max_value=max_val) for col in cols_num_validas}
                cfg_barras["Mes Año"] = st.column_config.TextColumn("Fecha", width="medium")

                st.dataframe(df_tabla, column_config=cfg_barras, hide_index=True, use_container_width=True, height=300)

            # --- SECCIÓN C: DESGLOSE ESPACIAL (ESTACIONES) ---
            st.divider()
            st.subheader("📍 Estaciones Utilizadas (Resumen Histórico)")
            
            if 'df_puntos' in locals() and not df_puntos.empty:
                # 1. DETECCIÓN INTELIGENTE DE COLUMNAS DE IDENTIFICACIÓN
                # Lista de posibles nombres para el ID de la estación
                posibles_ids = ['codigo', 'CODIGO', 'id_estacion', 'id', 'station_id', 'cod']
                
                # A) Buscar ID en datos de Lluvia (df_raw)
                col_id_raw = next((c for c in posibles_ids if c in df_raw.columns), None)
                
                # B) Buscar ID en datos de Estaciones (df_puntos)
                col_id_puntos = next((c for c in posibles_ids if c in df_puntos.columns), None)
                
                # C) Caso Especial: Si df_puntos no tiene 'codigo', intentamos usar 'nombre' si contiene [12345]
                if not col_id_puntos and 'nombre' in df_puntos.columns:
                     # Creamos una columna temporal 'codigo_extract' extrayendo números entre corchetes
                     df_puntos['codigo_extract'] = df_puntos['nombre'].astype(str).str.extract(r'\[(\d+)\]')
                     col_id_puntos = 'codigo_extract'

                # 2. PROCESAMIENTO
                if col_id_raw:
                    # Agrupar datos de lluvia por el ID encontrado
                    df_promedios_est = df_raw.groupby(col_id_raw)['valor'].mean().reset_index().rename(columns={'valor': 'Lluvia Media'})
                    
                    # Intentar unir si tenemos ID en ambos lados
                    if col_id_puntos:
                        # Asegurar tipos de datos iguales (string vs string)
                        df_puntos[col_id_puntos] = df_puntos[col_id_puntos].astype(str)
                        df_promedios_est[col_id_raw] = df_promedios_est[col_id_raw].astype(str)

                        df_est_detalle = pd.merge(
                            df_puntos, 
                            df_promedios_est, 
                            left_on=col_id_puntos, 
                            right_on=col_id_raw, 
                            how='left'
                        )
                    else:
                        # Si no hay ID en puntos, no podemos cruzar, usamos lo que haya
                        df_est_detalle = df_puntos.copy()
                        df_est_detalle['Lluvia Media'] = 0 # Valor default
                    
                    # Manejo de nombres de columnas (Normalización)
                    # Buscamos las columnas que el usuario quiere ver: Nombre y Municipio
                    col_nombre = next((c for c in ['nombre', 'nom_est', 'estacion'] if c in df_est_detalle.columns), 'Nombre')
                    col_mun = next((c for c in ['municipio', 'mun', 'ciudad'] if c in df_est_detalle.columns), 'Municipio')
                    col_alt = next((c for c in ['altitud', 'alt_est', 'elevacion'] if c in df_est_detalle.columns), 'Altitud')
                    
                    # Cálculos Hidrológicos (Estimación simple para la tabla)
                    altitud_safe = pd.to_numeric(df_est_detalle.get(col_alt, 1000), errors='coerce').fillna(1000)
                    lluvia_safe = df_est_detalle.get('Lluvia Media', 0).fillna(0)
                    
                    temp_est = np.maximum(5, 30 - (0.0065 * altitud_safe))
                    it_est = 300 + 25*temp_est + 0.05*(temp_est**3)
                    denom_est = np.sqrt(0.9 + (lluvia_safe / (np.maximum(it_est, 0.1)/12))**2)
                    
                    etr_est = np.where(denom_est > 0, lluvia_safe / denom_est, 0)
                    etr_real = np.minimum(etr_est * kc_ponderado, lluvia_safe)
                    recarga_est = (lluvia_safe - etr_real).clip(lower=0) * ki_final * kg_factor

                    # Construir DataFrame Final
                    df_show = pd.DataFrame({
                        'Estación': df_est_detalle[col_nombre],
                        'Municipio': df_est_detalle[col_mun],
                        'Altitud': altitud_safe,
                        'Lluvia (mm/mes)': lluvia_safe,
                        'Recarga (mm/mes)': recarga_est
                    })
                    
                    # Configuración Visual
                    cfg_est = {
                        "Estación": st.column_config.TextColumn("Estación", width="large"),
                        "Municipio": st.column_config.TextColumn("Municipio", width="medium"),
                        "Altitud": st.column_config.NumberColumn(format="%.0f m"),
                        "Lluvia (mm/mes)": st.column_config.ProgressColumn(format="%.0f", max_value=max(100, df_show['Lluvia (mm/mes)'].max())),
                        "Recarga (mm/mes)": st.column_config.ProgressColumn(format="%.0f", max_value=max(100, df_show['Lluvia (mm/mes)'].max()))
                    }
                    
                    st.dataframe(df_show.sort_values('Municipio'), column_config=cfg_est, hide_index=True, use_container_width=True)
                
                else:
                    st.warning(f"⚠️ No se encontró columna de ID en los datos de lluvia. Columnas disponibles: {list(df_raw.columns)}")
                    st.dataframe(df_puntos, hide_index=True, use_container_width=True)
            else:
                st.info("No hay información de estaciones para mostrar.")


    # --- TAB 2: CONTEXTO (TOOLTIPS RICOS) ---
    with tab2:
        if st.button("🔄 Recargar Mapa Contexto"): st.rerun()
        pad = 0.05
        bounds = [df_puntos.longitud.min()-pad, df_puntos.latitud.min()-pad, df_puntos.longitud.max()+pad, df_puntos.latitud.max()+pad]
        layers = hydrogeo_utils.cargar_capas_gis_optimizadas(engine, bounds)
        
        m = folium.Map(location=[df_puntos.latitud.mean(), df_puntos.longitud.mean()], zoom_start=11, tiles="CartoDB positron")
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        st.markdown("<style>.leaflet-tooltip {white-space: normal !important; max-width: 300px !important; font-size:11px;}</style>", unsafe_allow_html=True)

        def tooltip_ok(gdf, dic):
            cols = [c.lower().strip() for c in gdf.columns]
            f, a = [], []
            # Lógica permisiva: busca coincidencias parciales
            for k, v in dic.items():
                match = next((c for c in cols if k.lower() in c), None)
                if match:
                    f.append(match)
                    a.append(v)
            return folium.GeoJsonTooltip(fields=f, aliases=a, localize=True) if f else None

        # --- CAPA DE COBERTURAS (RASTER + TOOLTIPS) ---
        if os.path.exists(RUTA_RASTER):
            # 1. Capa Visual (Imagen bonita)
            img_cob, bounds_cob = land_cover.obtener_imagen_folium_coberturas(gdf_zona, RUTA_RASTER)
            
            if img_cob is not None:
                folium.raster_layers.ImageOverlay(
                    image=img_cob,
                    bounds=bounds_cob,
                    opacity=0.6,
                    name="Coberturas (Imagen)",
                    zindex=1
                ).add_to(m)

                # 2. Capa Interactiva (Vectores invisibles para Tooltip)
                # Solo la calculamos si la zona no es gigante (para no colgar el navegador)
                if len(gdf_zona) == 1: # Solo si es una cuenca/municipio específico
                    gdf_tooltips = land_cover.obtener_vector_coberturas_ligero(gdf_zona, RUTA_RASTER)
                    
                    if gdf_tooltips is not None:
                        folium.GeoJson(
                            gdf_tooltips,
                            name="Coberturas (Interactivo)",
                            style_function=lambda x: {
                                'color': 'transparent', 
                                'fillColor': 'transparent', 
                                'weight': 0, 
                                'fillOpacity': 0
                            },
                            tooltip=folium.GeoJsonTooltip(
                                fields=['Cobertura'],
                                aliases=['Tipo:'],
                                localize=True,
                                sticky=True
                            )
                        ).add_to(m)
        # -------------------------------------------

        # Diccionarios Expandidos para Tooltips
        if 'suelos' in layers:
            dic_suelos = {'ucs':'UCS:', 'litolo':'Litología:', 'caracter':'Caract:', 'paisaje':'Paisaje:', 'clima':'Clima:', 'component':'Comp:', 'porcent':'%:'}
            folium.GeoJson(layers['suelos'], name="Suelos", style_function=lambda x: {'color':'orange', 'weight':0.5, 'fillOpacity':0.2},
                           tooltip=tooltip_ok(layers['suelos'], dic_suelos)).add_to(m)
        if 'hidro' in layers:
            # --- FUNCIÓN SEMÁFORO (CORREGIDA: potencial_) ---
            def get_color_hidro(feature):
                props = feature.get('properties', {})
                
                # AQUI ESTABA EL DETALLE: Buscamos 'potencial_' (con guion bajo)
                val = props.get('potencial_') or props.get('potencial') or ''
                
                # Normalizar texto (minusculas)
                txt = str(val).lower().strip()
                
                # Escala de Colores (Semáforo Hidrogeológico)
                if 'muy alto' in txt: return '#006400'  # 🟢 Verde Oscuro
                if 'alto' in txt:     return '#32CD32'  # 🟢 Verde Lima
                if 'medio' in txt:    return '#F1C40F'  # 🟡 Amarillo
                if 'muy bajo' in txt: return '#8B0000'  # 🔴 Rojo Oscuro
                if 'bajo' in txt:     return '#E67E22'  # 🟠 Naranja
                
                return '#85C1E9' # Azul claro (si no tiene dato)

            # Diccionario para el tooltip (Ajustado también)
            dic_hidro = {
                'potencial_': 'Potencial:', # <--- Ajustado
                'unidad': 'Unidad:', 
                'sigla': 'Sigla:'
            }
            
            folium.GeoJson(
                layers['hidro'], 
                name="Hidrogeología (Potencial)", 
                style_function=lambda feature: {
                    'fillColor': get_color_hidro(feature),
                    'color': '#2c3e50',      
                    'weight': 0.5,
                    'fillOpacity': 0.6       
                },
                tooltip=tooltip_ok(layers['hidro'], dic_hidro)
            ).add_to(m)


        if 'bocatomas' in layers:
            # Diccionario exacto basado en tus tablas
            dic_boca = {
                'nombre_acu': 'Acueducto:',    # Nombre del Acueducto
                'tipo': 'Tipo:',               # Veredal, Municipal, etc.
                'fuente_aba': 'Fuente Sup:',   # Fuente Abastecedora (Superficial)
                'fuente_sub': 'Fuente Sub:',   # Fuente Subterránea (SI/NO)
                'pozos': 'Pozos:',             # Tiene pozos
                'entidad_ad': 'Entidad:',      # Entidad Administradora
                'suscriptor': 'Suscriptores:', # Número de suscriptores
                'q': 'Caudal (L/s):'           # Caudal (Q)
            }
            
            folium.GeoJson(
                layers['bocatomas'], 
                name="Bocatomas", 
                marker=folium.CircleMarker(radius=5, color='#d63031', fill_color='#ff7675', fill_opacity=0.8),
                tooltip=tooltip_ok(layers['bocatomas'], dic_boca)
            ).add_to(m)

        # --- ESTACIONES (Popups Completos) ---
        # 1. Crear Grupo de Capas para Estaciones
        fg_estaciones = folium.FeatureGroup(name="Estaciones", show=True)

        for _, r in df_mapa_stats.iterrows():
            # Formateador seguro
            def fmt(val, mult=12): 
                if pd.isnull(val): return "<span style='color:red'>N/D</span>"
                return f"{val*mult:,.0f} mm"

            mun = r.get('municipio', 'N/D')
            alt = r.get('alt_est', 0)
            std_val = r.get('std_lluvia', 0)
            
            html = f"""
            <div style='font-family:sans-serif; width:200px; font-size:12px;'>
                <b style="font-size:13px; color:#2c3e50;">{r['nom_est']}</b>
                <hr style='margin:4px 0; border-top: 1px solid #ccc;'>
                📍 <b>Mun:</b> {mun} <br>
                ⛰️ <b>Alt:</b> {alt:,.0f} m <br>
                <hr style='margin:4px 0; border-top: 1px dashed #ccc;'>
                🌧️ <b>Lluvia:</b> {fmt(r.get('p_media'))}<br>
                ☀️ <b>ETR:</b> {fmt(r.get('etr_media'))}<br>
                🌊 <b>Escorrentía:</b> {fmt(r.get('escorrentia_media'))}<br>
                💧 <b>Recarga:</b> <b style='color:#0000AA;'>{fmt(r.get('recarga_calc'))}</b><br>
                <div style="margin-top:4px; font-size:10px; color:#7f8c8d; text-align:right;">
                    (Desv. Std Lluvia: {std_val:.1f})
                </div>
            </div>"""
            
            folium.Marker(
                [r['latitud'], r['longitud']], 
                popup=folium.Popup(html, max_width=220), 
                icon=folium.Icon(color='black', icon='tint'),
                tooltip=r['nom_est']
            ).add_to(fg_estaciones) # <-- Agregamos al GRUPO, no al mapa directo

        # 2. Agregar el grupo completo al mapa
        fg_estaciones.add_to(m)

        # --- CONTROLES FINALES ---
        # 1. CONTROL DE CAPAS (Ahora reconocerá "Estaciones")
        folium.LayerControl(position='topright', collapsed=True).add_to(m)

        # 2. BOTÓN PANTALLA COMPLETA
        plugins.Fullscreen(
            position='topleft', 
            title='Pantalla Completa', 
            title_cancel='Salir', 
            force_separate_button=True
        ).add_to(m)

        # 3. RENDERIZAR MAPA
        st_folium(m, width=1400, height=600, key=f"ctx_{nombre_zona}")
        
        # 4. BOTÓN DESCARGA HTML
        map_html = m.get_root().render()
        st.download_button(
            label="🌍 Descargar Mapa Contexto (HTML)",
            data=map_html,
            file_name=f"Contexto_{nombre_zona}.html",
            mime="text/html",
            help="Descarga este mapa interactivo."
        )


    # --- TAB 3: RECARGA (BOTÓN + RASTER) ---
    with tab3:
        if st.button("🔄 Recargar Mapa Recarga"): st.rerun() # Botón Recuperado
        
        df_valid = df_mapa_stats.dropna(subset=['p_media'])
        if len(df_valid) < 4:
            st.warning("⚠️ Se requieren al menos 4 estaciones con datos válidos para interpolar.")

            st.session_state.raster_data = None # Limpiar si falla
        else:
            # Interpolación
            x, y, z = df_valid.longitud.values, df_valid.latitud.values, df_valid.p_media.values
            xi = np.linspace(bounds[0], bounds[2], 100)
            yi = np.linspace(bounds[1], bounds[3], 100)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((x, y), z, (Xi, Yi), method='linear')
            
            z_r = df_valid.recarga_calc.values * 12
            Zi_r = griddata((x, y), z_r, (Xi, Yi), method='linear')
            
            # Guardar en sesión
            st.session_state.raster_data = (Zi_r, xi, yi)
            
            # Mapa Base
            m_iso = folium.Map(location=[df_puntos.latitud.mean(), df_puntos.longitud.mean()], zoom_start=11, tiles="CartoDB positron")
            m_iso.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            
            # Capa Raster (Colores)
            vmin, vmax = np.nanmin(Zi_r), np.nanmax(Zi_r)
            try: cmap = plt.colormaps['viridis']
            except: cmap = cm.get_cmap('viridis')
            rgba = cmap((Zi_r - vmin)/(vmax - vmin)); rgba[np.isnan(Zi_r), 3] = 0
            
            folium.raster_layers.ImageOverlay(
                image=rgba, 
                bounds=[[yi.min(), xi.min()], [yi.max(), xi.max()]], 
                opacity=0.7, 
                origin='lower',
                name="Recarga (Raster)"
            ).add_to(m_iso)

            # 3. ISOLÍNEAS CON ETIQUETAS FIJAS (Números visibles)
            try:
                fig_c, ax_c = plt.subplots()
                # Generamos las curvas (menos niveles para no saturar el mapa con texto)
                cs = ax_c.contour(Xi, Yi, Zi_r, levels=10, colors='white', linewidths=0.8)
                plt.close(fig_c)
                
                for i, collection in enumerate(cs.allsegs):
                    level_val = cs.levels[i]
                    for segment in collection:
                        # Solo dibujamos si el segmento es relevante (> 5 puntos) para evitar ruido
                        if len(segment) > 5:
                            locs = [[pt[1], pt[0]] for pt in segment]
                            
                            # 1. Dibujar la Línea
                            folium.PolyLine(
                                locs, 
                                color='white', 
                                weight=1.0, 
                                opacity=0.8,
                                name="Isolíneas"
                            ).add_to(m_iso)
                            
                            # 2. Calcular punto medio para poner el texto
                            mid_idx = len(locs) // 2
                            lat_lbl, lon_lbl = locs[mid_idx]
                            
                            # 3. Crear Etiqueta de Texto Fija (DivIcon)
                            # Usamos text-shadow para que el número blanco se lea sobre fondo claro u oscuro
                            html_text = f"""
                                <div style="
                                    font-size: 9pt; 
                                    font-weight: bold; 
                                    color: white; 
                                    text-shadow: 1px 1px 2px black, -1px -1px 2px black;
                                    white-space: nowrap;
                                ">{int(level_val)}</div>
                            """
                            
                            folium.Marker(
                                location=[lat_lbl, lon_lbl],
                                icon=DivIcon(
                                    icon_size=(30, 10),
                                    icon_anchor=(15, 5), # Centrar el texto
                                    html=html_text
                                )
                            ).add_to(m_iso)
            except Exception as e:
                print(f"Error isolíneas: {e}")

            # 4. CONTORNO DE LA ZONA SELECCIONADA (Cuenca/Municipio)
            # Esto dibuja el límite exacto de lo que estás analizando
            if gdf_zona is not None:
                # Asegurar proyección correcta
                if gdf_zona.crs and gdf_zona.crs.to_string() != "EPSG:4326":
                    gdf_boundary = gdf_zona.to_crs("EPSG:4326")
                else:
                    gdf_boundary = gdf_zona

                folium.GeoJson(
                    gdf_boundary,
                    name=f"Límite: {nombre_zona}",
                    style_function=lambda x: {
                        'color': '#2c3e50',       # Color borde (Gris oscuro elegante)
                        'weight': 2.5,            # Grosor
                        'fillOpacity': 0.0,       # Relleno transparente (para ver el mapa debajo)
                        'dashArray': '5, 5',      # Línea punteada para diferenciar de isolíneas
                        'opacity': 1.0
                    },
                    tooltip=f"Zona: {nombre_zona}"
                ).add_to(m_iso)

            # 5. ESTACIONES (Puntos negros simples)
            for _, r in df_valid.iterrows(): # Usamos df_valid para mostrar solo las que tienen datos
                folium.CircleMarker(
                    location=[r['latitud'], r['longitud']],
                    radius=3,
                    color='black',
                    fill=True,
                    fill_color='white',
                    fill_opacity=1.0,
                    weight=1,
                    tooltip=f"{r['nom_est']}: {r['recarga_calc']*12:,.0f} mm",
                    name="Estaciones"
                ).add_to(m_iso)

            # 6. CONTROL DE CAPAS
            folium.LayerControl(position='topright', collapsed=True).add_to(m_iso)
            
            # 7. BOTÓN PANTALLA COMPLETA
            plugins.Fullscreen(
                position='topleft', 
                title='Pantalla Completa', 
                title_cancel='Salir', 
                force_separate_button=True
            ).add_to(m_iso)
            
            # 8. RENDERIZAR MAPA
            st_folium(m_iso, width=1400, height=600, key=f"iso_{nombre_zona}")

            # 9. BOTÓN DE DESCARGA HTML
            map_html_iso = m_iso.get_root().render()
            st.download_button(
                label="🌈 Descargar Mapa Recarga (HTML)",
                data=map_html_iso,
                file_name=f"Recarga_{nombre_zona}.html",
                mime="text/html",
                help="Descarga este mapa interactivo con isolíneas para compartir."
            )

# ... (Código anterior de las pestañas) ...

with tab4:
    st.header("📥 Centro de Descargas y Documentación")
    
    # 1. FICHA TÉCNICA ENRIQUECIDA
    with st.expander("📘 Ficha Técnica: Modelo Hidrológico Estocástico y de Balance (Leer antes de usar)", expanded=True):
        st.markdown("""
        ### 1. Marco Conceptual
        Este reporte implementa un **Modelo Hidrológico Híbrido** que integra el Balance Hídrico de largo plazo con un análisis estocástico de extremos. A diferencia de modelos simples lluvia-escorrentía, este sistema reconoce la **dualidad del flujo**:
        * **Componente Superficial:** Respuesta rápida a la precipitación (Escorrentía Directa).
        * **Componente Subterráneo (Flujo Base):** Aporte lento y sostenido del acuífero, calculado a partir de la Recarga Potencial.
        
        ### 2. Metodología de Cálculo
        * **Balance Hídrico:** Se utiliza el método de **Turc (1954)** modificado para condiciones tropicales, calculando la Evapotranspiración Real (ETR) y el Superávit Hídrico.
        * **Modelo Aditivo ($Q_{total}$):** El caudal medio no depende solo de la lluvia del mes. Se define como:
            $$Q_{total} = Q_{Directo}(P) + Q_{Base}(R)$$
            Donde $Q_{Base}$ actúa como un "suelo hidráulico" que impide que los ríos perennes aparezcan secos en el modelo, incluso en ausencia de lluvias.
        * **Análisis Estocástico de Extremos:**
            * **Máximos (Crecientes):** Se ajustan mediante la distribución de **Gumbel**, ideal para valores extremos superiores.
            * **Mínimos (Sequías):** Se utiliza la distribución **Log-Normal** de 2 parámetros. Esta elección matemática respeta la asintoticidad de las curvas de recesión de acuíferos (el caudal tiende a cero pero no toca el cero ni se vuelve negativo), garantizando proyecciones realistas para $T_r > 50$ años.
        * **Regionalización:** Ante la falta de estaciones in-situ, el sistema genera una "Estación Virtual" agregando datos de todas las estaciones en un **Buffer de 20 km** alrededor de la cuenca (Técnica de Vecino Próximo Ponderado).

        ### 3. Alcance y Utilidad
        * **Planificación del Recurso Hídrico:** Estimación de oferta hídrica neta para concesiones.
        * **Gestión del Riesgo:** Los valores $Q_{Max}^{100a}$ permiten dimensionar obras hidráulicas (puentes, box-culverts).
        * **Seguridad Hídrica:** Los valores $Q_{Min}^{50a}$ y $Q_{95}$ (Caudal Ecológico) establecen los límites críticos para el abastecimiento en escenarios de Cambio Climático.

        ### 4. Limitaciones e Interpretación
        * **Escala Temporal:** El modelo opera a paso mensual. Los picos de crecientes instantáneas (horas) podrían ser superiores a los $Q_{Max}$ mensuales reportados.
        * **Incertidumbre:** En cuencas sin estaciones dentro del radio de 20km, los datos son interpolaciones regionales que deben validarse en campo.
        * **Caudal Base:** Se asume un factor de recarga regional del 15% (30% infiltración $\times$ 50% percolación). Cuencas con geología kárstica o muy fracturada podrían tener caudales base superiores.

        ### 5. Fuentes de Información y Referencias
        * **Climatología:** IDEAM (Precipitación Histórica Mensual).
        * **Topografía:** ALOS PALSAR / SRTM (30m) para delimitación y morfometría.
        * **Referentes Académicos:** * *Chow, V. T. (1988). Applied Hydrology.* (Estadística Gumbel/Log-Normal).
            * *Turc, L. (1954). Le bilan d'eau des sols: relations entre les précipitations, l'évaporation et l'écoulement.*
        """)

    st.markdown("---")
    
    # 2. BOTONES DE DESCARGA
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.info("📊 **Serie Temporal Completa**\n\nDescarga los datos mensuales (Lluvia, Q) usados para los cálculos.")
        # (Aquí iría lógica para descargar serie si la tienes en memoria, o el botón que ya tenías)
        if 'df_rain_mensual' in locals(): # Ejemplo si estuviera disponible
             pass 
        else:
            st.write("*(Selecciona una cuenca en el mapa para habilitar esta descarga)*")

    with col_d2:
        st.success("📑 **Reporte Maestro Global (CSV)**\n\nTabla con todas las 51 cuencas, estadísticas y ecuaciones.")
        try:
            df_rep = pd.read_sql("SELECT * FROM reporte_cuencas", engine)
            csv_rep = df_rep.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar Reporte Global (.csv)",
                data=csv_rep,
                file_name="Reporte_Hidrologico_Completo_SIHCLI.csv",
                mime="text/csv"
            )
        except:
            st.warning("Primero debes generar el reporte en la sección inferior.")

# ==============================================================================
# SECCIÓN: REPORTE GLOBAL HIDROLÓGICO (VERSIÓN FINAL CORREGIDA)
# ==============================================================================
st.markdown("---")
with st.expander("📑 Reporte Maestro de Cuencas (Tabla Global)", expanded=False):
    
    st.info("Genera tabla maestra con Modelo Aditivo (Escorrentía Directa + Caudal Base) y Estadísticas Extremas (Log-Normal para sequías).")

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

    # 2. CONFIGURACIÓN
    st.write("#### ⚙️ Configuración")
    try:
        cols_bd = pd.read_sql("SELECT column_name FROM information_schema.columns WHERE table_name = 'cuencas' AND column_name != 'geometry'", engine)['column_name'].tolist()
        idx_def = next((i for i, c in enumerate(cols_bd) if c in ['n_nss3', 'subc_lbl']), 0)
        col_nombre_reporte = st.selectbox("🏷️ Columna para Nombres:", cols_bd, index=idx_def, key="sel_col_rep_final")
    except:
        col_nombre_reporte = 'nombre_cuenca'

    # 3. BOTÓN DE CÁLCULO
    if st.button(f"🚀 Generar Reporte Completo"):
        import rasterio
        from rasterio.mask import mask
        
        try:
            # A. CARGAR DATOS
            with st.spinner("Cargando geometrías y normalizando datos..."):
                # Cuencas
                gdf_all = gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry")
                if gdf_all.crs and gdf_all.crs.to_string() != "EPSG:3116":
                    gdf_all = gdf_all.to_crs("EPSG:3116")
                
                # Estaciones
                gdf_est = gpd.read_postgis("SELECT id_estacion, geom FROM estaciones", engine, geom_col="geom")
                if gdf_est.crs and gdf_est.crs.to_string() != "EPSG:3116":
                    gdf_est = gdf_est.to_crs("EPSG:3116")
                gdf_est['id_estacion'] = gdf_est['id_estacion'].astype(str)

                # Lluvias
                df_rain_anual = pd.read_sql("SELECT id_estacion_fk, AVG(precipitation)*12 as ppt_anual FROM precipitacion_mensual GROUP BY id_estacion_fk", engine)
                df_rain_anual['id_estacion_fk'] = df_rain_anual['id_estacion_fk'].astype(str)

                df_rain_mensual = pd.read_sql("SELECT id_estacion_fk, fecha_mes_año, precipitation FROM precipitacion_mensual", engine)
                df_rain_mensual['fecha'] = pd.to_datetime(df_rain_mensual['fecha_mes_año'])
                df_rain_mensual['id_estacion_fk'] = df_rain_mensual['id_estacion_fk'].astype(str)

            # B. PREPARAR DEM
            path_dem = "data/DemAntioquia_EPSG3116.tif"
            src_dem = None
            crs_dem_objetivo = None
            
            if os.path.exists(path_dem):
                src_dem = rasterio.open(path_dem)
                crs_dem_objetivo = src_dem.crs
                # Fix para Origen Nacional si no tiene CRS
                if not crs_dem_objetivo and src_dem.transform[2] > 4000000:
                    crs_dem_objetivo = "EPSG:9377"

            # C. BUCLE
            progreso = st.progress(0)
            status = st.empty()
            lista_resultados = []
            total = len(gdf_all)
            
            for i, row in gdf_all.iterrows():
                nom = str(row.get(col_nombre_reporte, f"Cuenca {i}"))
                status.text(f"Procesando {i+1}/{total}: {nom}...")
                
                # Geometría Base (Bogotá - 3116)
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
                # A. Parámetros Climáticos
                if alt_med == 0: alt_med = 1500
                temp = max(0, 28 - 0.006 * alt_med)
                L = 300 + 25*temp + 0.05*(temp**3)

                # B. Buffer y Lluvias
                buffer_geom = geom_base.buffer(20000) 
                est_in = gdf_est[gdf_est.geometry.within(buffer_geom)]
                n_est = len(est_in)
                
                ppt_cuenca = 0
                
                if n_est > 0:
                    ids = est_in['id_estacion'].astype(str).unique().tolist()
                    ppt_vals = df_rain_anual[df_rain_anual['id_estacion_fk'].isin(ids)]['ppt_anual']
                    if not ppt_vals.empty: ppt_cuenca = ppt_vals.mean()
                else:
                    ppt_cuenca = 2000 # Fallback

                # C. Balance Turc Anual
                etr = ppt_cuenca / np.sqrt(0.9 + (ppt_cuenca/L)**2) if (L>0 and ppt_cuenca>0) else 0
                etr = min(etr, ppt_cuenca)
                esc_total_anual = ppt_cuenca - etr 
                
                # Desglose Hidrogeológico
                inf = esc_total_anual * 0.30 
                recarga_mm = inf * 0.50 
                esc_directa_mm = esc_total_anual - inf 
                
                # Caudales Base (Modelo Aditivo)
                q_base_m3s = (recarga_mm * area_km2 * 1000) / 31536000
                q_medio_total = ((esc_directa_mm * area_km2 * 1000)/31536000) + q_base_m3s
                
                # Coeficiente para lo rápido
                c_directo = esc_directa_mm / ppt_cuenca if ppt_cuenca > 0 else 0.3

                # --- 3. ESTADÍSTICAS AVANZADAS ---
                ec_fdc = "N/A"
                stats_ext = {}
                
                if n_est > 0 and ppt_cuenca > 0 and analysis:
                    try:
                        ids = est_in['id_estacion'].astype(str).unique().tolist()
                        s_mensual = df_rain_mensual[df_rain_mensual['id_estacion_fk'].isin(ids)]
                        
                        if not s_mensual.empty:
                            s_sintetica = s_mensual.groupby('fecha')['precipitation'].mean()
                            
                            # Estadísticas con Suelo Hidrológico
                            stats_ext = analysis.calculate_hydrological_statistics(
                                s_sintetica, 
                                runoff_coeff=c_directo, 
                                area_km2=area_km2, 
                                q_base_m3s=q_base_m3s
                            )
                            
                            # FDC
                            fdc = analysis.calculate_duration_curve(s_sintetica, runoff_coeff=c_directo, area_km2=area_km2, q_base_m3s=q_base_m3s)
                            if fdc: ec_fdc = fdc.get("equation", "N/A")
                    except: 
                        pass # <--- AQUÍ ESTABA EL ERROR: Faltaba cerrar el try

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
                    "Q Max 5a": round(stats_ext.get("Q_Max_5a", 0), 3),
                    "Q Max 10a": round(stats_ext.get("Q_Max_10a", 0), 3),
                    "Q Max 25a": round(stats_ext.get("Q_Max_25a", 0), 3),
                    "Q Max 50a": round(stats_ext.get("Q_Max_50a", 0), 3),
                    "Q Max 100a": round(stats_ext.get("Q_Max_100a", 0), 3),
                    
                    # Mínimos
                    "Q Min 2.33a": round(stats_ext.get("Q_Min_2.33a", 0), 3),
                    "Q Min 5a": round(stats_ext.get("Q_Min_5a", 0), 3),
                    "Q Min 10a": round(stats_ext.get("Q_Min_10a", 0), 3),
                    "Q Min 25a": round(stats_ext.get("Q_Min_25a", 0), 3),
                    "Q Min 50a": round(stats_ext.get("Q_Min_50a", 0), 3),
                    "Q Min 100a": round(stats_ext.get("Q_Min_100a", 0), 3),
                }

                lista_resultados.append(fila)
                progreso.progress((i+1)/total)

            # GUARDAR EN BD
            df_final = pd.DataFrame(lista_resultados)
            df_final.to_sql("reporte_cuencas", engine, if_exists='replace', index=False)
            
            progreso.empty()
            status.success(f"✅ ¡Reporte Generado! ({len(df_final)} Cuencas).")
            st.rerun()

        except Exception as e:
            st.error(f"Error crítico generando el reporte: {e}")