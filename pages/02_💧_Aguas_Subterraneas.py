# pages/02_💧_Aguas_Subterraneas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import geopandas as gpd
from scipy.interpolate import griddata
import sys
import os
import folium
from streamlit_folium import st_folium
import json

# --- CONFIGURACIÓN E IMPORTS ---
st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors 
    # Intentar importar motor de base de datos
    try:
        from modules.db_manager import get_engine
    except ImportError:
        def get_engine(): return create_engine(st.secrets["DATABASE_URL"])
except ImportError:
    st.error("Error crítico importando módulos del sistema.")
    st.stop()

# --- PROPHET ---
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

st.title("💧 Aguas Subterráneas: Recarga y Zonificación")

# ==============================================================================
# 1. FUNCIONES DE LÓGICA (TURC, PROPHET, MATH) - RECUPERADAS
# ==============================================================================

def calculate_turc_row(p_anual, altitud, ki):
    """Cálculo puntual del Modelo Turc"""
    temp = 30 - (0.0065 * altitud)
    l_t = 300 + 25*temp + 0.05*(temp**3)
    if l_t == 0: l_t = 0.001
    etr = p_anual / np.sqrt(0.9 + (p_anual / l_t)**2)
    recarga = (p_anual - etr) * ki
    return etr, max(0, recarga)

def calculate_turc_advanced(df, ki):
    """Cálculo masivo (DataFrame) del Modelo Turc"""
    df = df.copy()
    df['alt_est'] = pd.to_numeric(df['alt_est'], errors='coerce').fillna(0)
    df['p_anual'] = pd.to_numeric(df['p_anual'], errors='coerce').fillna(0)
    
    # 1. Temperatura Estimada
    df['temp_est'] = 30 - (0.0065 * df['alt_est'])
    t = df['temp_est']
    
    # 2. Capacidad Evaporativa
    l_t = 300 + 25*t + 0.05*(t**3)
    
    # 3. ETR
    with np.errstate(divide='ignore', invalid='ignore'):
        df['etr_mm'] = df['p_anual'] / np.sqrt(0.9 + (df['p_anual'] / l_t)**2)
    
    # 4. Balance
    df['excedente_mm'] = (df['p_anual'] - df['etr_mm']).clip(lower=0)
    df['recarga_mm'] = df['excedente_mm'] * ki
    return df

def run_prophet_forecast_hybrid(df_hist, months_ahead, altitud_ref, ki, ruido_factor):
    """Proyección climática y cálculo de recarga futura"""
    if not PROPHET_AVAILABLE: return pd.DataFrame()
    
    # Preparar datos para Prophet
    df_prophet = df_hist.rename(columns={'fecha': 'ds', 'p_mensual': 'y'})
    last_date_real = df_prophet['ds'].max()
    
    # Modelo
    m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    m.fit(df_prophet)
    
    # Futuro
    future = m.make_future_dataframe(periods=months_ahead, freq='M')
    forecast = m.predict(future)
    
    # Merge
    df_merged = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], df_prophet[['ds', 'y']], on='ds', how='left')
    
    # Lógica híbrida (Real + Proyectado)
    df_merged['p_final'] = df_merged['y'].combine_first(df_merged['yhat'])
    df_merged['p_lower'] = df_merged['y'].combine_first(df_merged['yhat_lower'] * (1 - 0.1*ruido_factor))
    df_merged['p_upper'] = df_merged['y'].combine_first(df_merged['yhat_upper'] * (1 + 0.1*ruido_factor))
    
    # Anualizar tasas (mm/mes -> mm/año aprox para Turc instantáneo)
    df_merged['p_rate'] = df_merged['p_final'].clip(lower=0) * 12
    df_merged['p_rate_low'] = df_merged['p_lower'].clip(lower=0) * 12
    df_merged['p_rate_high'] = df_merged['p_upper'].clip(lower=0) * 12
    
    # Recálculo Turc Vectorizado para la serie
    def calc_vec(p): return calculate_turc_row(p, altitud_ref, ki)
    
    central = df_merged['p_rate'].apply(calc_vec)
    df_merged['etr_est'] = [x[0] for x in central]
    df_merged['recarga_est'] = [x[1] for x in central]
    
    low = df_merged['p_rate_low'].apply(calc_vec)
    df_merged['recarga_low'] = [x[1] for x in low]
    
    high = df_merged['p_rate_high'].apply(calc_vec)
    df_merged['recarga_high'] = [x[1] for x in high]
    
    df_merged['tipo'] = np.where(df_merged['ds'] <= last_date_real, 'Histórico', 'Proyección')
    return df_merged

def interpolacion_segura_suave(points, values, grid_x, grid_y):
    try:
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
        mask = np.isnan(grid_z)
        if np.any(mask):
            grid_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
            grid_z[mask] = grid_nearest[mask]
        return grid_z
    except: return griddata(points, values, (grid_x, grid_y), method='linear')

# ==============================================================================
# 2. FUNCIONES DE CARGA GIS (BASE DE DATOS)
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner="Cargando capas espaciales...")
def cargar_capas_gis_db():
    """Carga las capas de Suelos, Hidrogeología y Bocatomas desde PostgreSQL"""
    engine = get_engine()
    layers = {"suelos": None, "hidro": None, "bocatomas": None}
    
    if not engine: return layers
    
    try:
        with engine.connect() as conn:
            # A. SUELOS (Usando ST_AsGeoJSON para máxima compatibilidad)
            # Verificamos si la tabla existe primero
            try:
                q_suelos = text("""
                    SELECT id, unidad_suelo, textura, grupo_hidro, ST_AsGeoJSON(geom) as geometry 
                    FROM suelos
                """)
                df_s = pd.read_sql(q_suelos, conn)
                if not df_s.empty:
                    df_s['geometry'] = df_s['geometry'].apply(lambda x: json.loads(x) if x else None)
                    # Crear GeoDataFrame manual
                    from shapely.geometry import shape
                    df_s['geometry'] = df_s['geometry'].apply(lambda x: shape(x) if x else None)
                    layers["suelos"] = gpd.GeoDataFrame(df_s, geometry='geometry', crs="EPSG:4326")
            except Exception as e: pass # Tabla no existe o vacía

            # B. HIDROGEOLOGÍA
            try:
                q_hidro = text("""
                    SELECT id, nombre_zona, potencial, unidad_geo, ST_AsGeoJSON(geom) as geometry 
                    FROM zonas_hidrogeologicas
                """)
                df_h = pd.read_sql(q_hidro, conn)
                if not df_h.empty:
                    df_h['geometry'] = df_h['geometry'].apply(lambda x: json.loads(x) if x else None)
                    from shapely.geometry import shape
                    df_h['geometry'] = df_h['geometry'].apply(lambda x: shape(x) if x else None)
                    layers["hidro"] = gpd.GeoDataFrame(df_h, geometry='geometry', crs="EPSG:4326")
            except Exception: pass

            # C. BOCATOMAS (Si existe)
            try:
                q_boca = text("SELECT *, ST_AsGeoJSON(geom) as geometry FROM bocatomas")
                df_b = pd.read_sql(q_boca, conn)
                if not df_b.empty:
                    df_b['geometry'] = df_b['geometry'].apply(lambda x: json.loads(x) if x else None)
                    from shapely.geometry import shape
                    df_b['geometry'] = df_b['geometry'].apply(lambda x: shape(x) if x else None)
                    layers["bocatomas"] = gpd.GeoDataFrame(df_b, geometry='geometry', crs="EPSG:4326")
            except Exception: pass
            
    except Exception as e:
        # Silencioso para no asustar al usuario, pero imprime en consola
        print(f"Error GIS DB: {e}")
        
    return layers

# ==============================================================================
# 3. INTERFAZ: PANEL LATERAL RESTAURADO
# ==============================================================================

# Selector Espacial (Tu módulo original)
ids_dummy, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

st.sidebar.divider()
st.sidebar.header("🌱 Parámetros del Suelo")
st.sidebar.info("Ajuste los porcentajes de cobertura para calcular la infiltración.")

# --- ¡AQUÍ ESTÁN DE VUELTA TUS SLIDERS! ---
col_s1, col_s2 = st.sidebar.columns(2)
pct_bosque = col_s1.number_input("% Bosque", 0, 100, 60, help="Cobertura vegetal densa")
pct_cultivo = col_s2.number_input("% Agrícola", 0, 100, 30, help="Cultivos o pastos")
pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
st.sidebar.metric("% Urbano / Impermeable", f"{pct_urbano}%")

# Cálculo del Ki Ponderado
# Ki Bosque ~ 0.50, Cultivo ~ 0.30, Urbano ~ 0.10
ki_ponderado = ((pct_bosque * 0.50) + (pct_cultivo * 0.30) + (pct_urbano * 0.10)) / 100.0
st.sidebar.metric("Coef. Infiltración ($K_i$)", f"{ki_ponderado:.2f}")

st.sidebar.divider()
st.sidebar.header("⚙️ Configuración Pronóstico")
horizonte_meses = st.sidebar.slider("Meses Futuros:", 12, 60, 24)
ruido = st.sidebar.slider("Incertidumbre Climática:", 0.0, 2.0, 0.5)


# ==============================================================================
# 4. MOTOR PRINCIPAL
# ==============================================================================

if gdf_zona is not None and not gdf_zona.empty:
    engine = get_engine()
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    
    # 1. Cargar Estaciones (SQL)
    q_est = text("""
        SELECT id_estacion, nom_est, alt_est, ST_Y(geom::geometry) as lat, ST_X(geom::geometry) as lon
        FROM estaciones 
        WHERE ST_X(geom::geometry) BETWEEN :minx AND :maxx 
          AND ST_Y(geom::geometry) BETWEEN :miny AND :maxy
    """)
    
    try:
        # Cargar Datos Climáticos
        df_est = pd.read_sql(q_est, engine, params={"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})
        
        if not df_est.empty:
            sel_local = st.multiselect("📍 Filtrar Estaciones:", df_est['nom_est'].unique(), default=df_est['nom_est'].unique())
            df_est_filtered = df_est[df_est['nom_est'].isin(sel_local)]
            
            if not df_est_filtered.empty:
                ids_v = df_est_filtered['id_estacion'].unique()
                ids_s = ",".join([f"'{str(x)}'" for x in ids_v])
                
                # Cargar Lluvias (Promedios y Series)
                q_avg = text(f"SELECT id_estacion_fk as id_estacion, AVG(precipitation)*12 as p_anual FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY 1")
                df_avg = pd.read_sql(q_avg, engine)
                
                q_serie = text(f"SELECT fecha_mes_año as fecha, AVG(precipitation) as p_mensual FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY 1 ORDER BY 1")
                df_serie = pd.read_sql(q_serie, engine)
                
                # Unificar y Calcular Modelo Turc (CON TUS SLIDERS RESTAURADOS)
                df_est_filtered['id_estacion'] = df_est_filtered['id_estacion'].astype(str)
                df_avg['id_estacion'] = df_avg['id_estacion'].astype(str)
                df_work = pd.merge(df_est_filtered, df_avg, on='id_estacion', how='inner')
                df_work['alt_est'] = df_work['alt_est'].fillna(altitud_ref)
                
                # AQUÍ USAMOS EL KI QUE TÚ DEFINISTE EN EL SIDEBAR
                df_res_avg = calculate_turc_advanced(df_work, ki_ponderado)
                
                # KPIs Rápidos
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Lluvia Media (mm/año)", f"{df_res_avg['p_anual'].mean():.0f}")
                k2.metric("ETR (mm/año)", f"{df_res_avg['etr_mm'].mean():.0f}")
                k3.metric("Recarga Potencial (mm/año)", f"{df_res_avg['recarga_mm'].mean():.0f}", delta="Oferta Hídrica")
                k4.metric("Estaciones Analizadas", len(df_res_avg))
                
                st.divider()
                
                # --- PESTAÑAS (TODAS JUNTAS Y FUNCIONALES) ---
                tab_graf, tab_mapa, tab_data = st.tabs(["📈 Análisis Temporal & Pronóstico", "🗺️ Mapa Integrado (Capas)", "💾 Datos"])
                
                # ------------------------------------------------------------------
                # TAB 1: GRÁFICOS RESTAURADOS
                # ------------------------------------------------------------------
                with tab_graf:
                    if not df_serie.empty and PROPHET_AVAILABLE:
                        with st.spinner("Generando proyección climática..."):
                            df_fc = run_prophet_forecast_hybrid(df_serie, horizonte_meses, altitud_ref, ki_ponderado, ruido)
                            
                            if not df_fc.empty:
                                fig = go.Figure()
                                h = df_fc[df_fc['tipo']=='Histórico']
                                p = df_fc[df_fc['tipo']=='Proyección']
                                
                                # Gráfico Combinado Lluvia + ETR + Recarga
                                fig.add_trace(go.Bar(x=h['ds'], y=h['p_rate'], name='Lluvia', marker_color='rgba(135, 206, 235, 0.5)'))
                                fig.add_trace(go.Scatter(x=df_fc['ds'], y=df_fc['etr_est'], name='ETR (Evaporación)', line=dict(color='orange', width=2, dash='dot')))
                                fig.add_trace(go.Scatter(x=h['ds'], y=h['recarga_est'], name='Recarga Histórica', line=dict(color='blue', width=2), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'))
                                fig.add_trace(go.Scatter(x=p['ds'], y=p['recarga_est'], name='Recarga Proyectada', line=dict(color='dodgerblue', width=2, dash='dash')))
                                fig.add_trace(go.Scatter(x=p['ds'], y=p['recarga_high'], line=dict(width=0), showlegend=False))
                                fig.add_trace(go.Scatter(x=p['ds'], y=p['recarga_low'], line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.1)', name='Incertidumbre'))
                                
                                fig.update_layout(title="Dinámica Hídrica: Lluvia vs ETR vs Recarga", hovermode="x unified", height=500)
                                st.plotly_chart(fig, use_container_width=True)
                            else: st.error("Error calculando proyección.")
                    else: st.warning("Datos insuficientes para series de tiempo.")

                # ------------------------------------------------------------------
                # TAB 2: MAPA UNIFICADO (TURC + GIS DB)
                # ------------------------------------------------------------------
                with tab_mapa:
                    st.markdown("### Visor Integrado de Recursos Hídricos")
                    
                    # 1. Cargar Capas desde DB
                    capas = cargar_capas_gis_db()
                    gdf_suelos = capas["suelos"]
                    gdf_hidro = capas["hidro"]
                    gdf_bocas = capas["bocatomas"]
                    
                    # 2. Configurar Mapa Base
                    c_lat = df_est_filtered['lat'].mean()
                    c_lon = df_est_filtered['lon'].mean()
                    m = folium.Map(location=[c_lat, c_lon], zoom_start=10, tiles="CartoDB positron")
                    
                    # --- CAPA A: SUELOS (DB) ---
                    if gdf_suelos is not None:
                        fg_suelos = folium.FeatureGroup(name="🌱 Suelos (Fondo)")
                        
                        # Buscar columnas disponibles (tolerante a fallos)
                        cols_s = gdf_suelos.columns
                        field_u = next((c for c in ['unidad_suelo', 'Simbolo', 'Unidad'] if c in cols_s), None)
                        field_t = next((c for c in ['textura', 'Textura', 'TEXTURA'] if c in cols_s), None)
                        
                        tips = []
                        if field_u: tips.append(field_u)
                        if field_t: tips.append(field_t)
                        
                        folium.GeoJson(
                            gdf_suelos,
                            style_function=lambda x: {'fillColor': '#e5f5e0', 'color': '#a1d99b', 'weight': 0.5, 'fillOpacity': 0.4},
                            tooltip=folium.GeoJsonTooltip(fields=tips) if tips else None
                        ).add_to(fg_suelos)
                        fg_suelos.add_to(m)

                    # --- CAPA B: HIDROGEOLOGÍA (DB) ---
                    if gdf_hidro is not None:
                        fg_hidro = folium.FeatureGroup(name="💧 Potencial Hidrogeológico")
                        
                        cols_h = gdf_hidro.columns
                        field_p = next((c for c in ['potencial', 'Potencial_', 'POTENCIAL'] if c in cols_h), None)
                        field_ug = next((c for c in ['unidad_geo', 'Unidad_Geo', 'UNIDAD'] if c in cols_h), None)
                        
                        tips_h = []
                        if field_p: tips_h.append(field_p)
                        if field_ug: tips_h.append(field_ug)
                        
                        folium.GeoJson(
                            gdf_hidro,
                            style_function=lambda x: {'fillColor': '#3182bd', 'color': '#08519c', 'weight': 1, 'fillOpacity': 0.4},
                            tooltip=folium.GeoJsonTooltip(fields=tips_h) if tips_h else None
                        ).add_to(fg_hidro)
                        fg_hidro.add_to(m)

                    # --- CAPA C: RECARGA TURC (INTERPOLACIÓN) ---
                    if len(df_res_avg) >= 3:
                        fg_turc = folium.FeatureGroup(name="🌧️ Recarga Turc (Interpolada)", show=True)
                        gx, gy = np.mgrid[minx:maxx:200j, miny:maxy:200j]
                        grid = interpolacion_segura_suave(df_res_avg[['lon','lat']].values, df_res_avg['recarga_mm'].values, gx, gy)
                        
                        # Contornos (Isolíneas) usando GeoJSON Contour no es nativo fácil en Folium puro sin plugins pesados
                        # Usaremos Heatmap o CircleMarkers densos si no hay plugin, pero mejor: ImageOverlay
                        # Para simplicidad y rapidez: Marcadores de Estaciones con Color
                        for _, row in df_res_avg.iterrows():
                            val = row['recarga_mm']
                            color = 'blue' if val > 1000 else ('green' if val > 500 else 'orange')
                            folium.CircleMarker(
                                [row['lat'], row['lon']],
                                radius=6, color='white', weight=1,
                                fill=True, fill_color=color, fill_opacity=0.9,
                                popup=f"<b>{row['nom_est']}</b><br>Recarga: {val:.0f} mm",
                                tooltip=f"Recarga: {val:.0f} mm"
                            ).add_to(fg_turc)
                        fg_turc.add_to(m)
                    
                    # --- CAPA D: BOCATOMAS (DB) ---
                    if gdf_bocas is not None:
                        fg_boca = folium.FeatureGroup(name="🚰 Bocatomas", show=False)
                        cols_b = gdf_bocas.columns
                        field_n = next((c for c in ['Nombre', 'NOMBRE', 'BOCATOMA'] if c in cols_b), cols_b[0])
                        
                        for _, row in gdf_bocas.iterrows():
                            if row.geometry.geom_type == 'Point':
                                folium.CircleMarker(
                                    [row.geometry.y, row.geometry.x],
                                    radius=4, color='red', fill=True, fill_color='darkred',
                                    tooltip=f"{row[field_n]}"
                                ).add_to(fg_boca)
                        fg_boca.add_to(m)

                    folium.LayerControl().add_to(m)
                    st_folium(m, width="100%", height=600)

                # ------------------------------------------------------------------
                # TAB 3: DATOS
                # ------------------------------------------------------------------
                with tab_data:
                    st.dataframe(df_res_avg)
                    csv = df_res_avg.to_csv(index=False).encode('utf-8')
                    st.download_button("Descargar CSV", csv, "recarga_calculada.csv", "text/csv")
            
            else: st.warning("Sin estaciones en el área seleccionada.")
        else: st.warning("No hay datos climáticos en la base de datos para esta zona.")
    
    except Exception as e:
        st.error(f"Error en el motor de cálculo: {e}")

else:
    st.info("👈 Por favor, seleccione una zona en el menú lateral.")