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
from branca.colormap import LinearColormap
from shapely.geometry import shape

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors 
    try:
        from modules.db_manager import get_engine
    except ImportError:
        def get_engine(): return create_engine(st.secrets["DATABASE_URL"])
except ImportError:
    st.error("Error crítico importando módulos.")
    st.stop()

# --- PROPHET ---
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

st.title("💧 Aguas Subterráneas: Recarga y Zonificación")

# ==============================================================================
# 1. LÓGICA MATEMÁTICA (TURC & PROPHET) - RESTAURADA
# ==============================================================================

def calculate_turc_advanced(df, ki):
    df = df.copy()
    df['alt_est'] = pd.to_numeric(df['alt_est'], errors='coerce').fillna(0)
    df['p_anual'] = pd.to_numeric(df['p_anual'], errors='coerce').fillna(0)
    
    df['temp_est'] = 30 - (0.0065 * df['alt_est'])
    t = df['temp_est']
    l_t = 300 + 25*t + 0.05*(t**3)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        df['etr_mm'] = df['p_anual'] / np.sqrt(0.9 + (df['p_anual'] / l_t)**2)
    
    df['excedente_mm'] = (df['p_anual'] - df['etr_mm']).clip(lower=0)
    df['recarga_mm'] = df['excedente_mm'] * ki
    return df

def calculate_turc_row(p_anual, altitud, ki):
    temp = 30 - (0.0065 * altitud)
    l_t = 300 + 25*temp + 0.05*(temp**3)
    if l_t == 0: l_t = 0.001
    etr = p_anual / np.sqrt(0.9 + (p_anual / l_t)**2)
    recarga = (p_anual - etr) * ki
    return etr, max(0, recarga)

def run_prophet_forecast_hybrid(df_hist, months_ahead, altitud_ref, ki, ruido_factor):
    if not PROPHET_AVAILABLE: return pd.DataFrame()
    df_prophet = df_hist.rename(columns={'fecha': 'ds', 'p_mensual': 'y'})
    last_date_real = df_prophet['ds'].max()
    
    m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True).fit(df_prophet)
    future = m.make_future_dataframe(periods=months_ahead, freq='ME') 
    forecast = m.predict(future)
    
    df_merged = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], df_prophet[['ds', 'y']], on='ds', how='left')
    df_merged['p_final'] = df_merged['y'].combine_first(df_merged['yhat'])
    df_merged['p_lower'] = df_merged['y'].combine_first(df_merged['yhat_lower'] * (1 - 0.1*ruido_factor))
    df_merged['p_upper'] = df_merged['y'].combine_first(df_merged['yhat_upper'] * (1 + 0.1*ruido_factor))
    
    df_merged['p_rate'] = df_merged['p_final'].clip(lower=0) * 12
    df_merged['p_rate_low'] = df_merged['p_lower'].clip(lower=0) * 12
    df_merged['p_rate_high'] = df_merged['p_upper'].clip(lower=0) * 12
    
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
# 2. CARGA GIS INTELIGENTE (SIMPLIFICACIÓN POSTGIS) - LA SOLUCIÓN ESTRUCTURAL
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner="Optimizando geometrías en base de datos...")
def cargar_capas_gis_inteligente(minx, miny, maxx, maxy, is_full_region=False):
    """
    Usa ST_SimplifyPreserveTopology de PostGIS para reducir la carga.
    - Si es región completa: Simplificación agresiva (0.005 grados).
    - Si es zoom local: Simplificación leve o nula.
    """
    engine = get_engine()
    layers = {"suelos": None, "hidro": None, "bocatomas": None}
    
    if not engine: return layers
    
    # Tolerancia: 0.005 (~500m) para vista general, 0.0001 (~10m) para detalle
    tolerancia = 0.005 if is_full_region else 0.0001
    
    try:
        with engine.connect() as conn:
            # 1. SUELOS
            try:
                # Traemos columnas CLAVE + Geometría simplificada desde el SQL
                q_suelos = text(f"""
                    SELECT id, "UCS", "PAISAJE", "CLIMA", "LITOLOGÍA", "COMPONENTE", 
                           ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, {tolerancia})) as geometry 
                    FROM suelos
                    WHERE ST_Intersects(geom, ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326))
                """)
                df_s = pd.read_sql(q_suelos, conn)
                if not df_s.empty:
                    df_s['geometry'] = df_s['geometry'].apply(lambda x: shape(json.loads(x)) if x else None)
                    layers["suelos"] = gpd.GeoDataFrame(df_s, geometry='geometry', crs="EPSG:4326")
            except Exception as e: pass

            # 2. HIDROGEOLOGÍA
            try:
                q_hidro = text(f"""
                    SELECT id, nombre_zona, potencial, unidad_geo, 
                           ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, {tolerancia})) as geometry 
                    FROM zonas_hidrogeologicas
                    WHERE ST_Intersects(geom, ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326))
                """)
                df_h = pd.read_sql(q_hidro, conn)
                if not df_h.empty:
                    df_h['geometry'] = df_h['geometry'].apply(lambda x: shape(json.loads(x)) if x else None)
                    layers["hidro"] = gpd.GeoDataFrame(df_h, geometry='geometry', crs="EPSG:4326")
            except Exception: pass

            # 3. BOCATOMAS (Puntos no se simplifican, solo se filtran)
            try:
                q_boca = text(f"""
                    SELECT *, ST_AsGeoJSON(geom) as geometry 
                    FROM bocatomas
                    WHERE ST_Intersects(geom, ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326))
                """)
                df_b = pd.read_sql(q_boca, conn)
                if not df_b.empty:
                    df_b['geometry'] = df_b['geometry'].apply(lambda x: shape(json.loads(x)) if x else None)
                    layers["bocatomas"] = gpd.GeoDataFrame(df_b, geometry='geometry', crs="EPSG:4326")
            except Exception: pass
            
    except Exception as e: print(f"Error DB: {e}")
    return layers

# ==============================================================================
# 3. INTERFAZ Y PANEL LATERAL (RESTAURADO)
# ==============================================================================

ids_dummy, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

# Detectar si estamos viendo "Todo Antioquia" o un área grande (> 1 grado)
is_full_region = False
if gdf_zona is not None:
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    if (maxx - minx) > 1.0 or (maxy - miny) > 1.0:
        is_full_region = True

st.sidebar.divider()
st.sidebar.header("🌱 Parámetros del Suelo")

# Sliders Recuperados
col_s1, col_s2 = st.sidebar.columns(2)
pct_bosque = col_s1.number_input("% Bosque", 0, 100, 60)
pct_cultivo = col_s2.number_input("% Agrícola", 0, 100, 30)
pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
st.sidebar.metric("% Urbano", f"{pct_urbano}%")

ki_ponderado = ((pct_bosque * 0.50) + (pct_cultivo * 0.30) + (pct_urbano * 0.10)) / 100.0
st.sidebar.metric("Coef. Infiltración ($K_i$)", f"{ki_ponderado:.2f}")

st.sidebar.divider()
st.sidebar.header("⚙️ Configuración Pronóstico")
horizonte_meses = st.sidebar.slider("Meses Futuros:", 12, 60, 24)
ruido = st.sidebar.slider("Incertidumbre:", 0.0, 2.0, 0.5)

# ==============================================================================
# 4. MOTOR PRINCIPAL
# ==============================================================================

if gdf_zona is not None and not gdf_zona.empty:
    engine = get_engine()
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    
    # Cargar Estaciones y Lluvias
    try:
        q_est = text("""
            SELECT id_estacion, nom_est, alt_est, ST_Y(geom::geometry) as lat, ST_X(geom::geometry) as lon
            FROM estaciones 
            WHERE ST_X(geom::geometry) BETWEEN :minx AND :maxx 
              AND ST_Y(geom::geometry) BETWEEN :miny AND :maxy
        """)
        df_est = pd.read_sql(q_est, engine, params={"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})
        
        if not df_est.empty:
            sel_local = st.multiselect("📍 Filtrar Estaciones:", df_est['nom_est'].unique(), default=df_est['nom_est'].unique())
            df_est_filtered = df_est[df_est['nom_est'].isin(sel_local)]
            
            if not df_est_filtered.empty:
                ids_v = df_est_filtered['id_estacion'].unique()
                ids_s = ",".join([f"'{str(x)}'" for x in ids_v])
                
                # Datos para Gráficos
                q_avg = text(f"SELECT id_estacion_fk as id_estacion, AVG(precipitation)*12 as p_anual FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY 1")
                df_avg = pd.read_sql(q_avg, engine)
                q_serie = text(f"SELECT fecha_mes_año as fecha, AVG(precipitation) as p_mensual FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY 1 ORDER BY 1")
                df_serie = pd.read_sql(q_serie, engine)
                
                # Unificar
                df_est_filtered['id_estacion'] = df_est_filtered['id_estacion'].astype(str)
                df_avg['id_estacion'] = df_avg['id_estacion'].astype(str)
                df_work = pd.merge(df_est_filtered, df_avg, on='id_estacion', how='inner')
                df_work['alt_est'] = df_work['alt_est'].fillna(altitud_ref)
                
                # CÁLCULO TURC
                df_res_avg = calculate_turc_advanced(df_work, ki_ponderado)
                
                # KPIs
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Lluvia (mm/año)", f"{df_res_avg['p_anual'].mean():.0f}")
                k2.metric("ETR (mm/año)", f"{df_res_avg['etr_mm'].mean():.0f}")
                k3.metric("Recarga (mm/año)", f"{df_res_avg['recarga_mm'].mean():.0f}", delta="Oferta")
                k4.metric("Estaciones", len(df_res_avg))
                
                st.divider()
                
                # PESTAÑAS
                tab_graf, tab_mapa, tab_data = st.tabs(["📈 Análisis & Pronóstico", "🗺️ Mapa Integrado", "💾 Descargas"])
                
                # --- TAB 1: GRÁFICOS ---
                with tab_graf:
                    if not df_serie.empty and PROPHET_AVAILABLE:
                        with st.spinner("Proyectando..."):
                            df_fc = run_prophet_forecast_hybrid(df_serie, horizonte_meses, altitud_ref, ki_ponderado, ruido)
                            if not df_fc.empty:
                                fig = go.Figure()
                                h = df_fc[df_fc['tipo']=='Histórico']
                                p = df_fc[df_fc['tipo']=='Proyección']
                                
                                fig.add_trace(go.Bar(x=h['ds'], y=h['p_rate'], name='Lluvia', marker_color='rgba(135, 206, 235, 0.5)'))
                                fig.add_trace(go.Scatter(x=df_fc['ds'], y=df_fc['etr_est'], name='ETR', line=dict(color='orange', width=2, dash='dot')))
                                fig.add_trace(go.Scatter(x=h['ds'], y=h['recarga_est'], name='Recarga Hist', line=dict(color='blue', width=2), fill='tozeroy'))
                                fig.add_trace(go.Scatter(x=p['ds'], y=p['recarga_est'], name='Recarga Fut', line=dict(color='dodgerblue', width=2, dash='dash')))
                                fig.add_trace(go.Scatter(x=p['ds'], y=p['recarga_high'], line=dict(width=0), showlegend=False))
                                fig.add_trace(go.Scatter(x=p['ds'], y=p['recarga_low'], line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.1)', name='Incertidumbre'))
                                fig.update_layout(title="Dinámica Hídrica", height=450, hovermode="x unified")
                                st.plotly_chart(fig, use_container_width=True)
                    else: st.warning("Datos insuficientes.")

                # --- TAB 2: MAPA INTEGRADO (OPTIMIZADO) ---
                with tab_mapa:
                    st.markdown("### Visor de Recursos Hídricos")
                    if is_full_region:
                        st.info("ℹ️ Vista de región completa: Mostrando geometría simplificada para mayor velocidad.")
                    
                    # CARGA GIS INTELIGENTE
                    capas = cargar_capas_gis_inteligente(minx, miny, maxx, maxy, is_full_region)
                    gdf_suelos = capas["suelos"]
                    gdf_hidro = capas["hidro"]
                    gdf_bocas = capas["bocatomas"]
                    
                    c_lat = df_est_filtered['lat'].mean()
                    c_lon = df_est_filtered['lon'].mean()
                    m = folium.Map(location=[c_lat, c_lon], zoom_start=11 if not is_full_region else 9, tiles="CartoDB positron")
                    
                    # 1. SUELOS (Fondo)
                    if gdf_suelos is not None:
                        def style_suelos(feature):
                            p = str(feature['properties'].get('PAISAJE', '')).lower()
                            color = '#e5f5e0'
                            if 'montaña' in p: color = '#8c510a'
                            elif 'valle' in p: color = '#c7eae5'
                            elif 'lomerío' in p or 'lomerio' in p: color = '#35978f'
                            elif 'planicie' in p: color = '#dfc27d'
                            return {'fillColor': color, 'color': 'gray', 'weight': 0.5, 'fillOpacity': 0.5}
                        
                        fg_suelos = folium.FeatureGroup(name="🌱 Suelos")
                        # Tooltip con columnas REALES de tu archivo
                        folium.GeoJson(
                            gdf_suelos,
                            style_function=style_suelos,
                            tooltip=folium.GeoJsonTooltip(fields=['UCS', 'PAISAJE', 'CLIMA', 'LITOLOGÍA'], aliases=['UCS:', 'Paisaje:', 'Clima:', 'Litología:'])
                        ).add_to(fg_suelos)
                        fg_suelos.add_to(m)

                    # 2. HIDROGEOLOGÍA
                    if gdf_hidro is not None:
                        fg_hidro = folium.FeatureGroup(name="💧 Hidrogeología", show=False)
                        folium.GeoJson(
                            gdf_hidro,
                            style_function=lambda x: {'fillColor': '#4393c3', 'color': '#2166ac', 'weight': 1, 'fillOpacity': 0.5},
                            tooltip=folium.GeoJsonTooltip(fields=['nombre_zona', 'potencial', 'unidad_geo'])
                        ).add_to(fg_hidro)
                        fg_hidro.add_to(m)

                    # 3. RECARGA (Raster + Puntos)
                    if len(df_res_avg) >= 3:
                        fg_recarga = folium.FeatureGroup(name="🌧️ Recarga (Puntos)", show=True)
                        try:
                            gx, gy = np.mgrid[minx:maxx:200j, miny:maxy:200j]
                            grid = interpolacion_segura_suave(df_res_avg[['lon','lat']].values, df_res_avg['recarga_mm'].values, gx, gy)
                            cmap = LinearColormap(['orange', 'yellow', 'green', 'blue'], vmin=0, vmax=2500, caption="Recarga (mm/año)")
                            m.add_child(cmap)
                            folium.raster_layers.ImageOverlay(
                                image=grid.T, bounds=[[miny, minx], [maxy, maxx]], opacity=0.5,
                                colormap=lambda x: (x, 0, 0, x)
                            ).add_to(fg_recarga)
                        except: pass

                        for _, row in df_res_avg.iterrows():
                            val = row['recarga_mm']
                            color = 'blue' if val > 1500 else ('green' if val > 800 else 'orange')
                            folium.CircleMarker(
                                [row['lat'], row['lon']], radius=5, color='white', weight=1, fill=True, fill_color=color, fill_opacity=1,
                                tooltip=f"{row['nom_est']}: {val:.0f} mm",
                                popup=f"<b>Estación:</b> {row['nom_est']}<br><b>Recarga:</b> {val:.0f} mm"
                            ).add_to(fg_recarga)
                        fg_recarga.add_to(m)

                    # 4. BOCATOMAS (Case Insensitive Search)
                    if gdf_bocas is not None:
                        fg_boca = folium.FeatureGroup(name="🚰 Bocatomas", show=True)
                        # Buscar columna nombre sin importar mayúsculas
                        cols_b = {c.upper(): c for c in gdf_bocas.columns}
                        col_nom = cols_b.get('NOMBRE', cols_b.get('BOCATOMA', list(cols_b.values())[0]))
                        col_mun = cols_b.get('MUNICIPIO', '')

                        for _, row in gdf_bocas.iterrows():
                            if row.geometry.geom_type == 'Point':
                                txt = f"{row[col_nom]}"
                                if col_mun: txt += f" ({row[col_mun]})"
                                folium.CircleMarker(
                                    [row.geometry.y, row.geometry.x], radius=4, color='red', fill=True, fill_color='darkred', fill_opacity=1,
                                    tooltip=txt
                                ).add_to(fg_boca)
                        fg_boca.add_to(m)

                    folium.LayerControl().add_to(m)
                    st_folium(m, width="100%", height=600)

                # --- TAB 3: DESCARGAS ---
                with tab_data:
                    st.subheader("💾 Exportar Resultados")
                    c1, c2 = st.columns(2)
                    csv = df_res_avg.to_csv(index=False).encode('utf-8')
                    c1.download_button("📥 Descargar Tabla (CSV)", csv, "recarga_calculada.csv", "text/csv")
                    
                    gdf_exp = gpd.GeoDataFrame(df_res_avg, geometry=gpd.points_from_xy(df_res_avg.lon, df_res_avg.lat), crs="EPSG:4326")
                    c2.download_button("🗺️ Descargar Capa (GeoJSON)", gdf_exp.to_json(), "recarga_gis.geojson", "application/json")
                    
                    st.dataframe(df_res_avg)

            else: st.warning("Sin estaciones en zona.")
        else: st.warning("Sin datos climáticos.")
    except Exception as e: st.error(f"Error técnico: {e}")
else:
    st.info("👈 Seleccione una zona.")