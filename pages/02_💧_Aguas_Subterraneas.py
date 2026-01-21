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
from shapely.geometry import shape
from branca.colormap import LinearColormap

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

st.title("💧 Estimación de Recarga (Modelo Turc + Zonificación)")

# ==============================================================================
# 1. HERRAMIENTAS AUXILIARES
# ==============================================================================

def find_col(df, candidates):
    """Busca la primera columna que coincida (ignorando mayúsculas/minúsculas)."""
    cols = df.columns
    for cand in candidates:
        # Búsqueda exacta
        if cand in cols: return cand
        # Búsqueda insensible a mayúsculas
        for c in cols:
            if c.lower() == cand.lower(): return c
            # Búsqueda parcial (ej: 'potencial' en 'potencial_hidro')
            if cand.lower() in c.lower(): return c
    return None

def interpolacion_robusta(points, values, grid_x, grid_y):
    try:
        grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
        mask = np.isnan(grid_z)
        if np.any(mask):
            grid_n = griddata(points, values, (grid_x, grid_y), method='nearest')
            grid_z[mask] = grid_n[mask]
        return grid_z
    except: return griddata(points, values, (grid_x, grid_y), method='nearest')

# ==============================================================================
# 2. LÓGICA MATEMÁTICA (TURC)
# ==============================================================================

def calculate_turc_row(p_anual, altitud, ki):
    temp = 30 - (0.0065 * altitud)
    l_t = 300 + 25*temp + 0.05*(temp**3)
    if l_t == 0: l_t = 0.001
    etr = p_anual / np.sqrt(0.9 + (p_anual / l_t)**2)
    recarga = (p_anual - etr) * ki
    return etr, max(0, recarga)

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

def run_prophet_forecast_hybrid(df_hist, months_ahead, altitud_ref, ki, ruido_factor):
    if not PROPHET_AVAILABLE: return pd.DataFrame()
    df_prophet = df_hist.rename(columns={'fecha': 'ds', 'p_mensual': 'y'})
    m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True).fit(df_prophet)
    future = m.make_future_dataframe(periods=months_ahead, freq='ME') 
    forecast = m.predict(future)
    df_merged = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], df_prophet[['ds', 'y']], on='ds', how='left')
    df_merged['p_final'] = df_merged['y'].combine_first(df_merged['yhat'])
    df_merged['p_rate'] = df_merged['p_final'].clip(lower=0) * 12
    df_merged['p_lower'] = df_merged['p_rate'] * (1 - 0.1*ruido_factor)
    df_merged['p_upper'] = df_merged['p_rate'] * (1 + 0.1*ruido_factor)
    def calc_vec(p): return calculate_turc_row(p, altitud_ref, ki)
    central = df_merged['p_rate'].apply(calc_vec)
    df_merged['etr_est'] = [x[0] for x in central]
    df_merged['recarga_est'] = [x[1] for x in central]
    df_merged['recarga_low'] = df_merged['p_lower'].apply(lambda x: calculate_turc_row(x, altitud_ref, ki)[1])
    df_merged['recarga_high'] = df_merged['p_upper'].apply(lambda x: calculate_turc_row(x, altitud_ref, ki)[1])
    df_merged['tipo'] = np.where(df_merged['ds'] <= df_prophet['ds'].max(), 'Histórico', 'Proyección')
    return df_merged

# ==============================================================================
# 3. CARGA GIS INTELIGENTE (SELECT * + DETECCIÓN DE COLUMNAS)
# ==============================================================================

# ==============================================================================
# BLOQUE 1: REEMPLAZAR LA FUNCIÓN DE CARGA (Copiar y Pegar sobre la función existente)
# ==============================================================================

@st.cache_data(ttl=60, show_spinner="Cargando capas y detectando campos...")
def cargar_capas_gis_light():
    engine = get_engine()
    layers = {}
    
    if not engine: return layers
    
    # Tolerancia para polígonos (Suelos/Hidro)
    tol = 0.001
    
    try:
        with engine.connect() as conn:
            
            # --- 1. SUELOS (Buscando UCS y PAISAJE) ---
            try:
                # Usamos SELECT * para traer UCS, PAISAJE, etc.
                q = text(f'SELECT *, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, {tol})) as geometry_json FROM suelos LIMIT 1000')
                df = pd.read_sql(q, conn)
                if not df.empty:
                    df['geometry'] = df['geometry_json'].apply(lambda x: shape(json.loads(x)) if x else None)
                    layers['suelos'] = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
            except Exception as e: print(f"Error Suelos: {e}")

            # --- 2. HIDROGEOLOGÍA (Buscando COD y Unidad_Geo) ---
            try:
                q = text(f'SELECT *, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, {tol})) as geometry_json FROM zonas_hidrogeologicas LIMIT 1000')
                df = pd.read_sql(q, conn)
                if not df.empty:
                    df['geometry'] = df['geometry_json'].apply(lambda x: shape(json.loads(x)) if x else None)
                    layers['hidro'] = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
            except Exception as e: print(f"Error Hidro: {e}")

            # --- 3. BOCATOMAS (Buscando Nombre_Acu y Municipio) ---
            try:
                # Consulta simple sin filtros espaciales complejos para asegurar que carguen
                q = text('SELECT *, ST_AsGeoJSON(geom) as geometry_json FROM bocatomas LIMIT 2000')
                df = pd.read_sql(q, conn)
                if not df.empty:
                    df['geometry'] = df['geometry_json'].apply(lambda x: shape(json.loads(x)) if x else None)
                    layers['bocatomas'] = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
            except Exception as e: print(f"Error Bocatomas: {e}")
            
    except Exception as e: st.error(f"Error en BD: {e}")
    return layers

# ==============================================================================
# 4. INTERFAZ
# ==============================================================================

ids_dummy, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

st.sidebar.divider()
st.sidebar.header("🌱 Parámetros del Suelo")
col_s1, col_s2 = st.sidebar.columns(2)
pct_bosque = col_s1.number_input("% Bosque", 0, 100, 60)
pct_cultivo = col_s2.number_input("% Agrícola", 0, 100, 30)
pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
st.sidebar.metric("% Urbano", f"{pct_urbano}%")
ki_ponderado = ((pct_bosque * 0.50) + (pct_cultivo * 0.30) + (pct_urbano * 0.10)) / 100.0
st.sidebar.metric("Coef. Infiltración ($K_i$)", f"{ki_ponderado:.2f}")

st.sidebar.divider()
horizonte_meses = st.sidebar.slider("Meses Futuros", 12, 60, 24)
ruido = st.sidebar.slider("Incertidumbre", 0.0, 2.0, 0.5)

# ==============================================================================
# 5. MOTOR PRINCIPAL
# ==============================================================================

if gdf_zona is not None and not gdf_zona.empty:
    engine = get_engine()
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    
    # Cargar Datos Climáticos
    try:
        q_est = text("""
            SELECT id_estacion, nom_est, alt_est, ST_Y(geom::geometry) as lat, ST_X(geom::geometry) as lon
            FROM estaciones 
            WHERE ST_X(geom::geometry) BETWEEN :minx AND :maxx 
              AND ST_Y(geom::geometry) BETWEEN :miny AND :maxy
        """)
        df_est = pd.read_sql(q_est, engine, params={"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})
        
        if not df_est.empty:
            sel_local = st.multiselect("📍 Estaciones:", df_est['nom_est'].unique(), default=df_est['nom_est'].unique())
            df_est_filtered = df_est[df_est['nom_est'].isin(sel_local)]
            
            if not df_est_filtered.empty:
                ids_v = df_est_filtered['id_estacion'].unique()
                ids_s = ",".join([f"'{str(x)}'" for x in ids_v])
                
                # Datos Lluvia
                q_avg = text(f"SELECT id_estacion_fk as id_estacion, AVG(precipitation)*12 as p_anual FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY 1")
                df_avg = pd.read_sql(q_avg, engine)
                q_serie = text(f"SELECT fecha_mes_año as fecha, AVG(precipitation) as p_mensual FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY 1 ORDER BY 1")
                df_serie = pd.read_sql(q_serie, engine)
                
                # Turc
                df_est_filtered['id_estacion'] = df_est_filtered['id_estacion'].astype(str)
                df_avg['id_estacion'] = df_avg['id_estacion'].astype(str)
                df_work = pd.merge(df_est_filtered, df_avg, on='id_estacion', how='inner')
                df_work['alt_est'] = df_work['alt_est'].fillna(altitud_ref)
                df_res_avg = calculate_turc_advanced(df_work, ki_ponderado)
                
                # KPIs
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Lluvia", f"{df_res_avg['p_anual'].mean():.0f}")
                k2.metric("ETR", f"{df_res_avg['etr_mm'].mean():.0f}")
                k3.metric("Recarga", f"{df_res_avg['recarga_mm'].mean():.0f}", delta="mm/año")
                k4.metric("Estaciones", len(df_res_avg))
                
                # BLOQUE 2: 
                # ==============================================================================
                
                st.divider()
                
                # Definimos las pestañas (Incluyendo la NUEVA de Isolíneas)
                tab_evol, tab_mapa, tab_iso, tab_data = st.tabs([
                    "📈 Análisis Temporal", 
                    "🗺️ Mapa Integrado (Capas)", 
                    "🌈 Mapa Isolíneas Recarga", 
                    "💾 Datos"
                ])
                
                # --- PESTAÑA 1: GRÁFICOS ---
                with tab_evol:
                    if not df_serie.empty and PROPHET_AVAILABLE:
                        with st.spinner("Procesando proyección..."):
                            df_fc = run_prophet_forecast_hybrid(df_serie, horizonte_meses, altitud_ref, ki_ponderado, ruido)
                            if not df_fc.empty:
                                fig = go.Figure()
                                h = df_fc[df_fc['tipo']=='Histórico']
                                p = df_fc[df_fc['tipo']=='Proyección']
                                fig.add_trace(go.Bar(x=h['ds'], y=h['p_rate'], name='Lluvia', marker_color='rgba(135, 206, 235, 0.5)'))
                                fig.add_trace(go.Scatter(x=df_fc['ds'], y=df_fc['etr_est'], name='ETR', line=dict(color='orange', width=1.5, dash='dot')))
                                fig.add_trace(go.Scatter(x=h['ds'], y=h['recarga_est'], name='Recarga Histórica', line=dict(color='blue', width=2), fill='tozeroy'))
                                fig.add_trace(go.Scatter(x=p['ds'], y=p['recarga_est'], name='Recarga Futura', line=dict(color='dodgerblue', width=2, dash='dash')))
                                fig.update_layout(title="Dinámica Hídrica", height=400, hovermode="x unified")
                                st.plotly_chart(fig, use_container_width=True)

                # --- PESTAÑA 2: MAPA INTEGRADO (Capas Vectoriales) ---
                with tab_mapa:
                    st.markdown("### Visor de Capas: Suelos, Hidrogeología y Bocatomas")
                    
                    layers = cargar_capas_gis_light()
                    gdf_s = layers.get('suelos')
                    gdf_h = layers.get('hidro')
                    gdf_b = layers.get('bocatomas')
                    
                    # Mapa Base
                    c_lat = df_est_filtered['lat'].mean()
                    c_lon = df_est_filtered['lon'].mean()
                    m = folium.Map(location=[c_lat, c_lon], zoom_start=11, tiles="CartoDB positron")
                    
                    # A. SUELOS (Usando UCS y PAISAJE)
                    if gdf_s is not None and not gdf_s.empty:
                        # Buscar nombres de columna insensitivos a mayúsculas
                        cols = {c.lower(): c for c in gdf_s.columns}
                        col_ucs = cols.get('ucs', cols.get('unidad_suelo'))
                        col_paisaje = cols.get('paisaje', cols.get('textura'))
                        
                        def style_s(feature):
                            # Colorear por PAISAJE si existe
                            color = '#e5f5e0'
                            if col_paisaje:
                                p = str(feature['properties'].get(col_paisaje, '')).lower()
                                if 'montaña' in p: color = '#d95f0e'
                                elif 'valle' in p: color = '#fff7bc'
                                elif 'lomerío' in p: color = '#addd8e'
                            return {'fillColor': color, 'color': 'gray', 'weight': 0.5, 'fillOpacity': 0.4}
                        
                        fg_s = folium.FeatureGroup(name="🌱 Suelos")
                        
                        # Tooltip dinámico
                        fields_s = [c for c in [col_ucs, col_paisaje] if c]
                        folium.GeoJson(
                            gdf_s, style_function=style_s,
                            tooltip=folium.GeoJsonTooltip(fields=fields_s, aliases=fields_s) if fields_s else None
                        ).add_to(fg_s)
                        fg_s.add_to(m)

                    # B. HIDROGEOLOGÍA (Usando COD y Potencial_)
                    if gdf_h is not None and not gdf_h.empty:
                        cols = {c.lower(): c for c in gdf_h.columns}
                        col_cod = cols.get('cod', 'id')
                        col_pot = cols.get('potencial_', cols.get('potencial'))
                        col_uni = cols.get('unidad_geo')
                        
                        fg_h = folium.FeatureGroup(name="💧 Hidrogeología", show=False)
                        
                        fields_h = [c for c in [col_cod, col_pot, col_uni] if c]
                        folium.GeoJson(
                            gdf_h,
                            style_function=lambda x: {'fillColor': '#2c7fb8', 'color': '#253494', 'weight': 1, 'fillOpacity': 0.4},
                            tooltip=folium.GeoJsonTooltip(fields=fields_h, aliases=fields_h) if fields_h else None
                        ).add_to(fg_h)
                        fg_h.add_to(m)

                    # C. BOCATOMAS (Usando Nombre_Acu)
                    if gdf_b is not None and not gdf_b.empty:
                        fg_b = folium.FeatureGroup(name="🚰 Bocatomas", show=True)
                        
                        # Buscar columnas específicas que pediste
                        cols = {c.lower(): c for c in gdf_b.columns}
                        col_nom = cols.get('nombre_acu', cols.get('nombre'))
                        col_mun = cols.get('municipio')
                        col_ver = cols.get('veredas')
                        col_fte = cols.get('fuente_aba')
                        
                        # Construir tooltip rico
                        fields_b = [c for c in [col_nom, col_mun, col_ver, col_fte] if c]
                        aliases_b = [c + ":" for c in fields_b]
                        
                        if fields_b:
                            folium.GeoJson(
                                gdf_b,
                                marker=folium.CircleMarker(radius=4, color='red', fill=True, fill_color='darkred'),
                                tooltip=folium.GeoJsonTooltip(fields=fields_b, aliases=aliases_b)
                            ).add_to(fg_b)
                        else:
                            # Fallback si no encuentra columnas
                            for _, row in gdf_b.iterrows():
                                if row.geometry.geom_type == 'Point':
                                    folium.CircleMarker(
                                        [row.geometry.y, row.geometry.x], radius=4, color='red', fill=True, tooltip="Bocatoma"
                                    ).add_to(fg_b)
                                    
                        fg_b.add_to(m)

                    folium.LayerControl().add_to(m)
                    st_folium(m, width="100%", height=500)

                # --- PESTAÑA 3: MAPA ISOLÍNEAS (Exclusiva) ---
                with tab_iso:
                    st.markdown("### 🌈 Mapa de Isolíneas de Recarga")
                    if len(df_res_avg) >= 3:
                        m_iso = folium.Map(location=[c_lat, c_lon], zoom_start=11, tiles="CartoDB dark_matter")
                        
                        # Generar Grid de Interpolación
                        try:
                            # Usamos límites ajustados a los datos para evitar espacios vacíos
                            pad = 0.02
                            min_lon, max_lon = df_res_avg['lon'].min() - pad, df_res_avg['lon'].max() + pad
                            min_lat, max_lat = df_res_avg['lat'].min() - pad, df_res_avg['lat'].max() + pad
                            
                            gx, gy = np.mgrid[min_lon:max_lon:200j, min_lat:max_lat:200j]
                            grid = interpolacion_robusta(df_res_avg[['lon','lat']].values, df_res_avg['recarga_mm'].values, gx, gy)
                            
                            # Escala de Color
                            cmap = LinearColormap(['#ffffcc', '#a1dab4', '#41b6c4', '#225ea8'], vmin=0, vmax=2000, caption="Recarga (mm/año)")
                            m_iso.add_child(cmap)
                            
                            # Capa Raster (Imagen)
                            folium.raster_layers.ImageOverlay(
                                image=grid.T, bounds=[[min_lat, min_lon], [max_lat, max_lon]], opacity=0.7, 
                                colormap=lambda x: cmap(x)
                            ).add_to(m_iso)
                            
                            # Puntos de Control (Estaciones)
                            for _, row in df_res_avg.iterrows():
                                val = row['recarga_mm']
                                folium.CircleMarker(
                                    [row['lat'], row['lon']], radius=5, color='white', weight=1, fill=True, fill_color='black',
                                    tooltip=f"{row['nom_est']}: {val:.0f} mm"
                                ).add_to(m_iso)
                                
                            st_folium(m_iso, width="100%", height=600, key="mapa_isolineas")
                            
                        except Exception as e:
                            st.error(f"Error generando isolíneas: {e}")
                    else:
                        st.warning("⚠️ Se necesitan al menos 3 estaciones para generar isolíneas.")

                # --- PESTAÑA 4: DATOS ---
                with tab_data:
                    c1, c2 = st.columns(2)
                    csv = df_res_avg.to_csv(index=False).encode('utf-8')
                    c1.download_button("📥 Descargar CSV", csv, "balance.csv", "text/csv")
                    st.dataframe(df_res_avg)

            else: st.warning("Seleccione estaciones.")
        else: st.warning("Zona sin estaciones.")
    except Exception as e: st.error(f"Error técnico: {e}")
else: st.info("👈 Seleccione una zona.")