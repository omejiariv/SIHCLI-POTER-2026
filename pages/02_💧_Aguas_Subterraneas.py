import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import geopandas as gpd
from scipy.interpolate import griddata
import sys
import os

# --- PROPHET (Opcional) ---
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors 
except ImportError:
    st.error("Error al importar módulos base.")
    st.stop()

st.title("💧 Estimación de Recarga (Modelo Turc + Escenarios)")

# --- 1. DOCUMENTACIÓN ---
with st.expander("📘 Metodología: Modelo Turc y Proyecciones", expanded=False):
    st.markdown("""
    * **Balance Hídrico (Turc):** Estima la recarga como el excedente de la precipitación menos la evapotranspiración real (ETR), corregido por un coeficiente de infiltración ($K_i$).
    * **Proyección:** Utiliza el algoritmo Prophet para modelar la tendencia y estacionalidad de la lluvia, proyectando escenarios futuros.
    * **Interpolación:** Genera superficies continuas mediante métodos geoestadísticos para identificar zonas de recarga preferencial.
    """)

# --- FUNCIONES GIS ---
@st.cache_data(ttl=3600)
def load_geojson_cached(filename):
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
    if os.path.exists(filepath):
        try:
            gdf = gpd.read_file(filepath)
            if gdf.crs and gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
            return gdf
        except: pass
    return None

def add_context_layers(fig, gdf_zona):
    """Añade capas de contexto (Municipios, Cuencas)."""
    try:
        roi = gdf_zona.buffer(0.05)
        # Municipios
        gdf_m = load_geojson_cached("MunicipiosAntioquia.geojson")
        if gdf_m is not None:
            gdf_c = gpd.clip(gdf_m, roi)
            for _, r in gdf_c.iterrows():
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', line=dict(width=0.5, color='gray'), hoverinfo='skip'))
        # Cuencas
        gdf_cu = load_geojson_cached("SubcuencasAinfluencia.geojson")
        if gdf_cu is not None:
            gdf_c = gpd.clip(gdf_cu, roi)
            for _, r in gdf_c.iterrows():
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', line=dict(width=1, color='blue'), hoverinfo='skip'))
    except: pass

def interpolacion_segura(points, values, grid_x, grid_y):
    """
    Interpolación Robusta (Linear + Nearest). 
    Más estable que RBF para visualización web rápida.
    """
    # 1. Linear (Suave dentro del convex hull)
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear')
    # 2. Nearest (Rellena bordes y evita huecos)
    mask = np.isnan(grid_z0)
    if np.any(mask):
        grid_z1 = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z0[mask] = grid_z1[mask]
    return grid_z0

# --- CÁLCULOS TURC ---
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

# --- FORECASTING ---
def run_prophet_forecast_gap_filling(df_hist, months_ahead, altitud_ref, ki, ruido_factor):
    """
    Proyecta desde el último dato real hasta el futuro (cubriendo el presente).
    """
    if not PROPHET_AVAILABLE: return pd.DataFrame()

    # 1. Preparar histórico
    df_prophet = df_hist.rename(columns={'fecha': 'ds', 'p_mensual': 'y'})
    last_date = df_prophet['ds'].max()
    
    # 2. Configurar Modelo
    m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    m.fit(df_prophet)
    
    # 3. Definir Horizonte (Desde último dato hasta Hoy + Meses Futuros)
    target_date = pd.Timestamp.today() + pd.DateOffset(months=months_ahead)
    # Calcular cuántos meses faltan desde el último dato real hasta el objetivo
    months_gap = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
    if months_gap < 1: months_gap = 12
    
    future = m.make_future_dataframe(periods=months_gap, freq='M')
    forecast = m.predict(future)
    
    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    
    # 4. Calcular Hidrología (Mensual Anualizada)
    # Sin suavizado excesivo para ver variabilidad
    res['p_rate'] = res['yhat'].clip(lower=0) * 12
    
    # Aplicar Incertidumbre del usuario
    res['p_rate_low'] = (res['yhat_lower'] * (1 - 0.1*ruido_factor)).clip(lower=0) * 12
    res['p_rate_high'] = (res['yhat_upper'] * (1 + 0.1*ruido_factor)).clip(lower=0) * 12
    
    # Vectorización Turc
    def calc_vec(p): return calculate_turc_row(p, altitud_ref, ki)
    
    # Calcular Central
    central = res['p_rate'].apply(calc_vec)
    res['etr_est'] = [x[0] for x in central]
    res['recarga_est'] = [x[1] for x in central]
    
    # Calcular Bandas
    low = res['p_rate_low'].apply(calc_vec)
    res['recarga_low'] = [x[1] for x in low]
    
    high = res['p_rate_high'].apply(calc_vec)
    res['recarga_high'] = [x[1] for x in high]
    
    return res

# --- INTERFAZ ---
ids_dummy, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

st.sidebar.divider()
st.sidebar.header("🌱 Parámetros del Suelo")
col_s1, col_s2, col_s3 = st.sidebar.columns(3)
pct_bosque = col_s1.number_input("% Bosque", 0, 100, 60)
pct_cultivo = col_s2.number_input("% Agrícola", 0, 100, 30)
pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
col_s3.metric("% Urbano", f"{pct_urbano}%")
ki_ponderado = ((pct_bosque * 0.50) + (pct_cultivo * 0.30) + (pct_urbano * 0.10)) / 100.0
st.sidebar.metric("Coef. Infiltración ($K_i$)", f"{ki_ponderado:.2f}")

st.sidebar.divider()
st.sidebar.header("⚙️ Configuración Pronóstico")
horizonte_meses = st.sidebar.slider("Meses Futuros:", 12, 60, 24)
ruido = st.sidebar.slider("Incertidumbre Climática:", 0.0, 2.0, 0.5)

# --- MOTOR ---
if gdf_zona is not None and not gdf_zona.empty:
    engine = create_engine(st.secrets["DATABASE_URL"])
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    
    q_est = text("""
        SELECT id_estacion, nom_est, alt_est, ST_Y(geom::geometry) as lat, ST_X(geom::geometry) as lon
        FROM estaciones 
        WHERE ST_X(geom::geometry) BETWEEN :minx AND :maxx 
          AND ST_Y(geom::geometry) BETWEEN :miny AND :maxy
    """)
    
    try:
        df_est = pd.read_sql(q_est, engine, params={"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})
        
        if not df_est.empty:
            estaciones_disponibles = df_est['nom_est'].unique()
            sel_local = st.multiselect("📍 Filtrar Estaciones:", estaciones_disponibles, default=estaciones_disponibles)
            df_est_filtered = df_est[df_est['nom_est'].isin(sel_local)]
            
            if not df_est_filtered.empty:
                ids_v = df_est_filtered['id_estacion'].unique()
                ids_s = ",".join([f"'{str(x)}'" for x in ids_v])
                
                # --- DATOS PROMEDIO ---
                q_avg = text(f"""
                    SELECT id_estacion_fk as id_estacion, AVG(precipitation)*12 as p_anual 
                    FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY 1
                """)
                df_avg = pd.read_sql(q_avg, engine)
                
                # --- DATOS SERIE (HISTÓRICO) ---
                q_serie = text(f"""
                    SELECT fecha_mes_año as fecha, AVG(precipitation) as p_mensual
                    FROM precipitacion_mensual 
                    WHERE id_estacion_fk IN ({ids_s}) 
                    GROUP BY 1 ORDER BY 1
                """)
                df_serie = pd.read_sql(q_serie, engine)
                
                # Procesar KPIs y Mapa
                df_est_filtered['id_estacion'] = df_est_filtered['id_estacion'].astype(str)
                df_avg['id_estacion'] = df_avg['id_estacion'].astype(str)
                df_work = pd.merge(df_est_filtered, df_avg, on='id_estacion', how='inner')
                df_work['alt_est'] = df_work['alt_est'].fillna(altitud_ref)
                df_res_avg = calculate_turc_advanced(df_work, ki_ponderado)
                
                # KPIs
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Lluvia Media", f"{df_res_avg['p_anual'].mean():.0f} mm")
                c2.metric("ETR", f"{df_res_avg['etr_mm'].mean():.0f} mm")
                c3.metric("Coef. $K_i$", f"{ki_ponderado:.2f}")
                c4.metric("Recarga Total", f"{df_res_avg['recarga_mm'].mean():.0f} mm", delta="Oferta Hídrica")
                
                st.divider()
                
                # --- PESTAÑAS ---
                tab_evol, tab_mapa, tab_data = st.tabs(["📈 Evolución & Pronóstico", "🗺️ Mapa de Recarga", "💾 Descargas"])
                
                with tab_evol:
                    if not df_serie.empty and PROPHET_AVAILABLE:
                        with st.spinner("Proyectando escenarios climáticos..."):
                            alt_prom = df_est_filtered['alt_est'].mean() if not df_est_filtered['alt_est'].isna().all() else altitud_ref
                            
                            # Forecast Gap Filling
                            df_forecast = run_prophet_forecast_gap_filling(df_serie, horizonte_meses, alt_prom, ki_ponderado, ruido)
                            
                            if not df_forecast.empty:
                                fig = go.Figure()
                                now = pd.Timestamp.today()
                                
                                # Separar Pasado (Real/Simulado) y Futuro
                                # Prophet devuelve toda la serie temporal
                                
                                # 1. Recarga (Continua)
                                fig.add_trace(go.Scatter(
                                    x=df_forecast['ds'], y=df_forecast['recarga_est'],
                                    name='Recarga Estimada', line=dict(color='blue', width=2),
                                    fill='tozeroy', fillcolor='rgba(0,0,255,0.05)'
                                ))
                                
                                # 2. Lluvia (Fondo)
                                fig.add_trace(go.Bar(
                                    x=df_forecast['ds'], y=df_forecast['p_rate'],
                                    name='Precipitación', marker_color='rgba(135, 206, 235, 0.3)',
                                    hoverinfo='skip' # Para no saturar el hover
                                ))
                                
                                # 3. ETR
                                fig.add_trace(go.Scatter(
                                    x=df_forecast['ds'], y=df_forecast['etr_est'],
                                    name='ETR', line=dict(color='orange', width=1.5, dash='dot')
                                ))
                                
                                # 4. Incertidumbre Futura
                                fut_mask = df_forecast['ds'] > now
                                fig.add_trace(go.Scatter(
                                    x=df_forecast[fut_mask]['ds'], y=df_forecast[fut_mask]['recarga_high'],
                                    mode='lines', line=dict(width=0), showlegend=False
                                ))
                                fig.add_trace(go.Scatter(
                                    x=df_forecast[fut_mask]['ds'], y=df_forecast[fut_mask]['recarga_low'],
                                    mode='lines', line=dict(width=0), fill='tonexty', 
                                    fillcolor='rgba(0,0,255,0.1)', name='Rango Incertidumbre'
                                ))
                                
                                # Línea vertical de "Hoy"
                                fig.add_vline(x=now.timestamp() * 1000, line_width=1, line_dash="dash", line_color="green")
                                fig.add_annotation(x=now, y=df_forecast['recarga_est'].max(), text="Hoy", showarrow=False, yshift=10)

                                fig.update_layout(
                                    title="Dinámica de Recarga: Histórico + Proyección (Gap Filling)",
                                    yaxis_title="Tasa (mm/año)", hovermode="x unified", height=500
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Error en la generación del pronóstico.")
                    else:
                        st.warning("Datos históricos insuficientes para proyectar.")

                with tab_mapa:
                    if len(df_res_avg) >= 3:
                        with st.spinner("Generando superficie continua..."):
                            # Grid
                            gx, gy = np.mgrid[minx:maxx:100j, miny:maxy:100j]
                            
                            # Interpolación
                            grid_R = interpolacion_segura(
                                df_res_avg[['lon', 'lat']].values, 
                                df_res_avg['recarga_mm'].values, gx, gy
                            )
                            
                            fig_m = go.Figure()
                            
                            # 1. Superficie Continua (Contour)
                            fig_m.add_trace(go.Contour(
                                z=grid_R.T, x=np.linspace(minx, maxx, 100), y=np.linspace(miny, maxy, 100),
                                colorscale="Blues", 
                                colorbar=dict(title="Recarga (mm/año)"), 
                                hoverinfo='z', 
                                contours=dict(coloring='heatmap', showlabels=False),
                                opacity=0.7, 
                                connectgaps=True,
                                line_smoothing=0.85
                            ))
                            
                            # 2. Contexto
                            add_context_layers(fig_m, gdf_zona)
                            
                            # 3. Puntos Estaciones
                            fig_m.add_trace(go.Scattermapbox(
                                lon=df_res_avg['lon'], lat=df_res_avg['lat'],
                                mode='markers', marker=dict(size=8, color='black'),
                                text=df_res_avg['nom_est'] + '<br>Recarga: ' + df_res_avg['recarga_mm'].round(0).astype(str),
                                hoverinfo='text', name='Estaciones'
                            ))
                            
                            fig_m.update_layout(
                                mapbox_style="carto-positron",
                                mapbox=dict(center=dict(lat=df_res_avg['lat'].mean(), lon=df_res_avg['lon'].mean()), zoom=10),
                                margin={"r":0,"t":0,"l":0,"b":0}, height=650
                            )
                            # CLAVE ÚNICA PARA EVITAR PARPADEO
                            st.plotly_chart(fig_m, use_container_width=True, key="mapa_recarga_estable")
                    else:
                        st.warning("⚠️ Se necesitan al menos 3 estaciones para interpolar el mapa.")

                with tab_data:
                    st.subheader("Centro de Descargas")
                    c_d1, c_d2 = st.columns(2)
                    
                    # CSV
                    csv = df_res_avg.to_csv(index=False).encode('utf-8')
                    c_d1.download_button("📥 Descargar Tabla (CSV)", csv, "balance_hidrico.csv", "text/csv")
                    
                    # GeoJSON
                    gdf_exp = gpd.GeoDataFrame(df_res_avg, geometry=gpd.points_from_xy(df_res_avg.lon, df_res_avg.lat), crs="EPSG:4326")
                    c_d2.download_button("🗺️ Descargar Capa GIS (GeoJSON)", gdf_exp.to_json(), "recarga_gis.geojson", "application/json")
                    
                    st.dataframe(df_res_avg[['nom_est', 'p_anual', 'etr_mm', 'recarga_mm']])

            else: st.warning("Seleccione al menos una estación.")
        else: st.warning("Zona sin estaciones.")
    except Exception as e: st.error(f"Error técnico: {e}")
else: st.info("👈 Seleccione una zona.")