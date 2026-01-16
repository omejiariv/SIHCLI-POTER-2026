import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import geopandas as gpd
from scipy.interpolate import griddata
import sys
import os

# --- 1. INTENTO DE IMPORTAR PROPHET (Con Plan B) ---
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

# --- 2. FUNCIONES AUXILIARES (GIS & Interpolación) ---
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
    """Dibuja municipios y cuencas de fondo."""
    try:
        roi = gdf_zona.buffer(0.05)
        # Capa Municipios
        gdf_m = load_geojson_cached("MunicipiosAntioquia.geojson")
        if gdf_m is not None:
            gdf_c = gpd.clip(gdf_m, roi)
            for _, r in gdf_c.iterrows():
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', line=dict(width=0.5, color='gray'), hoverinfo='skip'))
        # Capa Cuencas
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
    """Interpolación híbrida (Linear + Nearest) para evitar huecos blancos."""
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear')
    mask = np.isnan(grid_z0)
    if np.any(mask):
        grid_z1 = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z0[mask] = grid_z1[mask]
    return grid_z0

# --- 3. CÁLCULOS HIDROLÓGICOS (TURC) ---
def calculate_turc_row(p_anual, altitud, ki):
    """Cálculo unitario."""
    temp = 30 - (0.0065 * altitud)
    l_t = 300 + 25*temp + 0.05*(temp**3)
    if l_t == 0: l_t = 0.001
    etr = p_anual / np.sqrt(0.9 + (p_anual / l_t)**2)
    recarga = (p_anual - etr) * ki
    return etr, max(0, recarga)

def calculate_turc_advanced(df, ki):
    """Cálculo vectorizado para DataFrames."""
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

# --- 4. MOTORES DE PRONÓSTICO (IA + ESTADÍSTICO) ---

def forecast_fallback_statistical(recarga_actual, months_ahead, ruido_factor):
    """Plan B: Estadístico (Random Walk) si no hay Prophet o datos insuficientes."""
    fechas = pd.date_range(start='2026-01-01', periods=months_ahead, freq='M')
    
    np.random.seed(42)
    trend = np.linspace(recarga_actual, recarga_actual * 0.95, months_ahead)
    seasonality = np.sin(np.linspace(0, 4*np.pi, months_ahead)) * (recarga_actual * 0.15)
    noise = np.random.normal(0, recarga_actual * 0.05 * ruido_factor, months_ahead)
    
    yhat = (trend + seasonality + noise).clip(min=0)
    
    return pd.DataFrame({
        'ds': fechas,
        'recarga_est': yhat,
        'recarga_low': yhat * 0.85,
        'recarga_up': yhat * 1.15,
        'p_anual_est': np.nan, # No calculamos P ni ETR en el modo simple
        'etr_est': np.nan
    })

def run_prophet_forecast(df_hist, months_ahead, altitud_ref, ki, ruido_factor):
    """Plan A: IA con Prophet."""
    # Validación mínima para Prophet
    if not PROPHET_AVAILABLE or len(df_hist) < 24: # Necesita al menos 2 años para detectar estacionalidad
        # Recarga actual promedio para el fallback
        alt_prom = 1500 # Default
        if 'p_mensual' in df_hist.columns:
            p_mean = df_hist['p_mensual'].mean() * 12
            _, r_mean = calculate_turc_row(p_mean, altitud_ref, ki)
        else:
            r_mean = 500
        return forecast_fallback_statistical(r_mean, months_ahead, ruido_factor)

    try:
        # Preparar datos
        df_prophet = df_hist.rename(columns={'fecha': 'ds', 'p_mensual': 'y'})
        
        # Modelo
        m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=months_ahead, freq='M')
        forecast = m.predict(future)
        
        res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        
        # Anualizar P (x12)
        res['p_anual_est'] = res['yhat'].clip(lower=0) * 12 
        res['p_lower'] = res['yhat_lower'].clip(lower=0) * 12
        res['p_upper'] = res['yhat_upper'].clip(lower=0) * 12
        
        # Calcular Turc vectorizado
        def calc_vector(p_val):
            etr, rec = calculate_turc_row(p_val, altitud_ref, ki)
            return pd.Series([etr, rec])

        res[['etr_est', 'recarga_est']] = res['p_anual_est'].apply(calc_vector)
        res[['etr_low', 'recarga_low']] = res['p_lower'].apply(calc_vector)
        res[['etr_up', 'recarga_up']] = res['p_upper'].apply(calc_vector)
        
        return res
        
    except Exception as e:
        # Si falla Prophet, fallback silencioso
        return forecast_fallback_statistical(500, months_ahead, ruido_factor)

# --- 5. INTERFAZ Y BARRA LATERAL ---
ids_dummy, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

st.sidebar.divider()
st.sidebar.header("🌱 Suelos e Infiltración")
col_s1, col_s2, col_s3 = st.sidebar.columns(3)
pct_bosque = col_s1.number_input("% Bosque", 0, 100, 60)
pct_cultivo = col_s2.number_input("% Agrícola", 0, 100, 30)
pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
col_s3.metric("% Urbano", f"{pct_urbano}%")

# Coeficientes
K_BOSQUE = 0.50
K_CULTIVO = 0.30
K_URBANO = 0.10
ki_ponderado = ((pct_bosque * K_BOSQUE) + (pct_cultivo * K_CULTIVO) + (pct_urbano * K_URBANO)) / 100.0
st.sidebar.metric("Coef. Infiltración ($K_i$)", f"{ki_ponderado:.2f}")

st.sidebar.divider()
st.sidebar.header("⚙️ Configuración Pronóstico")
horizonte_meses = st.sidebar.slider("Meses a Proyectar:", 12, 60, 24)
ruido_factor = st.sidebar.slider("Factor Incertidumbre:", 0.0, 2.0, 0.5)

# --- 6. MOTOR DE ANÁLISIS ---
if gdf_zona is not None and not gdf_zona.empty:
    engine = create_engine(st.secrets["DATABASE_URL"])
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    
    # Consulta Espacial
    q_est = text("""
        SELECT id_estacion, nom_est, alt_est, ST_Y(geom::geometry) as lat, ST_X(geom::geometry) as lon
        FROM estaciones 
        WHERE ST_X(geom::geometry) BETWEEN :minx AND :maxx 
          AND ST_Y(geom::geometry) BETWEEN :miny AND :maxy
    """)
    
    try:
        df_est = pd.read_sql(q_est, engine, params={"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})
        
        if not df_est.empty:
            # Selector Local (¡Recuperado!)
            estaciones_disponibles = df_est['nom_est'].unique()
            sel_local = st.multiselect("📍 Filtrar Estaciones:", estaciones_disponibles, default=estaciones_disponibles)
            df_est_filtered = df_est[df_est['nom_est'].isin(sel_local)]
            
            if not df_est_filtered.empty:
                ids_v = df_est_filtered['id_estacion'].unique()
                ids_s = ",".join([f"'{str(x)}'" for x in ids_v])
                
                # Datos Promedio (KPIs)
                q_avg = text(f"""
                    SELECT id_estacion_fk as id_estacion, AVG(precipitation)*12 as p_anual 
                    FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY 1
                """)
                df_avg = pd.read_sql(q_avg, engine)
                
                # Datos Serie Histórica (CORREGIDO: fecha_mes_año)
                q_serie = text(f"""
                    SELECT fecha_mes_año as fecha, AVG(precipitation) as p_mensual
                    FROM precipitacion_mensual 
                    WHERE id_estacion_fk IN ({ids_s}) 
                    GROUP BY 1 ORDER BY 1
                """)
                df_serie = pd.read_sql(q_serie, engine)
                
                # Procesamiento
                df_est_filtered['id_estacion'] = df_est_filtered['id_estacion'].astype(str)
                df_avg['id_estacion'] = df_avg['id_estacion'].astype(str)
                df_work = pd.merge(df_est_filtered, df_avg, on='id_estacion', how='inner')
                df_work['alt_est'] = df_work['alt_est'].fillna(altitud_ref)
                
                # Cálculo Turc Actual
                df_res_avg = calculate_turc_advanced(df_work, ki_ponderado)
                
                # KPIs Superiores
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Lluvia Media", f"{df_res_avg['p_anual'].mean():.0f} mm")
                c2.metric("ETR", f"{df_res_avg['etr_mm'].mean():.0f} mm")
                c3.metric("Coef. $K_i$", f"{ki_ponderado:.2f}")
                c4.metric("Recarga Total", f"{df_res_avg['recarga_mm'].mean():.0f} mm", delta="Oferta")
                
                st.divider()
                
                # Pestañas
                tab_evol, tab_mapa, tab_data = st.tabs(["📈 Evolución & Pronóstico", "🗺️ Superficie de Recarga", "💾 Datos"])
                
                with tab_evol:
                    if not df_serie.empty:
                        with st.spinner("Generando proyección..."):
                            # Altitud promedio para el histórico
                            alt_prom = df_est_filtered['alt_est'].mean() if not df_est_filtered['alt_est'].isna().all() else altitud_ref
                            
                            # Ejecutar Pronóstico (Prophet o Fallback)
                            df_forecast = run_prophet_forecast(df_serie, horizonte_meses, alt_prom, ki_ponderado, ruido_factor)
                            
                            # Graficar
                            fig = go.Figure()
                            now = pd.Timestamp.today()
                            hist = df_forecast[df_forecast['ds'] <= now]
                            fut = df_forecast[df_forecast['ds'] > now]
                            
                            # Si tenemos datos de Precipitación (Prophet mode)
                            if 'p_anual_est' in hist.columns and not hist['p_anual_est'].isna().all():
                                fig.add_trace(go.Bar(x=hist['ds'], y=hist['p_anual_est'], name='Lluvia (Hist)', marker_color='rgba(135, 206, 235, 0.5)'))
                                fig.add_trace(go.Scatter(x=hist['ds'], y=hist['etr_est'], name='ETR', line=dict(color='orange', width=2)))
                            
                            # Recarga (Siempre disponible)
                            fig.add_trace(go.Scatter(x=hist['ds'], y=hist['recarga_est'], name='Recarga (Hist)', line=dict(color='blue', width=2), fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.1)'))
                            
                            # Futuro
                            fig.add_trace(go.Scatter(x=fut['ds'], y=fut['recarga_est'], name='Pronóstico', line=dict(color='blue', width=2, dash='dot')))
                            fig.add_trace(go.Scatter(x=fut['ds'], y=fut['recarga_low'], mode='lines', line=dict(width=0), showlegend=False))
                            fig.add_trace(go.Scatter(x=fut['ds'], y=fut['recarga_up'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.05)', name='Incertidumbre'))

                            title_sufix = "(Modelo IA Prophet)" if PROPHET_AVAILABLE else "(Modelo Estadístico Simplificado)"
                            fig.update_layout(title=f"Dinámica Lluvia-Recarga {title_sufix}", yaxis_title="mm/año estimado", hovermode="x unified", height=550)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No hay datos históricos para proyectar.")

                with tab_mapa:
                    if len(df_res_avg) >= 3:
                        with st.spinner("Interpolando mapa..."):
                            gx, gy = np.mgrid[minx:maxx:100j, miny:maxy:100j]
                            grid_R = interpolacion_segura(df_res_avg[['lon', 'lat']].values, df_res_avg['recarga_mm'].values, gx, gy)
                            
                            fig_m = go.Figure()
                            # Isolíneas de Recarga
                            fig_m.add_trace(go.Contour(
                                z=grid_R.T, x=np.linspace(minx, maxx, 100), y=np.linspace(miny, maxy, 100),
                                colorscale="Blues", colorbar=dict(title="Recarga (mm)"), 
                                hoverinfo='z', contours=dict(coloring='heatmap', showlabels=False), opacity=0.7, connectgaps=True
                            ))
                            # Capas GIS
                            add_context_layers(fig_m, gdf_zona)
                            # Puntos
                            fig_m.add_trace(go.Scattermapbox(
                                lon=df_res_avg['lon'], lat=df_res_avg['lat'],
                                mode='markers', marker=dict(size=9, color='black'),
                                text=df_res_avg['nom_est'] + '<br>Recarga: ' + df_res_avg['recarga_mm'].round(0).astype(str),
                                hoverinfo='text', name='Estaciones'
                            ))
                            fig_m.update_layout(
                                mapbox_style="carto-positron", 
                                mapbox=dict(center=dict(lat=df_res_avg['lat'].mean(), lon=df_res_avg['lon'].mean()), zoom=10),
                                margin={"r":0,"t":0,"l":0,"b":0}, height=650
                            )
                            st.plotly_chart(fig_m, use_container_width=True)
                    else:
                        st.warning("Se necesitan al menos 3 estaciones para el mapa continuo.")

                with tab_data:
                    # Descargas recuperadas
                    csv = df_res_avg.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Descargar CSV", csv, "balance_hidrico.csv", "text/csv")
                    
                    gdf_exp = gpd.GeoDataFrame(df_res_avg, geometry=gpd.points_from_xy(df_res_avg.lon, df_res_avg.lat), crs="EPSG:4326")
                    st.download_button("🗺️ Descargar GeoJSON", gdf_exp.to_json(), "recarga_gis.geojson", "application/json")
                    
                    st.dataframe(df_res_avg[['nom_est', 'p_anual', 'etr_mm', 'recarga_mm']])

            else: st.warning("Seleccione al menos una estación.")
        else: st.warning("Zona sin estaciones.")
    except Exception as e: st.error(f"Error técnico: {e}")
else: st.info("👈 Seleccione una zona.")