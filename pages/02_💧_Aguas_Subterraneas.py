import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import geopandas as gpd
from scipy.interpolate import griddata
import sys
import os

# --- INTENTO DE IMPORTAR PROPHET (Con fallback por si falla) ---
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

# --- FUNCIONES AUXILIARES ---
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
    """Añade capas de contexto recortadas."""
    try:
        roi = gdf_zona.buffer(0.05)
        gdf_m = load_geojson_cached("MunicipiosAntioquia.geojson")
        if gdf_m is not None:
            gdf_c = gpd.clip(gdf_m, roi)
            for _, r in gdf_c.iterrows():
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', line=dict(width=0.5, color='gray'), hoverinfo='skip'))
        
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
    """Genera superficie continua usando Linear + Nearest para rellenar."""
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='linear')
    mask = np.isnan(grid_z0)
    if np.any(mask):
        grid_z1 = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z0[mask] = grid_z1[mask]
    return grid_z0

# --- CÁLCULOS HIDROLÓGICOS ---
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

# --- FORECASTING CON PROPHET ---
def run_prophet_forecast(df_hist, months_ahead, altitud_ref, ki):
    """
    Usa Prophet para proyectar Precipitación y luego calcula Turc sobre esa proyección.
    df_hist debe tener columnas: ['fecha', 'p_mensual']
    """
    if not PROPHET_AVAILABLE:
        st.warning("Prophet no instalado. Usando proyección lineal simple.")
        return pd.DataFrame()

    # Preparar datos para Prophet
    df_prophet = df_hist.rename(columns={'fecha': 'ds', 'p_mensual': 'y'})
    
    # Modelo
    m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    m.fit(df_prophet)
    
    # Futuro
    future = m.make_future_dataframe(periods=months_ahead, freq='M')
    forecast = m.predict(future)
    
    # Procesar resultados
    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    
    # Convertir Precipitación Mensual predicha a Anualizada (x12) para Turc
    # Nota: Turc es fórmula anual. P_mensual * 12 es una aproximación para ver tendencias.
    res['p_anual_est'] = res['yhat'].clip(lower=0) * 12 
    res['p_lower'] = res['yhat_lower'].clip(lower=0) * 12
    res['p_upper'] = res['yhat_upper'].clip(lower=0) * 12
    
    # Calcular Turc sobre la proyección
    # Usamos vectorización simple aplicando la función row a las columnas
    # Necesitamos aplicar la función fila a fila o vectorizada
    
    def calc_vector(p_val):
        etr, rec = calculate_turc_row(p_val, altitud_ref, ki)
        return pd.Series([etr, rec])

    res[['etr_est', 'recarga_est']] = res['p_anual_est'].apply(calc_vector)
    res[['etr_low', 'recarga_low']] = res['p_lower'].apply(calc_vector)
    res[['etr_up', 'recarga_up']] = res['p_upper'].apply(calc_vector)
    
    return res

# --- INTERFAZ BARRA LATERAL ---
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
st.sidebar.header("⚙️ Pronóstico")
horizonte_meses = st.sidebar.slider("Meses a Proyectar:", 12, 60, 24)

# --- MOTOR PRINCIPAL ---
if gdf_zona is not None and not gdf_zona.empty:
    engine = create_engine(st.secrets["DATABASE_URL"])
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    
    # 1. BÚSQUEDA SEGURA
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
                
                # --- DETECCIÓN DE COLUMNA FECHA ---
                try:
                    cols = [c.lower() for c in pd.read_sql("SELECT * FROM precipitacion_mensual LIMIT 0", engine).columns]
                    date_col = next((c for c in cols if c in ['fecha', 'date', 'time', 'fecha_registro']), None)
                    if not date_col: raise Exception("No se encontró columna de fecha")
                except: date_col = 'fecha' # Fallback

                # 2. DATOS PARA MAPA (PROMEDIO)
                q_avg = text(f"""
                    SELECT id_estacion_fk as id_estacion, AVG(precipitation)*12 as p_anual 
                    FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY 1
                """)
                df_avg = pd.read_sql(q_avg, engine)
                
                # 3. DATOS PARA SERIE (HISTÓRICO MENSUAL)
                q_serie = text(f"""
                    SELECT {date_col} as fecha, AVG(precipitation) as p_mensual
                    FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) 
                    GROUP BY 1 ORDER BY 1
                """)
                df_serie = pd.read_sql(q_serie, engine)
                
                # Procesar Mapa
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
                c4.metric("Recarga Total", f"{df_res_avg['recarga_mm'].mean():.0f} mm", delta="Oferta")
                
                st.divider()
                
                # --- PESTAÑAS ---
                tab_evol, tab_mapa, tab_data = st.tabs(["📈 Evolución & Pronóstico", "🗺️ Superficie de Recarga", "💾 Datos"])
                
                with tab_evol:
                    if not df_serie.empty and PROPHET_AVAILABLE:
                        with st.spinner("🧠 Ejecutando modelo Prophet + Turc..."):
                            # Usamos la altura promedio para el cálculo histórico agregado
                            alt_prom = df_est_filtered['alt_est'].mean() if not df_est_filtered['alt_est'].isna().all() else altitud_ref
                            
                            # Generar Pronóstico
                            df_forecast = run_prophet_forecast(df_serie, horizonte_meses, alt_prom, ki_ponderado)
                            
                            # GRAFICAR
                            fig = go.Figure()
                            
                            # Zona Histórica (Hasta hoy)
                            now = pd.Timestamp.today()
                            hist = df_forecast[df_forecast['ds'] <= now]
                            fut = df_forecast[df_forecast['ds'] > now]
                            
                            # 1. Precipitación (Barras)
                            fig.add_trace(go.Bar(
                                x=hist['ds'], y=hist['p_anual_est'], 
                                name='Precipitación (Hist)', marker_color='rgba(135, 206, 235, 0.5)'
                            ))
                            
                            # 2. ETR (Línea Naranja)
                            fig.add_trace(go.Scatter(
                                x=hist['ds'], y=hist['etr_est'],
                                name='ETR (Evapotranspiración)', line=dict(color='orange', width=2)
                            ))
                            
                            # 3. Recarga (Línea Azul y Relleno)
                            fig.add_trace(go.Scatter(
                                x=hist['ds'], y=hist['recarga_est'],
                                name='Recarga (Hist)', line=dict(color='blue', width=2),
                                fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.1)'
                            ))
                            
                            # 4. Proyecciones (Punteadas)
                            fig.add_trace(go.Scatter(
                                x=fut['ds'], y=fut['recarga_est'],
                                name='Pronóstico Recarga', line=dict(color='blue', width=2, dash='dot')
                            ))
                            
                            # Intervalo de Confianza Recarga
                            fig.add_trace(go.Scatter(
                                x=fut['ds'], y=fut['recarga_up'], mode='lines', line=dict(width=0), showlegend=False
                            ))
                            fig.add_trace(go.Scatter(
                                x=fut['ds'], y=fut['recarga_low'], mode='lines', line=dict(width=0),
                                fill='tonexty', fillcolor='rgba(0,0,255,0.05)', name='Incertidumbre'
                            ))

                            fig.update_layout(
                                title="Dinámica Temporal: Lluvia vs ETR vs Recarga (Histórico + Pronóstico)",
                                yaxis_title="Lámina (mm/año estimado)",
                                xaxis_title="Fecha", hovermode="x unified", height=550
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.info("ℹ️ La gráfica muestra valores mensuales anualizados para facilitar la comparación con promedios anuales.")
                    else:
                        st.warning("Datos insuficientes para pronóstico o Prophet no instalado.")

                with tab_mapa:
                    if len(df_res_avg) >= 3:
                        with st.spinner("Interpolando superficie..."):
                            # Grid
                            gx, gy = np.mgrid[minx:maxx:100j, miny:maxy:100j]
                            grid_R = interpolacion_segura(
                                df_res_avg[['lon', 'lat']].values, 
                                df_res_avg['recarga_mm'].values, gx, gy
                            )
                            
                            # Centro mapa
                            c_lat = df_res_avg['lat'].mean()
                            c_lon = df_res_avg['lon'].mean()
                            
                            fig_m = go.Figure()
                            
                            # 1. Superficie Continua
                            fig_m.add_trace(go.Contour(
                                z=grid_R.T, x=np.linspace(minx, maxx, 100), y=np.linspace(miny, maxy, 100),
                                colorscale="Blues", colorbar=dict(title="Recarga (mm)", len=0.8),
                                hoverinfo='z', contours=dict(coloring='heatmap', showlabels=False),
                                opacity=0.7, connectgaps=True, name='Recarga'
                            ))
                            
                            # 2. Contexto
                            add_context_layers(fig_m, gdf_zona)
                            
                            # 3. Puntos Estaciones
                            fig_m.add_trace(go.Scattermapbox(
                                lon=df_res_avg['lon'], lat=df_res_avg['lat'],
                                mode='markers', marker=dict(size=9, color='black'),
                                text=df_res_avg['nom_est'] + '<br>Recarga: ' + df_res_avg['recarga_mm'].round(0).astype(str),
                                hoverinfo='text', name='Estaciones'
                            ))
                            
                            fig_m.update_layout(
                                mapbox_style="carto-positron",
                                mapbox=dict(center=dict(lat=c_lat, lon=c_lon), zoom=10),
                                margin={"r":0,"t":0,"l":0,"b":0}, height=650
                            )
                            st.plotly_chart(fig_m, use_container_width=True)
                    else:
                        st.warning("Se necesitan al menos 3 estaciones para interpolar el mapa.")

                with tab_data:
                    csv = df_res_avg.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Descargar CSV (Balance Hídrico)", csv, "balance_hidrico.csv", "text/csv")
                    
                    # Exportar GeoJSON
                    gdf_exp = gpd.GeoDataFrame(df_res_avg, geometry=gpd.points_from_xy(df_res_avg.lon, df_res_avg.lat), crs="EPSG:4326")
                    st.download_button("🗺️ Descargar GeoJSON (GIS)", gdf_exp.to_json(), "recarga_gis.geojson", "application/json")
                    
                    st.dataframe(df_res_avg[['nom_est', 'p_anual', 'etr_mm', 'recarga_mm']])

            else: st.warning("Seleccione estaciones.")
        else: st.warning("Zona sin estaciones.")
    except Exception as e: st.error(f"Error: {e}")
else: st.info("👈 Seleccione una zona.")