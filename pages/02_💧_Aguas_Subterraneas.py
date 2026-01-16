import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
import geopandas as gpd
import sys
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors 
except ImportError:
    st.error("Error al importar módulos base.")
    st.stop()

st.title("💧 Modelo Hidrogeológico: Recarga y Escenarios")

# --- FUNCIONES AUXILIARES DE CAPAS ---
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
        # Municipios
        gdf_m = load_geojson_cached("MunicipiosAntioquia.geojson")
        if gdf_m is not None:
            gdf_c = gpd.clip(gdf_m, roi)
            for _, r in gdf_c.iterrows():
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scattermapbox(lon=list(x), lat=list(y), mode='lines', line=dict(width=1, color='gray'), hoverinfo='skip'))
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

# --- FUNCIONES MATEMÁTICAS ---
def calculate_turc_row(p_anual, altitud, ki):
    temp = 30 - (0.0065 * altitud)
    l_t = 300 + 25*temp + 0.05*(temp**3)
    etr = p_anual / np.sqrt(0.9 + (p_anual / l_t)**2)
    recarga = (p_anual - etr) * ki
    return etr, recarga

def calculate_turc_advanced(df, ki):
    df = df.copy()
    df['alt_est'] = pd.to_numeric(df['alt_est'], errors='coerce')
    df['p_anual'] = pd.to_numeric(df['p_anual'], errors='coerce')
    
    df['temp_est'] = 30 - (0.0065 * df['alt_est'])
    t = df['temp_est']
    l_t = 300 + 25*t + 0.05*(t**3)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        df['etr_mm'] = df['p_anual'] / np.sqrt(0.9 + (df['p_anual'] / l_t)**2)
    
    df['excedente_mm'] = (df['p_anual'] - df['etr_mm']).clip(lower=0)
    df['recarga_mm'] = df['excedente_mm'] * ki
    return df

def generate_projections(df_turc, years_ahead=1, ruido_factor=0.5):
    recarga_actual = df_turc['recarga_mm'].mean()
    if np.isnan(recarga_actual): recarga_actual = 0
    n_months = (years_ahead - 2025) * 12 + 12
    if n_months < 12: n_months = 12
    fechas = pd.date_range(start='2026-01-01', periods=n_months, freq='M')
    
    np.random.seed(42)
    trend = np.linspace(recarga_actual, recarga_actual * 0.95, n_months)
    seasonality = np.sin(np.linspace(0, 4*np.pi, n_months)) * (recarga_actual * 0.15)
    noise = np.random.normal(0, recarga_actual * 0.05 * ruido_factor, n_months)
    valores = trend + seasonality + noise
    
    return pd.DataFrame({
        'Fecha': fechas,
        'Recarga Estimada': valores.clip(min=0),
        'Limite Superior': (valores * 1.15).clip(min=0),
        'Limite Inferior': (valores * 0.85).clip(min=0)
    })

# --- BARRA LATERAL ---
ids_dummy, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

st.sidebar.divider()
st.sidebar.header("🌱 Parámetros del Suelo")
st.sidebar.info("Ajuste los coeficientes según la cobertura del suelo.")
k_bosque, k_cultivo, k_urbano = 0.50, 0.30, 0.10
col_s1, col_s2, col_s3 = st.sidebar.columns(3)
pct_bosque = col_s1.number_input("% Bosque", 0, 100, 60)
pct_cultivo = col_s2.number_input("% Agrícola", 0, 100, 30)
pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
col_s3.metric("% Urbano", f"{pct_urbano}%")
ki_ponderado = ((pct_bosque * k_bosque) + (pct_cultivo * k_cultivo) + (pct_urbano * k_urbano)) / 100.0
st.sidebar.metric("Coef. Infiltración ($K_i$)", f"{ki_ponderado:.2f}")

st.sidebar.divider()
st.sidebar.header("⚙️ Proyección")
activar_proyeccion = st.sidebar.checkbox("Activar Proyección", value=True)
horizonte = st.sidebar.selectbox("Horizonte:", [2026, 2027, 2030], index=0)
ruido = st.sidebar.slider("Incertidumbre (+/-)", 0.0, 2.0, 0.5)

# --- MOTOR DE DATOS ---
if gdf_zona is not None and not gdf_zona.empty:
    engine = create_engine(st.secrets["DATABASE_URL"])
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    
    # 1. Buscar Estaciones
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
                
                # 2. DATOS PROMEDIO
                q_clima_avg = text(f"""
                    SELECT id_estacion_fk as id_estacion, AVG(precipitation)*12 as p_anual 
                    FROM precipitacion_mensual 
                    WHERE id_estacion_fk IN ({ids_s}) 
                    GROUP BY id_estacion_fk
                """)
                df_clima_avg = pd.read_sql(q_clima_avg, engine)
                
                # Merge
                df_est_filtered['id_estacion'] = df_est_filtered['id_estacion'].astype(str)
                df_clima_avg['id_estacion'] = df_clima_avg['id_estacion'].astype(str)
                df_work = pd.merge(df_est_filtered, df_clima_avg, on='id_estacion', how='inner')
                df_work['alt_est'] = df_work['alt_est'].fillna(altitud_ref)
                
                # Cálculo
                df_res_avg = calculate_turc_advanced(df_work, ki_ponderado)
                
                # KPIs
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Precipitación Media", f"{df_res_avg['p_anual'].mean():.0f} mm/año")
                kpi2.metric("Evapotranspiración", f"{df_res_avg['etr_mm'].mean():.0f} mm/año")
                kpi3.metric("Coef. Infiltración", f"{ki_ponderado:.2f}")
                kpi4.metric("Recarga Total", f"{df_res_avg['recarga_mm'].mean():.0f} mm/año", delta="Infiltración Profunda")
                
                st.divider()

                # --- PESTAÑAS ---
                tab_hist, tab_serie, tab_mapa, tab_proy = st.tabs(["📊 Balance Hídrico", "📈 Evolución Histórica", "🗺️ Mapa de Recarga", "🔮 Proyección"])
                
                with tab_hist:
                    vals = [df_res_avg['p_anual'].mean(), df_res_avg['etr_mm'].mean(), df_res_avg['excedente_mm'].mean(), df_res_avg['recarga_mm'].mean()]
                    cats = ['Precipitación', 'ETR', 'Excedente', 'Recarga']
                    fig_hist = go.Figure(go.Bar(x=cats, y=vals, text=[f"{v:.0f}" for v in vals], textposition='auto', marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#17becf']))
                    fig_hist.update_layout(title="Balance Hídrico Medio Anual (Turc)", yaxis_title="mm/año", height=500)
                    st.plotly_chart(fig_hist, use_container_width=True)

                with tab_serie:
                    with st.spinner("Consultando historia climática..."):
                        # --- AUTODETECCIÓN DE COLUMNA DE FECHA (Nueva Lógica) ---
                        # Consultamos una fila para ver los nombres de columnas
                        check_cols = pd.read_sql("SELECT * FROM precipitacion_mensual LIMIT 1", engine)
                        cols = [c.lower() for c in check_cols.columns]
                        
                        # Buscamos el nombre correcto
                        date_col = 'fecha' # Default
                        if 'date' in cols: date_col = 'date'
                        elif 'fecha_registro' in cols: date_col = 'fecha_registro'
                        elif 'time' in cols: date_col = 'time'
                        
                        # Consulta Agrupada usando la columna detectada
                        q_hist = text(f"""
                            SELECT EXTRACT(YEAR FROM {date_col}) as anio, AVG(precipitation)*12 as p_anual 
                            FROM precipitacion_mensual 
                            WHERE id_estacion_fk IN ({ids_s}) 
                            GROUP BY 1 ORDER BY 1
                        """)
                        
                        try:
                            df_hist_serie = pd.read_sql(q_hist, engine)
                            
                            if not df_hist_serie.empty:
                                alt_prom = df_est_filtered['alt_est'].mean()
                                if pd.isna(alt_prom): alt_prom = altitud_ref
                                df_hist_serie['etr'], df_hist_serie['recarga'] = zip(*df_hist_serie['p_anual'].apply(lambda p: calculate_turc_row(p, alt_prom, ki_ponderado)))
                                
                                fig_serie = go.Figure()
                                fig_serie.add_trace(go.Bar(x=df_hist_serie['anio'], y=df_hist_serie['p_anual'], name='Precipitación', marker_color='#A9D0F5', opacity=0.6))
                                fig_serie.add_trace(go.Scatter(x=df_hist_serie['anio'], y=df_hist_serie['etr'], name='ETR', line=dict(color='orange', width=2)))
                                fig_serie.add_trace(go.Scatter(x=df_hist_serie['anio'], y=df_hist_serie['recarga'], name='Recarga Potencial', line=dict(color='blue', width=3), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'))
                                fig_serie.update_layout(title="Evolución Histórica: Lluvia vs Recarga", xaxis_title="Año", yaxis_title="mm/año", hovermode="x unified", height=500)
                                st.plotly_chart(fig_serie, use_container_width=True)
                            else:
                                st.warning("No hay datos históricos detallados disponibles.")
                        except Exception as e:
                            st.error(f"Error consultando histórico (Columna: {date_col}): {e}")

                with tab_mapa:
                    center_lat = df_res_avg['lat'].mean()
                    center_lon = df_res_avg['lon'].mean()
                    fig_map = go.Figure()
                    add_context_layers(fig_map, gdf_zona)
                    fig_map.add_trace(go.Scattermapbox(
                        lon=df_res_avg['lon'], lat=df_res_avg['lat'],
                        mode='markers',
                        marker=dict(size=12, color=df_res_avg['recarga_mm'], colorscale='RdYlBu', showscale=True, colorbar=dict(title="Recarga (mm)")),
                        text=df_res_avg['nom_est'] + '<br>Recarga: ' + df_res_avg['recarga_mm'].round(1).astype(str) + ' mm',
                        hoverinfo='text'
                    ))
                    fig_map.update_layout(mapbox_style="carto-positron", mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=10), margin={"r":0,"t":0,"l":0,"b":0}, height=600)
                    st.plotly_chart(fig_map, use_container_width=True)

                with tab_proy:
                    if activar_proyeccion:
                        df_proy = generate_projections(df_res_avg, years_ahead=int(horizonte), ruido_factor=ruido)
                        fig_proy = go.Figure()
                        fig_proy.add_trace(go.Scatter(x=df_proy['Fecha'], y=df_proy['Limite Superior'], mode='lines', line=dict(width=0), showlegend=False))
                        fig_proy.add_trace(go.Scatter(x=df_proy['Fecha'], y=df_proy['Limite Inferior'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', name='Incertidumbre'))
                        fig_proy.add_trace(go.Scatter(x=df_proy['Fecha'], y=df_proy['Recarga Estimada'], mode='lines', line=dict(color='blue', width=3), name='Recarga Estimada'))
                        fig_proy.update_layout(title=f"Proyección (Horizonte {horizonte})", yaxis_title="Recarga (mm/año)", hovermode="x unified", height=500)
                        st.plotly_chart(fig_proy, use_container_width=True)
            else:
                st.warning("Seleccione estaciones.")
        else:
            st.warning("Zona sin estaciones.")
            
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👈 Seleccione una zona.")