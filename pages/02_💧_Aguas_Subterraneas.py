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

# --- NUEVOS IMPORTS PARA MAPAS DE CAPAS (SHAPEFILES) ---
import folium
from streamlit_folium import st_folium

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

st.title("💧 Aguas Subterráneas y Recarga")

# --- 1. DOCUMENTACIÓN ---
with st.expander("📘 Metodología: Modelo Turc y Proyecciones", expanded=False):
    st.markdown("""
    ### 1. Marco Conceptual
    La recarga de aguas subterráneas es la fracción de la precipitación que se infiltra en el suelo y alcanza el nivel freático.
    ### 2. Metodología: Método de Turc (1954)
    Se utiliza el modelo empírico de Turc para estimar la Evapotranspiración Real (ETR).
    """)

# --- FUNCIONES GIS EXISTENTES ---
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

def get_name_from_row(row, type_layer):
    """Busca el nombre en las columnas disponibles (insensible a mayúsculas)."""
    cols = row.index.str.lower()
    if type_layer == 'muni':
        for c in ['mpio_cnmbr', 'nombre', 'municipio', 'mpio_nomb']:
            if c in cols: return row[c]
    elif type_layer == 'cuenca':
        for c in ['n-nss3', 'subc_lbl', 'n_nss1', 'nom_cuenca', 'nombre']:
            if c in cols: return row[c]
    return "Desconocido"

def add_context_layers_cartesian(fig, gdf_zona):
    """Añade capas de contexto (Municipios y Cuencas) con estilo sutil."""
    try:
        roi = gdf_zona.buffer(0.05)
        
        # A. MUNICIPIOS
        gdf_m = load_geojson_cached("MunicipiosAntioquia.geojson")
        if gdf_m is not None:
            gdf_c = gpd.clip(gdf_m, roi)
            gdf_c.columns = gdf_c.columns.str.lower()
            for _, r in gdf_c.iterrows():
                name = get_name_from_row(r, 'muni')
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', 
                        line=dict(width=0.7, color='rgba(100, 100, 100, 0.3)', dash='dot'), 
                        hoverinfo='text', text=f"Mpio: {name}", showlegend=False
                    ))
        
        # B. CUENCAS
        gdf_cu = load_geojson_cached("SubcuencasAinfluencia.geojson")
        if gdf_cu is not None:
            gdf_c = gpd.clip(gdf_cu, roi)
            gdf_c.columns = gdf_c.columns.str.lower()
            for _, r in gdf_c.iterrows():
                name = get_name_from_row(r, 'cuenca')
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', 
                        line=dict(width=0.7, color='rgba(50, 100, 200, 0.3)', dash='dash'), 
                        hoverinfo='text', text=f"Cuenca: {name}", showlegend=False
                    ))
    except Exception as e:
        print(f"Error cargando capas contexto: {e}")

# --- NUEVA FUNCIÓN: CARGAR SHAPEFILE ZONAS ---
@st.cache_data(show_spinner="Cargando Zonas Hidrogeológicas...")
def cargar_capa_hidrogeologica():
    # Ruta dinámica a data/shapefiles/
    base_dir = os.path.dirname(__file__)
    ruta_shp = os.path.abspath(os.path.join(base_dir, '..', 'data', 'shapefiles', 'Zonas_PotHidrogeologico.shp'))
    
    if not os.path.exists(ruta_shp):
        return None

    try:
        gdf = gpd.read_file(ruta_shp)
        # Reproyección obligatoria para mapas web
        if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as e:
        print(f"Error SHP: {e}")
        return None

# --- FUNCIONES MATEMÁTICAS EXISTENTES ---
def interpolacion_segura_suave(points, values, grid_x, grid_y):
    try:
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
        mask = np.isnan(grid_z)
        if np.any(mask):
            grid_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
            grid_z[mask] = grid_nearest[mask]
        return grid_z
    except Exception:
        return griddata(points, values, (grid_x, grid_y), method='linear')

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
    last_date_real = df_prophet['ds'].max()
    m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    m.fit(df_prophet)
    target_date = pd.Timestamp.today() + pd.DateOffset(months=months_ahead)
    months_gap = (target_date.year - last_date_real.year) * 12 + (target_date.month - last_date_real.month)
    if months_gap < 1: months_gap = 12
    future = m.make_future_dataframe(periods=months_gap, freq='M')
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

# --- MOTOR PRINCIPAL ---
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
                
                # DATOS
                q_avg = text(f"""
                    SELECT id_estacion_fk as id_estacion, AVG(precipitation)*12 as p_anual 
                    FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY 1
                """)
                df_avg = pd.read_sql(q_avg, engine)
                
                q_serie = text(f"""
                    SELECT fecha_mes_año as fecha, AVG(precipitation) as p_mensual
                    FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY 1 ORDER BY 1
                """)
                df_serie = pd.read_sql(q_serie, engine)
                
                # Procesamiento
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
                
                # --- PESTAÑAS (Aquí integramos la nueva) ---
                tab_evol, tab_mapa, tab_zonas, tab_data = st.tabs([
                    "📈 Evolución & Pronóstico", 
                    "🗺️ Mapa de Recarga", 
                    "💧 Zonas Hidrogeológicas", 
                    "💾 Descargas"
                ])
                
                with tab_evol:
                    if not df_serie.empty and PROPHET_AVAILABLE:
                        with st.spinner("Integrando datos reales y proyectados..."):
                            alt_prom = df_est_filtered['alt_est'].mean() if not df_est_filtered['alt_est'].isna().all() else altitud_ref
                            df_forecast = run_prophet_forecast_hybrid(df_serie, horizonte_meses, alt_prom, ki_ponderado, ruido)
                            
                            if not df_forecast.empty:
                                fig = go.Figure()
                                hist = df_forecast[df_forecast['tipo'] == 'Histórico']
                                fut = df_forecast[df_forecast['tipo'] == 'Proyección']
                                
                                fig.add_trace(go.Bar(x=hist['ds'], y=hist['p_rate'], name='Precipitación', marker_color='rgba(173, 216, 230, 0.4)', hoverinfo='y+name'))
                                fig.add_trace(go.Scatter(x=df_forecast['ds'], y=df_forecast['etr_est'], name='ETR', line=dict(color='orange', width=1.5, dash='dot')))
                                fig.add_trace(go.Scatter(x=hist['ds'], y=hist['recarga_est'], name='Recarga Histórica', line=dict(color='blue', width=2), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'))
                                fig.add_trace(go.Scatter(x=fut['ds'], y=fut['recarga_est'], name='Recarga Proyectada', line=dict(color='dodgerblue', width=2, dash='dash')))
                                fig.add_trace(go.Scatter(x=fut['ds'], y=fut['recarga_high'], mode='lines', line=dict(width=0), showlegend=False))
                                fig.add_trace(go.Scatter(x=fut['ds'], y=fut['recarga_low'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.1)', name='Rango Incertidumbre'))
                                
                                fig.update_layout(
                                    title="Dinámica de Recarga: Datos Reales + Proyección",
                                    yaxis_title="Tasa (mm/año)", hovermode="x unified", height=550,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else: st.warning("Error en pronóstico.")
                    else: st.warning("Datos insuficientes.")

                with tab_mapa:
                    if len(df_res_avg) >= 3:
                        with st.spinner("Generando superficie suavizada (Cubic)..."):
                            gx, gy = np.mgrid[minx:maxx:200j, miny:maxy:200j]
                            grid_R = interpolacion_segura_suave(
                                df_res_avg[['lon', 'lat']].values, 
                                df_res_avg['recarga_mm'].values, gx, gy
                            )
                            
                            fig_m = go.Figure()
                            fig_m.add_trace(go.Contour(
                                z=grid_R.T, x=np.linspace(minx, maxx, 200), y=np.linspace(miny, maxy, 200),
                                colorscale="Blues", colorbar=dict(title="Recarga (mm/año)"), 
                                hovertemplate="Recarga: %{z:.0f} mm<extra></extra>",
                                contours=dict(coloring='heatmap', showlabels=False),
                                opacity=0.7, connectgaps=True, line_smoothing=1.3
                            ))
                            add_context_layers_cartesian(fig_m, gdf_zona)
                            fig_m.add_trace(go.Scatter(
                                x=df_res_avg['lon'], y=df_res_avg['lat'],
                                mode='markers', marker=dict(size=8, color='black', line=dict(width=1, color='white')),
                                text=df_res_avg['nom_est'] + '<br>Recarga: ' + df_res_avg['recarga_mm'].round(0).astype(str),
                                hoverinfo='text', name='Estaciones'
                            ))
                            for _, row in gdf_zona.iterrows():
                                geom = row.geometry
                                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                                for p in polys:
                                    x, y = p.exterior.xy
                                    fig_m.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(color='black', width=2), hoverinfo='skip', showlegend=False))

                            fig_m.update_layout(
                                title=f"Mapa de Recarga: {nombre_seleccion}",
                                height=650, margin=dict(l=0,r=0,t=40,b=0),
                                xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False),
                                plot_bgcolor='white', legend=dict(x=0, y=0, bgcolor='rgba(255,255,255,0.7)')
                            )
                            st.plotly_chart(fig_m, use_container_width=True, key="mapa_final")
                    else: st.warning("Se necesitan al menos 3 estaciones.")

                # --- NUEVA PESTAÑA INTEGRADA ---
                with tab_zonas:
                    st.markdown("### 🗺️ Zonas de Potencial Hidrogeológico")
                    st.caption("Visualización basada en shapefile local.")
                    
                    gdf_pot = cargar_capa_hidrogeologica()
                    
                    if gdf_pot is not None:
                        # Calcular centro
                        centro_lat = gdf_pot.geometry.centroid.y.mean()
                        centro_lon = gdf_pot.geometry.centroid.x.mean()
                        
                        m = folium.Map(location=[centro_lat, centro_lon], zoom_start=9, tiles="CartoDB positron")
                        
                        def estilo_zona(feature):
                            return {
                                'fillColor': '#2b8cbe', 'color': 'black',
                                'weight': 1, 'fillOpacity': 0.5
                            }
                        
                        # Detectar campos para tooltip de forma segura
                        cols_dispo = gdf_pot.columns.tolist()
                        # Intentar buscar columnas típicas o usar las primeras 3
                        fields_tip = [c for c in ['Nombre', 'NOMBRE', 'ZONA', 'Zona', 'AREA', 'Area'] if c in cols_dispo]
                        if not fields_tip: fields_tip = cols_dispo[:3]
                        
                        folium.GeoJson(
                            gdf_pot,
                            name="Potencial Hidrogeológico",
                            style_function=estilo_zona,
                            tooltip=folium.GeoJsonTooltip(fields=fields_tip)
                        ).add_to(m)
                        
                        folium.LayerControl().add_to(m)
                        st_folium(m, width="100%", height=600)
                        
                        with st.expander("Ver Datos de Atributos"):
                            st.dataframe(gdf_pot.drop(columns='geometry'))
                    else:
                        st.warning("⚠️ No se encontró el archivo 'Zonas_PotHidrogeologico.shp' en la carpeta data/shapefiles/.")

                with tab_data:
                    c_d1, c_d2 = st.columns(2)
                    csv = df_res_avg.to_csv(index=False).encode('utf-8')
                    c_d1.download_button("📥 Descargar Tabla (CSV)", csv, "balance_hidrico.csv", "text/csv")
                    gdf_exp = gpd.GeoDataFrame(df_res_avg, geometry=gpd.points_from_xy(df_res_avg.lon, df_res_avg.lat), crs="EPSG:4326")
                    c_d2.download_button("🗺️ Descargar Capa GIS (GeoJSON)", gdf_exp.to_json(), "recarga_gis.geojson", "application/json")
                    st.dataframe(df_res_avg[['nom_est', 'p_anual', 'etr_mm', 'recarga_mm']])

            else: st.warning("Seleccione estaciones.")
        else: st.warning("Zona sin estaciones.")
    except Exception as e: st.error(f"Error técnico: {e}")
else: st.info("👈 Seleccione una zona.")