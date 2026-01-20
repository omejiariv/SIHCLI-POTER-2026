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

# --- PROPHET ---
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

# --- FUNCIONES GIS AUXILIARES ---
@st.cache_data(ttl=3600)
def load_geojson_cached(filename):
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
    if os.path.exists(filepath):
        try:
            gdf = gpd.read_file(filepath)
            if gdf.crs and gdf.crs.to_string() != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
            return gdf
        except: pass
    return None

def add_context_layers_cartesian(fig, gdf_zona):
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
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(width=0.7, color='grey', dash='dot'), showlegend=False, hoverinfo='skip'))
    except Exception: pass

# --- CARGA DE CAPAS CON CORRECTOR ---
def get_shapefile_path(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, '..', 'data', 'shapefiles', filename)

@st.cache_data(show_spinner="Cargando Capas...")
def cargar_capa_base(nombre_archivo):
    ruta = get_shapefile_path(nombre_archivo)
    if not os.path.exists(ruta): return None, None
    try:
        gdf = gpd.read_file(ruta)
        msg = f"Cargado: {len(gdf)} registros. CRS detectado: {gdf.crs}"
        return gdf, msg
    except Exception as e:
        return None, str(e)

# --- UTILS PARA BUSCAR COLUMNAS (Manejo de acentos y mayúsculas) ---
def buscar_columna(df, keywords):
    """Busca una columna que coincida con alguna de las keywords (case insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for k in keywords:
        if k.lower() in cols_lower:
            return cols_lower[k.lower()]
    return None

def generar_popup_html(row, campos_deseados):
    """Genera una tabla HTML bonita para el popup."""
    html = """
    <style>
        table {width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 11px;}
        td, th {border: 1px solid #ddd; padding: 4px;}
        tr:nth-child(even){background-color: #f2f2f2;}
        th {padding-top: 6px; padding-bottom: 6px; text-align: left; background-color: #2b8cbe; color: white;}
    </style>
    <table>
        <tr><th>Atributo</th><th>Valor</th></tr>
    """
    
    for label, posibles_nombres in campos_deseados.items():
        col_real = buscar_columna(pd.DataFrame([row]), posibles_nombres)
        valor = row[col_real] if col_real else "N/A"
        # Limpieza de valores nulos o feos
        if pd.isna(valor) or str(valor).strip() == "": valor = "-"
        
        html += f"<tr><td><b>{label}</b></td><td>{valor}</td></tr>"
    
    html += "</table>"
    return html

# --- LÓGICA TURC & PROPHET ---
def interpolacion_segura_suave(points, values, grid_x, grid_y):
    try:
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
        mask = np.isnan(grid_z)
        if np.any(mask):
            grid_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
            grid_z[mask] = grid_nearest[mask]
        return grid_z
    except: return griddata(points, values, (grid_x, grid_y), method='linear')

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
    m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=months_ahead, freq='M')
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
    df_merged['recarga_est'] = [x[1] for x in central]
    df_merged['etr_est'] = [x[0] for x in central]
    low = df_merged['p_rate_low'].apply(calc_vec)
    df_merged['recarga_low'] = [x[1] for x in low]
    high = df_merged['p_rate_high'].apply(calc_vec)
    df_merged['recarga_high'] = [x[1] for x in high]
    df_merged['tipo'] = np.where(df_merged['ds'] <= last_date_real, 'Histórico', 'Proyección')
    return df_merged

# --- INTERFAZ ---
ids_dummy, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

st.sidebar.divider()
st.sidebar.header("🌱 Suelo e Infiltración")
col_s1, col_s2 = st.sidebar.columns(2)
pct_bosque = col_s1.number_input("% Bosque", 0, 100, 60)
pct_cultivo = col_s2.number_input("% Agrícola", 0, 100, 30)
pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
st.sidebar.caption(f"Urbano calculado: {pct_urbano}%")
ki_ponderado = ((pct_bosque * 0.50) + (pct_cultivo * 0.30) + (pct_urbano * 0.10)) / 100.0
st.sidebar.metric("Coef. Ki", f"{ki_ponderado:.2f}")

st.sidebar.divider()
horizonte_meses = st.sidebar.slider("Meses Pronóstico", 12, 60, 24)
ruido = st.sidebar.slider("Incertidumbre", 0.0, 2.0, 0.5)

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
            sel_local = st.multiselect("📍 Filtrar Estaciones:", df_est['nom_est'].unique(), default=df_est['nom_est'].unique())
            df_est_filtered = df_est[df_est['nom_est'].isin(sel_local)]
            
            if not df_est_filtered.empty:
                # Cargar Datos SQL
                ids_v = df_est_filtered['id_estacion'].unique()
                ids_s = ",".join([f"'{str(x)}'" for x in ids_v])
                q_avg = text(f"SELECT id_estacion_fk as id_estacion, AVG(precipitation)*12 as p_anual FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY 1")
                df_avg = pd.read_sql(q_avg, engine)
                q_serie = text(f"SELECT fecha_mes_año as fecha, AVG(precipitation) as p_mensual FROM precipitacion_mensual WHERE id_estacion_fk IN ({ids_s}) GROUP BY 1 ORDER BY 1")
                df_serie = pd.read_sql(q_serie, engine)
                
                # Unificar
                df_est_filtered['id_estacion'] = df_est_filtered['id_estacion'].astype(str)
                df_avg['id_estacion'] = df_avg['id_estacion'].astype(str)
                df_work = pd.merge(df_est_filtered, df_avg, on='id_estacion', how='inner')
                df_work['alt_est'] = df_work['alt_est'].fillna(altitud_ref)
                df_res_avg = calculate_turc_advanced(df_work, ki_ponderado)
                
                # KPIs
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Lluvia (mm/año)", f"{df_res_avg['p_anual'].mean():.0f}")
                k2.metric("ETR (mm/año)", f"{df_res_avg['etr_mm'].mean():.0f}")
                k3.metric("Recarga (mm/año)", f"{df_res_avg['recarga_mm'].mean():.0f}", delta="Oferta")
                k4.metric("Estaciones", len(df_res_avg))
                
                st.divider()
                
                # --- PESTAÑAS ---
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis Temporal", "🗺️ Mapa Recarga (Turc)", "💧 Hidrogeología & Bocatomas", "💾 Datos"])
                
                # TAB 1: GRÁFICAS (ETR RECUPERADA)
                with tab1:
                    if not df_serie.empty and PROPHET_AVAILABLE:
                        with st.spinner("Proyectando..."):
                            df_fc = run_prophet_forecast_hybrid(df_serie, horizonte_meses, altitud_ref, ki_ponderado, ruido)
                            if not df_fc.empty:
                                fig = go.Figure()
                                h = df_fc[df_fc['tipo']=='Histórico']
                                p = df_fc[df_fc['tipo']=='Proyección']
                                
                                # 1. Lluvia (Barras Fondo)
                                fig.add_trace(go.Bar(x=h['ds'], y=h['p_rate'], name='Lluvia', marker_color='lightblue', opacity=0.6))
                                
                                # 2. ETR (Línea Naranja - IMPORTANTE)
                                fig.add_trace(go.Scatter(x=df_fc['ds'], y=df_fc['etr_est'], name='ETR (Evapotranspiración)', 
                                                         line=dict(color='darkorange', width=2, dash='dot')))
                                
                                # 3. Recarga Histórica (Relleno Azul)
                                fig.add_trace(go.Scatter(x=h['ds'], y=h['recarga_est'], name='Recarga Histórica', 
                                                         line=dict(color='blue'), fill='tozeroy'))
                                
                                # 4. Recarga Futura (Línea Punteada)
                                fig.add_trace(go.Scatter(x=p['ds'], y=p['recarga_est'], name='Recarga Futura', 
                                                         line=dict(color='dodgerblue', dash='dash')))
                                
                                # 5. Rango Incertidumbre
                                fig.add_trace(go.Scatter(x=p['ds'], y=p['recarga_low'], line=dict(width=0), showlegend=False))
                                fig.add_trace(go.Scatter(x=p['ds'], y=p['recarga_high'], line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.1)', name='Incertidumbre'))
                                
                                fig.update_layout(height=450, title="Dinámica: Lluvia vs ETR vs Recarga", hovermode="x unified")
                                st.plotly_chart(fig, use_container_width=True)
                    else: st.warning("Datos insuficientes para proyección.")
                
                # TAB 2: MAPA TURC (Sin cambios)
                with tab2:
                    if len(df_res_avg) >= 3:
                        gx, gy = np.mgrid[minx:maxx:200j, miny:maxy:200j]
                        grid = interpolacion_segura_suave(df_res_avg[['lon','lat']].values, df_res_avg['recarga_mm'].values, gx, gy)
                        fig_m = go.Figure()
                        fig_m.add_trace(go.Contour(z=grid.T, x=np.linspace(minx, maxx, 200), y=np.linspace(miny, maxy, 200), colorscale="Blues", opacity=0.7))
                        fig_m.add_trace(go.Scatter(x=df_res_avg['lon'], y=df_res_avg['lat'], mode='markers', marker=dict(color='black'), text=df_res_avg['nom_est']))
                        add_context_layers_cartesian(fig_m, gdf_zona)
                        fig_m.update_layout(height=600, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False)
                        st.plotly_chart(fig_m, use_container_width=True)
                    else: st.warning("Se requieren al menos 3 estaciones.")
                
                # TAB 3: MAPAS GIS (POPUPS + HOVER)
                with tab3:
                    st.markdown("### 🗺️ Mapa Integrado de Aguas Subterráneas")
                    
                    with st.expander("🛠️ Corrector de Coordenadas", expanded=True):
                        st.caption("Si ves la tabla pero no el mapa, cambia esta opción.")
                        epsg_manual = st.selectbox(
                            "Seleccionar Sistema de Origen:",
                            options=["Detectar Automático", "EPSG:9377 (Origen Nacional)", "EPSG:3116 (Magna Bogotá)", "EPSG:3115 (Magna Oeste)"],
                            index=0
                        )
                    
                    gdf_zonas_raw, msg_zonas = cargar_capa_base('Zonas_PotHidrogeologico.shp')
                    gdf_bocas_raw, msg_bocas = cargar_capa_base('Bocatomas_Ant.shp')
                    
                    # Reproyección manual
                    def aplicar_reproyeccion(gdf_in, opcion):
                        if gdf_in is None: return None
                        gdf = gdf_in.copy()
                        try:
                            if opcion == "Detectar Automático":
                                if gdf.crs and gdf.crs.to_string() != "EPSG:4326": return gdf.to_crs("EPSG:4326")
                                if not gdf.crs and abs(gdf.geometry.iloc[0].centroid.x) > 180:
                                    gdf.set_crs("EPSG:3116", inplace=True)
                                    return gdf.to_crs("EPSG:4326")
                                return gdf
                            codigo = opcion.split(" ")[0]
                            gdf.set_crs(codigo, inplace=True, allow_override=True)
                            return gdf.to_crs("EPSG:4326")
                        except: return None

                    gdf_zonas = aplicar_reproyeccion(gdf_zonas_raw, epsg_manual)
                    gdf_bocas = aplicar_reproyeccion(gdf_bocas_raw, epsg_manual)
                    
                    if gdf_zonas is not None:
                        c_lat = gdf_zonas.geometry.centroid.y.mean()
                        c_lon = gdf_zonas.geometry.centroid.x.mean()
                        m = folium.Map(location=[c_lat, c_lon], zoom_start=9, tiles="CartoDB positron")
                        
                        # --- CAPA A: ZONAS (HOVER PERSONALIZADO) ---
                        fg_zonas = folium.FeatureGroup(name="🟫 Zonas Hidrogeológicas")
                        
                        # Buscamos columnas reales
                        col_pot = buscar_columna(gdf_zonas, ['Potencial_', 'POTENCIAL', 'Potencial']) or 'Desconocido'
                        col_uni = buscar_columna(gdf_zonas, ['Unidad_Geo', 'UNIDAD_GEO', 'Unidad']) or 'Desconocido'
                        col_sig = buscar_columna(gdf_zonas, ['SIGLA', 'Sigla']) or 'Desconocido'
                        
                        # Creamos tooltip solo si encontramos al menos una columna
                        fields_tip = [c for c in [col_pot, col_uni, col_sig] if c != 'Desconocido']
                        aliases_tip = [f"{c}:" for c in fields_tip]
                        
                        folium.GeoJson(
                            gdf_zonas,
                            style_function=lambda x: {'fillColor': '#2b8cbe', 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.4},
                            tooltip=folium.GeoJsonTooltip(fields=fields_tip, aliases=aliases_tip) if fields_tip else None
                        ).add_to(fg_zonas)
                        fg_zonas.add_to(m)
                        
                        # --- CAPA B: BOCATOMAS (POPUP COMPLETO) ---
                        if gdf_bocas is not None:
                            fg_bocas = folium.FeatureGroup(name="🚰 Bocatomas")
                            
                            # Diccionario de campos pedidos -> nombres posibles en SHP
                            campos_map = {
                                'Municipio': ['Municipio', 'MUNICIPIO', 'MPIO_CNMBR'],
                                'Acuífero': ['Nombre_Acu', 'NOMBRE_ACU'],
                                'Tipo': ['Tipo', 'TIPO'],
                                'Veredas': ['Veredas', 'VEREDAS'],
                                'Fuente Aba': ['Fuente_Aba', 'FUENTE_ABA'],
                                'Fuente Sub': ['Fuente_Sub', 'FUENTE_SUB'],
                                'Pozos': ['Pozos', 'POZOS'],
                                'Fuente Sup': ['Fuente_Sup', 'FUENTE_SUP'],
                                'Prot. Amb': ['Prot_Amb', 'PROT_AMB'],
                                'Prot. Conta': ['Prot_Conta', 'PROT_CONTA'],
                                'Entidad': ['Entidad_Ad', 'ENTIDAD_AD'],
                                'Tipo Entidad': ['Tipo_Ent', 'TIPO_ENT'],
                                'Suscriptor': ['Suscriptor', 'SUSCRIPTOR'],
                                'Año Const': ['Año_Const', 'AÃ±o_Const', 'ANO_CONST'],
                                'Vida Útil': ['Vida_Útil', 'Vida_Util', 'VIDA_UTIL'],
                                'Tipo Capt': ['Tipo_Capt', 'TIPO_CAPT'],
                                'Forma Capt': ['Forma_Capt', 'FORMA_CAPT']
                            }

                            for _, row in gdf_bocas.iterrows():
                                if row.geometry.geom_type == 'Point':
                                    # Generar HTML
                                    html = generar_popup_html(row, campos_map)
                                    iframe = folium.IFrame(html, width=320, height=300)
                                    popup = folium.Popup(iframe, max_width=320)
                                    
                                    folium.CircleMarker(
                                        location=[row.geometry.y, row.geometry.x],
                                        radius=5, color='red', fill=True, fill_color='darkred', fill_opacity=0.8,
                                        popup=popup, # <--- AQUI ESTA EL POPUP
                                        tooltip="Clic para info"
                                    ).add_to(fg_bocas)
                            fg_bocas.add_to(m)
                        
                        folium.LayerControl().add_to(m)
                        st_folium(m, width="100%", height=600)
                    else:
                        st.error("No se pudo cargar mapa. Intenta cambiar el selector de coordenadas.")
                
                with tab4:
                    st.dataframe(df_res_avg)

            else: st.warning("Sin estaciones en filtro.")
    except Exception as e: st.error(f"Error: {e}")