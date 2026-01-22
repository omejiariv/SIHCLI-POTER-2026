# pages/02_💧_Aguas_Subterraneas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

from modules import db_manager, hydrogeo_utils, forecasting, interpolation, config

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

# --- FUNCIONES AUXILIARES ---
def haversine_vectorized(lat1, lon1, lat_series, lon_series):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat_series)
    dphi = np.radians(lat_series - lat1)
    dlambda = np.radians(lon_series - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# --- CONEXIÓN BD ---
engine = db_manager.get_engine()
if not engine:
    st.error("⚠️ Error crítico: No hay conexión a la base de datos.")
    st.stop()

st.title("💧 Sistema de Recarga y Aguas Subterráneas")

# ==============================================================================
# 1. SIDEBAR: FILTROS
# ==============================================================================
with st.sidebar:
    st.header("📍 Configuración Espacial")
    
    # Cargar Datos Maestros
    q_master = "SELECT id_estacion, nom_est, municipio, latitud, longitud FROM estaciones ORDER BY nom_est"
    df_master = pd.read_sql(q_master, engine)
    
    try:
        df_cuencas = pd.read_sql("SELECT nombre_cuenca, municipios_influencia FROM cuencas", engine)
    except:
        df_cuencas = pd.DataFrame(columns=['nombre_cuenca', 'municipios_influencia'])

    # Filtros Cascada
    sel_munis = st.multiselect("1. Municipios:", sorted(df_master['municipio'].dropna().unique()), placeholder="Todos")
    
    # Universo filtrado inicial
    df_filtered = df_master.copy()
    if sel_munis:
        df_filtered = df_filtered[df_filtered['municipio'].isin(sel_munis)]

    # Filtro Cuencas (Textual)
    all_cuencas = sorted(df_cuencas['nombre_cuenca'].dropna().unique())
    sel_cuencas = st.multiselect("2. Cuencas:", all_cuencas, placeholder="Todas")
    
    if sel_cuencas:
        munis_cuenca = set()
        for c in sel_cuencas:
            rows = df_cuencas[df_cuencas['nombre_cuenca'] == c]
            for _, r in rows.iterrows():
                if r['municipios_influencia']:
                    for m in df_master['municipio'].unique():
                        if m in r['municipios_influencia']: munis_cuenca.add(m)
        if munis_cuenca:
            df_filtered = df_filtered[df_filtered['municipio'].isin(munis_cuenca)]

    # Selector Estación
    st.markdown(f"**Estaciones Disponibles:** `{len(df_filtered)}`")
    if df_filtered.empty:
        st.warning("Sin estaciones.")
        st.stop()

    est_seleccion = st.selectbox("3. Estación Análisis:", df_filtered['id_estacion'] + " - " + df_filtered['nom_est'])
    id_est = est_seleccion.split(" - ")[0]
    
    est_central = df_filtered[df_filtered['id_estacion'] == id_est].iloc[0]
    lat_central, lon_central = est_central['latitud'], est_central['longitud']

    # Radio Buffer
    st.markdown("---")
    usar_buffer = st.toggle("Aplicar Radio (km)", value=True)
    radio_km = st.slider("Distancia", 5, 200, 40) if usar_buffer else 0

    # Fechas
    st.markdown("---")
    fechas = pd.read_sql(f"SELECT MIN(fecha_mes_año), MAX(fecha_mes_año) FROM precipitacion_mensual WHERE id_estacion_fk='{id_est}'", engine)
    if fechas.iloc[0,0]:
        start_dt, end_dt = fechas.iloc[0,0], fechas.iloc[0,1]
        date_range = st.slider("Periodo", min_value=start_dt.date(), max_value=end_dt.date(), value=(start_dt.date(), end_dt.date()))
    else:
        st.error("Sin datos de lluvia.")
        st.stop()

# ==============================================================================
# 2. PROCESAMIENTO
# ==============================================================================

# A. Datos Puntuales
q_geo = f"""
SELECT e.latitud, e.longitud, e.elevacion, s.infiltracion_ki, s.unidad_suelo, zh.potencial 
FROM estaciones e
LEFT JOIN suelos s ON ST_Intersects(e.geom, s.geom)
LEFT JOIN zonas_hidrogeologicas zh ON ST_Intersects(e.geom, zh.geom)
WHERE e.id_estacion = '{id_est}'
"""
geo_data = pd.read_sql(q_geo, engine)

q_lluvia = f"""
SELECT fecha_mes_año as {config.Config.DATE_COL}, precipitation as {config.Config.PRECIPITATION_COL}
FROM precipitacion_mensual 
WHERE id_estacion_fk = '{id_est}' 
ORDER BY fecha_mes_año
"""
df_lluvia = pd.read_sql(q_lluvia, engine)

df_vis = pd.DataFrame()
ki, potencial = 0.15, "N/A"
stats_qa = (0, 0, 0)

if not geo_data.empty and not df_lluvia.empty:
    lat, alt = geo_data.iloc[0]['latitud'], geo_data.iloc[0]['elevacion']
    ki = geo_data.iloc[0]['infiltracion_ki'] if pd.notnull(geo_data.iloc[0]['infiltracion_ki']) else 0.15
    potencial = geo_data.iloc[0]['potencial']
    
    df_balance = hydrogeo_utils.calcular_serie_recarga(df_lluvia, lat, alt, ki)
    mask_date = (df_balance[config.Config.DATE_COL].dt.date >= date_range[0]) & (df_balance[config.Config.DATE_COL].dt.date <= date_range[1])
    df_vis = df_balance[mask_date].copy()
    stats_qa = hydrogeo_utils.calcular_calidad_datos(df_vis, date_range[0], date_range[1])

# B. Datos Espaciales (Mapa)
df_map = hydrogeo_utils.obtener_datos_estaciones_recarga(engine)

if usar_buffer:
    df_map['distancia_km'] = haversine_vectorized(lat_central, lon_central, df_map['latitud'], df_map['longitud'])
    df_map = df_map[df_map['distancia_km'] <= radio_km]
else:
    ids_validos = df_filtered['id_estacion'].unique()
    df_map = df_map[df_map['id_estacion'].isin(ids_validos)]

# ==============================================================================
# 3. INTERFAZ
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis", "🔮 Pronóstico (IA)", "🗺️ Mapa Recarga", "📥 Descargas"])

# TAB 1: Balance
with tab1:
    if not df_vis.empty:
        st.info(f"📊 Datos encontrados: **{stats_qa[0]}** meses ({stats_qa[2]:.1f}% completitud).")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recarga Media", f"{(df_vis['recarga_mm'].mean()*12):,.0f} mm/año")
        c2.metric("Infiltración", f"{ki*100:.1f}%")
        c3.metric("Potencial", potencial or "N/A")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis[config.Config.PRECIPITATION_COL], mode='lines', name='Lluvia', line=dict(color='blue', width=1), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['recarga_mm'], mode='lines', name='Recarga', line=dict(color='#2ca02c', width=2)))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['escorrentia_sup_mm'], mode='lines', name='Escorrentía', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['etr_mm'], mode='lines', name='ETR', line=dict(color='red', dash='dot')))
        fig.update_layout(title="Balance Hídrico", hovermode="x unified", height=450)
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: Pronóstico (CORREGIDO)
with tab2:
    st.subheader("Pronóstico (Prophet)")
    h = st.slider("Meses:", 12, 60, 24)
    if st.button("Ejecutar"):
        with st.spinner("Calculando..."):
            try:
                # 1. ELIMINAR DUPLICADOS (Crucial para Prophet)
                # Re-muestrea a Inicio de Mes (MS) y promedia si hay >1 dato
                df_clean = df_vis.set_index(config.Config.DATE_COL).resample('MS').mean().reset_index()
                
                # 2. Formatear para módulo forecasting
                df_input = df_clean.rename(columns={config.Config.DATE_COL: 'ds', 'recarga_mm': config.Config.PRECIPITATION_COL})
                df_input = df_input.dropna(subset=[config.Config.PRECIPITATION_COL])

                if len(df_input) < 24:
                    st.error("Datos insuficientes (<24 meses limpios).")
                else:
                    _, forecast, metrics = forecasting.generate_prophet_forecast(df_input, h, 12)
                    
                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(x=df_input['ds'], y=df_input[config.Config.PRECIPITATION_COL], name="Histórico", line=dict(color='gray')))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Pronóstico", line=dict(color='blue')))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.2)', name="IC"))
                    st.plotly_chart(fig_fc, use_container_width=True)
                    st.success(f"MAE: {metrics['MAE']:.2f} mm")
            except Exception as e:
                st.error(f"Error Prophet: {e}")

# TAB 3: Mapa Rico
with tab3:
    st.subheader("Mapa Recarga Anual")
    c1, c2 = st.columns([1, 4])
    with c1:
        st.metric("Estaciones", len(df_map))
        metodo = st.radio("Método", ["IDW", "Kriging Ordinario"])
        res = st.select_slider("Resolución", [50, 100, 150], value=100)
    
    with c2:
        if len(df_map) < 4:
            st.warning("⚠️ Mínimo 4 estaciones.")
        else:
            m = 0.05
            b = [df_map.longitud.min()-m, df_map.latitud.min()-m, df_map.longitud.max()+m, df_map.latitud.max()+m]
            gx = np.linspace(b[0], b[2], res)
            gy = np.linspace(b[1], b[3], res)
            
            with st.spinner("Interpolando..."):
                val = 'recarga_anual'
                if metodo == "IDW":
                    z = interpolation.interpolate_idw(df_map.longitud.values, df_map.latitud.values, df_map[val].values, gx, gy)
                else:
                    gdf_p = gpd.GeoDataFrame(df_map, geometry=gpd.points_from_xy(df_map.longitud, df_map.latitud))
                    z, _ = interpolation.create_kriging_by_basin(_gdf_points=gdf_p, grid_lon=gx, grid_lat=gy, value_col=val)
            
            vmin, vmax = np.nanpercentile(z, 2), np.nanpercentile(z, 98)
            
            fig_map = go.Figure(data=go.Contour(
                z=z, x=gx, y=gy, colorscale="Viridis", zmin=vmin, zmax=vmax,
                colorbar=dict(title="Recarga (mm/año)"),
                contours=dict(start=vmin, end=vmax, size=(vmax-vmin)/15, showlines=False),
                hoverinfo='skip' # El contour es el fondo, el hover lo dan los puntos
            ))
            
            # POPUPS RICOS (Hover Data)
            # Creamos el texto formateado HTML
            df_map['hover_text'] = df_map.apply(lambda row: f"""
            <b>{row['nom_est']}</b><br>
            📍 {row.get('municipio', 'N/A')}<br>
            ⛰️ Elev: {row['elevacion']:.0f} m<br>
            💧 Ppt Anual: {row['ppt_anual']:.0f} mm<br>
            ☀️ ETR Anual: {row['etr_anual']:.0f} mm<br>
            ⬇️ <b>Recarga: {row['recarga_anual']:.0f} mm/año</b>
            """, axis=1)

            fig_map.add_trace(go.Scatter(
                x=df_map.longitud, y=df_map.latitud, mode='markers',
                marker=dict(color='black', size=5, opacity=0.6, line=dict(color='white', width=1)),
                text=df_map['hover_text'],
                hoverinfo='text',
                name='Estaciones'
            ))
            fig_map.add_trace(go.Scatter(x=[lon_central], y=[lat_central], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Tu Ubicación', hoverinfo='skip'))
            
            fig_map.update_layout(height=650, xaxis=dict(scaleanchor="y", scaleratio=1), margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_map, use_container_width=True)

# TAB 4: Descargas
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        if not df_vis.empty: st.download_button("Descargar CSV", df_vis.to_csv(index=False), f"recarga_{id_est}.csv")
    with c2:
        if 'z' in locals() and z is not None:
            try:
                tif = hydrogeo_utils.generar_geotiff_bytes(z, b)
                st.download_button("Descargar Raster", tif, "recarga.tif")
                geo = hydrogeo_utils.generar_geojson_bytes(df_map)
                st.download_button("Descargar GeoJSON", geo, "estaciones.geojson")
            except: pass