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

engine = db_manager.get_engine()
if not engine:
    st.error("⚠️ Sin conexión BD.")
    st.stop()

st.title("💧 Sistema de Recarga y Aguas Subterráneas")

# ==============================================================================
# 1. SIDEBAR: FILTROS SINCRONIZADOS
# ==============================================================================
with st.sidebar:
    st.header("📍 Configuración Espacial")
    
    # Cargar Maestros
    df_estaciones = pd.read_sql("SELECT id_estacion, nom_est, municipio, latitud, longitud FROM estaciones", engine)
    
    try:
        df_cuencas = pd.read_sql("SELECT DISTINCT nombre_cuenca, municipios_influencia FROM cuencas ORDER BY nombre_cuenca", engine)
    except:
        df_cuencas = pd.DataFrame()

    # --- A. Selección de CUENCA (Nivel Superior) ---
    lista_cuencas = ["Todas"] + df_cuencas['nombre_cuenca'].tolist() if not df_cuencas.empty else ["Todas"]
    sel_cuenca = st.selectbox("1. Cuenca Hidrográfica:", lista_cuencas)
    
    # Filtrar municipios basados en la cuenca
    munis_disponibles = sorted(df_estaciones['municipio'].dropna().unique().tolist())
    
    if sel_cuenca != "Todas":
        # Obtener texto de municipios de la cuenca
        row_c = df_cuencas[df_cuencas['nombre_cuenca'] == sel_cuenca]
        if not row_c.empty and row_c.iloc[0]['municipios_influencia']:
            txt_infl = row_c.iloc[0]['municipios_influencia']
            # Filtramos la lista de municipios disponibles para que solo muestre los de la cuenca
            munis_disponibles = [m for m in munis_disponibles if m in txt_infl]
    
    # --- B. Selección de MUNICIPIO (Filtrado por Cuenca) ---
    sel_muni = st.selectbox("2. Municipio:", ["Todos"] + munis_disponibles)
    
    # --- C. Filtrado del DataFrame de Estaciones ---
    df_filtered = df_estaciones.copy()
    
    # Aplicar Filtro Cuenca (Indirecto por municipio)
    if sel_cuenca != "Todas":
        df_filtered = df_filtered[df_filtered['municipio'].isin(munis_disponibles)]
        
    # Aplicar Filtro Municipio Específico
    if sel_muni != "Todos":
        df_filtered = df_filtered[df_filtered['municipio'] == sel_muni]

    # --- D. Selección ESTACIÓN FINAL ---
    st.markdown(f"**Estaciones encontradas:** `{len(df_filtered)}`")
    
    if df_filtered.empty:
        st.warning("No hay estaciones en esta zona.")
        st.stop()
        
    est_seleccion = st.selectbox(
        "3. Estación de Análisis:", 
        df_filtered['id_estacion'] + " - " + df_filtered['nom_est']
    )
    id_est = est_seleccion.split(" - ")[0]
    
    est_central = df_filtered[df_filtered['id_estacion'] == id_est].iloc[0]
    
    # --- E. Buffer ---
    st.markdown("---")
    usar_buffer = st.toggle("Radio de Búsqueda (Mapa)", value=True)
    radio_km = st.slider("Km:", 5, 100, 20) if usar_buffer else 0

    # --- F. Periodo ---
    st.markdown("---")
    fechas = pd.read_sql(f"SELECT MIN(fecha_mes_año), MAX(fecha_mes_año) FROM precipitacion_mensual WHERE id_estacion_fk='{id_est}'", engine)
    if fechas.iloc[0,0]:
        start_dt, end_dt = fechas.iloc[0,0], fechas.iloc[0,1]
        date_range = st.slider("Periodo", min_value=start_dt.date(), max_value=end_dt.date(), value=(start_dt.date(), end_dt.date()))
    else:
        st.error("Sin datos.")
        st.stop()

# ==============================================================================
# 2. CÁLCULOS
# ==============================================================================

# A. Datos Puntuales
q_geo = f"""
SELECT e.latitud, e.longitud, e.elevacion, s.infiltracion_ki, s.unidad_suelo, zh.potencial 
FROM estaciones e
LEFT JOIN suelos s ON ST_Intersects(e.geom, ST_Transform(s.geom, 4326)) 
LEFT JOIN zonas_hidrogeologicas zh ON ST_Intersects(e.geom, zh.geom)
WHERE e.id_estacion = '{id_est}'
"""
# Nota: Agregamos ST_Transform en la query puntual también por si acaso
try:
    geo_data = pd.read_sql(q_geo, engine)
except:
    # Fallback si falla SQL espacial complejo
    geo_data = pd.DataFrame({'latitud': [est_central.latitud], 'elevacion': [0], 'infiltracion_ki': [None], 'potencial': ['N/A']})

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

if not df_lluvia.empty:
    lat = est_central.latitud
    alt = geo_data.iloc[0]['elevacion'] if not geo_data.empty else 0
    ki = geo_data.iloc[0]['infiltracion_ki'] if not geo_data.empty and pd.notnull(geo_data.iloc[0]['infiltracion_ki']) else 0.15
    potencial = geo_data.iloc[0]['potencial'] if not geo_data.empty else "N/A"
    
    df_balance = hydrogeo_utils.calcular_serie_recarga(df_lluvia, lat, alt, ki)
    mask_date = (df_balance[config.Config.DATE_COL].dt.date >= date_range[0]) & (df_balance[config.Config.DATE_COL].dt.date <= date_range[1])
    df_vis = df_balance[mask_date].copy()
    stats_qa = hydrogeo_utils.calcular_calidad_datos(df_vis, date_range[0], date_range[1])

# B. Mapa
df_map = hydrogeo_utils.obtener_datos_estaciones_recarga(engine)

if usar_buffer:
    df_map['distancia_km'] = haversine_vectorized(est_central.latitud, est_central.longitud, df_map['latitud'], df_map['longitud'])
    df_map = df_map[df_map['distancia_km'] <= radio_km]
else:
    ids_validos = df_filtered['id_estacion'].unique()
    df_map = df_map[df_map['id_estacion'].isin(ids_validos)]

# ==============================================================================
# 3. DASHBOARD
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis", "🔮 Pronóstico", "🗺️ Mapa Recarga", "📥 Datos"])

with tab1:
    if not df_vis.empty:
        st.info(f"📅 Datos: **{stats_qa[0]}** meses ({stats_qa[2]:.1f}% completitud).")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recarga Media", f"{(df_vis['recarga_mm'].mean()*12):,.0f} mm/año")
        
        # Etiqueta visual si es estimado
        lbl_ki = f"{ki*100:.1f}%" + (" (Estimado)" if ki == 0.15 else " (Suelo)")
        c2.metric("Infiltración (Ki)", lbl_ki)
        c3.metric("Potencial", potencial)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis[config.Config.PRECIPITATION_COL], mode='lines', name='Lluvia', line=dict(color='blue', width=1), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['recarga_mm'], mode='lines', name='Recarga', line=dict(color='#2ca02c', width=2)))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['escorrentia_sup_mm'], mode='lines', name='Escorrentía', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['etr_mm'], mode='lines', name='ETR', line=dict(color='red', dash='dot')))
        fig.update_layout(title="Balance Hídrico", hovermode="x unified", height=450)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Pronóstico (Prophet)")
    h = st.slider("Meses:", 12, 60, 24)
    if st.button("Ejecutar"):
        with st.spinner("Procesando..."):
            try:
                # La limpieza ya se hizo en hydrogeo_utils, pero aseguramos formato
                df_input = df_vis.rename(columns={config.Config.DATE_COL: 'ds', 'recarga_mm': config.Config.PRECIPITATION_COL})
                if len(df_input) < 24:
                    st.error("Datos insuficientes (<24 meses).")
                else:
                    _, forecast, metrics = forecasting.generate_prophet_forecast(df_input, h, 12)
                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(x=df_input['ds'], y=df_input[config.Config.PRECIPITATION_COL], name="Histórico", line=dict(color='gray')))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Pronóstico", line=dict(color='blue')))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', width=0, showlegend=False))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', width=0, fill='tonexty', fillcolor='rgba(0,0,255,0.2)', name="IC"))
                    st.plotly_chart(fig_fc, use_container_width=True)
                    st.success(f"MAE: {metrics['MAE']:.2f} mm")
            except Exception as e:
                st.error(f"Error: {e}")

with tab3:
    c1, c2 = st.columns([1, 4])
    with c1:
        st.metric("Puntos", len(df_map))
        metodo = st.radio("Método", ["IDW", "Kriging Ordinario"])
        res = st.select_slider("Resolución", [50, 100], value=50)
    
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
            fig_map = go.Figure(data=go.Contour(z=z, x=gx, y=gy, colorscale="Viridis", zmin=vmin, zmax=vmax, colorbar=dict(title="mm/año"), hoverinfo='skip'))
            
            # Popups Ricos
            df_map['hover'] = df_map.apply(lambda r: f"<b>{r['nom_est']}</b><br>Ki: {r['ki_final']*100:.0f}%<br>Recarga: {r['recarga_anual']:.0f}", axis=1)
            
            fig_map.add_trace(go.Scatter(x=df_map.longitud, y=df_map.latitud, mode='markers', marker=dict(color='black', size=5), text=df_map['hover'], hoverinfo='text', name='Estaciones'))
            fig_map.add_trace(go.Scatter(x=[est_central.longitud], y=[est_central.latitud], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Centro'))
            fig_map.update_layout(height=600, xaxis=dict(scaleanchor="y", scaleratio=1), margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_map, use_container_width=True)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        if not df_vis.empty: st.download_button("Descargar CSV", df_vis.to_csv(index=False), f"recarga_{id_est}.csv")
    with c2:
        if 'z' in locals() and z is not None:
            try:
                tif = hydrogeo_utils.generar_geotiff_bytes(z, b)
                st.download_button("Descargar Raster", tif, "recarga.tif")
            except: pass