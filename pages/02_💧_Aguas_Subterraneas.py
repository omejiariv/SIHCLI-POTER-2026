# pages/02_💧_Aguas_Subterraneas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

from modules import db_manager, hydrogeo_utils, forecasting, interpolation, config

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

# --- FUNCIONES ---
def haversine_vectorized(lat1, lon1, lat_series, lon_series):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat_series)
    dphi = np.radians(lat_series - lat1)
    dlambda = np.radians(lon_series - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# --- INIT ---
engine = db_manager.get_engine()
if not engine:
    st.error("Sin conexión BD")
    st.stop()

st.title("💧 Sistema de Recarga y Aguas Subterráneas")

# ==============================================================================
# 1. SIDEBAR (FILTROS CORREGIDOS)
# ==============================================================================
with st.sidebar:
    st.header("📍 Configuración")
    
    # Cargar Maestros
    df_estaciones = pd.read_sql("SELECT id_estacion, nom_est, municipio, latitud, longitud FROM estaciones ORDER BY nom_est", engine)
    
    try:
        df_cuencas = pd.read_sql("SELECT DISTINCT nombre_cuenca, municipios_influencia FROM cuencas ORDER BY nombre_cuenca", engine)
        lista_cuencas = ["Todas"] + df_cuencas['nombre_cuenca'].tolist()
    except:
        lista_cuencas = ["Todas"]
        df_cuencas = pd.DataFrame()

    # --- A. Filtro Cuenca (Flexible) ---
    sel_cuenca = st.selectbox("1. Cuenca:", lista_cuencas)
    
    # Determinar municipios válidos
    munis_validos = sorted(df_estaciones['municipio'].dropna().unique())
    
    if sel_cuenca != "Todas" and not df_cuencas.empty:
        # Buscar la fila de la cuenca
        row_c = df_cuencas[df_cuencas['nombre_cuenca'] == sel_cuenca]
        if not row_c.empty:
            txt_infl = str(row_c.iloc[0]['municipios_influencia']).lower() # Convertir a minúsculas para buscar
            # Filtramos municipios que estén CONTENIDOS en el texto de influencia
            munis_validos = [m for m in munis_validos if m.lower() in txt_infl]
            
            if not munis_validos:
                st.warning(f"No se detectaron municipios vinculados textualmente a '{sel_cuenca}'. Mostrando todos.")
                munis_validos = sorted(df_estaciones['municipio'].dropna().unique())

    # --- B. Filtro Municipio ---
    sel_muni = st.selectbox("2. Municipio:", ["Todos"] + munis_validos)
    
    # --- C. Filtrar Estaciones ---
    df_final = df_estaciones.copy()
    
    # Aplicar Cuenca (vía municipios)
    if sel_cuenca != "Todas":
        df_final = df_final[df_final['municipio'].isin(munis_validos)]
    
    # Aplicar Municipio
    if sel_muni != "Todos":
        df_final = df_final[df_final['municipio'] == sel_muni]
        
    st.caption(f"Estaciones encontradas: {len(df_final)}")
    
    if df_final.empty:
        st.error("No hay estaciones con estos filtros.")
        st.stop()

    # --- D. Estación Central ---
    est_seleccion = st.selectbox("3. Estación:", df_final['id_estacion'] + " - " + df_final['nom_est'])
    id_est = est_seleccion.split(" - ")[0]
    est_central = df_final[df_final['id_estacion'] == id_est].iloc[0]

    # --- E. Buffer (Solo Visual) ---
    st.markdown("---")
    usar_buffer = st.toggle("Buffer (Mapa)", value=True)
    radio_km = st.slider("Radio (km)", 5, 100, 20) if usar_buffer else 0

    # --- F. Fechas ---
    st.markdown("---")
    fechas = pd.read_sql(f"SELECT MIN(fecha_mes_año), MAX(fecha_mes_año) FROM precipitacion_mensual WHERE id_estacion_fk='{id_est}'", engine)
    if fechas.iloc[0,0]:
        start_dt, end_dt = fechas.iloc[0,0], fechas.iloc[0,1]
        date_range = st.slider("Rango", min_value=start_dt.date(), max_value=end_dt.date(), value=(start_dt.date(), end_dt.date()))
    else:
        st.error("Estación vacía.")
        st.stop()

# ==============================================================================
# 2. CÁLCULO PUNTUAL (Serie de Tiempo)
# ==============================================================================
# Consulta Nearest Neighbor para el punto central (arregla el 15% fijo)
q_geo = f"""
SELECT e.latitud, e.longitud, e.elevacion, s.infiltracion_ki, s.unidad_suelo, zh.potencial 
FROM estaciones e
LEFT JOIN LATERAL (
    SELECT infiltracion_ki, unidad_suelo FROM suelos s ORDER BY e.geom <-> s.geom LIMIT 1
) s ON true
LEFT JOIN zonas_hidrogeologicas zh ON ST_Intersects(e.geom, zh.geom)
WHERE e.id_estacion = '{id_est}'
"""
geo_data = pd.read_sql(q_geo, engine)

q_lluvia = f"SELECT fecha_mes_año as {config.Config.DATE_COL}, precipitation as {config.Config.PRECIPITATION_COL} FROM precipitacion_mensual WHERE id_estacion_fk = '{id_est}' ORDER BY fecha_mes_año"
df_lluvia = pd.read_sql(q_lluvia, engine)

df_vis = pd.DataFrame()
ki, potencial = 0.15, "N/A"
stats_qa = (0,0,0)

if not df_lluvia.empty:
    lat, alt = est_central.latitud, geo_data.iloc[0]['elevacion']
    # Recuperar Ki del SQL
    ki_val = geo_data.iloc[0]['infiltracion_ki']
    ki = ki_val if pd.notnull(ki_val) else 0.15
    potencial = geo_data.iloc[0]['potencial']
    
    df_balance = hydrogeo_utils.calcular_serie_recarga(df_lluvia, lat, alt, ki)
    mask_date = (df_balance[config.Config.DATE_COL].dt.date >= date_range[0]) & (df_balance[config.Config.DATE_COL].dt.date <= date_range[1])
    df_vis = df_balance[mask_date].copy()
    stats_qa = hydrogeo_utils.calcular_calidad_datos(df_vis, date_range[0], date_range[1])

# ==============================================================================
# 3. CÁLCULO ESPACIAL (Mapa) - INDEPENDIENTE DEL FILTRO SIDEBAR
# ==============================================================================
# Traemos TODAS las estaciones siempre para poder calcular el buffer real
df_map_data_full = hydrogeo_utils.obtener_datos_estaciones_recarga(engine)

if usar_buffer:
    # Calculamos distancia desde la estación seleccionada a TODAS las demás
    df_map_data_full['distancia_km'] = haversine_vectorized(
        est_central.latitud, est_central.longitud, 
        df_map_data_full['latitud'], df_map_data_full['longitud']
    )
    # Filtramos por radio
    df_map_view = df_map_data_full[df_map_data_full['distancia_km'] <= radio_km].copy()
else:
    # Si no usa buffer, mostramos solo las estaciones que pasaron el filtro del sidebar (Cuenca/Muni)
    ids_permitidos = df_final['id_estacion'].unique()
    df_map_view = df_map_data_full[df_map_data_full['id_estacion'].isin(ids_permitidos)].copy()

# ==============================================================================
# 4. FRONTEND
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Balance", "🔮 Pronóstico", "🗺️ Mapa", "📥 Descargas"])

with tab1:
    if not df_vis.empty:
        st.info(f"Datos: {stats_qa[0]} meses ({stats_qa[2]:.1f}%). Ki real: {ki*100:.1f}%")
        c1, c2, c3 = st.columns(3)
        c1.metric("Recarga Anual", f"{(df_vis['recarga_mm'].mean()*12):,.0f} mm")
        c2.metric("Infiltración", f"{ki*100:.1f}%")
        c3.metric("Potencial", potencial)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis[config.Config.PRECIPITATION_COL], mode='lines', name='Lluvia', line=dict(color='blue', width=1), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['recarga_mm'], mode='lines', name='Recarga', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['escorrentia_sup_mm'], mode='lines', name='Escorrentía', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['etr_mm'], mode='lines', name='ETR', line=dict(color='red', dash='dot')))
        fig.update_layout(title="Balance Mensual", hovermode="x unified", height=450)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Pronóstico Prophet")
    h = st.slider("Horizonte:", 12, 60, 24)
    if st.button("Ejecutar"):
        with st.spinner("Calculando..."):
            try:
                # PREPARACIÓN CRÍTICA PARA PROPHET
                # 1. Asegurar índice mensual único
                df_proph = df_vis.set_index(config.Config.DATE_COL).resample('MS').mean().reset_index()
                # 2. Renombrar explícitamente a lo que espera Prophet
                df_proph = df_proph.rename(columns={config.Config.DATE_COL: 'ds', 'recarga_mm': 'y'})
                # 3. Eliminar nulos en la target
                df_proph = df_proph.dropna(subset=['y'])
                
                # Engañamos al módulo 'forecasting' pasándole 'y' renombrada a 'precipitation' si es lo que pide
                # Pero Prophet puro usa 'ds' y 'y'. Revisando tu módulo forecasting, usa 'ds' y 'y' internamente.
                # Si tu funcion forecasting.generate_prophet_forecast espera un DF con columna config.PRECIPITATION_COL:
                df_to_mod = df_proph.rename(columns={'y': config.Config.PRECIPITATION_COL, 'ds': config.Config.DATE_COL})
                
                if len(df_to_mod) < 24:
                    st.error("Datos insuficientes (<24 meses).")
                else:
                    _, forecast, metrics = forecasting.generate_prophet_forecast(df_to_mod, h, 12)
                    
                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(x=df_to_mod[config.Config.DATE_COL], y=df_to_mod[config.Config.PRECIPITATION_COL], name="Histórico", line=dict(color='gray')))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Pronóstico", line=dict(color='blue')))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', width=0, showlegend=False))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', width=0, fill='tonexty', fillcolor='rgba(0,0,255,0.2)', name="IC"))
                    st.plotly_chart(fig_fc, use_container_width=True)
                    st.success(f"MAE: {metrics['MAE']:.2f}")
            except Exception as e:
                st.error(f"Error Prophet: {e}")

with tab3:
    st.subheader("Mapa Recarga")
    c1, c2 = st.columns([1, 4])
    with c1:
        st.write(f"**Puntos:** {len(df_map_view)}")
        metodo = st.radio("Método", ["IDW", "Kriging Ordinario"])
        res = st.select_slider("Resolución", [50, 100], value=50)
    
    with c2:
        if len(df_map_view) < 4:
            st.warning("Se necesitan min 4 estaciones. Aumenta el radio.")
        else:
            m = 0.05
            b = [df_map_view.longitud.min()-m, df_map_view.latitud.min()-m, df_map_view.longitud.max()+m, df_map_view.latitud.max()+m]
            gx = np.linspace(b[0], b[2], res)
            gy = np.linspace(b[1], b[3], res)
            
            with st.spinner("Interpolando..."):
                val = 'recarga_anual'
                if metodo == "IDW":
                    z = interpolation.interpolate_idw(df_map_view.longitud.values, df_map_view.latitud.values, df_map_view[val].values, gx, gy)
                else:
                    gdf_p = gpd.GeoDataFrame(df_map_view, geometry=gpd.points_from_xy(df_map_view.longitud, df_map_view.latitud))
                    z, _ = interpolation.create_kriging_by_basin(_gdf_points=gdf_p, grid_lon=gx, grid_lat=gy, value_col=val)
            
            vmin, vmax = np.nanpercentile(z, 2), np.nanpercentile(z, 98)
            fig_map = go.Figure(data=go.Contour(z=z, x=gx, y=gy, colorscale="Viridis", zmin=vmin, zmax=vmax, colorbar=dict(title="mm/año"), hoverinfo='skip'))
            
            df_map_view['hover'] = df_map_view.apply(lambda r: f"<b>{r['nom_est']}</b><br>Recarga: {r['recarga_anual']:.0f}<br>Ki: {r['ki_final']:.2f}", axis=1)
            fig_map.add_trace(go.Scatter(x=df_map_view.longitud, y=df_map_view.latitud, mode='markers', marker=dict(color='black', size=5), text=df_map_view['hover'], hoverinfo='text', name='Estaciones'))
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
                geo = hydrogeo_utils.generar_geojson_bytes(df_map_view)
                st.download_button("Descargar GeoJSON", geo, "estaciones.geojson")
            except: pass