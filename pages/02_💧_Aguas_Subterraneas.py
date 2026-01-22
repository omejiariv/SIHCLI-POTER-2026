# pages/02_💧_Aguas_Subterraneas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

# Importación de módulos propios
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
# 1. SIDEBAR: FILTROS EN CASCADA (LOGICA CORREGIDA)
# ==============================================================================
with st.sidebar:
    st.header("📍 Configuración Espacial")
    
    # 1. Cargar Datos Maestros (Cacheado sería ideal, pero directo es más seguro ahora)
    # Traemos todo el "Universo" de estaciones con sus ubicaciones para filtrar en memoria (más rápido y flexible)
    q_master = """
    SELECT id_estacion, nom_est, municipio, latitud, longitud 
    FROM estaciones 
    ORDER BY nom_est
    """
    df_master = pd.read_sql(q_master, engine)
    
    # Cargar cuencas para cruce textual
    try:
        df_cuencas = pd.read_sql("SELECT nombre_cuenca, municipios_influencia FROM cuencas", engine)
    except:
        df_cuencas = pd.DataFrame(columns=['nombre_cuenca', 'municipios_influencia'])

    # --- FILTRO 1: MUNICIPIOS ---
    lista_munis_total = sorted(df_master['municipio'].dropna().unique().tolist())
    sel_munis = st.multiselect("1. Filtrar Municipios:", lista_munis_total, placeholder="Todos")
    
    # Aplicar Filtro 1 al Universo
    if sel_munis:
        df_filtered_1 = df_master[df_master['municipio'].isin(sel_munis)]
    else:
        df_filtered_1 = df_master

    # --- FILTRO 2: CUENCAS (Dinámico basado en Filtro 1) ---
    # Identificar qué cuencas tocan los municipios seleccionados (lógica inversa textual)
    cuencas_disponibles = []
    if sel_munis:
        for idx, row in df_cuencas.iterrows():
            if row['municipios_influencia']:
                # Ver si algún municipio seleccionado está en el texto de influencia
                if any(m in row['municipios_influencia'] for m in sel_munis):
                    cuencas_disponibles.append(row['nombre_cuenca'])
    else:
        cuencas_disponibles = sorted(df_cuencas['nombre_cuenca'].unique().tolist())
    
    # Eliminar duplicados y ordenar
    cuencas_disponibles = sorted(list(set(cuencas_disponibles)))
    
    sel_cuencas = st.multiselect("2. Filtrar Cuencas (Opcional):", cuencas_disponibles, placeholder="Todas")
    
    # Aplicar Filtro 2
    if sel_cuencas:
        # Recuperar municipios de estas cuencas
        munis_de_cuenca = set()
        for c in sel_cuencas:
            row = df_cuencas[df_cuencas['nombre_cuenca'] == c]
            if not row.empty and row.iloc[0]['municipios_influencia']:
                # Agregar todos los municipios que mencione esta cuenca
                # Esto filtra las estaciones que estén en esos municipios
                for m_candidate in lista_munis_total:
                    if m_candidate in row.iloc[0]['municipios_influencia']:
                        munis_de_cuenca.add(m_candidate)
        
        # Intersección: Estaciones que ya pasaron Filtro 1 Y están en municipios de la Cuenca
        if munis_de_cuenca:
            df_filtered_1 = df_filtered_1[df_filtered_1['municipio'].isin(munis_de_cuenca)]

    # --- SELECTOR FINAL ---
    # Mostramos cuántas estaciones quedan
    count_avail = len(df_filtered_1)
    st.markdown(f"**Estaciones Disponibles:** `{count_avail}`")
    
    if df_filtered_1.empty:
        st.warning("No hay estaciones con esta combinación.")
        st.stop()

    est_seleccion = st.selectbox(
        "3. Estación de Análisis:", 
        df_filtered_1['id_estacion'] + " - " + df_filtered_1['nom_est']
    )
    id_est = est_seleccion.split(" - ")[0]
    
    # Datos Centrales
    est_central = df_filtered_1[df_filtered_1['id_estacion'] == id_est].iloc[0]
    lat_central, lon_central = est_central['latitud'], est_central['longitud']

    # --- RADIO ---
    st.markdown("---")
    usar_buffer = st.toggle("Aplicar Radio de Búsqueda (km)", value=True)
    radio_km = 20
    if usar_buffer:
        radio_km = st.slider("Radio (km)", 5, 200, 40)
        st.caption(f"Mapa mostrará vecinos a {radio_km}km")

    # --- FECHAS ---
    st.markdown("---")
    fechas = pd.read_sql(f"SELECT MIN(fecha_mes_año), MAX(fecha_mes_año) FROM precipitacion_mensual WHERE id_estacion_fk='{id_est}'", engine)
    if fechas.iloc[0,0]:
        start_dt, end_dt = fechas.iloc[0,0], fechas.iloc[0,1]
        date_range = st.slider("Periodo Análisis", min_value=start_dt.date(), max_value=end_dt.date(), value=(start_dt.date(), end_dt.date()))
    else:
        st.error("Estación sin datos de lluvia.")
        st.stop()

# ==============================================================================
# 2. PROCESAMIENTO (CÁLCULOS)
# ==============================================================================

# A. Metadatos Puntuales
q_geo = f"""
SELECT e.latitud, e.longitud, e.elevacion, s.infiltracion_ki, s.unidad_suelo, zh.potencial 
FROM estaciones e
LEFT JOIN suelos s ON ST_Intersects(e.geom, s.geom)
LEFT JOIN zonas_hidrogeologicas zh ON ST_Intersects(e.geom, zh.geom)
WHERE e.id_estacion = '{id_est}'
"""
geo_data = pd.read_sql(q_geo, engine)

# B. Serie Lluvia
q_lluvia = f"""
SELECT fecha_mes_año as {config.Config.DATE_COL}, precipitation as {config.Config.PRECIPITATION_COL}
FROM precipitacion_mensual 
WHERE id_estacion_fk = '{id_est}' 
ORDER BY fecha_mes_año
"""
df_lluvia = pd.read_sql(q_lluvia, engine)

df_vis = pd.DataFrame()
ki = 0.15 
potencial = "N/A"
stats_qa = (0, 0, 0)

if not geo_data.empty and not df_lluvia.empty:
    lat, alt = geo_data.iloc[0]['latitud'], geo_data.iloc[0]['elevacion']
    ki_db = geo_data.iloc[0]['infiltracion_ki']
    potencial = geo_data.iloc[0]['potencial']
    ki = ki_db if pd.notnull(ki_db) else 0.15
    
    # Calcular Balance con Nueva Lógica
    df_balance = hydrogeo_utils.calcular_serie_recarga(df_lluvia, lat, alt, ki)
    
    # Filtro Fecha
    mask_date = (df_balance[config.Config.DATE_COL].dt.date >= date_range[0]) & (df_balance[config.Config.DATE_COL].dt.date <= date_range[1])
    df_vis = df_balance[mask_date].copy()
    
    # Métricas de Calidad
    stats_qa = hydrogeo_utils.calcular_calidad_datos(df_vis, date_range[0], date_range[1])

# C. Mapa Espacial
df_map_data = hydrogeo_utils.obtener_datos_estaciones_recarga(engine)

if usar_buffer:
    df_map_data['distancia_km'] = haversine_vectorized(lat_central, lon_central, df_map_data['latitud'], df_map_data['longitud'])
    df_map_data = df_map_data[df_map_data['distancia_km'] <= radio_km]
else:
    # Usar filtro del sidebar
    ids_validos = df_filtered_1['id_estacion'].unique()
    df_map_data = df_map_data[df_map_data['id_estacion'].isin(ids_validos)]

# ==============================================================================
# 3. VISUALIZACIÓN
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis Temporal", "🔮 Pronóstico (IA)", "🗺️ Mapa de Recarga", "📥 Descargas"])

# TAB 1
with tab1:
    if not df_vis.empty:
        # Info Box Calidad
        st.info(f"📊 **Calidad de Datos:** Se encontraron **{stats_qa[0]}** registros de lluvia para los **{stats_qa[1]}** meses del periodo seleccionado ({stats_qa[2]:.1f}% completitud).")
        
        c1, c2, c3 = st.columns(3)
        recarga_anual_prom = df_vis['recarga_mm'].mean() * 12
        c1.metric("Recarga Media Anual", f"{recarga_anual_prom:,.0f} mm/año")
        c2.metric("Infiltración (Ki)", f"{ki*100:.1f}%")
        c3.metric("Potencial", potencial or "Sin Dato")

        fig = go.Figure()
        # Lluvia
        fig.add_trace(go.Scatter(
            x=df_vis[config.Config.DATE_COL], y=df_vis[config.Config.PRECIPITATION_COL],
            mode='lines', name='Lluvia Total', line=dict(color='blue', width=1), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'
        ))
        # Recarga
        fig.add_trace(go.Scatter(
            x=df_vis[config.Config.DATE_COL], y=df_vis['recarga_mm'],
            mode='lines', name='Recarga', line=dict(color='#2ca02c', width=2)
        ))
        # Escorrentía
        fig.add_trace(go.Scatter(
            x=df_vis[config.Config.DATE_COL], y=df_vis['escorrentia_sup_mm'],
            mode='lines', name='Escorrentía', line=dict(color='orange', width=1.5)
        ))
        # ETR
        fig.add_trace(go.Scatter(
            x=df_vis[config.Config.DATE_COL], y=df_vis['etr_mm'],
            mode='lines', name='ETR (Real)', line=dict(color='red', width=1, dash='dot')
        ))
        
        fig.update_layout(title="Balance Hídrico Mensual", yaxis_title="mm", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)

# TAB 2
with tab2:
    st.subheader("Pronóstico (Prophet)")
    h = st.slider("Horizonte (Meses):", 12, 60, 24)
    if st.button("Ejecutar Pronóstico"):
        with st.spinner("Procesando..."):
            try:
                # Agrupación Mensual (Fix Duplicate Labels)
                df_clean = df_vis.groupby(pd.Grouper(key=config.Config.DATE_COL, freq='MS')).mean().reset_index()
                df_input = df_clean.rename(columns={config.Config.DATE_COL: 'ds', 'recarga_mm': config.Config.PRECIPITATION_COL})
                
                if len(df_input) < 24:
                    st.error("Datos insuficientes (<24 meses).")
                else:
                    _, forecast, metrics = forecasting.generate_prophet_forecast(df_input, h, 12)
                    
                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(x=df_clean[config.Config.DATE_COL], y=df_clean['recarga_mm'], name="Histórico", line=dict(color='black')))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Pronóstico", line=dict(color='blue')))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), showlegend=False))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.2)', name="IC"))
                    st.plotly_chart(fig_fc, use_container_width=True)
                    st.success(f"MAE: {metrics['MAE']:.2f} mm")
            except Exception as e:
                st.error(f"Error: {e}")

# TAB 3
with tab3:
    st.subheader("Mapa Recarga Anual")
    c1, c2 = st.columns([1, 4])
    with c1:
        st.write(f"**Puntos:** {len(df_map_data)}")
        metodo = st.radio("Método", ["IDW", "Kriging Ordinario"])
        res = st.select_slider("Resolución", [50, 100, 150], value=100)
    
    with c2:
        if len(df_map_data) < 4:
            st.warning("⚠️ Mínimo 4 estaciones requeridas.")
        else:
            m = 0.05
            b = [df_map_data.longitud.min()-m, df_map_data.latitud.min()-m, df_map_data.longitud.max()+m, df_map_data.latitud.max()+m]
            gx = np.linspace(b[0], b[2], res)
            gy = np.linspace(b[1], b[3], res)
            
            with st.spinner("Interpolando..."):
                val = 'recarga_anual'
                if metodo == "IDW":
                    z = interpolation.interpolate_idw(df_map_data.longitud.values, df_map_data.latitud.values, df_map_data[val].values, gx, gy)
                else:
                    gdf_p = gpd.GeoDataFrame(df_map_data, geometry=gpd.points_from_xy(df_map_data.longitud, df_map_data.latitud))
                    z, _ = interpolation.create_kriging_by_basin(_gdf_points=gdf_p, grid_lon=gx, grid_lat=gy, value_col=val)
            
            vmin, vmax = np.nanpercentile(z, 2), np.nanpercentile(z, 98)
            fig_map = go.Figure(data=go.Contour(z=z, x=gx, y=gy, colorscale="Viridis", zmin=vmin, zmax=vmax, colorbar=dict(title="mm/año")))
            fig_map.add_trace(go.Scatter(x=df_map_data.longitud, y=df_map_data.latitud, mode='markers', marker=dict(color='black', size=3), name='Estaciones'))
            fig_map.add_trace(go.Scatter(x=[lon_central], y=[lat_central], mode='markers', marker=dict(color='red', size=10, symbol='star'), name='Centro'))
            fig_map.update_layout(height=600, xaxis=dict(scaleanchor="y", scaleratio=1))
            st.plotly_chart(fig_map, use_container_width=True)

# TAB 4
with tab4:
    st.subheader("Descargas")
    c1, c2 = st.columns(2)
    with c1:
        if not df_vis.empty:
            st.download_button("Descargar CSV", df_vis.to_csv(index=False), f"recarga_{id_est}.csv")
    with c2:
        if 'z' in locals() and z is not None:
            try:
                tif = hydrogeo_utils.generar_geotiff_bytes(z, b)
                st.download_button("Descargar Raster", tif, "recarga.tif")
                geo = hydrogeo_utils.generar_geojson_bytes(df_map_data)
                st.download_button("Descargar Puntos", geo, "estaciones.geojson")
            except: pass