# pages/02_💧_Aguas_Subterraneas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium

# Importación de TUS módulos
from modules import db_manager, hydrogeo_utils, forecasting, interpolation, config

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

# Conexión DB
engine = db_manager.get_engine()
if not engine:
    st.error("Error de conexión a base de datos.")
    st.stop()

st.title("💧 Sistema de Recarga y Aguas Subterráneas")

# --- SIDEBAR: FILTROS ---
with st.sidebar:
    st.header("Configuración")
    
    # Selector de Estación
    estaciones = pd.read_sql("SELECT id_estacion, nom_est FROM estaciones ORDER BY nom_est", engine)
    est_seleccion = st.selectbox(
        "Estación de Análisis:", 
        estaciones['id_estacion'] + " - " + estaciones['nom_est']
    )
    id_est = est_seleccion.split(" - ")[0]
    
    # Filtro de Fechas
    fechas = pd.read_sql(f"SELECT MIN(fecha_mes_año), MAX(fecha_mes_año) FROM precipitacion_mensual WHERE id_estacion_fk='{id_est}'", engine)
    if fechas.iloc[0,0]:
        start_dt, end_dt = fechas.iloc[0,0], fechas.iloc[0,1]
        date_range = st.slider("Periodo de Análisis", min_value=start_dt.date(), max_value=end_dt.date(), value=(start_dt.date(), end_dt.date()))

# --- PROCESAMIENTO PRINCIPAL ---
# 1. Obtener Metadatos (Suelo, Altitud)
q_geo = f"""
SELECT e.latitud, e.longitud, e.elevacion, s.infiltracion_ki, s.unidad_suelo, zh.potencial 
FROM estaciones e
LEFT JOIN suelos s ON ST_Intersects(e.geom, s.geom)
LEFT JOIN zonas_hidrogeologicas zh ON ST_Intersects(e.geom, zh.geom)
WHERE e.id_estacion = '{id_est}'
"""
geo_data = pd.read_sql(q_geo, engine)

# 2. Obtener Serie de Lluvia
q_lluvia = f"""
SELECT fecha_mes_año as {config.Config.DATE_COL}, precipitation as {config.Config.PRECIPITATION_COL}
FROM precipitacion_mensual 
WHERE id_estacion_fk = '{id_est}' 
ORDER BY fecha_mes_año
"""
df_lluvia = pd.read_sql(q_lluvia, engine)

# 3. Calcular Recarga (Usando hydrogeo_utils + analysis)
if not geo_data.empty and not df_lluvia.empty:
    lat = geo_data.iloc[0]['latitud']
    alt = geo_data.iloc[0]['elevacion']
    ki = geo_data.iloc[0]['infiltracion_ki']
    
    df_balance = hydrogeo_utils.calcular_serie_recarga(df_lluvia, lat, alt, ki)
    
    # Filtro visual por fecha
    mask_date = (df_balance[config.Config.DATE_COL].dt.date >= date_range[0]) & (df_balance[config.Config.DATE_COL].dt.date <= date_range[1])
    df_vis = df_balance[mask_date]

# --- INTERFAZ DE TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis Temporal", "🔮 Pronóstico (IA)", "🗺️ Mapa de Recarga", "📥 Datos"])

# TAB 1: SERIES TEMPORALES
with tab1:
    col1, col2, col3 = st.columns(3)
    recarga_total = df_vis['recarga_mm'].sum()
    col1.metric("Recarga Total (Periodo)", f"{recarga_total:,.0f} mm")
    col2.metric("Infiltración Suelo (Ki)", f"{ki*100:.1f}%" if pd.notnull(ki) else "15% (Est.)")
    col3.metric("Potencial Hidrogeológico", geo_data.iloc[0]['potencial'] or "N/A")

    # Gráfico de Área Apilada
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['recarga_mm'], mode='lines', stackgroup='one', name='Recarga (Acuífero)', line=dict(width=0, color='#006400')))
    fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['escorrentia_sup_mm'], mode='lines', stackgroup='one', name='Escorrentía', line=dict(width=0, color='#FFA500')))
    fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['etr_mm'], mode='lines', stackgroup='one', name='ETR', line=dict(width=0, color='#87CEEB')))
    fig.update_layout(title="Balance Hídrico Detallado", yaxis_title="mm / mes", height=400)
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: PRONÓSTICOS (PROPHET)
with tab2:
    st.subheader("Pronóstico de Recarga a 5 años")
    st.info("Utilizando el motor `Prophet` configurado en `modules/forecasting.py`")
    
    if st.button("Generar Proyección"):
        with st.spinner("Entrenando modelo Prophet..."):
            # Preparamos los datos de entrada (solo columna recarga)
            df_prophet_input = df_balance[[config.Config.DATE_COL, 'recarga_mm']].rename(
                columns={'recarga_mm': config.Config.PRECIPITATION_COL} # El módulo espera esta columna o se renombra dentro
            )
            
            # Llamamos a TU función de forecasting
            # generate_prophet_forecast espera (df, horizon, test_size)
            model, forecast, metrics = forecasting.generate_prophet_forecast(
                df_prophet_input, 
                horizon=60, # 5 años * 12 meses
                test_size=12
            )
            
            # Gráfico de resultados
            fig_fc = go.Figure()
            # Histórico
            fig_fc.add_trace(go.Scatter(x=df_prophet_input[config.Config.DATE_COL], y=df_prophet_input[config.Config.PRECIPITATION_COL], name="Histórico", line=dict(color='gray')))
            # Pronóstico
            fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Pronóstico", line=dict(color='blue')))
            # Intervalo de confianza
            fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', marker=dict(color="#444"), line=dict(width=0), showlegend=False))
            fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', marker=dict(color="#444"), line=dict(width=0), fill='tonexty', fillcolor='rgba(68, 68, 68, 0.3)', name="Intervalo Confianza"))
            
            st.plotly_chart(fig_fc, use_container_width=True)
            st.write(f"**Precisión del Modelo (Test):** RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}")

# TAB 3: MAPA DINÁMICO (INTERPOLACIÓN)
with tab3:
    st.subheader("Mapa Interpolado de Recarga Media")
    
    # 1. Obtener datos puntuales de todas las estaciones
    df_map_data = hydrogeo_utils.obtener_datos_estaciones_recarga(engine)
    
    col_k1, col_k2 = st.columns([1, 3])
    with col_k1:
        st.write(f"Puntos base: {len(df_map_data)}")
        metodo = st.radio("Método Interpolación", ["IDW", "Kriging Ordinario"], index=0)
    
    with col_k2:
        # Preparar Grilla
        bounds = [df_map_data.longitud.min(), df_map_data.latitud.min(), df_map_data.longitud.max(), df_map_data.latitud.max()]
        grid_lon = np.linspace(bounds[0], bounds[2], 100)
        grid_lat = np.linspace(bounds[1], bounds[3], 100)
        
        # Llamar a TU módulo de interpolación
        if metodo == "Kriging Ordinario":
            # Usar GeoDataFrame ficticio para compatibilidad con tu funcion
            import geopandas as gpd
            gdf_p = gpd.GeoDataFrame(df_map_data, geometry=gpd.points_from_xy(df_map_data.longitud, df_map_data.latitud))
            z_grid, _ = interpolation.create_kriging_by_basin(gdf_p, grid_lon, grid_lat, value_col="recarga_media")
        else:
            # Usar IDW básico
            z_grid = interpolation.interpolate_idw(
                df_map_data.longitud.values, 
                df_map_data.latitud.values, 
                df_map_data.recarga_media.values, 
                grid_lon, grid_lat
            )
            
        # Visualizar con Plotly Contour
        fig_map = go.Figure(data=go.Contour(
            z=z_grid, x=grid_lon, y=grid_lat,
            colorscale="YlGnBu", 
            colorbar=dict(title="Recarga (mm/mes)")
        ))
        # Añadir puntos de estaciones
        fig_map.add_trace(go.Scatter(x=df_map_data.longitud, y=df_map_data.latitud, mode='markers', marker=dict(color='black', size=4), name='Estaciones'))
        st.plotly_chart(fig_map, use_container_width=True)

# TAB 4: DATOS
with tab4:
    st.dataframe(df_balance)