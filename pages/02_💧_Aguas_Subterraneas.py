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
    """Calcula distancia (km) usando fórmula de Haversine."""
    R = 6371  # Radio Tierra km
    phi1, phi2 = np.radians(lat1), np.radians(lat_series)
    dphi = np.radians(lat_series - lat1)
    dlambda = np.radians(lon_series - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# --- INICIO APP ---
engine = db_manager.get_engine()
if not engine:
    st.error("Error de conexión a base de datos.")
    st.stop()

st.title("💧 Sistema de Recarga y Aguas Subterráneas")

# --- 1. SIDEBAR: FILTROS ---
with st.sidebar:
    st.header("📍 Configuración Espacial")
    
    # A. Filtros Regionales
    st.markdown("### 1. Filtrar Región")
    try:
        lista_munis = pd.read_sql("SELECT DISTINCT municipio FROM estaciones ORDER BY municipio", engine)['municipio'].tolist()
        lista_cuencas = pd.read_sql("SELECT DISTINCT nombre_cuenca FROM cuencas ORDER BY nombre_cuenca", engine)['nombre_cuenca'].tolist()
    except:
        lista_munis, lista_cuencas = [], []
    
    sel_munis = st.multiselect("Municipios:", lista_munis, placeholder="Todos")
    sel_cuencas = st.multiselect("Cuencas Hidrográficas:", lista_cuencas, placeholder="Todas")
    
    # Construcción de Query Dinámica para SELECTBOX (Evitando el error SQL)
    filtros_where = []
    
    if sel_munis:
        m_str = "', '".join(sel_munis)
        filtros_where.append(f"municipio IN ('{m_str}')")
    
    if sel_cuencas:
        c_str = "', '".join(sel_cuencas)
        # SOLUCIÓN ERROR SQL: Usamos una subquery estándar en lugar de agregaciones en WHERE
        subquery_cuencas = f"""
        id_estacion IN (
            SELECT e2.id_estacion 
            FROM estaciones e2 
            JOIN cuencas c ON ST_Intersects(e2.geom, c.geom) 
            WHERE c.nombre_cuenca IN ('{c_str}')
        )
        """
        filtros_where.append(subquery_cuencas)

    clause = "WHERE " + " AND ".join(filtros_where) if filtros_where else ""
    q_selector = f"SELECT id_estacion, nom_est, latitud, longitud, municipio FROM estaciones {clause} ORDER BY nom_est"
    
    try:
        df_selector = pd.read_sql(q_selector, engine)
    except Exception as e:
        st.error(f"Error en filtro: {e}")
        st.stop()
        
    if df_selector.empty:
        st.warning("No hay estaciones con estos filtros.")
        st.stop()

    # B. Selección Estación Central
    st.markdown("### 2. Estación de Análisis")
    est_seleccion = st.selectbox(
        "Seleccione Estación:", 
        df_selector['id_estacion'] + " - " + df_selector['nom_est']
    )
    id_est = est_seleccion.split(" - ")[0]
    
    # Datos de la estación central
    est_central = df_selector[df_selector['id_estacion'] == id_est].iloc[0]
    lat_central, lon_central = est_central['latitud'], est_central['longitud']

    # C. Filtro de Radio (Buffer) - CORREGIDO
    st.markdown("### 3. Área de Influencia")
    usar_buffer = st.toggle("Aplicar Radio de Búsqueda (km)", value=True) # Default True suele ser mejor UX
    radio_km = 0
    if usar_buffer:
        radio_km = st.slider("Radio (km)", 5, 100, 20)
        st.caption(f"Buscar vecinos a {radio_km}km (Ignora límites municipales)")

    st.markdown("---")
    # Filtro Temporal
    fechas = pd.read_sql(f"SELECT MIN(fecha_mes_año), MAX(fecha_mes_año) FROM precipitacion_mensual WHERE id_estacion_fk='{id_est}'", engine)
    if fechas.iloc[0,0]:
        start_dt, end_dt = fechas.iloc[0,0], fechas.iloc[0,1]
        date_range = st.slider("Periodo de Análisis", min_value=start_dt.date(), max_value=end_dt.date(), value=(start_dt.date(), end_dt.date()))

# --- 2. PROCESAMIENTO (PUNTUAL) ---
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
ki = 0.15 # Default
if not geo_data.empty and not df_lluvia.empty:
    lat, alt, ki_db = geo_data.iloc[0]['latitud'], geo_data.iloc[0]['elevacion'], geo_data.iloc[0]['infiltracion_ki']
    ki = ki_db if pd.notnull(ki_db) else 0.15
    
    # Calcular Serie
    df_balance = hydrogeo_utils.calcular_serie_recarga(df_lluvia, lat, alt, ki)
    
    # Filtro fecha
    mask_date = (df_balance[config.Config.DATE_COL].dt.date >= date_range[0]) & (df_balance[config.Config.DATE_COL].dt.date <= date_range[1])
    df_vis = df_balance[mask_date].copy()

# --- 3. PROCESAMIENTO (ESPACIAL / MAPA) ---
# Paso A: Traer TODAS las estaciones con datos para tener universo de búsqueda
df_map_data = hydrogeo_utils.obtener_datos_estaciones_recarga(engine)

# Paso B: Filtrar Mapa (Lógica Corregida)
if usar_buffer:
    # Si hay buffer, buscamos vecinos por distancia (IGNORANDO filtro de municipio/cuenca)
    df_map_data['distancia_km'] = haversine_vectorized(
        lat_central, lon_central, df_map_data['latitud'], df_map_data['longitud']
    )
    df_map_data = df_map_data[df_map_data['distancia_km'] <= radio_km]
else:
    # Si NO hay buffer, respetamos estrictamente los filtros del sidebar
    ids_validos = df_selector['id_estacion'].unique()
    df_map_data = df_map_data[df_map_data['id_estacion'].isin(ids_validos)]

# --- INTERFAZ TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis Temporal", "🔮 Pronóstico (IA)", "🗺️ Mapa de Recarga", "📥 Descargas"])

# TAB 1: GRÁFICO COMPLETO
with tab1:
    if not df_vis.empty:
        # Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("Recarga Total (Periodo)", f"{df_vis['recarga_mm'].sum():,.0f} mm")
        c2.metric("Infiltración (Ki)", f"{ki*100:.1f}%")
        c3.metric("Potencial", geo_data.iloc[0]['potencial'] or "N/A")

        # Gráfico Multivariable
        fig = go.Figure()
        
        # 1. Lluvia (Barras al fondo o Línea Azul)
        fig.add_trace(go.Scatter(
            x=df_vis[config.Config.DATE_COL], y=df_vis[config.Config.PRECIPITATION_COL],
            mode='lines', name='Lluvia (Ppt)', line=dict(color='rgba(0,0,255,0.3)', width=1), fill='tozeroy'
        ))
        
        # 2. ETR (Línea Roja punteada)
        fig.add_trace(go.Scatter(
            x=df_vis[config.Config.DATE_COL], y=df_vis['etr_mm'],
            mode='lines', name='ETR', line=dict(color='red', width=1, dash='dot')
        ))

        # 3. Recarga (Área Verde)
        fig.add_trace(go.Scatter(
            x=df_vis[config.Config.DATE_COL], y=df_vis['recarga_mm'],
            mode='lines', stackgroup='one', name='Recarga (Acuífero)', line=dict(width=0, color='#2ca02c')
        ))
        
        # 4. Escorrentía (Área Naranja)
        fig.add_trace(go.Scatter(
            x=df_vis[config.Config.DATE_COL], y=df_vis['escorrentia_sup_mm'],
            mode='lines', stackgroup='one', name='Escorrentía', line=dict(width=0, color='#ff7f0e')
        ))
        
        fig.update_layout(
            title="Balance Hídrico (Lluvia, ETR, Recarga, Escorrentía)",
            yaxis_title="mm / mes",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: PRONÓSTICO REAL
with tab2:
    st.subheader("Pronóstico de Recarga (Prophet)")
    
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        horizonte = st.slider("Meses a pronosticar:", 12, 60, 24)
        ejecutar = st.button("Ejecutar IA")
    
    if ejecutar:
        with st.spinner("Entrenando modelo predictivo..."):
            try:
                # Preparar datos para Forecasting (ds, y)
                # Usamos la serie de recarga calculada como 'y'
                df_prophet = df_vis[[config.Config.DATE_COL, 'recarga_mm']].rename(
                    columns={config.Config.DATE_COL: 'ds', 'recarga_mm': 'y'}
                )
                
                # Llamar al módulo forecasting existente
                # Nota: generate_prophet_forecast retorna (model, forecast_df, metrics)
                # Ojo: Tu modulo forecasting.generate_prophet_forecast espera columnas específicas.
                # Aseguramos compatibilidad renombrando temporalmente para el modulo
                df_input_mod = df_vis.rename(columns={config.Config.DATE_COL: 'ds', 'recarga_mm': config.Config.PRECIPITATION_COL})
                
                _, forecast, metrics = forecasting.generate_prophet_forecast(
                    df_input_mod, 
                    horizon=horizonte,
                    test_size=12
                )
                
                # Visualización Pronóstico
                fig_fc = go.Figure()
                
                # Histórico
                fig_fc.add_trace(go.Scatter(
                    x=df_prophet['ds'], y=df_prophet['y'], 
                    name="Histórico", line=dict(color='gray', width=1)
                ))
                
                # Predicción
                fig_fc.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat'], 
                    name="Pronóstico", line=dict(color='blue', width=2)
                ))
                
                # Intervalo Confianza
                fig_fc.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_upper'], 
                    mode='lines', line=dict(width=0), showlegend=False
                ))
                fig_fc.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_lower'], 
                    mode='lines', line=dict(width=0), fill='tonexty', 
                    fillcolor='rgba(0,0,255,0.1)', name="Intervalo Confianza"
                ))
                
                st.plotly_chart(fig_fc, use_container_width=True)
                
                st.success(f"Modelo Entrenado. Error (RMSE): {metrics['RMSE']:.2f} mm")
                
            except Exception as e:
                st.error(f"Error en pronóstico: {e}")
                st.info("Verifica que existan suficientes datos históricos (>24 meses) para entrenar el modelo.")

# TAB 3: MAPA
with tab3:
    st.subheader("Interpolación Espacial de Recarga")
    c_set, c_map = st.columns([1, 4])
    
    with c_set:
        st.write(f"**Puntos usados:** {len(df_map_data)}")
        metodo = st.radio("Método:", ["IDW", "Kriging Ordinario"])
        resolucion = st.select_slider("Resolución:", [50, 100, 150], value=100)
    
    with c_map:
        if len(df_map_data) < 4:
            st.error("⚠️ Menos de 4 estaciones encontradas. Aumenta el radio de búsqueda en el Sidebar.")
        else:
            # Grid Dinámico
            margin = 0.05
            bounds = [
                df_map_data.longitud.min() - margin, df_map_data.latitud.min() - margin,
                df_map_data.longitud.max() + margin, df_map_data.latitud.max() + margin
            ]
            glon = np.linspace(bounds[0], bounds[2], resolucion)
            glat = np.linspace(bounds[1], bounds[3], resolucion)
            
            with st.spinner("Interpolando..."):
                if metodo == "IDW":
                    z_grid = interpolation.interpolate_idw(
                        df_map_data.longitud.values, df_map_data.latitud.values,
                        df_map_data.recarga_media.values, glon, glat
                    )
                else:
                    gdf_p = gpd.GeoDataFrame(df_map_data, geometry=gpd.points_from_xy(df_map_data.longitud, df_map_data.latitud))
                    z_grid, _ = interpolation.create_kriging_by_basin(
                        _gdf_points=gdf_p, grid_lon=glon, grid_lat=glat, value_col="recarga_media"
                    )
            
            # Mapa
            vmin, vmax = np.nanpercentile(z_grid, 5), np.nanpercentile(z_grid, 95)
            fig_map = go.Figure(data=go.Contour(
                z=z_grid, x=glon, y=glat,
                colorscale="Viridis", zmin=vmin, zmax=vmax,
                colorbar=dict(title="Recarga (mm)"),
                contours=dict(start=vmin, end=vmax, size=(vmax-vmin)/12, showlines=False)
            ))
            fig_map.add_trace(go.Scatter(
                x=df_map_data.longitud, y=df_map_data.latitud, mode='markers',
                marker=dict(color='black', size=4, opacity=0.5), name='Estaciones'
            ))
            # Estación central
            fig_map.add_trace(go.Scatter(
                x=[lon_central], y=[lat_central], mode='markers',
                marker=dict(color='red', size=12, symbol='star'), name='Tu Estación'
            ))
            fig_map.update_layout(height=600, xaxis=dict(scaleanchor="y", scaleratio=1))
            st.plotly_chart(fig_map, use_container_width=True)

# TAB 4: DESCARGAS
with tab4:
    st.subheader("Centro de Descargas")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("##### 📄 Datos CSV")
        if not df_vis.empty:
            st.download_button("Descargar Serie Histórica", df_vis.to_csv(index=False), f"recarga_{id_est}.csv", "text/csv")
    
    with col_d2:
        st.markdown("##### 🗺️ Mapas (Raster/Vector)")
        if 'z_grid' in locals() and z_grid is not None:
            try:
                # Descargar Raster
                tif_bytes = hydrogeo_utils.generar_geotiff_bytes(z_grid, bounds)
                st.download_button("Descargar Raster (GeoTIFF)", tif_bytes, "mapa_recarga.tif", "image/tiff")
                
                # Descargar Puntos
                geo_bytes = hydrogeo_utils.generar_geojson_bytes(df_map_data)
                st.download_button("Descargar Puntos (GeoJSON)", geo_bytes, "estaciones_recarga.geojson", "application/geo+json")
            except Exception as e:
                st.error(f"Error generando archivos de mapa: {e}")
        else:
            st.info("Primero genera el mapa en la pestaña anterior para descargar.")