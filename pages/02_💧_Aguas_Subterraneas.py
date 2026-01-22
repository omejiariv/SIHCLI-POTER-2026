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
    """Calcula distancia en km (fórmula Haversine)."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat_series)
    dphi = np.radians(lat_series - lat1)
    dlambda = np.radians(lon_series - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# --- CONEXIÓN ---
engine = db_manager.get_engine()
if not engine:
    st.error("⚠️ Error crítico: No hay conexión a la base de datos.")
    st.stop()

st.title("💧 Sistema de Recarga y Aguas Subterráneas")

# ==============================================================================
# 1. SIDEBAR: FILTROS Y CONFIGURACIÓN
# ==============================================================================
with st.sidebar:
    st.header("📍 Configuración Espacial")
    
    # --- A. Filtros Regionales ---
    st.markdown("### 1. Filtrar Región")
    try:
        lista_munis = pd.read_sql("SELECT DISTINCT municipio FROM estaciones ORDER BY municipio", engine)['municipio'].tolist()
        
        # CORRECCIÓN 4: Usamos DISTINCT para evitar nombres repetidos como 'R. Chico'
        df_cuencas = pd.read_sql("SELECT DISTINCT nombre_cuenca, municipios_influencia FROM cuencas ORDER BY nombre_cuenca", engine)
        lista_cuencas = df_cuencas['nombre_cuenca'].tolist()
    except:
        lista_munis, lista_cuencas = [], []

    sel_munis = st.multiselect("Municipios:", lista_munis, placeholder="Todos")
    sel_cuencas = st.multiselect("Cuencas Hidrográficas:", lista_cuencas, placeholder="Todas")
    
    # Lógica de filtrado textual
    munis_activos = set(sel_munis) if sel_munis else set()
    
    if sel_cuencas:
        for c in sel_cuencas:
            # Filtramos el dataframe localmente
            rows = df_cuencas[df_cuencas['nombre_cuenca'] == c]
            for _, row in rows.iterrows():
                if row['municipios_influencia']:
                    txt_infl = row['municipios_influencia']
                    for m in lista_munis:
                        if m in txt_infl: 
                            munis_activos.add(m)
    
    filtros_sql = []
    if munis_activos:
        m_str = "', '".join(list(munis_activos))
        filtros_sql.append(f"municipio IN ('{m_str}')")

    where_clause = "WHERE " + " AND ".join(filtros_sql) if filtros_sql else ""
    q_selector = f"SELECT id_estacion, nom_est, latitud, longitud FROM estaciones {where_clause} ORDER BY nom_est"
    
    try:
        df_selector = pd.read_sql(q_selector, engine)
    except Exception as e:
        st.error("Error al cargar estaciones.")
        st.stop()
        
    if df_selector.empty:
        st.warning("No se encontraron estaciones con estos filtros.")
        st.stop()

    # --- B. Selección Estación Central ---
    st.markdown("### 2. Estación de Análisis")
    est_seleccion = st.selectbox(
        "Seleccione Estación:", 
        df_selector['id_estacion'] + " - " + df_selector['nom_est']
    )
    id_est = est_seleccion.split(" - ")[0]
    
    # Datos Estación Central
    est_central = df_selector[df_selector['id_estacion'] == id_est].iloc[0]
    lat_central, lon_central = est_central['latitud'], est_central['longitud']

    # --- C. Filtro de Radio (Buffer) ---
    st.markdown("### 3. Área de Influencia")
    usar_buffer = st.toggle("Aplicar Radio de Búsqueda (km)", value=True)
    radio_km = 20
    if usar_buffer:
        radio_km = st.slider("Radio (km)", 5, 200, 40) # Default 40 para capturar vecinos en Abejorral
        st.caption(f"El mapa interpolará usando estaciones a {radio_km}km a la redonda.")

    st.markdown("---")
    # Filtro Temporal
    fechas = pd.read_sql(f"SELECT MIN(fecha_mes_año), MAX(fecha_mes_año) FROM precipitacion_mensual WHERE id_estacion_fk='{id_est}'", engine)
    if fechas.iloc[0,0]:
        start_dt, end_dt = fechas.iloc[0,0], fechas.iloc[0,1]
        date_range = st.slider("Periodo de Análisis", min_value=start_dt.date(), max_value=end_dt.date(), value=(start_dt.date(), end_dt.date()))

# ==============================================================================
# 2. PROCESAMIENTO DE DATOS (Backend)
# ==============================================================================

# A. Datos Puntuales (Para Tab 1 y 2)
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
ki = 0.15 
potencial = "N/A"

if not geo_data.empty and not df_lluvia.empty:
    lat, alt = geo_data.iloc[0]['latitud'], geo_data.iloc[0]['elevacion']
    ki_db = geo_data.iloc[0]['infiltracion_ki']
    potencial = geo_data.iloc[0]['potencial']
    ki = ki_db if pd.notnull(ki_db) else 0.15
    
    # Calcular Serie
    df_balance = hydrogeo_utils.calcular_serie_recarga(df_lluvia, lat, alt, ki)
    
    # Filtrar por fecha
    mask_date = (df_balance[config.Config.DATE_COL].dt.date >= date_range[0]) & (df_balance[config.Config.DATE_COL].dt.date <= date_range[1])
    df_vis = df_balance[mask_date].copy()

# B. Datos Espaciales (Para Tab 3)
# Traemos TODAS las estaciones con recarga anual ya calculada
df_map_data = hydrogeo_utils.obtener_datos_estaciones_recarga(engine)

# Aplicar Filtro al Mapa
if usar_buffer:
    # 1. Calcular distancia REAL usando Haversine
    df_map_data['distancia_km'] = haversine_vectorized(
        lat_central, lon_central, df_map_data['latitud'], df_map_data['longitud']
    )
    # 2. Filtrar
    df_map_data_filtered = df_map_data[df_map_data['distancia_km'] <= radio_km].copy()
    
    # Debug visual (Opcional, para verificar)
    # st.sidebar.write(f"Vecinos encontrados: {len(df_map_data_filtered)}")
else:
    ids_validos = df_selector['id_estacion'].unique()
    df_map_data_filtered = df_map_data[df_map_data['id_estacion'].isin(ids_validos)].copy()

# ==============================================================================
# 3. INTERFAZ VISUAL (Frontend)
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis Temporal", "🔮 Pronóstico (IA)", "🗺️ Mapa de Recarga", "📥 Descargas"])

# --- TAB 1: GRÁFICOS ---
with tab1:
    if not df_vis.empty:
        c1, c2, c3 = st.columns(3)
        recarga_anual_promedio = df_vis['recarga_mm'].mean() * 12
        c1.metric("Recarga Media Anual", f"{recarga_anual_promedio:,.0f} mm/año")
        c2.metric("Infiltración (Ki)", f"{ki*100:.1f}%")
        c3.metric("Potencial Hidrogeológico", potencial or "Sin Dato")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis[config.Config.PRECIPITATION_COL], mode='lines', name='Precipitación', line=dict(color='rgba(50, 50, 255, 0.2)', width=1), fill='tozeroy'))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['etr_mm'], mode='lines', name='ETR', line=dict(color='red', width=1.5, dash='dot')))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['recarga_mm'], mode='lines', name='Recarga', line=dict(color='#2ca02c', width=2)))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['escorrentia_sup_mm'], mode='lines', name='Escorrentía', line=dict(color='#ff7f0e', width=1)))
        
        fig.update_layout(title="Dinámica Hidroclimática Mensual", yaxis_title="Lámina (mm)", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: PRONÓSTICO (CORRECCIÓN PROPHET) ---
with tab2:
    st.subheader("Pronóstico de Recarga (Prophet)")
    h = st.slider("Horizonte de Pronóstico (Meses):", 12, 60, 24)
    
    if st.button("Ejecutar Pronóstico"):
        with st.spinner("Calibrando modelo IA..."):
            try:
                # 1. LIMPIEZA AGRESIVA DE DATOS PARA PROPHET
                # Agrupamos por mes (inicio de mes 'MS') y promediamos valores duplicados si existen
                df_clean = df_vis.copy()
                df_clean = df_clean.set_index(config.Config.DATE_COL).resample('MS').mean().reset_index()
                
                # 2. RENOMBRADO OBLIGATORIO
                # El módulo forecasting espera 'precipitation' como target. 
                # Le pasamos 'recarga_mm' disfrazada.
                df_input = df_clean.rename(columns={config.Config.DATE_COL: 'ds', 'recarga_mm': config.Config.PRECIPITATION_COL})
                
                # Limpiar NaNs que hayan podido quedar
                df_input = df_input.dropna(subset=[config.Config.PRECIPITATION_COL])

                if len(df_input) < 24:
                    st.error("Datos insuficientes (<24 meses limpios) para el pronóstico.")
                else:
                    # 3. EJECUCIÓN
                    _, forecast, metrics = forecasting.generate_prophet_forecast(df_input, h, 12)
                    
                    # 4. VISUALIZACIÓN
                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(x=df_input['ds'], y=df_input[config.Config.PRECIPITATION_COL], name="Histórico", line=dict(color='gray')))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Pronóstico", line=dict(color='blue')))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), showlegend=False))
                    fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,255,0.2)', name="Confianza"))
                    
                    st.plotly_chart(fig_fc, use_container_width=True)
                    st.success(f"Modelo calibrado. Error MAE: {metrics['MAE']:.2f} mm")
            except Exception as e:
                st.error(f"Error en el modelo: {e}")

# --- TAB 3: MAPA (CORRECCIÓN BUFFER & VISUALIZACIÓN) ---
with tab3:
    st.subheader("Mapa de Recarga Media Anual")
    c1, c2 = st.columns([1, 4])
    
    with c1:
        st.write(f"**Puntos usados:** {len(df_map_data_filtered)}")
        metodo = st.radio("Método:", ["IDW", "Kriging Ordinario"])
        resolucion = st.select_slider("Resolución:", [50, 100, 150], value=100)
    
    with c2:
        if len(df_map_data_filtered) < 4:
            st.warning("⚠️ Se necesitan al menos 4 estaciones. Amplía el radio de búsqueda.")
        else:
            # CORRECCIÓN BUFFER: Calculamos los límites (bounds) basándonos SOLO en los datos filtrados
            margin = 0.02 # Margen pequeño para que el mapa haga "zoom" a los puntos
            bounds = [
                df_map_data_filtered.longitud.min() - margin, df_map_data_filtered.latitud.min() - margin,
                df_map_data_filtered.longitud.max() + margin, df_map_data_filtered.latitud.max() + margin
            ]
            
            # Generar grilla basada en los límites filtrados
            gx = np.linspace(bounds[0], bounds[2], resolucion)
            gy = np.linspace(bounds[1], bounds[3], resolucion)
            
            with st.spinner("Interpolando superficie..."):
                val_col = 'recarga_anual'
                
                if metodo == "IDW":
                    z = interpolation.interpolate_idw(
                        df_map_data_filtered.longitud.values, df_map_data_filtered.latitud.values, 
                        df_map_data_filtered[val_col].values, gx, gy
                    )
                else:
                    gdf_p = gpd.GeoDataFrame(df_map_data_filtered, geometry=gpd.points_from_xy(df_map_data_filtered.longitud, df_map_data_filtered.latitud))
                    z, _ = interpolation.create_kriging_by_basin(
                        _gdf_points=gdf_p, grid_lon=gx, grid_lat=gy, value_col=val_col
                    )
            
            # Escala de Color adaptativa
            vmin, vmax = np.nanpercentile(z, 2), np.nanpercentile(z, 98)
            
            fig_map = go.Figure(data=go.Contour(
                z=z, x=gx, y=gy, colorscale="Viridis", zmin=vmin, zmax=vmax,
                colorbar=dict(title="Recarga (mm/año)")
            ))
            # Puntos filtrados
            fig_map.add_trace(go.Scatter(
                x=df_map_data_filtered.longitud, y=df_map_data_filtered.latitud, mode='markers',
                marker=dict(color='black', size=5, opacity=0.6, line=dict(width=1, color='white')), name='Estaciones Vecinas'
            ))
            # Centro
            fig_map.add_trace(go.Scatter(
                x=[lon_central], y=[lat_central], mode='markers',
                marker=dict(color='red', size=12, symbol='star'), name='Tu Ubicación'
            ))
            
            # Forzar el encuadre del mapa
            fig_map.update_layout(
                height=650, 
                xaxis=dict(scaleanchor="y", scaleratio=1, range=[bounds[0], bounds[2]]),
                yaxis=dict(range=[bounds[1], bounds[3]])
            )
            st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 4: DESCARGAS ---
with tab4:
    st.subheader("Centro de Descargas")
    c1, c2 = st.columns(2)
    with c1:
        if not df_vis.empty:
            st.download_button("Descargar CSV", df_vis.to_csv(index=False), f"recarga_{id_est}.csv")
    with c2:
        if 'z' in locals() and z is not None:
            try:
                tif = hydrogeo_utils.generar_geotiff_bytes(z, bounds)
                st.download_button("Descargar Raster TIFF", tif, "recarga_anual.tif")
                geo = hydrogeo_utils.generar_geojson_bytes(df_map_data_filtered)
                st.download_button("Descargar GeoJSON", geo, "estaciones.geojson")
            except: pass