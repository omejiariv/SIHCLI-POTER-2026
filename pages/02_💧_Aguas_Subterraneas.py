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

# --- FUNCIONES AUXILIARES (ESPACIALES) ---
def haversine_vectorized(lat1, lon1, lat_series, lon_series):
    """Calcula distancia (km) entre un punto y una serie de puntos usando Haversine."""
    R = 6371  # Radio de la Tierra en km
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

# --- 1. SIDEBAR: FILTROS AVANZADOS ---
with st.sidebar:
    st.header("📍 Configuración Espacial")
    
    # --- A. Filtros Regionales (Municipios y Cuencas) ---
    st.markdown("### 1. Filtrar Región")
    
    # Cargar listas para filtros
    try:
        lista_munis = pd.read_sql("SELECT DISTINCT municipio FROM estaciones ORDER BY municipio", engine)['municipio'].tolist()
        lista_cuencas = pd.read_sql("SELECT DISTINCT nombre_cuenca FROM cuencas ORDER BY nombre_cuenca", engine)['nombre_cuenca'].tolist()
    except Exception as e:
        st.warning(f"No se pudieron cargar listas de filtros: {e}")
        lista_munis, lista_cuencas = [], []
    
    sel_munis = st.multiselect("Municipios:", lista_munis, placeholder="Todos")
    sel_cuencas = st.multiselect("Cuencas Hidrográficas:", lista_cuencas, placeholder="Todas")
    
    # Lógica de filtrado para el SELECTBOX de estaciones
    # Construimos una query dinámica para traer solo las estaciones que cumplen los filtros
    filtros_sql = []
    if sel_munis:
        m_str = "', '".join(sel_munis)
        filtros_sql.append(f"municipio IN ('{m_str}')")
    
    # Filtro espacial por cuenca (PostGIS ST_Intersects es lo más preciso)
    if sel_cuencas:
        c_str = "', '".join(sel_cuencas)
        # Subquery espacial: Estaciones que intersectan las cuencas seleccionadas
        filtros_sql.append(f"""
            ST_Intersects(
                geom, 
                (SELECT ST_Union(geom) FROM cuencas WHERE nombre_cuenca IN ('{c_str}'))
            )
        """)

    where_clause = "WHERE " + " AND ".join(filtros_sql) if filtros_sql else ""
    query_estaciones = f"SELECT id_estacion, nom_est, latitud, longitud FROM estaciones {where_clause} ORDER BY nom_est"
    
    df_estaciones_filtradas = pd.read_sql(query_estaciones, engine)
    
    if df_estaciones_filtradas.empty:
        st.warning("No hay estaciones en la región seleccionada.")
        st.stop()

    # --- B. Selección de Estación Central ---
    st.markdown("### 2. Estación de Análisis")
    est_seleccion = st.selectbox(
        "Seleccione Estación:", 
        df_estaciones_filtradas['id_estacion'] + " - " + df_estaciones_filtradas['nom_est']
    )
    id_est = est_seleccion.split(" - ")[0]
    
    # Obtener coordenadas de la estación seleccionada para el Buffer
    est_central = df_estaciones_filtradas[df_estaciones_filtradas['id_estacion'] == id_est].iloc[0]
    lat_central, lon_central = est_central['latitud'], est_central['longitud']

    # --- C. Filtro de Radio (Buffer) ---
    st.markdown("### 3. Área de Influencia")
    usar_buffer = st.toggle("Aplicar Radio de Búsqueda (km)", value=False)
    radio_km = 0
    if usar_buffer:
        radio_km = st.slider("Radio (km)", 5, 100, 20)
        st.caption(f"Filtrando estaciones a {radio_km}km de {id_est}")

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
if not geo_data.empty and not df_lluvia.empty:
    lat, alt, ki = geo_data.iloc[0]['latitud'], geo_data.iloc[0]['elevacion'], geo_data.iloc[0]['infiltracion_ki']
    df_balance = hydrogeo_utils.calcular_serie_recarga(df_lluvia, lat, alt, ki)
    mask_date = (df_balance[config.Config.DATE_COL].dt.date >= date_range[0]) & (df_balance[config.Config.DATE_COL].dt.date <= date_range[1])
    df_vis = df_balance[mask_date]

# --- 3. PROCESAMIENTO (ESPACIAL / MAPA) ---
# Paso 1: Obtener datos de TODAS las estaciones base con su recarga media
df_map_data = hydrogeo_utils.obtener_datos_estaciones_recarga(engine)

# Paso 2: Aplicar Filtros Regionales (Municipios / Cuencas)
# Usamos la lista de IDs que ya filtramos en el Sidebar con SQL para ser consistentes
ids_validos = df_estaciones_filtradas['id_estacion'].unique()
df_map_data = df_map_data[df_map_data['id_estacion'].isin(ids_validos)]

# Paso 3: Aplicar Filtro de Radio (Buffer) - CÁLCULO EN MEMORIA
if usar_buffer and radio_km > 0:
    # Calculamos distancia de todas las estaciones a la central
    df_map_data['distancia_km'] = haversine_vectorized(
        lat_central, lon_central, 
        df_map_data['latitud'], df_map_data['longitud']
    )
    # Filtramos
    df_map_data = df_map_data[df_map_data['distancia_km'] <= radio_km]

# --- INTERFAZ DE TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis Temporal", "🔮 Pronóstico (IA)", "🗺️ Mapa de Recarga", "📥 Descargas"])

with tab1:
    if not df_vis.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Recarga Total", f"{df_vis['recarga_mm'].sum():,.0f} mm")
        c2.metric("Infiltración (Ki)", f"{ki*100:.1f}%" if pd.notnull(ki) else "15% (Def.)")
        c3.metric("Potencial", geo_data.iloc[0]['potencial'] or "N/A")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['recarga_mm'], mode='lines', stackgroup='one', name='Recarga', line=dict(width=0, color='#2ca02c')))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['escorrentia_sup_mm'], mode='lines', stackgroup='one', name='Escorrentía', line=dict(width=0, color='#ff7f0e')))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Pronóstico Prophet")
    if st.button("Ejecutar Pronóstico"):
        # Lógica de forecasting simulada para brevedad
        st.info("Funcionalidad de pronóstico lista (ver implementación previa).")

with tab3:
    st.subheader("Mapa Interpolado de Recarga Media")
    c_settings, c_map = st.columns([1, 4])
    
    with c_settings:
        st.write(f"**Puntos base:** {len(df_map_data)}")
        if usar_buffer:
            st.info(f"Radio: {radio_km} km")
        metodo = st.radio("Método:", ["IDW", "Kriging Ordinario"])
        resolucion = st.select_slider("Resolución:", options=[50, 100, 150], value=100)
    
    with c_map:
        if len(df_map_data) < 4:
            st.warning("Se necesitan al menos 4 estaciones para interpolar. Intenta ampliar el radio o los filtros.")
        else:
            # Configurar Grid dinámico basado en los datos filtrados
            margin = 0.05
            bounds = [
                df_map_data.longitud.min() - margin, 
                df_map_data.latitud.min() - margin, 
                df_map_data.longitud.max() + margin, 
                df_map_data.latitud.max() + margin
            ]
            grid_lon = np.linspace(bounds[0], bounds[2], resolucion)
            grid_lat = np.linspace(bounds[1], bounds[3], resolucion)
            
            with st.spinner("Interpolando..."):
                if metodo == "Kriging Ordinario":
                    gdf_p = gpd.GeoDataFrame(
                        df_map_data, 
                        geometry=gpd.points_from_xy(df_map_data.longitud, df_map_data.latitud)
                    )
                    # NOTA: Usamos el argumento _gdf_points corregido previamente
                    z_grid, _ = interpolation.create_kriging_by_basin(
                        _gdf_points=gdf_p, 
                        grid_lon=grid_lon, 
                        grid_lat=grid_lat, 
                        value_col="recarga_media"
                    )
                else:
                    z_grid = interpolation.interpolate_idw(
                        df_map_data.longitud.values, 
                        df_map_data.latitud.values, 
                        df_map_data.recarga_media.values, 
                        grid_lon, grid_lat
                    )
            
            # Escala de Color Robusta
            vmin, vmax = np.nanpercentile(z_grid, 5), np.nanpercentile(z_grid, 95)
            
            fig_map = go.Figure(data=go.Contour(
                z=z_grid, x=grid_lon, y=grid_lat,
                colorscale="Viridis", zmin=vmin, zmax=vmax,
                colorbar=dict(title="Recarga (mm)"),
                contours=dict(start=vmin, end=vmax, size=(vmax-vmin)/10, showlines=False)
            ))
            # Puntos de estaciones
            fig_map.add_trace(go.Scatter(
                x=df_map_data.longitud, y=df_map_data.latitud, mode='markers',
                marker=dict(color='black', size=4, opacity=0.6), name='Estaciones'
            ))
            # Resaltar estación central si hay Buffer
            if usar_buffer:
                fig_map.add_trace(go.Scatter(
                    x=[lon_central], y=[lat_central], mode='markers',
                    marker=dict(color='red', size=12, symbol='star'), name='Centro Análisis'
                ))

            fig_map.update_layout(height=600, xaxis=dict(scaleanchor="y", scaleratio=1))
            st.plotly_chart(fig_map, use_container_width=True)

with tab4:
    st.subheader("Descargas")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if not df_vis.empty:
            st.download_button("Descargar Serie (CSV)", df_vis.to_csv(index=False), f"recarga_{id_est}.csv", "text/csv")
    with col_d2:
        if 'z_grid' in locals() and z_grid is not None:
             try:
                tif_bytes = hydrogeo_utils.generar_geotiff_bytes(z_grid, bounds)
                st.download_button("Descargar Raster (TIFF)", tif_bytes, "mapa_recarga.tif", "image/tiff")
             except: pass