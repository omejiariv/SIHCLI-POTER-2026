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

# Conexión DB
engine = db_manager.get_engine()
if not engine:
    st.error("Error de conexión a base de datos.")
    st.stop()

st.title("💧 Sistema de Recarga y Aguas Subterráneas")

# --- 1. SIDEBAR: FILTROS ESPACIALES Y TEMPORALES ---
with st.sidebar:
    st.header("Configuración Espacial")
    
    # A. Filtros Regionales (Municipios y Cuencas)
    st.markdown("### 🌍 Filtrar Región")
    
    # Cargar listas únicas para filtros
    df_loc = pd.read_sql("SELECT DISTINCT municipio, nom_est, id_estacion FROM estaciones", engine)
    lista_munis = sorted(df_loc['municipio'].unique().tolist())
    
    sel_munis = st.multiselect("Municipios:", lista_munis, placeholder="Todos los municipios")
    
    # Filtrar estaciones basado en municipio
    if sel_munis:
        df_loc_filtered = df_loc[df_loc['municipio'].isin(sel_munis)]
    else:
        df_loc_filtered = df_loc

    # Selector de Estación (Filtrado)
    est_seleccion = st.selectbox(
        "Estación de Análisis Puntual:", 
        df_loc_filtered['id_estacion'] + " - " + df_loc_filtered['nom_est']
    )
    id_est = est_seleccion.split(" - ")[0]
    
    st.markdown("---")
    st.header("⏱️ Periodo")
    # Filtro de Fechas dinámico según la estación seleccionada
    fechas = pd.read_sql(f"SELECT MIN(fecha_mes_año), MAX(fecha_mes_año) FROM precipitacion_mensual WHERE id_estacion_fk='{id_est}'", engine)
    if fechas.iloc[0,0]:
        start_dt, end_dt = fechas.iloc[0,0], fechas.iloc[0,1]
        date_range = st.slider("Rango de Fechas", min_value=start_dt.date(), max_value=end_dt.date(), value=(start_dt.date(), end_dt.date()))

# --- 2. PROCESAMIENTO (PUNTUAL) ---
# Consultas SQL para el análisis puntual (Tabs 1 y 2)
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

# Calcular Recarga Puntual
df_balance = pd.DataFrame()
if not geo_data.empty and not df_lluvia.empty:
    lat, alt, ki = geo_data.iloc[0]['latitud'], geo_data.iloc[0]['elevacion'], geo_data.iloc[0]['infiltracion_ki']
    df_balance = hydrogeo_utils.calcular_serie_recarga(df_lluvia, lat, alt, ki)
    mask_date = (df_balance[config.Config.DATE_COL].dt.date >= date_range[0]) & (df_balance[config.Config.DATE_COL].dt.date <= date_range[1])
    df_vis = df_balance[mask_date]

# --- 3. PROCESAMIENTO (ESPACIAL / MAPA) ---
# Obtener datos de TODAS las estaciones para interpolar, aplicando filtros de municipio
df_map_data = hydrogeo_utils.obtener_datos_estaciones_recarga(engine)

# Aplicar filtro de Municipios al Mapa
if sel_munis:
    # Necesitamos traer el municipio en la función obtener_datos_estaciones_recarga o hacer un join aquí
    # Haremos un join rápido con df_loc que ya tenemos en memoria
    df_map_data = df_map_data.merge(df_loc[['id_estacion', 'municipio']], on='id_estacion', how='left')
    df_map_data = df_map_data[df_map_data['municipio_y'].isin(sel_munis)]

# --- INTERFAZ DE TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis Temporal", "🔮 Pronóstico (IA)", "🗺️ Mapa de Recarga", "📥 Descargas"])

# ... TAB 1 y TAB 2 (Mantener código anterior de gráficos y prophet) ...
with tab1:
    if not df_vis.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Recarga Total", f"{df_vis['recarga_mm'].sum():,.0f} mm")
        col2.metric("Infiltración (Ki)", f"{ki*100:.1f}%" if pd.notnull(ki) else "15% (Est.)")
        col3.metric("Potencial", geo_data.iloc[0]['potencial'] or "N/A")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['recarga_mm'], mode='lines', stackgroup='one', name='Recarga', line=dict(width=0, color='#2ca02c')))
        fig.add_trace(go.Scatter(x=df_vis[config.Config.DATE_COL], y=df_vis['escorrentia_sup_mm'], mode='lines', stackgroup='one', name='Escorrentía', line=dict(width=0, color='#ff7f0e')))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Pronóstico Prophet")
    if st.button("Ejecutar Pronóstico"):
        # ... Tu lógica de forecasting existente ...
        st.info("Pronóstico generado (simulado para brevedad en esta respuesta)")

# --- TAB 3: MAPA MEJORADO ---
with tab3:
    st.subheader("Mapa Interpolado de Recarga Media")
    
    c_settings, c_map = st.columns([1, 4])
    
    with c_settings:
        st.write(f"**Estaciones:** {len(df_map_data)}")
        metodo = st.radio("Método:", ["IDW", "Kriging Ordinario"])
        resolucion = st.select_slider("Resolución:", options=[50, 100, 200], value=100)
    
    with c_map:
        if len(df_map_data) < 4:
            st.warning("Se necesitan al menos 4 estaciones para interpolar.")
        else:
            # 1. Definir Grid
            margin = 0.05
            bounds = [
                df_map_data.longitud.min() - margin, 
                df_map_data.latitud.min() - margin, 
                df_map_data.longitud.max() + margin, 
                df_map_data.latitud.max() + margin
            ]
            grid_lon = np.linspace(bounds[0], bounds[2], resolucion)
            grid_lat = np.linspace(bounds[1], bounds[3], resolucion)
            
            # 2. Interpolación
            with st.spinner(f"Interpolando con {metodo}..."):
                if metodo == "Kriging Ordinario":
                    # Crear GeoDataFrame
                    gdf_p = gpd.GeoDataFrame(
                        df_map_data, 
                        geometry=gpd.points_from_xy(df_map_data.longitud, df_map_data.latitud)
                    )
                    # CORRECCIÓN DE LLAMADA: Pasamos el argumento que espera el _
                    z_grid, _ = interpolation.create_kriging_by_basin(
                        _gdf_points=gdf_p, # Nótese el nombre del argumento si usaste kwargs, o posicional
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
            
            # 3. Visualización Mejorada (Escala de Color Inteligente)
            # Calcular percentiles para ignorar outliers extremos en la escala de color
            vmin = np.nanpercentile(z_grid, 2)
            vmax = np.nanpercentile(z_grid, 98)
            
            fig_map = go.Figure(data=go.Contour(
                z=z_grid, x=grid_lon, y=grid_lat,
                colorscale="Viridis", # Viridis suele ser mejor para hidrogeología
                zmin=vmin, 
                zmax=vmax,
                colorbar=dict(title="Recarga (mm/año)"),
                contours=dict(
                    start=vmin,
                    end=vmax,
                    size=(vmax-vmin)/15, # 15 niveles automáticos
                    showlines=False
                )
            ))
            
            # Puntos encima
            fig_map.add_trace(go.Scatter(
                x=df_map_data.longitud, y=df_map_data.latitud, 
                mode='markers', 
                marker=dict(color='black', size=3, opacity=0.5),
                text=df_map_data['nom_est'],
                name='Estaciones'
            ))
            
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=600,
                xaxis=dict(scaleanchor="y", scaleratio=1) # Mantener proporción geográfica
            )
            st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 4: DESCARGAS ---
with tab4:
    st.subheader("Centro de Descargas")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("#### 📄 Datos Tabulares")
        if not df_balance.empty:
            st.download_button(
                "Descargar Serie Histórica (CSV)",
                data=df_balance.to_csv(index=False),
                file_name=f"recarga_{id_est}.csv",
                mime="text/csv"
            )
        
        st.markdown("#### 🗺️ Vectores")
        if not df_map_data.empty:
            geojson_bytes = hydrogeo_utils.generar_geojson_bytes(df_map_data)
            st.download_button(
                "Descargar Estaciones + Recarga (GeoJSON)",
                data=geojson_bytes,
                file_name="estaciones_recarga.geojson",
                mime="application/geo+json"
            )

    with col_d2:
        st.markdown("#### 🖼️ Raster (Mapa Interpolado)")
        if 'z_grid' in locals() and z_grid is not None:
            # Botón para GeoTIFF
            try:
                tif_bytes = hydrogeo_utils.generar_geotiff_bytes(z_grid, bounds)
                st.download_button(
                    "Descargar Mapa Recarga (GeoTIFF)",
                    data=tif_bytes,
                    file_name="mapa_recarga_interpolado.tif",
                    mime="image/tiff"
                )
                st.success("Raster listo para descargar (WGS84).")
            except Exception as e:
                st.error(f"Error generando TIFF: {e}")
        else:
            st.info("Primero genera el mapa en la pestaña 'Mapa de Recarga' para habilitar la descarga.")