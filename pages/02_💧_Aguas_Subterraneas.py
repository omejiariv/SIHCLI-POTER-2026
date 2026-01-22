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
        # Intentamos cargar cuencas, si falla la tabla, lista vacía
        try:
            lista_cuencas = pd.read_sql("SELECT DISTINCT nombre_cuenca FROM cuencas ORDER BY nombre_cuenca", engine)['nombre_cuenca'].tolist()
        except:
            lista_cuencas = []
    except:
        lista_munis = []

    sel_munis = st.multiselect("Municipios:", lista_munis, placeholder="Todos")
    sel_cuencas = st.multiselect("Cuencas (Ref. Textual):", lista_cuencas, placeholder="Todas")
    
    # Construcción de Query Dinámica para SELECTBOX
    filtros_where = []
    
    if sel_munis:
        m_str = "', '".join(sel_munis)
        filtros_where.append(f"municipio IN ('{m_str}')")
    
    if sel_cuencas:
        # CORRECCIÓN ERROR SQL: Tu tabla cuencas NO tiene geometría.
        # No podemos hacer filtro espacial real. 
        # Intentamos filtrar por texto si existe 'nombre_cuenca' en la tabla estaciones (a veces pasa)
        # O mostramos advertencia y no filtramos para no romper la app.
        st.warning("⚠️ La tabla 'cuencas' no tiene geometría en BD. El filtro será ignorado temporalmente.")
        # Si quisieras filtrar, necesitarías una columna 'cuenca' en la tabla 'estaciones'.

    clause = "WHERE " + " AND ".join(filtros_where) if filtros_where else ""
    q_selector = f"SELECT id_estacion, nom_est, latitud, longitud, municipio FROM estaciones {clause} ORDER BY nom_est"
    
    try:
        df_selector = pd.read_sql(q_selector, engine)
    except Exception as e:
        st.error(f"Error cargando lista de estaciones: {e}")
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

    # C. Filtro de Radio (Buffer)
    st.markdown("### 3. Área de Influencia")
    usar_buffer = st.toggle("Aplicar Radio de Búsqueda (km)", value=True)
    radio_km = 20
    if usar_buffer:
        radio_km = st.slider("Radio (km)", 5, 200, 40) # Aumenté el rango a 200km para asegurar vecinos
        st.caption(f"El Mapa usará estaciones a {radio_km}km")

    st.markdown("---")
    # Filtro Temporal
    fechas = pd.read_sql(f"SELECT MIN(fecha_mes_año), MAX(fecha_mes_año) FROM precipitacion_mensual WHERE id_estacion_fk='{id_est}'", engine)
    start_val, end_val = fechas.iloc[0,0], fechas.iloc[0,1]
    
    if start_val and end_val:
        date_range = st.slider("Periodo de Análisis", min_value=start_val.date(), max_value=end_val.date(), value=(start_val.date(), end_val.date()))
    else:
        st.error("Esta estación no tiene datos de lluvia.")
        st.stop()

# --- 2. PROCESAMIENTO PUNTUAL (TAB 1 y 2) ---
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
if not geo_data.empty and not df_lluvia.empty:
    lat, alt, ki_db = geo_data.iloc[0]['latitud'], geo_data.iloc[0]['elevacion'], geo_data.iloc[0]['infiltracion_ki']
    ki = ki_db if pd.notnull(ki_db) else 0.15
    
    # Calcular Serie
    df_balance = hydrogeo_utils.calcular_serie_recarga(df_lluvia, lat, alt, ki)
    
    # Filtro fecha
    mask_date = (df_balance[config.Config.DATE_COL].dt.date >= date_range[0]) & (df_balance[config.Config.DATE_COL].dt.date <= date_range[1])
    df_vis = df_balance[mask_date].copy()

# --- 3. PROCESAMIENTO ESPACIAL (MAPA) ---
# Paso A: Traer TODAS las estaciones con datos (Recarga ANUAL calculada en utils)
df_map_data = hydrogeo_utils.obtener_datos_estaciones_recarga(engine)

# Paso B: Filtrar Mapa
if usar_buffer:
    # Filtro Geográfico Puro (Distancia)
    df_map_data['distancia_km'] = haversine_vectorized(
        lat_central, lon_central, df_map_data['latitud'], df_map_data['longitud']
    )
    df_map_data = df_map_data[df_map_data['distancia_km'] <= radio_km]
else:
    # Filtro Administrativo (Municipios seleccionados)
    ids_validos = df_selector['id_estacion'].unique()
    df_map_data = df_map_data[df_map_data['id_estacion'].isin(ids_validos)]

# --- INTERFAZ TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis Temporal", "🔮 Pronóstico (IA)", "🗺️ Mapa de Recarga", "📥 Descargas"])

# TAB 1: GRÁFICO COMPLETO
with tab1:
    if not df_vis.empty:
        # Métricas
        c1, c2, c3 = st.columns(3)
        recarga_anual_prom = df_vis['recarga_mm'].mean() * 12
        c1.metric("Recarga Media Anual", f"{recarga_anual_prom:,.0f} mm/año")
        c2.metric("Infiltración (Ki)", f"{ki*100:.1f}%")
        c3.metric("Potencial", geo_data.iloc[0]['potencial'] or "N/A")

        # Gráfico Multivariable
        fig = go.Figure()
        
        # 1. Lluvia (Area Azul de fondo)
        fig.add_trace(go.Scatter(
            x=df_vis[config.Config.DATE_COL], y=df_vis[config.Config.PRECIPITATION_COL],
            mode='lines', name='Lluvia Total', line=dict(color='rgba(0,0,255,0.2)', width=1), fill='tozeroy'
        ))
        
        # 2. ETR (Línea Roja)
        fig.add_trace(go.Scatter(
            x=df_vis[config.Config.DATE_COL], y=df_vis['etr_mm'],
            mode='lines', name='Evapotranspiración (ETR)', line=dict(color='red', width=1.5, dash='dot')
        ))

        # 3. Recarga (Linea Verde Sólida)
        fig.add_trace(go.Scatter(
            x=df_vis[config.Config.DATE_COL], y=df_vis['recarga_mm'],
            mode='lines', name='Recarga Efectiva', line=dict(width=2, color='#2ca02c')
        ))
        
        # 4. Escorrentía (Linea Naranja)
        fig.add_trace(go.Scatter(
            x=df_vis[config.Config.DATE_COL], y=df_vis['escorrentia_sup_mm'],
            mode='lines', name='Escorrentía Sup.', line=dict(width=1, color='#ff7f0e')
        ))
        
        fig.update_layout(
            title=f"Balance Hídrico Detallado - {est_central['nom_est']}",
            yaxis_title="Lámina de Agua (mm/mes)",
            hovermode="x unified",
            height=500,
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: PRONÓSTICO CORREGIDO
with tab2:
    st.subheader("Pronóstico de Recarga (Prophet)")
    
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        horizonte = st.slider("Meses adelante:", 12, 60, 24)
        ejecutar = st.button("Ejecutar Pronóstico")
    
    if ejecutar:
        with st.spinner("Entrenando modelo IA..."):
            try:
                # 1. PREPARACIÓN DE DATOS (Corrección Duplicate Labels)
                # Agrupamos por fecha para eliminar duplicados si existen
                df_clean = df_vis.groupby(config.Config.DATE_COL, as_index=False).mean()
                
                # Seleccionamos columnas y renombramos para Prophet
                df_prophet_input = df_clean[[config.Config.DATE_COL, 'recarga_mm']].rename(
                    columns={config.Config.DATE_COL: 'ds', 'recarga_mm': 'y'}
                )
                
                # Verificar longitud mínima
                if len(df_prophet_input) < 24:
                    st.error("Datos insuficientes (<24 meses) para pronosticar.")
                else:
                    # El módulo forecasting espera config.Config.PRECIPITATION_COL como variable objetivo
                    # Engañamos al módulo pasando 'recarga' como si fuera 'precipitation'
                    df_mod = df_prophet_input.rename(columns={'y': config.Config.PRECIPITATION_COL, 'ds': config.Config.DATE_COL})
                    
                    # Llamada al módulo
                    _, forecast, metrics = forecasting.generate_prophet_forecast(
                        df_mod, 
                        horizon=horizonte,
                        test_size=12
                    )
                    
                    # Visualización
                    fig_fc = go.Figure()
                    
                    # Histórico (Negro)
                    fig_fc.add_trace(go.Scatter(
                        x=df_prophet_input['ds'], y=df_prophet_input['y'], 
                        name="Histórico", line=dict(color='black', width=1)
                    ))
                    
                    # Pronóstico (Azul)
                    fig_fc.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat'], 
                        name="Pronóstico", line=dict(color='blue', width=2)
                    ))
                    
                    # Banda Confianza
                    fig_fc.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat_upper'], 
                        mode='lines', line=dict(width=0), showlegend=False
                    ))
                    fig_fc.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat_lower'], 
                        mode='lines', line=dict(width=0), fill='tonexty', 
                        fillcolor='rgba(0,0,255,0.15)', name="Intervalo Confianza"
                    ))
                    
                    fig_fc.update_layout(title="Proyección de Recarga Futura", yaxis_title="Recarga (mm/mes)")
                    st.plotly_chart(fig_fc, use_container_width=True)
                    st.success(f"Modelo calibrado. Error promedio (MAE): {metrics['MAE']:.2f} mm")
                
            except Exception as e:
                st.error(f"Error técnico en pronóstico: {e}")

# TAB 3: MAPA (ESCALA ANUAL)
with tab3:
    st.subheader("Mapa Interpolado de Recarga Media Anual")
    c_set, c_map = st.columns([1, 4])
    
    with c_set:
        st.write(f"**Puntos usados:** {len(df_map_data)}")
        if usar_buffer:
            st.info(f"Radio aplicado: {radio_km} km")
        metodo = st.radio("Método:", ["IDW", "Kriging Ordinario"])
        resolucion = st.select_slider("Resolución Grid:", [50, 100, 150], value=100)
    
    with c_map:
        if len(df_map_data) < 4:
            st.error("⚠️ Menos de 4 estaciones en el área. Aumenta el radio de búsqueda.")
        else:
            # Grid
            margin = 0.05
            bounds = [
                df_map_data.longitud.min() - margin, df_map_data.latitud.min() - margin,
                df_map_data.longitud.max() + margin, df_map_data.latitud.max() + margin
            ]
            glon = np.linspace(bounds[0], bounds[2], resolucion)
            glat = np.linspace(bounds[1], bounds[3], resolucion)
            
            with st.spinner(f"Interpolando {len(df_map_data)} puntos..."):
                # Usamos 'recarga_anual' (mm/año) para que la escala tenga sentido
                vals = df_map_data.recarga_anual.values 
                
                if metodo == "IDW":
                    z_grid = interpolation.interpolate_idw(
                        df_map_data.longitud.values, df_map_data.latitud.values,
                        vals, glon, glat
                    )
                else:
                    gdf_p = gpd.GeoDataFrame(df_map_data, geometry=gpd.points_from_xy(df_map_data.longitud, df_map_data.latitud))
                    z_grid, _ = interpolation.create_kriging_by_basin(
                        _gdf_points=gdf_p, grid_lon=glon, grid_lat=glat, value_col="recarga_anual"
                    )
            
            # Escala Color (Percentiles para evitar outliers)
            vmin, vmax = np.nanpercentile(z_grid, 2), np.nanpercentile(z_grid, 98)
            
            fig_map = go.Figure(data=go.Contour(
                z=z_grid, x=glon, y=glat,
                colorscale="YlGnBu", zmin=vmin, zmax=vmax,
                colorbar=dict(title="Recarga (mm/año)"),
                contours=dict(start=vmin, end=vmax, size=(vmax-vmin)/15, showlines=False)
            ))
            
            # Estaciones
            fig_map.add_trace(go.Scatter(
                x=df_map_data.longitud, y=df_map_data.latitud, mode='markers',
                marker=dict(color='black', size=4, opacity=0.4), name='Estaciones'
            ))
            # Centro
            fig_map.add_trace(go.Scatter(
                x=[lon_central], y=[lat_central], mode='markers',
                marker=dict(color='red', size=12, symbol='cross'), name='Tu Ubicación'
            ))
            
            fig_map.update_layout(height=650, xaxis=dict(scaleanchor="y", scaleratio=1), template="plotly_white")
            st.plotly_chart(fig_map, use_container_width=True)

# TAB 4: DESCARGAS ACTIVAS
with tab4:
    st.subheader("Centro de Descargas")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("##### 📄 Datos Tabulares")
        if not df_vis.empty:
            st.download_button("Descargar Serie Histórica (CSV)", df_vis.to_csv(index=False), f"recarga_{id_est}.csv", "text/csv")
    
    with col_d2:
        st.markdown("##### 🗺️ Mapas Generados")
        if 'z_grid' in locals() and z_grid is not None:
            try:
                # 1. Raster TIFF
                tif_bytes = hydrogeo_utils.generar_geotiff_bytes(z_grid, bounds)
                st.download_button("Descargar Raster (GeoTIFF)", tif_bytes, "mapa_recarga_anual.tif", "image/tiff")
                
                # 2. Vector GeoJSON
                geo_bytes = hydrogeo_utils.generar_geojson_bytes(df_map_data)
                st.download_button("Descargar Puntos (GeoJSON)", geo_bytes, "estaciones_recarga.geojson", "application/geo+json")
                
                st.success("✅ Archivos generados en memoria listos para descargar.")
            except Exception as e:
                st.error(f"Error generando descargas: {e}")
        else:
            st.info("Genera primero el mapa en la pestaña anterior.")