# pages/02_💧_Aguas_Subterraneas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
import geopandas as gpd
from scipy.interpolate import griddata
import folium
from streamlit_folium import st_folium
from branca.colormap import LinearColormap

# Importaciones locales
from modules import db_manager, hydrogeo_utils, selectors

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

# --- CSS PARA ESTÉTICA ---
st.markdown("""
<style>
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #eee; }
    .stAlert { padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("💧 Sistema de Inteligencia: Aguas Subterráneas")
st.markdown("---")

# ==============================================================================
# 1. SELECTOR ESPACIAL (Recuperar Zona)
# ==============================================================================
# Usamos tu selector, pero ignoramos el primer valor (ids) ya que viene vacío
_, nombre_zona, altitud_ref, gdf_zona = selectors.render_selector_espacial()

if gdf_zona is None or gdf_zona.empty:
    st.info("👈 Seleccione una zona (Cuenca, Municipio o Depto) en el menú lateral.")
    st.stop()

# ==============================================================================
# 2. BÚSQUEDA DE ESTACIONES (Lógica Corregida)
# ==============================================================================
engine = db_manager.get_engine()
if not engine:
    st.error("Error de conexión a BD.")
    st.stop()

# A. Calcular límites de la zona seleccionada (Bounding Box)
minx, miny, maxx, maxy = gdf_zona.total_bounds

# B. Consultar estaciones dentro de esa caja geográfica
# Usamos latitud/longitud directas para máxima velocidad y compatibilidad
q_estaciones = f"""
SELECT id_estacion, nom_est, latitud, longitud, alt_est 
FROM estaciones 
WHERE longitud BETWEEN {minx} AND {maxx}
  AND latitud BETWEEN {miny} AND {maxy}
"""
df_puntos = pd.read_sql(q_estaciones, engine)

if df_puntos.empty:
    st.warning(f"No se encontraron estaciones dentro de los límites de {nombre_zona}.")
    st.stop()

# Lista de IDs encontrados
ids_estaciones = df_puntos['id_estacion'].tolist()
ids_sql = tuple(ids_estaciones)
if len(ids_sql) == 1: ids_sql = f"('{ids_estaciones[0]}')"

# ==============================================================================
# 3. CONFIGURACIÓN LATERAL (Parámetros)
# ==============================================================================
with st.sidebar:
    st.divider()
    st.header("⚙️ Parámetros Hidrogeológicos")
    
    col_s1, col_s2 = st.columns(2)
    pct_bosque = col_s1.number_input("% Bosque", 0, 100, 50)
    pct_cultivo = col_s2.number_input("% Agrícola", 0, 100, 30)
    pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
    
    # Ki Ponderado
    ki_ponderado = ((pct_bosque * 0.50) + (pct_cultivo * 0.30) + (pct_urbano * 0.10)) / 100.0
    st.metric("Coef. Infiltración ($K_i$)", f"{ki_ponderado:.2f}")
    
    st.divider()
    st.subheader("🔮 Configuración Pronóstico")
    meses_futuros = st.slider("Horizonte (Meses)", 12, 60, 24)
    ruido = st.slider("Incertidumbre", 0.0, 1.0, 0.1)

# ==============================================================================
# 4. CÁLCULO DEL BALANCE (Motor Híbrido)
# ==============================================================================

# A. Traer serie histórica PROMEDIO de la zona
q_serie = f"""
SELECT fecha_mes_año as fecha, AVG(precipitation) as precipitation 
FROM precipitacion_mensual 
WHERE id_estacion_fk IN {ids_sql} 
GROUP BY fecha_mes_año 
ORDER BY fecha_mes_año
"""
df_serie = pd.read_sql(q_serie, engine)

if df_serie.empty:
    st.error("Las estaciones encontradas no tienen datos históricos de precipitación.")
    st.stop()

# B. Ejecutar Pronóstico + Balance Hídrico (Usando hydrogeo_utils)
# Pasamos la altitud de referencia del selector (o promedio de estaciones si es nula)
alt_calc = altitud_ref if altitud_ref else df_puntos['alt_est'].mean()

df_proyeccion = hydrogeo_utils.ejecutar_pronostico_prophet(
    df_serie, meses_futuros, alt_calc, ki_ponderado, ruido
)

# KPIs Principales
if not df_proyeccion.empty:
    # Filtramos el último año histórico para los KPIs o el promedio histórico total
    df_hist = df_proyeccion[df_proyeccion['tipo'] == 'Histórico']
    
    p_anual = df_hist['p_final'].mean() * 12
    recarga_anual = df_hist['recarga_mm'].mean() * 12
    etr_anual = df_hist['etr_mm'].mean() * 12
    
    # Métricas en columnas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lluvia Media", f"{p_anual:,.0f} mm/año")
    c2.metric("ETR (Turc)", f"{etr_anual:,.0f} mm/año")
    c3.metric("Recarga Potencial", f"{recarga_anual:,.0f} mm/año", delta="Acuífero")
    c4.metric("Estaciones Activas", len(df_puntos))

# ==============================================================================
# 5. VISUALIZACIÓN (TABS)
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis Temporal", "🗺️ Mapa Interactivo", "🌈 Superficie Recarga", "📥 Descargas"])

# --- TAB 1: GRÁFICO HISTÓRICO Y PRONÓSTICO ---
with tab1:
    if not df_proyeccion.empty:
        fig = go.Figure()
        
        # Histórico
        hist = df_proyeccion[df_proyeccion['tipo'] == 'Histórico']
        fig.add_trace(go.Scatter(x=hist['fecha'], y=hist['p_final'], name='Lluvia Histórica', line=dict(color='gray', width=1)))
        fig.add_trace(go.Scatter(x=hist['fecha'], y=hist['recarga_mm'], name='Recarga Histórica', line=dict(color='#2ca02c', width=2), fill='tozeroy'))
        
        # Futuro
        fut = df_proyeccion[df_proyeccion['tipo'] == 'Proyección']
        fig.add_trace(go.Scatter(x=fut['fecha'], y=fut['p_final'], name='Lluvia Proyectada', line=dict(color='silver', dash='dot')))
        fig.add_trace(go.Scatter(x=fut['fecha'], y=fut['recarga_mm'], name='Recarga Futura', line=dict(color='#00ff00', width=2, dash='solid')))
        
        # Intervalo de confianza
        fig.add_trace(go.Scatter(x=fut['fecha'], y=fut['yhat_upper'], showlegend=False, line=dict(width=0)))
        fig.add_trace(go.Scatter(x=fut['fecha'], y=fut['yhat_lower'], fill='tonexty', fillcolor='rgba(0,255,0,0.1)', name='Incertidumbre', line=dict(width=0)))
        
        fig.update_layout(
            title=f"Dinámica Hídrica - {nombre_zona}", 
            yaxis_title="mm / mes", 
            hovermode="x unified",
            height=450,
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: VISOR GIS (Capas Base) ---
with tab2:
    st.markdown("#### Contexto Territorial")
    
    # Cargamos capas usando tu función optimizada
    layers = hydrogeo_utils.cargar_capas_gis_optimizadas(engine)
    
    # Mapa base centrado en la zona seleccionada
    m = folium.Map(location=[(miny+maxy)/2, (minx+maxx)/2], zoom_start=10, tiles="CartoDB positron")
    
    # Capa de la Zona Seleccionada (Borde rojo)
    folium.GeoJson(
        gdf_zona, 
        name="Zona Seleccionada",
        style_function=lambda x: {'color': 'red', 'fillColor': 'none', 'weight': 2}
    ).add_to(m)
    
    # Capas Contextuales
    if 'suelos' in layers:
        folium.GeoJson(layers['suelos'], name="Suelos", show=False,
                       style_function=lambda x: {'fillColor': 'orange', 'color': 'none', 'fillOpacity': 0.3},
                       tooltip=folium.GeoJsonTooltip(fields=['unidad_suelo', 'textura'])).add_to(m)
    
    if 'hidro' in layers:
        folium.GeoJson(layers['hidro'], name="Hidrogeología", show=True,
                       style_function=lambda x: {'fillColor': '#4287f5', 'color': '#4287f5', 'weight': 0.5, 'fillOpacity': 0.3},
                       tooltip=folium.GeoJsonTooltip(fields=['potencial', 'unidad_geo'])).add_to(m)
    
    if 'bocatomas' in layers:
        folium.GeoJson(layers['bocatomas'], name="Bocatomas",
                       marker=folium.CircleMarker(radius=3, color='black', fill=True),
                       tooltip=folium.GeoJsonTooltip(fields=['nombre'])).add_to(m)
    
    # Estaciones encontradas
    for _, row in df_puntos.iterrows():
        folium.Marker(
            [row['latitud'], row['longitud']], 
            popup=row['nom_est'],
            icon=folium.Icon(color="green", icon="tint", prefix='fa')
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=1400, height=600)

# --- TAB 3: INTERPOLACIÓN (Isolíneas) ---
with tab3:
    st.markdown("#### Superficie Interpolada de Recarga Media")
    
    if len(df_puntos) < 4:
        st.warning("⚠️ Se necesitan al menos 4 estaciones en la zona para generar isolíneas fiables.")
    else:
        # 1. Calcular Recarga puntual media por estación
        q_avg_est = f"""
        SELECT id_estacion_fk as id_estacion, AVG(precipitation) as p_media 
        FROM precipitacion_mensual 
        WHERE id_estacion_fk IN {ids_sql} 
        GROUP BY 1
        """
        df_avg_est = pd.read_sql(q_avg_est, engine)
        df_mapa = pd.merge(df_puntos, df_avg_est, on='id_estacion')
        
        # Aplicar Turc a cada punto (vectorizado)
        df_mapa['alt_est'] = df_mapa['alt_est'].fillna(alt_calc)
        t_est = 30 - (0.0065 * df_mapa['alt_est'])
        l_est = 300 + 25*t_est + 0.05*(t_est**3)
        # ETR media mensual estimada
        etr_est = df_mapa['p_media'] / np.sqrt(0.9 + (df_mapa['p_media'] / (l_est/12))**2)
        df_mapa['recarga_val'] = (df_mapa['p_media'] - etr_est).clip(lower=0) * ki_ponderado
        
        # 2. Interpolación
        x = df_mapa['longitud'].values
        y = df_mapa['latitud'].values
        z = df_mapa['recarga_val'].values
        
        # Grid
        pad = 0.02
        xi = np.linspace(minx-pad, maxx+pad, 100)
        yi = np.linspace(miny-pad, maxy+pad, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        Zi = griddata((x, y), z, (Xi, Yi), method='linear')
        
        # Relleno de bordes (Nearest)
        mask_nan = np.isnan(Zi)
        if np.any(mask_nan):
            Zi_nearest = griddata((x, y), z, (Xi, Yi), method='nearest')
            Zi[mask_nan] = Zi_nearest[mask_nan]
        
        # Mapa Isolíneas
        m_iso = folium.Map(location=[(miny+maxy)/2, (minx+maxx)/2], zoom_start=11, tiles="CartoDB dark_matter")
        
        # Colormap
        vmin, vmax = z.min(), z.max()
        if vmin == vmax: vmax += 0.1
        cmap = LinearColormap(['#ffffcc', '#a1dab4', '#41b6c4', '#225ea8'], vmin=vmin, vmax=vmax, caption="Recarga (mm/mes)")
        m_iso.add_child(cmap)
        
        folium.raster_layers.ImageOverlay(
            image=Zi[::-1],
            bounds=[[yi.min(), xi.min()], [yi.max(), xi.max()]],
            opacity=0.7,
            colormap=lambda x: cmap(x)
        ).add_to(m_iso)
        
        # Puntos con valor
        for _, row in df_mapa.iterrows():
            folium.CircleMarker(
                [row['latitud'], row['longitud']], radius=3, color='white', fill=True, fill_color='black',
                tooltip=f"{row['nom_est']}: {row['recarga_val']:.1f} mm"
            ).add_to(m_iso)
            
        st_folium(m_iso, width=1400, height=600)

# --- TAB 4: DESCARGAS ---
with tab4:
    st.subheader("💾 Centro de Datos")
    c1, c2 = st.columns(2)
    
    # CSV
    c1.download_button("📥 Descargar Serie (CSV)", df_proyeccion.to_csv(index=False), "serie_recarga.csv")
    
    # Raster (si existe)
    if 'Zi' in locals():
        try:
            tif_bytes = hydrogeo_utils.generar_geotiff(Zi[::-1], [xi.min(), yi.min(), xi.max(), yi.max()])
            c2.download_button("📥 Descargar Raster (GeoTIFF)", tif_bytes, "mapa_recarga.tif")
        except Exception as e:
            st.warning(f"Raster no disponible: {e}")