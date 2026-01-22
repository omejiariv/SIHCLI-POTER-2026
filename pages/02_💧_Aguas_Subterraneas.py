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

# --- CSS PERSONALIZADO ---
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("💧 Sistema de Inteligencia: Aguas Subterráneas")
st.markdown("---")

# ==============================================================================
# 1. SELECTOR ESPACIAL (Tu módulo)
# ==============================================================================
# Esto reemplaza todo el código manual de sidebar anterior
ids_estaciones, nombre_zona, altitud_ref, gdf_zona = selectors.render_selector_espacial()

if gdf_zona is None:
    st.info("👈 Por favor, selecciona una Cuenca o Municipio en el menú lateral para comenzar.")
    st.stop()

# ==============================================================================
# 2. CONFIGURACIÓN LATERAL ADICIONAL
# ==============================================================================
with st.sidebar:
    st.divider()
    st.header("⚙️ Parámetros del Modelo")
    
    # Tu excelente idea de sliders para Ki ponderado
    col_s1, col_s2 = st.columns(2)
    pct_bosque = col_s1.number_input("% Bosque", 0, 100, 50)
    pct_cultivo = col_s2.number_input("% Agrícola", 0, 100, 30)
    pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
    st.caption(f"Calculado: % Urbano/Otros: {pct_urbano}%")
    
    # Coeficiente Ki Ponderado
    ki_ponderado = ((pct_bosque * 0.50) + (pct_cultivo * 0.30) + (pct_urbano * 0.10)) / 100.0
    st.metric("Coef. Infiltración (Ki)", f"{ki_ponderado:.2f}")
    
    st.divider()
    st.subheader("🔮 Pronóstico")
    meses_futuros = st.slider("Horizonte (Meses)", 12, 60, 24)
    ruido = st.slider("Factor Incertidumbre", 0.0, 1.0, 0.1)

# ==============================================================================
# 3. CARGA DE DATOS
# ==============================================================================
engine = db_manager.get_engine()
if not engine:
    st.error("Error de conexión.")
    st.stop()

# Convertir lista de IDs a formato SQL
if not ids_estaciones:
    st.warning("La zona seleccionada no tiene estaciones activas.")
    st.stop()

ids_sql = tuple(ids_estaciones)
if len(ids_sql) == 1: ids_sql = f"('{ids_estaciones[0]}')" # Fix para tupla de 1 elemento

# A. Datos de Lluvia Histórica (Promedio de la zona)
q_serie = f"""
SELECT fecha_mes_año as fecha, AVG(precipitation) as precipitation 
FROM precipitacion_mensual 
WHERE id_estacion_fk IN {ids_sql} 
GROUP BY fecha_mes_año 
ORDER BY fecha_mes_año
"""
df_serie = pd.read_sql(q_serie, engine)

# B. Datos Geoespaciales de las Estaciones (Para el mapa)
q_puntos = f"""
SELECT id_estacion, nom_est, latitud, longitud, alt_est 
FROM estaciones 
WHERE id_estacion IN {ids_sql}
"""
df_puntos = pd.read_sql(q_puntos, engine)

# ==============================================================================
# 4. CÁLCULOS (Usando el módulo híbrido)
# ==============================================================================

# Calcular Proyección Híbrida (Histórico + Futuro)
df_proyeccion = hydrogeo_utils.ejecutar_pronostico_prophet(
    df_serie, meses_futuros, altitud_ref, ki_ponderado, ruido
)

# KPIs Generales (Promedios Anuales)
if not df_proyeccion.empty:
    df_hist = df_proyeccion[df_proyeccion['tipo'] == 'Histórico']
    p_anual = df_hist['p_final'].mean() * 12
    recarga_anual = df_hist['recarga_mm'].mean() * 12
    etr_anual = df_hist['etr_mm'].mean() * 12
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precipitación Media", f"{p_anual:,.0f} mm/año")
    c2.metric("ETR (Turc)", f"{etr_anual:,.0f} mm/año")
    c3.metric("Recarga Potencial", f"{recarga_anual:,.0f} mm/año", delta="Acuífero")
    c4.metric("Estaciones en Zona", len(ids_estaciones))

# ==============================================================================
# 5. VISUALIZACIÓN (TABS)
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis & Pronóstico", "🗺️ Mapa Interactivo", "🌈 Superficie Recarga", "📥 Descargas"])

# --- TAB 1: GRÁFICO ---
with tab1:
    if not df_proyeccion.empty:
        fig = go.Figure()
        
        # Histórico
        hist = df_proyeccion[df_proyeccion['tipo'] == 'Histórico']
        fig.add_trace(go.Scatter(x=hist['fecha'], y=hist['p_final'], name='Lluvia Histórica', line=dict(color='gray', width=1)))
        fig.add_trace(go.Scatter(x=hist['fecha'], y=hist['recarga_mm'], name='Recarga Histórica', line=dict(color='blue', width=2), fill='tozeroy'))
        
        # Futuro
        fut = df_proyeccion[df_proyeccion['tipo'] == 'Proyección']
        fig.add_trace(go.Scatter(x=fut['fecha'], y=fut['p_final'], name='Lluvia Proyectada', line=dict(color='silver', dash='dot')))
        fig.add_trace(go.Scatter(x=fut['fecha'], y=fut['recarga_mm'], name='Recarga Futura', line=dict(color='cyan', width=2, dash='solid')))
        
        # Bandas
        fig.add_trace(go.Scatter(x=fut['fecha'], y=fut['yhat_upper'], showlegend=False, line=dict(width=0)))
        fig.add_trace(go.Scatter(x=fut['fecha'], y=fut['yhat_lower'], fill='tonexty', fillcolor='rgba(0,255,255,0.1)', name='Incertidumbre', line=dict(width=0)))
        
        fig.update_layout(title="Dinámica de Recarga (Histórico + IA)", yaxis_title="mm/mes", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: VISOR GIS (Tu código optimizado) ---
with tab2:
    st.markdown("#### Contexto Hidrogeológico y Bocatomas")
    layers = hydrogeo_utils.cargar_capas_gis_optimizadas(engine)
    
    # Centro del mapa
    mean_lat = df_puntos['latitud'].mean()
    mean_lon = df_puntos['longitud'].mean()
    
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=11, tiles="CartoDB positron")
    
    # Capas
    if 'suelos' in layers:
        folium.GeoJson(layers['suelos'], name="Suelos", 
                       style_function=lambda x: {'fillColor': 'green', 'color': 'none', 'fillOpacity': 0.1},
                       tooltip=folium.GeoJsonTooltip(fields=['unidad_suelo', 'textura'])).add_to(m)
                       
    if 'hidro' in layers:
        folium.GeoJson(layers['hidro'], name="Hidrogeología", 
                       style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'fillOpacity': 0.1},
                       tooltip=folium.GeoJsonTooltip(fields=['potencial', 'unidad_geo'])).add_to(m)
                       
    if 'bocatomas' in layers:
        folium.GeoJson(layers['bocatomas'], name="Bocatomas",
                       marker=folium.CircleMarker(radius=4, color='red', fill=True),
                       tooltip=folium.GeoJsonTooltip(fields=['nombre', 'municipio'])).add_to(m)
    
    folium.LayerControl().add_to(m)
    st_folium(m, width=1400, height=600)

# --- TAB 3: INTERPOLACIÓN (Tu código de isolíneas mejorado) ---
with tab3:
    st.markdown("#### Superficie Interpolada de Recarga (mm/mes promedio)")
    
    if len(df_puntos) < 4:
        st.warning("Se necesitan al menos 4 estaciones en la zona para interpolar.")
    else:
        # Calcular Recarga Promedio por Estación para el mapa
        # Usamos la misma lógica de Turc pero por estación
        # 1. Traer lluvia media por estación
        q_avg = f"SELECT id_estacion_fk as id_estacion, AVG(precipitation) as p_media FROM precipitacion_mensual WHERE id_estacion_fk IN {ids_sql} GROUP BY 1"
        df_avg = pd.read_sql(q_avg, engine)
        df_mapa = pd.merge(df_puntos, df_avg, on='id_estacion')
        
        # Calcular Recarga puntual
        temp_est = 30 - (0.0065 * df_mapa['alt_est'])
        l_t = 300 + 25*temp_est + 0.05*(temp_est**3)
        etr = df_mapa['p_media'] / np.sqrt(0.9 + (df_mapa['p_media'] / (l_t/12))**2)
        df_mapa['recarga_val'] = (df_mapa['p_media'] - etr).clip(lower=0) * ki_ponderado
        
        # Grid Data
        x = df_mapa['longitud'].values
        y = df_mapa['latitud'].values
        z = df_mapa['recarga_val'].values
        
        # Crear Grid
        pad = 0.05
        xi = np.linspace(x.min()-pad, x.max()+pad, 100)
        yi = np.linspace(y.min()-pad, y.max()+pad, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolar (Linear con relleno Nearest para bordes)
        Zi = griddata((x, y), z, (Xi, Yi), method='linear')
        mask_nan = np.isnan(Zi)
        if np.any(mask_nan):
            Zi_nearest = griddata((x, y), z, (Xi, Yi), method='nearest')
            Zi[mask_nan] = Zi_nearest[mask_nan]
            
        # Mapa Folium
        m_iso = folium.Map(location=[mean_lat, mean_lon], zoom_start=11, tiles="CartoDB dark_matter")
        
        # Colormap
        vmin, vmax = z.min(), z.max()
        if vmin == vmax: vmax += 0.1
        cmap = LinearColormap(['#ffffcc', '#a1dab4', '#41b6c4', '#225ea8'], vmin=vmin, vmax=vmax, caption="Recarga (mm/mes)")
        m_iso.add_child(cmap)
        
        # Overlay Imagen
        folium.raster_layers.ImageOverlay(
            image=Zi[::-1], # Flip vertical para folium
            bounds=[[yi.min(), xi.min()], [yi.max(), xi.max()]],
            opacity=0.7,
            colormap=lambda x: cmap(x)
        ).add_to(m_iso)
        
        # Puntos
        for _, row in df_mapa.iterrows():
            folium.CircleMarker(
                [row['latitud'], row['longitud']], radius=3, color='white', fill=True,
                tooltip=f"{row['nom_est']}: {row['recarga_val']:.1f} mm"
            ).add_to(m_iso)
            
        st_folium(m_iso, width=1400, height=600)

# --- TAB 4: DESCARGAS ---
with tab4:
    c1, c2 = st.columns(2)
    c1.download_button("📥 Descargar Serie (CSV)", df_proyeccion.to_csv(index=False), "serie_recarga.csv")
    
    if 'Zi' in locals():
        tif_bytes = hydrogeo_utils.generar_geotiff(Zi[::-1], [xi.min(), yi.min(), xi.max(), yi.max()])
        c2.download_button("📥 Descargar Raster (GeoTIFF)", tif_bytes, "mapa_recarga.tif")