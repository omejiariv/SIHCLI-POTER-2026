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
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Importaciones locales
from modules import db_manager, hydrogeo_utils, selectors

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

# ==============================================================================
# 1. SELECTOR ESPACIAL Y FILTROS
# ==============================================================================
# Recuperamos la selección inicial
ids_estaciones_inicial, nombre_zona, altitud_ref, gdf_zona = selectors.render_selector_espacial()

st.sidebar.divider()
st.sidebar.header("📍 Estaciones")

engine = db_manager.get_engine()

# --- LÓGICA DE BÚSQUEDA DE ESTACIONES ---
# Si el selector no nos dio IDs (ej. búsqueda por polígono), buscamos espacialmente
df_puntos_base = pd.DataFrame()

if gdf_zona is not None:
    # Opción A: Tenemos un polígono (Cuenca o Municipio)
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    # Buscamos en el rectángulo primero (rápido)
    q_geo = f"""
    SELECT id_estacion, nom_est, latitud, longitud, alt_est 
    FROM estaciones 
    WHERE longitud BETWEEN {minx} AND {maxx} 
      AND latitud BETWEEN {miny} AND {maxy}
    """
    df_puntos_base = pd.read_sql(q_geo, engine)
    
    # Si tenemos el polígono exacto (no solo el box), filtramos fino aquí si fuera necesario
    # (Por ahora confiamos en el selector manual abajo para corregir excesos del Bounding Box)

elif ids_estaciones_inicial:
    # Opción B: El selector nos dio IDs específicos
    ids_sql = str(tuple(ids_estaciones_inicial)).replace(",)", ")")
    q_ids = f"SELECT id_estacion, nom_est, latitud, longitud, alt_est FROM estaciones WHERE id_estacion IN {ids_sql}"
    df_puntos_base = pd.read_sql(q_ids, engine)

# --- SELECTOR DE ESTACIONES (CASCADA) ---
if not df_puntos_base.empty:
    # Creamos lista de nombres para el selector
    opciones_estaciones = df_puntos_base['nom_est'].tolist()
    ids_map = dict(zip(df_puntos_base['nom_est'], df_puntos_base['id_estacion']))
    
    # Por defecto seleccionamos todas las encontradas
    seleccion_usuario = st.sidebar.multiselect(
        "Estaciones a incluir:",
        options=opciones_estaciones,
        default=opciones_estaciones
    )
    
    if not seleccion_usuario:
        st.warning("⚠️ Debe seleccionar al menos una estación.")
        st.stop()
        
    # Filtramos el DF base con la selección del usuario
    ids_finales = [ids_map[nom] for nom in seleccion_usuario]
    df_puntos = df_puntos_base[df_puntos_base['id_estacion'].isin(ids_finales)].copy()
else:
    st.warning("⚠️ No se encontraron estaciones en la zona buscada.")
    st.stop()

# ==============================================================================
# 2. PARÁMETROS DEL MODELO
# ==============================================================================
st.sidebar.divider()
st.sidebar.header("⚙️ Parámetros del Modelo")

col_s1, col_s2 = st.sidebar.columns(2)
pct_bosque = col_s1.number_input("% Bosque", 0, 100, 50)
pct_cultivo = col_s2.number_input("% Agrícola", 0, 100, 30)
pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
st.sidebar.metric("% Urbano/Otro", f"{pct_urbano}%")

ki_ponderado = ((pct_bosque * 0.50) + (pct_cultivo * 0.30) + (pct_urbano * 0.10)) / 100.0
st.sidebar.metric("Coef. Infiltración (Ki)", f"{ki_ponderado:.2f}")

st.sidebar.divider()
meses_futuros = st.sidebar.slider("Horizonte Pronóstico", 12, 60, 24)
ruido = st.sidebar.slider("Incertidumbre", 0.0, 1.0, 0.1)

# Inicializamos variables para exportación
Zi = None
xi, yi = None, None

# ==============================================================================
# 3. EJECUCIÓN (LÓGICA)
# ==============================================================================

# 1. Obtener datos de lluvia
ids_sql_final = str(tuple(ids_finales)).replace(",)", ")")
q_serie = f"""
SELECT fecha_mes_año, precipitation 
FROM precipitacion_mensual 
WHERE id_estacion_fk IN {ids_sql_final}
"""
df_raw = pd.read_sql(q_serie, engine)

if df_raw.empty:
    st.error("⚠️ Las estaciones seleccionadas no tienen datos históricos.")
    st.stop()

# 2. Ejecutar Modelo Prophet
alt_calc = altitud_ref if altitud_ref else df_puntos['alt_est'].mean()

with st.spinner("Calculando balance hídrico y proyecciones..."):
    df_res = hydrogeo_utils.ejecutar_pronostico_prophet(
        df_raw, meses_futuros, alt_calc, ki_ponderado, ruido
    )

if df_res is None or df_res.empty or 'tipo' not in df_res.columns:
    st.error("⚠️ Error al calcular el modelo.")
    st.stop()

# ==============================================================================
# 4. VISUALIZACIÓN
# ==============================================================================
st.markdown(f"### Análisis: {nombre_zona}")

# KPIs
df_hist = df_res[df_res['tipo'] == 'Histórico']
if not df_hist.empty:
    c1, c2, c3, c4 = st.columns(4)
    if 'p_final' in df_hist.columns:
        c1.metric("Lluvia Media", f"{df_hist['p_final'].mean()*12:,.0f} mm/año")
    if 'etr_mm' in df_hist.columns:
        c2.metric("ETR (Turc)", f"{df_hist['etr_mm'].mean()*12:,.0f} mm/año")
    if 'recarga_mm' in df_hist.columns:
        c3.metric("Recarga Potencial", f"{df_hist['recarga_mm'].mean()*12:,.0f} mm/año")
    c4.metric("Estaciones", len(df_puntos))

# PESTAÑAS
tab1, tab2, tab3, tab4 = st.tabs(["📈 Serie & Pronóstico", "🗺️ Mapa Contexto", "🌈 Interpolación", "📥 Descargas"])

# --- TAB 1: GRÁFICO ---
with tab1:
    fig = go.Figure()
    if not df_hist.empty:
        fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['p_final'], name='Lluvia Histórica', line=dict(color='gray', width=1)))
        fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['recarga_mm'], name='Recarga Histórica', line=dict(color='blue', width=2), fill='tozeroy'))
    
    df_fut = df_res[df_res['tipo'] == 'Proyección']
    if not df_fut.empty:
        fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['recarga_mm'], name='Recarga Futura', line=dict(color='cyan', width=2, dash='dot')))
        if 'yhat_upper' in df_fut.columns:
            fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_upper'], showlegend=False, line=dict(width=0)))
            fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_lower'], name='Incertidumbre', fill='tonexty', line=dict(width=0), fillcolor='rgba(0,255,255,0.1)'))
    
    fig.update_layout(height=450, hovermode="x unified", title="Dinámica de Recarga")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: MAPA CONTEXTO ---
with tab2:
    # Definir límites: Prioridad al polígono de la zona, sino usas las estaciones
    bounds = None
    if gdf_zona is not None:
        bounds = gdf_zona.total_bounds
    elif not df_puntos.empty:
        bounds = df_puntos.total_bounds # GeoDataFrame property si fuera GDF, pero es DF
        # Calculamos bounds manuales del DF de puntos
        bounds = [
            df_puntos['longitud'].min(), df_puntos['latitud'].min(),
            df_puntos['longitud'].max(), df_puntos['latitud'].max()
        ]

    # Cargar capas recortadas
    with st.spinner("Cargando capas geográficas..."):
        layers = hydrogeo_utils.cargar_capas_gis_optimizadas(engine, bounds)
    
    # Centro del mapa
    mean_lat = df_puntos['latitud'].mean()
    mean_lon = df_puntos['longitud'].mean()
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=11, tiles="CartoDB positron")
    
    # Renderizar capas
    if 'suelos' in layers: 
        folium.GeoJson(layers['suelos'], name="Suelos", 
                       style_function=lambda x: {'color': 'green', 'weight': 0.5, 'fillOpacity': 0.1},
                       tooltip=folium.GeoJsonTooltip(fields=['codigo'], aliases=['Suelo:'])).add_to(m)
    
    if 'hidro' in layers: 
        folium.GeoJson(layers['hidro'], name="Hidrogeología", 
                       style_function=lambda x: {'color': 'blue', 'weight': 0.5, 'fillOpacity': 0.1}).add_to(m)
    
    if 'bocatomas' in layers: 
        folium.GeoJson(layers['bocatomas'], name="Bocatomas", 
                       marker=folium.CircleMarker(radius=3, color='red', fill=True)).add_to(m)
        
    # Pintar estaciones seleccionadas
    for _, r in df_puntos.iterrows():
        folium.Marker(
            [r['latitud'], r['longitud']], 
            popup=f"Estación: {r['nom_est']}", 
            icon=folium.Icon(color='black', icon='tint')
        ).add_to(m)
        
    folium.LayerControl().add_to(m)
    st_folium(m, width=1400, height=600)

# --- TAB 3: INTERPOLACIÓN ---
with tab3:
    if len(df_puntos) < 4:
        st.warning(f"ℹ️ Se requieren al menos 4 estaciones para generar el mapa de calor (Interpolación). Actualmente seleccionadas: {len(df_puntos)}.")
        st.info("El análisis puntual y el pronóstico siguen siendo válidos.")
    else:
        # Lógica de interpolación existente
        q_ind = f"SELECT id_estacion_fk, AVG(precipitation) as p_media FROM precipitacion_mensual WHERE id_estacion_fk IN {ids_sql_final} GROUP BY 1"
        df_ind = pd.read_sql(q_ind, engine)
        df_mapa = pd.merge(df_puntos, df_ind, left_on='id_estacion', right_on='id_estacion_fk')
        
        t_est = 30 - (0.0065 * df_mapa['alt_est'].fillna(alt_calc))
        l_t = 300 + 25*t_est + 0.05*(t_est**3)
        etr = df_mapa['p_media'] / np.sqrt(0.9 + (df_mapa['p_media']/(l_t/12))**2)
        df_mapa['z'] = (df_mapa['p_media'] - etr).clip(lower=0) * ki_ponderado
        
        x, y, z = df_mapa['longitud'].values, df_mapa['latitud'].values, df_mapa['z'].values
        pad = 0.05
        xi = np.linspace(x.min()-pad, x.max()+pad, 100)
        yi = np.linspace(y.min()-pad, y.max()+pad, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        Zi = griddata((x, y), z, (Xi, Yi), method='linear')
        mask_nan = np.isnan(Zi)
        if np.any(mask_nan):
            Zi_n = griddata((x, y), z, (Xi, Yi), method='nearest')
            Zi[mask_nan] = Zi_n[mask_nan]
        
        vmin, vmax = np.nanmin(Zi), np.nanmax(Zi)
        if vmin == vmax: vmax += 0.1
        Zi_norm = (Zi - vmin) / (vmax - vmin)
        
        try:
            cmap_mpl = plt.colormaps['viridis']
        except:
            cmap_mpl = cm.get_cmap('viridis')
            
        rgba = cmap_mpl(Zi_norm)
        rgba[np.isnan(Zi), 3] = 0
        
        m_iso = folium.Map(location=[mean_lat, mean_lon], zoom_start=11, tiles="CartoDB dark_matter")
        folium.raster_layers.ImageOverlay(image=rgba, bounds=[[yi.min(), xi.min()], [yi.max(), xi.max()]], opacity=0.8, origin='lower').add_to(m_iso)
        
        # Puntos encima
        for _, r in df_mapa.iterrows():
             folium.CircleMarker([r['latitud'], r['longitud']], radius=3, color='white', fill=True, popup=f"{r['z']:.1f} mm").add_to(m_iso)

        m_iso.add_child(LinearColormap(['#440154', '#21918c', '#fde725'], vmin=vmin, vmax=vmax, caption="Recarga (mm/mes)"))
        st_folium(m_iso, width=1400, height=600)

# --- TAB 4: DESCARGAS ---
with tab4:
    c1, c2 = st.columns(2)
    c1.download_button("Descargar CSV", df_res.to_csv(index=False), "recarga.csv")
    
    if Zi is not None:
        try:
            tif = hydrogeo_utils.generar_geotiff(Zi[::-1], [xi.min(), yi.min(), xi.max(), yi.max()])
            c2.download_button("Descargar Raster TIFF", tif, "mapa_recarga.tif")
        except:
            c2.warning("Raster no generado.")
    else:
        c2.info("Raster no disponible (requiere interpolación).")