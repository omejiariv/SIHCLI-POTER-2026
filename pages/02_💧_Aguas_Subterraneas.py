# pages/02_💧_Aguas_Subterraneas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
import geopandas as gpd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import folium
from folium.features import DivIcon # Para las etiquetas de texto
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Importaciones locales
from modules import db_manager, hydrogeo_utils, selectors

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

# ==============================================================================
# FUNCIONES AUXILIARES (CACHED)
# ==============================================================================
@st.cache_data(show_spinner=False)
def calcular_interpolacion_suave(df_puntos, df_lluvia_total, alt_ref, ki, _ids_validos):
    """
    Calcula la malla interpolada y suavizada para evitar recálculos al hacer zoom.
    """
    # 1. Calcular promedios por estación
    q_agrupada = df_lluvia_total.groupby('id_estacion_fk')['precipitation'].mean().reset_index()
    q_agrupada.columns = ['id_estacion', 'p_media']
    
    df_mapa = pd.merge(df_puntos, q_agrupada, on='id_estacion')
    
    if len(df_mapa) < 4:
        return None, None, None, None, None

    # 2. Balance Hídrico Puntual
    # Si alt_ref es None, usamos la altura de la estación
    alt_uso = df_mapa['alt_est'].fillna(df_mapa['alt_est'].mean())
    if alt_ref: alt_uso = alt_ref
        
    t_est = 30 - (0.0065 * alt_uso)
    l_t = 300 + 25*t_est + 0.05*(t_est**3)
    etr = df_mapa['p_media'] / np.sqrt(0.9 + (df_mapa['p_media']/(l_t/12))**2)
    df_mapa['recarga_calc'] = (df_mapa['p_media'] - etr).clip(lower=0) * ki
    
    # 3. Interpolación
    x, y, z = df_mapa['longitud'].values, df_mapa['latitud'].values, df_mapa['recarga_calc'].values
    
    pad = 0.05
    xi = np.linspace(x.min()-pad, x.max()+pad, 300) # Más resolución
    yi = np.linspace(y.min()-pad, y.max()+pad, 300)
    Xi, Yi = np.meshgrid(xi, yi)
    
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')
    
    # Rellenar NaNs con nearest
    if np.any(np.isnan(Zi)):
        Zi_n = griddata((x, y), z, (Xi, Yi), method='nearest')
        mask = np.isnan(Zi)
        Zi[mask] = Zi_n[mask]
    
    # 4. SUAVIZADO GAUSSIANO (Elimina picos y cruces de líneas)
    Zi_smooth = gaussian_filter(Zi, sigma=2.0)
    
    return Xi, Yi, Zi_smooth, df_mapa, (xi, yi)

# ==============================================================================
# 1. SELECTOR ESPACIAL
# ==============================================================================
ids_estaciones_inicial, nombre_zona, altitud_ref, gdf_zona = selectors.render_selector_espacial()

st.sidebar.divider()
st.sidebar.header("📍 Estaciones")

engine = db_manager.get_engine()
df_puntos_base = pd.DataFrame()

if gdf_zona is not None:
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    pad = 0.02
    q_geo = f"""
    SELECT id_estacion, nom_est, municipio, latitud, longitud, alt_est 
    FROM estaciones 
    WHERE longitud BETWEEN {minx-pad} AND {maxx+pad} 
      AND latitud BETWEEN {miny-pad} AND {maxy+pad}
    """
    df_puntos_base = pd.read_sql(q_geo, engine)
elif ids_estaciones_inicial:
    ids_sql = str(tuple(ids_estaciones_inicial)).replace(",)", ")")
    q_ids = f"SELECT id_estacion, nom_est, municipio, latitud, longitud, alt_est FROM estaciones WHERE id_estacion IN {ids_sql}"
    df_puntos_base = pd.read_sql(q_ids, engine)

if not df_puntos_base.empty:
    opciones = df_puntos_base['nom_est'].tolist()
    ids_map = dict(zip(df_puntos_base['nom_est'], df_puntos_base['id_estacion']))
    seleccion = st.sidebar.multiselect(f"Estaciones ({len(opciones)}):", opciones, default=opciones)
    
    if not seleccion:
        st.warning("⚠️ Selecciona al menos una estación.")
        st.stop()
        
    ids_finales = [ids_map[x] for x in seleccion]
    df_puntos = df_puntos_base[df_puntos_base['id_estacion'].isin(ids_finales)].copy()
else:
    st.warning("⚠️ Zona sin estaciones.")
    st.stop()

# ==============================================================================
# 2. PARÁMETROS
# ==============================================================================
st.sidebar.divider()
st.sidebar.header("⚙️ Modelo")
c1, c2 = st.sidebar.columns(2)
pct_bosque = c1.number_input("% Bosque", 0, 100, 50)
pct_cultivo = c2.number_input("% Agrícola", 0, 100, 30)
pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
st.sidebar.metric("% Urbano", f"{pct_urbano}%")
ki = ((pct_bosque * 0.50) + (pct_cultivo * 0.30) + (pct_urbano * 0.10)) / 100.0
st.sidebar.metric("Ki", f"{ki:.2f}")
meses = st.sidebar.slider("Pronóstico (Meses)", 12, 60, 24)

# ==============================================================================
# 3. DATOS Y CÁLCULOS
# ==============================================================================
ids_sql_final = str(tuple(ids_finales)).replace(",)", ")")
# Traemos id_estacion_fk para poder agrupar por estación después
q_serie = f"SELECT id_estacion_fk, fecha_mes_año, precipitation FROM precipitacion_mensual WHERE id_estacion_fk IN {ids_sql_final}"
df_raw = pd.read_sql(q_serie, engine)

if df_raw.empty:
    st.error("Sin datos históricos.")
    st.stop()

alt_calc = altitud_ref if altitud_ref else df_puntos['alt_est'].mean()

# Modelo Global (Prophet)
with st.spinner("Calculando modelo..."):
    df_res = hydrogeo_utils.ejecutar_pronostico_prophet(df_raw, meses, alt_calc, ki, ruido=0.1)

# Interpolación (Cacheada)
Xi, Yi, Zi, df_mapa_stats, grid_coords = calcular_interpolacion_suave(
    df_puntos, df_raw, altitud_ref, ki, ids_finales
)

# ==============================================================================
# 4. VISUALIZACIÓN
# ==============================================================================
st.markdown(f"### {nombre_zona}")

# KPIs
df_hist = df_res[df_res['tipo'] == 'Histórico']
if not df_hist.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lluvia Media", f"{df_hist['p_final'].mean()*12:,.0f} mm/año")
    c2.metric("ETR", f"{df_hist['etr_mm'].mean()*12:,.0f} mm/año")
    c3.metric("Recarga", f"{df_hist['recarga_mm'].mean()*12:,.0f} mm/año")
    c4.metric("Estaciones", len(df_puntos))

tab1, tab2, tab3, tab4 = st.tabs(["📈 Series y Pronóstico", "🗺️ Mapa Contexto", "🌈 Mapa Recarga", "📥 Descargas"])

# TAB 1: Serie Completa (Corregida)
with tab1:
    fig = go.Figure()
    if not df_hist.empty:
        # 1. Lluvia Histórica
        fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['p_final'], name='Lluvia Histórica', line=dict(color='gray', width=1)))
        # 2. ETR Histórica (RECUPERADA)
        if 'etr_mm' in df_hist.columns:
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['etr_mm'], name='ETR (Turc)', line=dict(color='green', width=1.5, dash='dash')))
        # 3. Recarga Histórica
        fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['recarga_mm'], name='Recarga Histórica', line=dict(color='blue', width=2), fill='tozeroy'))
    
    df_fut = df_res[df_res['tipo'] == 'Proyección']
    if not df_fut.empty:
        # 4. Lluvia Pronosticada (RECUPERADA) - Usamos 'yhat' que es la predicción cruda del modelo
        if 'yhat' in df_fut.columns:
             fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat'], name='Lluvia Pronosticada', line=dict(color='gray', width=1, dash='dot')))

        # 5. Recarga Futura
        fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['recarga_mm'], name='Recarga Futura', line=dict(color='cyan', width=2, dash='dot')))
        
        # 6. Incertidumbre
        if 'yhat_upper' in df_fut.columns:
            fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_upper'], showlegend=False, line=dict(width=0)))
            fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_lower'], name='Incertidumbre', fill='tonexty', line=dict(width=0), fillcolor='rgba(200,200,200,0.2)'))
            
    fig.update_layout(height=450, hovermode="x unified", title="Dinámica Hidrológica Completa (Lluvia, ETR, Recarga)")
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: Contexto (Capas + Popups Ricos - VERSIÓN MAESTRA)
with tab2:
    # 1. Definir caja geográfica (bounds) basada en estaciones
    # Agregamos un pequeño margen (pad) para que no quede muy apretado
    pad = 0.05
    bounds = [
        df_puntos['longitud'].min() - pad,
        df_puntos['latitud'].min() - pad,
        df_puntos['longitud'].max() + pad,
        df_puntos['latitud'].max() + pad
    ]
    
    # 2. Cargar capas usando el motor robusto (Compatible con lo que arreglamos hoy)
    layers = hydrogeo_utils.cargar_capas_gis_optimizadas(engine, bounds)
    
    # 3. Mapa base
    m = folium.Map(location=[df_puntos['latitud'].mean(), df_puntos['longitud'].mean()], zoom_start=11, tiles="CartoDB positron")
    
    # 4. Capas GIS (Si existen en la BD)
    if 'suelos' in layers: 
        folium.GeoJson(
            layers['suelos'], 
            name="Suelos", 
            style_function=lambda x: {'color': 'green', 'weight': 0.5, 'fillOpacity': 0.1},
            tooltip=folium.GeoJsonTooltip(fields=['info'], aliases=['Suelo:'])
        ).add_to(m)
        
    if 'hidro' in layers: 
        folium.GeoJson(
            layers['hidro'], 
            name="Hidrogeología", 
            style_function=lambda x: {'color': 'blue', 'weight': 0.5, 'fillOpacity': 0.1},
            tooltip=folium.GeoJsonTooltip(fields=['info'], aliases=['Potencial:'])
        ).add_to(m)
        
    if 'bocatomas' in layers: 
        folium.GeoJson(
            layers['bocatomas'], 
            name="Bocatomas", 
            marker=folium.CircleMarker(radius=3, color='red', fill_color='red'),
            tooltip=folium.GeoJsonTooltip(fields=['info'], aliases=['Bocatoma:'])
        ).add_to(m)
    
    # 5. Estaciones con POPUP ENRIQUECIDO (Tu lógica original recuperada)
    # Usamos df_mapa_stats si existe (tiene promedios calculados), sino df_puntos crudo
    df_display = df_mapa_stats if df_mapa_stats is not None else df_puntos
    
    for _, r in df_display.iterrows():
        # Lógica de presentación de datos
        p_val = f"{r['p_media']*12:,.0f}" if 'p_media' in r else "N/A"
        rec_val = f"{r['recarga_calc']*12:,.0f}" if 'recarga_calc' in r else "N/A"
        mun = r['municipio'] if 'municipio' in r else "N/D"
        alt = f"{r['alt_est']:.0f}"
        
        # HTML Estilizado
        html = f"""
        <div style="font-family:sans-serif; width:160px; font-size:12px;">
            <b style="font-size:13px;">{r['nom_est']}</b><br>
            <hr style="margin:5px 0; border: 0; border-top: 1px solid #ccc;">
            📍 Mun: {mun}<br>
            ⛰️ Alt: {alt} m<br>
            <span style="color:#555;">🌧️ Lluvia:</span> <b>{p_val}</b> mm/año<br>
            <span style="color:#0000AA;">💧 Recarga:</span> <b style="color:#0000AA; font-size:13px;">{rec_val}</b> mm/año
        </div>
        """
        
        iframe = folium.IFrame(html, width=180, height=150)
        popup = folium.Popup(iframe, max_width=180)
        
        folium.Marker(
            [r['latitud'], r['longitud']], 
            popup=popup, 
            icon=folium.Icon(color='black', icon='tint', prefix='fa'),
            tooltip=r['nom_est']
        ).add_to(m)
        
    folium.LayerControl().add_to(m)
    Fullscreen().add_to(m) # ✅ Fullscreen asegurado
    
    # ✅ CLAVE ÚNICA (key): Evita que el mapa desaparezca al cambiar filtros
    st_folium(m, width=1400, height=600, key=f"map_ctx_{nombre_zona}_{len(ids_finales)}")

# TAB 3: Interpolación (CORREGIDA)
with tab3:
    if Zi is None: st.warning("Requiere 4+ estaciones.")
    else:
        vmin, vmax = np.nanmin(Zi), np.nanmax(Zi)
        
        # ✅ CAMBIO: Usamos 'CartoDB positron' (Claro) para que el texto sea negro y legible
        m_iso = folium.Map(location=[df_puntos['latitud'].mean(), df_puntos['longitud'].mean()], zoom_start=11, tiles="CartoDB positron")
        
        # 1. Raster
        try: cmap = plt.colormaps['viridis']
        except: cmap = cm.get_cmap('viridis')
        rgba = cmap((Zi - vmin) / (vmax - vmin))
        rgba[np.isnan(Zi), 3] = 0
        folium.raster_layers.ImageOverlay(image=rgba, bounds=[[grid_coords[1].min(), grid_coords[0].min()], [grid_coords[1].max(), grid_coords[0].max()]], opacity=0.7, origin='lower').add_to(m_iso)
        
        # 2. Isolíneas LIMPIAS
        try:
            fig_c, ax_c = plt.subplots()
            cs = ax_c.contour(Xi, Yi, Zi, levels=12, colors='white', linewidths=0.8)
            plt.close(fig_c)
            
            for i, collection in enumerate(cs.allsegs):
                level_val = cs.levels[i]
                for segment in collection:
                    if len(segment) < 2: continue
                    lat_lon = [[y, x] for x, y in segment]
                    if not lat_lon: continue
                    
                    folium.PolyLine(lat_lon, color='white', weight=1, opacity=0.9, tooltip=f"{level_val:.0f} mm").add_to(m_iso)
                    
                    # Etiquetas Flotantes
                    if len(lat_lon) > 15: 
                        mid_pt = lat_lon[len(lat_lon) // 2]
                        # Estilo mejorado para contraste
                        icon_html = f"""
                        <div style="
                            font-size: 9pt; 
                            font-weight: bold;
                            color: black; 
                            background: rgba(255,255,255,0.7); 
                            padding: 1px 4px; 
                            border-radius: 4px;
                            border: 1px solid #666;
                            text-shadow: 0px 0px 2px white;
                        ">{level_val:.0f}</div>
                        """
                        folium.map.Marker(mid_pt, icon=DivIcon(icon_size=(30, 15), icon_anchor=(15, 7), html=icon_html)).add_to(m_iso)

        except Exception as e: st.warning(f"Error visual: {e}")
            
        m_iso.add_child(LinearColormap(['#440154', '#21918c', '#fde725'], vmin=vmin, vmax=vmax, caption="Recarga (mm/mes)"))
        Fullscreen().add_to(m_iso) # ✅ Fullscreen asegurado
        
        # ✅ CLAVE ÚNICA (key): Misma estrategia para evitar errores de carga
        st_folium(m_iso, width=1400, height=600, key=f"map_iso_{nombre_zona}_{len(ids_finales)}_{meses}")
        
        st.session_state['last_contour'] = cs

# TAB 4: Descargas
with tab4:
    c1, c2 = st.columns(2)
    c1.download_button("CSV Serie", df_res.to_csv(index=False), "recarga_serie.csv")
    if Zi is not None:
        try:
            tif = hydrogeo_utils.generar_geotiff(Zi[::-1], [grid_coords[0].min(), grid_coords[1].min(), grid_coords[0].max(), grid_coords[1].max()])
            c2.download_button("GeoTIFF Recarga", tif, "recarga_mapa.tif")
        except: pass