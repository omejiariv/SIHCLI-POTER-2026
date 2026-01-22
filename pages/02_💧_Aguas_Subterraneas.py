# pages/02_💧_Aguas_Subterraneas.py

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

# --- 1. SELECTOR ESPACIAL ---
ids_estaciones, nombre_zona, altitud_ref, gdf_zona = selectors.render_selector_espacial()

# --- 2. PARÁMETROS ---
st.sidebar.divider()
st.sidebar.header("⚙️ Parámetros")

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

# Inicializamos variables de exportación para evitar errores si no se generan
Zi = None
xi, yi = None, None

# --- 3. LÓGICA ---
if gdf_zona is not None:
    engine = db_manager.get_engine()
    
    # Fallback si no hay IDs directos (Buscar por Buffer)
    if not ids_estaciones:
        minx, miny, maxx, maxy = gdf_zona.total_bounds
        buff = 0.1 
        q_geo = f"""
        SELECT id_estacion, nom_est, latitud, longitud, alt_est 
        FROM estaciones 
        WHERE longitud BETWEEN {minx-buff} AND {maxx+buff} 
          AND latitud BETWEEN {miny-buff} AND {maxy+buff}
        """
        df_puntos = pd.read_sql(q_geo, engine)
        if not df_puntos.empty:
            ids_estaciones = df_puntos['id_estacion'].tolist()
    else:
        ids_sql = str(tuple(ids_estaciones)).replace(",)", ")")
        df_puntos = pd.read_sql(f"SELECT id_estacion, nom_est, latitud, longitud, alt_est FROM estaciones WHERE id_estacion IN {ids_sql}", engine)

    if not ids_estaciones:
        st.warning("⚠️ No se encontraron estaciones dentro de la zona seleccionada.")
        st.stop()

    # Traer Serie Histórica
    ids_sql = str(tuple(ids_estaciones)).replace(",)", ")")
    q_serie = f"""
    SELECT fecha_mes_año, precipitation 
    FROM precipitacion_mensual 
    WHERE id_estacion_fk IN {ids_sql}
    """
    df_raw = pd.read_sql(q_serie, engine)
    
    if df_raw.empty:
        st.error("⚠️ Las estaciones encontradas no tienen datos históricos de precipitación.")
        st.stop()

    # --- EJECUCIÓN DEL MODELO ---
    alt_calc = altitud_ref if altitud_ref else df_puntos['alt_est'].mean()
    
    # Llamada al motor (Prophet)
    with st.spinner("Calculando balance hídrico y proyecciones..."):
        df_res = hydrogeo_utils.ejecutar_pronostico_prophet(
            df_raw, meses_futuros, alt_calc, ki_ponderado, ruido
        )

    # --- VERIFICACIÓN DE SEGURIDAD ROBUSTA ---
    # 1. Verificar si está vacío o es None
    if df_res is None or df_res.empty:
        st.warning("⚠️ El modelo no arrojó resultados válidos. Verifica la calidad de los datos históricos.")
        st.stop()

    # 2. Verificar estructura de columnas (Esto evita el KeyError: 'tipo')
    if 'tipo' not in df_res.columns:
        st.error(f"⚠️ Error estructural en el modelo. No se encontró la columna 'tipo'. Columnas recibidas: {list(df_res.columns)}")
        st.stop()

    # --- 4. VISUALIZACIÓN ---
    st.markdown(f"### Análisis: {nombre_zona}")
    
    # KPIs - Filtrado seguro
    df_hist = df_res[df_res['tipo'] == 'Histórico']
    
    if not df_hist.empty:
        c1, c2, c3, c4 = st.columns(4)
        
        # Validamos que existan las columnas numéricas antes de calcular medias
        if 'p_final' in df_hist.columns:
            c1.metric("Lluvia Media", f"{df_hist['p_final'].mean()*12:,.0f} mm/año")
        if 'etr_mm' in df_hist.columns:
            c2.metric("ETR (Turc)", f"{df_hist['etr_mm'].mean()*12:,.0f} mm/año")
        if 'recarga_mm' in df_hist.columns:
            c3.metric("Recarga Potencial", f"{df_hist['recarga_mm'].mean()*12:,.0f} mm/año")
        
        c4.metric("Estaciones", len(df_puntos))
    else:
        st.warning("El modelo corrió, pero no hay datos históricos clasificados.")

    # PESTAÑAS
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Serie & Pronóstico", "🗺️ Mapa Contexto", "🌈 Interpolación", "📥 Descargas"])

    # TAB 1: GRÁFICO
    with tab1:
        fig = go.Figure()
        
        if not df_hist.empty:
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['p_final'], name='Lluvia Histórica', line=dict(color='gray', width=1)))
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['recarga_mm'], name='Recarga Histórica', line=dict(color='blue', width=2), fill='tozeroy'))
        
        df_fut = df_res[df_res['tipo'] == 'Proyección']
        if not df_fut.empty:
            fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['recarga_mm'], name='Recarga Futura', line=dict(color='cyan', width=2, dash='dot')))
            
            # Bandas de incertidumbre (si existen las columnas)
            if 'yhat_upper' in df_fut.columns and 'yhat_lower' in df_fut.columns:
                fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_upper'], showlegend=False, line=dict(width=0)))
                fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_lower'], name='Incertidumbre', fill='tonexty', line=dict(width=0), fillcolor='rgba(0,255,255,0.1)'))
        
        fig.update_layout(height=450, hovermode="x unified", title="Dinámica de Recarga")
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2: MAPA BASE (Corregido para evitar ValueError)
    with tab2:
        layers = hydrogeo_utils.cargar_capas_gis_optimizadas(engine)
        mean_lat, mean_lon = df_puntos['latitud'].mean(), df_puntos['longitud'].mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=11, tiles="CartoDB positron")
        
        # CORRECCIÓN: Usamos 'in' para verificar existencia sin evaluar el DataFrame
        if 'suelos' in layers: 
            folium.GeoJson(layers['suelos'], name="Suelos", style_function=lambda x: {'color': 'green', 'weight': 0.5, 'fillOpacity': 0.1}).add_to(m)
        
        if 'hidro' in layers: 
            folium.GeoJson(layers['hidro'], name="Hidrogeología", style_function=lambda x: {'color': 'blue', 'weight': 0.5, 'fillOpacity': 0.1}).add_to(m)
        
        if 'bocatomas' in layers: 
            folium.GeoJson(layers['bocatomas'], name="Bocatomas", marker=folium.CircleMarker(radius=3, color='red')).add_to(m)
            
        for _, r in df_puntos.iterrows():
            folium.Marker([r['latitud'], r['longitud']], popup=r['nom_est'], icon=folium.Icon(color='black', icon='tint')).add_to(m)
            
        folium.LayerControl().add_to(m)
        st_folium(m, width=1400, height=600)

    # TAB 3: INTERPOLACIÓN
    with tab3:
        if len(df_puntos) < 4:
            st.warning("⚠️ Se requieren al menos 4 estaciones con datos para generar una interpolación espacial fiable.")
        else:
            q_ind = f"SELECT id_estacion_fk, AVG(precipitation) as p_media FROM precipitacion_mensual WHERE id_estacion_fk IN {ids_sql} GROUP BY 1"
            df_ind = pd.read_sql(q_ind, engine)
            df_mapa = pd.merge(df_puntos, df_ind, left_on='id_estacion', right_on='id_estacion_fk')
            
            # Modelo Turc Espacializado
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
            
            # Obtener colormap de manera segura
            try:
                cmap_mpl = plt.colormaps['viridis'] # Matplotlib moderno
            except:
                cmap_mpl = cm.get_cmap('viridis') # Fallback antiguo
                
            rgba = cmap_mpl(Zi_norm)
            rgba[np.isnan(Zi), 3] = 0
            
            m_iso = folium.Map(location=[mean_lat, mean_lon], zoom_start=11, tiles="CartoDB dark_matter")
            folium.raster_layers.ImageOverlay(image=rgba, bounds=[[yi.min(), xi.min()], [yi.max(), xi.max()]], opacity=0.8, origin='lower').add_to(m_iso)
            
            cmap_branca = LinearColormap(['#440154', '#21918c', '#fde725'], vmin=vmin, vmax=vmax, caption="Recarga (mm/mes)")
            m_iso.add_child(cmap_branca)
            
            st_folium(m_iso, width=1400, height=600)

    # TAB 4: DESCARGAS
    with tab4:
        c1, c2 = st.columns(2)
        c1.download_button("Descargar CSV (Serie Temporal)", df_res.to_csv(index=False), "recarga_temporal.csv")
        
        if Zi is not None and xi is not None and yi is not None:
            # Solo generamos el TIFF si la interpolación fue exitosa
            try:
                tif = hydrogeo_utils.generar_geotiff(Zi[::-1], [xi.min(), yi.min(), xi.max(), yi.max()])
                c2.download_button("Descargar Raster TIFF", tif, "recarga_mapa_interpolado.tif")
            except Exception as e:
                c2.warning(f"No se pudo generar el GeoTIFF: {e}")
        else:
            c2.info("El mapa raster no está disponible (se requieren > 4 estaciones para interpolar).")

else:
    st.info("👈 Seleccione una zona en el menú lateral para iniciar el análisis.")