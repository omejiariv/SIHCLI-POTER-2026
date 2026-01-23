# pages/02_💧_Aguas_Subterraneas.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
import geopandas as gpd
from scipy.interpolate import griddata
import folium
from folium.features import DivIcon
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Importaciones locales
from modules import db_manager, hydrogeo_utils, selectors

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

# --- MANTENIMIENTO: BOTÓN DE LIMPIEZA ---
if st.sidebar.button("🧹 Limpiar Memoria y Recargar"):
    st.cache_data.clear()
    st.rerun()

# --- 1. SELECTOR ESPACIAL ---
ids_estaciones, nombre_zona, altitud_ref, gdf_zona = selectors.render_selector_espacial()
engine = db_manager.get_engine()

# --- 2. PARÁMETROS LATERALES ---
st.sidebar.divider()
st.sidebar.header("🎛️ Parámetros del Modelo")

col_s1, col_s2 = st.sidebar.columns(2)
pct_bosque = col_s1.number_input("% Bosque", 0, 100, 50)
pct_cultivo = col_s2.number_input("% Agrícola", 0, 100, 30)
pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
st.sidebar.metric("% Urbano/Otro", f"{pct_urbano}%")

ki_ponderado = ((pct_bosque * 0.50) + (pct_cultivo * 0.30) + (pct_urbano * 0.10)) / 100.0
st.sidebar.metric("Coef. Infiltración (Ki)", f"{ki_ponderado:.2f}")

st.sidebar.divider()
meses_futuros = st.sidebar.slider("Horizonte Pronóstico (meses)", 12, 60, 24)
ruido = st.sidebar.slider("Incertidumbre Estocástica", 0.0, 1.0, 0.1)

# Inicialización
Zi = None
xi, yi = None, None
grid_coords = None

# --- 3. LÓGICA DE DATOS ---
if gdf_zona is not None:
    
    # 3.1 Obtener Estaciones (Lógica de Buffer si no hay IDs)
    if not ids_estaciones:
        minx, miny, maxx, maxy = gdf_zona.total_bounds
        buff = 0.05
        q_geo = text(f"""
            SELECT id_estacion, nom_est, latitud, longitud, alt_est 
            FROM estaciones 
            WHERE longitud BETWEEN {minx-buff} AND {maxx+buff} 
            AND latitud BETWEEN {miny-buff} AND {maxy+buff}
        """)
        df_puntos = pd.read_sql(q_geo, engine)
        if not df_puntos.empty:
            # Filtro fino con polígono
            points = gpd.points_from_xy(df_puntos.longitud, df_puntos.latitud)
            gdf_pts = gpd.GeoDataFrame(df_puntos, geometry=points, crs="EPSG:4326")
            df_puntos = gpd.sjoin(gdf_pts, gdf_zona, how="inner", predicate="within")[df_puntos.columns]
            ids_estaciones = df_puntos['id_estacion'].tolist()
    else:
        # IDs directos
        if len(ids_estaciones) == 1:
             df_puntos = pd.read_sql(text(f"SELECT id_estacion, nom_est, latitud, longitud, alt_est FROM estaciones WHERE id_estacion = '{ids_estaciones[0]}'"), engine)
        else:
             df_puntos = pd.read_sql(text("SELECT id_estacion, nom_est, latitud, longitud, alt_est FROM estaciones WHERE id_estacion IN :ids"), engine, params={'ids': tuple(ids_estaciones)})

    if df_puntos.empty:
        st.warning("⚠️ No se encontraron estaciones en esta zona.")
        st.stop()

    # 3.2 CÁLCULO DE ESTADÍSTICAS (Optimizado con Caché)
    # Esto evita el loop de recarga y trae los datos para los popups
    with st.spinner("Procesando datos hidrológicos..."):
        df_mapa_stats = hydrogeo_utils.obtener_estadisticas_estaciones(engine, df_puntos)

    # 3.3 Traer Serie Histórica (Solo para el gráfico agregado)
    # Usamos tabla 'precipitacion_mensual' si existe, o agregamos desde 'precipitacion'
    # Por simplicidad, asumimos que existe una vista o tabla mensualizada
    try:
        if len(ids_estaciones) == 1:
             q_serie = text(f"SELECT fecha_mes_año, precipitation FROM precipitacion_mensual WHERE id_estacion_fk = '{ids_estaciones[0]}'")
        else:
             q_serie = text("SELECT fecha_mes_año, precipitation FROM precipitacion_mensual WHERE id_estacion_fk IN :ids")
        
        df_raw = pd.read_sql(q_serie, engine, params={'ids': tuple(ids_estaciones)})
    except:
        # Fallback si no hay tabla mensual, se podría agregar lógica aquí pero por ahora avisamos
        df_raw = pd.DataFrame()

    # 3.4 Ejecutar Modelo Prophet (Agregado)
    df_res = pd.DataFrame()
    if not df_raw.empty:
        alt_calc = altitud_ref if altitud_ref else df_puntos['alt_est'].mean()
        df_res = hydrogeo_utils.ejecutar_pronostico_prophet(df_raw, meses_futuros, alt_calc, ki_ponderado, ruido)

    # --- 4. INTERFAZ VISUAL ---
    st.markdown(f"### 📍 Análisis: {nombre_zona}")
    
    # KPIs Rápidos
    if not df_res.empty:
        df_hist = df_res[df_res['tipo'] == 'Histórico']
        if not df_hist.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Lluvia Media Zona", f"{df_hist['p_final'].mean()*12:,.0f} mm/año")
            c2.metric("ETR Estimada", f"{df_hist['etr_mm'].mean()*12:,.0f} mm/año")
            c3.metric("Recarga Potencial", f"{df_hist['recarga_mm'].mean()*12:,.0f} mm/año")
            c4.metric("Estaciones", len(df_puntos))
    
    # PESTAÑAS
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Serie & Pronóstico", "🗺️ Mapa Contexto", "🌈 Mapa Recarga", "📥 Descargas"])

    # --- TAB 1: GRÁFICO ---
    with tab1:
        if not df_res.empty:
            df_hist = df_res[df_res['tipo'] == 'Histórico']
            df_fut = df_res[df_res['tipo'] == 'Proyección']
            
            fig = go.Figure()
            # Histórico
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['p_final'], name='Lluvia Histórica', line=dict(color='gray', width=1)))
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['recarga_mm'], name='Recarga Histórica', line=dict(color='blue', width=2), fill='tozeroy'))
            # Futuro
            fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['recarga_mm'], name='Proyección Recarga', line=dict(color='cyan', width=2, dash='dot')))
            
            # Incertidumbre
            if 'yhat_upper' in df_fut.columns:
                fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_upper'], showlegend=False, line=dict(width=0)))
                fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_lower'], name='Rango Incertidumbre', fill='tonexty', line=dict(width=0), fillcolor='rgba(0,255,255,0.1)'))
            
            fig.update_layout(height=450, hovermode="x unified", title="Dinámica de Recarga Estimada")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos históricos suficientes para generar el pronóstico agregado.")

    # --- TAB 2: MAPA CONTEXTO (Robustez + Tooltips) ---
    with tab2:
        col_btn, _ = st.columns([1, 5])
        if col_btn.button("🔄 Recargar Mapa"):
            st.rerun()

        # Bounds
        pad = 0.05
        min_lon, min_lat = df_puntos['longitud'].min() - pad, df_puntos['latitud'].min() - pad
        max_lon, max_lat = df_puntos['longitud'].max() + pad, df_puntos['latitud'].max() + pad
        bounds_vals = [min_lon, min_lat, max_lon, max_lat]
        
        # Cargar Capas (Desde Utils Optimizado)
        layers = hydrogeo_utils.cargar_capas_gis_optimizadas(engine, bounds_vals)
        
        # Mapa
        m = folium.Map(location=[df_puntos['latitud'].mean(), df_puntos['longitud'].mean()], zoom_start=11, tiles="CartoDB positron")
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

        # Estilo CSS para tooltips largos
        st.markdown("<style>.leaflet-tooltip {white-space: normal !important; max-width: 250px !important;}</style>", unsafe_allow_html=True)

        # Función Helper para Tooltips
        def crear_tooltip(gdf, campos_dict):
            cols_reales = [c.lower().strip() for c in gdf.columns]
            fields, aliases = [], []
            for col, alias in campos_dict.items():
                if col.lower().strip() in cols_reales:
                    idx = cols_reales.index(col.lower().strip())
                    fields.append(gdf.columns[idx])
                    aliases.append(alias)
            return folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True, sticky=False) if fields else None

        # A. SUELOS
        if 'suelos' in layers:
            dict_suelos = {'ucs': 'UCS:', 'ucs_f': 'UCS:', 'litologia': 'Litología:', 'caracteri': 'Caract:', 'paisaje': 'Paisaje:', 'clima': 'Clima:'}
            folium.GeoJson(layers['suelos'], name="Suelos", 
                           style_function=lambda x: {'color': 'orange', 'weight': 0.5, 'fillOpacity': 0.2},
                           tooltip=crear_tooltip(layers['suelos'], dict_suelos)).add_to(m)

        # B. HIDROGEOLOGIA
        if 'hidro' in layers:
            dict_hidro = {'potencial': 'Potencial:', 'potencial_': 'Potencial:', 'unidad_geo': 'Unidad:', 'sigla': 'Sigla:'}
            folium.GeoJson(layers['hidro'], name="Hidrogeología", 
                           style_function=lambda x: {'color': 'blue', 'weight': 0.5, 'fillOpacity': 0.2},
                           tooltip=crear_tooltip(layers['hidro'], dict_hidro)).add_to(m)

        # C. BOCATOMAS
        if 'bocatomas' in layers:
            dict_boca = {'nom_bocatoma': 'Nombre:', 'nombre_acu': 'Acueducto:', 'municipio': 'Mun:'}
            folium.GeoJson(layers['bocatomas'], name="Bocatomas", 
                           marker=folium.CircleMarker(radius=4, color='red', fill_color='red'),
                           tooltip=crear_tooltip(layers['bocatomas'], dict_boca)).add_to(m)

        # D. ESTACIONES (Popup Completo)
        for _, r in df_mapa_stats.iterrows():
             # Valores anualizados
             p_val = r.get('p_media', 0) * 12
             etr_val = r.get('etr_media', 0) * 12
             rec_val = r.get('recarga_calc', 0) * 12
             
             html = f"""
             <div style='font-family:sans-serif; width:180px; font-size:12px;'>
                 <b>{r['nom_est']}</b><hr style='margin:3px 0;'>
                 🌧️ Lluvia: <b>{p_val:,.0f} mm</b><br>
                 ☀️ ETR: <b>{etr_val:,.0f} mm</b><br>
                 💧 Recarga: <b style='color:blue;'>{rec_val:,.0f} mm</b>
             </div>
             """
             folium.Marker([r['latitud'], r['longitud']], 
                           popup=folium.Popup(folium.IFrame(html, width=200, height=130)),
                           icon=folium.Icon(color='black', icon='tint'),
                           tooltip=r['nom_est']).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, width=1400, height=600, key=f"map_ctx_{nombre_zona}_{len(ids_estaciones)}")

    # --- TAB 3: INTERPOLACIÓN (Mapa Recarga) ---
    with tab3:
        if len(df_puntos) < 4:
            st.warning("⚠️ Se requieren al menos 4 estaciones para interpolar.")
        else:
            # Interpolación usando df_mapa_stats que ya tiene 'p_media'
            x = df_mapa_stats['longitud'].values
            y = df_mapa_stats['latitud'].values
            z = df_mapa_stats['p_media'].values # Usamos lluvia media para interpolar patrón espacial
            
            # Modelo Turc Espacial
            # Recalculamos Z como Recarga
            # Nota: df_mapa_stats ya tiene 'recarga_calc', usémoslo directamente si es posible
            z_recarga = df_mapa_stats['recarga_calc'].values * 12 # Anual
            
            # Grid
            pad = 0.05
            xi = np.linspace(min_lon, max_lon, 100)
            yi = np.linspace(min_lat, max_lat, 100)
            Xi, Yi = np.meshgrid(xi, yi)
            
            Zi = griddata((x, y), z_recarga, (Xi, Yi), method='linear')
            grid_coords = np.array([xi, yi]) # Guardar para exportar
            
            # Mapa
            m_iso = folium.Map(location=[df_puntos['latitud'].mean(), df_puntos['longitud'].mean()], zoom_start=11, tiles="CartoDB positron")
            m_iso.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
            
            # Colorear
            vmin, vmax = np.nanmin(Zi), np.nanmax(Zi)
            try: cmap = plt.colormaps['viridis']
            except: cmap = cm.get_cmap('viridis')
            
            rgba = cmap((Zi - vmin) / (vmax - vmin))
            rgba[np.isnan(Zi), 3] = 0
            
            folium.raster_layers.ImageOverlay(image=rgba, bounds=[[yi.min(), xi.min()], [yi.max(), xi.max()]], opacity=0.7, origin='lower').add_to(m_iso)
            
            # Isolíneas
            fig_c, ax_c = plt.subplots()
            cs = ax_c.contour(Xi, Yi, Zi, levels=10, colors='white', linewidths=0.5)
            plt.close(fig_c)
            
            for i, collection in enumerate(cs.allsegs):
                level_val = cs.levels[i]
                for segment in collection:
                    if len(segment) > 2:
                        folium.PolyLine([[pt[1], pt[0]] for pt in segment], color='white', weight=1, tooltip=f"{level_val:.0f}").add_to(m_iso)

            m_iso.add_child(LinearColormap(['#440154', '#21918c', '#fde725'], vmin=vmin, vmax=vmax, caption="Recarga (mm/año)"))
            st_folium(m_iso, width=1400, height=600, key=f"map_iso_{nombre_zona}")

    # --- TAB 4: DESCARGAS ---
    with tab4:
        c1, c2 = st.columns(2)
        if not df_res.empty:
            c1.download_button("Descargar CSV (Serie)", df_res.to_csv(index=False), "serie_temporal.csv")
        
        if Zi is not None:
            try:
                tif_bytes = hydrogeo_utils.generar_geotiff(Zi[::-1], [xi.min(), yi.min(), xi.max(), yi.max()])
                c2.download_button("Descargar Raster (TIFF)", tif_bytes, "mapa_recarga.tif")
            except Exception as e:
                c2.error(f"Error generando TIFF: {e}")

else:
    st.info("👈 Selecciona una zona o municipio en el menú lateral.")