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

from modules import db_manager, hydrogeo_utils, selectors

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

# Botón de Auxilio
if st.sidebar.button("🧹 Limpiar Memoria y Recargar"):
    st.cache_data.clear()
    st.rerun()

ids_estaciones, nombre_zona, altitud_ref, gdf_zona = selectors.render_selector_espacial()
engine = db_manager.get_engine()

# --- PARÁMETROS ---
st.sidebar.divider()
st.sidebar.header("🎛️ Parámetros")
col_s1, col_s2 = st.sidebar.columns(2)
pct_bosque = col_s1.number_input("% Bosque", 0, 100, 50)
pct_cultivo = col_s2.number_input("% Agrícola", 0, 100, 30)
pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
st.sidebar.metric("% Urbano/Otro", f"{pct_urbano}%")
ki_ponderado = ((pct_bosque * 0.50) + (pct_cultivo * 0.30) + (pct_urbano * 0.10)) / 100.0
st.sidebar.metric("Coef. Infiltración (Ki)", f"{ki_ponderado:.2f}")
meses_futuros = st.sidebar.slider("Horizonte", 12, 60, 24)
ruido = st.sidebar.slider("Incertidumbre", 0.0, 1.0, 0.1)

Zi, xi, yi = None, None, None

# --- LÓGICA PRINCIPAL ---
if gdf_zona is not None:
    # 1. Recuperar Estaciones
    if not ids_estaciones:
        minx, miny, maxx, maxy = gdf_zona.total_bounds
        buff = 0.05
        q_geo = text(f"SELECT id_estacion, nom_est, latitud, longitud, alt_est FROM estaciones WHERE longitud BETWEEN {minx-buff} AND {maxx+buff} AND latitud BETWEEN {miny-buff} AND {maxy+buff}")
        df_puntos = pd.read_sql(q_geo, engine)
        if not df_puntos.empty:
            ids_estaciones = df_puntos['id_estacion'].tolist()
    else:
        if len(ids_estaciones) == 1:
            q = text(f"SELECT id_estacion, nom_est, latitud, longitud, alt_est FROM estaciones WHERE id_estacion = '{ids_estaciones[0]}'")
            df_puntos = pd.read_sql(q, engine)
        else:
            q = text("SELECT id_estacion, nom_est, latitud, longitud, alt_est FROM estaciones WHERE id_estacion IN :ids")
            df_puntos = pd.read_sql(q, engine, params={'ids': tuple(ids_estaciones)})

    if df_puntos.empty:
        st.error("❌ No se encontraron estaciones.")
        st.stop()

    # 2. Calcular Estadísticas (Motor arreglado)
    with st.spinner("Procesando hidrología..."):
        df_mapa_stats = hydrogeo_utils.obtener_estadisticas_estaciones(engine, df_puntos)

    # 3. Serie Histórica (Intento múltiple)
    df_raw = pd.DataFrame()
    intentos = [
        ('precipitacion', 'id_estacion', 'fecha', 'valor'),
        ('precipitacion_mensual', 'id_estacion_fk', 'fecha_mes_año', 'precipitation')
    ]
    
    for tbl, col_id, col_f, col_v in intentos:
        try:
            if len(ids_estaciones) == 1:
                q = text(f"SELECT {col_f} as fecha, {col_v} as valor FROM {tbl} WHERE {col_id} = '{ids_estaciones[0]}'")
                df_temp = pd.read_sql(q, engine)
            else:
                q = text(f"SELECT {col_f} as fecha, {col_v} as valor FROM {tbl} WHERE {col_id} IN :ids")
                df_temp = pd.read_sql(q, engine, params={'ids': tuple(ids_estaciones)})
            
            if not df_temp.empty:
                df_raw = df_temp
                break
        except: continue

    df_res = pd.DataFrame()
    if not df_raw.empty:
        alt_calc = altitud_ref if altitud_ref else df_puntos['alt_est'].mean()
        df_res = hydrogeo_utils.ejecutar_pronostico_prophet(df_raw, meses_futuros, alt_calc, ki_ponderado, ruido)

    # --- VISUALIZACIÓN ---
    st.markdown(f"### 📍 {nombre_zona}")
    
    if not df_res.empty:
        df_hist = df_res[df_res['tipo'] == 'Histórico']
        if not df_hist.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Lluvia Media", f"{df_hist['p_final'].mean()*12:,.0f} mm/año")
            c2.metric("ETR", f"{df_hist['etr_mm'].mean()*12:,.0f} mm/año")
            c3.metric("Recarga", f"{df_hist['recarga_mm'].mean()*12:,.0f} mm/año")
            c4.metric("Estaciones", len(df_puntos))
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Serie Completa", "🗺️ Mapa Contexto", "🌈 Mapa Recarga", "📥 Descargas"])

    with tab1:
        if not df_res.empty:
            df_hist = df_res[df_res['tipo'] == 'Histórico']
            df_fut = df_res[df_res['tipo'] == 'Proyección']
            fig = go.Figure()
            # Gráficas Completas
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['p_final'], name='Lluvia', line=dict(color='gray', width=1), opacity=0.5))
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['recarga_mm'], name='Recarga Hist.', line=dict(color='blue', width=2), fill='tozeroy'))
            fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['recarga_mm'], name='Recarga Proy.', line=dict(color='cyan', width=2, dash='dot')))
            # Incertidumbre
            if 'yhat_upper' in df_fut.columns:
                 fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_upper'], showlegend=False, line=dict(width=0)))
                 fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_lower'], name='Incertidumbre', fill='tonexty', line=dict(width=0), fillcolor='rgba(0,255,255,0.1)'))
            
            fig.update_layout(height=400, hovermode="x unified", title="Dinámica de Recarga (Balance Hídrico)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No se encontraron datos históricos de lluvia para generar el gráfico.")

    with tab2:
        if st.button("🔄 Recargar Mapa"): st.rerun()
        pad = 0.05
        bounds = [df_puntos.longitud.min()-pad, df_puntos.latitud.min()-pad, df_puntos.longitud.max()+pad, df_puntos.latitud.max()+pad]
        layers = hydrogeo_utils.cargar_capas_gis_optimizadas(engine, bounds)
        
        m = folium.Map(location=[df_puntos.latitud.mean(), df_puntos.longitud.mean()], zoom_start=11, tiles="CartoDB positron")
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        # Tooltip Helper (Limpia espacios y minúsculas)
        def tooltip_ok(gdf, dic):
            cols = [c.lower().strip() for c in gdf.columns]
            f, a = [], []
            for k, v in dic.items():
                k_clean = k.lower().strip()
                if k_clean in cols:
                    f.append(gdf.columns[cols.index(k_clean)])
                    a.append(v)
            return folium.GeoJsonTooltip(fields=f, aliases=a, localize=True) if f else None

        if 'suelos' in layers:
            folium.GeoJson(layers['suelos'], name="Suelos", style_function=lambda x: {'color':'orange', 'weight':0.5, 'fillOpacity':0.2},
                           tooltip=tooltip_ok(layers['suelos'], {'ucs_f':'UCS:', 'litologia':'Lit:', 'caracteri':'Caract:', 'paisaje':'Paisaje:'})).add_to(m)
        if 'hidro' in layers:
            folium.GeoJson(layers['hidro'], name="Hidro", style_function=lambda x: {'color':'blue', 'weight':0.5, 'fillOpacity':0.2},
                           tooltip=tooltip_ok(layers['hidro'], {'potencial':'Pot:', 'unidad_geo':'Uni:'})).add_to(m)
        if 'bocatomas' in layers:
            folium.GeoJson(layers['bocatomas'], name="Bocatomas", marker=folium.CircleMarker(radius=4, color='red'),
                           tooltip=tooltip_ok(layers['bocatomas'], {'nom_bocatoma':'Nom:', 'nombre_acu':'Acu:'})).add_to(m)

        # Popups
        for _, r in df_mapa_stats.iterrows():
            def fmt(val): return f"{val*12:,.0f} mm" if pd.notnull(val) else "<span style='color:red'>N/D</span>"
            html = f"""
            <div style='font-family:sans-serif; width:160px; font-size:12px;'>
                <b>{r['nom_est']}</b><hr style='margin:3px 0;'>
                🌧️ Lluvia: <b>{fmt(r.get('p_media'))}</b><br>
                💧 Recarga: <b>{fmt(r.get('recarga_calc'))}</b>
            </div>"""
            folium.Marker([r['latitud'], r['longitud']], popup=folium.Popup(html, max_width=200), icon=folium.Icon(color='black', icon='tint')).add_to(m)

        st_folium(m, width=1400, height=600, key=f"ctx_{nombre_zona}")

    with tab3:
        # Interpolación: Requiere al menos 4 puntos CON DATOS
        df_valid = df_mapa_stats.dropna(subset=['p_media'])
        
        if len(df_valid) < 4:
            st.warning(f"⚠️ Se encontraron {len(df_valid)} estaciones con datos válidos. Se requieren mínimo 4 para interpolar.")
        else:
            x, y, z = df_valid.longitud.values, df_valid.latitud.values, df_valid.p_media.values
            xi = np.linspace(bounds[0], bounds[2], 100)
            yi = np.linspace(bounds[1], bounds[3], 100)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((x, y), z, (Xi, Yi), method='linear')
            
            z_r = df_valid.recarga_calc.values * 12
            Zi_r = griddata((x, y), z_r, (Xi, Yi), method='linear')
            
            m_iso = folium.Map(location=[df_puntos.latitud.mean(), df_puntos.longitud.mean()], zoom_start=11, tiles="CartoDB positron")
            m_iso.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            
            vmin, vmax = np.nanmin(Zi_r), np.nanmax(Zi_r)
            try: cmap = plt.colormaps['viridis']
            except: cmap = cm.get_cmap('viridis')
            rgba = cmap((Zi_r - vmin)/(vmax - vmin)); rgba[np.isnan(Zi_r), 3] = 0
            
            folium.raster_layers.ImageOverlay(image=rgba, bounds=[[yi.min(), xi.min()], [yi.max(), xi.max()]], opacity=0.7, origin='lower').add_to(m_iso)
            st_folium(m_iso, width=1400, height=600, key=f"iso_{nombre_zona}")

    with tab4:
        c1, c2 = st.columns(2)
        if not df_res.empty: c1.download_button("Descargar CSV", df_res.to_csv(index=False), "serie.csv")

else:
    st.info("👈 Selecciona una zona.")