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
from modules import land_cover

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

if st.sidebar.button("🧹 Limpiar Memoria y Recargar"):
    st.cache_data.clear()
    st.rerun()

ids_estaciones, nombre_zona, altitud_ref, gdf_zona = selectors.render_selector_espacial()
engine = db_manager.get_engine()

# --- 2. PARÁMETROS HIDROLÓGICOS (5 GRUPOS) ---
st.sidebar.divider()
st.sidebar.header("🎛️ Parámetros")

# ⚠️ Asegúrate de que este archivo exista en tu carpeta data/
RUTA_RASTER = "data/Cob25m_WGS84.tif"

modo_params = st.sidebar.radio(
    "Fuente de Parámetros:", 
    ["Automático (Satélite)", "Manual (Simulación)"],
    help="Automático: Calcula desde el mapa de coberturas.\nManual: Ajusta valores para escenarios."
)

# Valores por defecto para evitar errores si no hay cálculo
pct_bosque, pct_agricola, pct_pecuario, pct_agua, pct_urbano = 40.0, 20.0, 30.0, 5.0, 5.0

if modo_params == "Automático (Satélite)" and gdf_zona is not None:
    with st.sidebar.status("🛰️ Analizando coberturas..."):
        # Llamamos al módulo land_cover para procesar el raster
        stats_raw = land_cover.calcular_estadisticas_zona(gdf_zona, RUTA_RASTER)
        # Agrupamos los resultados en tus 5 categorías
        p_bosque, p_agricola, p_pecuario, p_agua, p_urbano = land_cover.agrupar_coberturas_turc(stats_raw)
    
    if not stats_raw:
        st.sidebar.warning("⚠️ No se encontró el archivo raster en data/. Usando valores por defecto.")
    else:
        st.sidebar.success("✅ Datos extraídos")
        pct_bosque, pct_agricola, pct_pecuario, pct_agua, pct_urbano = p_bosque, p_agricola, p_pecuario, p_agua, p_urbano
        
        # Inputs visuales (deshabilitados porque son datos reales)
        st.sidebar.number_input("% Bosque", value=float(pct_bosque), disabled=True, format="%.1f")
        st.sidebar.number_input("% Agrícola", value=float(pct_agricola), disabled=True, format="%.1f")
        st.sidebar.number_input("% Pecuario", value=float(pct_pecuario), disabled=True, format="%.1f")
        st.sidebar.number_input("% Agua/Humedal", value=float(pct_agua), disabled=True, format="%.1f")
        st.sidebar.metric("% Urbano / Otro", f"{pct_urbano:.1f}%")
        
        st.sidebar.caption("Distribución Real:")
        st.sidebar.progress(int(pct_bosque), text=f"Bosque {pct_bosque:.0f}%")
        st.sidebar.progress(int(pct_pecuario), text=f"Pecuario {pct_pecuario:.0f}%")

else:
    # Modo Manual: Tú controlas los sliders
    st.sidebar.info("Modo Simulación: Ajusta los valores.")
    pct_bosque = st.sidebar.number_input("% Bosque", 0, 100, 40)
    pct_agricola = st.sidebar.number_input("% Agrícola", 0, 100, 20)
    pct_pecuario = st.sidebar.number_input("% Pecuario", 0, 100, 30)
    pct_agua = st.sidebar.number_input("% Agua/Humedal", 0, 100, 5)
    
    pct_urbano = max(0, 100 - (pct_bosque + pct_agricola + pct_pecuario + pct_agua))
    st.sidebar.metric("% Urbano / Otro", f"{pct_urbano}%")

# --- CÁLCULO DEL KI (5 GRUPOS) ---
# Coeficientes: Bosque: 0.50 | Agrícola: 0.30 | Pecuario: 0.30 | Agua: 1.00 | Urbano: 0.10
ki_ponderado = (
    (pct_bosque * 0.50) + 
    (pct_agricola * 0.30) + 
    (pct_pecuario * 0.30) + 
    (pct_agua * 1.00) +
    (pct_urbano * 0.10)
) / 100.0

st.sidebar.metric("Coef. Infiltración (Ki)", f"{ki_ponderado:.2f}", delta_color="normal")

st.sidebar.divider()
meses_futuros = st.sidebar.slider("Horizonte", 12, 60, 24)
ruido = st.sidebar.slider("Incertidumbre", 0.0, 1.0, 0.1)


# Inicializar variables de sesión para descargas
if 'raster_data' not in st.session_state: st.session_state.raster_data = None

if gdf_zona is not None:
    # 1. Recuperar Estaciones (CORREGIDO: Incluye columna 'municipio')
    if not ids_estaciones:
        minx, miny, maxx, maxy = gdf_zona.total_bounds
        buff = 0.05
        # AGREGADO: municipio en el SELECT
        q_geo = text(f"""
            SELECT id_estacion, nom_est, latitud, longitud, alt_est, municipio 
            FROM estaciones 
            WHERE longitud BETWEEN {minx-buff} AND {maxx+buff} 
            AND latitud BETWEEN {miny-buff} AND {maxy+buff}
        """)
        df_puntos = pd.read_sql(q_geo, engine)
        
        if not df_puntos.empty:
            # Intento de filtro fino por polígono
            try:
                points = gpd.points_from_xy(df_puntos.longitud, df_puntos.latitud)
                gdf_pts = gpd.GeoDataFrame(df_puntos, geometry=points, crs="EPSG:4326")
                
                if gdf_zona.crs is None: gdf_zona = gdf_zona.set_crs("EPSG:4326")
                else: gdf_zona = gdf_zona.to_crs("EPSG:4326")
                
                df_joined = gpd.sjoin(gdf_pts, gdf_zona, how="inner", predicate="intersects")
                if not df_joined.empty: 
                    df_puntos = df_joined[df_puntos.columns].copy()
            except: pass
            
            ids_estaciones = df_puntos['id_estacion'].tolist()
    else:
        # Selección por IDs (AGREGADO: municipio)
        if len(ids_estaciones) == 1:
            q = text(f"SELECT id_estacion, nom_est, latitud, longitud, alt_est, municipio FROM estaciones WHERE id_estacion = '{ids_estaciones[0]}'")
            df_puntos = pd.read_sql(q, engine)
        else:
            q = text("SELECT id_estacion, nom_est, latitud, longitud, alt_est, municipio FROM estaciones WHERE id_estacion IN :ids")
            df_puntos = pd.read_sql(q, engine, params={'ids': tuple(ids_estaciones)})

    if df_puntos.empty:
        st.error("❌ No se encontraron estaciones.")
        st.stop()


    # 2. Estadísticas (Cacheado)
    with st.spinner("Procesando hidrología..."):
        df_mapa_stats = hydrogeo_utils.obtener_estadisticas_estaciones(engine, df_puntos)

    # 3. Serie Histórica
    df_raw = pd.DataFrame()
    intentos = [('precipitacion', 'id_estacion', 'fecha', 'valor'), ('precipitacion_mensual', 'id_estacion_fk', 'fecha_mes_año', 'precipitation')]
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

    # --- TAB 1: GRÁFICO COMPLETO ---
    with tab1:
        if not df_res.empty:
            df_hist = df_res[df_res['tipo'] == 'Histórico']
            df_fut = df_res[df_res['tipo'] == 'Proyección']
            
            fig = go.Figure()
            
            # 1. Variables Históricas
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['p_final'], name='Lluvia Hist.', line=dict(color='#95a5a6', width=1)))
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['etr_mm'], name='ETR Hist.', line=dict(color='#e67e22', width=1.5)))
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['escorrentia_mm'], name='Escorrentía Hist.', line=dict(color='#27ae60', width=1.5)))
            fig.add_trace(go.Scatter(x=df_hist['fecha'], y=df_hist['recarga_mm'], name='Recarga Hist.', line=dict(color='#2980b9', width=2), fill='tozeroy'))
            
            # 2. Proyecciones
            fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['p_final'], name='Lluvia Proy.', line=dict(color='#95a5a6', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['recarga_mm'], name='Recarga Proy.', line=dict(color='#00d2d3', width=2, dash='dot')))
            
            # 3. Incertidumbre
            if 'yhat_upper' in df_fut.columns:
                 fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_upper'], showlegend=False, line=dict(width=0)))
                 fig.add_trace(go.Scatter(x=df_fut['fecha'], y=df_fut['yhat_lower'], name='Incertidumbre', fill='tonexty', line=dict(width=0), fillcolor='rgba(0,210,211,0.1)'))
            
            fig.update_layout(height=500, hovermode="x unified", title="Balance Hídrico Completo y Proyección", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Sin datos históricos suficientes para graficar.")

    # --- TAB 2: CONTEXTO (TOOLTIPS RICOS) ---
    with tab2:
        if st.button("🔄 Recargar Mapa Contexto"): st.rerun()
        pad = 0.05
        bounds = [df_puntos.longitud.min()-pad, df_puntos.latitud.min()-pad, df_puntos.longitud.max()+pad, df_puntos.latitud.max()+pad]
        layers = hydrogeo_utils.cargar_capas_gis_optimizadas(engine, bounds)
        
        m = folium.Map(location=[df_puntos.latitud.mean(), df_puntos.longitud.mean()], zoom_start=11, tiles="CartoDB positron")
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

        st.markdown("<style>.leaflet-tooltip {white-space: normal !important; max-width: 300px !important; font-size:11px;}</style>", unsafe_allow_html=True)

        def tooltip_ok(gdf, dic):
            cols = [c.lower().strip() for c in gdf.columns]
            f, a = [], []
            # Lógica permisiva: busca coincidencias parciales
            for k, v in dic.items():
                match = next((c for c in cols if k.lower() in c), None)
                if match:
                    f.append(match)
                    a.append(v)
            return folium.GeoJsonTooltip(fields=f, aliases=a, localize=True) if f else None

        # Diccionarios Expandidos para Tooltips
        if 'suelos' in layers:
            dic_suelos = {'ucs':'UCS:', 'litolo':'Litología:', 'caracter':'Caract:', 'paisaje':'Paisaje:', 'clima':'Clima:', 'component':'Comp:', 'porcent':'%:'}
            folium.GeoJson(layers['suelos'], name="Suelos", style_function=lambda x: {'color':'orange', 'weight':0.5, 'fillOpacity':0.2},
                           tooltip=tooltip_ok(layers['suelos'], dic_suelos)).add_to(m)
        if 'hidro' in layers:
            dic_hidro = {'potencial':'Potencial:', 'unidad':'Unidad:', 'sigla':'Sigla:', 'cod':'Cod:', 'area':'Area:'}
            folium.GeoJson(layers['hidro'], name="Hidro", style_function=lambda x: {'color':'blue', 'weight':0.5, 'fillOpacity':0.2},
                           tooltip=tooltip_ok(layers['hidro'], dic_hidro)).add_to(m)

        if 'bocatomas' in layers:
            # Diccionario exacto basado en tus tablas
            dic_boca = {
                'nombre_acu': 'Acueducto:',    # Nombre del Acueducto
                'tipo': 'Tipo:',               # Veredal, Municipal, etc.
                'fuente_aba': 'Fuente Sup:',   # Fuente Abastecedora (Superficial)
                'fuente_sub': 'Fuente Sub:',   # Fuente Subterránea (SI/NO)
                'pozos': 'Pozos:',             # Tiene pozos
                'entidad_ad': 'Entidad:',      # Entidad Administradora
                'suscriptor': 'Suscriptores:', # Número de suscriptores
                'q': 'Caudal (L/s):'           # Caudal (Q)
            }
            
            folium.GeoJson(
                layers['bocatomas'], 
                name="Bocatomas", 
                marker=folium.CircleMarker(radius=5, color='#d63031', fill_color='#ff7675', fill_opacity=0.8),
                tooltip=tooltip_ok(layers['bocatomas'], dic_boca)
            ).add_to(m)

        # --- ESTACIONES (Popups Completos) ---
        for _, r in df_mapa_stats.iterrows():
            # Formateador seguro
            def fmt(val, mult=12): 
                if pd.isnull(val): return "<span style='color:red'>N/D</span>"
                return f"{val*mult:,.0f} mm"

            mun = r.get('municipio', 'N/D')
            alt = r.get('alt_est', 0)
            std_val = r.get('std_lluvia', 0)
            
            html = f"""
            <div style='font-family:sans-serif; width:200px; font-size:12px;'>
                <b style="font-size:13px; color:#2c3e50;">{r['nom_est']}</b>
                <hr style='margin:4px 0; border-top: 1px solid #ccc;'>
                📍 <b>Mun:</b> {mun} <br>
                ⛰️ <b>Alt:</b> {alt:,.0f} m <br>
                <hr style='margin:4px 0; border-top: 1px dashed #ccc;'>
                🌧️ <b>Lluvia:</b> {fmt(r.get('p_media'))}<br>
                ☀️ <b>ETR:</b> {fmt(r.get('etr_media'))}<br>
                🌊 <b>Escorrentía:</b> {fmt(r.get('escorrentia_media'))}<br>
                💧 <b>Recarga:</b> <b style='color:#0000AA;'>{fmt(r.get('recarga_calc'))}</b><br>
                <div style="margin-top:4px; font-size:10px; color:#7f8c8d; text-align:right;">
                    (Desv. Std Lluvia: {std_val:.1f})
                </div>
            </div>"""
            
            folium.Marker(
                [r['latitud'], r['longitud']], 
                popup=folium.Popup(html, max_width=220), 
                icon=folium.Icon(color='black', icon='tint'),
                tooltip=r['nom_est']
            ).add_to(m)

        # Selector de Capas (Vital para activar/desactivar)
        folium.LayerControl(position='topright', collapsed=True).add_to(m)

        st_folium(m, width=1400, height=600, key=f"ctx_{nombre_zona}")

    # --- TAB 3: RECARGA (BOTÓN + RASTER) ---
    with tab3:
        if st.button("🔄 Recargar Mapa Recarga"): st.rerun() # Botón Recuperado
        
        df_valid = df_mapa_stats.dropna(subset=['p_media'])
        if len(df_valid) < 4:
            st.warning("⚠️ Se requieren al menos 4 estaciones con datos válidos para interpolar.")

            st.session_state.raster_data = None # Limpiar si falla
        else:
            # Interpolación
            x, y, z = df_valid.longitud.values, df_valid.latitud.values, df_valid.p_media.values
            xi = np.linspace(bounds[0], bounds[2], 100)
            yi = np.linspace(bounds[1], bounds[3], 100)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((x, y), z, (Xi, Yi), method='linear')
            
            z_r = df_valid.recarga_calc.values * 12
            Zi_r = griddata((x, y), z_r, (Xi, Yi), method='linear')
            
            # Guardar en sesión
            st.session_state.raster_data = (Zi_r, xi, yi)
            
            # Mapa Base
            m_iso = folium.Map(location=[df_puntos.latitud.mean(), df_puntos.longitud.mean()], zoom_start=11, tiles="CartoDB positron")
            m_iso.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            
            # Capa Raster (Colores)
            vmin, vmax = np.nanmin(Zi_r), np.nanmax(Zi_r)
            try: cmap = plt.colormaps['viridis']
            except: cmap = cm.get_cmap('viridis')
            rgba = cmap((Zi_r - vmin)/(vmax - vmin)); rgba[np.isnan(Zi_r), 3] = 0
            
            folium.raster_layers.ImageOverlay(
                image=rgba, 
                bounds=[[yi.min(), xi.min()], [yi.max(), xi.max()]], 
                opacity=0.7, 
                origin='lower',
                name="Recarga (Raster)"
            ).add_to(m_iso)

            # 3. ISOLÍNEAS CON ETIQUETAS FIJAS (Números visibles)
            try:
                fig_c, ax_c = plt.subplots()
                # Generamos las curvas (menos niveles para no saturar el mapa con texto)
                cs = ax_c.contour(Xi, Yi, Zi_r, levels=10, colors='white', linewidths=0.8)
                plt.close(fig_c)
                
                for i, collection in enumerate(cs.allsegs):
                    level_val = cs.levels[i]
                    for segment in collection:
                        # Solo dibujamos si el segmento es relevante (> 5 puntos) para evitar ruido
                        if len(segment) > 5:
                            locs = [[pt[1], pt[0]] for pt in segment]
                            
                            # 1. Dibujar la Línea
                            folium.PolyLine(
                                locs, 
                                color='white', 
                                weight=1.0, 
                                opacity=0.8,
                                name="Isolíneas"
                            ).add_to(m_iso)
                            
                            # 2. Calcular punto medio para poner el texto
                            mid_idx = len(locs) // 2
                            lat_lbl, lon_lbl = locs[mid_idx]
                            
                            # 3. Crear Etiqueta de Texto Fija (DivIcon)
                            # Usamos text-shadow para que el número blanco se lea sobre fondo claro u oscuro
                            html_text = f"""
                                <div style="
                                    font-size: 9pt; 
                                    font-weight: bold; 
                                    color: white; 
                                    text-shadow: 1px 1px 2px black, -1px -1px 2px black;
                                    white-space: nowrap;
                                ">{int(level_val)}</div>
                            """
                            
                            folium.Marker(
                                location=[lat_lbl, lon_lbl],
                                icon=DivIcon(
                                    icon_size=(30, 10),
                                    icon_anchor=(15, 5), # Centrar el texto
                                    html=html_text
                                )
                            ).add_to(m_iso)
            except Exception as e:
                print(f"Error isolíneas: {e}")

            # 4. CONTORNO DE LA ZONA SELECCIONADA (Cuenca/Municipio)
            # Esto dibuja el límite exacto de lo que estás analizando
            if gdf_zona is not None:
                # Asegurar proyección correcta
                if gdf_zona.crs and gdf_zona.crs.to_string() != "EPSG:4326":
                    gdf_boundary = gdf_zona.to_crs("EPSG:4326")
                else:
                    gdf_boundary = gdf_zona

                folium.GeoJson(
                    gdf_boundary,
                    name=f"Límite: {nombre_zona}",
                    style_function=lambda x: {
                        'color': '#2c3e50',       # Color borde (Gris oscuro elegante)
                        'weight': 2.5,            # Grosor
                        'fillOpacity': 0.0,       # Relleno transparente (para ver el mapa debajo)
                        'dashArray': '5, 5',      # Línea punteada para diferenciar de isolíneas
                        'opacity': 1.0
                    },
                    tooltip=f"Zona: {nombre_zona}"
                ).add_to(m_iso)

            # 5. ESTACIONES (Puntos negros simples)
            for _, r in df_valid.iterrows(): # Usamos df_valid para mostrar solo las que tienen datos
                folium.CircleMarker(
                    location=[r['latitud'], r['longitud']],
                    radius=3,
                    color='black',
                    fill=True,
                    fill_color='white',
                    fill_opacity=1.0,
                    weight=1,
                    tooltip=f"{r['nom_est']}: {r['recarga_calc']*12:,.0f} mm",
                    name="Estaciones"
                ).add_to(m_iso)

            # 6. SELECTOR DE CAPAS (Asegúrate que esto esté AL FINAL)
            folium.LayerControl(position='topright', collapsed=True).add_to(m_iso)
            
            st_folium(m_iso, width=1400, height=600, key=f"iso_{nombre_zona}")


    # --- TAB 4: DESCARGAS (RASTER + CSV) ---
    with tab4:
        c1, c2 = st.columns(2)
        if not df_res.empty: 
            c1.download_button("📥 Descargar Serie (CSV)", df_res.to_csv(index=False), "serie_hidrologica.csv")
        
        # Recuperar Raster de Sesión
        if st.session_state.raster_data is not None:
            Zi_s, xi_s, yi_s = st.session_state.raster_data
            try:
                # bounds para rasterio: minx, miny, maxx, maxy
                b_tif = [xi_s.min(), yi_s.min(), xi_s.max(), yi_s.max()]
                # GeoTIFF
                tif_bytes = hydrogeo_utils.generar_geotiff(Zi_s[::-1], b_tif)
                c2.download_button("🗺️ Descargar Mapa Recarga (GeoTIFF)", tif_bytes, "mapa_recarga.tif", mime="image/tiff")
            except Exception as e:
                c2.error(f"Error generando descarga: {e}")
        else:
            c2.info("El mapa raster se genera en la pestaña 'Mapa Recarga'. Visítala primero.")

else:
    st.info("👈 Selecciona una zona.")