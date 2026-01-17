# pages/05_🗺️_Isoyetas_HD.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import geopandas as gpd
from scipy.interpolate import Rbf
import os
import sys
import io

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Isoyetas HD", page_icon="🗺️", layout="wide")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules.config import Config
    try:
        from modules.data_processor import complete_series
    except ImportError:
        complete_series = None
except:
    complete_series = None
    pass

st.title("🗺️ Generador Avanzado de Isoyetas (Escenarios & Pronósticos)")

# --- 2. FICHA TÉCNICA (RESTAURADA) ---
with st.expander("📘 Ficha Técnica: Metodología, Utilidad y Fuentes", expanded=False):
    st.markdown("""
    ### 1. Concepto y Utilidad
    Las **isoyetas** son líneas que unen puntos de igual precipitación. Este mapa permite visualizar la distribución espacial de la lluvia acumulada en un año específico, identificando zonas de oferta hídrica (superávit) o déficit. Útil para la planificación de cuencas y gestión del riesgo.

    ### 2. Metodología de Interpolación
    Se utiliza el algoritmo **RBF (Radial Basis Function)** con la función núcleo *Thin-Plate Spline*. 
    * A diferencia de métodos simples (como IDW), el RBF genera una superficie suave y continua que minimiza la curvatura total, simulando el comportamiento físico de una lámina flexible que pasa por los puntos de medición.
    * Esto reduce artefactos visuales y mejora la estimación en zonas de transición.

    ### 3. Interpretación
    * **🟦 Azul Oscuro:** Máximos de precipitación. Zonas de recarga potencial.
    * **🟨 Amarillo/Claro:** Mínimos de precipitación. Zonas más secas.

    ### 4. Fuentes de Información
    * **Datos:** Base de datos consolidada SIHCLI (IDEAM, EPM, Cenicafé, etc.).
    * **Cartografía:** Límites político-administrativos (IGAC) y zonificación hidrográfica (CuencaVerde).
### 🛠️ Modos de Análisis Disponibles
    1.  **Año Específico:** Muestra la lluvia real acumulada en un año histórico.
    2.  **Mínimos/Máximos Históricos:** Identifica los extremos climáticos (años más secos o lluviosos) registrados en cada estación. Útil para mapas de amenaza.
    3.  **Promedio Multianual:** Calcula la "Normal Climatológica" promediando la lluvia en un rango de años seleccionado (ej. 1981-2010).
    4.  **Pronóstico (Tendencia):** Proyecta la lluvia futura (2026-2040) basándose en la tendencia lineal histórica de cada estación. *Nota: Es una proyección estadística, no un modelo físico.*

    ### 📐 Metodología de Interpolación
    Se utiliza **RBF (Thin-Plate Spline)** para generar superficies suaves a partir de los puntos, extendiendo el análisis a zonas sin medición mediante un buffer de búsqueda inteligente.
    """)

# --- 3. FUNCIONES DE SOPORTE ---
@st.cache_data(ttl=3600)
def load_geojson_cached(filename):
    possible_paths = [os.path.join("data", filename), os.path.join("..", "data", filename), os.path.join(os.path.dirname(__file__), '..', 'data', filename)]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                gdf = gpd.read_file(path)
                if gdf.crs and gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                return gdf
            except: continue
    return None

def detectar_columna(df, keywords):
    if df is None or df.empty: return None
    cols_orig = df.columns.tolist()
    for kw in keywords:
        kw_clean = kw.lower().replace('-', '').replace('_', '')
        for col in cols_orig:
            if kw_clean in col.lower().replace('-', '').replace('_', ''): return col
    return None

@st.cache_data(ttl=600)
def obtener_estaciones_enriquecidas():
    try:
        engine = create_engine(st.secrets["DATABASE_URL"])
        q = "SELECT *, ST_Y(geom::geometry) as lat_calc, ST_X(geom::geometry) as lon_calc FROM estaciones"
        df_est = pd.read_sql(q, engine)
        gdf_est = gpd.GeoDataFrame(df_est, geometry=gpd.points_from_xy(df_est.lon_calc, df_est.lat_calc), crs="EPSG:4326")
        
        gdf_cuencas = load_geojson_cached("SubcuencasAinfluencia.geojson")
        if gdf_cuencas is not None:
            col_cuenca_geo = detectar_columna(gdf_cuencas, ['n-nss3', 'n_nss3', 'nnss3', 'nombre', 'subcuenca'])
            if col_cuenca_geo:
                gdf_joined = gpd.sjoin(gdf_est, gdf_cuencas[[col_cuenca_geo, 'geometry']], how='left', predicate='within')
                gdf_joined = gdf_joined.rename(columns={col_cuenca_geo: 'CUENCA_GIS'})
                gdf_joined['CUENCA_GIS'] = gdf_joined['CUENCA_GIS'].fillna('Fuera de Jurisdicción')
                return gdf_joined, True
        return gdf_est, False
    except Exception as e:
        return pd.DataFrame(), False

def generar_raster_ascii(grid_z, minx, miny, cellsize, nrows, ncols):
    header = f"ncols        {ncols}\nnrows        {nrows}\nxllcorner    {minx}\nyllcorner    {miny}\ncellsize     {cellsize}\nNODATA_value -9999\n"
    grid_fill = np.nan_to_num(grid_z.T, nan=-9999)
    body = ""
    for row in np.flipud(grid_fill.T): 
        body += " ".join([f"{val:.2f}" for val in row]) + "\n"
    return header + body

def add_context_layers_ghost(fig, gdf_zona):
    if gdf_zona is None or gdf_zona.empty: return
    try:
        roi = gdf_zona.buffer(0.1)
        gdf_m = load_geojson_cached("MunicipiosAntioquia.geojson")
        gdf_cu = load_geojson_cached("SubcuencasAinfluencia.geojson")
        
        if gdf_m is not None:
            gdf_c = gpd.clip(gdf_m, roi)
            for _, r in gdf_c.iterrows():
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(width=0.5, color='rgba(100,100,100,0.2)', dash='dot'), hoverinfo='skip', showlegend=False))
        if gdf_cu is not None:
            gdf_c = gpd.clip(gdf_cu, roi)
            for _, r in gdf_c.iterrows():
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(width=0.8, color='rgba(50,100,200,0.4)', dash='dash'), hoverinfo='skip', showlegend=False))
    except: pass

def calcular_pronostico(df_hist, target_year):
    """Proyección lineal simple por estación."""
    proyecciones = []
    # df_hist debe tener: station_id, year, value
    for station in df_hist['station_id'].unique():
        datos_est = df_hist[df_hist['station_id'] == station].sort_values('year')
        if len(datos_est) >= 5: # Mínimo 5 años para tendencia
            x = datos_est['year'].values
            y = datos_est['value'].values
            # Regresión Lineal (Grado 1)
            slope, intercept = np.polyfit(x, y, 1)
            pred = (slope * target_year) + intercept
            proyecciones.append({'station_id': station, 'valor': max(0, pred)}) # No lluvia negativa
    return pd.DataFrame(proyecciones)

# --- 4. SIDEBAR & LÓGICA DE DATOS ---
st.sidebar.header("🔍 Configuración de Escenarios")

with st.spinner("Cargando ecosistema espacial..."):
    gdf_meta, exito_cruce = obtener_estaciones_enriquecidas()

if gdf_meta.empty:
    st.error("Error crítico de base de datos.")
    st.stop()

# Detectar columnas
col_id = detectar_columna(gdf_meta, ['id_estacion', 'codigo']) or 'id_estacion'
col_nom = detectar_columna(gdf_meta, ['nom_est', 'nombre']) or 'nom_est'
col_region = detectar_columna(gdf_meta, ['region', 'subregion', 'depto_region'])
col_muni = detectar_columna(gdf_meta, ['municipio', 'mpio'])
col_alt = detectar_columna(gdf_meta, ['alt_est', 'altitud'])
col_cuenca = 'CUENCA_GIS' if 'CUENCA_GIS' in gdf_meta.columns else None

# Filtros Jerárquicos
df_filtered_meta = gdf_meta.copy()

if col_region:
    regs = sorted(df_filtered_meta[col_region].dropna().astype(str).unique())
    sel_reg = st.sidebar.multiselect("📍 Región:", regs)
    if sel_reg: df_filtered_meta = df_filtered_meta[df_filtered_meta[col_region].isin(sel_reg)]

if col_cuenca:
    cuencas = sorted(df_filtered_meta[col_cuenca].dropna().astype(str).unique())
    sel_cuenca = st.sidebar.multiselect("🌊 Cuenca:", cuencas)
    if sel_cuenca: df_filtered_meta = df_filtered_meta[df_filtered_meta[col_cuenca].isin(sel_cuenca)]

if col_muni:
    munis = sorted(df_filtered_meta[col_muni].dropna().astype(str).unique())
    sel_muni = st.sidebar.multiselect("🏙️ Municipio:", munis)
    if sel_muni: df_filtered_meta = df_filtered_meta[df_filtered_meta[col_muni].isin(sel_muni)]

st.sidebar.markdown(f"**Estaciones en zona:** {len(df_filtered_meta)}")
st.sidebar.divider()

# --- CONFIGURACIÓN DE ESCENARIOS (NUEVO) ---
tipo_analisis = st.sidebar.selectbox(
    "📊 Modo de Análisis:", 
    ["Año Específico", "Promedio Multianual", "Mínimo Histórico", "Máximo Histórico", "Pronóstico Futuro"]
)

params_analisis = {}

if tipo_analisis == "Año Específico":
    params_analisis['year'] = st.sidebar.selectbox("📅 Año:", range(2025, 1980, -1))

elif tipo_analisis == "Promedio Multianual":
    rango = st.sidebar.slider("📅 Periodo de Referencia:", 1980, 2025, (1990, 2020))
    params_analisis['start'], params_analisis['end'] = rango

elif tipo_analisis == "Pronóstico Futuro":
    params_analisis['target'] = st.sidebar.slider("🔮 Año a Proyectar:", 2026, 2040, 2026)
    st.sidebar.info("Proyección basada en tendencia lineal histórica.")

# Buffer y Opciones
buffer_km = st.sidebar.slider("📡 Buffer Búsqueda (km):", 0, 50, 10)
buffer_deg = buffer_km / 111.0

c1, c2 = st.sidebar.columns(2)
ignore_zeros = c1.checkbox("🚫 No Ceros", value=True)
ignore_nulls = c2.checkbox("🚫 No Nulos", value=True)

do_interp_temp = False
if complete_series and tipo_analisis == "Año Específico":
    do_interp_temp = st.sidebar.checkbox("🔄 Interpolación Temporal", value=False)

suavidad = st.sidebar.slider("🎨 Suavizado (RBF):", 0.0, 2.0, 0.5)

# --- 5. MOTOR DE CÁLCULO Y VISUALIZACIÓN ---
if len(df_filtered_meta) > 0:
    gdf_target = df_filtered_meta
    minx, miny, maxx, maxy = gdf_target.total_bounds
    q_minx, q_miny = minx - buffer_deg, miny - buffer_deg
    q_maxx, q_maxy = maxx + buffer_deg, maxy + buffer_deg
    
    tab_mapa, tab_datos = st.tabs(["🗺️ Visualización Espacial", "💾 Descargas GIS"])
    
    with tab_mapa:
        try:
            engine = create_engine(st.secrets["DATABASE_URL"])
            df_agg = pd.DataFrame() # Contenedor de resultados
            
            # --- LÓGICA POR ESCENARIO ---
            
            if tipo_analisis == "Año Específico":
                q_data = text(f"""
                    SELECT p.id_estacion_fk as station_id, p.fecha_mes_año as fecha_safe, p.precipitation as val
                    FROM precipitacion_mensual p JOIN estaciones e ON p.id_estacion_fk = e.id_estacion
                    WHERE extract(year from p.fecha_mes_año) = :y
                    AND ST_X(e.geom::geometry) BETWEEN :mx AND :Mx AND ST_Y(e.geom::geometry) BETWEEN :my AND :My
                """)
                df_raw = pd.read_sql(q_data, engine, params={"y": params_analisis['year'], "mx":q_minx, "my":q_miny, "Mx":q_maxx, "My":q_maxy})
                
                if not df_raw.empty:
                    if do_interp_temp and complete_series:
                        df_proc = df_raw.rename(columns={'fecha_safe': 'fecha_mes_año', 'val': 'value'})
                        df_proc['fecha_mes_año'] = pd.to_datetime(df_proc['fecha_mes_año'])
                        with st.spinner("Interpolando tiempo..."):
                            df_filled = complete_series(df_proc)
                            df_agg = df_filled.groupby('station_id')['value'].sum().reset_index()
                    else:
                        df_agg = df_raw.groupby('station_id')['val'].sum().reset_index()
                    df_agg.columns = [col_id, 'valor']

            elif tipo_analisis in ["Promedio Multianual", "Mínimo Histórico", "Máximo Histórico", "Pronóstico Futuro"]:
                # Para estos necesitamos historia anual. Traemos todo el rango relevante.
                q_hist = text(f"""
                    SELECT p.id_estacion_fk as station_id, extract(year from p.fecha_mes_año) as year, SUM(p.precipitation) as value
                    FROM precipitacion_mensual p JOIN estaciones e ON p.id_estacion_fk = e.id_estacion
                    WHERE ST_X(e.geom::geometry) BETWEEN :mx AND :Mx AND ST_Y(e.geom::geometry) BETWEEN :my AND :My
                    GROUP BY 1, 2
                """)
                df_hist = pd.read_sql(q_hist, engine, params={"mx":q_minx, "my":q_miny, "Mx":q_maxx, "My":q_maxy})
                
                if not df_hist.empty:
                    if tipo_analisis == "Promedio Multianual":
                        mask = (df_hist['year'] >= params_analisis['start']) & (df_hist['year'] <= params_analisis['end'])
                        df_agg = df_hist[mask].groupby('station_id')['value'].mean().reset_index()
                    
                    elif tipo_analisis == "Mínimo Histórico":
                        df_agg = df_hist.groupby('station_id')['value'].min().reset_index()
                        
                    elif tipo_analisis == "Máximo Histórico":
                        df_agg = df_hist.groupby('station_id')['value'].max().reset_index()
                        
                    elif tipo_analisis == "Pronóstico Futuro":
                        with st.spinner(f"Calculando tendencias y proyectando al {params_analisis['target']}..."):
                            df_agg = calcular_pronostico(df_hist, params_analisis['target'])
                    
                    if not df_agg.empty:
                        df_agg.columns = [col_id, 'valor']

            # --- RENDERIZADO COMÚN ---
            if not df_agg.empty:
                # Merge con metadatos
                cols_merge = [col_id, col_nom, 'lat_calc', 'lon_calc']
                if col_muni: cols_merge.append(col_muni)
                if col_alt: cols_merge.append(col_alt)
                if col_cuenca: cols_merge.append(col_cuenca)
                cols_merge = list(set(cols_merge))
                
                df_final = pd.merge(df_agg, gdf_meta[cols_merge], on=col_id)
                
                if ignore_zeros: df_final = df_final[df_final['valor'] > 0]
                if ignore_nulls: df_final = df_final.dropna(subset=['valor'])
                
                if len(df_final) >= 3:
                    with st.spinner(f"Generando isoyetas ({len(df_final)} estaciones)..."):
                        grid_res = 200
                        gx, gy = np.mgrid[q_minx:q_maxx:complex(0, grid_res), q_miny:q_maxy:complex(0, grid_res)]
                        rbf = Rbf(df_final['lon_calc'], df_final['lat_calc'], df_final['valor'], function='thin_plate', smooth=suavidad)
                        grid_z = rbf(gx, gy)
                        
                        fig = go.Figure()
                        
                        # Título dinámico
                        if tipo_analisis == "Año Específico": tit = f"Isoyetas Año {params_analisis['year']}"
                        elif tipo_analisis == "Promedio Multianual": tit = f"Isoyetas Promedio {params_analisis['start']}-{params_analisis['end']}"
                        elif tipo_analisis == "Pronóstico Futuro": tit = f"Isoyetas Proyectadas {params_analisis['target']}"
                        else: tit = f"Isoyetas {tipo_analisis}"
                        
                        # Hover Info
                        custom_data = np.stack((
                            df_final[col_muni].fillna('-') if col_muni else ["-"]*len(df_final),
                            df_final[col_alt].fillna(0) if col_alt else [0]*len(df_final),
                            df_final[col_cuenca].fillna('-') if col_cuenca else ["-"]*len(df_final)
                        ), axis=-1)
                        
                        fig.add_trace(go.Contour(
                            z=grid_z.T, x=np.linspace(q_minx, q_maxx, grid_res), y=np.linspace(q_miny, q_maxy, grid_res),
                            colorscale="YlGnBu", colorbar=dict(title="Lluvia (mm)"),
                            hovertemplate="Lluvia: %{z:.0f} mm<extra></extra>",
                            contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                            opacity=0.8, connectgaps=True, line_smoothing=1.3
                        ))
                        add_context_layers_ghost(fig, gdf_target)
                        fig.add_trace(go.Scatter(
                            x=df_final['lon_calc'], y=df_final['lat_calc'], mode='markers',
                            marker=dict(size=6, color='black', line=dict(width=1, color='white')),
                            text=df_final[col_nom], customdata=custom_data,
                            hovertemplate="<b>%{text}</b><br>🌧️: %{marker.color:.0f} mm<br>🏙️: %{customdata[0]}<br>⛰️: %{customdata[1]} m<br>🌊: %{customdata[2]}<extra></extra>",
                            name="Estaciones"
                        ))
                        
                        fig.add_shape(type="rect", x0=minx, y0=miny, x1=maxx, y1=maxy, line=dict(color="Red", width=2, dash="dot"))
                        fig.update_layout(title=tit, height=650, margin=dict(l=0,r=0,t=30,b=0), xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False), plot_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Datos insuficientes para generar mapa (Mínimo 3 estaciones).")
            else:
                st.warning("No se encontraron datos para los parámetros seleccionados.")
                
        except Exception as e:
            st.error(f"Error en procesamiento: {e}")

    with tab_datos:
        if 'df_final' in locals() and not df_final.empty:
            st.subheader("💾 Descargas GIS")
            cols_show = [col_id, col_nom, 'valor']
            if col_cuenca in df_final.columns: cols_show.append(col_cuenca)
            st.dataframe(df_final[cols_show].head(50), use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            gdf_out = gpd.GeoDataFrame(df_final, geometry=gpd.points_from_xy(df_final.lon_calc, df_final.lat_calc), crs="EPSG:4326")
            c1.download_button("🌍 GeoJSON", gdf_out.to_json(), f"isoyetas_{tipo_analisis}.geojson", "application/json")
            if 'grid_z' in locals():
                asc = generar_raster_ascii(grid_z, q_minx, q_miny, (q_maxx-q_minx)/grid_res, grid_res, grid_res)
                c2.download_button("⬛ Raster (.asc)", asc, f"raster_{tipo_analisis}.asc", "text/plain")
            csv = df_final.to_csv(index=False).encode('utf-8')
            c3.download_button("📊 CSV", csv, f"datos_{tipo_analisis}.csv", "text/csv")
else:
    st.info("👈 Seleccione una zona en el sidebar.")# --- 3. FUNCIONES DE SOPORTE ---
@st.cache_data(ttl=3600)
def load_geojson_cached(filename):
    possible_paths = [os.path.join("data", filename), os.path.join("..", "data", filename), os.path.join(os.path.dirname(__file__), '..', 'data', filename)]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                gdf = gpd.read_file(path)
                if gdf.crs and gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                return gdf
            except: continue
    return None

def detectar_columna(df, keywords):
    if df is None or df.empty: return None
    cols_orig = df.columns.tolist()
    for kw in keywords:
        kw_clean = kw.lower().replace('-', '').replace('_', '')
        for col in cols_orig:
            if kw_clean in col.lower().replace('-', '').replace('_', ''): return col
    return None

@st.cache_data(ttl=600)
def obtener_estaciones_enriquecidas():
    try:
        engine = create_engine(st.secrets["DATABASE_URL"])
        q = "SELECT *, ST_Y(geom::geometry) as lat_calc, ST_X(geom::geometry) as lon_calc FROM estaciones"
        df_est = pd.read_sql(q, engine)
        gdf_est = gpd.GeoDataFrame(df_est, geometry=gpd.points_from_xy(df_est.lon_calc, df_est.lat_calc), crs="EPSG:4326")
        
        gdf_cuencas = load_geojson_cached("SubcuencasAinfluencia.geojson")
        if gdf_cuencas is not None:
            col_cuenca_geo = detectar_columna(gdf_cuencas, ['n-nss3', 'n_nss3', 'nnss3', 'nombre', 'subcuenca'])
            if col_cuenca_geo:
                gdf_joined = gpd.sjoin(gdf_est, gdf_cuencas[[col_cuenca_geo, 'geometry']], how='left', predicate='within')
                gdf_joined = gdf_joined.rename(columns={col_cuenca_geo: 'CUENCA_GIS'})
                gdf_joined['CUENCA_GIS'] = gdf_joined['CUENCA_GIS'].fillna('Fuera de Jurisdicción')
                return gdf_joined, True
        return gdf_est, False
    except Exception as e:
        return pd.DataFrame(), False

def generar_raster_ascii(grid_z, minx, miny, cellsize, nrows, ncols):
    header = f"ncols        {ncols}\nnrows        {nrows}\nxllcorner    {minx}\nyllcorner    {miny}\ncellsize     {cellsize}\nNODATA_value -9999\n"
    grid_fill = np.nan_to_num(grid_z.T, nan=-9999)
    body = ""
    for row in np.flipud(grid_fill.T): 
        body += " ".join([f"{val:.2f}" for val in row]) + "\n"
    return header + body

def add_context_layers_ghost(fig, gdf_zona):
    if gdf_zona is None or gdf_zona.empty: return
    try:
        roi = gdf_zona.buffer(0.1)
        gdf_m = load_geojson_cached("MunicipiosAntioquia.geojson")
        gdf_cu = load_geojson_cached("SubcuencasAinfluencia.geojson")
        
        if gdf_m is not None:
            gdf_c = gpd.clip(gdf_m, roi)
            for _, r in gdf_c.iterrows():
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(width=0.5, color='rgba(100,100,100,0.2)', dash='dot'), hoverinfo='skip', showlegend=False))
        if gdf_cu is not None:
            gdf_c = gpd.clip(gdf_cu, roi)
            for _, r in gdf_c.iterrows():
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(width=0.8, color='rgba(50,100,200,0.4)', dash='dash'), hoverinfo='skip', showlegend=False))
    except: pass

def calcular_pronostico(df_hist, target_year):
    """Proyección lineal simple por estación."""
    proyecciones = []
    # df_hist debe tener: station_id, year, value
    for station in df_hist['station_id'].unique():
        datos_est = df_hist[df_hist['station_id'] == station].sort_values('year')
        if len(datos_est) >= 5: # Mínimo 5 años para tendencia
            x = datos_est['year'].values
            y = datos_est['value'].values
            # Regresión Lineal (Grado 1)
            slope, intercept = np.polyfit(x, y, 1)
            pred = (slope * target_year) + intercept
            proyecciones.append({'station_id': station, 'valor': max(0, pred)}) # No lluvia negativa
    return pd.DataFrame(proyecciones)

# --- 4. SIDEBAR & LÓGICA DE DATOS ---
st.sidebar.header("🔍 Configuración de Escenarios")

with st.spinner("Cargando ecosistema espacial..."):
    gdf_meta, exito_cruce = obtener_estaciones_enriquecidas()

if gdf_meta.empty:
    st.error("Error crítico de base de datos.")
    st.stop()

# Detectar columnas
col_id = detectar_columna(gdf_meta, ['id_estacion', 'codigo']) or 'id_estacion'
col_nom = detectar_columna(gdf_meta, ['nom_est', 'nombre']) or 'nom_est'
col_region = detectar_columna(gdf_meta, ['region', 'subregion', 'depto_region'])
col_muni = detectar_columna(gdf_meta, ['municipio', 'mpio'])
col_alt = detectar_columna(gdf_meta, ['alt_est', 'altitud'])
col_cuenca = 'CUENCA_GIS' if 'CUENCA_GIS' in gdf_meta.columns else None

# Filtros Jerárquicos
df_filtered_meta = gdf_meta.copy()

if col_region:
    regs = sorted(df_filtered_meta[col_region].dropna().astype(str).unique())
    sel_reg = st.sidebar.multiselect("📍 Región:", regs)
    if sel_reg: df_filtered_meta = df_filtered_meta[df_filtered_meta[col_region].isin(sel_reg)]

if col_cuenca:
    cuencas = sorted(df_filtered_meta[col_cuenca].dropna().astype(str).unique())
    sel_cuenca = st.sidebar.multiselect("🌊 Cuenca:", cuencas)
    if sel_cuenca: df_filtered_meta = df_filtered_meta[df_filtered_meta[col_cuenca].isin(sel_cuenca)]

if col_muni:
    munis = sorted(df_filtered_meta[col_muni].dropna().astype(str).unique())
    sel_muni = st.sidebar.multiselect("🏙️ Municipio:", munis)
    if sel_muni: df_filtered_meta = df_filtered_meta[df_filtered_meta[col_muni].isin(sel_muni)]

st.sidebar.markdown(f"**Estaciones en zona:** {len(df_filtered_meta)}")
st.sidebar.divider()

# --- CONFIGURACIÓN DE ESCENARIOS (NUEVO) ---
tipo_analisis = st.sidebar.selectbox(
    "📊 Modo de Análisis:", 
    ["Año Específico", "Promedio Multianual", "Mínimo Histórico", "Máximo Histórico", "Pronóstico Futuro"]
)

params_analisis = {}

if tipo_analisis == "Año Específico":
    params_analisis['year'] = st.sidebar.selectbox("📅 Año:", range(2025, 1980, -1))

elif tipo_analisis == "Promedio Multianual":
    rango = st.sidebar.slider("📅 Periodo de Referencia:", 1980, 2025, (1990, 2020))
    params_analisis['start'], params_analisis['end'] = rango

elif tipo_analisis == "Pronóstico Futuro":
    params_analisis['target'] = st.sidebar.slider("🔮 Año a Proyectar:", 2026, 2040, 2026)
    st.sidebar.info("Proyección basada en tendencia lineal histórica.")

# Buffer y Opciones
buffer_km = st.sidebar.slider("📡 Buffer Búsqueda (km):", 0, 50, 10)
buffer_deg = buffer_km / 111.0

c1, c2 = st.sidebar.columns(2)
ignore_zeros = c1.checkbox("🚫 No Ceros", value=True)
ignore_nulls = c2.checkbox("🚫 No Nulos", value=True)

do_interp_temp = False
if complete_series and tipo_analisis == "Año Específico":
    do_interp_temp = st.sidebar.checkbox("🔄 Interpolación Temporal", value=False)

suavidad = st.sidebar.slider("🎨 Suavizado (RBF):", 0.0, 2.0, 0.5)

# --- 5. MOTOR DE CÁLCULO Y VISUALIZACIÓN ---
if len(df_filtered_meta) > 0:
    gdf_target = df_filtered_meta
    minx, miny, maxx, maxy = gdf_target.total_bounds
    q_minx, q_miny = minx - buffer_deg, miny - buffer_deg
    q_maxx, q_maxy = maxx + buffer_deg, maxy + buffer_deg
    
    tab_mapa, tab_datos = st.tabs(["🗺️ Visualización Espacial", "💾 Descargas GIS"])
    
    with tab_mapa:
        try:
            engine = create_engine(st.secrets["DATABASE_URL"])
            df_agg = pd.DataFrame() # Contenedor de resultados
            
            # --- LÓGICA POR ESCENARIO ---
            
            if tipo_analisis == "Año Específico":
                q_data = text(f"""
                    SELECT p.id_estacion_fk as station_id, p.fecha_mes_año as fecha_safe, p.precipitation as val
                    FROM precipitacion_mensual p JOIN estaciones e ON p.id_estacion_fk = e.id_estacion
                    WHERE extract(year from p.fecha_mes_año) = :y
                    AND ST_X(e.geom::geometry) BETWEEN :mx AND :Mx AND ST_Y(e.geom::geometry) BETWEEN :my AND :My
                """)
                df_raw = pd.read_sql(q_data, engine, params={"y": params_analisis['year'], "mx":q_minx, "my":q_miny, "Mx":q_maxx, "My":q_maxy})
                
                if not df_raw.empty:
                    if do_interp_temp and complete_series:
                        df_proc = df_raw.rename(columns={'fecha_safe': 'fecha_mes_año', 'val': 'value'})
                        df_proc['fecha_mes_año'] = pd.to_datetime(df_proc['fecha_mes_año'])
                        with st.spinner("Interpolando tiempo..."):
                            df_filled = complete_series(df_proc)
                            df_agg = df_filled.groupby('station_id')['value'].sum().reset_index()
                    else:
                        df_agg = df_raw.groupby('station_id')['val'].sum().reset_index()
                    df_agg.columns = [col_id, 'valor']

            elif tipo_analisis in ["Promedio Multianual", "Mínimo Histórico", "Máximo Histórico", "Pronóstico Futuro"]:
                # Para estos necesitamos historia anual. Traemos todo el rango relevante.
                q_hist = text(f"""
                    SELECT p.id_estacion_fk as station_id, extract(year from p.fecha_mes_año) as year, SUM(p.precipitation) as value
                    FROM precipitacion_mensual p JOIN estaciones e ON p.id_estacion_fk = e.id_estacion
                    WHERE ST_X(e.geom::geometry) BETWEEN :mx AND :Mx AND ST_Y(e.geom::geometry) BETWEEN :my AND :My
                    GROUP BY 1, 2
                """)
                df_hist = pd.read_sql(q_hist, engine, params={"mx":q_minx, "my":q_miny, "Mx":q_maxx, "My":q_maxy})
                
                if not df_hist.empty:
                    if tipo_analisis == "Promedio Multianual":
                        mask = (df_hist['year'] >= params_analisis['start']) & (df_hist['year'] <= params_analisis['end'])
                        df_agg = df_hist[mask].groupby('station_id')['value'].mean().reset_index()
                    
                    elif tipo_analisis == "Mínimo Histórico":
                        df_agg = df_hist.groupby('station_id')['value'].min().reset_index()
                        
                    elif tipo_analisis == "Máximo Histórico":
                        df_agg = df_hist.groupby('station_id')['value'].max().reset_index()
                        
                    elif tipo_analisis == "Pronóstico Futuro":
                        with st.spinner(f"Calculando tendencias y proyectando al {params_analisis['target']}..."):
                            df_agg = calcular_pronostico(df_hist, params_analisis['target'])
                    
                    if not df_agg.empty:
                        df_agg.columns = [col_id, 'valor']

            # --- RENDERIZADO COMÚN ---
            if not df_agg.empty:
                # Merge con metadatos
                cols_merge = [col_id, col_nom, 'lat_calc', 'lon_calc']
                if col_muni: cols_merge.append(col_muni)
                if col_alt: cols_merge.append(col_alt)
                if col_cuenca: cols_merge.append(col_cuenca)
                cols_merge = list(set(cols_merge))
                
                df_final = pd.merge(df_agg, gdf_meta[cols_merge], on=col_id)
                
                if ignore_zeros: df_final = df_final[df_final['valor'] > 0]
                if ignore_nulls: df_final = df_final.dropna(subset=['valor'])
                
                if len(df_final) >= 3:
                    with st.spinner(f"Generando isoyetas ({len(df_final)} estaciones)..."):
                        grid_res = 200
                        gx, gy = np.mgrid[q_minx:q_maxx:complex(0, grid_res), q_miny:q_maxy:complex(0, grid_res)]
                        rbf = Rbf(df_final['lon_calc'], df_final['lat_calc'], df_final['valor'], function='thin_plate', smooth=suavidad)
                        grid_z = rbf(gx, gy)
                        
                        fig = go.Figure()
                        
                        # Título dinámico
                        if tipo_analisis == "Año Específico": tit = f"Isoyetas Año {params_analisis['year']}"
                        elif tipo_analisis == "Promedio Multianual": tit = f"Isoyetas Promedio {params_analisis['start']}-{params_analisis['end']}"
                        elif tipo_analisis == "Pronóstico Futuro": tit = f"Isoyetas Proyectadas {params_analisis['target']}"
                        else: tit = f"Isoyetas {tipo_analisis}"
                        
                        # Hover Info
                        custom_data = np.stack((
                            df_final[col_muni].fillna('-') if col_muni else ["-"]*len(df_final),
                            df_final[col_alt].fillna(0) if col_alt else [0]*len(df_final),
                            df_final[col_cuenca].fillna('-') if col_cuenca else ["-"]*len(df_final)
                        ), axis=-1)
                        
                        fig.add_trace(go.Contour(
                            z=grid_z.T, x=np.linspace(q_minx, q_maxx, grid_res), y=np.linspace(q_miny, q_maxy, grid_res),
                            colorscale="YlGnBu", colorbar=dict(title="Lluvia (mm)"),
                            hovertemplate="Lluvia: %{z:.0f} mm<extra></extra>",
                            contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                            opacity=0.8, connectgaps=True, line_smoothing=1.3
                        ))
                        add_context_layers_ghost(fig, gdf_target)
                        fig.add_trace(go.Scatter(
                            x=df_final['lon_calc'], y=df_final['lat_calc'], mode='markers',
                            marker=dict(size=6, color='black', line=dict(width=1, color='white')),
                            text=df_final[col_nom], customdata=custom_data,
                            hovertemplate="<b>%{text}</b><br>🌧️: %{marker.color:.0f} mm<br>🏙️: %{customdata[0]}<br>⛰️: %{customdata[1]} m<br>🌊: %{customdata[2]}<extra></extra>",
                            name="Estaciones"
                        ))
                        
                        fig.add_shape(type="rect", x0=minx, y0=miny, x1=maxx, y1=maxy, line=dict(color="Red", width=2, dash="dot"))
                        fig.update_layout(title=tit, height=650, margin=dict(l=0,r=0,t=30,b=0), xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False), plot_bgcolor='white')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Datos insuficientes para generar mapa (Mínimo 3 estaciones).")
            else:
                st.warning("No se encontraron datos para los parámetros seleccionados.")
                
        except Exception as e:
            st.error(f"Error en procesamiento: {e}")

    with tab_datos:
        if 'df_final' in locals() and not df_final.empty:
            st.subheader("💾 Descargas GIS")
            cols_show = [col_id, col_nom, 'valor']
            if col_cuenca in df_final.columns: cols_show.append(col_cuenca)
            st.dataframe(df_final[cols_show].head(50), use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            gdf_out = gpd.GeoDataFrame(df_final, geometry=gpd.points_from_xy(df_final.lon_calc, df_final.lat_calc), crs="EPSG:4326")
            c1.download_button("🌍 GeoJSON", gdf_out.to_json(), f"isoyetas_{tipo_analisis}.geojson", "application/json")
            if 'grid_z' in locals():
                asc = generar_raster_ascii(grid_z, q_minx, q_miny, (q_maxx-q_minx)/grid_res, grid_res, grid_res)
                c2.download_button("⬛ Raster (.asc)", asc, f"raster_{tipo_analisis}.asc", "text/plain")
            csv = df_final.to_csv(index=False).encode('utf-8')
            c3.download_button("📊 CSV", csv, f"datos_{tipo_analisis}.csv", "text/csv")
else:
    st.info("👈 Seleccione una zona en el sidebar.")