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

st.title("🗺️ Mapas de Isoyetas de Alta Definición (RBF)")

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
    """)

# --- 3. FUNCIONES DE SOPORTE GIS ---
@st.cache_data(ttl=3600)
def load_geojson_cached(filename):
    possible_paths = [
        os.path.join("data", filename),
        os.path.join(os.path.dirname(__file__), '..', 'data', filename),
        os.path.join("..", "data", filename)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                gdf = gpd.read_file(path)
                if gdf.crs and gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
                return gdf
            except: continue
    return None

def detectar_columna(df, keywords):
    """Busca columnas ignorando mayúsculas y caracteres especiales."""
    if df is None or df.empty: return None
    cols_orig = df.columns.tolist()
    for kw in keywords:
        kw_clean = kw.lower().replace('-', '').replace('_', '')
        for col in cols_orig:
            col_clean = col.lower().replace('-', '').replace('_', '')
            if kw_clean in col_clean:
                return col
    return None

@st.cache_data(ttl=600)
def obtener_estaciones_enriquecidas():
    """Carga estaciones y cruza con Cuencas."""
    try:
        engine = create_engine(st.secrets["DATABASE_URL"])
        # Traemos alt_est para el popup
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

# --- 4. SIDEBAR Y LÓGICA ---
st.sidebar.header("🔍 Filtros & Configuración")

with st.spinner("Cargando capas espaciales..."):
    gdf_meta, exito_cruce = obtener_estaciones_enriquecidas()

if gdf_meta.empty:
    st.error("Error cargando base de datos.")
    st.stop()

# Configuración Filtros
try:
    col_id = detectar_columna(gdf_meta, ['id_estacion', 'codigo']) or 'id_estacion'
    col_nom = detectar_columna(gdf_meta, ['nom_est', 'nombre']) or 'nom_est'
    col_region = detectar_columna(gdf_meta, ['region', 'subregion', 'depto_region'])
    col_muni = detectar_columna(gdf_meta, ['municipio', 'mpio'])
    col_alt = detectar_columna(gdf_meta, ['alt_est', 'altitud', 'elevacion']) # Detectar altura
    col_cuenca = 'CUENCA_GIS' if 'CUENCA_GIS' in gdf_meta.columns else None

    df_filtered_meta = gdf_meta.copy()

    if col_region:
        regiones = sorted(df_filtered_meta[col_region].dropna().astype(str).unique())
        sel_region = st.sidebar.multiselect("📍 Región:", regiones)
        if sel_region: df_filtered_meta = df_filtered_meta[df_filtered_meta[col_region].isin(sel_region)]

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
    
    # --- BUFFER EN KILOMETROS (MEJORA) ---
    buffer_km = st.sidebar.slider("📡 Buffer Búsqueda (km):", 0, 50, 10, help="Extiende la búsqueda para interpolar bordes.")
    # Conversión aprox: 1 grado ~ 111 km
    buffer_deg = buffer_km / 111.0
    
    year_iso = st.sidebar.selectbox("📅 Año de Análisis:", range(2025, 1980, -1))
    
    c1, c2 = st.sidebar.columns(2)
    ignore_zeros = c1.checkbox("🚫 No Ceros", value=True)
    ignore_nulls = c2.checkbox("🚫 No Nulos", value=True)
    
    do_interp_temp = False
    if complete_series:
        do_interp_temp = st.sidebar.checkbox("🔄 Interpolación Temporal", value=False)
    
    suavidad = st.sidebar.slider("🎨 Suavizado (RBF):", 0.0, 2.0, 0.5)

except Exception as e:
    st.error(f"Error configuración: {e}")
    st.stop()

# --- 5. RENDERIZADO ---
if len(df_filtered_meta) > 0:
    gdf_target = df_filtered_meta
    minx, miny, maxx, maxy = gdf_target.total_bounds
    
    q_minx, q_miny = minx - buffer_deg, miny - buffer_deg
    q_maxx, q_maxy = maxx + buffer_deg, maxy + buffer_deg
    
    tab_mapa, tab_datos = st.tabs(["🗺️ Visualización Espacial", "💾 Descargas GIS"])
    
    with tab_mapa:
        try:
            engine = create_engine(st.secrets["DATABASE_URL"])
            q_data = text(f"""
                SELECT p.id_estacion_fk as {col_id}, 
                       p.fecha_mes_año as fecha_safe, 
                       p.precipitation
                FROM precipitacion_mensual p
                JOIN estaciones e ON p.id_estacion_fk = e.id_estacion
                WHERE extract(year from p.fecha_mes_año) = :anio
                AND ST_X(e.geom::geometry) BETWEEN :minx AND :maxx
                AND ST_Y(e.geom::geometry) BETWEEN :miny AND :maxy
            """)
            
            df_raw = pd.read_sql(q_data, engine, params={"anio": year_iso, "minx": q_minx, "miny": q_miny, "maxx": q_maxx, "maxy": q_maxy})
            
            if not df_raw.empty:
                if do_interp_temp and complete_series:
                    df_proc = df_raw.rename(columns={col_id: 'station_id', 'fecha_safe': 'fecha_mes_año', 'precipitation': 'value'})
                    df_proc['fecha_mes_año'] = pd.to_datetime(df_proc['fecha_mes_año'])
                    with st.spinner("Rellenando series..."):
                        df_filled = complete_series(df_proc) 
                        df_agg = df_filled.groupby('station_id')['value'].sum().reset_index()
                        df_agg.columns = [col_id, 'valor']
                else:
                    df_agg = df_raw.groupby(col_id)['precipitation'].sum().reset_index()
                    df_agg.columns = [col_id, 'valor']

                # Merge con todas las columnas necesarias para el Popup
                cols_to_merge = [col_id, col_nom, 'lat_calc', 'lon_calc']
                if col_muni: cols_to_merge.append(col_muni)
                if col_alt: cols_to_merge.append(col_alt)
                if col_cuenca: cols_to_merge.append(col_cuenca)
                
                # Eliminamos duplicados por si acaso
                cols_to_merge = list(set(cols_to_merge))
                
                df_final = pd.merge(df_agg, gdf_meta[cols_to_merge], on=col_id)
                
                if ignore_zeros: df_final = df_final[df_final['valor'] > 0]
                if ignore_nulls: df_final = df_final.dropna(subset=['valor'])
                
                if len(df_final) >= 3:
                    with st.spinner(f"Interpolando {len(df_final)} estaciones..."):
                        grid_res = 200
                        gx, gy = np.mgrid[q_minx:q_maxx:complex(0, grid_res), q_miny:q_maxy:complex(0, grid_res)]
                        rbf = Rbf(df_final['lon_calc'], df_final['lat_calc'], df_final['valor'], function='thin_plate', smooth=suavidad)
                        grid_z = rbf(gx, gy)
                        
                        fig = go.Figure()
                        
                        # CONSTRUCCIÓN DEL POPUP (HOVER TEXT)
                        hover_template = (
                            "<b>%{text}</b><br>" +
                            "🌧️ Lluvia: %{marker.color:.0f} mm<br>" +
                            "🏙️ Mpio: %{customdata[0]}<br>" +
                            "⛰️ Alt: %{customdata[1]} m<br>" +
                            "🌊 Cuenca: %{customdata[2]}<extra></extra>"
                        )
                        
                        # Preparamos datos auxiliares (Custom Data) para el hover
                        # Orden: [Municipio, Altura, Cuenca]
                        c_muni = df_final[col_muni].fillna('N/A') if col_muni else ["-"]*len(df_final)
                        c_alt = df_final[col_alt].fillna(0).astype(int) if col_alt else [0]*len(df_final)
                        c_cuenca = df_final[col_cuenca].fillna('N/A') if col_cuenca else ["-"]*len(df_final)
                        custom_data = np.stack((c_muni, c_alt, c_cuenca), axis=-1)

                        # Isoyetas
                        fig.add_trace(go.Contour(
                            z=grid_z.T, x=np.linspace(q_minx, q_maxx, grid_res), y=np.linspace(q_miny, q_maxy, grid_res),
                            colorscale="YlGnBu", colorbar=dict(title="Lluvia (mm)"),
                            hovertemplate="Lluvia Interpolada: %{z:.0f} mm<extra></extra>",
                            contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                            opacity=0.8, connectgaps=True, line_smoothing=1.3
                        ))
                        add_context_layers_ghost(fig, gdf_target)
                        
                        # Puntos Estaciones con Popup Mejorado
                        fig.add_trace(go.Scatter(
                            x=df_final['lon_calc'], y=df_final['lat_calc'], mode='markers',
                            marker=dict(size=6, color='black', line=dict(width=1, color='white')),
                            text=df_final[col_nom],
                            customdata=custom_data,
                            hovertemplate=hover_template,
                            name="Estaciones"
                        ))
                        
                        fig.add_shape(type="rect", x0=minx, y0=miny, x1=maxx, y1=maxy, line=dict(color="Red", width=2, dash="dot"))
                        fig.update_layout(height=650, margin=dict(l=0,r=0,t=20,b=0), xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False), plot_bgcolor='white', title=f"Isoyetas {year_iso}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Datos insuficientes. Intente aumentar el Buffer (km).")
            else:
                st.warning("No hay datos para esta zona/año.")
        except Exception as e:
            st.error(f"Error procesando mapa: {e}")

    with tab_datos:
        if 'df_final' in locals() and not df_final.empty:
            st.subheader("💾 Descargas GIS")
            cols_show = [col_id, col_nom, 'valor']
            if col_cuenca in df_final.columns: cols_show.append(col_cuenca)
            st.dataframe(df_final[cols_show].head(50), use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            gdf_out = gpd.GeoDataFrame(df_final, geometry=gpd.points_from_xy(df_final.lon_calc, df_final.lat_calc), crs="EPSG:4326")
            c1.download_button("🌍 GeoJSON", gdf_out.to_json(), f"estaciones_{year_iso}.geojson", "application/json")
            if 'grid_z' in locals():
                asc = generar_raster_ascii(grid_z, q_minx, q_miny, (q_maxx-q_minx)/grid_res, grid_res, grid_res)
                c2.download_button("⬛ Raster (.asc)", asc, f"isoyetas_{year_iso}.asc", "text/plain")
            csv = df_final.to_csv(index=False).encode('utf-8')
            c3.download_button("📊 CSV", csv, f"datos_{year_iso}.csv", "text/csv")
else:
    st.info("👈 Utilice el sidebar para seleccionar una zona.")