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

# --- 2. FUNCIONES DE SOPORTE GIS ---
@st.cache_data(ttl=3600)
def load_geojson_cached(filename):
    # Intentamos varias rutas para asegurar que encuentre el archivo
    possible_paths = [
        os.path.join("data", filename),
        os.path.join(os.path.dirname(__file__), '..', 'data', filename),
        os.path.join("..", "data", filename)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                gdf = gpd.read_file(path)
                if gdf.crs and gdf.crs != "EPSG:4326": 
                    gdf = gdf.to_crs("EPSG:4326")
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
    """
    1. Carga estaciones de BD.
    2. Carga GeoJSON de Cuencas.
    3. Realiza Spatial Join para asignar 'Cuenca' a cada estación.
    """
    debug_msg = []
    try:
        engine = create_engine(st.secrets["DATABASE_URL"])
        
        # A. Cargar Estaciones
        q = "SELECT *, ST_Y(geom::geometry) as lat_calc, ST_X(geom::geometry) as lon_calc FROM estaciones"
        df_est = pd.read_sql(q, engine)
        gdf_est = gpd.GeoDataFrame(
            df_est, 
            geometry=gpd.points_from_xy(df_est.lon_calc, df_est.lat_calc),
            crs="EPSG:4326"
        )
        
        # B. Cargar Cuencas (SubcuencasAinfluencia.geojson)
        gdf_cuencas = load_geojson_cached("SubcuencasAinfluencia.geojson")
        
        if gdf_cuencas is not None:
            # Buscar columna N-NSS3 o similar
            col_cuenca_geo = detectar_columna(gdf_cuencas, ['n-nss3', 'n_nss3', 'nnss3', 'nombre', 'subcuenca'])
            
            if col_cuenca_geo:
                # C. SPATIAL JOIN: Puntos dentro de Polígonos
                gdf_joined = gpd.sjoin(gdf_est, gdf_cuencas[[col_cuenca_geo, 'geometry']], how='left', predicate='within')
                
                # Renombrar para estandarizar
                gdf_joined = gdf_joined.rename(columns={col_cuenca_geo: 'CUENCA_GIS'})
                
                # Rellenar nulos con 'Desconocida'
                gdf_joined['CUENCA_GIS'] = gdf_joined['CUENCA_GIS'].fillna('Fuera de Jurisdicción')
                
                return gdf_joined, True, f"✅ Cuencas cruzadas usando columna: {col_cuenca_geo}"
            else:
                return gdf_est, False, f"⚠️ GeoJSON cargado pero no se halló columna de nombre (Cols: {list(gdf_cuencas.columns)})"
        else:
            return gdf_est, False, "⚠️ No se encontró el archivo SubcuencasAinfluencia.geojson"
            
    except Exception as e:
        return pd.DataFrame(), False, f"❌ Error crítico: {str(e)}"

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

# --- 3. SIDEBAR Y CARGA DE DATOS ---
st.sidebar.header("🔍 Filtros & Configuración")

# A. Carga de Datos Enriquecidos
with st.spinner("Cargando y cruzando capas espaciales..."):
    gdf_meta, exito_cruce, msg_debug = obtener_estaciones_enriquecidas()

# Debug opcional en sidebar para ver qué pasó con las cuencas
with st.sidebar.expander("🛠️ Diagnóstico de Datos"):
    st.write(msg_debug)
    if not gdf_meta.empty:
        st.write("Columnas disponibles:", list(gdf_meta.columns))

if gdf_meta.empty:
    st.error("Error cargando base de datos de estaciones.")
    st.stop()

# B. Configuración de Filtros
try:
    col_id = detectar_columna(gdf_meta, ['id_estacion', 'codigo']) or 'id_estacion'
    col_nom = detectar_columna(gdf_meta, ['nom_est', 'nombre']) or 'nom_est'
    col_region = detectar_columna(gdf_meta, ['region', 'subregion', 'depto_region'])
    col_muni = detectar_columna(gdf_meta, ['municipio', 'mpio'])
    
    # La columna de Cuenca es la calculada por el Spatial Join
    col_cuenca = 'CUENCA_GIS' if 'CUENCA_GIS' in gdf_meta.columns else None

    df_filtered_meta = gdf_meta.copy()

    # 1. Filtro Región
    if col_region:
        regiones = sorted(df_filtered_meta[col_region].dropna().astype(str).unique())
        sel_region = st.sidebar.multiselect("📍 Región:", regiones)
        if sel_region:
            df_filtered_meta = df_filtered_meta[df_filtered_meta[col_region].isin(sel_region)]

    # 2. Filtro Cuenca (Ahora sí debe funcionar)
    if col_cuenca:
        cuencas = sorted(df_filtered_meta[col_cuenca].dropna().astype(str).unique())
        sel_cuenca = st.sidebar.multiselect("🌊 Cuenca:", cuencas)
        if sel_cuenca:
            df_filtered_meta = df_filtered_meta[df_filtered_meta[col_cuenca].isin(sel_cuenca)]
    else:
        st.sidebar.warning("Filtro de Cuenca no disponible (falló cruce espacial).")

    # 3. Filtro Municipio
    if col_muni:
        munis = sorted(df_filtered_meta[col_muni].dropna().astype(str).unique())
        sel_muni = st.sidebar.multiselect("🏙️ Municipio:", munis)
        if sel_muni:
            df_filtered_meta = df_filtered_meta[df_filtered_meta[col_muni].isin(sel_muni)]

    st.sidebar.markdown(f"**Estaciones en zona:** {len(df_filtered_meta)}")
    
    st.sidebar.divider()
    buffer_deg = st.sidebar.slider("📡 Buffer Búsqueda (°):", 0.0, 0.5, 0.1, 0.01)
    year_iso = st.sidebar.selectbox("📅 Año de Análisis:", range(2025, 1980, -1))
    
    c1, c2 = st.sidebar.columns(2)
    ignore_zeros = c1.checkbox("🚫 No Ceros", value=True)
    ignore_nulls = c2.checkbox("🚫 No Nulos", value=True)
    
    do_interp_temp = False
    if complete_series:
        do_interp_temp = st.sidebar.checkbox("🔄 Interpolación Temporal", value=False)
    
    suavidad = st.sidebar.slider("🎨 Suavizado (RBF):", 0.0, 2.0, 0.5)

except Exception as e:
    st.error(f"Error en configuración de filtros: {e}")
    st.stop()

# --- 4. MAPA Y LÓGICA ---
if len(df_filtered_meta) > 0:
    gdf_target = df_filtered_meta
    minx, miny, maxx, maxy = gdf_target.total_bounds
    
    q_minx, q_miny = minx - buffer_deg, miny - buffer_deg
    q_maxx, q_maxy = maxx + buffer_deg, maxy + buffer_deg
    
    tab_mapa, tab_datos = st.tabs(["🗺️ Visualización Espacial", "💾 Descargas GIS"])
    
    with tab_mapa:
        try:
            engine = create_engine(st.secrets["DATABASE_URL"])
            
            # SOLUCIÓN ERROR FECHA: Usamos alias 'fecha_safe' para extracción segura
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
            
            df_raw = pd.read_sql(q_data, engine, params={
                "anio": year_iso, "minx": q_minx, "miny": q_miny, "maxx": q_maxx, "maxy": q_maxy
            })
            
            if not df_raw.empty:
                # SOLUCIÓN ERROR FECHA PARTE 2: Renombrar a lo que complete_series espera
                # Si complete_series espera 'fecha_mes_año', se lo damos.
                
                if do_interp_temp and complete_series:
                    # Renombramos 'fecha_safe' -> 'fecha_mes_año' para que el módulo esté feliz
                    df_proc = df_raw.rename(columns={'fecha_safe': 'fecha_mes_año'})
                    
                    # Asegurar tipo datetime
                    df_proc['fecha_mes_año'] = pd.to_datetime(df_proc['fecha_mes_año'])
                    
                    with st.spinner("🔄 Rellenando huecos temporales..."):
                        # Llamamos al módulo externo con el nombre de columna correcto
                        df_filled = complete_series(df_proc) 
                        
                        # Agrupar. Nota: complete_series suele devolver el DF relleno.
                        # Asumimos que mantiene la columna de fecha.
                        df_agg = df_filled.groupby(col_id)['precipitation'].sum().reset_index()
                        df_agg.columns = [col_id, 'valor']
                else:
                    # Sin interpolación, suma directa
                    df_agg = df_raw.groupby(col_id)['precipitation'].sum().reset_index()
                    df_agg.columns = [col_id, 'valor']

                # Merge usando gdf_meta (el dataset completo enriquecido)
                df_final = pd.merge(df_agg, gdf_meta, on=col_id)
                
                # Filtros de valor
                if ignore_zeros: df_final = df_final[df_final['valor'] > 0]
                if ignore_nulls: df_final = df_final.dropna(subset=['valor'])
                
                if len(df_final) >= 3:
                    with st.spinner(f"Generando isoyetas ({len(df_final)} estaciones)..."):
                        grid_res = 200
                        gx, gy = np.mgrid[q_minx:q_maxx:complex(0, grid_res), q_miny:q_maxy:complex(0, grid_res)]
                        rbf = Rbf(df_final['lon_calc'], df_final['lat_calc'], df_final['valor'], function='thin_plate', smooth=suavidad)
                        grid_z = rbf(gx, gy)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Contour(
                            z=grid_z.T, x=np.linspace(q_minx, q_maxx, grid_res), y=np.linspace(q_miny, q_maxy, grid_res),
                            colorscale="YlGnBu", colorbar=dict(title="Lluvia (mm)"),
                            hovertemplate="Precipitación: %{z:.0f} mm<extra></extra>",
                            contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                            opacity=0.8, connectgaps=True, line_smoothing=1.3
                        ))
                        add_context_layers_ghost(fig, gdf_target)
                        fig.add_trace(go.Scatter(
                            x=df_final['lon_calc'], y=df_final['lat_calc'], mode='markers',
                            marker=dict(size=5, color='black', line=dict(width=1, color='white')),
                            text=df_final[col_nom] + ': ' + df_final['valor'].round(0).astype(str) + ' mm',
                            hoverinfo='text', name="Estaciones"
                        ))
                        fig.add_shape(type="rect", x0=minx, y0=miny, x1=maxx, y1=maxy, line=dict(color="Red", width=2, dash="dot"))
                        fig.update_layout(height=650, margin=dict(l=0,r=0,t=20,b=0), xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False), plot_bgcolor='white', title=f"Isoyetas {year_iso}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Datos insuficientes. Intente aumentar el Buffer.")
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