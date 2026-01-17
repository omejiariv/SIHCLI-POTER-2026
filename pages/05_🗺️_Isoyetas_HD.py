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
    # Intentamos importar la función de completado de series si existe
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
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
    if os.path.exists(filepath):
        try:
            gdf = gpd.read_file(filepath)
            if gdf.crs and gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
            return gdf
        except: pass
    return None

def get_name_from_row_v2(row, type_layer):
    cols = row.index.str.lower()
    if type_layer == 'muni':
        for c in ['mpio_cnmbr', 'nombre', 'municipio', 'mpio_nomb']:
            if c in cols: return row[c]
    elif type_layer == 'cuenca':
        for c in ['n-nss3', 'subc_lbl', 'nom_cuenca', 'nombre', 'cuenca']:
            if c in cols: return row[c]
    return ""

def detectar_columna(df, keywords):
    """Busca una columna en el DF que coincida con las keywords."""
    cols_orig = df.columns
    cols_lower = [c.lower() for c in cols_orig]
    
    for kw in keywords:
        kw_lower = kw.lower()
        for i, col_name_lower in enumerate(cols_lower):
            # Coincidencia exacta o parcial
            if kw_lower == col_name_lower or kw_lower in col_name_lower:
                return cols_orig[i] # Retorna el nombre real (con mayúsculas originales)
    return None

def generar_raster_ascii(grid_z, minx, miny, cellsize, nrows, ncols):
    """Genera un archivo Raster formato ESRI ASCII Grid (.asc)."""
    header = f"""ncols        {ncols}
nrows        {nrows}
xllcorner    {minx}
yllcorner    {miny}
cellsize     {cellsize}
NODATA_value -9999
"""
    grid_fill = np.nan_to_num(grid_z.T, nan=-9999)
    body = ""
    # Flip vertical para coincidir con coordenadas cartesianas vs matriz
    for row in np.flipud(grid_fill.T): 
        body += " ".join([f"{val:.2f}" for val in row]) + "\n"
    return header + body

def add_context_layers_ghost(fig, gdf_zona):
    try:
        if gdf_zona is None or gdf_zona.empty: return
        roi = gdf_zona.buffer(0.1) 
        
        gdf_m = load_geojson_cached("MunicipiosAntioquia.geojson")
        gdf_cu = load_geojson_cached("SubcuencasAinfluencia.geojson")
        
        if gdf_m is not None:
            gdf_c = gpd.clip(gdf_m, roi)
            for _, r in gdf_c.iterrows():
                name = get_name_from_row_v2(r, 'muni')
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', 
                        line=dict(width=0.5, color='rgba(100, 100, 100, 0.2)', dash='dot'), 
                        hoverinfo='text', text=f"Mpio: {name}", showlegend=False
                    ))
        
        if gdf_cu is not None:
            gdf_c = gpd.clip(gdf_cu, roi)
            for _, r in gdf_c.iterrows():
                name = get_name_from_row_v2(r, 'cuenca')
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', 
                        line=dict(width=0.8, color='rgba(50, 100, 200, 0.4)', dash='dash'), 
                        hoverinfo='text', text=f"Cuenca: {name}", showlegend=False
                    ))
    except Exception as e: print(f"Ghost Error: {e}")

# --- 3. SIDEBAR DE HIDROLOGÍA ---
st.sidebar.header("🔍 Filtros & Configuración")

try:
    engine = create_engine(st.secrets["DATABASE_URL"])
    
    # CONSULTA BASE METADATOS
    q_meta = """
        SELECT *, 
               ST_Y(geom::geometry) as lat_calc, 
               ST_X(geom::geometry) as lon_calc 
        FROM estaciones
    """
    df_meta_raw = pd.read_sql(q_meta, engine)
    
    # Detección de columnas
    col_id = detectar_columna(df_meta_raw, ['id_estacion', 'codigo']) or 'id_estacion'
    col_nom = detectar_columna(df_meta_raw, ['nom_est', 'nombre']) or 'nom_est'
    
    # Columnas Geográficas
    col_region = detectar_columna(df_meta_raw, ['region', 'subregion', 'zona'])
    # AQUÍ ESTÁ EL CAMBIO CLAVE: Priorizamos 'n-nss3' explícitamente
    col_cuenca = detectar_columna(df_meta_raw, ['n-nss3', 'n_nss3', 'cuenca', 'basin'])
    col_muni = detectar_columna(df_meta_raw, ['municipio', 'mpio', 'ciud'])

    # --- A. FILTROS JERÁRQUICOS ---
    df_filtered_meta = df_meta_raw.copy()

    # 1. Región
    sel_region = []
    if col_region:
        regiones = sorted(df_filtered_meta[col_region].dropna().astype(str).unique())
        sel_region = st.sidebar.multiselect(f"📍 Región:", regiones)
        if sel_region:
            df_filtered_meta = df_filtered_meta[df_filtered_meta[col_region].isin(sel_region)]

    # 2. Cuenca (Ahora debería aparecer si N-NSS3 existe)
    sel_cuenca = []
    if col_cuenca:
        # Filtrar opciones basadas en selección anterior
        cuencas = sorted(df_filtered_meta[col_cuenca].dropna().astype(str).unique())
        # Usamos el nombre real de la columna en el label
        sel_cuenca = st.sidebar.multiselect(f"🌊 Cuenca ({col_cuenca}):", cuencas)
        if sel_cuenca:
            df_filtered_meta = df_filtered_meta[df_filtered_meta[col_cuenca].isin(sel_cuenca)]
    else:
        # Debug para saber si falla la detección
        st.sidebar.warning("Columna 'Cuenca/N-NSS3' no detectada.")

    # 3. Municipio
    sel_muni = []
    if col_muni:
        munis = sorted(df_filtered_meta[col_muni].dropna().astype(str).unique())
        sel_muni = st.sidebar.multiselect("🏙️ Municipio:", munis)
        if sel_muni:
            df_filtered_meta = df_filtered_meta[df_filtered_meta[col_muni].isin(sel_muni)]

    st.sidebar.markdown(f"**Estaciones en zona:** {len(df_filtered_meta)}")
    
    # --- B. BUFFER DE BÚSQUEDA ---
    st.sidebar.divider()
    buffer_deg = st.sidebar.slider("📡 Buffer Búsqueda (°):", 0.0, 0.5, 0.1, 0.01, help="Amplía búsqueda a vecinos.")
    
    # --- C. TIEMPO Y PROCESAMIENTO ---
    year_iso = st.sidebar.selectbox("📅 Año de Análisis:", range(2025, 1980, -1))
    
    c1, c2 = st.sidebar.columns(2)
    ignore_zeros = c1.checkbox("🚫 No Ceros", value=True)
    ignore_nulls = c2.checkbox("🚫 No Nulos", value=True)
    
    do_interp_temp = False
    if complete_series:
        do_interp_temp = st.sidebar.checkbox("🔄 Interpolación Temporal", value=False, help="Rellena huecos en series.")
    
    suavidad = st.sidebar.slider("🎨 Suavizado Visual (RBF):", 0.0, 2.0, 0.5)

except Exception as e:
    st.error(f"Error cargando metadatos: {e}")
    st.stop()

# --- 4. LÓGICA ESPACIAL INTELIGENTE ---
if len(df_filtered_meta) > 0:
    # Asegurar coordenadas
    if 'lat' not in df_filtered_meta.columns: df_filtered_meta['lat'] = df_filtered_meta['lat_calc']
    if 'lon' not in df_filtered_meta.columns: df_filtered_meta['lon'] = df_filtered_meta['lon_calc']
    
    # 1. Definir Zona Objetivo (TARGET)
    gdf_target = gpd.GeoDataFrame(df_filtered_meta, geometry=gpd.points_from_xy(df_filtered_meta.lon, df_filtered_meta.lat), crs="EPSG:4326")
    minx, miny, maxx, maxy = gdf_target.total_bounds
    
    # 2. Expandir Búsqueda (QUERY)
    q_minx, q_miny = minx - buffer_deg, miny - buffer_deg
    q_maxx, q_maxy = maxx + buffer_deg, maxy + buffer_deg
    
    tab_mapa, tab_datos = st.tabs(["🗺️ Visualización Espacial", "💾 Descargas GIS"])
    
    with tab_mapa:
        try:
            # 3. Consulta Espacial Ampliada
            # CORRECCIÓN CLAVE: Usamos ALIAS 'fecha' para evitar error con 'ñ' en Pandas
            q_data = text(f"""
                SELECT p.id_estacion_fk as {col_id}, 
                       p.fecha_mes_año as fecha, 
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
                # 4. Procesamiento Temporal
                if do_interp_temp and complete_series:
                    # Renombrar usando 'fecha' (sin ñ) que viene del alias SQL
                    # Mapeamos: col_id -> station_id, 'fecha' -> date, 'precipitation' -> value
                    df_proc = df_raw.rename(columns={col_id: 'station_id', 'fecha': 'date', 'precipitation': 'value'})
                    
                    # Asegurar formato fecha
                    df_proc['date'] = pd.to_datetime(df_proc['date'])
                    
                    with st.spinner("🔄 Interpolando series temporales..."):
                        df_filled = complete_series(df_proc) 
                        # Sumar el año completo
                        df_agg = df_filled.groupby('station_id')['value'].sum().reset_index()
                        df_agg.columns = [col_id, 'valor']
                else:
                    # Suma directa sin interpolar
                    df_agg = df_raw.groupby(col_id)['precipitation'].sum().reset_index()
                    df_agg.columns = [col_id, 'valor']

                # 5. Merge con Metadatos
                # Aseguramos coords en meta_raw
                if 'lat' not in df_meta_raw.columns: df_meta_raw['lat'] = df_meta_raw['lat_calc']
                if 'lon' not in df_meta_raw.columns: df_meta_raw['lon'] = df_meta_raw['lon_calc']
                
                df_final = pd.merge(df_agg, df_meta_raw, on=col_id)
                
                # Filtros de valor
                if ignore_zeros: df_final = df_final[df_final['valor'] > 0]
                if ignore_nulls: df_final = df_final.dropna(subset=['valor'])
                
                if len(df_final) >= 3:
                    with st.spinner(f"Interpolando {len(df_final)} estaciones..."):
                        # Grid
                        grid_res = 200
                        gx, gy = np.mgrid[q_minx:q_maxx:complex(0, grid_res), q_miny:q_maxy:complex(0, grid_res)]
                        
                        rbf = Rbf(df_final['lon'], df_final['lat'], df_final['valor'], function='thin_plate', smooth=suavidad)
                        grid_z = rbf(gx, gy)
                        
                        fig = go.Figure()
                        
                        # Isoyetas
                        fig.add_trace(go.Contour(
                            z=grid_z.T, x=np.linspace(q_minx, q_maxx, grid_res), y=np.linspace(q_miny, q_maxy, grid_res),
                            colorscale="YlGnBu", colorbar=dict(title="Lluvia (mm)"),
                            hovertemplate="Precipitación: %{z:.0f} mm<extra></extra>",
                            contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                            opacity=0.8, connectgaps=True, line_smoothing=1.3
                        ))
                        
                        # Contexto
                        add_context_layers_ghost(fig, gdf_target)
                        
                        # Puntos
                        fig.add_trace(go.Scatter(
                            x=df_final['lon'], y=df_final['lat'], mode='markers',
                            marker=dict(size=5, color='black', line=dict(width=1, color='white')),
                            text=df_final[col_nom] + ': ' + df_final['valor'].round(0).astype(str) + ' mm',
                            hoverinfo='text', name="Estaciones"
                        ))
                        
                        # Zona Seleccionada (Referencia)
                        fig.add_shape(type="rect",
                            x0=minx, y0=miny, x1=maxx, y1=maxy,
                            line=dict(color="Red", width=2, dash="dot"),
                        )

                        fig.update_layout(
                            height=650, margin=dict(l=0,r=0,t=20,b=0),
                            xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False),
                            plot_bgcolor='white',
                            title=f"Isoyetas {year_iso} (Marco Rojo = Zona Seleccionada)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Datos insuficientes. Intente aumentar el Buffer.")
            else:
                st.warning("No hay datos de precipitación para esta zona/año.")
                
        except Exception as e:
            st.error(f"Error procesando mapa: {e}")

    with tab_datos:
        if 'df_final' in locals() and not df_final.empty:
            st.subheader("💾 Centro de Descargas GIS")
            
            # Tabla
            cols_show = [col_id, col_nom, 'valor']
            if col_muni in df_final.columns: cols_show.append(col_muni)
            st.dataframe(df_final[cols_show].head(100), use_container_width=True)
            
            col_d1, col_d2, col_d3 = st.columns(3)
            
            # GeoJSON
            gdf_out = gpd.GeoDataFrame(df_final, geometry=gpd.points_from_xy(df_final.lon, df_final.lat), crs="EPSG:4326")
            col_d1.download_button("🌍 Descargar GeoJSON", 
                                   gdf_out.to_json(), f"estaciones_{year_iso}.geojson", "application/json")
            
            # Raster
            if 'grid_z' in locals():
                asc_content = generar_raster_ascii(grid_z, q_minx, q_miny, (q_maxx-q_minx)/grid_res, grid_res, grid_res)
                col_d2.download_button("⬛ Descargar Raster (.asc)", 
                                       asc_content, f"isoyetas_{year_iso}.asc", "text/plain")
            
            # CSV
            csv = df_final.to_csv(index=False).encode('utf-8')
            col_d3.download_button("📊 Descargar CSV", csv, f"datos_{year_iso}.csv", "text/csv")

else:
    st.info("👈 Utilice el sidebar para seleccionar una zona.")