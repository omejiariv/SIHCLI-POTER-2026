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

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Isoyetas HD", page_icon="🗺️", layout="wide")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules.config import Config
except:
    pass

st.title("🗺️ Mapas de Isoyetas de Alta Definición (RBF)")

# --- 2. FUNCIONES DE SOPORTE ---
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
    """Extrae nombre de geojson de forma segura."""
    cols = row.index.str.lower()
    if type_layer == 'muni':
        for c in ['mpio_cnmbr', 'nombre', 'municipio', 'mpio_nomb']:
            if c in cols: return row[c]
    elif type_layer == 'cuenca':
        for c in ['n-nss3', 'subc_lbl', 'nom_cuenca', 'nombre', 'cuenca']:
            if c in cols: return row[c]
    return ""

def detectar_columna(df, keywords):
    """Busca una columna en el DF que coincida con las keywords (insensible a mayúsculas)."""
    cols_lower = [c.lower() for c in df.columns]
    for kw in keywords:
        kw_lower = kw.lower()
        # Buscamos coincidencia exacta o parcial
        for i, col_name in enumerate(cols_lower):
            if kw_lower in col_name:
                return df.columns[i] # Retorna el nombre real de la columna
    return None

def add_context_layers_ghost(fig, gdf_zona):
    try:
        if gdf_zona is None or gdf_zona.empty: return
        roi = gdf_zona.buffer(0.05)
        
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

# --- 3. FICHA TÉCNICA ---
with st.expander("📘 Ficha Técnica: Metodología e Interpretación", expanded=False):
    st.markdown("""
    ### 1. Concepto
    Mapas de isolíneas de precipitación interpoladas mediante algoritmos matemáticos (RBF) para estimar lluvia en zonas no instrumentadas.

    ### 2. Metodología
    **Thin-Plate Spline (RBF):** Simula una superficie flexible que se ajusta suavemente a los datos, ideal para variables continuas como la lluvia.

    ### 3. Interpretación
    * **🟦 Azul Oscuro:** Alta precipitación.
    * **🟨 Amarillo:** Baja precipitación.
    """)

# --- 4. SIDEBAR DE HIDROLOGÍA ---
st.sidebar.header("🔍 Filtros Hidrológicos")

try:
    engine = create_engine(st.secrets["DATABASE_URL"])
    
    # CONSULTA GENERAL (Traemos todo para buscar N-NSS3)
    q_meta = """
        SELECT *, 
               ST_Y(geom::geometry) as lat_calc, 
               ST_X(geom::geometry) as lon_calc 
        FROM estaciones
    """
    df_meta = pd.read_sql(q_meta, engine)
    
    # Detección de columnas
    col_id = detectar_columna(df_meta, ['id_estacion', 'codigo']) or 'id_estacion'
    col_nom = detectar_columna(df_meta, ['nom_est', 'nombre']) or 'nom_est'
    
    # 1. Filtro CUENCA (Prioridad: N-NSS3)
    col_cuenca = detectar_columna(df_meta, ['n-nss3', 'n_nss3', 'cuenca', 'basin', 'zona_hidro'])
    
    sel_cuenca = []
    if col_cuenca:
        cuencas_disp = sorted(df_meta[col_cuenca].astype(str).unique())
        sel_cuenca = st.sidebar.multiselect(f"🌊 Cuenca ({col_cuenca}):", cuencas_disp)
        if sel_cuenca:
            df_meta = df_meta[df_meta[col_cuenca].isin(sel_cuenca)]
    else:
        st.sidebar.warning("⚠️ No se encontró la columna 'N-NSS3' ni similares.")

    # 2. Filtro MUNICIPIO
    col_muni = detectar_columna(df_meta, ['municipio', 'mpio', 'ciud'])
    
    sel_muni = []
    if col_muni:
        munis_disp = sorted(df_meta[col_muni].astype(str).unique())
        sel_muni = st.sidebar.multiselect("🏙️ Municipio:", munis_disp)
        if sel_muni:
            df_meta = df_meta[df_meta[col_muni].isin(sel_muni)]

    ids_filtrados = tuple(df_meta[col_id].unique())
    st.sidebar.markdown(f"**Estaciones encontradas:** {len(df_meta)}")
    
    # 3. Parametros
    st.sidebar.divider()
    year_iso = st.sidebar.selectbox("📅 Año de Análisis:", range(2025, 1980, -1))
    
    c1, c2 = st.sidebar.columns(2)
    ignore_zeros = c1.checkbox("🚫 No Ceros", value=True)
    ignore_nulls = c2.checkbox("🚫 No Nulos", value=True)
    suavidad = st.sidebar.slider("🎨 Suavizado (RBF):", 0.0, 2.0, 0.5)

except Exception as e:
    st.error(f"Error cargando metadatos: {e}")
    st.stop()

# --- 5. LÓGICA PRINCIPAL ---
if len(df_meta) > 0:
    # Asegurar Lat/Lon
    if 'lat' not in df_meta.columns: df_meta['lat'] = df_meta['lat_calc']
    if 'lon' not in df_meta.columns: df_meta['lon'] = df_meta['lon_calc']
    
    # Crear Geometría
    gdf_puntos = gpd.GeoDataFrame(df_meta, geometry=gpd.points_from_xy(df_meta.lon, df_meta.lat), crs="EPSG:4326")
    minx, miny, maxx, maxy = gdf_puntos.total_bounds
    
    # PESTAÑAS
    tab_mapa, tab_datos = st.tabs(["🗺️ Visualización Espacial", "💾 Datos y Descargas"])
    
    with tab_mapa:
        try:
            if len(ids_filtrados) == 1: ids_sql = f"('{ids_filtrados[0]}')"
            else: ids_sql = str(ids_filtrados)
            
            # Consulta de LLUVIA
            q_data = text(f"""
                SELECT id_estacion_fk as {col_id}, SUM(precipitation) as valor
                FROM precipitacion_mensual
                WHERE extract(year from fecha_mes_año) = :anio
                AND id_estacion_fk IN {ids_sql}
                GROUP BY 1
            """)
            df_rain = pd.read_sql(q_data, engine, params={"anio": year_iso})
            
            # Merge
            df_final = pd.merge(df_rain, df_meta, on=col_id)
            
            # Limpieza
            if ignore_zeros: df_final = df_final[df_final['valor'] > 0]
            if ignore_nulls: df_final = df_final.dropna(subset=['valor'])
            
            if len(df_final) >= 3:
                with st.spinner("Interpolando superficie hidrológica..."):
                    grid_res = 200
                    gx, gy = np.mgrid[minx:maxx:complex(0, grid_res), miny:maxy:complex(0, grid_res)]
                    
                    rbf = Rbf(df_final['lon'], df_final['lat'], df_final['valor'], function='thin_plate', smooth=suavidad)
                    grid_z = rbf(gx, gy)
                    
                    fig = go.Figure()
                    
                    # Contornos
                    fig.add_trace(go.Contour(
                        z=grid_z.T, x=np.linspace(minx, maxx, grid_res), y=np.linspace(miny, maxy, grid_res),
                        colorscale="YlGnBu", colorbar=dict(title="Lluvia (mm)"),
                        hovertemplate="Precipitación: %{z:.0f} mm<extra></extra>",
                        contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                        opacity=0.8, connectgaps=True, line_smoothing=1.3
                    ))
                    
                    # Contexto
                    add_context_layers_ghost(fig, gdf_puntos)
                    
                    # Puntos
                    fig.add_trace(go.Scatter(
                        x=df_final['lon'], y=df_final['lat'], mode='markers',
                        marker=dict(size=6, color='black', line=dict(width=1, color='white')),
                        text=df_final[col_nom] + ': ' + df_final['valor'].round(0).astype(str) + ' mm',
                        hoverinfo='text', name="Estaciones"
                    ))
                    
                    fig.update_layout(
                        height=650, margin=dict(l=0,r=0,t=20,b=0),
                        xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False),
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"⚠️ Datos insuficientes en {year_iso} (Mínimo 3 estaciones con datos).")
                
        except Exception as e:
            st.error(f"Error procesando mapa: {e}")

    with tab_datos:
        if 'df_final' in locals() and not df_final.empty:
            st.subheader(f"📋 Tabla de Datos: Año {year_iso}")
            cols_show = [col_id, col_nom, 'valor']
            if col_muni: cols_show.append(col_muni)
            if col_cuenca: cols_show.append(col_cuenca)
            
            st.dataframe(df_final[cols_show].rename(columns={'valor': 'Lluvia (mm)'}), use_container_width=True, hide_index=True)
            
            c1, c2 = st.columns(2)
            csv = df_final.to_csv(index=False).encode('utf-8')
            c1.download_button("📥 Descargar CSV", csv, f"Isoyetas_{year_iso}.csv", "text/csv")
            
            try:
                import io
                buffer = io.StringIO()
                fig.write_html(buffer)
                c2.download_button("🗺️ Descargar Mapa HTML", buffer.getvalue().encode(), f"Mapa_{year_iso}.html", "text/html")
            except: pass
        else:
            st.info("Genere el mapa para ver los datos.")

else:
    st.info("👈 No se encontraron estaciones con los filtros actuales.")