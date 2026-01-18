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
    try:
        from modules.data_processor import complete_series
    except ImportError:
        complete_series = None
except:
    complete_series = None

st.title("🗺️ Generador Avanzado de Isoyetas (Escenarios & Pronósticos)")

# --- 2. FICHA TÉCNICA (TEXTO ACTUALIZADO) ---
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

    ### 5. Modos de Análisis
    * **📅 Año Específico:** Lluvia total acumulada en un año.
    * **📉 Mínimo/Máximo Histórico:** Extremos climáticos registrados.
    * **➗ Promedio Multianual:** Normal Climatológica (ej. 1981-2010).
    * **📊 Variabilidad Temporal (Desv. Estándar):** Muestra qué tan inestable es la lluvia en cada zona. Valores altos indican que la lluvia cambia drásticamente de un año a otro; valores bajos indican un clima constante.
    * **🔮 Pronóstico Futuro:** Proyección lineal a 2026-2040.
    ### 6. Metodología
    Interpolación **Thin-Plate Spline (RBF)** con corrección de escala dinámica y recorte de valores negativos.

    ### 7. Interpretación Visual
    * **Líneas Negras Punteadas:** Límites Municipales.
    * **Líneas Azules:** Límites de Cuencas Hidrográficas.
    * **Colores:** Representan la intensidad de la precipitación según la escala seleccionada.
    """)

# --- 3. FUNCIONES DE SOPORTE ---
@st.cache_data(ttl=3600)
def load_geojson_cached(filename):
    possible_paths = [
        os.path.join("data", filename),
        os.path.join("..", "data", filename),
        os.path.join(os.path.dirname(__file__), '..', 'data', filename)
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

def get_name_from_row_v2(row, type_layer):
    cols = row.index.str.lower()
    if type_layer == 'muni':
        for c in ['mpio_cnmbr', 'nombre', 'municipio', 'mpio_nomb']:
            if c in cols: return row[c]
    elif type_layer == 'cuenca':
        for c in ['n-nss3', 'subc_lbl', 'nom_cuenca', 'nombre', 'cuenca']:
            if c in cols: return row[c]
    return "Zona"

def add_context_layers_robust(fig, minx, miny, maxx, maxy):
    """
    Versión mejorada: Usa .cx para filtrar por caja delimitadora + iteración robusta de geometrías.
    """
    try:
        # Buffer visual para traer geometrías cercanas al borde
        pad = 0.05 
        b_minx, b_miny, b_maxx, b_maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
        
        gdf_m = load_geojson_cached("MunicipiosAntioquia.geojson")
        gdf_cu = load_geojson_cached("SubcuencasAinfluencia.geojson")
        
        # 1. Capa Municipios
        if gdf_m is not None:
            try:
                gdf_c = gdf_m.cx[b_minx:b_maxx, b_miny:b_maxy]
            except: gdf_c = gdf_m 

            for _, r in gdf_c.iterrows():
                name = get_name_from_row_v2(r, 'muni')
                geom = r.geometry
                # Manejo robusto de MultiPolygon
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', 
                        line=dict(width=1.5, color='rgba(50, 50, 50, 0.7)', dash='dot'), # Gris oscuro visible
                        text=f"🏙️ Mpio: {name}", hoverinfo='text',
                        showlegend=False
                    ))
        
        # 2. Capa Cuencas
        if gdf_cu is not None:
            try:
                gdf_c = gdf_cu.cx[b_minx:b_maxx, b_miny:b_maxy]
            except: gdf_c = gdf_cu

            for _, r in gdf_c.iterrows():
                name = get_name_from_row_v2(r, 'cuenca')
                geom = r.geometry
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', 
                        line=dict(width=2.0, color='rgba(0, 100, 255, 0.8)'), # Azul brillante sólido
                        text=f"🌊 Cuenca: {name}", hoverinfo='text',
                        showlegend=False
                    ))
    except Exception as e:
        print(f"Error pintando capas: {e}")

def calcular_pronostico(df_anual, target_year):
    proyecciones = []
    for station in df_anual['station_id'].unique():
        datos_est = df_anual[df_anual['station_id'] == station].dropna()
        if len(datos_est) >= 5: 
            try:
                x = datos_est['year'].values
                y = datos_est['total_anual'].values
                slope, intercept = np.polyfit(x, y, 1)
                pred = (slope * target_year) + intercept
                proyecciones.append({'station_id': station, 'valor': max(0, pred)}) 
            except: pass
    return pd.DataFrame(proyecciones)

def generar_analisis_texto_corregido(df_stats, tipo_analisis):
    """Lógica calibrada para reportar la realidad del gradiente."""
    if df_stats.empty: return "No hay datos suficientes."
    
    avg_val = df_stats['valor'].mean()
    min_val = df_stats['valor'].min()
    max_val = df_stats['valor'].max()
    
    # Diferencia absoluta
    diff = max_val - min_val
    
    # Identificar estaciones extremas
    est_max = df_stats.loc[df_stats['valor'].idxmax()]['nom_est']
    est_min = df_stats.loc[df_stats['valor'].idxmin()]['nom_est']
    
    # Lógica de interpretación CORREGIDA (Basada en Rango Absoluto)
    if diff < 600:
        conclusion = "un comportamiento regional relativamente uniforme."
    elif diff < 1500:
        conclusion = "un gradiente de precipitación moderado."
    else:
        conclusion = "una **fuerte variabilidad orográfica** y contrastes climáticos marcados."
    
    unit = "mm"
    
    reporte = f"""
    ### 📝 Análisis Automático del Territorio
    
    **1. Resumen Ejecutivo:**
    * Promedio Regional: **{avg_val:,.0f} {unit}**.
    * Rango de Variación (Max - Min): **{diff:,.0f} {unit}**.
    
    **2. Diagnóstico de Variabilidad:**
    * Existe una diferencia de **{diff:,.0f} {unit}** entre los extremos, lo que sugiere {conclusion}
    
    **3. Puntos Críticos:**
    * 🌧️ **Zona más Húmeda:** Estación **{est_max}** ({max_val:,.0f} {unit}).
    * ☀️ **Zona más Seca:** Estación **{est_min}** ({min_val:,.0f} {unit}).
    """
    return reporte

# --- 4. SIDEBAR ---
st.sidebar.header("🔍 Configuración")

with st.spinner("Cargando datos espaciales..."):
    gdf_meta, exito_cruce = obtener_estaciones_enriquecidas()

if gdf_meta.empty:
    st.error("Error crítico: Base de datos no disponible.")
    st.stop()

# Depuración de Archivos GIS
with st.sidebar.expander("🛠️ Diagnóstico de Archivos GIS"):
    geo_muni = load_geojson_cached("MunicipiosAntioquia.geojson")
    geo_cuenca = load_geojson_cached("SubcuencasAinfluencia.geojson")
    st.write(f"Municipios cargado: {'✅' if geo_muni is not None else '❌'}")
    st.write(f"Cuencas cargado: {'✅' if geo_cuenca is not None else '❌'}")

col_id = detectar_columna(gdf_meta, ['id_estacion', 'codigo']) or 'id_estacion'
col_nom = detectar_columna(gdf_meta, ['nom_est', 'nombre']) or 'nom_est'
col_region = detectar_columna(gdf_meta, ['region', 'subregion', 'depto_region'])
col_muni = detectar_columna(gdf_meta, ['municipio', 'mpio'])
col_alt = detectar_columna(gdf_meta, ['alt_est', 'altitud'])
col_cuenca = 'CUENCA_GIS' if 'CUENCA_GIS' in gdf_meta.columns else None

df_filtered_meta = gdf_meta.copy()

if col_region:
    regs = sorted(df_filtered_meta[col_region].dropna().astype(str).unique())
    sel_reg = st.sidebar.multiselect("📍 Región:", regs, key='filter_reg')
    if sel_reg: df_filtered_meta = df_filtered_meta[df_filtered_meta[col_region].isin(sel_reg)]

if col_cuenca:
    cuencas = sorted(df_filtered_meta[col_cuenca].dropna().astype(str).unique())
    sel_cuenca = st.sidebar.multiselect("🌊 Cuenca:", cuencas, key='filter_cuenca')
    if sel_cuenca: df_filtered_meta = df_filtered_meta[df_filtered_meta[col_cuenca].isin(sel_cuenca)]

if col_muni:
    munis = sorted(df_filtered_meta[col_muni].dropna().astype(str).unique())
    sel_muni = st.sidebar.multiselect("🏙️ Municipio:", munis, key='filter_muni')
    if sel_muni: df_filtered_meta = df_filtered_meta[df_filtered_meta[col_muni].isin(sel_muni)]

st.sidebar.markdown(f"**Estaciones en zona:** {len(df_filtered_meta)}")
st.sidebar.divider()

# Escenarios
tipo_analisis = st.sidebar.selectbox("📊 Modo de Análisis:", ["Año Específico", "Promedio Multianual", "Variabilidad Temporal", "Mínimo Histórico", "Máximo Histórico", "Pronóstico Futuro"], key='analisis_mode')

params_analisis = {}
if tipo_analisis == "Año Específico":
    params_analisis['year'] = st.sidebar.selectbox("📅 Año:", range(2025, 1980, -1), key='sel_year')
elif tipo_analisis in ["Promedio Multianual", "Variabilidad Temporal"]:
    rango = st.sidebar.slider("📅 Periodo:", 1980, 2025, (1990, 2020), key='sel_period')
    params_analisis['start'], params_analisis['end'] = rango
elif tipo_analisis == "Pronóstico Futuro":
    params_analisis['target'] = st.sidebar.slider("🔮 Proyección:", 2026, 2040, 2026, key='sel_proj')

paleta_colores = st.sidebar.selectbox("🎨 Escala de Color:", options=["YlGnBu", "Jet", "Portland", "Viridis", "RdBu"], index=0)
buffer_km = st.sidebar.slider("📡 Buffer Búsqueda (km):", 0, 100, 20, key='buff_km')
buffer_deg = buffer_km / 111.0

c1, c2 = st.sidebar.columns(2)
ignore_zeros = c1.checkbox("🚫 No Ceros", value=True, key='chk_zeros')
ignore_nulls = c2.checkbox("🚫 No Nulos", value=True, key='chk_nulls')

do_interp_temp = False
if complete_series: do_interp_temp = st.sidebar.checkbox("🔄 Interpolación Temporal", value=False, key='chk_interp')

suavidad = st.sidebar.slider("🖌️ Suavizado (RBF):", 0.0, 2.0, 0.0, key='slider_smooth')

# --- 5. LÓGICA ESPACIAL ---
if len(df_filtered_meta) > 0:
    gdf_target = df_filtered_meta
    minx, miny, maxx, maxy = gdf_target.total_bounds
    q_minx, q_miny = minx - buffer_deg, miny - buffer_deg
    q_maxx, q_maxy = maxx + buffer_deg, maxy + buffer_deg
    
    tab_mapa, tab_datos = st.tabs(["🗺️ Visualización Espacial", "💾 Descargas GIS"])
    
    with tab_mapa:
        try:
            engine = create_engine(st.secrets["DATABASE_URL"])
            df_agg = pd.DataFrame()
            
            q_raw = text(f"""
                SELECT p.id_estacion_fk, p.fecha_mes_año, p.precipitation
                FROM precipitacion_mensual p JOIN estaciones e ON p.id_estacion_fk = e.id_estacion
                WHERE ST_X(e.geom::geometry) BETWEEN :mx AND :Mx AND ST_Y(e.geom::geometry) BETWEEN :my AND :My
            """)
            df_raw = pd.read_sql(q_raw, engine, params={"mx":q_minx, "my":q_miny, "Mx":q_maxx, "My":q_maxy})
            
            if not df_raw.empty:
                df_proc = df_raw.rename(columns={'id_estacion_fk': 'id_estacion', 'precipitation': 'precipitation'})
                df_proc['fecha_mes_año'] = pd.to_datetime(df_proc['fecha_mes_año'])
                df_proc = df_proc.groupby(['id_estacion', 'fecha_mes_año'])['precipitation'].mean().reset_index()
                
                if do_interp_temp and complete_series:
                    with st.spinner("Interpolando series..."):
                        df_processed = complete_series(df_proc) 
                else:
                    df_processed = df_proc.copy()
                
                df_processed['year'] = df_processed['fecha_mes_año'].dt.year
                year_counts = df_processed.groupby(['id_estacion', 'year'])['precipitation'].count().reset_index(name='count')
                
                if not do_interp_temp:
                    valid_years = year_counts[year_counts['count'] >= 10]
                    df_processed = pd.merge(df_processed, valid_years[['id_estacion', 'year']], on=['id_estacion', 'year'])
                
                df_annual_sums = df_processed.groupby(['id_estacion', 'year'])['precipitation'].sum().reset_index(name='total_anual')
                df_annual_sums = df_annual_sums.rename(columns={'id_estacion': 'station_id'})

                if tipo_analisis == "Año Específico":
                    df_agg = df_annual_sums[df_annual_sums['year'] == params_analisis['year']].copy()
                    df_agg = df_agg.rename(columns={'total_anual': 'valor'})
                elif tipo_analisis == "Mínimo Histórico":
                    df_agg = df_annual_sums.groupby('station_id')['total_anual'].min().reset_index(name='valor')
                elif tipo_analisis == "Máximo Histórico":
                    df_agg = df_annual_sums.groupby('station_id')['total_anual'].max().reset_index(name='valor')
                elif tipo_analisis == "Promedio Multianual":
                    mask = (df_annual_sums['year'] >= params_analisis['start']) & (df_annual_sums['year'] <= params_analisis['end'])
                    df_agg = df_annual_sums[mask].groupby('station_id')['total_anual'].mean().reset_index(name='valor')
                elif tipo_analisis == "Variabilidad Temporal":
                    mask = (df_annual_sums['year'] >= params_analisis['start']) & (df_annual_sums['year'] <= params_analisis['end'])
                    df_agg = df_annual_sums[mask].groupby('station_id')['total_anual'].std().reset_index(name='valor')
                elif tipo_analisis == "Pronóstico Futuro":
                    with st.spinner("Proyectando..."):
                        df_agg = calcular_pronostico(df_annual_sums, params_analisis['target'])

            if not df_agg.empty:
                df_agg = df_agg.rename(columns={'station_id': col_id})
                cols_merge = [col_id, col_nom, 'lat_calc', 'lon_calc']
                if col_muni: cols_merge.append(col_muni)
                if col_alt: cols_merge.append(col_alt)
                if col_cuenca: cols_merge.append(col_cuenca)
                
                df_final = pd.merge(df_agg, gdf_meta[list(set(cols_merge))], on=col_id)
                
                df_final = df_final.groupby(['lat_calc', 'lon_calc']).agg({
                    col_id: 'first', col_nom: 'first', 'valor': 'mean', 
                    col_muni: 'first', col_alt: 'first', col_cuenca: 'first'
                }).reset_index()

                if ignore_zeros: df_final = df_final[df_final['valor'] > 5]
                if ignore_nulls: df_final = df_final.dropna(subset=['valor'])
                
                if len(df_final) >= 3:
                    with st.spinner(f"Generando isoyetas..."):
                        grid_res = 200
                        
                        x_raw, y_raw, z_raw = df_final['lon_calc'].values, df_final['lat_calc'].values, df_final['valor'].values
                        x_mean, x_std = x_raw.mean(), x_raw.std()
                        y_mean, y_std = y_raw.mean(), y_raw.std()
                        x_norm = (x_raw - x_mean) / x_std
                        y_norm = (y_raw - y_mean) / y_std
                        
                        gx_raw, gy_raw = np.mgrid[q_minx:q_maxx:complex(0, grid_res), q_miny:q_maxy:complex(0, grid_res)]
                        gx_norm = (gx_raw - x_mean) / x_std
                        gy_norm = (gy_raw - y_mean) / y_std
                        
                        rbf = Rbf(x_norm, y_norm, z_raw, function='thin_plate', smooth=suavidad)
                        grid_z = rbf(gx_norm, gy_norm)
                        grid_z = np.maximum(grid_z, 0)
                        
                        z_min = df_final['valor'].min()
                        z_max = df_final['valor'].max()
                        if z_max == z_min: z_max += 0.1
                        
                        fig = go.Figure()
                        tit = f"Isoyetas: {tipo_analisis}"
                        if tipo_analisis == "Año Específico": tit += f" ({params_analisis['year']})"
                        
                        df_final['hover_val'] = df_final['valor'].apply(lambda x: f"{x:,.0f}")
                        c_muni = df_final[col_muni].fillna('-') if col_muni else ["-"]*len(df_final)
                        c_alt = df_final[col_alt].fillna(0) if col_alt else [0]*len(df_final)
                        c_cuenca = df_final[col_cuenca].fillna('-') if col_cuenca else ["-"]*len(df_final)
                        custom_data = np.stack((c_muni, c_alt, c_cuenca, df_final['hover_val']), axis=-1)
                        
                        fig.add_trace(go.Contour(
                            z=grid_z.T, x=np.linspace(q_minx, q_maxx, grid_res), y=np.linspace(q_miny, q_maxy, grid_res),
                            colorscale=paleta_colores, zmin=z_min, zmax=z_max,
                            colorbar=dict(title="mm"),
                            contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                            opacity=0.8, connectgaps=True, line_smoothing=1.3
                        ))
                        
                        # --- CAPAS DE CONTEXTO VISIBLES ---
                        add_context_layers_robust(fig, q_minx, q_miny, q_maxx, q_maxy)
                        
                        fig.add_trace(go.Scatter(
                            x=df_final['lon_calc'], y=df_final['lat_calc'], mode='markers',
                            marker=dict(size=6, color='black', line=dict(width=1, color='white')),
                            text=df_final[col_nom], customdata=custom_data,
                            hovertemplate="<b>%{text}</b><br>Valor: %{customdata[3]}<br>🏙️: %{customdata[0]}<br>⛰️: %{customdata[1]} m<br>🌊: %{customdata[2]}<extra></extra>",
                            name="Estaciones"
                        ))
                        fig.add_shape(type="rect", x0=minx, y0=miny, x1=maxx, y1=maxy, line=dict(color="Red", width=2, dash="dot"))
                        fig.update_layout(title=tit, height=650, margin=dict(l=0,r=0,t=30,b=0), xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False), plot_bgcolor='white', hovermode='closest')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # --- ANÁLISIS AUTOMÁTICO CORREGIDO ---
                        st.info(generar_analisis_texto_corregido(df_final, tipo_analisis))
                        
                else:
                    st.warning("⚠️ Datos insuficientes. Aumente el Buffer (km).")
            else:
                st.warning("No hay datos para esta selección.")
            
            with st.expander("🔍 Ver Datos Crudos", expanded=False):
                if not df_agg.empty: st.dataframe(df_final)

        except Exception as e:
            st.error(f"Error técnico: {e}")

    with tab_datos:
        if 'df_final' in locals() and not df_final.empty:
            st.subheader("💾 Descargas GIS")
            cols_show = [col_id, col_nom, 'valor']
            if col_cuenca in df_final.columns: cols_show.append(col_cuenca)
            st.dataframe(df_final[cols_show].head(100), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            gdf_out = gpd.GeoDataFrame(df_final, geometry=gpd.points_from_xy(df_final.lon_calc, df_final.lat_calc), crs="EPSG:4326")
            c1.download_button("🌍 GeoJSON", gdf_out.to_json(), f"isoyetas_{tipo_analisis}.geojson", "application/json")
            if 'grid_z' in locals():
                asc = generar_raster_ascii(grid_z, q_minx, q_miny, (q_maxx-q_minx)/grid_res, grid_res, grid_res)
                c2.download_button("⬛ Raster (.asc)", asc, f"raster_{tipo_analisis}.asc", "text/plain")
            csv = df_final.to_csv(index=False).encode('utf-8')
            c3.download_button("📊 CSV", csv, f"datos_{tipo_analisis}.csv", "text/csv")

else:
    st.info("👈 Seleccione una Región o Cuenca en el menú lateral.")