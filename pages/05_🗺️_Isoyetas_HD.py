# pages/05_🗺️_Isoyetas_HD.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import geopandas as gpd
from scipy.interpolate import griddata, Rbf
import os
import sys

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Isoyetas HD", page_icon="🗺️", layout="wide")

# Intentar importar módulos compartidos si existen, sino usar locales
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules.config import Config
except:
    pass

st.title("🗺️ Mapas de Isoyetas de Alta Definición (RBF)")

# --- 2. FUNCIONES DE SOPORTE (Estilo Fantasma y Carga) ---
@st.cache_data(ttl=3600)
def load_geojson_cached(filename):
    # Ajusta esta ruta si tu carpeta data está en otro nivel
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
        for c in ['n-nss3', 'subc_lbl', 'nom_cuenca', 'nombre']:
            if c in cols: return row[c]
    return ""

def add_context_layers_ghost(fig, gdf_zona):
    """Añade capas de contexto (Municipios/Cuencas) con estilo fantasma."""
    try:
        if gdf_zona is None or gdf_zona.empty: return
        roi = gdf_zona.buffer(0.05)
        
        # Cargar GeoJSONs
        gdf_m = load_geojson_cached("MunicipiosAntioquia.geojson")
        gdf_cu = load_geojson_cached("SubcuencasAinfluencia.geojson")
        
        # Capa Municipios
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
        
        # Capa Cuencas
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

# --- 3. FICHA TÉCNICA (DOCUMENTACIÓN) ---
with st.expander("📘 Ficha Técnica: Metodología e Interpretación", expanded=False):
    st.markdown("""
    ### 1. Concepto
    Las **isoyetas** son isolíneas que conectan puntos de igual precipitación. Este módulo utiliza técnicas avanzadas de interpolación para estimar la lluvia en lugares donde no existen estaciones, generando una superficie continua.

    ### 2. Metodología RBF (Radial Basis Function)
    Se implementa el algoritmo **Thin-Plate Spline** de la librería `SciPy`. 
    * A diferencia de la interpolación lineal (que crea triángulos y picos), el RBF simula el comportamiento de una hoja de metal delgada que se dobla suavemente para pasar por los puntos de medición.
    * Esto elimina el efecto artificial de "conos" o "ojos de buey" alrededor de las estaciones.

    ### 3. Interpretación del Mapa
    * **🟦 Azul Oscuro:** Alta precipitación (> 3000 mm). Zonas de oferta hídrica o riesgo de saturación.
    * **🟨 Amarillo/Claro:** Baja precipitación (< 1500 mm). Posible déficit hídrico o sombra de lluvia.
    
    ### 4. Fuentes de Datos
    * **Precipitación:** Base de datos SIHCLI (Consolidado IDEAM, EPM, Cenicafé).
    * **Cartografía:** IGAC (Municipios) y Corporación CuencaVerde (Delimitación hidrográfica).
    """)

# --- 4. SIDEBAR DE HIDROLOGÍA (FILTROS) ---
st.sidebar.header("🔍 Filtros Hidrológicos")

# Conexión a BD
try:
    engine = create_engine(st.secrets["DATABASE_URL"])
    
    # Cargar metadatos de estaciones para los filtros
    q_meta = "SELECT id_estacion, nom_est, lat, lon, alt_est, municipio, cuenca FROM estaciones"
    df_meta = pd.read_sql(q_meta, engine)
    
    # Filtro 1: CUENCA (Lo que pediste)
    cuencas_disp = sorted(df_meta['cuenca'].dropna().unique())
    sel_cuenca = st.sidebar.multiselect("🌊 Seleccionar Cuenca:", cuencas_disp)
    
    # Filtro 2: MUNICIPIO (Reactivo a la cuenca)
    if sel_cuenca:
        df_meta = df_meta[df_meta['cuenca'].isin(sel_cuenca)]
        
    munis_disp = sorted(df_meta['municipio'].dropna().unique())
    sel_muni = st.sidebar.multiselect("🏙️ Seleccionar Municipio:", munis_disp)
    
    # Aplicar Filtros al DF Metadata
    if sel_muni:
        df_meta = df_meta[df_meta['municipio'].isin(sel_muni)]
        
    ids_filtrados = tuple(df_meta['id_estacion'].unique())
    st.sidebar.markdown(f"**Estaciones encontradas:** {len(df_meta)}")
    
    # Filtro 3: TIEMPO Y PARÁMETROS
    st.sidebar.divider()
    year_iso = st.sidebar.selectbox("📅 Año de Análisis:", range(2025, 1980, -1))
    
    col_opt1, col_opt2 = st.sidebar.columns(2)
    ignore_zeros = col_opt1.checkbox("🚫 No Ceros", value=True)
    ignore_nulls = col_opt2.checkbox("🚫 No Nulos", value=True)
    
    suavidad = st.sidebar.slider("🎨 Suavizado (RBF):", 0.0, 2.0, 0.5, help="Mayor valor = curvas más relajadas")

except Exception as e:
    st.error(f"Error de conexión: {e}")
    st.stop()

# --- 5. LÓGICA PRINCIPAL ---
if len(df_meta) > 0:
    # Definir zona geográfica para el mapa
    gdf_puntos = gpd.GeoDataFrame(df_meta, geometry=gpd.points_from_xy(df_meta.lon, df_meta.lat), crs="EPSG:4326")
    minx, miny, maxx, maxy = gdf_puntos.total_bounds
    
    # PESTAÑAS PARA ORGANIZAR
    tab_mapa, tab_datos = st.tabs(["🗺️ Visualización Espacial", "💾 Datos y Descargas"])
    
    with tab_mapa:
        # Consulta de DATOS DE LLUVIA
        try:
            if len(ids_filtrados) == 1: ids_sql = f"('{ids_filtrados[0]}')"
            else: ids_sql = str(ids_filtrados)
            
            q_data = text(f"""
                SELECT id_estacion_fk as id_estacion, SUM(precipitation) as valor
                FROM precipitacion_mensual
                WHERE extract(year from fecha_mes_año) = :anio
                AND id_estacion_fk IN {ids_sql}
                GROUP BY 1
            """)
            df_rain = pd.read_sql(q_data, engine, params={"anio": year_iso})
            
            # Unir con metadatos (Lat/Lon)
            df_final = pd.merge(df_rain, df_meta, on='id_estacion')
            
            # Limpieza
            if ignore_zeros: df_final = df_final[df_final['valor'] > 0]
            if ignore_nulls: df_final = df_final.dropna(subset=['valor'])
            
            if len(df_final) >= 3:
                with st.spinner("Interpolando superficie hidrológica..."):
                    # Grid 200x200
                    grid_res = 200
                    gx, gy = np.mgrid[minx:maxx:complex(0, grid_res), miny:maxy:complex(0, grid_res)]
                    
                    # Interpolación RBF
                    rbf = Rbf(df_final['lon'], df_final['lat'], df_final['valor'], function='thin_plate', smooth=suavidad)
                    grid_z = rbf(gx, gy)
                    
                    # Graficar
                    fig = go.Figure()
                    
                    # 1. Contornos
                    fig.add_trace(go.Contour(
                        z=grid_z.T, x=np.linspace(minx, maxx, grid_res), y=np.linspace(miny, maxy, grid_res),
                        colorscale="YlGnBu", colorbar=dict(title="Lluvia (mm)"),
                        hovertemplate="Precipitación: %{z:.0f} mm<extra></extra>",
                        contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                        opacity=0.8, connectgaps=True, line_smoothing=1.3
                    ))
                    
                    # 2. Contexto Fantasma
                    add_context_layers_ghost(fig, gdf_puntos)
                    
                    # 3. Puntos Estaciones
                    fig.add_trace(go.Scatter(
                        x=df_final['lon'], y=df_final['lat'], mode='markers',
                        marker=dict(size=6, color='black', line=dict(width=1, color='white')),
                        text=df_final['nom_est'] + ': ' + df_final['valor'].round(0).astype(str) + ' mm',
                        hoverinfo='text', name="Estaciones"
                    ))
                    
                    fig.update_layout(
                        height=650, margin=dict(l=0,r=0,t=20,b=0),
                        xaxis=dict(visible=False, scaleanchor="y"), yaxis=dict(visible=False),
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            else:
                st.warning(f"⚠️ Datos insuficientes en {year_iso} para las estaciones seleccionadas (Mínimo 3).")
                
        except Exception as e:
            st.error(f"Error procesando mapa: {e}")

    with tab_datos:
        if 'df_final' in locals() and not df_final.empty:
            st.subheader(f"📋 Tabla de Datos: Año {year_iso}")
            st.markdown("Datos consolidados utilizados para la generación de las isoyetas.")
            
            # Tabla interactiva
            st.dataframe(
                df_final[['id_estacion', 'nom_est', 'municipio', 'cuenca', 'valor']].rename(columns={'valor': 'Lluvia (mm)'}),
                use_container_width=True, hide_index=True
            )
            
            # Botones de descarga
            c_d1, c_d2 = st.columns(2)
            csv = df_final.to_csv(index=False).encode('utf-8')
            c_d1.download_button("📥 Descargar Excel/CSV", csv, f"Isoyetas_{year_iso}.csv", "text/csv")
            
            # Descargar Mapa HTML
            try:
                import io
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                c_d2.download_button("🗺️ Descargar Mapa Interactivo (HTML)", html_bytes, f"Mapa_Isoyetas_{year_iso}.html", "text/html")
            except: pass
            
        else:
            st.info("Genere el mapa primero para ver los datos.")

else:
    st.info("👈 Utilice el sidebar para seleccionar una Cuenca o Municipio.")