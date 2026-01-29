import streamlit as st
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
import json
import rasterio
from shapely.geometry import box

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Centro de Diagnóstico", page_icon="🕵️", layout="wide")

st.title("🕵️ Centro de Diagnóstico y Detective")
st.markdown("---")

# --- GESTIÓN DE CONEXIÓN ---
try:
    from modules.db_manager import get_engine
except ImportError:
    def get_engine(): return create_engine(st.secrets["DATABASE_URL"])

engine = get_engine()

# --- PESTAÑAS PARA ORGANIZAR (LO NUEVO + LO VIEJO) ---
tab_dem, tab_bd = st.tabs(["⛰️ Diagnóstico DEM vs Cuencas", "🔍 Explorador de Tablas (Original)"])

# ==============================================================================
# TAB 1: DIAGNÓSTICO DEM (NUEVA FUNCIONALIDAD)
# ==============================================================================
with tab_dem:
    st.header("🕵️ Detective de Coordenadas: ¿Por qué mi reporte da Ceros?")
    st.info("Esta herramienta verifica si el archivo DEM y las Cuencas en BD están 'viviendo' en el mismo sistema de coordenadas.")
    
    PATH_DEM = "data/DemAntioquia_EPSG3116.tif"

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("1. Análisis del DEM (Archivo)")
        try:
            if not rasterio:
                st.error("Librería rasterio no instalada.")
            else:
                try:
                    with rasterio.open(PATH_DEM) as src:
                        st.success(f"✅ DEM Cargado: {PATH_DEM}")
                        dem_crs = src.crs
                        dem_bounds = src.bounds
                        st.code(f"CRS DEM:\n{dem_crs}\n\nLímites (Metros):\nIzquierda: {dem_bounds.left:,.0f}\nAbajo:     {dem_bounds.bottom:,.0f}\nDerecha:   {dem_bounds.right:,.0f}\nArriba:    {dem_bounds.top:,.0f}")
                        
                        # Diagnóstico de Origen
                        if dem_bounds.left > 4000000:
                            st.info("ℹ️ TIPO: Parece ser MAGNA ORIGEN NACIONAL (CTM12)")
                        elif dem_bounds.left > 800000:
                            st.info("ℹ️ TIPO: Parece ser MAGNA BOGOTÁ (EPSG:3116)")
                        else:
                            st.info("ℹ️ TIPO: Probablemente Grados (WGS84)")
                except FileNotFoundError:
                    st.error(f"❌ No se encontró el archivo: {PATH_DEM}")
        except Exception as e:
            st.error(f"Error analizando DEM: {e}")

    with c2:
        st.subheader("2. Análisis de una Cuenca (Base de Datos)")
        try:
            # Cargar una cuenca cualquiera para probar
            gdf_test = gpd.read_postgis("SELECT * FROM cuencas LIMIT 1", engine, geom_col="geometry")
            
            if not gdf_test.empty:
                st.success(f"✅ Cuenca cargada: {gdf_test.iloc[0].get('nombre_cuenca', 'Sin Nombre')}")
                
                # Proyección original
                st.write(f"**CRS Original en BD:** {gdf_test.crs}")
                
                # Intentar reproyectar al CRS del DEM (si el DEM cargó bien)
                if 'dem_crs' in locals():
                    try:
                        gdf_reproj = gdf_test.to_crs(dem_crs)
                        poly_bounds = gdf_reproj.total_bounds
                        st.code(f"Límites Cuenca (Reproyectada al CRS del DEM):\nIzquierda: {poly_bounds[0]:,.0f}\nAbajo:     {poly_bounds[1]:,.0f}\nDerecha:   {poly_bounds[2]:,.0f}\nArriba:    {poly_bounds[3]:,.0f}")
                        
                        # PRUEBA DE FUEGO: ¿Se tocan?
                        dem_box = box(*dem_bounds)
                        cuenca_box = box(*poly_bounds)
                        
                        if dem_box.intersects(cuenca_box):
                            st.success("🎉 ¡HAY INTERSECCIÓN! Los datos se tocan físicamente.")
                        else:
                            st.error("❌ NO SE TOCAN. Están en lugares diferentes.")
                            st.warning("Esto explica por qué el reporte da valores en CERO.")
                            
                            # Calcular distancia del error
                            dist_x = abs(dem_bounds.left - poly_bounds[0])
                            st.write(f"Distancia X entre ellos: {dist_x:,.0f} metros")
                            if 3500000 < dist_x < 4500000:
                                st.error("⚠️ La diferencia es ~4,000,000m. Conflicto Origen Nacional vs Bogotá confirmado.")
                    except Exception as e:
                        st.error(f"Error reproyectando: {e}")
            else:
                st.warning("La tabla 'cuencas' está vacía.")

        except Exception as e:
            st.error(f"Error consultando Cuenca: {e}")

# ==============================================================================
# TAB 2: EXPLORADOR BD (TU CÓDIGO ORIGINAL)
# ==============================================================================
with tab_bd:
    st.header("🔍 Explorador de Tablas (Modo Forense)")
    
    # 1. LISTAR TABLAS
    with st.container():
        try:
            with engine.connect() as conn:
                # Consulta para ver todas las tablas públicas
                tables = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'", conn)
                
                if not tables.empty:
                    table_list = tables['table_name'].tolist()
                    selected_table = st.selectbox("Selecciona la tabla a investigar:", table_list)
                else:
                    st.error("No se encontraron tablas en la base de datos.")
                    selected_table = None
        except Exception as e:
            st.error(f"Error conectando a BD: {e}")
            selected_table = None

    # 2. RADIOGRAFÍA DE LA TABLA
    if selected_table:
        st.markdown(f"### 🔬 Analizando: `{selected_table}`")
        
        try:
            with engine.connect() as conn:
                # A. Conteo Total
                count = pd.read_sql(text(f"SELECT count(*) as total FROM {selected_table}"), conn).iloc[0]['total']
                
                # B. Estructura de Columnas
                cols_df = pd.read_sql(text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{selected_table}'"), conn)
                
                c1, c2 = st.columns(2)
                c1.metric("Filas Totales", count)
                with c2:
                    with st.expander("Ver Columnas y Tipos de Dato"):
                        st.dataframe(cols_df, hide_index=True)

                # C. DETECTOR DE COORDENADAS (La parte vital)
                st.markdown("#### 🌍 Auditoría de Geometría")
                
                # Verificar si tiene columna geométrica
                geom_col = "geom" # Asumimos standard
                if "geometry" in cols_df['column_name'].values: geom_col = "geometry"
                
                if geom_col in cols_df['column_name'].values:
                    try:
                        # Consulta forense
                        q_geo = text(f"""
                            SELECT 
                                ST_SRID({geom_col}) as srid_detectado, 
                                ST_AsText({geom_col}) as ejemplo_coordenada,
                                ST_IsValid({geom_col}) as es_valido
                            FROM {selected_table} 
                            LIMIT 1
                        """)
                        geo_sample = pd.read_sql(q_geo, conn).iloc[0]
                        
                        st.write("**Sistema de Referencia (SRID) en BD:**", f"`{geo_sample['srid_detectado']}`")
                        st.write("**Ejemplo de Coordenada Real:**", f"`{geo_sample['ejemplo_coordenada']}`")
                        
                        # ANÁLISIS AUTOMÁTICO
                        coord_text = geo_sample['ejemplo_coordenada']
                        if "POINT" in coord_text or "POLYGON" in coord_text:
                            import re
                            nums = re.findall(r"[-+]?\d*\.\d+|\d+", coord_text)
                            if nums:
                                first_num = float(nums[0])
                                if abs(first_num) <= 180:
                                    st.success("✅ **DIAGNÓSTICO:** Las coordenadas parecen ser **GRADOS (Lat/Lon)**.")
                                else:
                                    st.error(f"🚨 **DIAGNÓSTICO:** Las coordenadas parecen ser **METROS** (Valor: {first_num:.0f}).")
                    except Exception as e:
                        st.warning(f"No se pudo analizar la geometría: {e}")
                else:
                    st.info("Esta tabla no parece tener columna espacial (geom).")

                # D. MUESTRA DE DATOS
                st.markdown("#### 📄 Primeras 7 Filas (Datos Crudos)")
                cols_safe = [c for c in cols_df['column_name'] if c != geom_col]
                cols_query = ", ".join([f'"{c}"' for c in cols_safe])
                
                try:
                    df_preview = pd.read_sql(text(f"SELECT {cols_query} FROM {selected_table} LIMIT 7"), conn)
                    st.dataframe(df_preview)
                except Exception as e:
                    st.error(f"Error cargando vista previa: {e}")
        except Exception as e:
             st.error(f"Error en análisis de tabla: {e}")

    # 3. CONSOLA SQL LIBRE
    st.markdown("---")
    st.subheader("🛠️ Consola SQL Manual")
    query = st.text_area("Ejecutar SQL personalizado:", f"SELECT * FROM {selected_table if selected_table else 'cuencas'} LIMIT 10")
    if st.button("Ejecutar Query"):
        with engine.connect() as conn:
            try:
                res = pd.read_sql(text(query), conn)
                st.dataframe(res)
            except Exception as e:
                st.error(f"Error SQL: {e}")