import streamlit as st
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
import json

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Detective de Datos", page_icon="🕵️", layout="wide")

try:
    from modules.db_manager import get_engine
except ImportError:
    def get_engine(): return create_engine(st.secrets["DATABASE_URL"])

st.title("🕵️ Detective de Base de Datos Espacial")
st.markdown("---")

engine = get_engine()

# 1. LISTAR TABLAS
with st.container():
    st.subheader("1. Explorador de Tablas")
    try:
        with engine.connect() as conn:
            # Consulta para ver todas las tablas públicas
            tables = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'", conn)
            
            if not tables.empty:
                table_list = tables['table_name'].tolist()
                selected_table = st.selectbox("Selecciona la tabla a investigar:", table_list)
            else:
                st.error("No se encontraron tablas en la base de datos.")
                st.stop()
    except Exception as e:
        st.error(f"Error conectando a BD: {e}")
        st.stop()

# 2. RADIOGRAFÍA DE LA TABLA
if selected_table:
    st.markdown(f"### 🔬 Analizando: `{selected_table}`")
    
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
                # Consulta forense: Trae el SRID y una muestra de texto de la coordenada
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
                    # Extraer un número para ver si es Lat/Lon o Metros
                    import re
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", coord_text)
                    if nums:
                        first_num = float(nums[0])
                        if abs(first_num) <= 180:
                            st.success("✅ **DIAGNÓSTICO:** Las coordenadas parecen ser **GRADOS (Lat/Lon)**. Esto es correcto para mapas web.")
                        else:
                            st.error(f"🚨 **DIAGNÓSTICO:** Las coordenadas parecen ser **METROS** (Valor: {first_num:.0f}).")
                            st.warning("""
                            **SOLUCIÓN:** El mapa web espera Lat/Lon (-75.5). Si tus datos son Metros (1170000), 
                            debes usar el 'Corrector de Coordenadas' en la página de Aguas y seleccionar 'Magna Sirgas' o 'Origen Nacional'.
                            """)
            except Exception as e:
                st.warning(f"No se pudo analizar la geometría: {e}")
        else:
            st.info("Esta tabla no parece tener columna espacial (geom).")

        # D. MUESTRA DE DATOS
        st.markdown("#### 📄 Primeras 5 Filas (Datos Crudos)")
        # Evitamos traer la columna geom pesada para la vista previa
        cols_safe = [c for c in cols_df['column_name'] if c != geom_col]
        cols_query = ", ".join([f'"{c}"' for c in cols_safe]) # Comillas para manejar mayúsculas
        
        try:
            df_preview = pd.read_sql(text(f"SELECT {cols_query} FROM {selected_table} LIMIT 5"), conn)
            st.dataframe(df_preview)
        except Exception as e:
            st.error(f"Error cargando vista previa: {e}")

# 3. CONSOLA SQL LIBRE
st.markdown("---")
st.subheader("🛠️ Consola SQL Manual")
query = st.text_area("Ejecutar SQL personalizado:", f"SELECT * FROM {selected_table if selected_table else 'suelos'} LIMIT 10")
if st.button("Ejecutar Query"):
    with engine.connect() as conn:
        try:
            res = pd.read_sql(text(query), conn)
            st.dataframe(res)
        except Exception as e:
            st.error(f"Error SQL: {e}")