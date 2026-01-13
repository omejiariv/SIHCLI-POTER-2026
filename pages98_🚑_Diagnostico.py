import streamlit as st
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
from modules import data_processor

st.set_page_config(page_title="Diagnóstico de Datos", page_icon="🚑", layout="wide")

st.title("🚑 Diagnóstico de Carga de Datos")

# 1. VERIFICAR BASE DE DATOS DIRECTA
st.subheader("1. Inspección Directa a la Base de Datos")
try:
    engine = create_engine(st.secrets["DATABASE_URL"])
    with engine.connect() as conn:
        # Contamos qué hay en la tabla geometrias por tipo
        query = "SELECT tipo_geometria, COUNT(*) as cantidad FROM geometrias GROUP BY tipo_geometria"
        df_conteo = pd.read_sql(text(query), conn)
        
    if not df_conteo.empty:
        st.success("✅ Conexión exitosa. Resumen de la tabla 'geometrias':")
        st.dataframe(df_conteo)
        
        if 'cuenca' not in df_conteo['tipo_geometria'].values and 'subcuenca' not in df_conteo['tipo_geometria'].values:
            st.error("❌ ALERTA: No existen filas con tipo 'cuenca' o 'subcuenca' en la base de datos.")
            st.info("💡 Solución: Debes subir tus Shapefiles a la tabla 'geometrias'.")
    else:
        st.warning("⚠️ La tabla 'geometrias' está vacía.")
        
except Exception as e:
    st.error(f"Error conectando a BD: {e}")

st.divider()

# 2. VERIFICAR EL PROCESADOR DE DATOS
st.subheader("2. Inspección de 'data_processor.py'")
try:
    with st.spinner("Ejecutando load_and_process_all_data()..."):
        # Forzamos recarga sin caché para probar
        data_processor.load_and_process_all_data.clear()
        all_data = data_processor.load_and_process_all_data()
        
    st.write(f"📦 La función devolvió **{len(all_data)} elementos**.")
    
    # Inspeccionamos el elemento [2] (Supuestamente Cuencas)
    obj_cuencas = all_data[2]
    st.write(f"Tipo de objeto en índice [2]: `{type(obj_cuencas)}`")
    
    if isinstance(obj_cuencas, (pd.DataFrame, gpd.GeoDataFrame)):
        st.write(f"Filas: {len(obj_cuencas)}")
        st.write("Columnas:", obj_cuencas.columns.tolist())
        st.dataframe(obj_cuencas.head())
        
        if len(obj_cuencas) == 0:
            st.error("❌ El DataFrame de cuencas está vacío.")
    else:
        st.error("❌ El objeto en el índice [2] NO es un DataFrame.")

except Exception as e:
    st.error(f"Error ejecutando el procesador: {e}")