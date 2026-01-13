import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Detective de Datos", page_icon="🕵️")
st.title("🕵️ Auditoría de Estructura de Tablas")

try:
    engine = create_engine(st.secrets["DATABASE_URL"])
    with engine.connect() as conn:
        
        # 1. Investigar Tabla ESTACIONES
        st.subheader("1. Columnas de la tabla: 'estaciones'")
        query_est = "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'estaciones';"
        df_est = pd.read_sql(text(query_est), conn)
        st.dataframe(df_est)

        st.divider()

        # 2. Investigar Tabla PRECIPITACION_MENSUAL
        st.subheader("2. Columnas de la tabla: 'precipitacion_mensual'")
        query_precip = "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'precipitacion_mensual';"
        df_precip = pd.read_sql(text(query_precip), conn)
        st.dataframe(df_precip)

except Exception as e:
    st.error(f"Error conectando: {e}")