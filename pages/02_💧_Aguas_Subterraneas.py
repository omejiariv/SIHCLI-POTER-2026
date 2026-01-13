import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os

# Agregar la carpeta raíz al path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import hydrogeo_utils

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

st.title("💧 Estimación de Recarga de Acuíferos")
st.markdown("Este módulo utiliza datos de **precipitación mensual** para estimar la recarga potencial.")

# --- 1. CONEXIÓN ---
try:
    db_url = st.secrets["DATABASE_URL"]
    engine = create_engine(db_url)
    
    # CORRECCIÓN 1: Usamos los nombres reales de la tabla 'estaciones'
    # Renombramos 'id_estacion' -> 'codigo' y 'nom_est' -> 'nombre' al vuelo
    query_estaciones = """
        SELECT id_estacion AS codigo, nom_est AS nombre 
        FROM estaciones 
        ORDER BY nom_est
    """
    
    with engine.connect() as conn:
        df_estaciones = pd.read_sql(text(query_estaciones), conn)
        
except Exception as e:
    st.error(f"Error conectando a la base de datos: {e}")
    st.stop()

# --- 2. CONFIGURACIÓN ---
with st.sidebar:
    st.header("Parámetros")
    
    # Selector de Estación
    estacion_selec = st.selectbox(
        "Seleccione Estación:", 
        options=df_estaciones['codigo'],
        format_func=lambda x: df_estaciones[df_estaciones['codigo'] == x]['nombre'].values[0]
    )
    
    st.divider()
    
    st.subheader("Suelo y Cobertura")
    tipo_suelo = st.selectbox(
        "Tipo de Suelo:",
        ["Arenoso (Alta Infiltración)", "Franco (Media Infiltración)", "Arcilloso (Baja Infiltración)", "Urbano/Impermeable"]
    )
    
    coef_sugerido = hydrogeo_utils.obtener_clasificacion_suelo(tipo_suelo)
    coef_final = st.slider("Coeficiente de Infiltración (%)", 0.0, 1.0, coef_sugerido)
    st.info(f"Recarga estimada: **{coef_final*100:.0f}%** de la lluvia mensual.")

# --- 3. ANÁLISIS ---
if estacion_selec:
    # CORRECCIÓN 2: Usamos nombres reales de 'precipitacion_mensual'
    # 'id_estacion_fk' es el enlace. 'precipitation' es el dato.
    query_datos = f"""
        SELECT fecha_mes_año AS fecha, precipitation AS valor 
        FROM precipitacion_mensual 
        WHERE id_estacion_fk = '{estacion_selec}' 
        ORDER BY fecha_mes_año
    """
    
    try:
        with engine.connect() as conn:
            df_lluvia = pd.read_sql(text(query_datos), conn)
            
        if not df_lluvia.empty:
            df_lluvia['fecha'] = pd.to_datetime(df_lluvia['fecha'])
            
            # Cálculo (ahora sí funcionará porque df_lluvia tiene columna 'valor')
            df_resultado = hydrogeo_utils.calcular_recarga_simple(df_lluvia, coef_final)
            
            # Métricas
            col1, col2 = st.columns(2)
            total_lluvia = df_resultado['valor'].sum()
            total_recarga = df_resultado['recarga_estimada'].sum()
            
            with col1:
                st.metric("Lluvia Histórica Total", f"{total_lluvia:,.0f} mm")
            with col2:
                st.metric("Agua Infiltrada (Estimada)", f"{total_recarga:,.0f} mm",
                          delta=f"{(total_recarga/total_lluvia)*100:.1f}% Eficiencia")
            
            # Gráfica
            st.subheader("Dinámica Mensual: Lluvia vs. Recarga")
            st.line_chart(df_resultado.set_index('fecha')[['valor', 'recarga_estimada']], color=["#87CEEB", "#00008B"])
            st.caption("Azul Claro: Lluvia | Azul Oscuro: Recarga al Acuífero")
            
        else:
            st.warning(f"La estación seleccionada ({estacion_selec}) no tiene datos en la tabla 'precipitacion_mensual'.")
            
    except Exception as e:
        st.error("Error en la consulta de datos.")
        st.error(f"Detalle: {e}")