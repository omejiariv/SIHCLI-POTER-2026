import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os

# Agregar path de módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import hydrogeo_utils

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

st.title("💧 Estimación de Recarga: Escala Multiescalar")
st.markdown("Análisis de infiltración y recarga de acuíferos desde nivel de estación hasta escala regional.")

# --- 1. CONEXIÓN Y CARGA DE LISTAS ---
try:
    db_url = st.secrets["DATABASE_URL"]
    engine = create_engine(db_url)
    
    # Consultamos las listas para los filtros
    with engine.connect() as conn:
        # Lista de Estaciones
        q_est = "SELECT id_estacion AS codigo, nom_est AS nombre FROM estaciones ORDER BY nom_est"
        df_estaciones = pd.read_sql(text(q_est), conn)
        
        # Lista de Municipios (Agrupación Espacial)
        q_mun = "SELECT DISTINCT municipio FROM estaciones WHERE municipio IS NOT NULL ORDER BY municipio"
        df_municipios = pd.read_sql(text(q_mun), conn)

        # Lista de Regiones (Agrupación Regional)
        q_reg = "SELECT DISTINCT depto_region FROM estaciones WHERE depto_region IS NOT NULL ORDER BY depto_region"
        df_regiones = pd.read_sql(text(q_reg), conn)

except Exception as e:
    st.error(f"Error conectando a BD: {e}")
    st.stop()

# --- 2. BARRA LATERAL (CONTROLES) ---
with st.sidebar:
    st.header("⚙️ Configuración del Análisis")
    
    # INTERRUPTOR DE ESCALA
    tipo_analisis = st.radio(
        "Nivel de Agregación:",
        ["📍 Por Estación (Puntual)", "🏙️ Por Municipio", "🌍 Por Región"]
    )
    
    st.divider()
    
    # SELECTOR DINÁMICO
    seleccion_id = None
    seleccion_nombre = ""
    
    if tipo_analisis == "📍 Por Estación (Puntual)":
        seleccion_id = st.selectbox(
            "Seleccione Estación:", 
            options=df_estaciones['codigo'],
            format_func=lambda x: df_estaciones[df_estaciones['codigo'] == x]['nombre'].values[0]
        )
        seleccion_nombre = df_estaciones[df_estaciones['codigo'] == seleccion_id]['nombre'].values[0]
        
    elif tipo_analisis == "🏙️ Por Municipio":
        seleccion_id = st.selectbox("Seleccione Municipio:", options=df_municipios['municipio'])
        seleccion_nombre = f"Municipio de {seleccion_id}"
        
    elif tipo_analisis == "🌍 Por Región":
        seleccion_id = st.selectbox("Seleccione Región:", options=df_regiones['depto_region'])
        seleccion_nombre = f"Región {seleccion_id}"

    st.divider()
    
    # PARÁMETROS DE SUELO (Aplican a toda la selección)
    st.subheader("Propiedades del Suelo")
    tipo_suelo = st.selectbox(
        "Tipo de Suelo Dominante:",
        ["Arenoso (Alta Infiltración)", "Franco (Media Infiltración)", "Arcilloso (Baja Infiltración)", "Urbano/Impermeable"]
    )
    coef_sugerido = hydrogeo_utils.obtener_clasificacion_suelo(tipo_suelo)
    coef_final = st.slider("Coeficiente de Infiltración (%)", 0.0, 1.0, coef_sugerido)
    st.info(f"Se asume infiltración del **{coef_final*100:.0f}%**.")

# --- 3. LÓGICA DE CONSULTA Y ANÁLISIS ---
if seleccion_id:
    
    # Construcción de la Query según el tipo de análisis
    if tipo_analisis == "📍 Por Estación (Puntual)":
        # Query Simple (la que ya tenías)
        query = f"""
            SELECT fecha_mes_año AS fecha, precipitation AS valor 
            FROM precipitacion_mensual 
            WHERE id_estacion_fk = '{seleccion_id}' 
            ORDER BY fecha_mes_año
        """
        metric_label = "Estación"
        
    else:
        # Query Agregada (El promedio regional)
        # Hacemos JOIN entre tablas para filtrar por municipio/región
        filtro_col = "municipio" if "Municipio" in tipo_analisis else "depto_region"
        
        query = f"""
            SELECT 
                p.fecha_mes_año AS fecha, 
                AVG(p.precipitation) AS valor 
            FROM precipitacion_mensual p
            JOIN estaciones e ON p.id_estacion_fk = e.id_estacion
            WHERE e.{filtro_col} = '{seleccion_id}'
            GROUP BY p.fecha_mes_año
            ORDER BY p.fecha_mes_año
        """
        metric_label = "Promedio Areal"

    # --- EJECUCIÓN ---
    try:
        with engine.connect() as conn:
            df_data = pd.read_sql(text(query), conn)
            
        if not df_data.empty:
            df_data['fecha'] = pd.to_datetime(df_data['fecha'])
            
            # Cálculo de Recarga
            df_resultado = hydrogeo_utils.calcular_recarga_simple(df_data, coef_final)
            
            # --- DASHBOARD DE RESULTADOS ---
            st.subheader(f"Resultados para: {seleccion_nombre}")
            
            # KPIs
            col1, col2, col3 = st.columns(3)
            total_lluvia = df_resultado['valor'].sum()
            total_recarga = df_resultado['recarga_estimada'].sum()
            
            with col1:
                st.metric("Lluvia Acumulada (Serie)", f"{total_lluvia:,.0f} mm")
            with col2:
                st.metric("Recarga Potencial Total", f"{total_recarga:,.0f} mm")
            with col3:
                st.metric("Eficiencia de Recarga", f"{coef_final*100:.0f}%", help="Porcentaje de lluvia que se convierte en agua subterránea")

            # Gráficas
            tab1, tab2 = st.tabs(["📉 Serie Temporal", "📊 Análisis Anual"])
            
            with tab1:
                st.markdown("##### Dinámica Mensual Histórica")
                st.line_chart(df_resultado.set_index('fecha')[['valor', 'recarga_estimada']], color=["#87CEEB", "#00008B"])
            
            with tab2:
                # Agregación por año para ver tendencias macro
                df_anual = df_resultado.resample('YE', on='fecha').sum()
                st.markdown("##### Recarga Total por Año")
                st.bar_chart(df_anual['recarga_estimada'], color="#00008B")

        else:
            st.warning(f"No se encontraron datos de precipitación para {seleccion_nombre}.")

    except Exception as e:
        st.error("Error procesando los datos.")
        st.write(e)