import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os

# --- IMPORTACIÓN DE TU MOTOR DE CÁLCULO ---
# Aseguramos que Python encuentre la carpeta modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import analysis  # <--- Aquí importamos tu archivo analysis.py

# Configuración de la página
st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

st.title("💧 Estimación de Recarga (Modelo Turc)")
st.markdown("""
Este módulo estima la recarga potencial calculando primero el **Balance Hídrico de Turc**.
Se descuenta la **Evapotranspiración Real (ETR)** antes de calcular la infiltración.
""")

# --- 1. CONEXIÓN Y CARGA DE DATOS ---
try:
    db_url = st.secrets["DATABASE_URL"]
    engine = create_engine(db_url)
    
    # Consultamos las estaciones INCLUYENDO LA ALTITUD (alt_est)
    # Necesaria para estimar la temperatura con analysis.py
    query_est = """
        SELECT id_estacion AS codigo, nom_est AS nombre, alt_est AS altitud, municipio, depto_region 
        FROM estaciones 
        ORDER BY nom_est
    """
    with engine.connect() as conn:
        df_estaciones = pd.read_sql(text(query_est), conn)
        
    # Listas para filtros
    lista_municipios = sorted(df_estaciones['municipio'].dropna().unique())
    lista_regiones = sorted(df_estaciones['depto_region'].dropna().unique())

except Exception as e:
    st.error(f"Error conectando a BD: {e}")
    st.stop()

# --- 2. BARRA LATERAL (CONTROLES) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selector de Escala
    tipo_analisis = st.radio(
        "Nivel de Análisis:",
        ["📍 Por Estación", "🏙️ Por Municipio"]
    )
    
    st.divider()
    
    seleccion_ids = []
    seleccion_nombre = ""
    altitud_promedio = 1500 # Valor por defecto
    
    if tipo_analisis == "📍 Por Estación":
        cod_sel = st.selectbox(
            "Seleccione Estación:", 
            options=df_estaciones['codigo'],
            format_func=lambda x: df_estaciones[df_estaciones['codigo'] == x]['nombre'].values[0]
        )
        seleccion_ids = [cod_sel]
        # Obtenemos datos de la estación seleccionada
        fila_est = df_estaciones[df_estaciones['codigo'] == cod_sel].iloc[0]
        seleccion_nombre = fila_est['nombre']
        altitud_promedio = fila_est['altitud'] if pd.notnull(fila_est['altitud']) else 1500
        
    elif tipo_analisis == "🏙️ Por Municipio":
        mun_sel = st.selectbox("Seleccione Municipio:", options=lista_municipios)
        # Filtramos todas las estaciones de ese municipio
        estaciones_mun = df_estaciones[df_estaciones['municipio'] == mun_sel]
        seleccion_ids = estaciones_mun['codigo'].tolist()
        seleccion_nombre = f"Municipio de {mun_sel}"
        altitud_promedio = estaciones_mun['altitud'].mean() if not estaciones_mun['altitud'].isnull().all() else 1500

    st.info(f"📍 **Altitud ref:** {altitud_promedio:.0f} msnm")
    
    # Calculamos Temperatura estimada usando TU librería analysis.py
    temp_estimada = analysis.estimate_temperature(altitud_promedio)
    st.info(f"🌡️ **Temp. estimada:** {temp_estimada:.1f} °C")

    st.divider()
    
    st.subheader("Propiedades del Suelo")
    tipo_suelo = st.selectbox(
        "Tipo de Suelo / Permeabilidad:",
        ["Arenoso (Alta)", "Franco (Media)", "Arcilloso (Baja)", "Rocoso/Fracturado"]
    )
    
    # Mapeo simple de coeficientes (podrías mejorarlo luego con mapas reales)
    mapa_coef = {"Arenoso (Alta)": 0.50, "Franco (Media)": 0.30, "Arcilloso (Baja)": 0.10, "Rocoso/Fracturado": 0.05}
    coef_final = st.slider("Coeficiente Infiltración (%)", 0.0, 1.0, mapa_coef[tipo_suelo])

# --- 3. PROCESAMIENTO ---
if seleccion_ids:
    # Convertimos lista a formato SQL tuple
    ids_sql = str(tuple(seleccion_ids)).replace(',)', ')') 
    
    query_datos = f"""
        SELECT fecha_mes_año AS fecha, precipitation AS valor 
        FROM precipitacion_mensual 
        WHERE id_estacion_fk IN {ids_sql} 
        ORDER BY fecha_mes_año
    """
    
    try:
        with engine.connect() as conn:
            df_data = pd.read_sql(text(query_datos), conn)
            
        if not df_data.empty:
            df_data['fecha'] = pd.to_datetime(df_data['fecha'])
            
            # Si es regional, agrupamos por fecha (promedio de todas las estaciones)
            df_procesado = df_data.groupby('fecha')['valor'].mean().reset_index()
            
            # --- AQUÍ OCURRE LA MAGIA CON analysis.py ---
            # Aplicamos la función Turc fila por fila
            def aplicar_turc(row):
                ppt = row['valor']
                # Llamamos a tu función existente en analysis.py
                etr, q = analysis.calculate_water_balance_turc(ppt, temp_estimada)
                return pd.Series([etr, q])

            df_procesado[['etr', 'excedente_hídrico']] = df_procesado.apply(aplicar_turc, axis=1)
            
            # Calculamos Recarga sobre el excedente (Agua que sobra)
            df_procesado['recarga'] = df_procesado['excedente_hídrico'] * coef_final
            
            # --- VISUALIZACIÓN ---
            st.subheader(f"Balance Hídrico: {seleccion_nombre}")
            
            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            sum_ppt = df_procesado['valor'].sum()
            sum_etr = df_procesado['etr'].sum()
            sum_recarga = df_procesado['recarga'].sum()
            
            c1.metric("Lluvia Total", f"{sum_ppt:,.0f} mm")
            c2.metric("Evapotranspiración (ETR)", f"{sum_etr:,.0f} mm", delta="- Pérdida", delta_color="inverse")
            c3.metric("Recarga Estimada", f"{sum_recarga:,.0f} mm", delta="Agua Subterránea")
            c4.metric("Eficiencia Real", f"{(sum_recarga/sum_ppt)*100:.1f}%", help="% de lluvia que llega al acuífero")
            
            # Gráfica
            st.markdown("##### Dinámica del Balance (Turc)")
            st.line_chart(
                df_procesado.set_index('fecha')[['valor', 'etr', 'recarga']],
                color=["#87CEEB", "#FFA500", "#00008B"] # Celeste (Lluvia), Naranja (ETR), Azul Oscuro (Recarga)
            )
            st.caption("Celeste: Lluvia | Naranja: Evaporación | Azul Oscuro: Recarga")
            
        else:
            st.warning("No hay datos de precipitación para la selección.")
            
    except Exception as e:
        st.error(f"Error en el cálculo: {e}")