import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os

# --- IMPORTS MODULARES ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import analysis      # Tu cerebro matemático
from modules import selectors     # <--- TU NUEVO SELECTOR UNIVERSAL

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

st.title("💧 Estimación de Recarga (Modelo Turc)")
st.markdown("Balance hídrico integrado (Lluvia - ETR) para estimación de recarga de acuíferos.")

# --- 1. BARRA LATERAL (EL NUEVO CEREBRO) ---
# Esta línea carga estaciones, municipios, regiones Y CUENCAS automáticamente
ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

# Configuración específica del módulo (lo que es único de hidrogeología)
with st.sidebar:
    st.divider()
    st.subheader("Propiedades del Suelo")
    tipo_suelo = st.selectbox(
        "Permeabilidad:",
        ["Arenoso (Alta)", "Franco (Media)", "Arcilloso (Baja)", "Rocoso"]
    )
    mapa_coef = {"Arenoso (Alta)": 0.50, "Franco (Media)": 0.30, "Arcilloso (Baja)": 0.10, "Rocoso": 0.05}
    coef_final = st.slider("Coeficiente Infiltración (%)", 0.0, 1.0, mapa_coef[tipo_suelo])
    
    # Datos de referencia calculados
    temp_estimada = analysis.estimate_temperature(altitud_ref)
    st.info(f"📍 **Zona:** {nombre_seleccion}")
    st.info(f"🌡️ **Temp. Base:** {temp_estimada:.1f}°C")

# --- 2. CÁLCULOS Y VISUALIZACIÓN ---
if ids_seleccionados:
    try:
        # Conexión para traer las series de tiempo
        engine = create_engine(st.secrets["DATABASE_URL"])
        
        # Convertir lista a tuple SQL
        if len(ids_seleccionados) == 1:
            ids_sql = f"('{ids_seleccionados[0]}')"
        else:
            ids_sql = str(tuple(ids_seleccionados))

        query = f"""
            SELECT fecha_mes_año AS fecha, precipitation AS valor 
            FROM precipitacion_mensual 
            WHERE id_estacion_fk IN {ids_sql} 
            ORDER BY fecha_mes_año
        """
        
        with engine.connect() as conn:
            df_data = pd.read_sql(text(query), conn)
            
        if not df_data.empty:
            df_data['fecha'] = pd.to_datetime(df_data['fecha'])
            
            # AGRUPACIÓN: Promediamos todas las estaciones de la cuenca/municipio
            df_procesado = df_data.groupby('fecha')['valor'].mean().reset_index()
            
            # --- MODELO DE TURC (Usando analysis.py) ---
            def aplicar_turc(row):
                etr, q = analysis.calculate_water_balance_turc(row['valor'], temp_estimada)
                return pd.Series([etr, q])

            df_procesado[['etr', 'excedente']] = df_procesado.apply(aplicar_turc, axis=1)
            df_procesado['recarga'] = df_procesado['excedente'] * coef_final
            
            # --- DASHBOARD DE RESULTADOS ---
            st.subheader(f"Balance Hídrico: {nombre_seleccion}")
            
            # Métricas
            c1, c2, c3, c4 = st.columns(4)
            ppt_total = df_procesado['valor'].sum()
            recarga_total = df_procesado['recarga'].sum()
            
            c1.metric("Lluvia Histórica (Media Areal)", f"{ppt_total:,.0f} mm")
            c2.metric("Evapotranspiración (ETR)", f"{df_procesado['etr'].sum():,.0f} mm", delta="- Pérdida", delta_color="inverse")
            c3.metric("Recarga Potencial", f"{recarga_total:,.0f} mm", delta="Agua Subterránea")
            c4.metric("Eficiencia de Recarga", f"{(recarga_total/ppt_total)*100:.1f}%")
            
            # Gráfica
            st.area_chart(
                df_procesado.set_index('fecha')[['valor', 'recarga']],
                color=["#87CEEB", "#00008B"]
            )
            st.caption("Celeste: Lluvia Areal | Azul Oscuro: Recarga Efectiva al Acuífero")
            
        else:
            st.warning("Las estaciones seleccionadas no tienen datos históricos de precipitación.")
            
    except Exception as e:
        st.error(f"Error procesando datos: {e}")
else:
    st.info("👈 Seleccione una Cuenca, Municipio o Estación en el menú lateral para comenzar.")