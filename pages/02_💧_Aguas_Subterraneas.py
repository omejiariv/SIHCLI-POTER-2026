import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os

# --- IMPORTS MODULARES ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import analysis      
from modules import selectors     
from modules import land_cover as lc # Importamos el módulo de coberturas
from modules.config import Config    # Para la ruta del raster

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

st.title("💧 Estimación de Recarga (Modelo Turc)")
st.markdown("Balance hídrico integrado (Lluvia - ETR) para estimación de recarga de acuíferos.")

# --- 1. BARRA LATERAL (SELECTOR + INTELIGENCIA) ---
ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

with st.sidebar:
    st.divider()
    st.subheader("Propiedades del Suelo")
    
    # --- LÓGICA DE COEFICIENTE INTELIGENTE ---
    coef_default = 0.30
    mensaje_sugerencia = "Selección manual"
    
    # Si tenemos una zona válida (Cuenca/Municipio), calculamos coberturas
    if gdf_zona is not None and not gdf_zona.empty:
        try:
            with st.spinner("Analizando cobertura del suelo..."):
                # 1. Calcular porcentajes de cobertura (Bosque, Pasto, etc.)
                stats = lc.calculate_cover_stats(gdf_zona, Config.LAND_COVER_RASTER_PATH)
                
                if stats:
                    # 2. Obtener sugerencia ponderada
                    coef_sugerido, razon = lc.get_infiltration_suggestion(stats)
                    coef_default = coef_sugerido
                    mensaje_sugerencia = f"✨ Sugerido por IA: {razon}"
                    
                    # Mostrar desglose visual
                    st.caption("Distribución de Cobertura Detectada:")
                    df_stats = pd.DataFrame(list(stats.items()), columns=["Tipo", "%"])
                    st.dataframe(df_stats.set_index("Tipo").sort_values("%", ascending=False).head(3), height=100)
                else:
                    mensaje_sugerencia = "⚠️ No hay datos de cobertura (Raster no cubre zona)"
        except Exception as e:
            print(f"Error land cover: {e}")

    # Slider con el valor sugerido por defecto
    coef_final = st.slider(
        "Coeficiente Infiltración (%)", 
        0.0, 1.0, 
        float(coef_default), # Valor dinámico
        help="Porcentaje del Excedente Hídrico que se convierte en recarga."
    )
    st.info(mensaje_sugerencia)
    
    # Datos Climáticos
    temp_estimada = analysis.estimate_temperature(altitud_ref)
    st.divider()
    st.caption(f"📍 Zona: {nombre_seleccion}")
    st.caption(f"🌡️ Temp. Base: {temp_estimada:.1f}°C")

# --- 2. CÁLCULOS Y VISUALIZACIÓN ---
if ids_seleccionados:
    try:
        engine = create_engine(st.secrets["DATABASE_URL"])
        
        # Convertir lista a tuple SQL
        ids_sql = str(tuple(ids_seleccionados)).replace(',)', ')')
        
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
            
            # Agrupación Areal
            df_procesado = df_data.groupby('fecha')['valor'].mean().reset_index()
            
            # Modelo Turc
            def aplicar_turc(row):
                etr, q = analysis.calculate_water_balance_turc(row['valor'], temp_estimada)
                return pd.Series([etr, q])

            df_procesado[['etr', 'excedente']] = df_procesado.apply(aplicar_turc, axis=1)
            
            # Recarga usando el coeficiente (que puede venir de la IA)
            df_procesado['recarga'] = df_procesado['excedente'] * coef_final
            
            # --- DASHBOARD ---
            st.subheader(f"Balance Hídrico: {nombre_seleccion}")
            
            c1, c2, c3, c4 = st.columns(4)
            ppt_total = df_procesado['valor'].sum()
            recarga_total = df_procesado['recarga'].sum()
            
            c1.metric("Lluvia (Media Areal)", f"{ppt_total:,.0f} mm")
            c2.metric("ETR (Evaporación)", f"{df_procesado['etr'].sum():,.0f} mm", delta="Pérdida", delta_color="inverse")
            c3.metric("Recarga Potencial", f"{recarga_total:,.0f} mm", delta="Ganancia Acuífero")
            c4.metric("Tasa de Recarga", f"{(recarga_total/ppt_total)*100:.1f}%")
            
            st.area_chart(
                df_procesado.set_index('fecha')[['valor', 'recarga']],
                color=["#87CEEB", "#00008B"]
            )
            
        else:
            st.warning("Las estaciones seleccionadas no tienen datos históricos.")
            
    except Exception as e:
        st.error(f"Error procesando datos: {e}")
else:
    st.info("👈 Seleccione una zona en el menú lateral.")