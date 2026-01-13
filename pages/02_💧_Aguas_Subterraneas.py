import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import analysis, selectors, interpolation #
from modules import land_cover as lc
from modules.config import Config

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")
st.title("💧 Estimación de Recarga (Modelo Turc)")

# --- 1. CONFIGURACIÓN ---
ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

with st.sidebar:
    st.divider()
    st.subheader("Parametrización")
    
    # Coeficiente Inteligente
    coef_default = 0.30
    if gdf_zona is not None and not gdf_zona.empty:
        try:
            stats = lc.calculate_cover_stats(gdf_zona, Config.LAND_COVER_RASTER_PATH)
            if stats:
                c_sug, razon = lc.get_infiltration_suggestion(stats)
                coef_default = c_sug
                st.caption(f"✨ IA: {razon}")
        except: pass

    coef_final = st.slider("Coef. Infiltración", 0.0, 1.0, float(coef_default))
    temp_estimada = analysis.estimate_temperature(altitud_ref)

# --- 2. MOTOR DE CÁLCULO ---
if ids_seleccionados:
    engine = create_engine(st.secrets["DATABASE_URL"])
    ids_sql = str(tuple(ids_seleccionados)).replace(',)', ')')
    
    # Traemos también coordenadas para poder interpolar
    q = f"""
        SELECT p.fecha_mes_año AS fecha, p.precipitation AS valor, 
               e.longitude, e.latitude, e.alt_est
        FROM precipitacion_mensual p
        JOIN estaciones e ON p.id_estacion_fk = e.id_estacion
        WHERE p.id_estacion_fk IN {ids_sql}
        ORDER BY fecha
    """
    
    with engine.connect() as conn:
        df_full = pd.read_sql(text(q), conn)
        
    if not df_full.empty:
        df_full['fecha'] = pd.to_datetime(df_full['fecha'])
        
        # --- PESTAÑAS DE ANÁLISIS ---
        tab1, tab2 = st.tabs(["📉 Análisis Temporal (Rápido)", "🗺️ Análisis Espacial Distribuido (Avanzado)"])
        
        # === TAB 1: SERIE DE TIEMPO (LO QUE YA TENÍAS) ===
        with tab1:
            df_ts = df_full.groupby('fecha')['valor'].mean().reset_index()
            df_ts['etr'], df_ts['excedente'] = zip(*df_ts.apply(
                lambda x: analysis.calculate_water_balance_turc(x['valor'], temp_estimada), axis=1
            ))
            df_ts['recarga'] = df_ts['excedente'] * coef_final
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Lluvia Promedio Año", f"{df_ts['valor'].mean()*12:,.0f} mm")
            c2.metric("Recarga Promedio Año", f"{df_ts['recarga'].mean()*12:,.0f} mm")
            c3.metric("Eficiencia", f"{(df_ts['recarga'].sum()/df_ts['valor'].sum())*100:.1f}%")
            
            st.area_chart(df_ts.set_index('fecha')[['valor', 'recarga']], color=["#87CEEB", "#00008B"])

        # === TAB 2: MODELO ESPACIAL (LO NUEVO) ===
        with tab2:
            st.markdown("##### Interpolación de Balance Hídrico Anual")
            st.info("Este modelo genera una superficie continua usando las estaciones internas y vecinas (Buffer).")
            
            if gdf_zona is not None:
                # 1. Preparar Datos: Promedio Anual por Estación
                df_annual_st = df_full.groupby(['longitude', 'latitude'])['valor'].mean().reset_index()
                df_annual_st['valor'] = df_annual_st['valor'] * 12 # Llevar a anual
                
                if len(df_annual_st) >= 3:
                    # 2. Generar Malla (Grid)
                    bounds = gdf_zona.total_bounds # (minx, miny, maxx, maxy)
                    # Ojo: total_bounds es (x_min, y_min, x_max, y_max)
                    # interpolation espera (minx, maxx, miny, maxy)
                    bbox = (bounds[0], bounds[2], bounds[1], bounds[3]) 
                    
                    gx, gy = interpolation.generate_grid_coordinates(bbox, resolution=100j)
                    
                    # 3. Interpolar Lluvia (P)
                    grid_P = interpolation.interpolate_spatial(df_annual_st, 'valor', gx, gy, method='rbf')
                    
                    # 4. Calcular Balance Distribuido (Turc en cada pixel)
                    # Asumimos T constante por ahora (se podría interpolar T también si hay datos)
                    # L(t) es constante para la zona en esta versión simplificada
                    L_t = 300 + 25*temp_estimada + 0.05*(temp_estimada**3)
                    
                    # Fórmula Turc matricial: P / sqrt(0.9 + (P/L)^2)
                    grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
                    grid_Recarga = (grid_P - grid_ETR) * coef_final
                    
                    # 5. Visualizar Mapa de Recarga
                    st.write(f"Balance Espacial (Media Anual):")
                    
                    fig = go.Figure(data=go.Contour(
                        z=grid_Recarga.T, # Transpuesta necesaria para plotly
                        x=gx[:,0], # Eje X
                        y=gy[0,:], # Eje Y
                        colorscale="Blues",
                        colorbar=dict(title="Recarga (mm/año)")
                    ))
                    fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Botón descarga raster (Opcional futuro)
                else:
                    st.warning("Se necesitan al menos 3 estaciones (incluyendo vecinas) para interpolar.")
            else:
                st.warning("Seleccione una Cuenca para activar el modo espacial.")

    else:
        st.warning("No hay datos.")