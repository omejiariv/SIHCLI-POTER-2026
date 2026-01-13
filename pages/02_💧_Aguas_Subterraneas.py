import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import sys
import os

# --- IMPORTS MODULARES ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import analysis, selectors, interpolation, data_processor # <--- Agregamos data_processor
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
    
    # CORRECCIÓN: Consulta SQL simplificada (Sin lat/lon/alt)
    # Solo traemos los datos temporales y el ID para cruzar después
    q = f"""
        SELECT fecha_mes_año AS fecha, precipitation AS valor, id_estacion_fk AS id_estacion
        FROM precipitacion_mensual 
        WHERE id_estacion_fk IN {ids_sql}
        ORDER BY fecha_mes_año
    """
    
    with engine.connect() as conn:
        df_precip = pd.read_sql(text(q), conn)
        
    if not df_precip.empty:
        df_precip['fecha'] = pd.to_datetime(df_precip['fecha'])
        
        # --- CRUZAR CON METADATOS (Lat/Lon) EN PYTHON ---
        # 1. Cargamos estaciones desde el procesador (que sí tiene lat/lon)
        try:
            all_data = data_processor.load_and_process_all_data()
            gdf_stations = all_data[0] # Índice 0 son las estaciones
            
            # 2. Hacemos el Merge (Unión) usando 'id_estacion'
            # Seleccionamos solo columnas necesarias para evitar duplicados
            cols_meta = ['id_estacion', 'latitude', 'longitude', 'alt_est']
            # Filtramos columnas por si alguna no existe
            cols_meta = [c for c in cols_meta if c in gdf_stations.columns]
            
            df_full = pd.merge(df_precip, gdf_stations[cols_meta], on='id_estacion', how='left')
            
        except Exception as e:
            st.error(f"Error cruzando metadatos de estaciones: {e}")
            df_full = df_precip # Fallback (sin coordenadas)

        # --- PESTAÑAS DE ANÁLISIS ---
        tab1, tab2 = st.tabs(["📉 Análisis Temporal (Rápido)", "🗺️ Análisis Espacial Distribuido (Avanzado)"])
        
        # === TAB 1: SERIE DE TIEMPO ===
        with tab1:
            df_ts = df_full.groupby('fecha')['valor'].mean().reset_index()
            # Aplicar Turc
            turc_results = df_ts.apply(
                lambda x: analysis.calculate_water_balance_turc(x['valor'], temp_estimada), 
                axis=1
            )
            # Desempaquetar resultados (etr, q)
            df_ts['etr'] = [x[0] for x in turc_results]
            df_ts['excedente'] = [x[1] for x in turc_results]
            df_ts['recarga'] = df_ts['excedente'] * coef_final
            
            c1, c2, c3 = st.columns(3)
            # Suma anual promedio (Total / (meses/12))
            n_years = len(df_ts)/12 if len(df_ts) > 0 else 1
            c1.metric("Lluvia Promedio Año", f"{df_ts['valor'].sum()/n_years:,.0f} mm")
            c2.metric("Recarga Promedio Año", f"{df_ts['recarga'].sum()/n_years:,.0f} mm")
            c3.metric("Eficiencia Global", f"{(df_ts['recarga'].sum()/df_ts['valor'].sum())*100:.1f}%")
            
            st.area_chart(df_ts.set_index('fecha')[['valor', 'recarga']], color=["#87CEEB", "#00008B"])

        # === TAB 2: MODELO ESPACIAL ===
        with tab2:
            st.markdown("##### Interpolación de Balance Hídrico Anual")
            
            # Verificamos que tengamos coordenadas para interpolar
            if 'longitude' in df_full.columns and 'latitude' in df_full.columns:
                
                if gdf_zona is not None:
                    # 1. Preparar Datos: Promedio Anual por Estación (Espacial)
                    df_annual_st = df_full.groupby(['longitude', 'latitude'])['valor'].mean().reset_index()
                    df_annual_st['valor'] = df_annual_st['valor'] * 12 # Llevar a anual
                    
                    if len(df_annual_st) >= 3:
                        # 2. Generar Malla (Grid)
                        bounds = gdf_zona.total_bounds
                        bbox = (bounds[0], bounds[2], bounds[1], bounds[3]) 
                        
                        gx, gy = interpolation.generate_grid_coordinates(bbox, resolution=100j)
                        
                        # 3. Interpolar Lluvia (P)
                        grid_P = interpolation.interpolate_spatial(df_annual_st, 'valor', gx, gy, method='rbf')
                        
                        if grid_P is not None:
                            # 4. Calcular Balance Distribuido
                            L_t = 300 + 25*temp_estimada + 0.05*(temp_estimada**3)
                            
                            # Fórmula Turc matricial
                            grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
                            grid_Recarga = (grid_P - grid_ETR) * coef_final
                            
                            # 5. Visualizar
                            fig = go.Figure(data=go.Contour(
                                z=grid_Recarga.T,
                                x=gx[:,0],
                                y=gy[0,:],
                                colorscale="Blues",
                                colorbar=dict(title="Recarga (mm/año)"),
                                contours=dict(start=0, end=np.nanmax(grid_Recarga), size=50)
                            ))
                            fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No se pudo generar la interpolación.")
                    else:
                        st.warning("⚠️ Se necesitan al menos 3 estaciones con coordenadas para interpolar.")
                else:
                    st.info("Seleccione una 'Cuenca' en el menú lateral para ver el mapa.")
            else:
                st.warning("⚠️ No se pudieron cargar las coordenadas de las estaciones.")

    else:
        st.warning("No hay datos históricos para esta selección.")