import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import sys
import os

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="Aguas Subterráneas",
    page_icon="💧",
    layout="wide"
)

# Importar módulos propios
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors, models, scenarios
except ImportError as e:
    st.error(f"Error al importar módulos: {e}")
    st.stop()

st.title("💧 Estimación de Recarga (Modelo Turc + Escenarios)")

# --- 1. BARRA LATERAL Y SELECTORES ---
# Usamos el selector espacial moderno
ids_seleccionados_dummy, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

# --- SECCIÓN DE ESCENARIOS ---
st.sidebar.divider()
st.sidebar.subheader("⚙️ Pronóstico & Escenarios")
activar_proyeccion = st.sidebar.checkbox("Activar Proyección", value=True)
horizonte_meses = st.sidebar.selectbox("Horizonte (meses):", [12, 24, 60], index=1)
st.sidebar.write("**Configuración del Modelo:**")
usar_sarima = st.sidebar.checkbox("Simular Variabilidad Real", value=True, help="Usa SARIMA para añadir patrones estacionales complejos.")
intensidad_ruido = st.sidebar.slider("Intensidad Variabilidad:", 0.0, 2.0, 1.0, 0.1)

# --- LÓGICA PRINCIPAL ---
# CORRECCIÓN CLAVE: Dependemos de gdf_zona, no de ids_seleccionados
if gdf_zona is not None and not gdf_zona.empty:
    
    engine = create_engine(st.secrets["DATABASE_URL"])
    
    # 1. BÚSQUEDA ESPACIAL: Encontrar estaciones dentro de la zona seleccionada
    with st.spinner(f"🔍 Buscando estaciones en {nombre_seleccion}..."):
        minx, miny, maxx, maxy = gdf_zona.total_bounds
        
        # Consulta optimizada usando el bounding box y ST_X/ST_Y por si las dudas con los nombres
        q_spatial = text("""
            SELECT id_estacion 
            FROM estaciones 
            WHERE ST_X(geom::geometry) BETWEEN :minx AND :maxx 
              AND ST_Y(geom::geometry) BETWEEN :miny AND :maxy
        """)
        
        try:
            df_ids = pd.read_sql(q_spatial, engine, params={"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})
            ids_en_zona = df_ids['id_estacion'].unique().tolist()
            
        except Exception as e:
            st.error(f"Error en búsqueda espacial: {e}")
            ids_en_zona = []

    # 2. SI ENCONTRAMOS ESTACIONES, PROCEDEMOS CON EL MODELO TURC
    if ids_en_zona:
        with st.spinner("⏳ Procesando datos históricos y climáticos..."):
            try:
                # Cargar datos históricos
                df_historico = models.get_data_for_turc(ids_en_zona, engine)
                
                if not df_historico.empty:
                    # Calcular Turc Histórico
                    df_turc_hist = models.calculate_turc_model(df_historico)
                    
                    # Agregar datos de referencia (Altitud y Nombre)
                    q_ref = text(f"SELECT id_estacion, nom_est, alt_est FROM estaciones WHERE id_estacion IN ({','.join(map(str, ids_en_zona))})")
                    df_ref = pd.read_sql(q_ref, engine)
                    df_turc_hist = pd.merge(df_turc_hist, df_ref, left_on='id_estacion', right_on='id_estacion', how='left')
                    
                    # --- PESTAÑAS DE RESULTADOS ---
                    tab_hist, tab_proy = st.tabs(["🏛️ Histórico & Balance", "🔮 Proyección a Futuro"])
                    
                    with tab_hist:
                        # Resumen KPI
                        recarga_prom = df_turc_hist['recarga_mm'].mean()
                        st.metric("Recarga Potencial Promedio (Histórica)", f"{recarga_prom:.1f} mm/año")
                        
                        st.subheader("Balance Hídrico por Estación")
                        st.dataframe(
                            df_turc_hist[['nom_est', 'alt_est', 'p_anual', 'etr_mm', 'recarga_mm', 'coef_escorrentia']]
                            .rename(columns={'nom_est':'Estación', 'alt_est':'Altitud', 'p_anual':'Precipitación', 'etr_mm':'ETR', 'recarga_mm':'Recarga', 'coef_escorrentia':'Coef. Escorrentía'})
                            .style.format("{:.1f}", subset=['Altitud', 'Precipitación', 'ETR', 'Recarga']).format("{:.2f}", subset=['Coef. Escorrentía']),
                            use_container_width=True
                        )

                    with tab_proy:
                        if activar_proyeccion:
                            st.subheader(f"Escenarios de Recarga a {horizonte_meses} meses")
                            
                            # Generar escenarios
                            with st.spinner("💡 Generando futuros posibles..."):
                                df_proyecciones = scenarios.generate_scenarios(
                                    df_turc_hist, 
                                    horizonte_meses=horizonte_meses, 
                                    usar_sarima=usar_sarima, 
                                    ruido_factor=intensidad_ruido
                                )
                            
                            # Visualización
                            if not df_proyecciones.empty:
                                df_plot = df_proyecciones.melt('Fecha', var_name='Escenario', value_name='Recarga_mm')
                                
                                fig = go.Figure()
                                
                                # Áreas sombreadas para escenarios extremos
                                fig.add_trace(go.Scatter(
                                    x=df_plot[df_plot['Escenario']=='Optimista (Húmedo)']['Fecha'],
                                    y=df_plot[df_plot['Escenario']=='Optimista (Húmedo)']['Recarga_mm'],
                                    mode='lines', line=dict(width=0), showlegend=False, name='Max'
                                ))
                                fig.add_trace(go.Scatter(
                                    x=df_plot[df_plot['Escenario']=='Pesimista (Seco)']['Fecha'],
                                    y=df_plot[df_plot['Escenario']=='Pesimista (Seco)']['Recarga_mm'],
                                    mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,255,0.1)', showlegend=False, name='Min'
                                ))

                                # Líneas principales
                                colors = {'Neutro (Tendencial)': 'blue', 'Optimista (Húmedo)': 'green', 'Pesimista (Seco)': 'red'}
                                for escenario, color in colors.items():
                                    subset = df_plot[df_plot['Escenario'] == escenario]
                                    fig.add_trace(go.Scatter(
                                        x=subset['Fecha'], y=subset['Recarga_mm'],
                                        mode='lines', name=escenario, line=dict(color=color, width=3 if escenario == 'Neutro (Tendencial)' else 2)
                                    ))

                                fig.update_layout(
                                    title="Proyección Agregada de Recarga Potencial",
                                    yaxis_title="Recarga Mensual (mm)", hovermode="x unified",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                with st.expander("Ver datos de proyección"):
                                    st.dataframe(df_proyecciones.style.format("{:.1f}"), use_container_width=True)
                            else:
                                st.warning("No se pudieron generar proyecciones.")
                        else:
                            st.info("Active la casilla 'Activar Proyección' en la barra lateral.")
                else:
                    st.warning("No hay suficientes datos históricos de precipitación para las estaciones de esta zona.")
            except Exception as e:
                st.error(f"Error en el cálculo del modelo: {e}")
    else:
        st.warning(f"No se encontraron estaciones monitoreadas dentro de la zona '{nombre_seleccion}'. Pruebe aumentando el Radio Buffer.")

else:
    # Mensaje de bienvenida cuando no hay nada seleccionado
    st.info("👈 Seleccione una Cuenca, Municipio o el Departamento en la barra lateral para comenzar el análisis.")