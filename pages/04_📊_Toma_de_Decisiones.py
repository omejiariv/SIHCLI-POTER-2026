# Módulo de Soporte a Decisiones
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import sys
import os

# --- SETUP ---
st.set_page_config(page_title="Matriz de Decisiones", page_icon="🎯", layout="wide")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors, interpolation, analysis
except Exception as e:
    st.error(f"Error de sistema: {e}")
    st.stop()

st.title("🎯 Priorización de Áreas de Intervención")
st.markdown("""
**Análisis Multicriterio (AHP Simplificado):** Cruzamos la oferta hídrica con la sensibilidad ecosistémica para identificar **Lotes Prioritarios** para conservación o restauración.
""")

# --- 1. SELECTOR (Reutilizamos el cerebro rápido) ---
ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

# --- 2. PONDERACIÓN (Barra Lateral) ---
with st.sidebar:
    st.divider()
    st.header("⚖️ Criterios de Decisión")
    
    st.info("Define qué importancia tiene cada variable para el objetivo actual (ej: Propuesta Syngenta vs Nutresa).")
    
    w_agua = st.slider("💧 Peso: Importancia Hídrica (Recarga)", 0, 100, 60, 5)
    w_bio = st.slider("🍃 Peso: Valor Ecosistémico", 0, 100, 40, 5)
    
    # Normalización automática para que sume 100%
    total = w_agua + w_bio
    if total == 0: total = 1
    pct_agua = w_agua / total
    pct_bio = w_bio / total
    
    st.caption(f"**Distribución Final:** Agua {pct_agua:.0%} | Bio {pct_bio:.0%}")
    
    st.divider()
    st.subheader("Umbrales de Gestión")
    umbral_prioridad = st.slider("Filtrar solo Prioridad Alta (%)", 0, 90, 70, help="Muestra solo áreas con puntaje superior a este valor.")

# --- 3. MOTOR DE ANÁLISIS ---
if ids_seleccionados and gdf_zona is not None:
    engine = create_engine(st.secrets["DATABASE_URL"])
    
    # A. TRAER DATOS CLIMÁTICOS (P y T)
    # Usamos la lógica del Módulo 2 pero simplificada para velocidad
    ids_sql = str(tuple(ids_seleccionados)).replace(',)', ')')
    q = f"""
        SELECT 
            p.id_estacion_fk as id_estacion, 
            AVG(p.precipitation) * 12 as p_anual,
            e.latitude, e.longitude, e.alt_est
        FROM precipitacion_mensual p
        JOIN estaciones e ON p.id_estacion_fk = e.id_estacion
        WHERE p.id_estacion_fk IN {ids_sql}
        GROUP BY p.id_estacion_fk, e.latitude, e.longitude, e.alt_est
    """
    
    df_data = pd.read_sql(q, engine)
    
    if len(df_data) >= 3: # Necesitamos mínimo 3 puntos para interpolar un plano
        with st.spinner("🧮 Calculando matriz de priorización territorial..."):
            
            # 1. Generar Rejilla (Grid) sobre la zona seleccionada
            bounds = gdf_zona.total_bounds
            # Resolución media (50x50) para que sea instantáneo
            gx, gy = interpolation.generate_grid_coordinates((bounds[0], bounds[2], bounds[1], bounds[3]), resolution=60j)
            
            # 2. Interpolación de Precipitación (Capa Agua Base)
            grid_P = interpolation.interpolate_spatial(df_data, 'p_anual', gx, gy, method='rbf')
            
            # 3. Estimación de Temperatura (basada en altitud proxy o interpolada si tuvieramos)
            # Como no tenemos raster de elevación cargado aquí, usaremos la P como proxy inverso o interpolaremos Altitud si hay datos
            grid_Alt = interpolation.interpolate_spatial(df_data, 'alt_est', gx, gy, method='linear')
            if grid_Alt is None: grid_Alt = np.full_like(grid_P, altitud_ref)
            
            # Temp estimada = 30 - 0.0065 * Altura
            grid_T = 30 - (0.0065 * grid_Alt)
            
            # 4. CÁLCULO DE CAPAS (NORMALIZADAS 0-1)
            
            # --- CAPA 1: RECARGA POTENCIAL (TURC) ---
            # L(t) = 300 + 25T + 0.05T^3
            L_t = 300 + 25*grid_T + 0.05*(grid_T**3)
            # ETR
            with np.errstate(divide='ignore', invalid='ignore'):
                grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
                grid_R = grid_P - grid_ETR
            grid_R = np.nan_to_num(grid_R, nan=0).clip(min=0)
            
            # Normalizar (0 a 1)
            max_R = np.max(grid_R)
            norm_R = grid_R / max_R if max_R > 0 else grid_R

            # --- CAPA 2: VALOR ECOSISTÉMICO (SIMULADO PARA MVP) ---
            # En el futuro, esto leerá el raster de Cobertura. 
            # Por ahora, asumiremos que zonas más altas y húmedas tienen más valor bio (paramos)
            # Lógica: Mayor Altitud + Mayor Lluvia = Mayor probabilidad de ecosistema estratégico
            raw_Bio = (grid_Alt * 0.7) + (grid_P * 0.3)
            max_B = np.max(raw_Bio)
            norm_Bio = raw_Bio / max_B if max_B > 0 else raw_Bio
            
            # 5. SUPERPOSICIÓN PONDERADA (ALGORITMO DE DECISIÓN)
            # Score = (w1 * R) + (w2 * B)
            grid_Score = (norm_R * pct_agua) + (norm_Bio * pct_bio)
            
            # Filtrar por umbral (máscara)
            mask = grid_Score >= (umbral_prioridad / 100.0)
            grid_Score_Filtered = np.where(mask, grid_Score, np.nan)

            # --- VISUALIZACIÓN ---
            
            col_map, col_stats = st.columns([3, 1])
            
            with col_map:
                fig = go.Figure()
                
                # Mapa de Calor (Prioridad)
                fig.add_trace(go.Contour(
                    z=grid_Score_Filtered,
                    x=gx[0], y=gy[:,0],
                    colorscale="RdYlGn", # Rojo (Bajo) a Verde (Alto Prioridad)
                    reversescale=False,
                    connectgaps=False,
                    line_smoothing=0.85,
                    opacity=0.8,
                    colorbar=dict(title="Índice de Prioridad", len=0.8),
                    hoverinfo='z',
                    name="Prioridad"
                ))
                
                # Contorno de la zona
                for idx, row in gdf_zona.iterrows():
                    geom = row.geometry
                    polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                    for poly in polys:
                        x, y = poly.exterior.xy
                        fig.add_trace(go.Scatter(
                            x=list(x), y=list(y), mode='lines', 
                            line=dict(color='black', width=2), hoverinfo='skip', showlegend=False
                        ))
                
                # Estaciones como referencia
                fig.add_trace(go.Scatter(
                    x=df_data['longitude'], y=df_data['latitude'],
                    mode='markers', marker=dict(color='black', size=5),
                    name='Puntos de Control'
                ))

                fig.update_layout(
                    title=f"Mapa de Priorización: {nombre_seleccion}",
                    height=600,
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col_stats:
                st.subheader("Estadísticas del Escenario")
                
                # Simular Hectáreas (Aprox)
                # Area total aprox del bounding box * porcentaje de celdas activas
                total_cells = grid_Score.size
                active_cells = np.count_nonzero(~np.isnan(grid_Score_Filtered))
                pct_area = active_cells / total_cells
                
                st.metric("Área Prioritaria", f"{pct_area:.1%}", delta="del territorio")
                
                st.markdown("### Recomendación:")
                if pct_agua > 0.7:
                    st.success("🎯 **Enfoque: Seguridad Hídrica.** Ideal para proyectos de Pagos por Servicios Ambientales (PSA) enfocados en recarga.")
                elif pct_bio > 0.7:
                    st.success("🎯 **Enfoque: Conservación Estricta.** Ideal para ampliación de áreas protegidas y corredores biológicos.")
                else:
                    st.info("🎯 **Enfoque: Gestión Integral.** Territorio balanceado. Se recomiendan sistemas agroforestales o restauración productiva.")
                
                st.markdown("---")
                st.write("**Datos base:**")
                st.write(f"- Estaciones usadas: {len(df_data)}")
                st.write(f"- Ponderación Hídrica: {pct_agua:.0%}")
                
    else:
        st.warning("⚠️ Necesitamos al menos 3 estaciones con datos en la zona (o vecinas) para triangular la priorización.")
        st.info("Prueba aumentando el 'Radio Buffer' en la barra lateral izquierda.")

else:
    st.info("👈 Seleccione una zona para iniciar la matriz de decisiones.")