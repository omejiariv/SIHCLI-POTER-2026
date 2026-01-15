# Módulo de Soporte a Decisiones
import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import sys
import os

# --- SETUP ---
st.set_page_config(page_title="Matriz de Decisiones", page_icon="🎯", layout="wide")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors, interpolation
except Exception as e:
    st.error(f"Error de sistema: {e}")
    st.stop()

st.title("🎯 Priorización de Áreas de Intervención")
st.markdown("""
**Análisis Multicriterio (AHP Simplificado):** Cruzamos la oferta hídrica con la sensibilidad ecosistémica para identificar **Lotes Prioritarios** para conservación o restauración.
""")

# --- 1. SELECTOR ---
# Ahora ids_seleccionados vendrá vacío, pero gdf_zona traerá la geometría clave
ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

# --- 2. PONDERACIÓN (Barra Lateral) ---
with st.sidebar:
    st.divider()
    st.header("⚖️ Criterios de Decisión")
    
    st.info("Define qué importancia tiene cada variable para el objetivo actual.")
    
    w_agua = st.slider("💧 Peso: Importancia Hídrica (Recarga)", 0, 100, 60, 5)
    w_bio = st.slider("🍃 Peso: Valor Ecosistémico", 0, 100, 40, 5)
    
    # Normalización automática
    total = w_agua + w_bio
    if total == 0: total = 1
    pct_agua = w_agua / total
    pct_bio = w_bio / total
    
    st.caption(f"**Distribución Final:** Agua {pct_agua:.0%} | Bio {pct_bio:.0%}")
    
    st.divider()
    st.subheader("Umbrales de Gestión")
    umbral_prioridad = st.slider("Filtrar solo Prioridad Alta (%)", 0, 90, 70)

# --- 3. MOTOR DE ANÁLISIS ---
# CORRECCIÓN: Ya no dependemos de ids_seleccionados, solo de que exista una zona
if gdf_zona is not None and not gdf_zona.empty:
    engine = create_engine(st.secrets["DATABASE_URL"])
    
    # 1. Obtener Bounding Box de la zona para filtrar SQL rápido
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    
    # 2. Consulta Espacial a BD: Traer estaciones que caigan en ese recuadro
    q = text(f"""
        SELECT 
            e.id_estacion, 
            e.latitude, e.longitude, e.alt_est,
            AVG(p.precipitation) * 12 as p_anual
        FROM estaciones e
        JOIN precipitacion_mensual p ON e.id_estacion = p.id_estacion_fk
        WHERE e.longitude BETWEEN :minx AND :maxx 
          AND e.latitude BETWEEN :miny AND :maxy
        GROUP BY e.id_estacion, e.latitude, e.longitude, e.alt_est
    """)
    
    try:
        df_raw = pd.read_sql(q, engine, params={"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})
        
        # 3. Filtrado Fino (Clip Geométrico)
        # Convertimos estaciones a GeoDataFrame
        if not df_raw.empty:
            gdf_estaciones = gpd.GeoDataFrame(
                df_raw, geometry=gpd.points_from_xy(df_raw.longitude, df_raw.latitude), crs="EPSG:4326"
            )
            
            # Recortamos con la forma exacta de la cuenca (para quitar las que están en el recuadro pero fuera de la cuenca)
            # Nota: Si el usuario aplicó buffer en el selector, gdf_zona ya tiene el buffer.
            gdf_estaciones_clip = gpd.clip(gdf_estaciones, gdf_zona)
            df_data = pd.DataFrame(gdf_estaciones_clip.drop(columns='geometry'))
        else:
            df_data = pd.DataFrame()

        # --- VALIDACIÓN DE DATOS SUFICIENTES ---
        if len(df_data) >= 3: 
            with st.spinner("🧮 Calculando matriz de priorización territorial..."):
                
                # A. Generar Rejilla (Grid)
                gx, gy = interpolation.generate_grid_coordinates((minx, maxx, miny, maxy), resolution=60j)
                
                # B. Interpolación (Agua y Altitud)
                grid_P = interpolation.interpolate_spatial(df_data, 'p_anual', gx, gy, method='rbf')
                grid_Alt = interpolation.interpolate_spatial(df_data, 'alt_est', gx, gy, method='linear')
                
                # Fallback si falla altitud
                if grid_Alt is None: grid_Alt = np.full_like(grid_P, altitud_ref)
                
                # C. Modelo Matemático
                # Temp estimada
                grid_T = 30 - (0.0065 * grid_Alt)
                
                # Turc (Recarga)
                L_t = 300 + 25*grid_T + 0.05*(grid_T**3)
                with np.errstate(divide='ignore', invalid='ignore'):
                    grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
                    grid_R = grid_P - grid_ETR
                grid_R = np.nan_to_num(grid_R, nan=0).clip(min=0)
                
                # Normalización (0-1)
                max_R = np.max(grid_R)
                norm_R = grid_R / max_R if max_R > 0 else grid_R

                # Valor Ecosistémico (Simulado: Altura + Lluvia)
                raw_Bio = (grid_Alt * 0.7) + (grid_P * 0.3)
                max_B = np.max(raw_Bio)
                norm_Bio = raw_Bio / max_B if max_B > 0 else raw_Bio
                
                # D. Score Final Ponderado
                grid_Score = (norm_R * pct_agua) + (norm_Bio * pct_bio)
                
                # Máscara de visualización
                mask = grid_Score >= (umbral_prioridad / 100.0)
                grid_Score_Filtered = np.where(mask, grid_Score, np.nan)

                # --- VISUALIZACIÓN ---
                col_map, col_stats = st.columns([3, 1])
                
                with col_map:
                    fig = go.Figure()
                    
                    # Mapa de Prioridad
                    fig.add_trace(go.Contour(
                        z=grid_Score_Filtered,
                        x=gx[0], y=gy[:,0],
                        colorscale="RdYlGn", 
                        colorbar=dict(title="Índice Prioridad", len=0.8),
                        hoverinfo='z', name="Prioridad"
                    ))
                    
                    # Contorno Zona
                    for idx, row in gdf_zona.iterrows():
                        geom = row.geometry
                        polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                        for poly in polys:
                            x, y = poly.exterior.xy
                            fig.add_trace(go.Scatter(
                                x=list(x), y=list(y), mode='lines', 
                                line=dict(color='black', width=2), hoverinfo='skip', showlegend=False
                            ))
                    
                    # Estaciones usadas
                    fig.add_trace(go.Scatter(
                        x=df_data['longitude'], y=df_data['latitude'],
                        mode='markers', marker=dict(color='blue', size=5, symbol='x'),
                        name='Estaciones Base'
                    ))

                    fig.update_layout(
                        title=f"Mapa de Priorización: {nombre_seleccion}",
                        height=600, margin=dict(l=0, r=0, t=40, b=0),
                        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_stats:
                    st.subheader("Análisis")
                    active_cells = np.count_nonzero(~np.isnan(grid_Score_Filtered))
                    total_cells = grid_Score.size
                    pct_area = active_cells / total_cells
                    
                    st.metric("Área Prioritaria", f"{pct_area:.1%}", delta="del territorio total")
                    
                    if pct_agua > 0.7:
                        st.success("💧 **Estrategia: Hídrica.** Maximizar recarga de acuíferos.")
                    elif pct_bio > 0.7:
                        st.success("🍃 **Estrategia: Conservación.** Proteger ecosistemas altoandinos.")
                    else:
                        st.info("⚖️ **Estrategia: Integral.** Manejo balanceado del territorio.")
                        
                    st.caption(f"Basado en {len(df_data)} estaciones meteorológicas.")

        else:
            # MENSAJE DE AYUDA SI NO HAY DATOS
            st.warning(f"⚠️ **Datos insuficientes en esta zona.**")
            st.write(f"Se encontraron **{len(df_data)} estaciones** dentro del área seleccionada. Se necesitan mínimo 3 para triangular y crear el mapa de calor.")
            st.info("👉 **Solución:** Aumenta el 'Radio Buffer (km)' en la barra lateral izquierda para incluir estaciones vecinas.")
            
    except Exception as e:
        st.error(f"Error en el motor de cálculo: {e}")

else:
    st.info("👈 Seleccione una zona para iniciar la matriz de decisiones.")