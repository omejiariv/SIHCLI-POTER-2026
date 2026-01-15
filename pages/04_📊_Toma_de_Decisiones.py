# Módulo de Soporte a Decisiones

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from sqlalchemy import create_engine
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
**Análisis Multicriterio:** Este mapa identifica las zonas donde la inversión en conservación tendrá mayor impacto, cruzando oferta hídrica y valor ecosistémico.
""")

# --- 1. SELECTOR ---
ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

# --- 2. PONDERACIÓN ---
with st.sidebar:
    st.divider()
    st.header("⚖️ Criterios de Decisión")
    w_agua = st.slider("💧 Peso: Hídrico", 0, 100, 60, 5)
    w_bio = st.slider("🍃 Peso: Ecosistémico", 0, 100, 40, 5)
    
    total = w_agua + w_bio
    pct_agua = w_agua / (total if total > 0 else 1)
    pct_bio = w_bio / (total if total > 0 else 1)
    
    st.caption(f"**Distribución:** Agua {pct_agua:.0%} | Bio {pct_bio:.0%}")
    st.divider()
    # Bajamos el default a 0 para ver TODO el mapa primero, luego el usuario filtra
    umbral_prioridad = st.slider("Filtrar Prioridad Alta (%)", 0, 90, 0, help="Sube este valor para ver solo las zonas críticas.")

# --- 3. MOTOR DE ANÁLISIS ---
if gdf_zona is not None and not gdf_zona.empty:
    engine = create_engine(st.secrets["DATABASE_URL"])
    
    try:
        # A. CARGAR CATÁLOGO MAESTRO (CSV)
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mapaCVENSO.csv')
        
        if os.path.exists(csv_path):
            df_estaciones_all = pd.read_csv(
                csv_path, 
                sep=';',            
                decimal=',',        
                encoding='latin-1', 
                usecols=['Id_estacio', 'Longitud_geo', 'Latitud_geo', 'alt_est', 'Nom_Est']
            )
            
            df_estaciones_all.rename(columns={
                'Id_estacio': 'id_estacion',
                'Longitud_geo': 'longitude',
                'Latitud_geo': 'latitude',
                'Nom_Est': 'nombre'
            }, inplace=True)
            
            # Limpieza
            df_estaciones_all['latitude'] = pd.to_numeric(df_estaciones_all['latitude'], errors='coerce')
            df_estaciones_all['longitude'] = pd.to_numeric(df_estaciones_all['longitude'], errors='coerce')
            df_estaciones_all['alt_est'] = pd.to_numeric(df_estaciones_all['alt_est'], errors='coerce')
            df_estaciones_all.dropna(subset=['latitude', 'longitude'], inplace=True)
            df_estaciones_all['alt_est'] = df_estaciones_all['alt_est'].fillna(altitud_ref)

            # B. FILTRO ESPACIAL
            # Usamos un buffer pequeño para asegurar que atrapamos estaciones en el borde
            minx, miny, maxx, maxy = gdf_zona.buffer(0.02).total_bounds
            mask = (
                (df_estaciones_all['longitude'] >= minx) & (df_estaciones_all['longitude'] <= maxx) &
                (df_estaciones_all['latitude'] >= miny) & (df_estaciones_all['latitude'] <= maxy)
            )
            df_filtrada = df_estaciones_all[mask].copy()
            
            # C. DATOS DE PRECIPITACIÓN (SQL)
            if not df_filtrada.empty:
                ids_validos = df_filtrada['id_estacion'].unique()
                ids_sql = ",".join([f"'{str(x)}'" for x in ids_validos])
                
                if ids_sql:
                    q_ppt = f"""
                        SELECT id_estacion_fk as id_estacion, AVG(precipitation) * 12 as p_anual
                        FROM precipitacion_mensual
                        WHERE id_estacion_fk IN ({ids_sql})
                        GROUP BY id_estacion_fk
                    """
                    df_ppt = pd.read_sql(q_ppt, engine)
                    
                    df_filtrada['id_estacion'] = df_filtrada['id_estacion'].astype(str)
                    df_ppt['id_estacion'] = df_ppt['id_estacion'].astype(str)
                    
                    df_data = pd.merge(df_filtrada, df_ppt, on='id_estacion', how='inner')
                else:
                    df_data = pd.DataFrame()
            else:
                df_data = pd.DataFrame()

            # --- D. CÁLCULOS VISUALES ---
            if len(df_data) >= 3:
                with st.spinner("🎨 Pintando mapa de prioridades..."):
                    # 1. Grilla (Aumentamos un poco la resolución para suavidad)
                    gx, gy = interpolation.generate_grid_coordinates((minx, maxx, miny, maxy), resolution=80j)
                    
                    # 2. Interpolación (CORREGIDO: Usamos 'nearest' o 'rbf' para llenar huecos)
                    # 'linear' dejaba huecos blancos. 'nearest' llena todo el rectángulo.
                    grid_P = interpolation.interpolate_spatial(df_data, 'p_anual', gx, gy, method='rbf') # RBF es suave y llena todo
                    grid_Alt = interpolation.interpolate_spatial(df_data, 'alt_est', gx, gy, method='nearest') # Nearest es robusto para relieve
                    
                    # Fallback de seguridad
                    if grid_Alt is None: grid_Alt = np.full_like(grid_P, altitud_ref)
                    if grid_P is None: 
                         st.warning("No se pudo interpolar la lluvia. Verifica los datos.")
                         st.stop()
                    
                    # 3. Modelo Matemático (Turc & Bio)
                    grid_T = 30 - (0.0065 * grid_Alt)
                    L_t = 300 + 25*grid_T + 0.05*(grid_T**3)
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
                        grid_R = grid_P - grid_ETR
                    grid_R = np.nan_to_num(grid_R, nan=0).clip(min=0)
                    
                    # Normalización
                    max_R = np.nanmax(grid_R)
                    norm_R = grid_R / max_R if max_R > 0 else grid_R
                    
                    raw_Bio = (grid_Alt * 0.7) + (grid_P * 0.3)
                    max_B = np.nanmax(raw_Bio)
                    norm_Bio = raw_Bio / max_B if max_B > 0 else raw_Bio
                    
                    # 4. Score Final
                    grid_Score = (norm_R * pct_agua) + (norm_Bio * pct_bio)
                    
                    # Filtro de Umbral (Visualización)
                    # Si el usuario pone el umbral en 0, se ve todo.
                    mask_score = grid_Score >= (umbral_prioridad / 100.0)
                    grid_Final = np.where(mask_score, grid_Score, np.nan)

                    # 5. Visualización (Plotly)
                    col_map, col_res = st.columns([3, 1])
                    
                    with col_map:
                        fig = go.Figure()
                        
                        # A. Mapa de Calor (Contour)
                        # connectgaps=True es clave para evitar agujeros
                        fig.add_trace(go.Contour(
                            z=grid_Final, x=gx[0], y=gy[:,0],
                            colorscale="RdYlGn", 
                            colorbar=dict(title="Prioridad (0-1)", len=0.8),
                            hoverinfo='z+x+y', 
                            name="Prioridad",
                            connectgaps=True, 
                            line_smoothing=0.5,
                            contours=dict(coloring='heatmap') # Relleno sólido tipo heatmap
                        ))
                        
                        # B. Límites de la Zona (Negro)
                        for idx, row in gdf_zona.iterrows():
                            geom = row.geometry
                            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                            for poly in polys:
                                x, y = poly.exterior.xy
                                fig.add_trace(go.Scatter(
                                    x=list(x), y=list(y), mode='lines', 
                                    line=dict(color='black', width=3), showlegend=False, hoverinfo='skip'
                                ))
                        
                        # C. Estaciones (Puntos Azules)
                        fig.add_trace(go.Scatter(
                            x=df_data['longitude'], y=df_data['latitude'],
                            mode='markers', 
                            marker=dict(color='blue', size=6, symbol='circle', line=dict(width=1, color='white')),
                            name='Estaciones',
                            text=df_data['nombre'],
                            hoverinfo='text'
                        ))

                        fig.update_layout(
                            title=f"Mapa de Priorización: {nombre_seleccion}",
                            height=650, margin=dict(l=20,r=20,t=50,b=20),
                            xaxis=dict(visible=False, showgrid=False), 
                            yaxis=dict(visible=False, showgrid=False, scaleanchor="x"),
                            plot_bgcolor='rgba(0,0,0,0)' # Fondo transparente
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_res:
                        st.markdown("### 📊 Resultados")
                        
                        valid_cells = np.count_nonzero(~np.isnan(grid_Final))
                        total_grid_cells = grid_Final.size
                        pct_visible = valid_cells / total_grid_cells if total_grid_cells > 0 else 0
                        
                        # Métricas más claras
                        st.metric("Cobertura del Mapa", f"{pct_visible:.1%}", help="Porcentaje del área que supera el umbral seleccionado.")
                        
                        st.info(f"**Estaciones Base:** {len(df_data)}")
                        
                        st.markdown("---")
                        st.markdown("**Interpretación:**")
                        st.markdown("🟢 **Verde:** Alta Prioridad (Conservar/Restaurar)")
                        st.markdown("🔴 **Rojo:** Baja Prioridad (O menor impacto relativo)")

            else:
                st.warning("⚠️ Datos insuficientes para interpolar.")
                st.write(f"Estaciones encontradas: {len(df_data)} (Mínimo 3 requeridas).")
                st.info("💡 Aumenta el 'Radio Buffer' en la barra lateral.")

        else:
            st.error("Archivo 'mapaCVENSO.csv' no encontrado.")

    except Exception as e:
        st.error(f"Error en el proceso: {e}")
else:
    st.info("👈 Seleccione una zona para comenzar.")