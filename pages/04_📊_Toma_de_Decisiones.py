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
    umbral_prioridad = st.slider("Filtrar Prioridad Alta (%)", 0, 90, 70)

# --- 3. MOTOR DE ANÁLISIS ---
if gdf_zona is not None and not gdf_zona.empty:
    engine = create_engine(st.secrets["DATABASE_URL"])
    
    try:
        # A. CARGAR CATÁLOGO MAESTRO DESDE CSV
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mapaCVENSO.csv')
        
        if os.path.exists(csv_path):
            # CORRECCIÓN DE CODIFICACIÓN PARA LA 'Ñ' (latin-1)
            df_estaciones_all = pd.read_csv(
                csv_path, 
                usecols=['Id_estacio', 'Longitud_geo', 'Latitud_geo', 'alt_est', 'Nom_Est'],
                encoding='latin-1'  # <--- AQUÍ ESTÁ LA MAGIA
            )
            
            # RENOMBRAR PARA ESTANDARIZAR
            df_estaciones_all.rename(columns={
                'Id_estacio': 'id_estacion',
                'Longitud_geo': 'longitude',
                'Latitud_geo': 'latitude',
                'Nom_Est': 'nombre'
            }, inplace=True)
            
            # Rellenar altitud
            df_estaciones_all['alt_est'] = df_estaciones_all['alt_est'].fillna(altitud_ref)

            # B. FILTRO ESPACIAL
            minx, miny, maxx, maxy = gdf_zona.total_bounds
            mask = (
                (df_estaciones_all['longitude'] >= minx) & (df_estaciones_all['longitude'] <= maxx) &
                (df_estaciones_all['latitude'] >= miny) & (df_estaciones_all['latitude'] <= maxy)
            )
            df_filtrada = df_estaciones_all[mask].copy()
            
            # C. TRAER PRECIPITACIÓN DESDE SQL
            if not df_filtrada.empty:
                ids_validos = tuple(df_filtrada['id_estacion'].unique())
                
                if len(ids_validos) == 1:
                    ids_sql = f"({ids_validos[0]})"
                else:
                    ids_sql = str(ids_validos)
                
                q_ppt = f"""
                    SELECT id_estacion_fk as id_estacion, AVG(precipitation) * 12 as p_anual
                    FROM precipitacion_mensual
                    WHERE id_estacion_fk IN {ids_sql}
                    GROUP BY id_estacion_fk
                """
                df_ppt = pd.read_sql(q_ppt, engine)
                
                df_data = pd.merge(df_filtrada, df_ppt, on='id_estacion', how='inner')
                
                # Clip Geométrico
                gdf_pts = gpd.GeoDataFrame(
                    df_data, geometry=gpd.points_from_xy(df_data.longitude, df_data.latitude), crs="EPSG:4326"
                )
                gdf_pts = gpd.clip(gdf_pts, gdf_zona)
                df_data = pd.DataFrame(gdf_pts.drop(columns='geometry'))
            else:
                df_data = pd.DataFrame()

            # --- D. CÁLCULOS ---
            if len(df_data) >= 3:
                with st.spinner("🧮 Modelando territorio..."):
                    # 1. Grilla
                    gx, gy = interpolation.generate_grid_coordinates((minx, maxx, miny, maxy), resolution=60j)
                    
                    # 2. Interpolación
                    grid_P = interpolation.interpolate_spatial(df_data, 'p_anual', gx, gy, method='rbf')
                    grid_Alt = interpolation.interpolate_spatial(df_data, 'alt_est', gx, gy, method='linear')
                    
                    if grid_Alt is None: grid_Alt = np.full_like(grid_P, altitud_ref)
                    
                    # 3. Turc & Bio
                    grid_T = 30 - (0.0065 * grid_Alt)
                    L_t = 300 + 25*grid_T + 0.05*(grid_T**3)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
                        grid_R = grid_P - grid_ETR
                    grid_R = np.nan_to_num(grid_R, nan=0).clip(min=0)
                    
                    # Normalizar
                    max_R = np.max(grid_R)
                    norm_R = grid_R / max_R if max_R > 0 else grid_R
                    
                    raw_Bio = (grid_Alt * 0.7) + (grid_P * 0.3)
                    max_B = np.max(raw_Bio)
                    norm_Bio = raw_Bio / max_B if max_B > 0 else raw_Bio
                    
                    # 4. Score
                    grid_Score = (norm_R * pct_agua) + (norm_Bio * pct_bio)
                    mask_score = grid_Score >= (umbral_prioridad / 100.0)
                    grid_Final = np.where(mask_score, grid_Score, np.nan)

                    # 5. Visualización
                    col_map, col_res = st.columns([3, 1])
                    
                    with col_map:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Contour(
                            z=grid_Final, x=gx[0], y=gy[:,0],
                            colorscale="RdYlGn", colorbar=dict(title="Índice", len=0.8),
                            hoverinfo='z', name="Prioridad",
                            connectgaps=False
                        ))
                        
                        for idx, row in gdf_zona.iterrows():
                            geom = row.geometry
                            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                            for poly in polys:
                                x, y = poly.exterior.xy
                                fig.add_trace(go.Scatter(
                                    x=list(x), y=list(y), mode='lines', 
                                    line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'
                                ))
                        
                        fig.add_trace(go.Scatter(
                            x=df_data['longitude'], y=df_data['latitude'],
                            mode='markers', marker=dict(color='blue', size=5, symbol='x'),
                            name='Estaciones'
                        ))

                        fig.update_layout(
                            title=f"Mapa de Priorización: {nombre_seleccion}",
                            height=600, margin=dict(l=0,r=0,t=40,b=0),
                            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_res:
                        st.subheader("Resultados")
                        pct_area = np.count_nonzero(~np.isnan(grid_Final)) / grid_Final.size
                        st.metric("Área Prioritaria", f"{pct_area:.1%}")
                        st.caption(f"Análisis basado en {len(df_data)} estaciones.")
                        
                        if pct_agua > 0.6: st.success("Enfoque: **Seguridad Hídrica**")
                        elif pct_bio > 0.6: st.success("Enfoque: **Biodiversidad**")
                        else: st.info("Enfoque: **Integral**")

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