import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import sys
import os
import rasterio
from rasterio.transform import from_origin
import io

# --- IMPORTS MODULARES ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import analysis, selectors, interpolation, data_processor
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

    coef_final = st.slider("Coef. Infiltración", 0.0, 1.0, float(coef_default), help="Fracción del excedente hídrico que infiltra.")
    temp_estimada = analysis.estimate_temperature(altitud_ref)

# --- FUNCIÓN AUXILIAR DESCARGA RASTER ---
def get_geotiff_bytes(grid_data, transform, crs):
    """Genera un archivo GeoTIFF en memoria"""
    mem_file = io.BytesIO()
    with rasterio.open(
        mem_file, 'w',
        driver='GTiff',
        height=grid_data.shape[0],
        width=grid_data.shape[1],
        count=1,
        dtype=grid_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(grid_data, 1)
    return mem_file.getvalue()

# --- 2. MOTOR DE CÁLCULO ---
if ids_seleccionados:
    engine = create_engine(st.secrets["DATABASE_URL"])
    ids_sql = str(tuple(ids_seleccionados)).replace(',)', ')')
    
    # Consulta optimizada
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
        df_precip['año'] = df_precip['fecha'].dt.year
        
        # Cruzar con coordenadas (Python Merge)
        try:
            all_data = data_processor.load_and_process_all_data()
            gdf_stations = all_data[0]
            cols_meta = ['id_estacion', 'latitude', 'longitude', 'nom_est']
            cols_meta = [c for c in cols_meta if c in gdf_stations.columns]
            df_full = pd.merge(df_precip, gdf_stations[cols_meta], on='id_estacion', how='left')
        except:
            df_full = df_precip

        # --- PESTAÑAS ---
        tab1, tab2 = st.tabs(["📉 Análisis Temporal", "🗺️ Mapa de Recarga Distribuida"])
        
        # === TAB 1: ANÁLISIS TEMPORAL (Corregido) ===
        with tab1:
            st.markdown("##### Balance Hídrico Anual (Promedio Areal)")
            
            # 1. Agrupar por AÑO primero (Turc funciona mejor anualmente)
            df_anual = df_full.groupby('año')['valor'].mean().reset_index() # Promedio mensual del año
            df_anual['valor_anual'] = df_anual['valor'] * 12 # Convertir a total anual
            
            # 2. Aplicar Turc a los totales anuales
            turc_res = df_anual.apply(
                lambda x: analysis.calculate_water_balance_turc(x['valor_anual'], temp_estimada), axis=1
            )
            df_anual['etr'] = [x[0] for x in turc_res]
            df_anual['excedente'] = [x[1] for x in turc_res]
            df_anual['recarga'] = df_anual['excedente'] * coef_final
            
            # Métricas Globales (Promedio de la serie histórica anual)
            c1, c2, c3, c4 = st.columns(4)
            ppt_avg = df_anual['valor_anual'].mean()
            recarga_avg = df_anual['recarga'].mean()
            
            c1.metric("Precipitación Media", f"{ppt_avg:,.0f} mm/año")
            c2.metric("ETR (Evaporación)", f"{df_anual['etr'].mean():,.0f} mm/año")
            c3.metric("Recarga Total", f"{recarga_avg:,.0f} mm/año")
            c4.metric("Tasa de Recarga", f"{(recarga_avg/ppt_avg)*100:.1f}%")
            
            # GRÁFICO INTERACTIVO (PLOTLY)
            fig_t = go.Figure()
            
            # Capa Lluvia
            fig_t.add_trace(go.Bar(
                x=df_anual['año'], y=df_anual['valor_anual'],
                name='Precipitación', marker_color='#87CEEB', opacity=0.6
            ))
            
            # Capa ETR
            fig_t.add_trace(go.Scatter(
                x=df_anual['año'], y=df_anual['etr'],
                name='Evapotranspiración (ETR)', line=dict(color='#FFA500', width=2, dash='dot')
            ))
            
            # Capa Recarga
            fig_t.add_trace(go.Scatter(
                x=df_anual['año'], y=df_anual['recarga'],
                name='Recarga Efectiva', line=dict(color='#00008B', width=3), fill='tozeroy'
            ))
            
            fig_t.update_layout(
                title="Dinámica Histórica Anual",
                yaxis_title="Lámina de Agua (mm)",
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_t, use_container_width=True)

        # === TAB 2: MAPA DISTRIBUIDO ===
        with tab2:
            st.markdown("##### Modelo Espacial de Recarga")
            
            if 'longitude' in df_full.columns and gdf_zona is not None:
                # Datos para interpolar (Promedio Histórico Anual por Estación)
                df_spatial = df_full.groupby(['id_estacion', 'nom_est', 'longitude', 'latitude'])['valor'].mean().reset_index()
                df_spatial['valor_anual'] = df_spatial['valor'] * 12
                
                if len(df_spatial) >= 3:
                    # Grid
                    bounds = gdf_zona.total_bounds
                    bbox = (bounds[0], bounds[2], bounds[1], bounds[3])
                    gx, gy = interpolation.generate_grid_coordinates(bbox, resolution=100j)
                    
                    # Interpolar Precipitación
                    grid_P = interpolation.interpolate_spatial(df_spatial, 'valor_anual', gx, gy, method='rbf')
                    
                    if grid_P is not None:
                        # Turc Matricial
                        L_t = 300 + 25*temp_estimada + 0.05*(temp_estimada**3)
                        grid_ETR = grid_P / np.sqrt(0.9 + (grid_P/L_t)**2)
                        grid_Recarga = (grid_P - grid_ETR) * coef_final
                        grid_Recarga = np.nan_to_num(grid_Recarga, nan=0.0) # Limpieza
                        
                        # --- FIGURA MAPA ---
                        fig_map = go.Figure()
                        
                        # 1. Contornos (Recarga)
                        fig_map.add_trace(go.Contour(
                            z=grid_Recarga.T, x=gx[:,0], y=gy[0,:],
                            colorscale="Blues",
                            colorbar=dict(title="Recarga (mm/año)"),
                            name="Recarga Distribuida",
                            contours=dict(start=0, end=np.nanmax(grid_Recarga), size=50, showlabels=True)
                        ))
                        
                        # 2. Estaciones (Puntos)
                        fig_map.add_trace(go.Scatter(
                            x=df_spatial['longitude'], y=df_spatial['latitude'],
                            mode='markers',
                            marker=dict(color='black', size=8, line=dict(width=1, color='white')),
                            text=df_spatial['nom_est'] + ": " + df_spatial['valor_anual'].round(0).astype(str) + " mm",
                            name="Estaciones Usadas"
                        ))
                        
                        # Ajuste visual
                        fig_map.update_layout(
                            height=600, 
                            margin=dict(l=0, r=0, t=20, b=0),
                            xaxis=dict(showgrid=False, zeroline=False, visible=False),
                            yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1)
                        )
                        st.plotly_chart(fig_map, use_container_width=True)
                        
                        # --- DESCARGAS ---
                        col_dl1, col_dl2 = st.columns(2)
                        
                        # Preparar GeoTIFF
                        transform = from_origin(gx[0,0], gy[0,-1], (gx[1,0]-gx[0,0]), (gy[0,0]-gy[0,1]))
                        tiff_bytes = get_geotiff_bytes(np.flipud(grid_Recarga.T), transform, "EPSG:4326")
                        
                        with col_dl1:
                            st.download_button(
                                label="💾 Descargar Raster (GeoTIFF)",
                                data=tiff_bytes,
                                file_name=f"recarga_{nombre_seleccion}.tif",
                                mime="image/tiff"
                            )
                        with col_dl2:
                            csv = pd.DataFrame(df_spatial).to_csv(index=False)
                            st.download_button(
                                "📄 Descargar Datos Estaciones (CSV)",
                                csv, f"estaciones_{nombre_seleccion}.csv", "text/csv"
                            )
                    else:
                        st.warning("Error en interpolación.")
                else:
                    st.warning("Insuficientes estaciones para el mapa (Mínimo 3).")
            else:
                st.info("Seleccione una zona válida.")
    else:
        st.warning("No hay datos históricos.")