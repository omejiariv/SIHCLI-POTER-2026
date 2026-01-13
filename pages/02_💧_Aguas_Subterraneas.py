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
from prophet import Prophet # Motor de Inteligencia Artificial para Pronósticos

# --- IMPORTS MODULARES ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import analysis, selectors, interpolation, data_processor
from modules import land_cover as lc
from modules.config import Config

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")
st.title("💧 Estimación de Recarga (Modelo Turc + Pronóstico)")

# --- 1. CONFIGURACIÓN ---
ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

with st.sidebar:
    st.divider()
    st.subheader("🤖 Pronóstico Hidrológico")
    usar_forecast = st.checkbox("Activar Proyección IA", value=False)
    meses_forecast = 12
    if usar_forecast:
        meses_forecast = st.selectbox("Horizonte de proyección:", [6, 12, 18, 24, 36], index=1)
        st.caption(f"Proyectando hasta {meses_forecast} meses en el futuro usando Prophet.")

    st.divider()
    st.subheader("Parametrización Suelo")
    
    # Coeficiente Inteligente
    coef_default = 0.30
    if gdf_zona is not None and not gdf_zona.empty:
        try:
            stats = lc.calculate_cover_stats(gdf_zona, Config.LAND_COVER_RASTER_PATH)
            if stats:
                c_sug, razon = lc.get_infiltration_suggestion(stats)
                coef_default = c_sug
                st.caption(f"✨ IA Cobertura: {razon}")
        except: pass

    coef_final = st.slider("Coef. Infiltración", 0.0, 1.0, float(coef_default), help="Fracción del excedente hídrico que infiltra.")
    temp_estimada = analysis.estimate_temperature(altitud_ref)

# --- FUNCIÓN AUXILIAR DESCARGA RASTER ---
def get_geotiff_bytes(grid_data, transform, crs):
    mem_file = io.BytesIO()
    with rasterio.open(
        mem_file, 'w', driver='GTiff',
        height=grid_data.shape[0], width=grid_data.shape[1],
        count=1, dtype=grid_data.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(grid_data, 1)
    return mem_file.getvalue()

# --- 2. MOTOR DE CÁLCULO ---
if ids_seleccionados:
    engine = create_engine(st.secrets["DATABASE_URL"])
    ids_sql = str(tuple(ids_seleccionados)).replace(',)', ')')
    
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
        
        # Merge Metadatos
        try:
            all_data = data_processor.load_and_process_all_data()
            gdf_stations = all_data[0]
            cols_meta = ['id_estacion', 'latitude', 'longitude', 'nom_est', 'municipio', 'alt_est']
            cols_meta = [c for c in cols_meta if c in gdf_stations.columns]
            df_full = pd.merge(df_precip, gdf_stations[cols_meta], on='id_estacion', how='left')
        except:
            df_full = df_precip

        tab1, tab2 = st.tabs(["📉 Análisis Temporal y Pronóstico", "🗺️ Mapa de Recarga Distribuida"])
        
        # === TAB 1: ANÁLISIS TEMPORAL + PRONÓSTICO ===
        with tab1:
            st.markdown(f"##### Dinámica Histórica y Proyección: {nombre_seleccion}")
            
            # 1. Preparar Serie Temporal Agregada (Promedio de la zona)
            df_ts_monthly = df_full.groupby('fecha')['valor'].mean().reset_index()
            
            df_final_ts = df_ts_monthly.copy()
            df_final_ts['tipo'] = 'Histórico'

            # 2. MOTOR DE PRONÓSTICO (PROPHET)
            if usar_forecast and len(df_ts_monthly) > 24:
                with st.spinner("🧠 Entrenando modelo de IA (Prophet) para proyección climática..."):
                    # Preparar formato para Prophet (ds, y)
                    df_prophet = df_ts_monthly.rename(columns={'fecha': 'ds', 'valor': 'y'})
                    
                    # Entrenar Modelo
                    m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
                    m.fit(df_prophet)
                    
                    # Crear fechas futuras
                    future = m.make_future_dataframe(periods=meses_forecast, freq='MS')
                    forecast = m.predict(future)
                    
                    # Filtrar solo la parte futura para unirla
                    last_date = df_ts_monthly['fecha'].max()
                    df_future = forecast[forecast['ds'] > last_date][['ds', 'yhat']].rename(columns={'ds': 'fecha', 'yhat': 'valor'})
                    df_future['tipo'] = 'Pronóstico'
                    
                    # Unir Historia + Futuro
                    df_final_ts = pd.concat([df_final_ts, df_future])
                    
                    st.success(f"✅ Proyección generada hasta {df_future['fecha'].max().date()}")

            # 3. Calcular Balance Hídrico (Turc) sobre la serie extendida
            # Agrupamos por año para Turc, pero mantenemos la distinción de Tipo
            df_final_ts['año'] = df_final_ts['fecha'].dt.year
            
            # Cálculo Anual
            df_anual = df_final_ts.groupby(['año', 'tipo'])['valor'].sum().reset_index() # Suma de meses para el total anual
            # Filtrar años incompletos en el histórico (opcional, pero mejora la gráfica)
            
            # Aplicar Turc
            turc_res = df_anual.apply(
                lambda x: analysis.calculate_water_balance_turc(x['valor'], temp_estimada), axis=1
            )
            df_anual['etr'] = [x[0] for x in turc_res]
            df_anual['excedente'] = [x[1] for x in turc_res]
            df_anual['recarga'] = df_anual['excedente'] * coef_final
            
            # GRÁFICO
            fig_t = go.Figure()
            
            # Datos Históricos
            hist = df_anual[df_anual['tipo'] == 'Histórico']
            fig_t.add_trace(go.Bar(x=hist['año'], y=hist['valor'], name='Precipitación Histórica', marker_color='#87CEEB'))
            fig_t.add_trace(go.Scatter(x=hist['año'], y=hist['recarga'], name='Recarga Histórica', line=dict(color='#00008B', width=3)))
            
            # Datos Pronosticados
            if usar_forecast:
                pred = df_anual[df_anual['tipo'] == 'Pronóstico']
                if not pred.empty:
                    fig_t.add_trace(go.Bar(x=pred['año'], y=pred['valor'], name='Lluvia Proyectada', marker_color='#ADD8E6', opacity=0.5))
                    fig_t.add_trace(go.Scatter(x=pred['año'], y=pred['recarga'], name='Recarga Proyectada', line=dict(color='#00008B', width=3, dash='dot')))

            fig_t.update_layout(title="Balance Hídrico: Historia + Proyección", hovermode="x unified", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_t, use_container_width=True)
            
            # TABLA DE DATOS (SOLICITUD 1)
            with st.expander("📄 Ver Tabla de Datos Detallada", expanded=False):
                st.dataframe(df_anual.style.format("{:.1f}"))
                csv = df_anual.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Descargar Tabla (CSV)",
                    csv,
                    f"balance_anual_{nombre_seleccion}.csv",
                    "text/csv",
                    key='download-csv'
                )

        # === TAB 2: MAPA DISTRIBUIDO ===
        with tab2:
            # SOLICITUD 2: Nombre en el título
            st.markdown(f"##### Modelo Espacial de Recarga: {nombre_seleccion}")
            
            if 'longitude' in df_full.columns and gdf_zona is not None:
                # Mapa basado solo en históricos (para precisión espacial)
                df_spatial = df_full.groupby(['id_estacion', 'nom_est', 'longitude', 'latitude', 'municipio', 'alt_est'])['valor'].mean().reset_index()
                df_spatial['valor_anual'] = df_spatial['valor'] * 12
                
                # Cálculo puntual para Popups
                L_t = 300 + 25*temp_estimada + 0.05*(temp_estimada**3)
                def calc_pt(ppt):
                    with np.errstate(divide='ignore'): etr = ppt / np.sqrt(0.9 + (ppt/L_t)**2)
                    return min(etr, ppt), (ppt - min(etr, ppt)) * coef_final
                
                df_spatial['etr_pt'], df_spatial['rec_pt'] = zip(*df_spatial['valor_anual'].apply(calc_pt))
                
                # Popup Estaciones
                df_spatial['hover_txt'] = df_spatial.apply(
                    lambda r: f"<b>{r['nom_est']}</b><br>🌧️ P: {r['valor_anual']:.0f}<br>💧 R: {r['rec_pt']:.0f}", axis=1
                )

                if len(df_spatial) >= 3:
                    bounds = gdf_zona.total_bounds
                    gx, gy = interpolation.generate_grid_coordinates((bounds[0], bounds[2], bounds[1], bounds[3]), resolution=100j)
                    grid_P = interpolation.interpolate_spatial(df_spatial, 'valor_anual', gx, gy, method='rbf')
                    
                    if grid_P is not None:
                        grid_R = (grid_P - (grid_P / np.sqrt(0.9 + (grid_P/L_t)**2))) * coef_final
                        grid_R = np.nan_to_num(grid_R, nan=0.0)
                        
                        fig_map = go.Figure()
                        
                        # Capa Heatmap
                        fig_map.add_trace(go.Contour(
                            z=grid_R.T, x=gx[:,0], y=gy[0,:],
                            colorscale="Blues", name="Recarga",
                            colorbar=dict(title="mm/año")
                        ))
                        
                        # SOLICITUD 3: Hover en Cuencas
                        for geom in gdf_zona.geometry:
                            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                                if geom.geom_type == 'Polygon': polys = [geom]
                                else: polys = geom.geoms
                                
                                for poly in polys:
                                    x, y = poly.exterior.xy
                                    fig_map.add_trace(go.Scatter(
                                        x=list(x), y=list(y),
                                        mode='lines', line=dict(color='black', width=2),
                                        name='Límite Cuenca',
                                        text=f"Cuenca: {nombre_seleccion}", # <--- AQUÍ EL NOMBRE
                                        hoverinfo='text' # Solo muestra el texto
                                    ))

                        # Capa Estaciones
                        fig_map.add_trace(go.Scatter(
                            x=df_spatial['longitude'], y=df_spatial['latitude'],
                            mode='markers', marker=dict(color='black', size=6),
                            text=df_spatial['hover_txt'], hoverinfo='text',
                            name="Estaciones"
                        ))
                        
                        fig_map.update_layout(height=600, margin=dict(t=20, b=0, l=0, r=0), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
                        st.plotly_chart(fig_map, use_container_width=True)
                        
                        # Descargas
                        c1, c2 = st.columns(2)
                        tiff = get_geotiff_bytes(np.flipud(grid_R.T), from_origin(gx[0,0], gy[0,-1], gx[1,0]-gx[0,0], gy[0,0]-gy[0,1]), "EPSG:4326")
                        c1.download_button("💾 Raster (TIF)", tiff, f"recarga_{nombre_seleccion}.tif")
                        c2.download_button("📄 Estaciones (CSV)", df_spatial.drop(columns=['hover_txt']).to_csv(index=False), f"estaciones_{nombre_seleccion}.csv")
                    else: st.warning("Error interpolando.")
                else: st.warning("Mínimo 3 estaciones requeridas.")
    else: st.warning("Sin datos.")