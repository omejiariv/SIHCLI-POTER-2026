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
from prophet import Prophet
from datetime import datetime

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
    st.subheader("🤖 Pronóstico Hidrológico (ARIMA-Like)")
    usar_forecast = st.checkbox("Activar Proyección", value=True)
    
    meses_extra = 12
    if usar_forecast:
        # Contexto: Estamos en 2026 o queremos proyectar X meses adelante desde HOY
        st.info("El modelo llenará el hueco desde el último dato histórico hasta hoy + el horizonte seleccionado.")
        meses_extra = st.selectbox("Meses adelante (desde hoy):", [6, 12, 18, 24, 60], index=1)

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
        
        # --- MERGE ROBUSTO DE METADATOS ---
        try:
            all_data = data_processor.load_and_process_all_data()
            gdf_stations = all_data[0]
            
            # Aseguramos columnas clave
            cols_meta = ['id_estacion', 'latitude', 'longitude', 'nom_est', 'municipio', 'alt_est']
            # Filtramos solo las que existen en gdf_stations para evitar KeyError
            cols_existentes = [c for c in cols_meta if c in gdf_stations.columns]
            
            # Merge
            df_full = pd.merge(df_precip, gdf_stations[cols_existentes], on='id_estacion', how='left')
        except Exception as e:
            st.warning(f"Advertencia: No se pudieron cruzar metadatos completos. {e}")
            df_full = df_precip

        tab1, tab2 = st.tabs(["📉 Análisis Temporal y Pronóstico", "🗺️ Mapa de Recarga Distribuida"])
        
        # === TAB 1: ANÁLISIS TEMPORAL + PRONÓSTICO ===
        with tab1:
            st.markdown(f"##### Dinámica Histórica y Proyección: {nombre_seleccion}")
            
            # 1. Agrupación mensual regional
            df_ts_monthly = df_full.groupby('fecha')['valor'].mean().reset_index()
            df_final_ts = df_ts_monthly.copy()
            df_final_ts['tipo'] = 'Histórico'

            # 2. MOTOR DE PRONÓSTICO (PROPHET - GAP FILLING)
            if usar_forecast and len(df_ts_monthly) > 12:
                with st.spinner("🧠 Entrenando modelo estacional (Prophet) para completar la serie..."):
                    try:
                        # Preparar datos
                        df_prophet = df_ts_monthly.rename(columns={'fecha': 'ds', 'valor': 'y'})
                        
                        # Modelo: Estacionalidad Multiplicativa (Ideal para lluvias tropicales)
                        m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
                        m.fit(df_prophet)
                        
                        # Calcular horizonte necesario
                        last_hist_date = df_ts_monthly['fecha'].max()
                        target_date = datetime.now() # Fecha real actual
                        
                        # Calcular diferencia de meses entre el último dato y (hoy + horizonte)
                        diff_months = (target_date.year - last_hist_date.year) * 12 + (target_date.month - last_hist_date.month) + meses_extra
                        periods_needed = max(diff_months, 1)
                        
                        # Generar fechas futuras
                        future = m.make_future_dataframe(periods=periods_needed, freq='MS')
                        forecast = m.predict(future)
                        
                        # Filtrar solo la parte nueva
                        df_future = forecast[forecast['ds'] > last_hist_date][['ds', 'yhat']].rename(columns={'ds': 'fecha', 'yhat': 'valor'})
                        df_future['tipo'] = 'Pronóstico'
                        df_future['valor'] = df_future['valor'].clip(lower=0)
                        
                        # Concatenar
                        df_final_ts = pd.concat([df_final_ts, df_future])
                        
                        st.success(f"✅ Serie completada: {last_hist_date.year} ➝ {df_future['fecha'].max().year}")
                    except Exception as e:
                        st.error(f"Error en pronóstico: {e}")

            # 3. Calcular Balance Anual (Turc)
            df_final_ts['año'] = df_final_ts['fecha'].dt.year
            
            # Agrupar por año y tipo
            df_anual = df_final_ts.groupby(['año', 'tipo'])['valor'].sum().reset_index()
            
            # Turc sobre totales anuales
            turc_res = df_anual.apply(
                lambda x: analysis.calculate_water_balance_turc(x['valor'], temp_estimada), axis=1
            )
            df_anual['etr'] = [x[0] for x in turc_res]
            df_anual['excedente'] = [x[1] for x in turc_res]
            df_anual['recarga'] = df_anual['excedente'] * coef_final
            
            # GRÁFICO TEMPORAL
            fig_t = go.Figure()
            
            # Histórico
            hist = df_anual[df_anual['tipo'] == 'Histórico']
            fig_t.add_trace(go.Bar(x=hist['año'], y=hist['valor'], name='Precipitación Histórica', marker_color='#87CEEB'))
            # ETR (RECUPERADO)
            fig_t.add_trace(go.Scatter(x=hist['año'], y=hist['etr'], name='Evapotranspiración (ETR)', line=dict(color='#FFA500', width=2, dash='dot')))
            # Recarga
            fig_t.add_trace(go.Scatter(x=hist['año'], y=hist['recarga'], name='Recarga Histórica', line=dict(color='#00008B', width=3)))
            
            # Pronóstico
            if usar_forecast:
                pred = df_anual[df_anual['tipo'] == 'Pronóstico']
                if not pred.empty:
                    fig_t.add_trace(go.Bar(x=pred['año'], y=pred['valor'], name='Lluvia Proyectada', marker_color='#ADD8E6', opacity=0.5))
                    fig_t.add_trace(go.Scatter(x=pred['año'], y=pred['etr'], name='ETR Proyectada', line=dict(color='#FFA500', width=2, dash='dot'), showlegend=False))
                    fig_t.add_trace(go.Scatter(x=pred['año'], y=pred['recarga'], name='Recarga Proyectada', line=dict(color='#00008B', width=3, dash='dot')))

            fig_t.update_layout(title="Balance Hídrico: Historia + Proyección", hovermode="x unified", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_t, use_container_width=True)
            
            # Tabla Descargable
            with st.expander("📄 Ver Tabla de Datos Detallada", expanded=False):
                format_dict = {'valor': "{:,.1f}", 'etr': "{:,.1f}", 'excedente': "{:,.1f}", 'recarga': "{:,.1f}"}
                st.dataframe(df_anual.style.format(format_dict))
                csv = df_anual.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Descargar CSV", csv, f"balance_{nombre_seleccion}.csv", "text/csv")

        # === TAB 2: MAPA DISTRIBUIDO (MEJORADO) ===
        with tab2:
            st.markdown(f"##### Modelo Espacial: {nombre_seleccion}")
            
            if 'longitude' in df_full.columns and gdf_zona is not None:
                # Agrupación espacial con metadatos completos para Popups
                cols_grp = ['id_estacion', 'nom_est', 'longitude', 'latitude']
                # Agregar opcionales si existen
                if 'municipio' in df_full.columns: cols_grp.append('municipio')
                if 'alt_est' in df_full.columns: cols_grp.append('alt_est')
                
                df_spatial = df_full.groupby(cols_grp)['valor'].mean().reset_index()
                df_spatial['valor_anual'] = df_spatial['valor'] * 12
                
                # Cálculo puntual
                L_t = 300 + 25*temp_estimada + 0.05*(temp_estimada**3)
                def calc_pt(ppt):
                    with np.errstate(divide='ignore'): etr = ppt / np.sqrt(0.9 + (ppt/L_t)**2)
                    return min(etr, ppt), (ppt - min(etr, ppt)) * coef_final
                
                df_spatial['etr_pt'], df_spatial['rec_pt'] = zip(*df_spatial['valor_anual'].apply(calc_pt))
                
                # POPUP MEJORADO (SOLICITUD 4)
                def build_popup(row):
                    muni = row['municipio'] if 'municipio' in row else 'N/D'
                    alt = f"{row['alt_est']:.0f}" if 'alt_est' in row and pd.notnull(row['alt_est']) else "N/D"
                    return (
                        f"<b>{row['nom_est']}</b><br>"
                        f"🏙️ {muni} | ⛰️ {alt} msnm<br>"
                        f"🌧️ P: {row['valor_anual']:.0f}<br>"
                        f"☀️ ETR: {row['etr_pt']:.0f}<br>"
                        f"💧 <b>R: {row['rec_pt']:.0f}</b>"
                    )
                df_spatial['hover_txt'] = df_spatial.apply(build_popup, axis=1)

                if len(df_spatial) >= 3:
                    bounds = gdf_zona.total_bounds
                    gx, gy = interpolation.generate_grid_coordinates((bounds[0], bounds[2], bounds[1], bounds[3]), resolution=100j)
                    grid_P = interpolation.interpolate_spatial(df_spatial, 'valor_anual', gx, gy, method='rbf')
                    
                    if grid_P is not None:
                        grid_R = (grid_P - (grid_P / np.sqrt(0.9 + (grid_P/L_t)**2))) * coef_final
                        grid_R = np.nan_to_num(grid_R, nan=0.0)
                        
                        fig_map = go.Figure()
                        
                        # Capa Heatmap (Leyenda abajo)
                        fig_map.add_trace(go.Contour(
                            z=grid_R.T, x=gx[:,0], y=gy[0,:],
                            colorscale="Blues", name="Recarga (mm)",
                            colorbar=dict(title="mm/año", len=0.5, y=-0.2, orientation='h'), # LEYENDA ABAJO
                            showscale=True
                        ))
                        
                        # CAPA SUBCUENCAS CON NOMBRES (SOLICITUD 5 y 6)
                        # Buscamos la columna del nombre
                        candidates = ['nombre', 'subcuenca', 'name', 'microcuenca', 'cuenca', 'nom_subcuenca']
                        col_name_sub = next((c for c in gdf_zona.columns if any(x in c.lower() for x in candidates)), None)
                        
                        # Iteramos sobre filas para tener nombre + geometría
                        for idx, row in gdf_zona.iterrows():
                            geom = row.geometry
                            # Nombre específico de este polígono
                            name_sub = row[col_name_sub] if col_name_sub else "Subcuenca"
                            
                            if geom.geom_type == 'Polygon': polys = [geom]
                            elif geom.geom_type == 'MultiPolygon': polys = geom.geoms
                            else: polys = []
                            
                            for poly in polys:
                                x, y = poly.exterior.xy
                                fig_map.add_trace(go.Scatter(
                                    x=list(x), y=list(y),
                                    mode='lines', 
                                    line=dict(color='black', width=1.5),
                                    name=str(name_sub), # Nombre real en la leyenda
                                    text=f"Zona: {name_sub}",
                                    hoverinfo='text',
                                    showlegend=False # Evitar saturar leyenda con 100 líneas
                                ))

                        # Capa Estaciones
                        fig_map.add_trace(go.Scatter(
                            x=df_spatial['longitude'], y=df_spatial['latitude'],
                            mode='markers', marker=dict(color='red', size=6, line=dict(color='white', width=1)),
                            text=df_spatial['hover_txt'], hoverinfo='text',
                            name="Estaciones"
                        ))
                        
                        fig_map.update_layout(
                            height=650, 
                            margin=dict(t=10, b=50, l=0, r=0),
                            xaxis=dict(visible=False), 
                            yaxis=dict(visible=False, scaleanchor="x"),
                            hoverlabel=dict(bgcolor="white", font_size=12)
                        )
                        st.plotly_chart(fig_map, use_container_width=True)
                        
                        # Descargas
                        c1, c2 = st.columns(2)
                        tiff = get_geotiff_bytes(np.flipud(grid_R.T), from_origin(gx[0,0], gy[0,-1], gx[1,0]-gx[0,0], gy[0,0]-gy[0,1]), "EPSG:4326")
                        c1.download_button("💾 Raster (TIF)", tiff, f"recarga_{nombre_seleccion}.tif")
                        c2.download_button("📄 Estaciones (CSV)", df_spatial.drop(columns=['hover_txt']).to_csv(index=False), f"estaciones_{nombre_seleccion}.csv")
                    else: st.warning("Error interpolando.")
                else: st.warning("Mínimo 3 estaciones requeridas.")
    else: st.warning("Sin datos.")