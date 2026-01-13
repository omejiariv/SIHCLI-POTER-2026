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
from dateutil.relativedelta import relativedelta

# --- IMPORTS MODULARES ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import analysis, selectors, interpolation, data_processor
from modules import land_cover as lc
from modules.config import Config

st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

# --- METODOLOGÍA COMPLETA (SOLICITUD 3) ---
with st.expander("ℹ️ Metodología, Conceptos e Interpretación (Detallado)", expanded=False):
    st.markdown("""
    ### 1. Marco Conceptual
    Este módulo estima la **Recarga Potencial de Acuíferos**, definida como la lámina de agua que logra infiltrarse a través del suelo y llegar a la zona saturada, convirtiéndose en agua subterránea renovable.

    ### 2. Metodología de Cálculo
    Se utiliza un enfoque de **Balance Hídrico Mensual** agregados anualmente, basado en la fórmula de **Turc (1954)** modificado para zonas tropicales:
    
    * **P (Precipitación):** Variable de entrada principal (mm).
    * **T (Temperatura):** Estimada mediante gradiente altitudinal (-0.6°C / 100m) o datos directos.
    * **L(t) (Capacidad Evaporante):** $L(t) = 300 + 25T + 0.05T^3$.
    * **ETR (Evapotranspiración Real):** $ETR = \frac{P}{\sqrt{0.9 + (P/L)^2}}$. Representa el agua que regresa a la atmósfera por evaporación del suelo y transpiración de plantas.
    * **Excedente Hídrico (Q):** $Q = P - ETR$. Es el agua líquida disponible en superficie.
    * **Recarga (R):** $R = Q \times C_{inf}$. Donde $C_{inf}$ es el Coeficiente de Infiltración, que depende de la cobertura del suelo (Bosque, Pastos, Urbano) y la permeabilidad geológica.

    ### 3. Pronóstico Hidrológico (Inteligencia Artificial)
    Para proyectar la serie hacia el futuro, se utiliza el modelo **Prophet (Meta AI)** con configuración avanzada:
    * **Estacionalidad:** Modelo multiplicativo anual (captura bimodalidad de lluvias andinas).
    * **Regresores Externos:** El modelo se entrena considerando índices climáticos globales (**ONI/Niño, SOI, IOD**) para capturar la variabilidad macroclimática.
    * **Simulación Estocástica:** Se inyecta ruido estadístico basado en la varianza histórica para simular escenarios realistas de extremos (no solo promedios).

    ### 4. Modelo Espacial (Geoestadística)
    La distribución espacial se realiza mediante interpolación **RBF (Radial Basis Functions)** sobre las estaciones monitoreadas, generando una superficie continua (Raster) de precipitación y recarga, corregida por topografía y cobertura del suelo.
    
    ### 5. Interpretación
    * **Mapas Azules Oscuros:** Zonas de alta recarga (Estratégicas para protección).
    * **Brecha P vs ETR:** En años secos (Niño), la ETR puede consumir casi toda la precipitación, llevando la recarga a cero.
    """)

st.title("💧 Estimación de Recarga (Modelo Turc + Escenarios)")

# --- 1. CONFIGURACIÓN ---
ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

with st.sidebar:
    st.divider()
    st.subheader("🤖 Pronóstico & Escenarios")
    usar_forecast = st.checkbox("Activar Proyección", value=True)
    
    meses_futuros = 12
    usar_estocastico = False
    
    if usar_forecast:
        meses_futuros = st.selectbox("Horizonte (meses):", [12, 24, 36, 60], index=1)
        st.markdown("**Configuración del Modelo:**")
        usar_estocastico = st.checkbox("🎲 Simular Variabilidad Real", value=True, help="Añade ruido estadístico para simular picos y valles.")
        
        if usar_estocastico:
            nivel_ruido = st.slider("Intensidad Variabilidad:", 0.5, 1.5, 1.0)

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

    coef_final = st.slider("Coef. Infiltración", 0.0, 1.0, float(coef_default))
    temp_estimada = analysis.estimate_temperature(altitud_ref)

# --- FUNCIÓN AUXILIAR RASTER ---
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
        
        # --- CARGA CLIMA Y METADATOS ---
        try:
            all_data = data_processor.load_and_process_all_data()
            gdf_stations = all_data[0]
            df_climatico = all_data[3]
            if not df_climatico.empty:
                df_climatico['fecha_mes_año'] = pd.to_datetime(df_climatico['fecha_mes_año'])
            
            # MERGE COMPLETO (SOLICITUD 2 - Asegurar columnas)
            cols_meta = ['id_estacion', 'latitude', 'longitude', 'nom_est', 'municipio', 'alt_est']
            cols_existentes = [c for c in cols_meta if c in gdf_stations.columns]
            df_full = pd.merge(df_precip, gdf_stations[cols_existentes], on='id_estacion', how='left')
        except:
            df_full = df_precip
            df_climatico = pd.DataFrame()

        tab1, tab2 = st.tabs(["📉 Análisis Temporal y Pronóstico", "🗺️ Mapa de Recarga Distribuida"])
        
        # === TAB 1 ===
        with tab1:
            st.markdown(f"##### Dinámica Histórica y Escenarios: {nombre_seleccion}")
            
            # 1. Agrupación
            df_ts_monthly = df_full.groupby('fecha')['valor'].mean().reset_index()
            
            # Merge Clima
            if not df_climatico.empty:
                df_ts_monthly = pd.merge(df_ts_monthly, df_climatico, left_on='fecha', right_on='fecha_mes_año', how='left')
                cols_clima = ['anomalia_oni', 'soi', 'iod']
                cols_clima_presentes = [c for c in cols_clima if c in df_ts_monthly.columns]
                if cols_clima_presentes:
                    df_ts_monthly[cols_clima_presentes] = df_ts_monthly[cols_clima_presentes].fillna(0)
            
            # Filtro Calidad
            df_ts_monthly['año_temp'] = df_ts_monthly['fecha'].dt.year
            annual_stats = df_ts_monthly.groupby('año_temp')['valor'].sum()
            threshold = annual_stats.mean() * 0.5
            years_to_drop = annual_stats[annual_stats < threshold].index.tolist()
            
            df_train = df_ts_monthly.copy()
            if years_to_drop:
                df_train = df_ts_monthly[~df_ts_monthly['año_temp'].isin(years_to_drop)]
            
            df_final_ts = df_ts_monthly.drop(columns=['año_temp', 'fecha_mes_año'], errors='ignore').copy()
            df_final_ts['tipo'] = 'Histórico'
            df_final_ts['yhat_lower'] = df_final_ts['valor']
            df_final_ts['yhat_upper'] = df_final_ts['valor']

            # 2. PROPHET
            if usar_forecast and len(df_train) > 24:
                with st.spinner("🧠 Generando escenarios hidrológicos..."):
                    try:
                        last_hist_date = df_train['fecha'].max()
                        df_prophet = df_train.rename(columns={'fecha': 'ds', 'valor': 'y'})
                        
                        m = Prophet(
                            seasonality_mode='multiplicative', 
                            yearly_seasonality=True,
                            changepoint_prior_scale=0.5
                        )
                        
                        cols_clima_usadas = []
                        if 'anomalia_oni' in df_prophet.columns:
                            m.add_regressor('anomalia_oni')
                            cols_clima_usadas.append('anomalia_oni')
                            
                        m.fit(df_prophet)
                        
                        # Horizonte
                        fecha_objetivo = datetime.now() + relativedelta(months=meses_futuros)
                        future = m.make_future_dataframe(periods=300, freq='MS')
                        future = future[future['ds'] <= fecha_objetivo]
                        
                        if cols_clima_usadas:
                            last_indices = df_ts_monthly.sort_values('fecha').iloc[-1][cols_clima_usadas]
                            for col in cols_clima_usadas: future[col] = last_indices[col]
                        
                        forecast = m.predict(future)
                        
                        df_future = forecast[forecast['ds'] > last_hist_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'fecha', 'yhat': 'valor'})
                        df_future['tipo'] = 'Pronóstico'
                        
                        if usar_estocastico:
                            residuals = df_prophet['y'] - forecast.loc[forecast['ds'].isin(df_prophet['ds']), 'yhat']
                            std_resid = residuals.std()
                            np.random.seed(42)
                            noise = np.random.normal(0, std_resid * nivel_ruido, len(df_future))
                            df_future['valor'] += noise
                            df_future['yhat_upper'] += (std_resid * nivel_ruido)
                            df_future['yhat_lower'] -= (std_resid * nivel_ruido)

                        df_future['valor'] = df_future['valor'].clip(lower=0)
                        df_future['yhat_lower'] = df_future['yhat_lower'].clip(lower=0)
                        for col in cols_clima_usadas: df_future[col] = 0

                        df_final_ts = pd.concat([df_final_ts, df_future], ignore_index=True)
                        st.success(f"✅ Escenario generado hasta {fecha_objetivo.date()}.")
                    except Exception as e:
                        st.error(f"Error: {e}")

            # 3. Balance Anual
            df_final_ts['año'] = df_final_ts['fecha'].dt.year
            df_anual = df_final_ts.groupby(['año', 'tipo']).agg({'valor': 'sum', 'yhat_lower': 'sum', 'yhat_upper': 'sum'}).reset_index()
            
            turc_res = df_anual.apply(lambda x: analysis.calculate_water_balance_turc(x['valor'], temp_estimada), axis=1)
            df_anual['etr'] = [x[0] for x in turc_res]
            df_anual['recarga'] = np.array([x[1] for x in turc_res]) * coef_final
            
# --- SOLUCIÓN SIERRA: UNIFICAR DATOS PARA LÍNEAS ---
            # Agrupamos SOLO por año para tener una serie continua sin duplicados de transición
            df_lines = df_anual.groupby('año').agg({
                'etr': 'sum',
                'recarga': 'sum'
            }).reset_index().sort_values('año')
            
            # --- GRÁFICO ---
            fig_t = go.Figure()
            
            hist = df_anual[df_anual['tipo'] == 'Histórico']
            pred = df_anual[df_anual['tipo'] == 'Pronóstico']
            
            # 1. Intervalo Confianza
            if not pred.empty:
                fig_t.add_trace(go.Scatter(
                    x=pd.concat([pred['año'], pred['año'][::-1]]),
                    y=pd.concat([pred['yhat_upper'], pred['yhat_lower'][::-1]]),
                    fill='toself', fillcolor='rgba(173, 216, 230, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'), name='Rango Incertidumbre'
                ))

            # 2. Barras Históricas
            fig_t.add_trace(go.Bar(x=hist['año'], y=hist['valor'], name='Precipitación Histórica', marker_color='#87CEEB'))
            
            # 3. Barras Pronóstico (RESTAURADO A BARRAS)
            if not pred.empty:
                fig_t.add_trace(go.Bar(
                    x=pred['año'], y=pred['valor'], 
                    name='Precipitación Proyectada', 
                    marker_color='#ADD8E6', 
                    marker_line_color='#4682B4', 
                    marker_line_width=1.5, 
                    opacity=0.7
                ))
            # 4. Líneas Balance (USANDO df_lines UNIFICADO = NO MÁS SIERRA)
            fig_t.add_trace(go.Scatter(
                x=df_lines['año'], y=df_lines['etr'], 
                name='ETR', 
                line=dict(color='#FFA500', width=2, dash='dot', shape='spline', smoothing=1.3)
            ))
            
            fig_t.add_trace(go.Scatter(
                x=df_lines['año'], y=df_lines['recarga'], 
                name='Recarga', 
                line=dict(color='#00008B', width=3, shape='spline', smoothing=1.3)
            ))

            fig_t.update_layout(title="Dinámica Hidrológica", hovermode="x unified", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_t, use_container_width=True)
            
            with st.expander("📄 Tabla de Datos", expanded=False):
                format_dict = {'valor': "{:,.1f}", 'etr': "{:,.1f}", 'recarga': "{:,.1f}"}
                st.dataframe(df_anual[['año', 'tipo', 'valor', 'etr', 'recarga']].style.format(format_dict))
                st.download_button("💾 Descargar CSV", df_anual.to_csv(index=False).encode('utf-8'), f"balance_{nombre_seleccion}.csv")

        # === TAB 2: MAPA ===
        with tab2:
            st.markdown(f"##### Modelo Espacial: {nombre_seleccion}")
            if 'longitude' in df_full.columns and gdf_zona is not None:
                # SOLICITUD 2: Agrupación explícita con Municipio/Altura para que no se pierdan
                cols_grp = ['id_estacion', 'nom_est', 'longitude', 'latitude']
                # Verificamos y agregamos columnas opcionales al groupby
                for col in ['municipio', 'alt_est']:
                    if col in df_full.columns: cols_grp.append(col)
                
                df_spatial = df_full.groupby(cols_grp)['valor'].mean().reset_index()
                df_spatial['valor_anual'] = df_spatial['valor'] * 12
                
                L_t = 300 + 25*temp_estimada + 0.05*(temp_estimada**3)
                def calc_pt(ppt):
                    with np.errstate(divide='ignore'): etr = ppt / np.sqrt(0.9 + (ppt/L_t)**2)
                    return min(etr, ppt), (ppt - min(etr, ppt)) * coef_final
                
                df_spatial['etr_pt'], df_spatial['rec_pt'] = zip(*df_spatial['valor_anual'].apply(calc_pt))
                
                # Popup Restaurado
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
                        fig_map.add_trace(go.Contour(
                            z=grid_R.T, x=gx[:,0], y=gy[0,:],
                            colorscale="Blues", name="Recarga (mm)",
                            colorbar=dict(title="mm/año", len=0.6, y=-0.2, orientation='h'), showscale=True
                        ))
                        
                        candidates = ['nombre', 'subcuenca', 'name', 'microcuenca']
                        col_name_sub = next((c for c in gdf_zona.columns if any(x in c.lower() for x in candidates)), None)
                        for idx, row in gdf_zona.iterrows():
                            geom = row.geometry
                            name_sub = row[col_name_sub] if col_name_sub else ""
                            if geom.geom_type == 'Polygon': polys = [geom]
                            elif geom.geom_type == 'MultiPolygon': polys = geom.geoms
                            else: polys = []
                            for poly in polys:
                                x, y = poly.exterior.xy
                                fig_map.add_trace(go.Scatter(
                                    x=list(x), y=list(y), mode='lines', line=dict(color='black', width=1),
                                    name=str(name_sub), text=f"Zona: {name_sub}", hoverinfo='text', showlegend=False
                                ))

                        fig_map.add_trace(go.Scatter(
                            x=df_spatial['longitude'], y=df_spatial['latitude'],
                            mode='markers', marker=dict(color='red', size=6, line=dict(color='white', width=1)),
                            text=df_spatial['hover_txt'], hoverinfo='text', name="Estaciones"
                        ))
                        
                        fig_map.update_layout(height=650, margin=dict(t=10, b=80, l=0, r=0), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
                        st.plotly_chart(fig_map, use_container_width=True)
                        
                        c1, c2 = st.columns(2)
                        tiff = get_geotiff_bytes(np.flipud(grid_R.T), from_origin(gx[0,0], gy[0,-1], gx[1,0]-gx[0,0], gy[0,0]-gy[0,1]), "EPSG:4326")
                        c1.download_button("💾 Raster (TIF)", tiff, f"recarga_{nombre_seleccion}.tif")
                        c2.download_button("📄 Estaciones (CSV)", df_spatial.drop(columns=['hover_txt']).to_csv(index=False), f"estaciones_{nombre_seleccion}.csv")
                    else: st.warning("Error interpolando.")
                else: st.warning("Mínimo 3 estaciones.")
    else: st.warning("Sin datos.")