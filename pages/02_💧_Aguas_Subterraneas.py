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

# --- CAJA INFORMATIVA METODOLÓGICA (SOLICITUD 2) ---
with st.expander("ℹ️ Metodología, Conceptos e Interpretación", expanded=False):
    st.markdown("""
    ### 🧠 Metodología del Modelo
    Este módulo estima la recarga potencial de acuíferos utilizando un enfoque híbrido físico-estadístico:

    1.  **Balance Hídrico (Método de Turc, 1954):**
        * Calcula la **Evapotranspiración Real (ETR)** anual en función de la Precipitación (P) y la Temperatura media (T).
        * Fórmula: $ETR = \frac{P}{\sqrt{0.9 + (P/L(T))^2}}$, donde $L(T)$ es la capacidad evaporante del aire.
        * El **Excedente Hídrico** es el agua que sobra: $Q = P - ETR$.
    
    2.  **Estimación de Recarga:**
        * Se asume que una fracción del Excedente Hídrico se infiltra al subsuelo.
        * $Recarga = Excedente \times C_{infiltración}$
        * El $C_{infiltración}$ se sugiere automáticamente mediante IA analizando la cobertura del suelo (Bosque > Pastos > Urbano) o se ajusta manualmente.

    3.  **Pronóstico Climático Multivariado (IA - Prophet):**
        * Utiliza el modelo **Prophet** (desarrollado por Meta) para proyectar la precipitación futura.
        * **Modelo Avanzado:** No es solo una proyección de tendencia. El modelo aprende de la estacionalidad histórica y se ve influenciado por índices climáticos globales (**ONI/Niño, SOI, IOD**) si están disponibles, mejorando la precisión en escenarios de variabilidad climática.
        * **Control de Calidad:** Años con registros incompletos (<50% del promedio histórico) se excluyen del entrenamiento.

    4.  **Interpolación Espacial (Análisis Distribuido):**
        * Para el mapa, se utiliza **Kriging/RBF (Radial Basis Functions)**, una técnica geoestadística avanzada que genera una superficie continua de lluvia y recarga a partir de los puntos de las estaciones, considerando la topografía y la influencia de estaciones vecinas (buffer).

    ### 🔎 Interpretación
    * **Recarga Potencial:** Es una estimación del agua que *podría* llegar al acuífero. La recarga real depende de la geología profunda no considerada aquí.
    * **Uso:** Herramienta de planificación para identificar zonas de alta importancia hidrogeológica y evaluar el impacto de escenarios climáticos futuros en la disponibilidad de agua subterránea.
    
    **Fuentes:** IDEAM (Datos hidrometeorológicos), NOAA (Índices Climáticos), SIHCLI-POTER (Procesamiento y Coberturas).
    """)

st.title("💧 Estimación de Recarga (Modelo Turc + Pronóstico Multivariado)")

# --- 1. CONFIGURACIÓN ---
ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

with st.sidebar:
    st.divider()
    st.subheader("🤖 Pronóstico Hidrológico IA")
    usar_forecast = st.checkbox("Activar Proyección Multivariada", value=True, help="Usa índices climáticos (Niño/Niña) para mejorar la proyección.")
    
    meses_futuros = 12
    if usar_forecast:
        st.info("El modelo llenará el hueco desde el último dato válido hasta hoy + el horizonte futuro.")
        meses_futuros = st.selectbox("Meses adelante (desde HOY):", [6, 12, 18, 24, 60], index=1)

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
        
        # --- CARGA DE DATOS CLIMÁTICOS (SOLICITUD 3) ---
        try:
            all_data = data_processor.load_and_process_all_data()
            gdf_stations = all_data[0]
            # Índice 3 tiene los datos climáticos (ONI, SOI, IOD) según diagnóstico previo
            df_climatico = all_data[3]
            if not df_climatico.empty:
                df_climatico['fecha_mes_año'] = pd.to_datetime(df_climatico['fecha_mes_año'])
            
            # Merge Metadatos Estaciones
            cols_meta = ['id_estacion', 'latitude', 'longitude', 'nom_est', 'municipio', 'alt_est']
            cols_existentes = [c for c in cols_meta if c in gdf_stations.columns]
            df_full = pd.merge(df_precip, gdf_stations[cols_existentes], on='id_estacion', how='left')
        except Exception as e:
            st.warning(f"Advertencia: No se pudieron cargar metadatos o índices climáticos completos. {e}")
            df_full = df_precip
            df_climatico = pd.DataFrame()

        tab1, tab2 = st.tabs(["📉 Análisis Temporal y Pronóstico", "🗺️ Mapa de Recarga Distribuida"])
        
        # === TAB 1: ANÁLISIS TEMPORAL + PRONÓSTICO ===
        with tab1:
            st.markdown(f"##### Dinámica Histórica y Proyección: {nombre_seleccion}")
            
            # 1. Agrupación mensual
            df_ts_monthly = df_full.groupby('fecha')['valor'].mean().reset_index()
            
            # --- MERGE CON ÍNDICES CLIMÁTICOS ---
            if not df_climatico.empty:
                df_ts_monthly = pd.merge(df_ts_monthly, df_climatico, left_on='fecha', right_on='fecha_mes_año', how='left')
                # Llenar nulos históricos con 0 (condición neutral) por si acaso
                cols_clima = ['anomalia_oni', 'soi', 'iod']
                cols_clima_presentes = [c for c in cols_clima if c in df_ts_monthly.columns]
                if cols_clima_presentes:
                    df_ts_monthly[cols_clima_presentes] = df_ts_monthly[cols_clima_presentes].fillna(0)
            
            # --- FILTRO DE CALIDAD DE DATOS (SOLICITUD 1) ---
            df_ts_monthly['año_temp'] = df_ts_monthly['fecha'].dt.year
            annual_stats = df_ts_monthly.groupby('año_temp')['valor'].sum()
            long_term_avg = annual_stats.mean()
            threshold = long_term_avg * 0.5
            years_to_drop = annual_stats[annual_stats < threshold].index.tolist()
            
            df_train = df_ts_monthly.copy()
            if years_to_drop:
                st.warning(f"⚠️ Control de Calidad: Se excluyeron los años {years_to_drop} del entrenamiento por tener registros incompletos (<50% del promedio histórico).")
                df_train = df_ts_monthly[~df_ts_monthly['año_temp'].isin(years_to_drop)]
            
            df_final_ts = df_ts_monthly.drop(columns=['año_temp', 'fecha_mes_año'], errors='ignore').copy()
            df_final_ts['tipo'] = 'Histórico'

            # 2. MOTOR DE PRONÓSTICO MULTIVARIADO (PROPHET)
            if usar_forecast and len(df_train) > 24:
                with st.spinner("🧠 Entrenando modelo IA multivariado (Precipitación + Índices Climáticos)..."):
                    try:
                        last_hist_date = df_train['fecha'].max()
                        
                        # Preparar datos entrenamiento
                        df_prophet = df_train.rename(columns={'fecha': 'ds', 'valor': 'y'})
                        
                        # Modelo con Regresores Externos
                        m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
                        
                        # Agregar índices como regresores si existen
                        cols_clima_usadas = []
                        if 'anomalia_oni' in df_prophet.columns:
                            m.add_regressor('anomalia_oni')
                            cols_clima_usadas.append('anomalia_oni')
                        if 'soi' in df_prophet.columns:
                            m.add_regressor('soi')
                            cols_clima_usadas.append('soi')
                        if 'iod' in df_prophet.columns:
                            m.add_regressor('iod')
                            cols_clima_usadas.append('iod')
                            
                        m.fit(df_prophet)
                        
                        # Horizonte de tiempo
                        fecha_objetivo = datetime.now() + relativedelta(months=meses_futuros)
                        future = m.make_future_dataframe(periods=300, freq='MS')
                        future = future[future['ds'] <= fecha_objetivo]
                        
                        # --- PREPARAR REGRESORES FUTUROS ---
                        # Asumimos persistencia: los índices mantienen el último valor conocido
                        if cols_clima_usadas:
                            last_indices = df_ts_monthly.sort_values('fecha').iloc[-1][cols_clima_usadas]
                            for col in cols_clima_usadas:
                                future[col] = last_indices[col]
                        
                        # Predicción
                        forecast = m.predict(future)
                        
                        # Filtrar futuro
                        df_future = forecast[forecast['ds'] > last_hist_date][['ds', 'yhat']].rename(columns={'ds': 'fecha', 'yhat': 'valor'})
                        df_future['tipo'] = 'Pronóstico'
                        df_future['valor'] = df_future['valor'].clip(lower=0)
                        
                        # Rellenar columnas de clima en el futuro (para consistencia al concatenar)
                        for col in cols_clima_usadas:
                            df_future[col] = last_indices[col]

                        df_final_ts = pd.concat([df_final_ts, df_future], ignore_index=True)
                        
                        msg_clima = f" con soporte de índices climáticos ({', '.join(cols_clima_usadas)})" if cols_clima_usadas else ""
                        st.success(f"✅ Proyección generada{msg_clima} hasta {fecha_objetivo.date()}.")
                        
                    except Exception as e:
                        st.error(f"Error en pronóstico: {e}")
            elif usar_forecast:
                st.warning("No hay suficientes datos históricos válidos (>24 meses) para entrenar el modelo.")

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
            fig_t.add_trace(go.Scatter(x=hist['año'], y=hist['etr'], name='ETR Histórica', line=dict(color='#FFA500', width=2, dash='dot')))
            fig_t.add_trace(go.Scatter(x=hist['año'], y=hist['recarga'], name='Recarga Histórica', line=dict(color='#00008B', width=3)))
            
            # Pronóstico
            if usar_forecast:
                pred = df_anual[df_anual['tipo'] == 'Pronóstico']
                if not pred.empty:
                    fig_t.add_trace(go.Bar(x=pred['año'], y=pred['valor'], name='Lluvia Proyectada', marker_color='#ADD8E6', opacity=0.5))
                    fig_t.add_trace(go.Scatter(x=pred['año'], y=pred['etr'], name='ETR Proyectada', line=dict(color='#FFD700', width=2, dash='dot'), showlegend=False))
                    fig_t.add_trace(go.Scatter(x=pred['año'], y=pred['recarga'], name='Recarga Proyectada', line=dict(color='#00008B', width=3, dash='dot')))

            fig_t.update_layout(title="Balance Hídrico: Historia + Reconstrucción + Proyección", hovermode="x unified", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_t, use_container_width=True)
            
            # Tabla Descargable
            with st.expander("📄 Ver Tabla de Datos Detallada", expanded=False):
                format_dict = {'valor': "{:,.1f}", 'etr': "{:,.1f}", 'excedente': "{:,.1f}", 'recarga': "{:,.1f}"}
                st.dataframe(df_anual.style.format(format_dict))
                csv = df_anual.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Descargar CSV", csv, f"balance_{nombre_seleccion}.csv", "text/csv")
            
        # === TAB 2: MAPA DISTRIBUIDO (Igual que versión anterior) ===
        with tab2:
            st.markdown(f"##### Modelo Espacial: {nombre_seleccion}")
            
            if 'longitude' in df_full.columns and gdf_zona is not None:
                # Agrupación espacial
                cols_grp = ['id_estacion', 'nom_est', 'longitude', 'latitude']
                if 'municipio' in df_full.columns: cols_grp.append('municipio')
                if 'alt_est' in df_full.columns: cols_grp.append('alt_est')
                
                # Usamos solo datos históricos para el mapa base
                df_spatial = df_full.groupby(cols_grp)['valor'].mean().reset_index()
                df_spatial['valor_anual'] = df_spatial['valor'] * 12
                
                # Cálculo puntual
                L_t = 300 + 25*temp_estimada + 0.05*(temp_estimada**3)
                def calc_pt(ppt):
                    with np.errstate(divide='ignore'): etr = ppt / np.sqrt(0.9 + (ppt/L_t)**2)
                    return min(etr, ppt), (ppt - min(etr, ppt)) * coef_final
                
                df_spatial['etr_pt'], df_spatial['rec_pt'] = zip(*df_spatial['valor_anual'].apply(calc_pt))
                
                # Popup
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
                        
                        # Heatmap (Leyenda corregida)
                        fig_map.add_trace(go.Contour(
                            z=grid_R.T, x=gx[:,0], y=gy[0,:],
                            colorscale="Blues", name="Recarga (mm)",
                            colorbar=dict(title="mm/año", len=0.6, y=-0.2, orientation='h'),
                            showscale=True
                        ))
                        
                        # Subcuencas
                        candidates = ['nombre', 'subcuenca', 'name', 'microcuenca', 'cuenca', 'nom_subcuenca']
                        col_name_sub = next((c for c in gdf_zona.columns if any(x in c.lower() for x in candidates)), None)
                        
                        for idx, row in gdf_zona.iterrows():
                            geom = row.geometry
                            name_sub = row[col_name_sub] if col_name_sub else "Subcuenca"
                            
                            if geom.geom_type == 'Polygon': polys = [geom]
                            elif geom.geom_type == 'MultiPolygon': polys = geom.geoms
                            else: polys = []
                            
                            for poly in polys:
                                x, y = poly.exterior.xy
                                fig_map.add_trace(go.Scatter(
                                    x=list(x), y=list(y),
                                    mode='lines', 
                                    line=dict(color='black', width=1),
                                    name=str(name_sub),
                                    text=f"Zona: {name_sub}",
                                    hoverinfo='text',
                                    showlegend=False
                                ))

                        # Estaciones
                        fig_map.add_trace(go.Scatter(
                            x=df_spatial['longitude'], y=df_spatial['latitude'],
                            mode='markers', marker=dict(color='red', size=6, line=dict(color='white', width=1)),
                            text=df_spatial['hover_txt'], hoverinfo='text',
                            name="Estaciones"
                        ))
                        
                        fig_map.update_layout(
                            height=650, 
                            margin=dict(t=10, b=80, l=0, r=0),
                            xaxis=dict(visible=False), 
                            yaxis=dict(visible=False, scaleanchor="x"),
                            hoverlabel=dict(bgcolor="white", font_size=12)
                        )
                        st.plotly_chart(fig_map, use_container_width=True)
                        
                        c1, c2 = st.columns(2)
                        tiff = get_geotiff_bytes(np.flipud(grid_R.T), from_origin(gx[0,0], gy[0,-1], gx[1,0]-gx[0,0], gy[0,0]-gy[0,1]), "EPSG:4326")
                        c1.download_button("💾 Raster (TIF)", tiff, f"recarga_{nombre_seleccion}.tif")
                        c2.download_button("📄 Estaciones (CSV)", df_spatial.drop(columns=['hover_txt']).to_csv(index=False), f"estaciones_{nombre_seleccion}.csv")
                    else: st.warning("Error interpolando.")
                else: st.warning("Mínimo 3 estaciones requeridas.")
    else: st.warning("Sin datos.")