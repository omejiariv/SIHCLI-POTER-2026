import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import sys
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Aguas Subterráneas", page_icon="💧", layout="wide")

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors 
except ImportError:
    st.error("Error al importar módulos base.")
    st.stop()

st.title("💧 Modelo Hidrogeológico: Recarga y Escenarios")

# --- 1. BARRA LATERAL: PARÁMETROS DEL MODELO ---
# A. Selector Geográfico Global
ids_dummy, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

st.sidebar.divider()
st.sidebar.header("🌱 Parámetros del Suelo")
st.sidebar.info("Ajuste los coeficientes según la cobertura del suelo en la zona para refinar la tasa de infiltración.")

# B. Simulación de Cobertura (Recuperado)
# Coeficientes típicos (K)
k_bosque = 0.50
k_cultivo = 0.30
k_urbano = 0.10

col_s1, col_s2, col_s3 = st.sidebar.columns(3)
pct_bosque = col_s1.number_input("% Bosque", 0, 100, 60)
pct_cultivo = col_s2.number_input("% Agrícola", 0, 100, 30)
# El resto es urbano/impermeable automáticamente
pct_urbano = max(0, 100 - (pct_bosque + pct_cultivo))
col_s3.metric("% Urbano/Otros", f"{pct_urbano}%")

# Cálculo del Coeficiente de Infiltración Ponderado (Ki)
# Ki = ( (%B * Kb) + (%C * Kc) + (%U * Ku) ) / 100
ki_ponderado = ((pct_bosque * k_bosque) + (pct_cultivo * k_cultivo) + (pct_urbano * k_urbano)) / 100.0
st.sidebar.metric("Coef. Infiltración ($K_i$)", f"{ki_ponderado:.2f}")

st.sidebar.divider()
st.sidebar.header("⚙️ Proyección Climática")
activar_proyeccion = st.sidebar.checkbox("Activar Proyección Futura", value=True)
horizonte = st.sidebar.selectbox("Horizonte:", [2026, 2027, 2030], index=0)
ruido = st.sidebar.slider("Incertidumbre Climática (+/-)", 0.0, 2.0, 0.5)

# --- LÓGICA MATEMÁTICA (TURC + INFILTRACIÓN) ---
def calculate_turc_advanced(df, ki):
    """Calcula Turc y aplica factor de infiltración."""
    df = df.copy()
    # Limpieza numérica
    df['alt_est'] = pd.to_numeric(df['alt_est'], errors='coerce')
    df['p_anual'] = pd.to_numeric(df['p_anual'], errors='coerce')
    
    # 1. Temperatura estimada
    df['temp_est'] = 30 - (0.0065 * df['alt_est'])
    
    # 2. Capacidad evaporativa del aire L(t)
    t = df['temp_est']
    l_t = 300 + 25*t + 0.05*(t**3)
    
    # 3. Evapotranspiración Real (ETR) - Fórmula Turc
    p = df['p_anual']
    # Evitar división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        df['etr_mm'] = p / np.sqrt(0.9 + (p / l_t)**2)
    
    # 4. Excedente Hídrico (Lluvia neta disponible)
    df['excedente_mm'] = (df['p_anual'] - df['etr_mm']).clip(lower=0)
    
    # 5. Recarga Real (Infiltración profunda)
    # Aquí aplicamos el Ki definido por el usuario
    df['recarga_mm'] = df['excedente_mm'] * ki
    
    return df

def generate_projections(df_turc, years_ahead=1, ruido_factor=0.5):
    """Genera proyección anualizada."""
    recarga_actual = df_turc['recarga_mm'].mean()
    if np.isnan(recarga_actual): recarga_actual = 0
    
    # Crear eje de tiempo futuro (Mensual para suavidad visual, pero valores anualizados)
    n_months = (years_ahead - 2025) * 12 + 12 # Asumiendo base 2025
    if n_months < 12: n_months = 12
    
    fechas = pd.date_range(start='2026-01-01', periods=n_months, freq='M')
    
    # Simulación estocástica (Random Walk con deriva estacional)
    np.random.seed(42)
    trend = np.linspace(recarga_actual, recarga_actual * 0.95, n_months) # Ligera tendencia a la baja (cambio climático)
    seasonality = np.sin(np.linspace(0, 4*np.pi, n_months)) * (recarga_actual * 0.15)
    noise = np.random.normal(0, recarga_actual * 0.05 * ruido_factor, n_months)
    
    valores = trend + seasonality + noise
    
    # Bandas de confianza
    upper = valores * 1.15
    lower = valores * 0.85
    
    return pd.DataFrame({
        'Fecha': fechas,
        'Recarga Estimada': valores.clip(min=0),
        'Limite Superior': upper.clip(min=0),
        'Limite Inferior': lower.clip(min=0)
    })

# --- MOTOR DE DATOS ---
if gdf_zona is not None and not gdf_zona.empty:
    engine = create_engine(st.secrets["DATABASE_URL"])
    
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    
    # 1. Buscar Estaciones
    q_est = text("""
        SELECT id_estacion, nom_est, alt_est, ST_Y(geom::geometry) as lat, ST_X(geom::geometry) as lon
        FROM estaciones 
        WHERE ST_X(geom::geometry) BETWEEN :minx AND :maxx 
          AND ST_Y(geom::geometry) BETWEEN :miny AND :maxy
    """)
    
    try:
        df_est = pd.read_sql(q_est, engine, params={"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})
        
        if not df_est.empty:
            # --- 2. SELECTOR LOCAL DE ESTACIONES (Recuperado) ---
            # Permite filtrar dentro de la cuenca
            estaciones_disponibles = df_est['nom_est'].unique()
            sel_local = st.multiselect("📍 Filtrar Estaciones Específicas:", estaciones_disponibles, default=estaciones_disponibles)
            
            # Filtrar DF base
            df_est_filtered = df_est[df_est['nom_est'].isin(sel_local)]
            
            if not df_est_filtered.empty:
                ids_v = df_est_filtered['id_estacion'].unique()
                ids_s = ",".join([f"'{str(x)}'" for x in ids_v])
                
                # 3. Datos Climáticos
                q_clima = text(f"""
                    SELECT id_estacion_fk as id_estacion, AVG(precipitation)*12 as p_anual 
                    FROM precipitacion_mensual 
                    WHERE id_estacion_fk IN ({ids_s}) 
                    GROUP BY id_estacion_fk
                """)
                df_clima = pd.read_sql(q_clima, engine)
                
                # Merge
                df_est_filtered['id_estacion'] = df_est_filtered['id_estacion'].astype(str)
                df_clima['id_estacion'] = df_clima['id_estacion'].astype(str)
                df_work = pd.merge(df_est_filtered, df_clima, on='id_estacion', how='inner')
                df_work['alt_est'] = df_work['alt_est'].fillna(altitud_ref)
                
                if not df_work.empty:
                    # 4. CÁLCULO DEL MODELO
                    df_res = calculate_turc_advanced(df_work, ki_ponderado)
                    
                    # KPIs Globales
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Precipitación Media", f"{df_res['p_anual'].mean():.0f} mm/año")
                    kpi2.metric("Evapotranspiración", f"{df_res['etr_mm'].mean():.0f} mm/año")
                    kpi3.metric("Coef. Infiltración", f"{ki_ponderado:.2f}", delta=f"Bosque: {pct_bosque}%")
                    kpi4.metric("Recarga Total", f"{df_res['recarga_mm'].mean():.0f} mm/año", delta="Infiltración Profunda")
                    
                    st.divider()

                    # --- PESTAÑAS DE ANÁLISIS ---
                    tab_hist, tab_proy, tab_data = st.tabs(["📊 Serie Histórica", "🔮 Proyección Futura", "📋 Datos Detallados"])
                    
                    with tab_hist:
                        # 5. GRÁFICO HISTÓRICO (Revivido)
                        # Agrupamos promedio de todas las estaciones seleccionadas
                        fig_hist = go.Figure()
                        
                        # Usamos índices ficticios para representar "Promedio Anual" comparativo
                        # (P vs ETR vs R)
                        vals = [df_res['p_anual'].mean(), df_res['etr_mm'].mean(), df_res['excedente_mm'].mean(), df_res['recarga_mm'].mean()]
                        cats = ['Precipitación', 'ETR', 'Excedente Hídrico', 'Recarga (Acuífero)']
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#17becf']
                        
                        fig_hist.add_trace(go.Bar(
                            x=cats, y=vals,
                            text=[f"{v:.0f}" for v in vals],
                            textposition='auto',
                            marker_color=colors
                        ))
                        
                        fig_hist.update_layout(
                            title="Balance Hídrico Medio Anual (Turc)",
                            yaxis_title="Lámina de Agua (mm/año)",
                            template="plotly_white",
                            height=500
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                        st.caption(f"Este gráfico muestra cómo se distribuye la lluvia según el coeficiente de infiltración seleccionado ({ki_ponderado}).")

                    with tab_proy:
                        if activar_proyeccion:
                            # 6. PROYECCIÓN (Unidades Corregidas)
                            df_proy = generate_projections(df_res, years_ahead=int(horizonte), ruido_factor=ruido)
                            
                            fig_proy = go.Figure()
                            
                            # Rango de Incertidumbre
                            fig_proy.add_trace(go.Scatter(
                                x=df_proy['Fecha'], y=df_proy['Limite Superior'],
                                mode='lines', line=dict(width=0), showlegend=False
                            ))
                            fig_proy.add_trace(go.Scatter(
                                x=df_proy['Fecha'], y=df_proy['Limite Inferior'],
                                mode='lines', line=dict(width=0), fill='tonexty', 
                                fillcolor='rgba(0, 100, 255, 0.1)', name='Incertidumbre'
                            ))
                            
                            # Línea Central
                            fig_proy.add_trace(go.Scatter(
                                x=df_proy['Fecha'], y=df_proy['Recarga Estimada'],
                                mode='lines', line=dict(color='blue', width=3), name='Recarga Estimada'
                            ))
                            
                            fig_proy.update_layout(
                                title=f"Proyección de Recarga de Acuíferos (Horizonte {horizonte})",
                                yaxis_title="Recarga (mm/año)", # <--- UNIDAD CORREGIDA
                                hovermode="x unified",
                                template="plotly_white",
                                height=500
                            )
                            st.plotly_chart(fig_proy, use_container_width=True)
                            
                    with tab_data:
                        # 7. TABLA DE DATOS (Formato arreglado)
                        st.dataframe(
                            df_res[['nom_est', 'alt_est', 'p_anual', 'etr_mm', 'excedente_mm', 'recarga_mm']].style.format({
                                'alt_est': '{:.0f}',
                                'p_anual': '{:.1f}',
                                'etr_mm': '{:.1f}',
                                'excedente_mm': '{:.1f}',
                                'recarga_mm': '{:.1f}'
                            }),
                            use_container_width=True
                        )
                else:
                    st.warning("Las estaciones seleccionadas no tienen datos históricos de precipitación.")
            else:
                st.warning("Seleccione al menos una estación del filtro local.")
        else:
            st.warning("No hay estaciones en esta zona geográfica.")
            
    except Exception as e:
        st.error(f"Error en el módulo: {e}")
else:
    st.info("👈 Seleccione una zona en la barra lateral para comenzar.")