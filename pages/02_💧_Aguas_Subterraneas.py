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

st.title("💧 Estimación de Recarga (Modelo Turc + Escenarios)")

# --- FUNCIONES MATEMÁTICAS ---
def calculate_turc_model(df):
    """Calcula balance hídrico usando el método de Turc."""
    df = df.copy()
    df['temp_est'] = 30 - (0.0065 * df['alt_est'])
    t = df['temp_est']
    l_t = 300 + 25*t + 0.05*(t**3)
    p = df['p_anual']
    df['etr_mm'] = p / np.sqrt(0.9 + (p / l_t)**2)
    df['recarga_mm'] = (df['p_anual'] - df['etr_mm']).clip(lower=0)
    return df

def generate_simple_scenarios(df_turc, meses=12, ruido=1.0):
    recarga_base = df_turc['recarga_mm'].mean()
    if np.isnan(recarga_base): recarga_base = 0
    fechas = pd.date_range(start=pd.Timestamp.today(), periods=meses, freq='M')
    x = np.linspace(0, 4*np.pi, meses)
    estacionalidad = np.sin(x) * (recarga_base * 0.2)
    neutro = np.full(meses, recarga_base) + estacionalidad
    np.random.seed(42)
    ruido_arr = np.random.normal(0, recarga_base * 0.1 * ruido, meses)
    optimista = neutro * 1.2 + ruido_arr
    pesimista = neutro * 0.8 - ruido_arr
    return pd.DataFrame({
        'Fecha': fechas,
        'Neutro (Tendencial)': neutro.clip(min=0),
        'Optimista (Húmedo)': optimista.clip(min=0),
        'Pesimista (Seco)': pesimista.clip(min=0)
    })

# --- INTERFAZ ---
ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

st.sidebar.divider()
st.sidebar.subheader("⚙️ Pronóstico")
activar_proyeccion = st.sidebar.checkbox("Activar Proyección", value=True)
horizonte = st.sidebar.selectbox("Horizonte (meses):", [12, 24, 60], index=1)
ruido = st.sidebar.slider("Variabilidad Climática:", 0.0, 2.0, 0.5)

# --- LÓGICA PRINCIPAL ---
if gdf_zona is not None and not gdf_zona.empty:
    engine = create_engine(st.secrets["DATABASE_URL"])
    
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    
    # 1. CORRECCIÓN SQL: Usamos ST_X y ST_Y para extraer coordenadas de 'geom'
    # Esto soluciona el error "column latitud does not exist"
    q_spatial = text("""
        SELECT 
            id_estacion, 
            nom_est, 
            alt_est, 
            ST_Y(geom::geometry) as latitud, 
            ST_X(geom::geometry) as longitud
        FROM estaciones 
        WHERE ST_X(geom::geometry) BETWEEN :minx AND :maxx 
          AND ST_Y(geom::geometry) BETWEEN :miny AND :maxy
    """)
    
    try:
        df_estaciones = pd.read_sql(q_spatial, engine, params={"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})
        
        if not df_estaciones.empty:
            ids = tuple(df_estaciones['id_estacion'].unique())
            # Formateo seguro de IDs para SQL
            if len(ids) == 1:
                ids_sql = f"({ids[0]})"
            else:
                ids_sql = str(ids)
            
            # 2. Traer Datos Climáticos
            q_clima = text(f"""
                SELECT id_estacion_fk as id_estacion, AVG(precipitation)*12 as p_anual 
                FROM precipitacion_mensual 
                WHERE id_estacion_fk IN {ids_sql} 
                GROUP BY id_estacion_fk
            """)
            df_clima = pd.read_sql(q_clima, engine)
            
            # Unir todo
            df_full = pd.merge(df_estaciones, df_clima, on='id_estacion', how='inner')
            df_full['alt_est'] = df_full['alt_est'].fillna(altitud_ref)
            
            if not df_full.empty:
                # 3. Calcular Modelo
                df_res = calculate_turc_model(df_full)
                
                # --- VISUALIZACIÓN ---
                tab1, tab2 = st.tabs(["📊 Balance Hídrico Actual", "🔮 Proyección Futura"])
                
                with tab1:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Precipitación Media", f"{df_res['p_anual'].mean():.0f} mm")
                    c2.metric("ETR Estimada", f"{df_res['etr_mm'].mean():.0f} mm")
                    c3.metric("Recarga Potencial", f"{df_res['recarga_mm'].mean():.0f} mm", delta="Oferta Hídrica")
                    
                    st.dataframe(
                        df_res[['nom_est', 'alt_est', 'p_anual', 'recarga_mm']].style.format("{:.1f}"),
                        use_container_width=True # Mantenemos compatibility standard
                    )
                
                with tab2:
                    if activar_proyeccion:
                        df_proy = generate_simple_scenarios(df_res, meses=horizonte, ruido=ruido)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_proy['Fecha'], y=df_proy['Optimista (Húmedo)'], mode='lines', line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=df_proy['Fecha'], y=df_proy['Pesimista (Seco)'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,255,0.1)', name='Rango Incertidumbre'))
                        fig.add_trace(go.Scatter(x=df_proy['Fecha'], y=df_proy['Neutro (Tendencial)'], mode='lines', name='Tendencia', line=dict(color='blue')))
                        
                        fig.update_layout(title="Proyección de Recarga", yaxis_title="mm/mes", hovermode="x unified")
                        # CORRECCIÓN DEPRECACIÓN: width="stretch"
                        st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Estaciones encontradas pero sin datos de precipitación.")
        else:
            st.warning("No se encontraron estaciones en esta zona. Aumenta el Radio Buffer.")
            
    except Exception as e:
        st.error(f"Error técnico: {e}")
else:
    st.info("👈 Seleccione una zona.")