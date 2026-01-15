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
    from modules import selectors # Solo importamos selectores, el resto lo hacemos aquí
except ImportError:
    st.error("Error al importar módulos base.")
    st.stop()

st.title("💧 Estimación de Recarga (Modelo Turc + Escenarios)")

# --- FUNCIONES MATEMÁTICAS INTEGRADAS (Para evitar errores de importación) ---
def calculate_turc_model(df):
    """Calcula balance hídrico usando el método de Turc."""
    df = df.copy()
    # Temperatura estimada por altitud (Gradiente térmico)
    df['temp_est'] = 30 - (0.0065 * df['alt_est'])
    
    # Capacidad evaporativa del aire L(t)
    # L(t) = 300 + 25T + 0.05T^3
    t = df['temp_est']
    l_t = 300 + 25*t + 0.05*(t**3)
    
    # Evapotranspiración Real (ETR)
    # ETR = P / sqrt(0.9 + (P/L(t))^2)
    p = df['p_anual']
    df['etr_mm'] = p / np.sqrt(0.9 + (p / l_t)**2)
    
    # Recarga Potencial (R = P - ETR)
    df['recarga_mm'] = df['p_anual'] - df['etr_mm']
    df['recarga_mm'] = df['recarga_mm'].clip(lower=0) # No puede haber recarga negativa
    
    # Coeficiente de Escorrentía simple (Referencial)
    df['coef_escorrentia'] = df['recarga_mm'] / df['p_anual']
    
    return df

def generate_simple_scenarios(df_turc, meses=12, ruido=1.0):
    """Genera proyecciones estocásticas simples basadas en los datos actuales."""
    recarga_base = df_turc['recarga_mm'].mean()
    if np.isnan(recarga_base): recarga_base = 0
    
    fechas = pd.date_range(start=pd.Timestamp.today(), periods=meses, freq='M')
    
    # Tendencia estacional simulada (senoidal)
    x = np.linspace(0, 4*np.pi, meses)
    estacionalidad = np.sin(x) * (recarga_base * 0.2)
    
    # Escenarios
    neutro = np.full(meses, recarga_base) + estacionalidad
    
    # Ruido aleatorio
    np.random.seed(42)
    ruido_arr = np.random.normal(0, recarga_base * 0.1 * ruido, meses)
    
    optimista = neutro * 1.2 + ruido_arr
    pesimista = neutro * 0.8 - ruido_arr
    
    # Dataframe resultado
    df_res = pd.DataFrame({
        'Fecha': fechas,
        'Neutro (Tendencial)': neutro.clip(min=0),
        'Optimista (Húmedo)': optimista.clip(min=0),
        'Pesimista (Seco)': pesimista.clip(min=0)
    })
    return df_res

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
    
    # 1. Búsqueda Espacial (Bounding Box)
    minx, miny, maxx, maxy = gdf_zona.total_bounds
    
    # Buscamos estaciones en el rectángulo de la zona
    q_spatial = text("""
        SELECT id_estacion, nom_est, alt_est, latitud, longitud
        FROM estaciones 
        WHERE longitud BETWEEN :minx AND :maxx 
          AND latitud BETWEEN :miny AND :maxy
    """)
    
    try:
        df_estaciones = pd.read_sql(q_spatial, engine, params={"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})
        
        if not df_estaciones.empty:
            ids = tuple(df_estaciones['id_estacion'].unique())
            ids_sql = str(ids).replace(',)', ')') if len(ids) > 1 else f"({ids[0]})"
            
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
                        use_container_width=True
                    )
                
                with tab2:
                    if activar_proyeccion:
                        df_proy = generate_simple_scenarios(df_res, meses=horizonte, ruido=ruido)
                        
                        fig = go.Figure()
                        # Optimista
                        fig.add_trace(go.Scatter(x=df_proy['Fecha'], y=df_proy['Optimista (Húmedo)'], mode='lines', line=dict(width=0), showlegend=False))
                        # Pesimista (Relleno)
                        fig.add_trace(go.Scatter(x=df_proy['Fecha'], y=df_proy['Pesimista (Seco)'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,255,0.1)', name='Rango Incertidumbre'))
                        # Neutro
                        fig.add_trace(go.Scatter(x=df_proy['Fecha'], y=df_proy['Neutro (Tendencial)'], mode='lines', name='Tendencia', line=dict(color='blue')))
                        
                        fig.update_layout(title="Proyección de Recarga", yaxis_title="mm/mes", hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Estaciones encontradas pero sin datos de precipitación.")
        else:
            st.warning("No se encontraron estaciones en esta zona. Aumenta el Radio Buffer.")
            
    except Exception as e:
        st.error(f"Error técnico: {e}")

else:
    st.info("👈 Seleccione una zona.")