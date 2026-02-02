import streamlit as st
import pandas as pd
from sqlalchemy import text
from modules.db_manager import get_engine

st.set_page_config(page_title="Diagnóstico de Cruces", layout="wide")
st.title("🕵️‍♂️ Diagnóstico Forense de Datos")

engine = get_engine()

c1, c2 = st.columns(2)

with c1:
    st.subheader("1. Tabla: ESTACIONES")
    try:
        # Traemos IDs y Nombres
        q1 = "SELECT id_estacion, nombre FROM estaciones LIMIT 5"
        df_est = pd.read_sql(q1, engine)
        st.dataframe(df_est)
        
        if not df_est.empty:
            sample_id = df_est.iloc[0]['id_estacion']
            st.info(f"📌 Tipo de dato (Python): {type(sample_id)}")
            st.text(f"Valor crudo: '{sample_id}'")
            st.text(f"Longitud: {len(str(sample_id))}")
    except Exception as e:
        st.error(f"Error: {e}")

with c2:
    st.subheader("2. Tabla: PRECIPITACION")
    try:
        # Traemos IDs
        q2 = "SELECT id_estacion, fecha, valor FROM precipitacion LIMIT 5"
        df_rain = pd.read_sql(q2, engine)
        st.dataframe(df_rain)
        
        if not df_rain.empty:
            sample_id_rain = df_rain.iloc[0]['id_estacion']
            st.info(f"📌 Tipo de dato (Python): {type(sample_id_rain)}")
            st.text(f"Valor crudo: '{sample_id_rain}'")
            st.text(f"Longitud: {len(str(sample_id_rain))}")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.subheader("3. PRUEBA DE CRUCE (JOIN)")

try:
    # Intento de cruce directo
    q_join = """
    SELECT e.id_estacion, e.nombre, p.valor 
    FROM estaciones e
    JOIN precipitacion p ON e.id_estacion = p.id_estacion
    LIMIT 5
    """
    df_join = pd.read_sql(q_join, engine)
    
    if df_join.empty:
        st.error("❌ EL CRUCE DIRECTO FALLÓ. No hay coincidencia exacta.")
        
        # Intento de cruce con TRIM (Espacios)
        q_trim = """
        SELECT e.id_estacion, p.valor
        FROM estaciones e JOIN precipitacion p 
        ON TRIM(CAST(e.id_estacion AS TEXT)) = TRIM(CAST(p.id_estacion AS TEXT))
        LIMIT 5
        """
        df_trim = pd.read_sql(q_trim, engine)
        
        if not df_trim.empty:
            st.warning("⚠️ ¡AJÁ! El cruce funciona SOLO si usamos TRIM. Significa que tus IDs tienen espacios en blanco ocultos.")
            st.success("✅ La solución es usar TRIM en todas las consultas (ya lo hicimos en los archivos anteriores).")
        else:
            st.error("☠️ Ni siquiera con TRIM funciona. Revisa si los IDs son totalmente diferentes (ej: '1205' vs '1205000').")
            
            st.write("Muestra Estaciones:", df_est['id_estacion'].head().tolist())
            st.write("Muestra Lluvia:", df_rain['id_estacion'].head().tolist())
            
    else:
        st.success("✅ ¡EL CRUCE FUNCIONA PERFECTAMENTE! El problema no es la base de datos.")
        st.dataframe(df_join)

except Exception as e:
    st.error(f"Error técnico en el cruce: {e}")


import streamlit as st
import pandas as pd
from sqlalchemy import text
from modules.db_manager import get_engine

st.set_page_config(page_title="Diagnóstico Coordenadas", layout="wide")
st.title("🕵️‍♂️ Diagnóstico de Coordenadas")

engine = get_engine()

st.subheader("🔎 Revisión de Latitud y Longitud")
try:
    # Traemos una muestra de coordenadas
    q = "SELECT id_estacion, nombre, latitud, longitud FROM estaciones LIMIT 10"
    df = pd.read_sql(q, engine)
    
    st.dataframe(df)
    
    if not df.empty:
        lat_val = df.iloc[0]['latitud']
        lon_val = df.iloc[0]['longitud']
        
        c1, c2, c3 = st.columns(3)
        c1.info(f"Tipo Latitud: {type(lat_val)}")
        c2.info(f"Valor Lat: {lat_val}")
        c3.info(f"Valor Lon: {lon_val}")
        
        # Análisis
        if isinstance(lat_val, str):
            st.error("🚨 ¡ERROR CRÍTICO! Las coordenadas son TEXTO. Deben ser NÚMEROS.")
            if "," in str(lat_val):
                st.warning("💡 Pista: Tienen comas (,) en vez de puntos (.).")
        elif lat_val > 1000:
            st.warning("⚠️ Parece que son Coordenadas Planas (Magna-Sirgas). El mapa web espera Geográficas (Lat ~6.0, Lon ~-75.0).")
        else:
            st.success("✅ Parecen coordenadas Geográficas numéricas válidas.")
            
except Exception as e:
    st.error(f"Error: {e}")