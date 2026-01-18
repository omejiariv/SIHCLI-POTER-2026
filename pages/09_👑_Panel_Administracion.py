# pages/09_👑_Panel_Administracion.py

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import time

# --- 1. CONFIGURACIÓN Y SEGURIDAD ---
st.set_page_config(page_title="Admin Panel", page_icon="👑", layout="wide")

ADMIN_PASSWORD = "sihcli2026" 

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🔐 Acceso Restringido")
        st.info("Área exclusiva para la gestión de datos maestros.")
        pwd = st.text_input("Contraseña de Administrador:", type="password")
        if st.button("Ingresar al Panel"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("⛔ Acceso Denegado")
    return False

if not check_password():
    st.stop()

# --- 2. CONEXIÓN ---
def get_engine():
    return create_engine(st.secrets["DATABASE_URL"])

# --- 3. INTERFAZ ---
st.title("👑 Panel de Administración y Edición de Datos")
st.markdown("---")

tab_estaciones, tab_predios, tab_sql = st.tabs([
    "📡 Editar Estaciones", 
    "🏡 Gestión de Predios", 
    "🛠️ Consola SQL"
])

# ==============================================================================
# TAB 1: EDITOR DE ESTACIONES (CALIBRADO PARA TU ESQUEMA)
# ==============================================================================
with tab_estaciones:
    st.subheader("Modificar Metadatos de Estaciones")
    
    try:
        engine = get_engine()
        # 1. Selector de Estación
        df_list = pd.read_sql("SELECT id_estacion, nom_est FROM estaciones ORDER BY nom_est", engine)
        opciones = {f"{row['nom_est']} ({row['id_estacion']})": row['id_estacion'] for index, row in df_list.iterrows()}
        
        seleccion = st.selectbox("🔍 Buscar Estación a Editar:", options=list(opciones.keys()))
        
        if seleccion:
            id_sel = opciones[seleccion]
            
            # 2. Cargar datos específicos (Usando tus nombres de columna reales)
            query_data = text(f"SELECT * FROM estaciones WHERE id_estacion = '{id_sel}'")
            with engine.connect() as conn:
                df_est = pd.read_sql(query_data, conn)
            
            if not df_est.empty:
                curr = df_est.iloc[0]
                
                st.info(f"Editando ID: **{id_sel}**")
                
                with st.form("form_estacion"):
                    c1, c2 = st.columns(2)
                    
                    # Campos de Texto (Manejando posibles nulos)
                    val_nombre = curr['nom_est'] if pd.notnull(curr['nom_est']) else ""
                    val_muni = curr['municipio'] if pd.notnull(curr['municipio']) else ""
                    
                    new_name = c1.text_input("Nombre Estación:", value=val_nombre)
                    new_muni = c2.text_input("Municipio:", value=val_muni)
                    
                    c3, c4 = st.columns(2)
                    
                    # Campos Numéricos (Tus columnas son 'latitude' y 'longitude')
                    val_lat = float(curr['latitude']) if pd.notnull(curr['latitude']) else 0.0
                    val_lon = float(curr['longitude']) if pd.notnull(curr['longitude']) else 0.0
                    
                    new_lat = c3.number_input("Latitud (latitude):", value=val_lat, format="%.6f")
                    new_lon = c4.number_input("Longitud (longitude):", value=val_lon, format="%.6f")
                    
                    st.caption("Nota: Al guardar, se actualizarán las columnas 'latitude', 'longitude' y la geometría 'geometry' para los mapas.")
                    
                    # Botón de Guardar
                    submitted = st.form_submit_button("💾 Guardar Cambios")
                    
                    if submitted:
                        try:
                            # 3. SQL UPDATE (Sincronizando columnas planas y geometría PostGIS)
                            sql_update = text("""
                                UPDATE estaciones 
                                SET 
                                    nom_est = :nm, 
                                    municipio = :mu,
                                    latitude = :la,
                                    longitude = :lo,
                                    geometry = ST_SetSRID(ST_Point(:lo, :la), 4326)
                                WHERE id_estacion = :id
                            """)
                            
                            params = {
                                "nm": new_name, 
                                "mu": new_muni, 
                                "la": new_lat, 
                                "lo": new_lon, 
                                "id": id_sel
                            }
                            
                            with engine.connect() as conn:
                                conn.execute(sql_update, params)
                                conn.commit()
                            
                            st.success("✅ Datos actualizados correctamente.")
                            time.sleep(1.5)
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error crítico al guardar en base de datos: {e}")

    except Exception as e:
        st.error(f"Error cargando el módulo de edición: {e}")

# ==============================================================================
# TAB 2: GESTIÓN DE PREDIOS
# ==============================================================================
with tab_predios:
    st.subheader("🏡 Gestión de Predios y Adquisiciones")
    
    try:
        engine = get_engine()
        # Verificar si la tabla existe
        check_table = text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'predios_gestion')")
        with engine.connect() as conn:
            existe_tabla = conn.execute(check_table).scalar()
        
        if not existe_tabla:
            st.warning("⚠️ La tabla 'predios_gestion' no existe en la base de datos.")
            with st.expander("🛠️ Script SQL para crear la tabla"):
                st.code("""
                CREATE TABLE predios_gestion (
                    id SERIAL PRIMARY KEY,
                    codigo_catastral VARCHAR(50),
                    propietario VARCHAR(100),
                    estado_gestion VARCHAR(50), 
                    area_ha FLOAT,
                    geom GEOMETRY(Polygon, 4326)
                );
                """, language="sql")
        else:
            col_busqueda, col_accion = st.columns([2, 1])
            with col_busqueda:
                id_predio = st.text_input("🔍 Buscar por Cédula Catastral:", placeholder="Ej: 001-234-567")
                if st.button("Buscar Predio"):
                    st.info("Funcionalidad de búsqueda lista para conectar.")

    except Exception as e:
        st.error(f"Error de conexión: {e}")

# ==============================================================================
# TAB 3: CONSOLA SQL
# ==============================================================================
with tab_sql:
    st.error("⚠️ Zona de Peligro: Los cambios aquí son irreversibles.")
    
    col_q, col_res = st.columns([1, 1])
    
    with col_q:
        sql_query = st.text_area("Comando SQL:", height=200, placeholder="SELECT * FROM estaciones LIMIT 5;")
        run_btn = st.button("▶️ Ejecutar Sentencia", type="primary")
    
    with col_res:
        if run_btn and sql_query:
            if any(x in sql_query.upper() for x in ["DROP", "DELETE", "TRUNCATE"]):
                st.error("🚫 Comandos destructivos bloqueados.")
            else:
                try:
                    engine = get_engine()
                    with engine.connect() as conn:
                        if sql_query.strip().upper().startswith("SELECT"):
                            df_res = pd.read_sql(text(sql_query), conn)
                            st.dataframe(df_res)
                        else:
                            res = conn.execute(text(sql_query))
                            conn.commit()
                            st.success(f"✅ Ejecutado. Filas afectadas: {res.rowcount}")
                except Exception as e:
                    st.error(f"Error SQL: {e}")