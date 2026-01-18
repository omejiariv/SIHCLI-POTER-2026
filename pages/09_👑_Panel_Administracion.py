# pages/09_👑_Panel_Administracion.py

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import time

# --- 1. CONFIGURACIÓN Y SEGURIDAD ---
st.set_page_config(page_title="Admin Panel", page_icon="👑", layout="wide")

# Contraseña simple para proteger el módulo (Idealmente mover a st.secrets)
ADMIN_PASSWORD = "sihcli2026" 

def check_password():
    """Retorna True si el usuario ingresó la contraseña correcta."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    st.title("🔐 Acceso Restringido")
    pwd = st.text_input("Ingrese la contraseña de administrador:", type="password")
    
    if st.button("Ingresar"):
        if pwd == ADMIN_PASSWORD:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    return False

if not check_password():
    st.stop()

# --- 2. CONEXIÓN A BASE DE DATOS ---
def get_engine():
    return create_engine(st.secrets["DATABASE_URL"])

# --- 3. INTERFAZ PRINCIPAL ---
st.title("👑 Panel de Administración y Edición de Datos")
st.markdown("---")

tab_estaciones, tab_predios, tab_sql = st.tabs([
    "📡 Editar Estaciones", 
    "🏡 Gestión de Predios", 
    "🛠️ Consola SQL"
])

# ==============================================================================
# TAB 1: EDITOR DE ESTACIONES
# ==============================================================================
with tab_estaciones:
    st.subheader("Modificar Metadatos de Estaciones")
    
    # 1. Cargar lista de estaciones para el selector
    try:
        engine = get_engine()
        # Traemos ID y Nombre para facilitar la búsqueda
        df_list = pd.read_sql("SELECT id_estacion, nom_est FROM estaciones ORDER BY nom_est", engine)
        
        # Crear un diccionario para el selectbox: "Nombre (ID)" -> ID
        opciones = {f"{row['nom_est']} ({row['id_estacion']})": row['id_estacion'] for index, row in df_list.iterrows()}
        seleccion = st.selectbox("🔍 Buscar Estación a Editar:", options=list(opciones.keys()))
        
        if seleccion:
            id_sel = opciones[seleccion]
            
            # 2. Cargar datos completos de la estación seleccionada
            query_data = text(f"SELECT * FROM estaciones WHERE id_estacion = '{id_sel}'")
            with engine.connect() as conn:
                df_est = pd.read_sql(query_data, conn)
            
            if not df_est.empty:
                st.info(f"Editando: **{df_est.iloc[0]['nom_est']}**")
                
                # Formulario de Edición
                with st.form("form_estacion"):
                    c1, c2 = st.columns(2)
                    # Usamos los valores actuales como default
                    curr = df_est.iloc[0]
                    
                    new_name = c1.text_input("Nombre de la Estación:", value=curr['nom_est'])
                    new_lat = c2.number_input("Latitud:", value=float(curr['latitud']) if curr['latitud'] else 0.0, format="%.5f")
                    new_lon = c2.number_input("Longitud:", value=float(curr['longitud']) if curr['longitud'] else 0.0, format="%.5f")
                    new_muni = c1.text_input("Municipio:", value=curr['municipio'] if 'municipio' in curr else "")
                    new_cat = c1.selectbox("Categoría:", ["Pluviométrica", "Climatológica", "Limnimétrica"], index=0) # Ajustar según tus datos reales
                    
                    submitted = st.form_submit_button("💾 Guardar Cambios en Base de Datos")
                    
                    if submitted:
                        try:
                            # 3. Ejecutar UPDATE
                            update_query = text("""
                                UPDATE estaciones 
                                SET nom_est = :nm, latitud = :la, longitud = :lo, municipio = :mu
                                WHERE id_estacion = :id
                            """)
                            with engine.connect() as conn:
                                conn.execute(update_query, {"nm": new_name, "la": new_lat, "lo": new_lon, "mu": new_muni, "id": id_sel})
                                conn.commit()
                            st.success("✅ Estación actualizada correctamente.")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al guardar: {e}")
            else:
                st.error("No se encontraron datos para esta estación.")
                
    except Exception as e:
        st.error(f"Error de conexión: {e}")

# ==============================================================================
# TAB 2: GESTIÓN DE PREDIOS
# ==============================================================================
with tab_predios:
    st.subheader("Estado de Ejecución de Predios")
    st.caption("Cambie el estado de gestión de un predio (Ej: De 'Identificado' a 'Ejecutado').")
    
    # Simulación: Asumimos que tienes una tabla 'predios' con columna 'estado' y 'codigo_catastral'
    # Si no tienes tabla aún, esto servirá de plantilla.
    
    try:
        # Aquí deberías ajustar el query a tu tabla real de predios
        # q_predios = "SELECT codigo, propietario, estado FROM predios LIMIT 100" 
        # df_predios = pd.read_sql(q_predios, engine)
        
        st.warning("⚠️ Módulo en construcción: Requiere conectar con la tabla real de 'Predios'.")
        
        # Ejemplo Visual de cómo funcionará:
        col_id_predio = st.text_input("Ingrese Código Catastral o ID del Predio:")
        
        if st.button("Buscar Predio"):
            # Lógica futura: Buscar en DB
            st.success("Predio encontrado: La Finca (ID: 12345)")
            new_status = st.selectbox("Actualizar Estado:", ["Pendiente", "En Negociación", "Ejecutado/Conservado", "Descartado"])
            
            if st.button("Actualizar Estado"):
                # update_query = text("UPDATE predios SET estado = :st WHERE codigo = :cd")
                # conn.execute(update_query, ...)
                st.toast(f"Predio 12345 actualizado a: {new_status}")

    except Exception as e:
        st.error(f"Error cargando predios: {e}")

# ==============================================================================
# TAB 3: CONSOLA SQL (SOLO EXPERTOS)
# ==============================================================================
with tab_sql:
    with st.expander("⚠️ Consola de Comandos Directos (Peligro)", expanded=False):
        st.warning("Esta herramienta ejecuta SQL directo. Úsela solo para correcciones rápidas de nombres de municipios o cuencas.")
        
        sql_command = st.text_area("Comando SQL:", placeholder="UPDATE municipios SET nombre = 'Nuevo Nombre' WHERE id = 5;")
        
        if st.button("Ejecutar Comando SQL"):
            if "DROP" in sql_command.upper() or "DELETE" in sql_command.upper():
                st.error("🚫 Comandos destructivos bloqueados por seguridad.")
            else:
                try:
                    with engine.connect() as conn:
                        result = conn.execute(text(sql_command))
                        conn.commit()
                        st.success(f"Comando ejecutado. Filas afectadas: {result.rowcount}")
                except Exception as e:
                    st.error(f"Error SQL: {e}")